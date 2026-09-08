"""Whisper judge, via transformers. Default: openai/whisper-large-v3 on cuda:1.

Pinned to a different device than the TTS model by default. Co-tenancy on one
card tested clean on this hardware, so this is not a workaround - with two 96 GB
cards it is free isolation, and it keeps generation throughput unaffected by ASR
load during an RL rollout.

Known limitation, deliberately accepted: Whisper picks its own output script and
is not always self-consistent on code-switched audio. Our normalizer romanizes
both sides, so a *consistent* choice costs nothing; only mixing scripts within
one utterance hurts. Calibration measures exactly what that costs, and that
measurement is what would justify moving to IndicConformer later.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from .base import AsrBackend, AsrResult, AsrStatus, Clip, classify

WHISPER_RATE = 16_000
DEFAULT_MODEL = "openai/whisper-large-v3"


class WhisperAsr(AsrBackend):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        device: str = "cuda:1",
        dtype: str = "float16",
        batch_size: int = 8,
        chunk_length_s: int = 30,
    ) -> None:
        self.name = f"whisper:{model_id.split('/')[-1]}"
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.chunk_length_s = chunk_length_s
        self._pipe = None  # built once, on first use

    # -- model is loaded exactly once, not per clip -------------------------
    def _pipeline(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline

            torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                           "float32": torch.float32}[self.dtype]
            self._pipe = pipeline(
                task="automatic-speech-recognition",
                model=self.model_id,
                device=self.device,
                torch_dtype=torch_dtype,
                chunk_length_s=self.chunk_length_s,
            )
        return self._pipe

    # -- resampling is the backend's job, so the true rate stays visible ----
    @staticmethod
    def _to_whisper_rate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == WHISPER_RATE:
            return np.asarray(audio, dtype=np.float32)
        import torch
        import torchaudio

        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        out = torchaudio.functional.resample(wav, sample_rate, WHISPER_RATE)
        return out.squeeze(0).numpy().astype(np.float32)

    def transcribe(self, clips: Sequence[Clip]) -> list[AsrResult]:
        if not clips:
            return []

        results: list[AsrResult | None] = [None] * len(clips)

        # Whisper takes one language per call, so batch by language rather than
        # forcing a single one. Hardcoding "en" over an Indic corpus is exactly
        # the upstream bug we are not repeating.
        by_language: dict[str, list[int]] = {}
        for i, clip in enumerate(clips):
            by_language.setdefault(clip.language or "hi", []).append(i)

        pipe = self._pipeline()
        for language, idxs in by_language.items():
            inputs = [
                {"raw": self._to_whisper_rate(clips[i].audio, clips[i].sample_rate),
                 "sampling_rate": WHISPER_RATE}
                for i in idxs
            ]
            t0 = time.time()
            try:
                out = pipe(
                    inputs,
                    batch_size=min(self.batch_size, len(inputs)),
                    generate_kwargs={"language": language, "task": "transcribe"},
                )
                texts = [str((o or {}).get("text", "")).strip() for o in out]
                failed = None
            except Exception as exc:  # a backend crash must not look like bad speech
                texts = [""] * len(idxs)
                failed = f"{type(exc).__name__}: {exc}"
            elapsed = time.time() - t0
            per_clip = elapsed / max(len(idxs), 1)

            for slot, i in enumerate(idxs):
                clip = clips[i]
                text = texts[slot]
                status = AsrStatus.FAILED if failed else classify(text, clip.audio)
                results[i] = AsrResult(
                    # Do not hand back a transcript we would not score. A
                    # hallucinated string produces a plausible CER, which is
                    # worse than an obvious failure.
                    text=text if status is AsrStatus.OK else "",
                    status=status,
                    clip_id=clip.clip_id,
                    language=language,
                    seconds=clip.seconds,
                    infer_seconds=per_clip,
                )

        return [r for r in results if r is not None]
