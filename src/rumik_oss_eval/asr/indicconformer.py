"""AI4Bharat IndicConformer judge — 600M multilingual, all 22 Indian languages.

Measured against `whisper-large-v3` on the same generated audio, the same
references and the same normalizer (`promptsets/indic_v1.yaml`, 13 Indic rows):

    whisper-large-v3   mean CER 0.1521
    IndicConformer CTC          0.0716
    IndicConformer RNNT         0.0643      <- default

Two of our open issues closed outright by switching judge rather than by writing
code:

* **Numbers.** Whisper rendered `नौ` as `9` and `પંદર` as `15`, which is a
  guaranteed mismatch against a word-form reference. IndicConformer writes the
  words. `hi-003` went 0.172 -> 0.000. That removed the need to hand-build
  numeral tables for nine languages, which `num2words` cannot cover (it has
  en/te/kn/bn only).
* **Script consistency.** Whisper returned Gurmukhi audio as mixed
  Devanagari+Gurmukhi (`सत्चੀ ਏਕਾਲ`). IndicConformer stays in-script;
  `pa-001` went 0.370 -> 0.148.

What did *not* change is the useful part: `hing-001` scores ~0.167 under both
judges because the model really does speak the speaker name ("सिल्वर्स"). What
persists across judges is the model; what moves is the judge.

Runs on **CPU** in practice. onnxruntime-gpu 1.28 wants CUDA 13 / cuDNN 9 and the
box has the CUDA 12.8 runtime that torch ships, so the CUDA provider fails to
load and it falls back. That is fine and arguably preferable: ~10-30x realtime,
and it leaves both GPUs entirely to generation.
"""

from __future__ import annotations

import sys as _sys
import time
from collections.abc import Sequence

import numpy as np

from .base import AsrBackend, AsrResult, AsrStatus, Clip, classify

TARGET_RATE = 16_000
DEFAULT_MODEL = "ai4bharat/indic-conformer-600m-multilingual"

#: The 22 scheduled languages the model covers. English is NOT among them, so an
#: English row must go to a different judge rather than be silently mistranscribed.
SUPPORTED = frozenset({
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml",
    "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
})


class IndicConformerAsr(AsrBackend):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        decoder: str = "rnnt",
        fallback: AsrBackend | None = None,
    ) -> None:
        if decoder not in ("ctc", "rnnt"):
            raise ValueError("decoder must be 'ctc' or 'rnnt'")
        self.name = f"indicconformer:{decoder}"
        self.model_id = model_id
        self.decoder = decoder
        #: Used for languages outside SUPPORTED (English, mainly). Without one,
        #: those clips are reported as failures rather than guessed at.
        self.fallback = fallback
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModel

            # The bundled from_pretrained always calls snapshot_download, so it
            # needs the repo id -- a local path raises HFValidationError. The
            # snapshot is resolved from cache when already present.
            self._model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        return self._model

    #: Distinct failure reasons already printed, so a batch of 2000 clips
    #: failing the same way prints one line and not 2000.
    _reported: set = set()

    @staticmethod
    def _prepare(audio: np.ndarray, sample_rate: int):
        import torch

        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        if sample_rate != TARGET_RATE:
            import torchaudio

            wav = torchaudio.functional.resample(wav, sample_rate, TARGET_RATE)
        return wav

    def transcribe(self, clips: Sequence[Clip]) -> list[AsrResult]:
        if not clips:
            return []

        results: list[AsrResult | None] = [None] * len(clips)

        # Route unsupported languages before loading anything.
        unsupported = [i for i, c in enumerate(clips) if (c.language or "") not in SUPPORTED]
        if unsupported:
            if self.fallback is not None:
                for i, r in zip(unsupported, self.fallback.transcribe([clips[i] for i in unsupported])):
                    results[i] = r
            else:
                for i in unsupported:
                    results[i] = AsrResult(
                        text="", status=AsrStatus.FAILED, clip_id=clips[i].clip_id,
                        language=clips[i].language, seconds=clips[i].seconds,
                    )

        todo = [i for i in range(len(clips)) if results[i] is None]
        if todo:
            model = self._load()
            for i in todo:
                clip = clips[i]
                t0 = time.time()
                try:
                    out = model(self._prepare(clip.audio, clip.sample_rate),
                                clip.language, self.decoder)
                    text = str(out[0] if isinstance(out, (list, tuple)) else out).strip()
                    status = classify(text, clip.audio)
                except Exception as exc:
                    text, status = "", AsrStatus.FAILED
                    # Report each distinct reason once. Silently discarding this
                    # is how an environment problem masquerades as "the model
                    # cannot say this language" -- which has already happened on
                    # this project and cost a whole training run.
                    reason = f"{type(exc).__name__}: {exc}"
                    if reason not in self._reported:
                        self._reported.add(reason)
                        print(f"[indicconformer] transcription failed "
                              f"({clip.language}, {clip.clip_id}): {reason}",
                              file=_sys.stderr, flush=True)
                results[i] = AsrResult(
                    # Never hand back a transcript we would not score: a
                    # plausible string yields a plausible CER, which is worse
                    # than an obvious failure.
                    text=text if status is AsrStatus.OK else "",
                    status=status, clip_id=clip.clip_id, language=clip.language,
                    seconds=clip.seconds, infer_seconds=time.time() - t0,
                )

        return [r for r in results if r is not None]
