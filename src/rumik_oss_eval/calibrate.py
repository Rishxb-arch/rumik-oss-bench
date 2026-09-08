"""Noise floor: what CER does this pipeline report on known-good human audio?

This is the gate. Every downstream number is meaningless without it - a reward
improvement smaller than the floor is unmeasurable, and a policy optimised
against a noisy reward will find the noise rather than the speech.

Deliberately **no TTS model is loaded**. Real recordings go through the same ASR
and the same normalizer that a generation would, so what comes out is the error
the measurement apparatus contributes by itself.

Two passes, because they answer different questions:

* **original** - the wav as recorded. Isolates ASR error plus normalizer error.
* **roundtrip** - encoded to Mimi codes and decoded back. Adds codec loss, and is
  the more honest floor: the model emits codes, so it can never beat what the
  codec can represent.

The gap between them is how much of the floor the codec owns.
"""

from __future__ import annotations

import json
import random
import subprocess
import wave
from collections.abc import Sequence
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .asr.base import AsrBackend, Clip
from .metrics.cer_wer import Score, Status, score_pair
from .normalize.pipeline import normalize_for_scoring
from .normalize.translit import detect_scripts

# Corpus location is deployment-specific: point CALIBRATION_CORPUS at a
# directory or bucket of (audio, reference) pairs. The calibration itself is
# corpus-agnostic -- it measures what CER the judge and normaliser produce on
# known-good human speech, whatever that speech is.
BUCKET = os.environ.get("CALIBRATION_CORPUS", "")
MIMI_RATE = 24_000


@dataclass(slots=True)
class CalibrationSample:
    sample_id: str
    text: str                      # reference, still carrying [tone] / <event> markup
    emotion: str = ""
    duration_s: float | None = None

    @property
    def audio_name(self) -> str:
        """The sample_id is used VERBATIM. Do not zero-pad it.

        The manifest carries ids of mixed width ('005654' and '8953' both occur)
        and the bucket mirrors that exactly: filenames exist at every width from
        1 to 6 digits. Critically, `7629.wav` and `007629.wav` are *different
        recordings*, and 4,961 short ids have a zero-padded twin that is a
        separate manifest row.

        Padding therefore does not "fix" a short id - it silently swaps in
        someone else's audio. That defect put 12.5% of a calibration run at
        CER 1.0 and more than doubled the reported noise floor (0.19 vs 0.08)
        before it was caught.
        """
        return f"{self.sample_id}.wav"


@dataclass(slots=True)
class CalibrationRow:
    sample_id: str
    pass_name: str                 # "original" | "roundtrip"
    reference: str
    hypothesis: str
    ref_folded: str
    hyp_folded: str
    score: Score
    asr_status: str
    scripts: tuple[str, ...] = ()
    seconds: float = 0.0
    emotion: str = ""


# --- manifest -------------------------------------------------------------

def load_manifest(
    local_manifest: Path, n: int, *, seed: int = 42, min_seconds: float = 1.0
) -> list[CalibrationSample]:
    """Reservoir-free sampling: the manifest is ~20 MB, so just read and sample."""
    rows: list[CalibrationSample] = []
    with local_manifest.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(d.get("sample_id", "")).strip()
            text = str(d.get("text", "")).strip()
            if not sid or not text or not sid.isdigit():
                continue
            dur = d.get("duration_s")
            if dur is not None and float(dur) < min_seconds:
                continue
            rows.append(CalibrationSample(
                sample_id=sid, text=text,
                emotion=str(d.get("emotion", "") or ""),
                duration_s=float(dur) if dur is not None else None,
            ))
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def fetch_audio(samples: Sequence[CalibrationSample], dest: Path) -> dict[str, Path]:
    """Batch-copy the wavs we need. One gsutil -m call, not one per file."""
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {s.sample_id: dest / s.audio_name for s in samples}
    missing = [s for s in samples if not wanted[s.sample_id].exists()]
    if missing:
        uris = [f"{BUCKET}/audio/{s.audio_name}" for s in missing]
        proc = subprocess.run(
            ["gsutil", "-m", "-q", "cp", *uris, str(dest)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            # Partial failure is normal (a few ids may not exist); keep whatever landed.
            pass
    return {sid: p for sid, p in wanted.items() if p.exists()}


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        n_ch = w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = data.astype(np.float32) / 32768.0
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch).mean(axis=1)
    return audio, rate


# --- codec round trip (no LM) --------------------------------------------

def load_codec_only(bundle_path: str | Path, *, device: str = "cuda:0"):
    """Just the codec and its feature extractor. No language model."""
    import torch
    from transformers import AutoFeatureExtractor, MimiModel

    codec_dir = Path(bundle_path) / "codec"
    codec = MimiModel.from_pretrained(codec_dir, dtype=torch.float32).eval().to(device)
    fe = AutoFeatureExtractor.from_pretrained(codec_dir)
    return codec, fe


def mimi_roundtrip(codec, feature_extractor, audio: np.ndarray, rate: int,
                   *, num_quantizers: int = 8, device: str = "cuda:0") -> np.ndarray:
    """Encode to Q8 codes and decode back - exactly what a generation can express."""
    import torch

    inputs = feature_extractor(raw_audio=audio, sampling_rate=rate, return_tensors="pt")
    values = inputs["input_values"].to(device)
    if values.dim() == 2:                          # (B, N) -> (B, 1, N)
        values = values.unsqueeze(1)
    with torch.no_grad():
        codes = codec.encode(values, num_quantizers=num_quantizers).audio_codes  # (B,Q,T)
        out = codec.decode(codes).audio_values                                   # (B,1,N)
    return out[0, 0].float().cpu().numpy()


# --- the run --------------------------------------------------------------

def score_one(reference: str, hypothesis: str, *, asr_status: str) -> tuple[Score, str, str, tuple[str, ...]]:
    ref = normalize_for_scoring(reference)
    hyp = normalize_for_scoring(hypothesis)
    forced = None if asr_status == "ok" else Status.ASR_FAILED
    return (score_pair(ref.folded, hyp.folded, status=forced),
            ref.folded, hyp.folded, tuple(sorted(ref.scripts)))


def summarize(rows: Sequence[CalibrationRow]) -> dict:
    """Overall and per-script means, per pass. Only `valid` rows count."""
    out: dict = {}
    for pass_name in sorted({r.pass_name for r in rows}):
        subset = [r for r in rows if r.pass_name == pass_name and r.score.valid]
        cers = [r.score.cer for r in subset]
        wers = [r.score.wer for r in subset]
        by_script: dict[str, list[float]] = {}
        for r in subset:
            key = "+".join(r.scripts) if r.scripts else "latin_only"
            by_script.setdefault(key, []).append(r.score.cer)
        out[pass_name] = {
            "n": len(subset),
            "cer_mean": (sum(cers) / len(cers)) if cers else None,
            "cer_median": float(np.median(cers)) if cers else None,
            "wer_mean": (sum(wers) / len(wers)) if wers else None,
            "asr_failures": sum(1 for r in rows
                                if r.pass_name == pass_name and r.asr_status != "ok"),
            "by_script": {k: {"n": len(v), "cer_mean": sum(v) / len(v)}
                          for k, v in sorted(by_script.items())},
        }
    return out
