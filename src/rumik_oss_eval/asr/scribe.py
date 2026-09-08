"""ElevenLabs Scribe as an ASR judge, behind the same Protocol as the others.

Why bother with a paid remote judge: every per-language conclusion we draw is
conditional on the judge, and a second *independent* opinion is the only way to
tell "the model said it wrong" from "this ASR is weak here". IndicConformer and
Whisper are both open models with overlapping training data; Scribe is a
genuinely different system, which is what makes it useful as a control rather
than as a replacement.

Two things to keep in mind when reading its numbers:

* **It costs money per call and it leaves the machine.** Unlike the local judges
  this sends audio to a third party, so it is opt-in per run, never a default.
* **It emits punctuation and casing.** IndicConformer largely does not. That
  difference alone accounted for 10% of the measured error mass in the vendor
  benchmark, so compare Scribe against Scribe, not against a local judge, unless
  the text has been normalised.
"""

from __future__ import annotations

import io
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .base import AsrResult, AsrStatus, Clip, classify

ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"
DEFAULT_MODEL = "scribe_v2"
#: Scribe reports ISO-639-3; our prompt sets use ISO-639-1. Only the languages we
#: actually route matter here, and an unmapped code simply skips the hint.
ISO1_TO_ISO3 = {
    "as": "asm", "bn": "ben", "brx": "brx", "doi": "doi", "gu": "guj", "hi": "hin",
    "kn": "kan", "ks": "kas", "kok": "kok", "mai": "mai", "ml": "mal", "mni": "mni",
    "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan", "sa": "san", "sat": "sat",
    "sd": "snd", "ta": "tam", "te": "tel", "ur": "urd", "en": "eng",
}


def _key(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    for var in ("ELEVENLABS_API_KEY", "ELEVEN_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    for path in ("~/.elevenlabs_key", "~/.config/elevenlabs/key"):
        p = os.path.expanduser(path)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                return fh.read().strip()
    raise RuntimeError(
        "No ElevenLabs key. Set ELEVENLABS_API_KEY or write ~/.elevenlabs_key (mode 600)."
    )


def _wav_bytes(audio, sample_rate: int) -> bytes:
    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype="float32"), int(sample_rate), format="WAV",
             subtype="PCM_16")
    return buf.getvalue()


@dataclass(slots=True)
class ScribeAsr:
    model_id: str = DEFAULT_MODEL
    api_key: str | None = None
    workers: int = 4            # modest: this is someone else's rate limit
    timeout: float = 120.0
    retries: int = 3
    language_hint: bool = True
    _key_cache: str | None = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"scribe:{self.model_id}"

    def _resolved_key(self) -> str:
        if self._key_cache is None:
            self._key_cache = _key(self.api_key)
        return self._key_cache

    def _one(self, clip: Clip) -> AsrResult:
        import requests

        data = {"model_id": self.model_id}
        if self.language_hint and clip.language:
            iso3 = ISO1_TO_ISO3.get(clip.language)
            if iso3:
                data["language_code"] = iso3
        files = {"file": (f"{clip.clip_id}.wav",
                          _wav_bytes(clip.audio, clip.sample_rate), "audio/wav")}
        last = ""
        for attempt in range(self.retries):
            try:
                r = requests.post(ENDPOINT, headers={"xi-api-key": self._resolved_key()},
                                  data=data, files=files, timeout=self.timeout)
                if r.status_code == 429:            # rate limited: back off and retry
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                text = (r.json() or {}).get("text", "") or ""
                return AsrResult(text=text.strip(),
                                 status=classify(text.strip(), clip.audio),
                                 clip_id=clip.clip_id, language=clip.language)
            except Exception as exc:                # noqa: BLE001 - report, do not crash a run
                last = f"{type(exc).__name__}: {exc}"
                time.sleep(0.5 * (attempt + 1))
        # A failed call is a measurement failure, not a model failure: the rubric
        # gates on FAILED rather than scoring an empty transcript as silence.
        return AsrResult(text="", status=AsrStatus.FAILED, clip_id=clip.clip_id,
                         language=clip.language)

    def transcribe(self, clips: Sequence[Clip]) -> list[AsrResult]:
        if not clips:
            return []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self._one, clips))
