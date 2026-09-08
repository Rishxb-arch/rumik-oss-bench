"""The ASR judge interface.

Whatever sits behind this Protocol *becomes the objective* once RL starts — the
policy will move toward whatever the judge rewards. So the interface is designed
to make judge swaps cheap and judge failures visible, rather than to hide them.

Two rules the implementations must honour, both learned from the training repo's
version (`ayatts_runtime/deepspeed_runner.py:6384`), which violated all of them:

1. **Load the model once.** Upstream calls `whisper.load_model("base")` inside
   the per-sample function, reloading the checkpoint on every clip.
2. **Respect the sample rate.** Upstream accepts `sample_rate` and ignores it.
   Mimi decodes at 24 kHz and Whisper assumes 16 kHz, so audio was fed ~1.5x too
   fast — which inflates WER on its own, independent of the model.

And one rule of our own: a transcript we do not trust must be reported as such.
An ASR that hallucinates on silence produces a *plausible* string, and a
plausible string scored against a reference yields a plausible CER. That is far
worse than an obvious failure, because nothing downstream can tell it went wrong.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np


class AsrStatus(StrEnum):
    OK = "ok"
    EMPTY = "empty"                  # model returned nothing
    SILENT_INPUT = "silent_input"    # audio itself is (near) silence
    DEGENERATE = "degenerate"        # looping/repeating output, classic hallucination
    FAILED = "failed"                # backend raised


#: Statuses whose transcript must NOT be scored as if it were a real hypothesis.
UNTRUSTWORTHY = frozenset({AsrStatus.EMPTY, AsrStatus.SILENT_INPUT,
                           AsrStatus.DEGENERATE, AsrStatus.FAILED})


@dataclass(slots=True)
class Clip:
    """One piece of audio to transcribe.

    `audio` is 1-D float32 in [-1, 1] at `sample_rate`. Backends resample as
    needed; callers must not pre-resample, so that the true rate stays visible.
    """

    audio: np.ndarray
    sample_rate: int
    clip_id: str = ""
    language: str = "hi"

    @property
    def seconds(self) -> float:
        return float(len(self.audio)) / float(self.sample_rate or 1)


@dataclass(slots=True)
class AsrResult:
    text: str
    status: AsrStatus
    clip_id: str = ""
    language: str = ""
    seconds: float = 0.0
    infer_seconds: float = 0.0

    @property
    def trustworthy(self) -> bool:
        return self.status not in UNTRUSTWORTHY


@runtime_checkable
class AsrBackend(Protocol):
    """Any judge. `name` goes into the run summary so a score is attributable."""

    name: str

    def transcribe(self, clips: Sequence[Clip]) -> list[AsrResult]:
        """Transcribe a batch, returning one result per clip, in order."""
        ...


# --- shared guards, so every backend flags the same failure modes ----------

def is_silent(audio: np.ndarray, *, rms_threshold: float = 1e-4) -> bool:
    """Near-silent audio. Whisper reliably invents text over silence."""
    if audio is None or len(audio) == 0:
        return True
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) < rms_threshold


def is_degenerate(text: str, *, min_words: int = 6, repeat_ratio: float = 0.5) -> bool:
    """Looping output — the same token repeated over and over.

    Whisper's characteristic hallucination is a short phrase repeated to fill
    the window. A transcript where one word is more than half of all words is
    not a transcript.
    """
    words = (text or "").split()
    if len(words) < min_words:
        return False
    most_common = max((words.count(w) for w in set(words)), default=0)
    return most_common / len(words) > repeat_ratio


def classify(text: str, audio: np.ndarray) -> AsrStatus:
    """Order matters: silence is about the input, so it outranks output checks."""
    if is_silent(audio):
        return AsrStatus.SILENT_INPUT
    if not (text or "").strip():
        return AsrStatus.EMPTY
    if is_degenerate(text):
        return AsrStatus.DEGENERATE
    return AsrStatus.OK
