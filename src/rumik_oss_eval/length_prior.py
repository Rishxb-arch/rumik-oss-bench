"""Expected duration for a piece of text, fitted from real recordings.

Without this the length term is **inert**: `expected_seconds` is None for every
prompt, so `score_sample` renormalises the term away and the reward becomes pure
CER/WER. That matters because slowing down and over-articulating is *the* CER
reward hack, and the length band is what blocks it — so the guard is missing
exactly when RL would start exploiting it.

The estimator is deliberately crude: characters per second, per language, fitted
by median. A duration model does not need to be good, it needs to be unbiased
and stable. The rubric only penalises deviation outside a wide dead zone
(0.8-1.25), so what matters is that the centre is right, not that the prediction
is tight.

Median rather than mean because recording corpora carry long-tailed junk —
clips with trailing silence, or transcripts that do not match the audio.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .normalize.markup import strip_markup

#: Fallback when a language has no fitted entry. Roughly the pace of the
#: hinglish60k corpus; only used so a missing language degrades rather than
#: disables the length term.
DEFAULT_CHARS_PER_SECOND = 14.0

#: Below this many samples a per-language fit is noise, so we fall back.
MIN_SAMPLES = 30

_WS = re.compile(r"\s+")


def scoreable_length(text: str) -> int:
    """Characters that are actually spoken: markup stripped, whitespace collapsed.

    Counting the raw prompt would make a long `<description="...">` inflate the
    expected duration, which is exactly backwards — the description is never
    voiced.
    """
    spoken = strip_markup(text or "").spoken
    return len(_WS.sub(" ", spoken).strip())


@dataclass(slots=True)
class LengthPrior:
    chars_per_second: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    default_cps: float = DEFAULT_CHARS_PER_SECOND

    def cps(self, language: str) -> float:
        return self.chars_per_second.get(language, self.default_cps)

    def expected_seconds(self, text: str, language: str = "hi") -> float | None:
        """Predicted duration, or None for text with nothing to say."""
        n = scoreable_length(text)
        if n <= 0:
            return None
        return n / self.cps(language)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "chars_per_second": self.chars_per_second,
            "counts": self.counts,
            "default_cps": self.default_cps,
        }, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "LengthPrior":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(chars_per_second=d.get("chars_per_second", {}),
                   counts=d.get("counts", {}),
                   default_cps=float(d.get("default_cps", DEFAULT_CHARS_PER_SECOND)))


def fit_prior(
    rows: Iterable[tuple[str, str, float]],
    *,
    min_seconds: float = 0.8,
    max_seconds: float = 30.0,
    min_samples: int = MIN_SAMPLES,
) -> LengthPrior:
    """Fit chars/second per language from `(text, language, duration_s)` rows.

    Clips outside the duration bounds are dropped: very short ones are dominated
    by leading/trailing silence, and very long ones are usually concatenations
    where the transcript does not track the audio.
    """
    per_lang: dict[str, list[float]] = {}
    for text, language, duration in rows:
        if duration is None or not (min_seconds <= float(duration) <= max_seconds):
            continue
        n = scoreable_length(text)
        if n <= 0:
            continue
        per_lang.setdefault(language or "hi", []).append(n / float(duration))

    cps, counts = {}, {}
    pooled = [v for vals in per_lang.values() for v in vals]
    default = statistics.median(pooled) if pooled else DEFAULT_CHARS_PER_SECOND
    for lang, vals in per_lang.items():
        counts[lang] = len(vals)
        if len(vals) >= min_samples:
            cps[lang] = round(statistics.median(vals), 3)
    return LengthPrior(chars_per_second=cps, counts=counts,
                       default_cps=round(float(default), 3))


def fit_from_manifest(path: str | Path, *, language: str = "hi", **kwargs) -> LengthPrior:
    """Fit from a hinglish60k-style JSONL (`text`, `duration_s`, one language)."""
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("duration_s") is None or not d.get("text"):
                continue
            rows.append((str(d["text"]), language, float(d["duration_s"])))
    return fit_prior(rows, **kwargs)
