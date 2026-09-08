"""The single normalization entry point used for BOTH reference and hypothesis.

Using one function for both sides is the point: any asymmetry between how the
reference and the hypothesis are prepared shows up as fake error. Every
intermediate stage is retained on the result so a bad number can be traced to
the stage that caused it -- without that, a broken normalizer is invisible and
you end up "measuring" the normalizer instead of the model.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fold import fold
from .markup import strip_markup
from .numbers import neutralise_numbers
from .translit import detect_scripts, romanize


@dataclass(slots=True)
class Normalized:
    raw: str
    spoken: str        # markup removed
    romanized: str     # Indic spans mapped to ISO Latin
    folded: str        # coarse phonetic form -- this is what gets scored
    scripts: frozenset[str]
    numbers_replaced: int = 0   # numeric tokens neutralised, for visibility
    speaker: str | None = None
    tones: tuple[str, ...] = ()
    events: tuple[str, ...] = ()

    def __str__(self) -> str:  # pragma: no cover - debugging affordance
        return self.folded


def normalize_for_scoring(
    text: str,
    *,
    strip_prompt_markup: bool = True,
    allow_unknown_speaker: bool = True,
    neutralise_numeric: bool = True,
) -> Normalized:
    """Reduce text to the form CER/WER are computed on.

    `strip_prompt_markup` should stay True for references (which carry speaker,
    tone and event tags) and is harmless for ASR hypotheses, which contain no
    markup -- keeping it symmetric avoids an asymmetry between the two sides.
    """
    raw = text or ""
    if strip_prompt_markup:
        stripped = strip_markup(raw, allow_unknown_speaker=allow_unknown_speaker)
        spoken, speaker = stripped.spoken, stripped.speaker
        tones, events = tuple(stripped.tones), tuple(stripped.events)
    else:
        spoken, speaker, tones, events = raw, None, (), ()

    scripts = frozenset(detect_scripts(spoken))
    romanized = romanize(spoken)
    folded = fold(romanized)

    # Numbers are neutralised AFTER folding, because the word list is written in
    # the folded alphabet -- "three" only looks like "tri" once fold() has run.
    replaced = 0
    if neutralise_numeric:
        n = neutralise_numbers(folded)
        folded, replaced = n.text, n.replaced

    return Normalized(
        raw=raw,
        spoken=spoken,
        romanized=romanized,
        folded=folded,
        scripts=scripts,
        speaker=speaker,
        tones=tones,
        events=events,
        numbers_replaced=replaced,
    )

def normalized_tier(text: str) -> str:
    """The published benchmark's own `normalized` tier: NFC, drop Unicode P/S,
    collapse whitespace, casefold.

    This is the tier the six-vendor run and the 583 benchmark report as `cer`,
    and therefore the ONLY one comparable to their numbers. Our `folded` output
    is the `rubric` tier: romanised and equivalence-folded, deliberately more
    forgiving, and about 40% lower on the same audio. Scoring on `folded` and
    comparing against vendors measured on `normalized` is not a comparison.

    One deliberate wart. Punctuation is DELETED, not replaced with a space, so
    `राम-श्याम` collapses to one token where the ASR emits two. That is a known
    defect in the published tier (it affects 245 of 1,885 references and inflates
    the punctuation-heavy WER lift from 1.24x to 1.33x). It is reproduced here
    on purpose: the point of this function is comparability with their numbers,
    not correctness. Fixing it would silently move us off their scale again.
    """
    import unicodedata

    s = unicodedata.normalize("NFC", text or "")
    kept = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S"):
            continue          # deleted, not spaced -- see docstring
        kept.append(ch)
    return " ".join("".join(kept).split()).casefold()
