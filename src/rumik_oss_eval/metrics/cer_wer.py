"""CER and WER, defined once, with reward-safe failure semantics.

Three defects in the training repo's versions are fixed here deliberately.

1. There were three divergent implementations that disagreed on the CER
   denominator: `torchtitan/train.py:1083` divides by `len(ref_norm)` *including*
   spaces, while `deepspeed_runner.py:6424` strips spaces first. We pick one --
   spaces stripped on both sides -- and pin it with a test.

2. They return unbounded ratios. Levenshtein / len(ref) exceeds 1.0 whenever the
   hypothesis is longer than the reference, which a babbling model produces
   constantly. An unbounded, one-sided reward term wrecks GRPO's within-group
   normalisation, so both metrics are clamped to [0, 1].

3. They return NaN when the hypothesis is empty, and the aggregator
   (`research_eval.safe_mean`) *filters NaN out*. A model that emits silence is
   therefore excluded from the average instead of punished -- under RL that makes
   "generate nothing" the optimal policy. Empty or undecodable output scores the
   worst possible value here, never NaN.

The `valid` flag separates the two kinds of failure that must not be conflated:

  * model failure  -> valid=True,  cer=1.0   (score it, punish it)
  * data failure   -> valid=False, cer=1.0   (a broken prompt; exclude from means)

Aggregation must average over `valid` rows only, but must NOT drop invalid-
looking model failures. That distinction is the whole point of the flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import regex

from .edit import edit_distance


class Status(StrEnum):
    OK = "ok"
    EMPTY_HYPOTHESIS = "empty_hypothesis"      # model produced no words
    EMPTY_REFERENCE = "empty_reference"        # bad prompt -> not scoreable
    MALFORMED_DECODE = "malformed_decode"      # no complete audio frames
    NO_EOS = "no_eos"                          # hit max_new_tokens, truncated
    ASR_FAILED = "asr_failed"                  # transcription backend errored


#: Statuses that mean "the model failed", as opposed to "we cannot score this".
MODEL_FAILURE_STATUSES = frozenset(
    {Status.EMPTY_HYPOTHESIS, Status.MALFORMED_DECODE, Status.NO_EOS}
)


@dataclass(slots=True)
class Score:
    cer: float
    wer: float
    status: Status
    valid: bool
    ref_words: int = 0
    hyp_words: int = 0
    ref_chars: int = 0
    hyp_chars: int = 0
    #: Edit rate over Unicode extended grapheme clusters. `cer` above is over
    #: code points and is retained unchanged so historical runs stay comparable;
    #: `gcer` is the one to trust for Indic scripts.
    gcer: float = 1.0
    ref_graphemes: int = 0
    hyp_graphemes: int = 0

    @property
    def is_model_failure(self) -> bool:
        return self.status in MODEL_FAILURE_STATUSES


_GRAPHEME = regex.compile(r"\X")


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _chars_no_space(s: str) -> list[str]:
    return [c for c in s if not c.isspace()]


def grapheme_clusters(s: str) -> list[str]:
    r"""Split into Unicode extended grapheme clusters, dropping whitespace.

    `_chars_no_space` splits into code points, which is wrong for every Indic
    script here. "नमस्ते" is 6 code points but 3 clusters, because a consonant
    plus virama plus vowel sign is one written unit. Dividing the edit distance
    by 6 rather than 3 understates CER, and it understates it by a different
    factor per script, so cross language comparison is distorted in favour of
    scripts that use more combining marks.

    `regex` implements \X, the Unicode extended grapheme cluster. It is already
    a dependency via rapidfuzz's stack, so this costs nothing new.
    """
    return [g for g in _GRAPHEME.findall(s) if not g.isspace()]


def score_pair(
    reference: str,
    hypothesis: str,
    *,
    status: Status | None = None,
) -> Score:
    """Score one normalized reference against one normalized hypothesis.

    Both inputs must already be through `normalize.pipeline` -- this function
    does no normalization of its own, so that the scoring definition and the
    normalization policy stay independently testable.

    Pass `status` to force a known upstream failure (malformed decode, no EOS,
    ASR error); it is recorded and scored as the worst value rather than
    silently producing a plausible-looking number from a bad transcript.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    ref_chars = _chars_no_space(reference)
    hyp_chars = _chars_no_space(hypothesis)
    ref_gr = grapheme_clusters(reference)
    hyp_gr = grapheme_clusters(hypothesis)

    counts = dict(
        ref_words=len(ref_words),
        hyp_words=len(hyp_words),
        ref_chars=len(ref_chars),
        hyp_chars=len(hyp_chars),
        ref_graphemes=len(ref_gr),
        hyp_graphemes=len(hyp_gr),
    )

    # A missing reference is a corpus bug, not a model failure. Score it worst so
    # nothing can exploit it, but mark it invalid so aggregation can drop it.
    if not ref_chars:
        return Score(cer=1.0, wer=1.0, gcer=1.0, status=Status.EMPTY_REFERENCE,
                     valid=False, **counts)

    # Caller-declared upstream failure wins over whatever text we happen to hold.
    if status is not None and status is not Status.OK:
        return Score(cer=1.0, wer=1.0, gcer=1.0, status=status, valid=True, **counts)

    if not hyp_chars:
        return Score(cer=1.0, wer=1.0, gcer=1.0, status=Status.EMPTY_HYPOTHESIS,
                     valid=True, **counts)

    cer = _clamp01(edit_distance(ref_chars, hyp_chars) / len(ref_chars))
    gcer = (_clamp01(edit_distance(ref_gr, hyp_gr) / len(ref_gr)) if ref_gr else 1.0)
    wer = (
        _clamp01(edit_distance(ref_words, hyp_words) / len(ref_words))
        if ref_words
        else 1.0
    )
    return Score(cer=cer, wer=wer, gcer=gcer, status=Status.OK, valid=True, **counts)
