"""CER/WER semantics, including the reward-safety properties we depend on."""

from __future__ import annotations

import pytest

from rumik_oss_eval.metrics.cer_wer import Score, Status, score_pair
from rumik_oss_eval.metrics.edit import _levenshtein_python, edit_distance


# --- edit distance ---------------------------------------------------------

@pytest.mark.parametrize(
    "a,b,expected",
    [("", "", 0), ("abc", "abc", 0), ("abc", "", 3), ("", "abc", 3),
     ("kitten", "sitting", 3), ("flaw", "lawn", 2)],
)
def test_edit_distance_known_values(a, b, expected):
    assert edit_distance(a, b) == expected
    assert _levenshtein_python(a, b) == expected


def test_backend_agrees_with_reference_implementation():
    """rapidfuzz and the pure-Python fallback must not diverge."""
    cases = [("namaste", "namastey"), ("main ghar", "mai gar"),
             ("नमस्ते", "नमसते"), ("", "x"), ("abcdef", "fedcba")]
    for a, b in cases:
        assert edit_distance(a, b) == _levenshtein_python(a, b)


def test_edit_distance_on_word_lists():
    assert edit_distance(["main", "ghar", "ja"], ["main", "ghar", "gaya"]) == 1


# --- hand-computed CER/WER -------------------------------------------------

def test_exact_match_is_zero():
    s = score_pair("main ghar ja raha hoon", "main ghar ja raha hoon")
    assert s.cer == 0.0 and s.wer == 0.0
    assert s.status is Status.OK and s.valid


def test_one_word_substitution():
    s = score_pair("main ghar ja raha hoon", "main ghar ja raha hun")
    assert s.wer == pytest.approx(1 / 5)          # 1 of 5 words wrong
    # chars w/o spaces: ref "mainghatjarahahoon" = 18 chars.
    # "hoon" -> "hun" is TWO edits (substitute o->u, delete o), not one.
    assert s.ref_chars == 18
    assert s.cer == pytest.approx(2 / 18, abs=1e-6)


def test_wer_denominator_is_reference_word_count():
    s = score_pair("a b c d", "a b")
    assert s.wer == pytest.approx(2 / 4)
    assert s.ref_words == 4 and s.hyp_words == 2


# --- regression: the CER denominator disagreement --------------------------

def test_cer_denominator_excludes_spaces_regression():
    """train.py divided by len(ref) WITH spaces; the runtime path stripped them.

    We pin the spaces-stripped definition. 'ab cd' has 4 scoreable characters,
    not 5, so a single character error is 1/4 -- never 1/5.
    """
    s = score_pair("ab cd", "ab ce")
    assert s.ref_chars == 4
    assert s.cer == pytest.approx(1 / 4)


def test_whitespace_differences_alone_do_not_affect_cer():
    assert score_pair("ab cd", "abcd").cer == 0.0


# --- regression: NaN made silence the optimal policy -----------------------

def test_empty_hypothesis_scores_worst_not_nan_regression():
    """The bug: empty output returned NaN and safe_mean filtered NaN rows out,
    so a silent model was excluded from the average rather than punished."""
    s = score_pair("main ghar ja raha hoon", "")
    assert s.cer == 1.0 and s.wer == 1.0
    assert s.status is Status.EMPTY_HYPOTHESIS
    assert s.valid is True          # must be COUNTED, not dropped
    assert s.is_model_failure


def test_whitespace_only_hypothesis_is_empty():
    s = score_pair("hello", "   \n\t ")
    assert s.status is Status.EMPTY_HYPOTHESIS and s.cer == 1.0


# --- regression: unbounded ratios ------------------------------------------

def test_much_longer_hypothesis_is_clamped_to_one():
    """Raw distance/len(ref) exceeds 1.0 when the model babbles; an unbounded
    reward term destabilises GRPO's within-group normalisation."""
    s = score_pair("hi", "completely different and far far longer transcript here")
    assert s.cer == 1.0 and s.wer == 1.0
    assert s.status is Status.OK        # not a failure, just very wrong


def test_scores_always_within_unit_interval():
    for ref, hyp in [("abc", "xyz"), ("a", "aaaaaaaaaa"), ("aaaaaaaaaa", "a"),
                     ("नमस्ते", "hello world"), ("x", "")]:
        s = score_pair(ref, hyp)
        assert 0.0 <= s.cer <= 1.0 and 0.0 <= s.wer <= 1.0


# --- data failure vs model failure -----------------------------------------

def test_empty_reference_is_invalid_not_a_model_failure():
    s = score_pair("", "the model said something")
    assert s.status is Status.EMPTY_REFERENCE
    assert s.valid is False             # excluded from means: it is a corpus bug
    assert s.is_model_failure is False
    assert s.cer == 1.0                 # still worst, so nothing can exploit it


@pytest.mark.parametrize("status", [Status.MALFORMED_DECODE, Status.NO_EOS, Status.ASR_FAILED])
def test_forced_upstream_status_overrides_text(status):
    """A malformed decode must not be scored from whatever text happens to exist."""
    s = score_pair("hello world", "hello world", status=status)
    assert s.status is status
    assert s.cer == 1.0 and s.wer == 1.0
    assert s.valid is True


def test_ok_status_passed_explicitly_is_scored_normally():
    s = score_pair("hello world", "hello world", status=Status.OK)
    assert s.cer == 0.0 and s.status is Status.OK


# --- shape -----------------------------------------------------------------

def test_counts_are_reported():
    s = score_pair("ab cd", "ab")
    assert isinstance(s, Score)
    assert (s.ref_words, s.hyp_words, s.ref_chars, s.hyp_chars) == (2, 1, 4, 2)


def test_devanagari_counts_by_codepoint():
    """CER on Indic text counts combining marks as characters; that is intended
    and is why the folding stage exists upstream of this function."""
    s = score_pair("नमस्ते", "नमस्ते")
    assert s.cer == 0.0
    assert s.ref_chars == len("नमस्ते")
