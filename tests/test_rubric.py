"""Reward aggregation: bounds, gates, and weight renormalisation."""

from __future__ import annotations

import pytest

from rumik_oss_eval.asr.base import AsrStatus
from rumik_oss_eval.metrics.cer_wer import Status, score_pair
from rumik_oss_eval.metrics.integrity import check_integrity
from rumik_oss_eval.metrics.length import length_penalty
from rumik_oss_eval.rubric import RubricWeights, score_sample

Q = 2
END = 999
CODE_MAP = {100 + c * Q + q: (c, q) for c in range(10) for q in range(Q)}


def good_integrity(frames: int = 3):
    emitted = [100 + c * Q + q for c in range(frames) for q in range(Q)] + [END]
    return check_integrity(emitted, num_quantizers=Q, audio_code_id_map=CODE_MAP, audio_end_id=END)


def bad_integrity():
    return check_integrity([100], num_quantizers=Q, audio_code_id_map=CODE_MAP, audio_end_id=END)


def truncated_integrity():
    emitted = [100 + c * Q + q for c in range(3) for q in range(Q)]      # no EOS
    return check_integrity(emitted, num_quantizers=Q, audio_code_id_map=CODE_MAP, audio_end_id=END)


PERFECT = score_pair("namaste kaise hain", "namaste kaise hain")
BAD = score_pair("namaste kaise hain", "totally different words here")
OK_LEN = length_penalty(6.0, 6.0)


# --- bounds ----------------------------------------------------------------

def test_perfect_sample_scores_one():
    r = score_sample(score=PERFECT, integrity=good_integrity(), length=OK_LEN)
    assert r.reward == pytest.approx(1.0)
    assert not r.gated and r.valid


def test_reward_always_within_unit_interval():
    for sc in (PERFECT, BAD, score_pair("a", "a" * 200)):
        for ln in (OK_LEN, length_penalty(60.0, 6.0), length_penalty(0.5, 6.0)):
            r = score_sample(score=sc, integrity=good_integrity(), length=ln)
            assert 0.0 <= r.reward <= 1.0


def test_worse_cer_lowers_reward():
    good = score_sample(score=PERFECT, integrity=good_integrity(), length=OK_LEN).reward
    bad = score_sample(score=BAD, integrity=good_integrity(), length=OK_LEN).reward
    assert bad < good


# --- hard gates ------------------------------------------------------------

def test_malformed_decode_gates_to_zero():
    """Even with a perfect-looking transcript, no frames means no sample."""
    r = score_sample(score=PERFECT, integrity=bad_integrity(), length=OK_LEN)
    assert r.reward == 0.0 and r.gated
    assert r.gate == str(Status.MALFORMED_DECODE)


def test_truncation_gates_to_zero():
    r = score_sample(score=PERFECT, integrity=truncated_integrity(), length=OK_LEN)
    assert r.reward == 0.0 and r.gate == str(Status.NO_EOS)


@pytest.mark.parametrize("status", [AsrStatus.SILENT_INPUT, AsrStatus.DEGENERATE,
                                    AsrStatus.EMPTY, AsrStatus.FAILED])
def test_untrustworthy_asr_gates_to_zero(status):
    """A hallucinated transcript must not yield a plausible reward."""
    r = score_sample(score=PERFECT, integrity=good_integrity(), length=OK_LEN,
                     asr_status=status)
    assert r.reward == 0.0 and r.gated and r.gate.startswith("asr:")
    assert r.valid is True          # counted as a failure, not dropped


def test_gate_beats_every_other_term():
    """Gates run before the weighted sum, so weights cannot rescue a dead sample."""
    r = score_sample(score=PERFECT, integrity=bad_integrity(), length=OK_LEN,
                     weights=RubricWeights(cer=1.0, wer=0.0, length=0.0))
    assert r.reward == 0.0


def test_empty_hypothesis_gates():
    empty = score_pair("namaste kaise hain", "")
    r = score_sample(score=empty, integrity=good_integrity(), length=OK_LEN)
    assert r.reward == 0.0 and r.gate == str(Status.EMPTY_HYPOTHESIS)


# --- data failure vs model failure ----------------------------------------

def test_empty_reference_is_invalid_not_gated():
    """A missing reference is a corpus bug: zero, but excluded from means."""
    r = score_sample(score=score_pair("", "the model said something"),
                     integrity=good_integrity(), length=OK_LEN)
    assert r.reward == 0.0
    assert r.valid is False
    assert r.gated is False


# --- weight renormalisation ------------------------------------------------

def test_missing_length_term_renormalises_rather_than_penalising():
    """No expected duration must be neither a free penalty nor free credit."""
    no_expectation = length_penalty(6.0, 0.0)      # valid=False
    assert no_expectation.valid is False
    r = score_sample(score=PERFECT, integrity=good_integrity(), length=no_expectation)
    assert r.reward == pytest.approx(1.0)          # perfect on the terms we have
    assert set(r.contributions) == {"cer", "wer"}
    assert sum(r.contributions.values()) == pytest.approx(1.0)


def test_length_term_absent_entirely_is_handled():
    r = score_sample(score=PERFECT, integrity=good_integrity(), length=None)
    assert r.reward == pytest.approx(1.0)
    assert "length" not in r.contributions


def test_contributions_sum_to_reward():
    r = score_sample(score=BAD, integrity=good_integrity(),
                     length=length_penalty(20.0, 6.0))
    assert sum(r.contributions.values()) == pytest.approx(r.reward)


def test_weights_are_respected():
    """All weight on length: a perfect-duration sample scores 1 despite bad CER."""
    r = score_sample(score=BAD, integrity=good_integrity(), length=OK_LEN,
                     weights=RubricWeights(cer=0.0, wer=0.0, length=1.0))
    assert r.reward == pytest.approx(1.0)


def test_every_term_is_reported_even_when_unweighted():
    """Terms are emitted per sample so weights can be retuned offline."""
    r = score_sample(score=BAD, integrity=good_integrity(3), length=OK_LEN,
                     weights=RubricWeights(cer=1.0, wer=0.0, length=0.0))
    assert {"cer", "wer", "length_penalty", "frames", "drop_rate", "emitted_eos"} <= set(r.terms)
    assert r.terms["frames"] == 3.0
