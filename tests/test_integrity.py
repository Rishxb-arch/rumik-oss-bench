"""Integrity gates: the structural checks that must override CER/WER."""

from __future__ import annotations

import pytest

from rumik_oss_eval.metrics.cer_wer import Status, score_pair
from rumik_oss_eval.metrics.integrity import Integrity, check_integrity

Q = 2
END = 999
# <code_q> -> id, matching the vendored parser's (code, q) convention.
CODE_MAP = {100 + c * Q + q: (c, q) for c in range(10) for q in range(Q)}


def tok(code: int, q: int) -> int:
    return 100 + code * Q + q


def frame(code: int) -> list[int]:
    return [tok(code, q) for q in range(Q)]


def check(emitted, **kw):
    return check_integrity(emitted, num_quantizers=Q, audio_code_id_map=CODE_MAP,
                           audio_end_id=END, **kw)


# --- the healthy case ------------------------------------------------------

def test_clean_generation_passes():
    r = check(frame(1) + frame(2) + frame(3) + [END])
    assert r.frames_committed == 3
    assert r.tokens_dropped == 0
    assert r.emitted_eos is True
    assert r.decode_ok and r.passed
    assert r.gate is None
    assert r.drop_rate == 0.0


def test_tokens_after_eos_are_ignored():
    """Generation may keep sampling past its own EOS; only the span counts."""
    r = check(frame(1) + [END] + frame(5) + frame(6))
    assert r.frames_committed == 1 and r.emitted_eos


# --- gate: no decodable frame ---------------------------------------------

def test_no_complete_frame_is_malformed():
    r = check([tok(1, 0), END])            # half a frame
    assert r.frames_committed == 0
    assert r.decode_ok is False
    assert r.gate is Status.MALFORMED_DECODE


def test_empty_generation_is_malformed():
    r = check([])
    assert r.gate is Status.MALFORMED_DECODE
    assert r.drop_rate == 0.0              # no tokens, so no meaningful rate


def test_malformed_outranks_missing_eos():
    """Both are wrong; 'no frames at all' is the more specific diagnosis."""
    r = check([tok(1, 0)])                 # no frame AND no eos
    assert r.gate is Status.MALFORMED_DECODE


# --- gate: truncation ------------------------------------------------------

def test_missing_eos_is_gated():
    r = check(frame(1) + frame(2))
    assert r.frames_committed == 2 and r.decode_ok
    assert r.emitted_eos is False
    assert r.gate is Status.NO_EOS
    assert not r.passed


def test_hit_token_cap_is_recorded():
    emitted = frame(1) + frame(2)
    r = check(emitted, max_new_tokens=len(emitted))
    assert r.hit_token_cap is True and r.gate is Status.NO_EOS


def test_cap_not_flagged_when_eos_arrived_in_time():
    emitted = frame(1) + [END]
    r = check(emitted, max_new_tokens=len(emitted))
    assert r.hit_token_cap is False and r.passed


# --- dropped tokens --------------------------------------------------------

def test_out_of_order_quantizer_drops_the_partial_frame():
    # q0, q1, then a stray q1 (expected q0) -> reset, losing nothing committed
    r = check(frame(1) + [tok(3, 1)] + frame(4) + [END])
    assert r.frames_committed == 2
    assert r.tokens_dropped == 1
    assert r.drop_rate > 0


def test_restart_on_q0_abandons_the_in_progress_frame():
    """A fresh q0 mid-frame restarts; the tokens already banked are lost."""
    r = check([tok(1, 0), tok(2, 0), tok(2, 1), END])
    assert r.frames_committed == 1        # only the (2,*) frame completes
    assert r.tokens_dropped == 1          # the orphaned tok(1,0)


def test_foreign_token_resets_but_parsing_resumes():
    r = check(frame(1) + [55555] + frame(2) + [END])
    assert r.frames_committed == 2
    assert r.tokens_dropped == 0          # both frames were whole


def test_trailing_partial_frame_counts_as_dropped():
    r = check(frame(1) + [tok(7, 0)] + [END])
    assert r.frames_committed == 1 and r.tokens_dropped == 1


def test_drop_rate_is_a_fraction_of_audio_tokens():
    r = check([tok(1, 0), tok(2, 0), tok(2, 1), END])
    assert r.tokens_emitted == 3
    assert r.drop_rate == pytest.approx(1 / 3)


# --- composition with scoring ---------------------------------------------

def test_gate_forces_worst_score_regardless_of_transcript():
    """A malformed decode must not be scored from whatever text exists."""
    r = check([tok(1, 0)])
    s = score_pair("the reference text", "a plausible transcript", status=r.gate)
    assert s.cer == 1.0 and s.wer == 1.0
    assert s.status is Status.MALFORMED_DECODE
    assert s.valid is True            # counted as a model failure, not dropped


def test_passing_gate_scores_normally():
    r = check(frame(1) + [END])
    s = score_pair("hello world", "hello world", status=r.gate)
    assert s.cer == 0.0 and s.status is Status.OK


def test_returns_dataclass():
    assert isinstance(check([END]), Integrity)


# --- the accounting invariant ---------------------------------------------

@pytest.mark.parametrize(
    "emitted",
    [
        frame(1) + frame(2) + [END],
        frame(1) + [tok(3, 1)] + frame(4) + [END],
        [tok(1, 0), tok(2, 0), tok(2, 1), END],
        frame(1) + [tok(7, 0), END],
        frame(1) + [55555] + frame(2) + [END],
        [tok(0, 1), tok(0, 1), tok(5, 0), tok(5, 1), END],
        [END],
        [],
    ],
)
def test_every_audio_token_is_either_committed_or_dropped(emitted):
    """The accounting must balance, always:

        tokens_emitted == frames_committed * Q + tokens_dropped

    This is a stronger statement than any hand-counted example, and it is what
    makes `drop_rate` meaningful as a degradation signal.
    """
    r = check(emitted)
    assert r.tokens_emitted == r.frames_committed * Q + r.tokens_dropped
