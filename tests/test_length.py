"""Duration penalty: the timing arithmetic and the shape of the penalty curve."""

from __future__ import annotations

import math

import pytest

from rumik_oss_eval.metrics.length import (
    FRAME_SECONDS,
    FRAMES_PER_SECOND,
    MIMI_ENCODE_DOWNSAMPLE,
    MIMI_SAMPLE_RATE,
    frames_from_tokens,
    length_penalty,
    length_penalty_from_tokens,
    seconds_from_frames,
    seconds_from_tokens,
)


# --- the timing constants (a wrong number here silently skews every reward) ---

def test_mimi_timing_identities():
    assert MIMI_ENCODE_DOWNSAMPLE == 1920
    assert MIMI_SAMPLE_RATE == 24000
    assert FRAMES_PER_SECOND == 12.5
    assert FRAME_SECONDS == 0.08
    # 8 quantizers x 12.5 frames/s = 100 tokens per second of audio
    assert 8 * FRAMES_PER_SECOND == 100


@pytest.mark.parametrize("frames,seconds", [(0, 0.0), (1, 0.08), (12.5 * 2, 2.0), (300, 24.0)])
def test_seconds_from_frames(frames, seconds):
    assert seconds_from_frames(int(frames)) == pytest.approx(seconds)


def test_empirically_observed_decode_lengths():
    """Cross-check against real decode output captured from the codec:
    T=30 -> 57,600 samples and T=300 -> 576,000 samples, both at 24 kHz."""
    for frames, samples in [(30, 57_600), (300, 576_000)]:
        assert frames * MIMI_ENCODE_DOWNSAMPLE == samples
        assert seconds_from_frames(frames) == pytest.approx(samples / MIMI_SAMPLE_RATE)


# --- partial frames are not decodable -------------------------------------

def test_partial_trailing_frame_is_not_counted():
    """len(emitted)/Q is NOT the frame count: the parser drops partial frames."""
    assert frames_from_tokens(8, 8) == 1
    assert frames_from_tokens(15, 8) == 1        # 7 trailing tokens are dropped
    assert frames_from_tokens(16, 8) == 2
    assert frames_from_tokens(7, 8) == 0         # no complete frame at all
    assert seconds_from_tokens(15, 8) == pytest.approx(0.08)


def test_frames_from_tokens_rejects_bad_q():
    with pytest.raises(ValueError):
        frames_from_tokens(16, 0)


# --- the dead zone ---------------------------------------------------------

@pytest.mark.parametrize("ratio", [0.80, 0.9, 1.0, 1.1, 1.25])
def test_inside_dead_zone_is_free(ratio):
    s = length_penalty(ratio * 10.0, 10.0)
    assert s.penalty == 0.0
    assert s.verdict == "ok" and s.within_band and s.valid


@pytest.mark.parametrize("ratio", [0.79, 0.5, 1.26, 2.0])
def test_outside_dead_zone_is_penalised(ratio):
    assert length_penalty(ratio * 10.0, 10.0).penalty > 0.0


def test_verdicts():
    assert length_penalty(5.0, 10.0).verdict == "short"
    assert length_penalty(20.0, 10.0).verdict == "long"
    assert length_penalty(10.0, 10.0).verdict == "ok"


# --- asymmetry: running long is worse than running short -------------------

def test_long_is_penalised_harder_than_equally_short():
    """2x too long must cost more than 2x too short - over-generation signals a
    degenerate/looping policy, and slow over-articulation is the CER reward hack
    this term exists to block."""
    too_long = length_penalty(20.0, 10.0)     # ratio 2.0
    too_short = length_penalty(5.0, 10.0)     # ratio 0.5
    assert too_long.penalty > too_short.penalty


def test_asymmetry_is_controlled_by_long_weight():
    sym_long = length_penalty(20.0, 10.0, long_weight=1.0).penalty
    sym_short = length_penalty(5.0, 10.0, long_weight=1.0).penalty
    # With the tilt removed the two sides are comparable in log space
    # (|log(2/1.25)| vs |log(0.5/0.8)| are equal by construction).
    assert sym_long == pytest.approx(sym_short, abs=1e-9)


# --- monotonicity and bounds ----------------------------------------------

def test_penalty_increases_monotonically_away_from_the_band():
    longer = [length_penalty(r * 10.0, 10.0).penalty for r in (1.25, 1.5, 2.0, 3.0, 6.0)]
    assert longer == sorted(longer)
    shorter = [length_penalty(r * 10.0, 10.0).penalty for r in (0.80, 0.6, 0.4, 0.2, 0.05)]
    assert shorter == sorted(shorter)


@pytest.mark.parametrize("gen", [0.001, 0.5, 5.0, 10.0, 100.0, 10_000.0])
def test_penalty_always_within_unit_interval(gen):
    s = length_penalty(gen, 10.0)
    assert 0.0 <= s.penalty <= 1.0


def test_extreme_deviations_saturate_at_one():
    assert length_penalty(1000.0, 1.0).penalty == 1.0
    assert length_penalty(0.001, 100.0).penalty == 1.0


# --- failure semantics, mirroring cer_wer ---------------------------------

def test_no_audio_scores_worst_and_is_counted():
    """Emitting nothing is a model failure: worst score, never NaN, and counted."""
    s = length_penalty(0.0, 10.0)
    assert s.penalty == 1.0 and s.verdict == "empty"
    assert s.valid is True
    assert not math.isnan(s.penalty)


def test_missing_expectation_is_excluded_not_punished():
    """No expected duration is a corpus problem, not something the model did."""
    for bad in (0.0, -1.0, None):
        s = length_penalty(6.0, bad)
        assert s.verdict == "no_expectation"
        assert s.valid is False
        assert s.penalty == 0.0


def test_invalid_configuration_raises():
    with pytest.raises(ValueError):
        length_penalty(5.0, 10.0, lo=1.5, hi=1.2)
    with pytest.raises(ValueError):
        length_penalty(5.0, 10.0, scale=0.0)


# --- token-count entry point ----------------------------------------------

def test_from_tokens_matches_manual_computation():
    # 600 tokens at Q=8 -> 75 frames -> 6.0 s, against a 6.0 s expectation
    s = length_penalty_from_tokens(600, 8, 6.0)
    assert s.generated_seconds == pytest.approx(6.0)
    assert s.penalty == 0.0 and s.verdict == "ok"


def test_hundred_tokens_per_second_holds_only_on_frame_boundaries():
    """'100 tokens/second' is exact only when the count divides evenly by Q.

    100 tokens at Q=8 is 12 complete frames plus a 4-token remainder that is not
    decodable, so it is 0.96 s and not 1.0 s. Rounding it up would overstate the
    duration of every truncated generation.
    """
    assert seconds_from_tokens(100, 8) == pytest.approx(0.96)   # 12 frames, 4 dropped
    assert seconds_from_tokens(800, 8) == pytest.approx(8.0)    # exactly 100 frames
