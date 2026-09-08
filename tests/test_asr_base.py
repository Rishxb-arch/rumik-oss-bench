"""Judge-failure semantics. These run on CPU with synthetic audio, no model."""

from __future__ import annotations

import numpy as np
import pytest

from rumik_oss_eval.asr.base import (
    UNTRUSTWORTHY,
    AsrBackend,
    AsrResult,
    AsrStatus,
    Clip,
    classify,
    is_degenerate,
    is_silent,
)


def speech(seconds: float = 1.0, rate: int = 24_000) -> np.ndarray:
    """Non-silent audio: a tone, loud enough to clear the RMS gate."""
    t = np.linspace(0, seconds, int(rate * seconds), dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# --- silence detection -----------------------------------------------------

def test_silence_detected():
    assert is_silent(np.zeros(24_000, dtype=np.float32))
    assert is_silent(np.array([], dtype=np.float32))
    assert is_silent(np.full(24_000, 1e-6, dtype=np.float32))


def test_real_audio_is_not_silent():
    assert not is_silent(speech())


# --- degenerate output -----------------------------------------------------

def test_looping_output_is_degenerate():
    """Whisper's signature hallucination: one phrase repeated to fill the window."""
    assert is_degenerate("hai hai hai hai hai hai hai hai hai hai")


def test_normal_transcript_is_not_degenerate():
    assert not is_degenerate("namaste main abhi office se nikli hoon raaste mein traffic hai")


def test_short_transcript_is_never_degenerate():
    """Too little evidence to call it looping."""
    assert not is_degenerate("hai hai hai")


# --- classification order --------------------------------------------------

def test_silence_outranks_empty_text():
    """Silence is a fact about the input, so it beats any output-side check."""
    assert classify("", np.zeros(1000, dtype=np.float32)) is AsrStatus.SILENT_INPUT


def test_silent_input_with_hallucinated_text_is_still_silent():
    """The dangerous case: real-looking text over silence must not read OK."""
    assert classify("namaste kaise hain aap", np.zeros(24_000, dtype=np.float32)) is AsrStatus.SILENT_INPUT


def test_empty_text_over_real_audio():
    assert classify("   ", speech()) is AsrStatus.EMPTY


def test_good_transcript_is_ok():
    assert classify("namaste main ghar ja raha hoon abhi", speech()) is AsrStatus.OK


# --- result semantics ------------------------------------------------------

@pytest.mark.parametrize("status", sorted(UNTRUSTWORTHY))
def test_untrustworthy_statuses_are_flagged(status):
    assert AsrResult(text="anything", status=status).trustworthy is False


def test_ok_is_trustworthy():
    assert AsrResult(text="namaste", status=AsrStatus.OK).trustworthy is True


def test_clip_seconds():
    assert Clip(audio=np.zeros(48_000, dtype=np.float32), sample_rate=24_000).seconds == pytest.approx(2.0)
    assert Clip(audio=np.zeros(0, dtype=np.float32), sample_rate=24_000).seconds == 0.0


# --- the Protocol ----------------------------------------------------------

def test_a_fake_backend_satisfies_the_protocol():
    """Swapping judges must be cheap; this pins the contract."""

    class FakeAsr:
        name = "fake"

        def transcribe(self, clips):
            return [AsrResult(text="x", status=AsrStatus.OK, clip_id=c.clip_id) for c in clips]

    backend = FakeAsr()
    assert isinstance(backend, AsrBackend)
    out = backend.transcribe([Clip(speech(), 24_000, clip_id="a")])
    assert len(out) == 1 and out[0].clip_id == "a"


def test_results_preserve_input_order_contract():
    """Order matters: callers zip results back against references positionally."""

    class OrderedFake:
        name = "ordered"

        def transcribe(self, clips):
            return [AsrResult(text=c.clip_id, status=AsrStatus.OK, clip_id=c.clip_id) for c in clips]

    clips = [Clip(speech(0.1), 24_000, clip_id=f"c{i}", language="hi" if i % 2 else "en")
             for i in range(5)]
    out = OrderedFake().transcribe(clips)
    assert [r.clip_id for r in out] == [c.clip_id for c in clips]
