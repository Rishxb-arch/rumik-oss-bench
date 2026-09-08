"""IndicConformer backend: routing and Protocol conformance (no model loaded)."""

from __future__ import annotations

import numpy as np
import pytest

from rumik_oss_eval.asr.base import AsrBackend, AsrResult, AsrStatus, Clip
from rumik_oss_eval.asr.indicconformer import SUPPORTED, IndicConformerAsr


def clip(lang, cid="c"):
    t = np.linspace(0, 0.5, 8000, dtype=np.float32)
    return Clip(audio=(0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
                sample_rate=16000, clip_id=cid, language=lang)


class FakeFallback:
    name = "fake-fallback"

    def transcribe(self, clips):
        return [AsrResult(text="fallback", status=AsrStatus.OK, clip_id=c.clip_id,
                          language=c.language) for c in clips]


def test_satisfies_the_protocol():
    assert isinstance(IndicConformerAsr(), AsrBackend)


def test_decoder_must_be_valid():
    IndicConformerAsr(decoder="ctc")
    IndicConformerAsr(decoder="rnnt")
    with pytest.raises(ValueError):
        IndicConformerAsr(decoder="beam")


def test_default_decoder_is_rnnt():
    """RNNT measured 0.0643 mean CER vs CTC 0.0716 on the Indic prompt set."""
    assert IndicConformerAsr().decoder == "rnnt"
    assert IndicConformerAsr().name == "indicconformer:rnnt"


def test_english_is_not_supported():
    """The model covers the 22 scheduled languages; English is not one."""
    assert "en" not in SUPPORTED
    assert {"hi", "gu", "pa", "bn", "te", "ta", "kn", "mr"} <= SUPPORTED
    assert len(SUPPORTED) == 22


def test_unsupported_language_fails_loudly_without_a_fallback():
    """Silently mistranscribing English would yield a plausible, wrong CER."""
    out = IndicConformerAsr().transcribe([clip("en", "e1")])
    assert len(out) == 1
    assert out[0].status is AsrStatus.FAILED
    assert out[0].text == ""


def test_unsupported_language_routes_to_the_fallback():
    asr = IndicConformerAsr(fallback=FakeFallback())
    out = asr.transcribe([clip("en", "e1")])
    assert out[0].text == "fallback" and out[0].status is AsrStatus.OK


def test_routing_preserves_order_when_mixed():
    """Callers zip results back against references positionally."""
    asr = IndicConformerAsr(fallback=FakeFallback())
    clips = [clip("en", "a"), clip("en", "b"), clip("en", "c")]
    assert [r.clip_id for r in asr.transcribe(clips)] == ["a", "b", "c"]


def test_empty_input():
    assert IndicConformerAsr().transcribe([]) == []
