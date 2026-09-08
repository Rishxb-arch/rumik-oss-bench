"""Script-routed judge selection."""

from __future__ import annotations

import numpy as np
import pytest

from rumik_oss_eval.asr.base import AsrBackend, AsrResult, AsrStatus, Clip
from rumik_oss_eval.asr.routed import ScriptRoutedAsr


class Stub:
    def __init__(self, label):
        self.name = label
        self.seen = []

    def transcribe(self, clips):
        self.seen.extend(c.clip_id for c in clips)
        return [AsrResult(text=self.name, status=AsrStatus.OK, clip_id=c.clip_id,
                          language=c.language) for c in clips]


def clip(cid, lang="hi"):
    return Clip(audio=np.zeros(1600, dtype=np.float32), sample_rate=16000,
                clip_id=cid, language=lang)


@pytest.mark.parametrize("ref,expected", [
    ("नमस्ते, आप कैसे हैं?", "indic"),
    ("આવ્યા છું", "indic"),
    ("ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "indic"),
    ("நாளைக்கு", "indic"),
    ("main abhi office se nikli hoon", "latin"),
    ("The quick brown fox", "latin"),
    ("", "latin"),
])
def test_route_follows_the_reference_script(ref, expected):
    assert ScriptRoutedAsr.route_for(ref) == expected


def test_code_switched_reference_routes_indic():
    """Any Indic content sends it to the Indic judge; that is the side whose
    recognition matters most on such audio."""
    assert ScriptRoutedAsr.route_for("नमस्ते, main office se nikli hoon") == "indic"


def test_clips_go_to_the_right_backend():
    indic, latin = Stub("icf"), Stub("whisper")
    r = ScriptRoutedAsr(indic=indic, latin=latin)
    clips = [clip("a"), clip("b"), clip("c")]
    refs = ["नमस्ते", "hello there", "ನಾನು"]
    out = r.transcribe_with_references(clips, refs)
    assert indic.seen == ["a", "c"]
    assert latin.seen == ["b"]
    assert [x.text for x in out] == ["icf", "whisper", "icf"]


def test_order_is_preserved_across_buckets():
    """Callers zip results back positionally, so interleaving must not reorder."""
    r = ScriptRoutedAsr(indic=Stub("icf"), latin=Stub("whisper"))
    clips = [clip(f"c{i}") for i in range(6)]
    refs = ["नमस्ते", "latin", "नमस्ते", "latin", "latin", "नमस्ते"]
    got = [x.clip_id for x in r.transcribe_with_references(clips, refs)]
    assert got == [f"c{i}" for i in range(6)]


def test_length_mismatch_is_rejected():
    r = ScriptRoutedAsr(indic=Stub("icf"), latin=Stub("whisper"))
    with pytest.raises(ValueError):
        r.transcribe_with_references([clip("a")], ["x", "y"])


def test_protocol_fallback_routes_on_language():
    """Without references, an Indic language code still reaches the Indic judge."""
    indic, latin = Stub("icf"), Stub("whisper")
    r = ScriptRoutedAsr(indic=indic, latin=latin)
    r.transcribe([clip("a", "hi"), clip("b", "en")])
    assert indic.seen == ["a"] and latin.seen == ["b"]


def test_satisfies_the_protocol():
    assert isinstance(ScriptRoutedAsr(indic=Stub("i"), latin=Stub("l")), AsrBackend)


def test_name_records_both_judges():
    assert ScriptRoutedAsr(indic=Stub("icf"), latin=Stub("whisper")).name == "routed(icf|whisper)"
