"""Regression: ASR systems annotate non-speech events, and we scored them.

ElevenLabs Scribe writes "[laughing]" and "[music]"; Whisper writes
"[BLANK_AUDIO]". None of it was spoken. Left in the hypothesis, "[laughing]"
alone is nine characters of pure insertion error charged to the model for the
transcriber's editorial note.

Found while adding Scribe as a second judge: it emitted "[laughing]" on a
Kannada clip whose reference had no such tag.
"""

from __future__ import annotations

import pytest

from rumik_oss_eval.metrics.cer_wer import score_pair
from rumik_oss_eval.normalize.markup import strip_markup
from rumik_oss_eval.normalize.pipeline import normalize_for_scoring


@pytest.mark.parametrize("tag", [
    "[laughing]", "[laughter]", "[music]", "[applause]", "[noise]",
    "[inaudible]", "[BLANK_AUDIO]", "[Music]", "(inaudible)", "[crosstalk]",
])
def test_asr_event_tags_are_removed(tag):
    out = strip_markup(f"hello there {tag} how are you").spoken
    assert "[" not in out and "]" not in out
    assert out == "hello there how are you"


def test_tag_is_recorded_not_silently_dropped():
    """Callers can assert on what was removed; a silent strip hides a judge
    that is annotating heavily."""
    assert "laughing" in strip_markup("ok [laughing] fine").events


def test_indic_text_around_the_tag_survives():
    out = strip_markup("ಸ್ವಲ್ಪ ಕಾಯಿರಿ [laughing] ಸ್ವಲ್ಪ ಕಾಯಿರಿ").spoken
    assert out == "ಸ್ವಲ್ಪ ಕಾಯಿರಿ ಸ್ವಲ್ಪ ಕಾಯಿರಿ"


def test_bracketed_indic_content_is_left_alone():
    """The tag list is a whitelist on purpose: arbitrary bracketed text may be
    real content, and stripping it would delete words the model did say."""
    assert strip_markup("यह [कुछ] है").spoken == "यह [कुछ] है"


def test_prompt_tone_tags_still_work():
    s = strip_markup("[happy] नमस्ते")
    assert s.spoken == "नमस्ते" and "happy" in s.tones


# --- the reason it matters -------------------------------------------------

def test_a_stray_tag_no_longer_costs_the_model_anything():
    ref = "ಸ್ವಲ್ಪ ಕಾಯಿರಿ ಸ್ವಲ್ಪ ಕಾಯಿರಿ"
    hyp = "ಸ್ವಲ್ಪ ಕಾಯಿರಿ [laughing] ಸ್ವಲ್ಪ ಕಾಯಿರಿ"
    assert score_pair(normalize_for_scoring(ref).folded,
                      normalize_for_scoring(hyp).folded).cer == 0.0


def test_untagged_transcripts_are_unaffected():
    ref = "main office se nikli hoon"
    assert score_pair(normalize_for_scoring(ref).folded,
                      normalize_for_scoring(ref).folded).cer == 0.0
