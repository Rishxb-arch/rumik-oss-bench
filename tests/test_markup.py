"""Markup stripping, tested against strings taken verbatim from the real corpora.

Sources of the fixtures below:
  * the hinglish60k style corpus (internal)
  * the hinglish60k style eval prompt set (internal)
  * Pranavz/583 README (description dialect, speaker first)
  * rumik-ai/silk-oss-v2 usage snippet (description dialect, speaker second)
"""

from __future__ import annotations

import pytest

from rumik_oss_eval.normalize.markup import Stripped, strip_markup, spoken_text


# --- real hinglish60k rows -------------------------------------------------

def test_hinglish60k_tone_prefix():
    row = "[angry] What? So pehle toh tum mera phone check kar rahe the, that's not enough for you?"
    got = strip_markup(row)
    assert got.tones == ["angry"]
    assert got.speaker is None
    assert got.spoken.startswith("What? So pehle toh tum")
    assert "[angry]" not in got.spoken


def test_hinglish60k_tone_and_inline_event():
    row = "[happy] <laugh> Programming ka funda chalu hai kya? Last name match karne mein help chahiye na?"
    got = strip_markup(row)
    assert got.tones == ["happy"]
    assert got.events == ["laugh"]
    assert got.spoken.startswith("Programming ka funda")
    assert "<laugh>" not in got.spoken


# --- eval prompt dialect: speaker + tone + inline event, multi-script ------

def test_eval_prompt_speaker_tone_event_devanagari():
    row = ("Ira: [happy] नमस्ते, main abhi office se nikli hoon <laugh>, "
           "रास्ते में traffic kaafi light hai.")
    got = strip_markup(row)
    assert got.speaker == "ira"
    assert got.tones == ["happy"]
    assert got.events == ["laugh"]
    # Devanagari must survive untouched; only the markup goes.
    assert got.spoken.startswith("नमस्ते, main abhi office se nikli hoon")
    assert "रास्ते में traffic kaafi light hai." in got.spoken
    for junk in ("Ira:", "[happy]", "<laugh>"):
        assert junk not in got.spoken


@pytest.mark.parametrize(
    "row,speaker,script_fragment",
    [
        ("Aisha: [angry] તમે ફરીથી late આવ્યા? main already pandrah minute wait kar chuki hoon.",
         "aisha", "તમે ફરીથી late આવ્યા?"),
        ("Siya: [whisper] విను, secret code hai: zero nine alpha, please kisi ko mat bolna <sigh>.",
         "siya", "విను, secret code hai"),
        ("Zoya: [sad] আমি আজ station e আটকে আছি, train baar baar cancel ho rahi hai.",
         "zoya", "আমি আজ station e আটকে আছি"),
        ("Silver: [neutral] ਸਤ ਸ੍ਰੀ ਅਕਾਲ, order number P N seven four hai.",
         "silver", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ"),
    ],
)
def test_eval_prompts_all_scripts(row, speaker, script_fragment):
    got = strip_markup(row)
    assert got.speaker == speaker
    assert script_fragment in got.spoken
    assert not got.spoken.startswith(speaker.capitalize() + ":")


def test_speaker_colon_inside_text_is_kept():
    """'secret code hai: zero nine alpha' -- only the LEADING speaker is stripped."""
    row = "Siya: [whisper] विनु, secret code hai: zero nine alpha."
    got = strip_markup(row)
    assert got.speaker == "siya"
    assert "secret code hai: zero nine alpha." in got.spoken


# --- description dialect, both speaker orderings ---------------------------

def test_description_then_speaker_silk_dialect():
    row = ('<description="happy, indian accent">silver: '
           'Hey this is a demo text you know? is it all right?')
    got = strip_markup(row)
    assert got.description == "happy, indian accent"
    assert got.speaker == "silver"
    assert got.spoken == "Hey this is a demo text you know? is it all right?"


def test_speaker_then_description_583_dialect():
    row = ('Ira: <description="happy, Telugu accent, fast pace"> '
           'namaste, aap kaise hain? main bilkul theek hoon.')
    got = strip_markup(row)
    assert got.description == "happy, Telugu accent, fast pace"
    assert got.speaker == "ira"
    assert got.spoken == "namaste, aap kaise hain? main bilkul theek hoon."


# --- structural / defensive ------------------------------------------------

def test_structural_and_lang_tokens_removed():
    row = "<text><lang_hi>Ira: [neutral] hello there<audio></audio>"
    got = strip_markup(row)
    assert got.languages == ["hi"]
    assert got.spoken == "hello there"


def test_audio_code_tokens_are_flagged_and_removed():
    """Flat Mimi code tokens must never be scored as if they were speech."""
    got = strip_markup("Ira: hello <1573_4><12_0> world")
    assert got.had_audio_codes is True
    assert got.spoken == "hello world"


def test_empty_and_markup_only_inputs():
    assert strip_markup("").spoken == ""
    assert strip_markup("   ").spoken == ""
    # A prompt that is nothing but markup has no spoken content at all.
    got = strip_markup('Ira: [happy] <laugh> <description="x">')
    assert got.spoken == ""
    assert got.tones == ["happy"] and got.events == ["laugh"]


def test_whitespace_is_collapsed_not_merged_across_words():
    got = strip_markup("Ira:  [happy]   main   ghar   ja   raha   hoon  ")
    assert got.spoken == "main ghar ja raha hoon"


def test_unknown_speaker_can_be_disabled():
    row = "Note: this is the actual sentence."
    assert strip_markup(row, allow_unknown_speaker=False).spoken == row
    assert strip_markup(row, allow_unknown_speaker=True).spoken == "this is the actual sentence."


def test_spoken_text_wrapper_matches():
    row = "Ira: [happy] hello <laugh>"
    assert spoken_text(row) == strip_markup(row).spoken


def test_returns_dataclass():
    assert isinstance(strip_markup("hi"), Stripped)
