"""Regression: characters indic-transliteration silently passes through.

`transliterate()` returns Malayalam chillu letters unchanged rather than
raising, so they survived romanization *and* folding and reached the edit
distance as raw Indic codepoints. Measured on the indic_v2 prompts before the
fix: 46 un-romanized characters across 5 Malayalam samples, 3-5% of every folded
string - in one of the two languages the model is worst at, so it was corrupting
exactly the numbers we most needed to trust.

The class of bug matters more than the five characters: a silent pass-through
looks like working code. These tests assert the *absence* of Indic codepoints
downstream, which catches the next such character too.
"""

from __future__ import annotations

import pytest

from rumik_oss_eval.normalize.fold import fold
from rumik_oss_eval.normalize.pipeline import normalize_for_scoring
from rumik_oss_eval.normalize.translit import romanize


def indic_left(text: str) -> str:
    """Characters above the Latin/diacritic range that survived."""
    return "".join(c for c in text if ord(c) > 0x2FF)


# --- the five Malayalam chillus ------------------------------------------

@pytest.mark.parametrize("chillu,expected", [
    ("ൽ", "l"),      # U+0D7D chillu L
    ("ർ", "r"),      # U+0D7C chillu RR
    ("ൾ", "l"),      # U+0D7E chillu LL  (ḷ, marks stripped)
    ("ൺ", "n"),      # U+0D7A chillu NN  (ṇ, marks stripped)
    ("ൻ", "n"),      # U+0D7B chillu N
])
def test_chillu_romanizes_to_its_consonant(chillu, expected):
    got = romanize(chillu)
    assert indic_left(got) == "", f"{chillu!r} survived as {got!r}"
    assert got == expected


def test_chillu_k_is_covered_too():
    """ൿ is rarer but the same failure; included so the table is complete."""
    assert indic_left(romanize("ൿ")) == ""


# --- in running text ------------------------------------------------------

def test_malayalam_word_with_chillu_fully_romanizes():
    # "ഒടുവിൽ" (finally) - the chillu is word-final, where it is most common.
    out = romanize("ഒടുവിൽ")
    assert indic_left(out) == "", f"chillu survived in {out!r}"
    assert out.endswith("l")


def test_chillu_does_not_survive_folding():
    """fold() runs after romanize() and would happily carry an Indic codepoint
    through to the edit distance."""
    assert indic_left(fold(romanize("ഒടുവിൽ പൂർണ്ണമായി"))) == ""


def test_full_pipeline_leaves_no_malayalam_codepoints():
    text = ("എത്ര കാത്തിരുന്ന ജോലി ഒടുവിൽ പൂർണ്ണമായി തീർന്നെന്നറിഞ്ഞപ്പോൾ "
            "ഹൃദയം ഭാരംകുറഞ്ഞുപോയി.")
    n = normalize_for_scoring(text)
    assert indic_left(n.romanized) == "", f"romanized: {indic_left(n.romanized)!r}"
    assert indic_left(n.folded) == "", f"folded: {indic_left(n.folded)!r}"


# --- Assamese wa ----------------------------------------------------------

def test_assamese_wa_romanizes():
    """U+09F1 is in the Bengali block so it routed to the Bengali scheme, which
    then passed it through untouched."""
    assert indic_left(romanize("ৱ")) == ""


def test_assamese_sentence_leaves_no_codepoints():
    n = normalize_for_scoring("মোৰ গাঁৱৰ ওচৰত এটা পথাৰ আছে।")
    assert indic_left(n.folded) == "", f"folded: {indic_left(n.folded)!r}"


# --- the scripts that already worked must not regress ---------------------

@pytest.mark.parametrize("text", [
    "नमस्ते, आप कैसे हैं?",              # devanagari
    "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਹਾਡਾ ਆਰਡਰ ਤਿਆਰ ਹੈ।",  # gurmukhi
    "நாளைக்கு கடற்கரைக்கு போகலாம்",       # tamil
    "నేను ఈరోజు చాలా సంతోషంగా ఉన్నాను",   # telugu
    "ನಾನು ಈಗ ಮನೆಗೆ ಬರಲು ಆಗುತ್ತಿಲ್ಲ",       # kannada
    "আমি ভালো আছি",                       # bengali
    "હું ઘરે જાઉં છું",                    # gujarati
    "ମୁଁ ଭଲ ଅଛି",                          # oriya
])
def test_other_scripts_still_fully_romanize(text):
    assert indic_left(normalize_for_scoring(text).folded) == ""


def test_latin_is_untouched_by_the_fixups():
    assert romanize("main office se nikli hoon") == "main office se nikli hoon"
