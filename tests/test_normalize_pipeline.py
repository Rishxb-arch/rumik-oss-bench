"""Romanization, folding, and the end-to-end normalization contract.

The headline test is `test_real_asr_pair_*`: a genuine (prompt, ASR output) pair
captured from this stack -- TinyAya generated the audio, the local AI4Bharat
IndicConformer transcribed it, and the speech is correct. Scored raw it looks
like total failure; scored normalized it looks correct. Those two assertions
together are the entire justification for this package.
"""

from __future__ import annotations

import pytest

from rumik_oss_eval.metrics.cer_wer import score_pair
from rumik_oss_eval.normalize.fold import fold, strip_diacritics
from rumik_oss_eval.normalize.pipeline import normalize_for_scoring
from rumik_oss_eval.normalize.translit import detect_scripts, has_indic, romanize


# --- the real captured pair ------------------------------------------------

REAL_PROMPT = ("Silver: [happy] नमस्ते, main abhi office se nikli hoon <laugh>, "
               "raaste mein traffic kaafi light hai.")
REAL_ASR = "नमस्ते मैं अभी ऑफिस से निकली हूँ रास्ते में ट्रैफिक काफ़ी लाइट है"


def test_real_asr_pair_scores_terribly_without_normalization():
    """Baseline: mixed-script reference vs single-script ASR looks like failure."""
    s = score_pair(REAL_PROMPT, REAL_ASR)
    assert s.cer > 0.85
    assert s.wer >= 0.9


def test_real_asr_pair_scores_correctly_after_normalization():
    """Same audio, same transcript, honest number. This is the whole point."""
    ref = normalize_for_scoring(REAL_PROMPT)
    hyp = normalize_for_scoring(REAL_ASR)
    s = score_pair(ref.folded, hyp.folded)
    assert s.cer < 0.10, f"CER regressed to {s.cer:.3f}: {ref.folded!r} vs {hyp.folded!r}"
    assert s.wer < 0.20, f"WER regressed to {s.wer:.3f}"


def test_real_pair_word_alignment():
    """All but the known 'traffic' mismatch should align exactly."""
    ref = normalize_for_scoring(REAL_PROMPT).folded.split()
    hyp = normalize_for_scoring(REAL_ASR).folded.split()
    assert len(ref) == len(hyp)
    mismatches = [(a, b) for a, b in zip(ref, hyp) if a != b]
    # Hindi spells 'traffic' with ऐ, English with 'a' -- a real pronunciation
    # difference, not a normalizer bug. Anything beyond it is a regression.
    assert len(mismatches) <= 1, f"unexpected mismatches: {mismatches}"


# --- transliteration -------------------------------------------------------

def test_detect_scripts():
    assert detect_scripts("नमस्ते") == {"devanagari"}
    assert detect_scripts("તમે ફરીથી") == {"gujarati"}
    assert detect_scripts("hello") == set()
    assert detect_scripts("नमस्ते and విను") == {"devanagari", "telugu"}
    assert has_indic("main नमस्ते") and not has_indic("plain latin")


@pytest.mark.parametrize(
    "text,script",
    [("नमस्ते", "devanagari"), ("তমি", "bengali"), ("ਸਤ", "gurmukhi"),
     ("તમે", "gujarati"), ("విను", "telugu"), ("ಬರಲು", "kannada")],
)
def test_every_prompt_script_romanizes_to_latin(text, script):
    """All seven scripts in the eval prompts must leave the Indic range."""
    out = romanize(text)
    assert out and not has_indic(out), f"{script} left residue: {out!r}"


def test_latin_passes_through_romanize_untouched():
    assert romanize("main abhi office se") == "main abhi office se"


def test_code_switched_string_comes_out_all_latin():
    out = romanize("नमस्ते, main office se nikli")
    assert not has_indic(out)
    assert "main office se nikli" in out


def test_iso_handles_the_o_vowel_that_other_schemes_drop():
    """U+0911 ऑ (English 'o' in ऑफिस) survives IAST/ITRANS/HK untransliterated;
    ISO is used precisely because it does not."""
    assert not has_indic(romanize("ऑफिस"))


# --- schwa deletion --------------------------------------------------------

def test_schwa_deleted_in_penultimate_syllable():
    """Transliteration restores the inherent vowel that Hinglish omits."""
    assert "nikli" in fold(romanize("निकली"))


def test_schwa_not_deleted_from_first_syllable_regression():
    """An earlier rule turned 'namaste' into 'nmaste', destroying a real vowel."""
    out = romanize("नमस्ते")
    assert out.startswith("nama"), out


# --- folding ---------------------------------------------------------------

def test_strip_diacritics():
    assert strip_diacritics("namastē maiṁ abhī rāstē ṭraiphika") == \
        "namaste maim abhi raste traiphika"


@pytest.mark.parametrize(
    "a,b",
    [("office", "ofis"), ("hoon", "hun"), ("kaafi", "kafi"),
     ("nikli", "nikli"), ("light", "let"), ("phone", "fon")],
)
def test_english_words_fold_to_phonetic_forms(a, b):
    assert fold(a) == b


def test_soft_c_regression():
    """A blanket c->k turned 'office' into 'ofike' and broke the match."""
    assert fold("office") == "ofis"
    assert fold("cat") == "kat"          # hard c still becomes k


def test_short_words_keep_their_final_vowel_regression():
    """The silent-e rule used to eat the vowel in 'se', turning it into 's'."""
    assert fold("se") == "se"
    assert fold("de") == "de"
    # m/n are deliberately merged (maiṁ ~ main), so "me" -> "ne" is intended
    # behaviour, not the silent-e bug this test guards against.
    assert fold("me") == "ne"


def test_spelling_variants_of_the_same_word_converge():
    """Romanized Hinglish has no fixed spelling; folding must absorb that."""
    for group in [("hoon", "hun"), ("kaafi", "kafi"), ("raaste", "raste")]:
        folded = {fold(w) for w in group}
        assert len(folded) == 1, f"{group} -> {folded}"


def test_punctuation_and_case_removed():
    assert fold("Hello, World!") == fold("hello world")


# --- pipeline contract -----------------------------------------------------

def test_pipeline_retains_every_stage_for_debugging():
    n = normalize_for_scoring(REAL_PROMPT)
    assert n.raw == REAL_PROMPT
    assert "[happy]" not in n.spoken and "Silver:" not in n.spoken
    assert not has_indic(n.romanized)
    assert n.folded and n.folded == n.folded.lower()
    assert n.speaker == "silver" and n.tones == ("happy",) and n.events == ("laugh",)
    assert n.scripts == frozenset({"devanagari"})


def test_pipeline_is_idempotent_on_its_own_output():
    once = normalize_for_scoring(REAL_ASR).folded
    twice = normalize_for_scoring(once).folded
    assert once == twice


def test_empty_input():
    n = normalize_for_scoring("")
    assert n.folded == "" and n.scripts == frozenset()


def test_same_function_used_for_both_sides():
    """Reference and hypothesis must not be prepared differently -- any
    asymmetry becomes fake error."""
    text = "नमस्ते main office"
    assert normalize_for_scoring(text).folded == normalize_for_scoring(text).folded


# --- regressions found by calibration on real hinglish60k audio ------------

CH_PAIRS = [
    ("Poochh", "पूछ"), ("chhakke", "छक्के"), ("chal", "चल"),
    ("achha", "अच्छा"), ("bachche", "बच्चे"), ("chaar", "चार"), ("kuch", "कुछ"),
]
AA_PAIRS = [("bacha", "बचा"), ("laga", "लगा"), ("gaya", "गया"), ("kaafi", "काफ़ी")]


@pytest.mark.parametrize("latin,deva", CH_PAIRS)
def test_ch_convention_converges_regression(latin, deva):
    """Romanized Hinglish spells च as 'ch'; ISO 15919 spells it bare 'c'.

    Two bugs conspired here. In fold(), the blanket c->k ran BEFORE the chh/ch
    rules and destroyed the 'c' they needed, so "Poochh" folded to "pukh". And
    the Indic side emitted bare 'c', which then hit the hard-c rule and became
    'k'. Every ch/chh word in the corpus mismatched, inflating the noise floor.
    """
    assert normalize_for_scoring(latin).folded == normalize_for_scoring(deva).folded


@pytest.mark.parametrize("latin,deva", AA_PAIRS)
def test_final_long_a_survives_regression(latin, deva):
    """Schwa deletion must not eat a real final 'ā'.

    Diacritics used to be stripped before the schwa rules ran, which made the
    inherent schwa 'a' and the long vowel 'ā' indistinguishable - so "बचा"
    became "bac" while romanized "bacha" stayed "baca". Hindi words ending in ā
    are everywhere, so this was systematic.
    """
    assert normalize_for_scoring(latin).folded == normalize_for_scoring(deva).folded


def test_hard_and_soft_c_still_work_after_the_ch_fix():
    """The ch protection must not break English c handling."""
    assert fold("cat") == "kat"        # hard c
    assert fold("office") == "ofis"    # soft c
    assert fold("city") == "siti"


def test_shared_danda_does_not_imply_devanagari_regression():
    """। (U+0964) sits in the Devanagari block but is shared Indic punctuation.

    Without excluding it, Bengali or Gurmukhi text ending in a danda reports as
    containing Devanagari, which corrupts per-script reporting - and per-script
    reporting is how we will know whether the non-Latin scripts actually work.
    """
    assert detect_scripts("আমি আজ স্টেশনে আটকে আছি।") == {"bengali"}
    assert detect_scripts("ਸਤ ਸ੍ਰੀ ਅਕਾਲ।") == {"gurmukhi"}
    assert detect_scripts("नमस्ते।") == {"devanagari"}      # real Devanagari still detected


def test_numeral_vocabulary_is_fully_built_regression():
    """num2words must be installed, not silently skipped.

    Its guarded import once left the VM with an 83-entry numeral vocabulary
    while the dev machine had 224 - two environments scoring differently with
    no error raised anywhere, and a threshold tuned against the wrong one.
    """
    from rumik_oss_eval.normalize.numbers import GENERATED_AVAILABLE, _NUMBER_WORDS
    assert GENERATED_AVAILABLE, "num2words missing: numeral vocabulary is incomplete"
    assert len(_NUMBER_WORDS) > 150
