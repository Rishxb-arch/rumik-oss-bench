"""Romanize Indic scripts to a common Latin space, span by span.

Why this exists: our references are code-switched across up to seven scripts,
while an ASR returns a single script of its own choosing. Measured on real
output from this stack, a correct utterance scores CER 0.90 against a
mixed-script reference -- indistinguishable from total failure. Mapping both
sides into one alphabet before scoring is what makes the number mean anything.

ISO 15919 is used rather than IAST/ITRANS/HK because it is the only one of the
four that transliterates the full Devanagari range we see in practice: the
others leave characters such as U+0911 (ऑ, used for the English 'o' in ऑफिस)
untransliterated, which then survives into the folded string as a foreign
codepoint and counts as pure error.

Schwa deletion is applied *here*, per span, rather than in the folding stage.
Transliteration restores Devanagari's inherent vowel ("निकली" -> "nikalī")
whereas romanized Hinglish already omits it ("nikli"). The correction therefore
belongs only to text that came from an Indic script; applying it globally also
eats real vowels in English words ("kaafi" -> "kfi", "traffic" -> "trfik"),
which costs the metric discriminating power for no benefit.
"""

from __future__ import annotations

import re
import unicodedata

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

#: Unicode blocks for the scripts that appear in the eval prompts, plus the
#: rest of the IndicConformer language set so ASR output is always covered.
_SCRIPT_RANGES: dict[str, tuple[str, str]] = {
    "devanagari": ("ऀ", "ॿ"),   # hi, mr, ne, sa, doi, mai, brx, ks, kok, sd
    "bengali": ("ঀ", "৿"),      # bn, as, mni
    "gurmukhi": ("਀", "੿"),     # pa
    "gujarati": ("઀", "૿"),     # gu
    "oriya": ("଀", "୿"),        # or
    "tamil": ("஀", "௿"),        # ta
    "telugu": ("ఀ", "౿"),       # te
    "kannada": ("ಀ", "೿"),      # kn
    "malayalam": ("ഀ", "ൿ"),    # ml
}

_SANSCRIPT_NAME: dict[str, str] = {
    "devanagari": sanscript.DEVANAGARI,
    "bengali": sanscript.BENGALI,
    "gurmukhi": sanscript.GURMUKHI,
    "gujarati": sanscript.GUJARATI,
    "oriya": sanscript.ORIYA,
    "tamil": sanscript.TAMIL,
    "telugu": sanscript.TELUGU,
    "kannada": sanscript.KANNADA,
    "malayalam": sanscript.MALAYALAM,
}

_SPAN_RES: dict[str, re.Pattern[str]] = {
    name: re.compile(f"[{lo}-{hi}]+") for name, (lo, hi) in _SCRIPT_RANGES.items()
}
_ANY_INDIC_RE = re.compile(
    "[" + "".join(f"{lo}-{hi}" for lo, hi in _SCRIPT_RANGES.values()) + "]"
)

# Hindi schwa deletion, applied only to transliterated spans. The classic rule
# drops the inherent vowel of a non-final syllable when the following syllable
# carries its own vowel: nikalī -> niklī. Iterated because a word can contain
# more than one deletable schwa.
#
# Schwa rules run on the ISO text BEFORE diacritics are stripped, so that the
# inherent schwa "a" stays distinguishable from a real long "ā". Stripping first
# conflates them and deletes genuine final vowels: "बचा" -> "baca" -> "bac",
# which then never matches the romanized "bacha" -> "baca". Hindi words ending
# in ā are everywhere, so that was a systematic error.
#
# The consonant class therefore has to include ISO's marked consonants (ṭ ḍ ṇ ...),
# which is what stripping first was working around.
_CONS = "kgcjtdnpbmyrlvshfzṭḍṇṣśñṅṛḷ"
_VOWELS = "aeiouāīūēōṛ"

# The leading vowel is load-bearing: it anchors the rule to the penultimate
# syllable and never the first. Without it "namaste" becomes "nmaste", which
# destroys a real vowel and costs the metric discriminating power.
_SCHWA_MEDIAL_RE = re.compile(
    rf"([{_VOWELS}][{_CONS}])a([{_CONS}])([{_VOWELS}])"
)
#: Only a bare "a" is an inherent schwa; "ā" is a real vowel and must survive.
_SCHWA_FINAL_RE = re.compile(rf"([{_CONS}])a\b")


def _strip_marks(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    )


#: Danda and double danda live in the Devanagari block but are shared
#: punctuation across Indic scripts, so Bengali or Gurmukhi text ending in "।"
#: would otherwise be reported as containing Devanagari. They are stripped by
#: folding anyway; excluding them here only keeps per-script reporting honest.
_SHARED_PUNCT = str.maketrans({"।": " ", "॥": " "})


def detect_scripts(text: str) -> set[str]:
    """Which Indic scripts appear in `text` (empty set for pure Latin)."""
    cleaned = (text or "").translate(_SHARED_PUNCT)
    return {name for name, rx in _SPAN_RES.items() if rx.search(cleaned)}


def has_indic(text: str) -> bool:
    return bool(_ANY_INDIC_RE.search(text or ""))


def _delete_schwa(word: str) -> str:
    prev = None
    while prev != word:
        prev = word
        word = _SCHWA_MEDIAL_RE.sub(r"\1\2\3", word)
    return _SCHWA_FINAL_RE.sub(r"\1", word)


#: Characters `indic-transliteration` passes through unchanged, mapped to a form
#: it does know. Without this they survive romanization *and* folding, so they
#: reach the edit distance as raw Indic codepoints and every one is a guaranteed
#: mismatch against a hypothesis spelled any other way.
#:
#: Malayalam chillus are consonants written without an inherent vowel; the
#: standard decomposition is consonant + virama, which ISO-15919 then handles
#: (ൽ -> ല് -> "l"). Measured on the indic_v2 prompts before this fix: 46
#: un-romanized characters across 5 Malayalam samples, 3-5% of each folded
#: string, in one of the two languages the model is worst at.
_PASSTHROUGH_FIXUPS = {
    "ൽ": "ല്",   # U+0D7D chillu L
    "ർ": "ര്",   # U+0D7C chillu RR
    "ൾ": "ള്",   # U+0D7E chillu LL
    "ൺ": "ണ്",   # U+0D7A chillu NN
    "ൻ": "ന്",   # U+0D7B chillu N
    "ൿ": "ക്",   # U+0D7F chillu K
    "ৱ": "ৰ",    # U+09F1 Assamese wa -> ra, which BENGALI does transliterate
}
_FIXUP_RE = re.compile("[" + "".join(_PASSTHROUGH_FIXUPS) + "]")


def _apply_fixups(span: str) -> str:
    return _FIXUP_RE.sub(lambda m: _PASSTHROUGH_FIXUPS[m.group(0)], span)


def _romanize_span(span: str, script: str) -> str:
    span = _apply_fixups(span)
    latin = transliterate(span, _SANSCRIPT_NAME[script], sanscript.ISO)
    # Diacritics are stripped BEFORE schwa deletion, not after. ISO writes
    # retroflex and long forms as combining marks, so "lāiṭa" ends in
    # U+1E6D + "a" and would never match an ASCII consonant class. Stripping
    # first lets the schwa rules see plain letters. fold() would strip these
    # anyway, so nothing is lost downstream.
    # Schwa deletion FIRST, while "a" and "ā" are still distinguishable, then
    # strip marks. Schwa deletion is word-wise; a span can hold several words.
    latin = " ".join(_delete_schwa(w) if w else w for w in latin.split(" "))
    latin = _strip_marks(latin)
    # ISO 15919 writes च as bare "c" and छ as "ch", but romanized Hinglish spells
    # both "ch". Left alone, "चल" -> "cal" folds to "kal" (the blanket hard-c
    # rule) while the romanized "chal" folds to "cal", and they never match.
    # Re-spell both to the "ch" convention here, where we still know the text
    # came from an Indic script; fold() cannot make that distinction.
    latin = latin.replace("ch", "c")    # छ -> च, collapse the aspirate
    latin = latin.replace("c", "ch")    # both now use the Hinglish spelling
    return latin


def romanize(text: str) -> str:
    """Replace every Indic span with its ISO-15919 romanization.

    Latin text passes through untouched, so a code-switched string comes out
    entirely in Latin with both halves in the same alphabet.
    """
    if not text:
        return ""
    out = text
    for script, rx in _SPAN_RES.items():
        if rx.search(out):
            out = rx.sub(lambda m, s=script: _romanize_span(m.group(0), s), out)
    return out
