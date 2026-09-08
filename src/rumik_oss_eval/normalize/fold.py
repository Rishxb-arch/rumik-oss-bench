"""Fold Latin text into a coarse phonetic space so spelling variance stops
counting as speech error.

After romanization both sides are Latin, but they are still spelled by two
different conventions: English orthography on one side ("office", "traffic",
"light") and ISO transliteration of how an Indian speaker pronounces those same
words on the other ("ôphisa", "ṭraiphika", "lāiṭa"). On top of that, romanized
Hinglish has no fixed spelling at all -- hoon/hun, kaafi/kafi, nikli/nikalee are
all attested for the same word.

Folding collapses both conventions onto shared approximate phones. It is lossy
by design: it trades some ability to detect fine phonetic errors for the much
larger gain of not reporting 0.90 CER on correct audio.

Every rule below is justified by a real mismatch observed on this stack's
output, not added speculatively -- each one is pinned by a test.
"""

from __future__ import annotations

import re
import unicodedata

# Digraphs and English spellings, longest-first. Order matters: 'igh' must fire
# before 'gh', and soft c before the blanket c->k.
_RULES: tuple[tuple[str, str], ...] = (
    ("ough", "o"), ("augh", "a"), ("igh", "ai"),
    ("tion", "san"), ("sion", "san"),
    ("ph", "f"), ("ck", "k"),
    ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"),
    ("bh", "b"), ("jh", "j"), ("sh", "s"),
    ("qu", "kw"), ("x", "ks"),
    ("ee", "i"), ("oo", "u"), ("aa", "a"), ("ie", "i"), ("ea", "i"),
    ("ai", "e"), ("ay", "e"), ("ei", "e"),
    ("au", "o"), ("ow", "o"), ("ou", "u"),
    ("q", "k"), ("z", "j"), ("w", "v"), ("y", "i"),
)

_SOFT_C_RE = re.compile(r"c(?=[eiy])")
_DOUBLE_RE = re.compile(r"(.)\1+")
_SILENT_E_RE = re.compile(r"(?<=[bcdfgjklmnprstv])e\b")
_NASAL_RE = re.compile(r"[mn]+")
_TRAILING_VOWELS_RE = re.compile(r"([aeiou])[aeiou]+\b")


def strip_diacritics(text: str) -> str:
    """Drop combining marks: ē->e, ī->i, ū->u, ā->a, ṁ->m, ṭ->t, m̐->m.

    ISO 15919 encodes vowel length and retroflexion as diacritics, none of
    which romanized Hinglish writes, so they are pure noise for matching.
    """
    decomposed = unicodedata.normalize("NFD", text or "")
    without = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", without)


# The "ch" sound is protected through the blanket c->k pass with a sentinel.
# Ordering here is load-bearing and was a real bug: running c->k first destroys
# the "c" that the chh/ch rules need, so "Poochh" folded to "pukh" while the
# Devanagari it should match ("पूछ" -> "pucha") folded to "puc". Every ch/chh
# word in the corpus was mismatching, which inflated the measured noise floor.
_CH_SENTINEL = "\x01"


def _fold_word(word: str) -> str:
    word = _SOFT_C_RE.sub("s", word)      # offi[c]e -> s
    word = word.replace("chh", "ch")      # collapse the aspirate first
    word = word.replace("ch", _CH_SENTINEL)   # protect the ch sound
    word = word.replace("c", "k")         # remaining hard c
    word = word.replace(_CH_SENTINEL, "c")
    for src, dst in _RULES:
        word = word.replace(src, dst)
    # Silent-e only on words long enough for the "e" to plausibly be silent.
    # Ungated, this eats the real vowel in short words: "se" -> "s", "de" -> "d".
    if len(word) > 3:
        word = _SILENT_E_RE.sub("", word)  # offise -> offis
    word = _DOUBLE_RE.sub(r"\1", word)  # traffic -> trafic, hoon handled above
    word = _NASAL_RE.sub("n", word)    # m/n merge: maim ~ main
    word = _TRAILING_VOWELS_RE.sub(r"\1", word)
    return word


def fold(text: str) -> str:
    """Casefold, strip punctuation and diacritics, then apply phonetic rules."""
    if not text:
        return ""
    text = strip_diacritics(text).casefold()
    # Keep letters/digits/space; everything else becomes a separator. This also
    # removes the punctuation that an ASR does not emit but a reference has.
    text = "".join(
        ch if (ch.isalnum() or ch.isspace()) else " " for ch in text
    )
    return " ".join(_fold_word(w) for w in text.split() if w)
