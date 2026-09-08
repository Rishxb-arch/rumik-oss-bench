"""Strip TinyAya prompt markup down to the text that is actually spoken.

This exists because the training repo does not do it. Its normalizer
(`ayatts_runtime/deepspeed_runner.py:_normalize_text_for_eval`) removes only
`<audio>`, `<lang_*>`, `<speak_*>` and `<|...|>`. It leaves the speaker prefix,
the tone tag, the inline event tags and `<description="...">` in the reference
string -- so every WER/CER number computed against it penalises the model for
markup it correctly did *not* voice. That is a large, systematic error on
exactly the prompts we care about.

Two prompt dialects exist in this lineage and the speaker sits on a different
side of the description tag in each:

    Ira: <description="happy, Telugu accent, fast pace"> namaste, aap kaise hain?
    <description="happy, indian accent">silver: Hey this is a demo text

so the description is removed before the speaker prefix is matched, and the
speaker is then matched at the start of what remains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Canonical tag sets, from recipes/tinyaya/mimi/pretrain/config/scenario/*.yaml
# (`data.text_special_tokens`) and docs/runbooks/tinyaya_training_from_scratch.md.
# `<cough>` is not in the 583/v4 vocab but appears in the Stage AB v3 data mix,
# so it is accepted here: stripping a tag the model never emits is harmless,
# leaving one in the reference is not.
TONE_TAGS: frozenset[str] = frozenset(
    {"happy", "angry", "neutral", "sad", "whisper", "excited"}
)
EVENT_TAGS: frozenset[str] = frozenset({"laugh", "chuckle", "sigh", "cough"})

#: Non-speech annotations that ASR systems emit inside square brackets.
#: ElevenLabs Scribe writes "[laughing]" and "[music]"; Whisper writes
#: "[BLANK_AUDIO]" and "[Music]". None of it was spoken, so scoring it charges
#: the model for the transcriber's editorial notes -- "[laughing]" alone is nine
#: characters of pure insertion error. Stripped from BOTH sides, because a
#: transformation applied to one side only is a bias, not a normalisation.
ASR_EVENT_TAGS: frozenset[str] = frozenset({
    "laughing", "laughter", "laughs", "chuckles", "chuckling", "sighs", "sighing",
    "coughing", "coughs", "clears throat", "throat clearing", "breathing",
    "music", "musical", "applause", "clapping", "noise", "background noise",
    "silence", "pause", "inaudible", "unintelligible", "indistinct", "crosstalk",
    "foreign", "foreign language", "blank_audio", "blank audio", "speaking in foreign language",
})
SPEAKERS: frozenset[str] = frozenset({"ira", "aisha", "siya", "zoya", "silver"})

_DESCRIPTION_RE = re.compile(r"""<\s*description\s*=\s*(["'])(.*?)\1\s*>""", re.I | re.S)
_TONE_RE = re.compile(r"\[\s*(" + "|".join(sorted(TONE_TAGS)) + r")\s*\]", re.I)
_ASR_EVENT_RE = re.compile(
    r"[\[(]\s*(" + "|".join(re.escape(t) for t in sorted(ASR_EVENT_TAGS, key=len, reverse=True))
    + r")\s*[\])]", re.I)
_EVENT_RE = re.compile(r"<\s*(" + "|".join(sorted(EVENT_TAGS)) + r")\s*>", re.I)
_STRUCTURAL_RE = re.compile(r"</?\s*(text|audio|speak)\s*>", re.I)
_LANG_RE = re.compile(r"<\s*lang[_-]([A-Za-z_-]+)\s*>", re.I)
_SPECIAL_PIPE_RE = re.compile(r"<\|.*?\|>", re.S)
# Flat Mimi audio-code tokens, e.g. <1573_4>. Defensive: these should never
# reach a reference string, but if a decode leaks them we must not score them.
_AUDIO_CODE_RE = re.compile(r"<\d+_\d+>")
_SPEAKER_RE = re.compile(r"^\s*(" + "|".join(sorted(SPEAKERS)) + r")\s*:\s*", re.I)
# Fallback for speakers outside the known set (opt-in; see strip_markup).
_GENERIC_SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_-]{0,15})\s*:\s*")


@dataclass(slots=True)
class Stripped:
    """Result of stripping. Keeps what was removed so callers can assert on it."""

    spoken: str
    speaker: str | None = None
    tones: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    description: str | None = None
    languages: list[str] = field(default_factory=list)
    had_audio_codes: bool = False


def strip_markup(text: str, *, allow_unknown_speaker: bool = True) -> Stripped:
    """Reduce a prompt or reference to the words that should be audible.

    `allow_unknown_speaker` also strips a leading ``Word:`` for speakers outside
    SPEAKERS. It is on by default because unseen speaker names appear in ad-hoc
    prompt sets, but it is a heuristic: a reference that genuinely begins
    "Note: ..." would lose that word. Turn it off for corpora where the leading
    token is real text.
    """
    if not text:
        return Stripped(spoken="")

    out = text
    tones: list[str] = []
    events: list[str] = []
    languages: list[str] = []

    description = None
    m = _DESCRIPTION_RE.search(out)
    if m:
        description = m.group(2).strip()
    out = _DESCRIPTION_RE.sub(" ", out)

    had_codes = bool(_AUDIO_CODE_RE.search(out))
    out = _AUDIO_CODE_RE.sub(" ", out)

    out = _SPECIAL_PIPE_RE.sub(" ", out)
    out = _STRUCTURAL_RE.sub(" ", out)

    for lang_match in _LANG_RE.finditer(out):
        languages.append(lang_match.group(1).lower().replace("-", "_"))
    out = _LANG_RE.sub(" ", out)

    for tone_match in _TONE_RE.finditer(out):
        tones.append(tone_match.group(1).lower())
    out = _TONE_RE.sub(" ", out)

    for event_match in _EVENT_RE.finditer(out):
        events.append(event_match.group(1).lower())
    out = _EVENT_RE.sub(" ", out)

    for asr_match in _ASR_EVENT_RE.finditer(out):
        events.append(asr_match.group(1).lower())
    out = _ASR_EVENT_RE.sub(" ", out)

    # Speaker last: the description may have been sitting in front of it.
    speaker = None
    sm = _SPEAKER_RE.match(out)
    if sm:
        speaker = sm.group(1).lower()
        out = out[sm.end():]
    elif allow_unknown_speaker:
        gm = _GENERIC_SPEAKER_RE.match(out)
        if gm:
            speaker = gm.group(1).lower()
            out = out[gm.end():]

    # A tone tag can follow the speaker ("Ira: [happy] text"); the pass above
    # already removed it, but re-running is cheap and covers odd orderings.
    for tone_match in _TONE_RE.finditer(out):
        tones.append(tone_match.group(1).lower())
    out = _TONE_RE.sub(" ", out)

    return Stripped(
        spoken=" ".join(out.split()),
        speaker=speaker,
        tones=tones,
        events=events,
        description=description,
        languages=languages,
        had_audio_codes=had_codes,
    )


def spoken_text(text: str, **kwargs) -> str:
    """Convenience wrapper when only the spoken string is wanted."""
    return strip_markup(text, **kwargs).spoken
