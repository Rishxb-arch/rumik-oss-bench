"""Prompt sets: what to synthesise, and what it should sound like.

The existing `*_eval_prompts.yaml` in the training repo carry only `{text: str}`.
That is not enough to score against, for two reasons:

* The prompt contains markup the model must **not** voice - speaker prefix, tone
  tag, `<description="...">`, inline events. The spoken reference is the prompt
  with all of that removed.
* Scoring Indic audio needs a **per-prompt language** for the ASR judge.
  Hardcoding one language across a multi-script set is the upstream bug we are
  not repeating.

So a prompt row separates three things: the text sent to the model, the words
expected to be audible, and the language to transcribe in.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .normalize.markup import strip_markup
from .normalize.translit import detect_scripts


@dataclass(slots=True)
class Prompt:
    id: str
    text: str                       # what the model is asked to say, markup included
    speaker: str | None = None
    language: str = "hi"            # ASR language code for this row
    expected_seconds: float | None = None
    spoken: str | None = None       # override; otherwise derived from `text`
    tags: tuple[str, ...] = ()

    @property
    def reference(self) -> str:
        """The words that should be audible. Derived unless explicitly given."""
        return self.spoken if self.spoken is not None else strip_markup(self.text).spoken

    @property
    def scripts(self) -> tuple[str, ...]:
        return tuple(sorted(detect_scripts(self.reference)))

    @property
    def script_key(self) -> str:
        return "+".join(self.scripts) if self.scripts else "latin_only"


@dataclass(slots=True)
class PromptSet:
    name: str
    prompts: list[Prompt] = field(default_factory=list)
    version: int = 1

    def __iter__(self) -> Iterator[Prompt]:
        return iter(self.prompts)

    def __len__(self) -> int:
        return len(self.prompts)

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(sorted({p.language for p in self.prompts}))

    @property
    def script_keys(self) -> tuple[str, ...]:
        return tuple(sorted({p.script_key for p in self.prompts}))


def load_promptset(path: str | Path) -> PromptSet:
    """Load a prompt set. `defaults` fills any field a row omits."""
    import yaml

    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = data.get("defaults", {}) or {}
    rows = data.get("prompts", []) or []

    prompts: list[Prompt] = []
    for i, row in enumerate(rows):
        if isinstance(row, str):            # bare string shorthand
            row = {"text": row}
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        prompts.append(Prompt(
            id=str(row.get("id") or f"{path.stem}-{i:04d}"),
            text=text,
            speaker=row.get("speaker", defaults.get("speaker")),
            language=str(row.get("language", defaults.get("language", "hi"))),
            expected_seconds=(float(row["expected_seconds"])
                              if row.get("expected_seconds") is not None else
                              (float(defaults["expected_seconds"])
                               if defaults.get("expected_seconds") is not None else None)),
            spoken=row.get("spoken"),
            tags=tuple(row.get("tags", ()) or ()),
        ))
    return PromptSet(name=str(data.get("name", path.stem)),
                     prompts=prompts, version=int(data.get("version", 1)))


def convert_eval_prompts(path: str | Path, *, language: str = "hi") -> PromptSet:
    """Read a training-repo `*_eval_prompts.yaml` (rows of `{text: ...}`).

    Speaker and tone live inside the text there, so `reference` strips them and
    the speaker is recovered rather than assumed.
    """
    import yaml

    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    prompts: list[Prompt] = []
    for i, row in enumerate(data.get("prompts", []) or []):
        text = str((row or {}).get("text", "")).strip()
        if not text:
            continue
        stripped = strip_markup(text)
        prompts.append(Prompt(
            id=f"{path.stem}-{i:04d}",
            # The model expects the markup, so `text` keeps it; only the
            # reference is stripped.
            text=text,
            speaker=stripped.speaker.capitalize() if stripped.speaker else None,
            language=language,
            spoken=stripped.spoken,
        ))
    return PromptSet(name=path.stem, prompts=prompts)
