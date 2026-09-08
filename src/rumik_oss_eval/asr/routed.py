"""Pick the judge by the script the *reference* is written in.

Measured, same audio and same normalizer:

    reference script      IndicConformer   Whisper
    native Indic (n=13)         0.0643      0.1521
    romanized Latin (n=300)     0.1168      0.0800      (numeric refs excluded)

Neither judge is better in general. IndicConformer always emits Indic script, so
against a romanized-Latin reference its output has to cross the transliteration
bridge and pays for it. Whisper can echo Latin directly, but is far weaker at
actually recognising Indic speech.

Routing on the reference's script therefore picks the judge whose output
convention matches what it will be compared against. That is not cheating: the
reference is the target text, and choosing a transcription convention is not the
same as leaking the answer. The judge still has to hear the audio correctly.

The routing key comes from `detect_scripts`, the same function the per-script
reporting uses, so a row's judge and its reported script bucket always agree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..normalize.translit import detect_scripts
from .base import AsrBackend, AsrResult, Clip


@dataclass(slots=True)
class ScriptRoutedAsr(AsrBackend):
    """Route each clip to a judge based on its reference text's script.

    `indic` handles references written in an Indic script; `latin` handles
    romanized and English references.
    """

    indic: AsrBackend
    latin: AsrBackend
    name: str = field(default="routed")

    def __post_init__(self) -> None:
        self.name = f"routed({self.indic.name}|{self.latin.name})"

    @staticmethod
    def route_for(reference: str) -> str:
        """'indic' if the reference uses an Indic script, else 'latin'."""
        return "indic" if detect_scripts(reference or "") else "latin"

    def transcribe_with_references(
        self, clips: Sequence[Clip], references: Sequence[str]
    ) -> list[AsrResult]:
        """Transcribe, routing each clip by its reference's script.

        Kept separate from `transcribe` because the plain Protocol has no place
        for the reference - a judge normally never sees it.
        """
        if len(clips) != len(references):
            raise ValueError("clips and references must be the same length")

        buckets: dict[str, list[int]] = {}
        for i, ref in enumerate(references):
            buckets.setdefault(self.route_for(ref), []).append(i)

        out: list[AsrResult | None] = [None] * len(clips)
        for key, idxs in buckets.items():
            backend = self.indic if key == "indic" else self.latin
            for i, r in zip(idxs, backend.transcribe([clips[i] for i in idxs])):
                out[i] = r
        return [r for r in out if r is not None]

    def transcribe(self, clips: Sequence[Clip]) -> list[AsrResult]:
        """Protocol conformance: with no reference available, route on the
        clip's declared language instead - Indic language codes go to the Indic
        judge. Prefer `transcribe_with_references` when references exist."""
        from .indicconformer import SUPPORTED

        refs = ["ऀ" if (c.language or "") in SUPPORTED else "" for c in clips]
        return self.transcribe_with_references(clips, refs)
