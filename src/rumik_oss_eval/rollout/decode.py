"""Generated tokens -> Mimi codes -> waveform.

The span selection here is the one that matters: always the LAST `<audio>` span.
In an ICL / voice-clone prompt the reference audio is itself wrapped in
`<audio>...</audio>`, so taking the first span decodes the reference clip — audio
that sounds perfectly fine and has nothing to do with what the model produced.
Every real call site in the training repo uses the last-span variant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..metrics.integrity import Integrity, check_integrity
from ..vendor.audio_token_parser import (extract_audio_codes_bqt_from_token_ids,
                                         normalize_waveform_for_logging)


@dataclass(slots=True)
class Decoded:
    audio: np.ndarray | None          # 1-D float32 at bundle.sample_rate
    sample_rate: int
    frames: int
    integrity: Integrity

    @property
    def seconds(self) -> float:
        return (len(self.audio) / self.sample_rate) if self.audio is not None else 0.0

    @property
    def ok(self) -> bool:
        return self.audio is not None


def codes_from_emitted(bundle, emitted):
    """Emitted ids -> codes (1, Q, T), or None when no complete frame exists.

    `emitted` is post-prompt, so the whole list is the generated span and there
    is no earlier `<audio>` to confuse things. The span helper is still applied
    for the case where a caller passes a full sequence including the prompt.
    """
    import torch

    ids = list(emitted)
    if bundle.audio_start_id in ids:
        from ..vendor.audio_token_parser import get_last_audio_span_indices
        start, end = get_last_audio_span_indices(ids, bundle.audio_start_id, bundle.audio_end_id)
    else:
        start = 0
        end = ids.index(bundle.audio_end_id) if bundle.audio_end_id in ids else len(ids)

    codes = extract_audio_codes_bqt_from_token_ids(
        ids, bundle.num_quantizers, bundle.code_map, start_idx=start, end_idx=end
    )
    if codes is None:
        return None
    # The parser hands back a transpose, i.e. a non-contiguous view.
    return codes.contiguous() if isinstance(codes, torch.Tensor) else codes


def decode_generation(bundle, generation, *, max_new_tokens: int | None = None) -> Decoded:
    """Full token -> audio path, with the structural gates evaluated alongside."""
    import torch

    integrity = check_integrity(
        generation.emitted,
        num_quantizers=bundle.num_quantizers,
        audio_code_id_map=bundle.code_map,
        audio_end_id=bundle.audio_end_id,
        max_new_tokens=max_new_tokens,
    )

    codes = codes_from_emitted(bundle, generation.emitted)
    if codes is None:
        return Decoded(None, bundle.sample_rate, 0, integrity)

    with torch.no_grad():
        decoded = bundle.codec.decode(codes.to(bundle.codec_device))
    values = getattr(decoded, "audio_values", None)
    if values is None:                      # some adapters return a tuple
        values = decoded[0]
    audio = normalize_waveform_for_logging(values[0].float().cpu().numpy())

    return Decoded(audio, bundle.sample_rate, int(codes.shape[2]), integrity)


def write_wav(audio: np.ndarray, path, sample_rate: int) -> None:
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
