"""Structural checks on a generation, derived from the token stream alone.

These are **gates on CER/WER, not extra rubric terms**. A generation that never
closed its audio span, or that produced no decodable frame, must score worst
rather than yield a plausible number from whatever partial text survived. The
`Status` enum in `cer_wer.py` already carries the corresponding values; this
module is what decides which one applies.

Free to compute: no ASR, no codec, no audio. Just the emitted ids and the
vocabulary map, so this can gate a rollout before paying for decode + ASR — which
matters when an RL step produces hundreds of generations and some fraction are
structurally dead on arrival.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .cer_wer import Status


@dataclass(slots=True)
class Integrity:
    tokens_emitted: int
    frames_committed: int
    tokens_dropped: int
    emitted_eos: bool
    hit_token_cap: bool

    @property
    def decode_ok(self) -> bool:
        return self.frames_committed > 0

    @property
    def drop_rate(self) -> float:
        """Share of audio tokens that never formed a complete frame.

        A healthy generation is ~0. A rising drop rate means the model is
        emitting quantizers out of order, which is a real degradation signal
        that CER alone would only see indirectly.
        """
        total = self.tokens_emitted
        return (self.tokens_dropped / total) if total else 0.0

    @property
    def gate(self) -> Status | None:
        """The status that must override scoring, or None if the sample is scorable.

        Order matters: a stream with no frames is malformed regardless of whether
        it also hit the cap, and that is the more specific diagnosis.
        """
        if not self.decode_ok:
            return Status.MALFORMED_DECODE
        if not self.emitted_eos:
            return Status.NO_EOS
        return None

    @property
    def passed(self) -> bool:
        return self.gate is None


def check_integrity(
    emitted: Sequence[int],
    *,
    num_quantizers: int,
    audio_code_id_map: Mapping[int, tuple[int, int]],
    audio_end_id: int,
    max_new_tokens: int | None = None,
) -> Integrity:
    """Walk the emitted stream with the same state machine the parser uses.

    `emitted` is what generation produced *after* the prompt, so index 0 is the
    first generated token. Counting here rather than reusing the parser keeps the
    dropped-token tally, which the parser discards.
    """
    frames = 0
    dropped = 0
    partial = 0
    expected_q = 0
    audio_tokens = 0
    saw_eos = False

    for tok in emitted:
        if tok == audio_end_id:
            saw_eos = True
            break
        mapped = audio_code_id_map.get(tok)
        if mapped is None:
            dropped += partial
            partial, expected_q = 0, 0
            continue
        audio_tokens += 1
        _code, q_idx = mapped
        if q_idx == expected_q:
            partial += 1
            expected_q += 1
            if expected_q == num_quantizers:
                frames += 1
                partial, expected_q = 0, 0
        elif q_idx == 0:
            # A fresh q0 abandons the in-progress frame but starts a new one,
            # so this token itself is not lost.
            dropped += partial
            partial, expected_q = 1, 1
        else:
            # Out of order and not a restart: the banked partial AND this token
            # are both unusable. Counting the stray keeps the invariant
            #   tokens_emitted == frames_committed * Q + tokens_dropped
            dropped += partial + 1
            partial, expected_q = 0, 0

    dropped += partial                  # trailing incomplete frame is undecodable

    hit_cap = bool(max_new_tokens) and len(emitted) >= int(max_new_tokens) and not saw_eos

    return Integrity(
        tokens_emitted=audio_tokens,
        frames_committed=frames,
        tokens_dropped=dropped,
        emitted_eos=saw_eos,
        hit_token_cap=hit_cap,
    )
