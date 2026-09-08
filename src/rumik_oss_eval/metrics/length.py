"""Duration penalty, computed from the token stream alone.

No audio, no model, no GPU: Mimi's frame rate is fixed, so the duration of a
generation is a function of how many *complete* frames it emitted.

    MIMI_ENCODE_DOWNSAMPLE = 1920 samples per frame   (pretokenize_emilia.py:24)
    24000 / 1920 = 12.5 frames per second
    1 frame = Q tokens = 80 ms  ->  100 tokens per second of audio at Q=8

Why a duration term belongs in a CER-driven reward: the cheapest way for a TTS
policy to reduce ASR error is to slow down and over-articulate. That is the
classic reward hack for this objective, it is invisible to CER, and a duration
band blocks it directly. The penalty is deliberately asymmetric - running long
(looping, trailing silence, repeated syllables) is punished harder than running
slightly short, because it is both the more common failure and the one that
signals a degenerate policy.

The penalty is flat zero inside a dead zone. Speech has legitimate natural
variation and we do not want gradient pressure toward some exact duration; we
only want to punish clearly wrong ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Samples per Mimi frame. Named constant in the training repo
#: (`scripts/pretokenize_emilia.py:24`), corroborated by the config comment in
#: `recipes/.../scenario/en_pretrain_v1.yaml:7` and by empirical decode
#: (T=30 -> 57,600 samples; T=300 -> 576,000).
MIMI_ENCODE_DOWNSAMPLE = 1920
MIMI_SAMPLE_RATE = 24000
FRAME_SECONDS = MIMI_ENCODE_DOWNSAMPLE / MIMI_SAMPLE_RATE  # 0.08
FRAMES_PER_SECOND = MIMI_SAMPLE_RATE / MIMI_ENCODE_DOWNSAMPLE  # 12.5

# Defaults. The dead zone is wide because natural speech-rate variation is wide;
# `long_weight` is what makes the penalty asymmetric.
DEFAULT_LO = 0.80
DEFAULT_HI = 1.25
DEFAULT_SCALE = 1.0
DEFAULT_LONG_WEIGHT = 2.0


@dataclass(slots=True)
class LengthScore:
    penalty: float          # [0, 1]; 0.0 means "within the acceptable band"
    ratio: float            # generated / expected
    generated_seconds: float
    expected_seconds: float
    verdict: str            # "ok" | "short" | "long" | "empty" | "no_expectation"
    valid: bool             # False => no usable expectation; exclude from means

    @property
    def within_band(self) -> bool:
        return self.verdict == "ok"


def seconds_from_frames(complete_frames: int) -> float:
    """Duration of `complete_frames` Mimi frames, in seconds."""
    return max(0, int(complete_frames)) * FRAME_SECONDS


def frames_from_tokens(num_tokens: int, num_quantizers: int) -> int:
    """Complete frames representable by `num_tokens` audio tokens.

    Integer division is the point: a trailing partial frame is not decodable and
    the parser drops it, so `len(emitted) / Q` is *not* the frame count.
    """
    if num_quantizers <= 0:
        raise ValueError("num_quantizers must be positive")
    return max(0, int(num_tokens)) // int(num_quantizers)


def seconds_from_tokens(num_tokens: int, num_quantizers: int) -> float:
    """Duration implied by a raw audio-token count, ignoring any partial frame."""
    return seconds_from_frames(frames_from_tokens(num_tokens, num_quantizers))


def length_penalty(
    generated_seconds: float,
    expected_seconds: float,
    *,
    lo: float = DEFAULT_LO,
    hi: float = DEFAULT_HI,
    scale: float = DEFAULT_SCALE,
    long_weight: float = DEFAULT_LONG_WEIGHT,
) -> LengthScore:
    """Asymmetric dead-zone penalty on the duration ratio.

        ratio inside [lo, hi]        -> 0.0
        ratio below lo               -> clamp(|log(r/lo)| / scale, 0, 1)
        ratio above hi               -> clamp(long_weight * |log(r/hi)| / scale, 0, 1)

    Log-space because durations are naturally multiplicative: being 2x too long
    and 2x too short should be comparable distances before `long_weight` tilts
    them.

    A missing expectation is a data problem, not a model failure, so it scores
    0.0 with `valid=False` - the sample should be excluded from means rather
    than punished for something the model did not do.
    """
    if lo <= 0 or hi <= 0 or lo > hi:
        raise ValueError(f"invalid dead zone: lo={lo} hi={hi}")
    if scale <= 0:
        raise ValueError("scale must be positive")

    if expected_seconds is None or expected_seconds <= 0:
        return LengthScore(0.0, float("nan"), float(generated_seconds or 0.0),
                           float(expected_seconds or 0.0), "no_expectation", False)

    generated_seconds = float(generated_seconds or 0.0)

    # Nothing decodable: worst possible, and it must never be NaN - the whole
    # point of the reward-safety rules in cer_wer.py.
    if generated_seconds <= 0:
        return LengthScore(1.0, 0.0, 0.0, float(expected_seconds), "empty", True)

    ratio = generated_seconds / float(expected_seconds)
    if lo <= ratio <= hi:
        verdict, raw = "ok", 0.0
    elif ratio < lo:
        verdict, raw = "short", abs(math.log(ratio / lo)) / scale
    else:
        verdict, raw = "long", long_weight * abs(math.log(ratio / hi)) / scale

    return LengthScore(
        penalty=min(1.0, max(0.0, raw)),
        ratio=ratio,
        generated_seconds=generated_seconds,
        expected_seconds=float(expected_seconds),
        verdict=verdict,
        valid=True,
    )


def length_penalty_from_tokens(
    num_tokens: int,
    num_quantizers: int,
    expected_seconds: float,
    **kwargs,
) -> LengthScore:
    """Convenience wrapper: raw audio-token count -> penalty."""
    return length_penalty(
        seconds_from_tokens(num_tokens, num_quantizers), expected_seconds, **kwargs
    )
