"""Combine the metric terms into one bounded scalar reward.

Three properties matter more than the exact weights, because they are what make
the number safe to optimise against:

1. **Bounded in [0, 1].** GRPO normalises advantages within a group; an
   unbounded or NaN term wrecks that.
2. **Hard gates run first.** A malformed decode, a truncated generation, or an
   untrustworthy transcript scores zero outright. Scoring such a sample from
   whatever partial text survived produces a plausible number, which is worse
   than an obvious failure because nothing downstream can tell it went wrong.
3. **Every component is emitted per sample.** Weights can then be retuned
   offline from a stored run, without re-running generation or ASR — which
   matters enormously once RL sweeps start.

The default weighting leans on CER. WER is reported but not leaned on: romanized
Hinglish has no fixed spelling, so word-level agreement is far noisier than
character-level agreement even when the speech is right.
"""

from __future__ import annotations

import math as _math

from dataclasses import dataclass, field

from .asr.base import AsrStatus
from .metrics.cer_wer import Score, Status
from .metrics.integrity import Integrity
from .metrics.length import LengthScore


@dataclass(slots=True, frozen=True)
class RubricWeights:
    cer: float = 0.7
    wer: float = 0.2
    length: float = 0.1
    #: Grapheme cluster edit rate. 0.0 keeps the historical behaviour exactly.
    #: Set --w-cer 0 --w-gcer 0.6 to switch the content term over: `cer` splits
    #: code points, which understates error on every Indic script here and by a
    #: different factor per script.
    gcer: float = 0.0
    #: Teacher forced CTC NLL from the ASR, per target token. Dense where the
    #: edit rates tie. Needs a scorer to be wired in; 0.0 leaves it inert.
    nll: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"cer": self.cer, "wer": self.wer, "length": self.length,
                "gcer": self.gcer, "nll": self.nll}


@dataclass(slots=True)
class RubricResult:
    reward: float
    terms: dict[str, float] = field(default_factory=dict)      # raw metric values
    contributions: dict[str, float] = field(default_factory=dict)  # weighted, post-renorm
    gate: str | None = None
    status: str = "ok"
    valid: bool = True

    @property
    def gated(self) -> bool:
        return self.gate is not None


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def score_sample(
    *,
    score: Score,
    integrity: Integrity | None = None,
    length: LengthScore | None = None,
    asr_status: AsrStatus | str = AsrStatus.OK,
    weights: RubricWeights | None = None,
    nll: float | None = None,
    nll_scale: float = 4.0,
) -> RubricResult:
    """Reward for one generation. Gates first, then a weighted sum of goodness.

    A term whose input is unavailable (no expected duration, say) is **dropped
    and the remaining weights renormalised** — not scored zero, which would be a
    free penalty, and not scored one, which would be free credit.
    """
    weights = weights or RubricWeights()
    asr_status = AsrStatus(asr_status) if not isinstance(asr_status, AsrStatus) else asr_status

    terms = {
        "gcer": float(getattr(score, "gcer", 1.0)),
        "nll": float(nll) if nll is not None else float("nan"),
        "cer": float(score.cer),
        "wer": float(score.wer),
        "length_penalty": float(length.penalty) if length is not None else 0.0,
    }
    if integrity is not None:
        terms["frames"] = float(integrity.frames_committed)
        terms["drop_rate"] = float(integrity.drop_rate)
        terms["emitted_eos"] = float(integrity.emitted_eos)

    # --- hard gates -------------------------------------------------------
    gate: str | None = None
    if integrity is not None and integrity.gate is not None:
        gate = str(integrity.gate)
    elif asr_status is not AsrStatus.OK:
        # An untrustworthy transcript is a measurement failure. It is still the
        # model's sample, so it is counted (valid=True) and scored worst.
        gate = f"asr:{asr_status}"
    elif score.status in {Status.MALFORMED_DECODE, Status.NO_EOS,
                          Status.EMPTY_HYPOTHESIS, Status.ASR_FAILED}:
        gate = str(score.status)

    if gate is not None:
        return RubricResult(reward=0.0, terms=terms, contributions={},
                            gate=gate, status=gate, valid=score.valid)

    # A missing reference is a corpus bug, not something the model did: score it
    # zero so nothing can exploit it, but mark it invalid so means exclude it.
    if not score.valid:
        return RubricResult(reward=0.0, terms=terms, contributions={},
                            gate=None, status=str(score.status), valid=False)

    # --- weighted sum over the terms we actually have ---------------------
    available: dict[str, tuple[float, float]] = {
        "cer": (weights.cer, 1.0 - _clamp01(score.cer)),
        "wer": (weights.wer, 1.0 - _clamp01(score.wer)),
    }
    if weights.gcer:
        available["gcer"] = (weights.gcer,
                             1.0 - _clamp01(float(getattr(score, "gcer", 1.0))))
    if weights.nll and nll is not None and nll == nll:
        # NLL is unbounded above, so it is squashed into [0,1] before entering a
        # weighted sum whose other terms are bounded. nll_scale is the per token
        # NLL that maps to a goodness of exp(-1); it wants calibrating per
        # language, since a language the ASR models poorly has a higher floor.
        available["nll"] = (weights.nll, _clamp01(_math.exp(-float(nll) / max(nll_scale, 1e-6))))
    if length is not None and length.valid:
        available["length"] = (weights.length, 1.0 - _clamp01(length.penalty))

    total_w = sum(w for w, _ in available.values())
    if total_w <= 0:
        return RubricResult(reward=0.0, terms=terms, contributions={},
                            status="no_weights", valid=False)

    contributions = {k: (w / total_w) * g for k, (w, g) in available.items()}
    reward = _clamp01(sum(contributions.values()))

    return RubricResult(reward=reward, terms=terms, contributions=contributions,
                        gate=None, status="ok", valid=True)
