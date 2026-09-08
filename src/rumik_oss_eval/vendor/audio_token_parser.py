"""VENDORED — do not edit to change behaviour.

Source: tinyaya-training-main/torchtitan/tools/audio_token_parser.py
Copied verbatim except for the two marked adaptations below, so that the
upstream contract test (tests/test_audio_token_parser.py, also vendored) passes
against this file unmodified. That test is the only real protection against
drifting from the training repo's tokenisation, and drift here produces audio
that sounds plausible and is wrong.

Adaptations:
  1. `transformers.LogitsProcessor` is imported optionally. It is a marker base
     class; HF's LogitsProcessorList does not isinstance-check, so the fallback
     keeps this a drop-in without making transformers a hard dependency.
  2. `extract_frames()` is ADDED (not modified) below the vendored code: a
     pure-Python (T, Q) view of the same state machine, so the frame logic and
     the (T,Q) -> (Q,T) transpose can be tested independently. The transpose is
     the trap this package exists to avoid getting wrong.

Two conventions this file locks in, both load-bearing:
  * `build_audio_code_id_map` returns {token_id: (code, q)} — code first.
  * `extract_audio_codes_bqt_from_token_ids` returns (B=1, Q, T) — quantizer
    major, which is what MimiModel.decode expects. The training data path is
    (T, Q) frame-major everywhere else.
"""

import re
from collections.abc import Mapping, Sequence

import numpy as np
import torch
try:  # vendored adaptation 1
    from transformers import LogitsProcessor
except ImportError:  # transformers is not a hard dependency here
    class LogitsProcessor:  # type: ignore[no-redef]
        pass


_AUDIO_TOKEN_RE = re.compile(r"^<(\d+)_(\d+)>$")
_SPARK_SEMANTIC_TOKEN_RE = re.compile(r"^<\|bicodec_semantic_(\d+)\|>$")
_SPARK_GLOBAL_TOKEN_RE = re.compile(r"^<\|bicodec_global_(\d+)\|>$")


def build_audio_code_id_map(vocab: Mapping[str, int]) -> dict[int, tuple[int, int]]:
    code_id_map: dict[int, tuple[int, int]] = {}
    for token, token_id in vocab.items():
        match = _AUDIO_TOKEN_RE.match(token)
        if match:
            code_id_map[token_id] = (int(match.group(1)), int(match.group(2)))
            continue
        spark_match = _SPARK_SEMANTIC_TOKEN_RE.match(token)
        if spark_match:
            # Spark BiCodec semantic stream is single-codebook.
            code_id_map[token_id] = (int(spark_match.group(1)), 0)
    return code_id_map


def build_spark_global_id_map(vocab: Mapping[str, int]) -> dict[int, int]:
    global_id_map: dict[int, int] = {}
    for token, token_id in vocab.items():
        match = _SPARK_GLOBAL_TOKEN_RE.match(token)
        if match:
            global_id_map[token_id] = int(match.group(1))
    return global_id_map


def filter_tokens_by_attention_mask(
    token_ids: Sequence[int], attention_mask: Sequence[int]
) -> list[int]:
    return [tok for tok, mask in zip(token_ids, attention_mask) if int(mask) == 1]


def get_audio_span_indices(
    token_ids: Sequence[int], audio_start_id: int | None, audio_end_id: int | None
) -> tuple[int, int]:
    start_idx = 0
    if audio_start_id is not None and audio_start_id in token_ids:
        start_idx = token_ids.index(audio_start_id) + 1

    end_idx = len(token_ids)
    if audio_end_id is not None:
        for i in range(start_idx, len(token_ids)):
            if token_ids[i] == audio_end_id:
                end_idx = i
                break

    return start_idx, end_idx


def get_last_audio_span_indices(
    token_ids: Sequence[int], audio_start_id: int | None, audio_end_id: int | None
) -> tuple[int, int]:
    if audio_start_id is None:
        return 0, len(token_ids)

    last_start_idx = None
    for idx in range(len(token_ids) - 1, -1, -1):
        if token_ids[idx] == audio_start_id:
            last_start_idx = idx
            break

    if last_start_idx is None:
        return 0, len(token_ids)

    start_idx = last_start_idx + 1
    end_idx = len(token_ids)
    if audio_end_id is not None:
        for idx in range(start_idx, len(token_ids)):
            if token_ids[idx] == audio_end_id:
                end_idx = idx
                break

    return start_idx, end_idx


def extract_audio_codes_bqt_from_token_ids(
    token_ids: Sequence[int],
    num_quantizers: int,
    audio_code_id_map: Mapping[int, tuple[int, int]],
    audio_start_id: int | None = None,
    audio_end_id: int | None = None,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> torch.Tensor | None:
    if not audio_code_id_map:
        return None

    if start_idx is None or end_idx is None:
        start_idx, end_idx = get_audio_span_indices(
            token_ids, audio_start_id=audio_start_id, audio_end_id=audio_end_id
        )

    segment = token_ids[start_idx:end_idx]
    vals: list[int] = []
    expected_q = 0
    frame_vals: list[int] = []

    for tok_id in segment:
        mapped = audio_code_id_map.get(tok_id)
        if mapped is None:
            expected_q = 0
            frame_vals.clear()
            continue

        code, q_idx = mapped
        if q_idx == expected_q:
            frame_vals.append(code)
            expected_q += 1
            if expected_q == num_quantizers:
                vals.extend(frame_vals)
                frame_vals.clear()
                expected_q = 0
        elif q_idx == 0:
            frame_vals = [code]
            expected_q = 1
        else:
            expected_q = 0
            frame_vals.clear()

    if not vals:
        return None
    vals = vals[: len(vals) - len(vals) % num_quantizers]
    if not vals:
        return None

    tensor_btq = torch.tensor(vals).reshape(1, -1, num_quantizers)
    return tensor_btq.transpose(1, 2)  # (B, Q, T)


def extract_spark_global_token_ids(
    token_ids: Sequence[int],
    spark_global_id_map: Mapping[int, int],
    start_global_id: int | None = None,
    end_global_id: int | None = None,
) -> torch.Tensor | None:
    if not spark_global_id_map:
        return None

    if (
        start_global_id is not None
        and end_global_id is not None
        and start_global_id in token_ids
    ):
        start_idx = token_ids.index(start_global_id) + 1
        end_idx = len(token_ids)
        for i in range(start_idx, len(token_ids)):
            if token_ids[i] == end_global_id:
                end_idx = i
                break
        candidate = [spark_global_id_map[tok] for tok in token_ids[start_idx:end_idx] if tok in spark_global_id_map]
        if candidate:
            return torch.tensor(candidate, dtype=torch.long).unsqueeze(0)

    # Fallback: collect all global ids in sequence order.
    collected = [spark_global_id_map[tok] for tok in token_ids if tok in spark_global_id_map]
    if not collected:
        return None
    return torch.tensor(collected, dtype=torch.long).unsqueeze(0)


def normalize_waveform_for_logging(audio_np: np.ndarray) -> np.ndarray:
    if audio_np.ndim > 1:
        audio_np = audio_np[0]
    audio_np = np.ascontiguousarray(audio_np.reshape(-1).astype(np.float32))
    return np.clip(audio_np, -1.0, 1.0)


class AllowTokenIdsLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: Sequence[int]):
        # keep only unique non-negative ids
        uniq = sorted({int(i) for i in allowed_token_ids if int(i) >= 0})
        self.allowed_token_ids = torch.tensor(uniq, dtype=torch.long)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.allowed_token_ids.numel() == 0:
            return scores

        allowed = self.allowed_token_ids.to(scores.device)
        allowed = allowed[allowed < scores.size(1)]
        if allowed.numel() == 0:
            return scores

        filtered = torch.full_like(scores, torch.finfo(scores.dtype).min)
        filtered.index_copy_(1, allowed, scores.index_select(1, allowed))
        return filtered


# ---------------------------------------------------------------------------
# Vendored adaptation 2: ADDED, not modified. Nothing above this line changes.
# ---------------------------------------------------------------------------

def extract_frames(
    token_ids: Sequence[int],
    num_quantizers: int,
    audio_code_id_map: Mapping[int, tuple[int, int]],
    audio_start_id: int | None = None,
    audio_end_id: int | None = None,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> list[list[int]]:
    """The same frame state machine, returning (T, Q) frame-major nested lists.

    Exists so the frame logic is testable without torch, and so the
    (T, Q) -> (Q, T) transpose that `extract_audio_codes_bqt_from_token_ids`
    performs is visible and separately assertable rather than implicit.

    Partial frames are dropped, exactly as upstream: a trailing group of fewer
    than `num_quantizers` tokens is not decodable, so `len(token_ids) // Q` is
    NOT the frame count once any resync has happened.
    """
    if not audio_code_id_map:
        return []
    if start_idx is None or end_idx is None:
        start_idx, end_idx = get_audio_span_indices(
            token_ids, audio_start_id=audio_start_id, audio_end_id=audio_end_id
        )

    frames: list[list[int]] = []
    frame_vals: list[int] = []
    expected_q = 0
    for tok_id in token_ids[start_idx:end_idx]:
        mapped = audio_code_id_map.get(tok_id)
        if mapped is None:
            expected_q = 0
            frame_vals = []
            continue
        code, q_idx = mapped
        if q_idx == expected_q:
            frame_vals.append(code)
            expected_q += 1
            if expected_q == num_quantizers:
                frames.append(frame_vals)
                frame_vals = []
                expected_q = 0
        elif q_idx == 0:
            frame_vals = [code]
            expected_q = 1
        else:
            expected_q = 0
            frame_vals = []
    return frames
