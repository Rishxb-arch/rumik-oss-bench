"""Coverage the upstream contract test does not provide, for things we rely on.

The vendored `test_audio_token_parser.py` pins the frame state machine and the
(code, q) map, but leaves three functions untested that this package depends on:
`get_last_audio_span_indices`, `AllowTokenIdsLogitsProcessor`, and
`normalize_waveform_for_logging`. It also never states the (T,Q) -> (Q,T)
relationship explicitly, which is the single easiest thing to get wrong here.
"""

from __future__ import annotations

import numpy as np
import pytest

# The audio-token parser returns torch tensors, so this module needs the
# judges extra:  pip install -e ".[judges]"
pytest.importorskip("torch", reason="needs the [judges] extra")

import torch

from rumik_oss_eval.vendor.audio_token_parser import (

    AllowTokenIdsLogitsProcessor,
    build_audio_code_id_map,
    extract_audio_codes_bqt_from_token_ids,
    extract_frames,
    get_audio_span_indices,
    get_last_audio_span_indices,
    normalize_waveform_for_logging,
)

Q = 2
AUDIO_START, AUDIO_END = 10, 11
# <code_q> -> id.  Codes 5,8,9,4 used below.
VOCAB = {f"<{c}_{q}>": 100 + c * Q + q for c in range(10) for q in range(Q)}
VOCAB.update({"<audio>": AUDIO_START, "</audio>": AUDIO_END})
CODE_MAP = build_audio_code_id_map(VOCAB)


def tok(code: int, q: int) -> int:
    return VOCAB[f"<{code}_{q}>"]


# --- the ICL trap: first span vs last span --------------------------------

def _two_span_sequence():
    """Mimics an ICL / voice-clone prompt: reference audio, then generated audio.

        <audio> REF </audio> ... <audio> GEN </audio>
    """
    ref = [tok(1, 0), tok(1, 1)]
    gen = [tok(7, 0), tok(7, 1)]
    return [AUDIO_START] + ref + [AUDIO_END] + [99] + [AUDIO_START] + gen + [AUDIO_END], ref, gen


def test_last_span_selects_the_generation_not_the_reference():
    """THE correctness trap. In a cloning prompt the reference audio is itself
    wrapped in <audio>...</audio>, so the first-span variant decodes the
    reference clip — audio that sounds perfectly fine and is completely wrong.
    Every real call site in the training repo uses the last-span variant."""
    ids, ref, gen = _two_span_sequence()

    first_s, first_e = get_audio_span_indices(ids, AUDIO_START, AUDIO_END)
    last_s, last_e = get_last_audio_span_indices(ids, AUDIO_START, AUDIO_END)

    assert ids[first_s:first_e] == ref, "first-span must land on the REFERENCE"
    assert ids[last_s:last_e] == gen, "last-span must land on the GENERATION"
    assert (first_s, first_e) != (last_s, last_e)


def test_last_span_decodes_different_codes_than_first_span():
    """Same trap, expressed as decoded codes rather than indices."""
    ids, _, _ = _two_span_sequence()
    first = extract_frames(ids, Q, CODE_MAP, AUDIO_START, AUDIO_END)
    s, e = get_last_audio_span_indices(ids, AUDIO_START, AUDIO_END)
    last = extract_frames(ids, Q, CODE_MAP, start_idx=s, end_idx=e)
    assert first == [[1, 1]]      # the reference
    assert last == [[7, 7]]       # the generation


def test_last_span_without_start_id_returns_whole_sequence():
    ids = [tok(3, 0), tok(3, 1)]
    assert get_last_audio_span_indices(ids, None, AUDIO_END) == (0, len(ids))


def test_last_span_when_start_marker_absent():
    ids = [tok(3, 0), tok(3, 1)]
    assert get_last_audio_span_indices(ids, AUDIO_START, AUDIO_END) == (0, len(ids))


def test_last_span_runs_to_end_when_no_end_marker():
    ids = [AUDIO_START, tok(3, 0), tok(3, 1)]
    assert get_last_audio_span_indices(ids, AUDIO_START, AUDIO_END) == (1, 3)


# --- the (T,Q) -> (Q,T) transpose -----------------------------------------

def test_frames_are_TQ_and_tensor_is_QT():
    """The layout trap, stated explicitly: frames[t][q] == tensor[0][q][t]."""
    ids = [AUDIO_START, tok(5, 0), tok(8, 1), tok(9, 0), tok(4, 1), AUDIO_END]
    frames = extract_frames(ids, Q, CODE_MAP, AUDIO_START, AUDIO_END)
    tensor = extract_audio_codes_bqt_from_token_ids(ids, Q, CODE_MAP, AUDIO_START, AUDIO_END)

    assert frames == [[5, 8], [9, 4]]          # (T=2, Q=2) frame-major
    assert tuple(tensor.shape) == (1, Q, 2)    # (B=1, Q, T) quantizer-major

    for t, frame in enumerate(frames):
        for q, code in enumerate(frame):
            assert tensor[0][q][t].item() == code, f"transpose broken at t={t} q={q}"


def test_a_transposed_tensor_would_have_different_values():
    """Guards against 'it has the right shape so it must be right' when Q == T."""
    ids = [AUDIO_START, tok(5, 0), tok(8, 1), tok(9, 0), tok(4, 1), AUDIO_END]
    tensor = extract_audio_codes_bqt_from_token_ids(ids, Q, CODE_MAP, AUDIO_START, AUDIO_END)
    assert tensor.tolist() == [[[5, 9], [8, 4]]]          # (B,Q,T) — correct
    assert tensor.tolist() != [[[5, 8], [9, 4]]]          # (B,T,Q) — the bug


def test_partial_trailing_frame_is_dropped():
    """len(tokens)//Q is not the frame count once a frame is incomplete."""
    ids = [AUDIO_START, tok(5, 0), tok(8, 1), tok(9, 0), AUDIO_END]  # 3 tokens, Q=2
    assert extract_frames(ids, Q, CODE_MAP, AUDIO_START, AUDIO_END) == [[5, 8]]


def test_no_complete_frame_returns_empty_and_none():
    ids = [AUDIO_START, tok(5, 0), AUDIO_END]
    assert extract_frames(ids, Q, CODE_MAP, AUDIO_START, AUDIO_END) == []
    assert extract_audio_codes_bqt_from_token_ids(ids, Q, CODE_MAP, AUDIO_START, AUDIO_END) is None


def test_empty_code_map_is_handled():
    assert extract_frames([1, 2, 3], Q, {}) == []


# --- AllowTokenIdsLogitsProcessor -----------------------------------------

def test_logits_processor_preserves_shape_and_allowed_values():
    scores = torch.randn(3, 50)
    allowed = [4, 9, 17]
    out = AllowTokenIdsLogitsProcessor(allowed)(torch.zeros(3, 1, dtype=torch.long), scores)
    assert out.shape == scores.shape and out.dtype == scores.dtype
    for i in allowed:
        assert torch.equal(out[:, i], scores[:, i])


def test_logits_processor_masks_with_finfo_min_not_inf():
    """-inf would make softmax produce NaN if every allowed id were masked
    elsewhere; finfo.min keeps the distribution finite."""
    scores = torch.randn(2, 20)
    out = AllowTokenIdsLogitsProcessor([3])(torch.zeros(2, 1, dtype=torch.long), scores)
    blocked = out[:, 0]
    assert torch.isfinite(blocked).all()
    assert (blocked == torch.finfo(scores.dtype).min).all()
    assert torch.isfinite(torch.softmax(out, dim=-1)).all()


def test_logits_processor_does_not_mutate_input():
    scores = torch.randn(2, 20)
    before = scores.clone()
    AllowTokenIdsLogitsProcessor([1, 2])(torch.zeros(2, 1, dtype=torch.long), scores)
    assert torch.equal(scores, before)


def test_logits_processor_empty_allowlist_is_passthrough():
    scores = torch.randn(2, 10)
    out = AllowTokenIdsLogitsProcessor([])(torch.zeros(2, 1, dtype=torch.long), scores)
    assert torch.equal(out, scores)


def test_logits_processor_ignores_ids_beyond_vocab():
    """An expanded tokenizer against a smaller head must not index-error."""
    scores = torch.randn(1, 10)
    out = AllowTokenIdsLogitsProcessor([2, 999])(torch.zeros(1, 1, dtype=torch.long), scores)
    assert out.shape == scores.shape
    assert torch.equal(out[:, 2], scores[:, 2])


def test_logits_processor_deduplicates_and_drops_negatives():
    p = AllowTokenIdsLogitsProcessor([5, 5, -1, 2])
    assert p.allowed_token_ids.tolist() == [2, 5]


# --- normalize_waveform_for_logging ---------------------------------------

def test_waveform_normalisation_flattens_and_clips():
    wav = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]], dtype=np.float32)  # (1, N)
    out = normalize_waveform_for_logging(wav)
    assert out.ndim == 1
    assert out.dtype == np.float32
    assert out.min() >= -1.0 and out.max() <= 1.0
    assert out.tolist() == [-1.0, -0.5, 0.0, 0.5, 1.0]


def test_waveform_normalisation_takes_first_channel():
    """Mimi decode yields (B, 1, N); callers strip to 1-D before logging."""
    wav = np.zeros((2, 4), dtype=np.float32)
    wav[0] = 0.25
    wav[1] = 0.75
    assert normalize_waveform_for_logging(wav).tolist() == pytest.approx([0.25] * 4)
