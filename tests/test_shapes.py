"""Shape contracts and the arithmetic identities the tensor-flow doc claims.

Run against BOTH numpy and torch fixtures: the contracts are duck-typed so they
must work either way, and proving that now is what lets the same assertions run
unchanged against a real collated batch later.
"""

from __future__ import annotations

import numpy as np
import pytest

# The audio-token parser returns torch tensors, so this module needs the
# judges extra:  pip install -e ".[judges]"
pytest.importorskip("torch", reason="needs the [judges] extra")

import torch

from rumik_oss_eval.shapes import (
    CODEBOOK_FRAME_BATCH,
    DEFAULT_CODEBOOK_SIZE,
    DEFAULT_NUM_QUANTIZERS,
    FRAME_SECONDS,
    FRAMES_PER_SECOND,
    ICL_BATCH,
    IGNORE_INDEX,
    MIMI_ENCODE_DOWNSAMPLE,
    MIMI_SAMPLE_RATE,
    AudioVocabLayout,
    assert_batch,
    frames_from_samples,
    samples_from_frames,
    tokens_per_second,
)
from rumik_oss_eval.vendor.audio_token_parser import build_audio_code_id_map


B, S, R, T_r, Q = 3, 64, 16, 5, DEFAULT_NUM_QUANTIZERS


def _icl_batch(lib):
    if lib == "torch":
        z = lambda *s: torch.zeros(s, dtype=torch.long)  # noqa: E731
    else:
        z = lambda *s: np.zeros(s, dtype=np.int64)       # noqa: E731
    return {"input_ids": z(B, S), "attention_mask": z(B, S),
            "labels": z(B, S), "ref_length_hint": z(B)}


def _codebook_batch(lib):
    if lib == "torch":
        z = lambda *s: torch.zeros(s, dtype=torch.long)  # noqa: E731
    else:
        z = lambda *s: np.zeros(s, dtype=np.int64)       # noqa: E731
    return {"input_ids": z(B, S), "attention_mask": z(B, S), "labels": z(B, S),
            "ref_input_ids": z(B, R), "ref_attention_mask": z(B, R),
            "ref_audio_codes": z(B, T_r, Q), "ref_audio_mask": z(B, T_r)}


# --- both variants, both libraries ----------------------------------------

@pytest.mark.parametrize("lib", ["numpy", "torch"])
def test_icl_batch_validates(lib):
    dims = assert_batch(_icl_batch(lib), ICL_BATCH)
    assert dims["B"] == B and dims["S"] == S


@pytest.mark.parametrize("lib", ["numpy", "torch"])
def test_codebook_frame_batch_validates(lib):
    dims = assert_batch(_codebook_batch(lib), CODEBOOK_FRAME_BATCH)
    assert (dims["B"], dims["S"], dims["R"], dims["T_r"], dims["Q"]) == (B, S, R, T_r, Q)


def test_the_two_variants_are_mutually_exclusive():
    """ICL pops the ref_* tensors; codebook_frame has no ref_length_hint.
    Accepting the wrong one means the model silently gets no reference."""
    with pytest.raises(AssertionError, match="must be absent"):
        assert_batch(_codebook_batch("torch"), ICL_BATCH)
    with pytest.raises(AssertionError, match="must be absent"):
        assert_batch(_icl_batch("torch"), CODEBOOK_FRAME_BATCH)


def test_R_and_T_r_are_distinct_dimensions():
    """Upstream reuses one local name (`ref_target_len`) for reference TOKENS
    and reference FRAMES in the same function; they differ by roughly Q."""
    dims = assert_batch(_codebook_batch("torch"), CODEBOOK_FRAME_BATCH)
    assert dims["R"] != dims["T_r"]


# --- the validator actually catches things --------------------------------

def test_inconsistent_batch_dim_is_caught():
    batch = _icl_batch("torch")
    batch["labels"] = torch.zeros((B + 1, S), dtype=torch.long)
    with pytest.raises(AssertionError, match="already"):
        assert_batch(batch, ICL_BATCH)


def test_wrong_rank_is_caught():
    batch = _icl_batch("torch")
    batch["input_ids"] = torch.zeros((B, S, 1), dtype=torch.long)
    with pytest.raises(AssertionError, match="expected 2 dims"):
        assert_batch(batch, ICL_BATCH)


def test_wrong_dtype_is_caught():
    batch = _icl_batch("torch")
    batch["input_ids"] = torch.zeros((B, S), dtype=torch.float32)
    with pytest.raises(AssertionError, match="dtype"):
        assert_batch(batch, ICL_BATCH)


def test_missing_required_key_is_caught():
    batch = _icl_batch("torch")
    del batch["labels"]
    with pytest.raises(AssertionError, match="missing required"):
        assert_batch(batch, ICL_BATCH)


def test_transposed_ref_codes_are_caught_when_dims_disagree():
    """(B,Q,T_r) instead of (B,T_r,Q) is caught because Q is pinned elsewhere."""
    batch = _codebook_batch("torch")
    batch["ref_audio_codes"] = torch.zeros((B, Q, T_r), dtype=torch.long)
    with pytest.raises(AssertionError):
        assert_batch(batch, CODEBOOK_FRAME_BATCH, dims={"Q": Q, "T_r": T_r})


# --- arithmetic identities the doc claims ---------------------------------

def test_mimi_constants():
    assert MIMI_ENCODE_DOWNSAMPLE == 1920
    assert MIMI_SAMPLE_RATE == 24000
    assert FRAMES_PER_SECOND == 12.5
    assert FRAME_SECONDS == 0.08
    assert IGNORE_INDEX == -100


def test_tokens_per_second_is_100_at_q8():
    assert tokens_per_second(8) == 100.0
    assert tokens_per_second(4) == 50.0


@pytest.mark.parametrize("frames,samples", [(1, 1920), (30, 57_600), (300, 576_000)])
def test_samples_from_frames_matches_observed_decode(frames, samples):
    assert samples_from_frames(frames) == samples


def test_frames_from_samples_rounds_up():
    """T = ceil(N / 1920): the encoder pads a partial trailing window."""
    assert frames_from_samples(1920) == 1
    assert frames_from_samples(1921) == 2
    assert frames_from_samples(1) == 1
    assert frames_from_samples(0) == 0


# --- audio vocab layout ---------------------------------------------------

def test_closed_form_matches_the_real_583_vocab_ids():
    """Anchored to real observed ids: <0_0>=261008 and <2047_7>=277391."""
    layout = AudioVocabLayout(base_id=261008, num_quantizers=8, codebook_size=2048,
                              audio_start_id=277393, audio_end_id=277394)
    assert layout.token_id(0, 0) == 261008
    assert layout.token_id(2047, 7) == 277391
    assert layout.code_q(261008) == (0, 0)
    assert layout.code_q(277391) == (2047, 7)
    assert layout.num_audio_tokens == 16384


def test_token_id_and_code_q_are_inverse():
    layout = AudioVocabLayout(base_id=1000)
    for code in (0, 1, 17, 2047):
        for q in range(DEFAULT_NUM_QUANTIZERS):
            assert layout.code_q(layout.token_id(code, q)) == (code, q)


def test_out_of_range_is_rejected():
    layout = AudioVocabLayout(base_id=1000)
    with pytest.raises(ValueError):
        layout.token_id(DEFAULT_CODEBOOK_SIZE, 0)
    with pytest.raises(ValueError):
        layout.token_id(0, DEFAULT_NUM_QUANTIZERS)
    with pytest.raises(ValueError):
        layout.code_q(999)


def test_verify_against_vocab_derived_map_agrees():
    """The startup check: closed form vs the map rebuilt from vocab strings."""
    base, q_n, c_n = 500, 2, 4
    vocab = {f"<{c}_{q}>": base + c * q_n + q for c in range(c_n) for q in range(q_n)}
    layout = AudioVocabLayout(base_id=base, num_quantizers=q_n, codebook_size=c_n)
    assert layout.verify_against(build_audio_code_id_map(vocab)) is True


def test_verify_against_detects_a_non_affine_vocab():
    """expand_tokenizer_with_unit_tokens skips pre-existing tokens, which breaks
    the affine mapping. verify_against must notice rather than corrupt decoding."""
    base, q_n, c_n = 500, 2, 4
    vocab = {f"<{c}_{q}>": base + c * q_n + q for c in range(c_n) for q in range(q_n)}
    vocab["<3_1>"] = 9999                      # one token landed elsewhere
    layout = AudioVocabLayout(base_id=base, num_quantizers=q_n, codebook_size=c_n)
    assert layout.verify_against(build_audio_code_id_map(vocab)) is False


def test_restricted_generation_support_is_17x_smaller_than_full_vocab():
    """The GRPO memory lever: sampling was constrained to this set, so the
    logprob must be computed over it — correct AND ~17x cheaper."""
    layout = AudioVocabLayout(base_id=261008, audio_end_id=277394)
    allowed = layout.allowed_generation_ids
    assert len(allowed) == 2048 * 8 + 1 == 16_385
    assert 277404 / len(allowed) > 16.9
