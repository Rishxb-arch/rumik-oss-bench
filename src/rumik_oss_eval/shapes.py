"""Executable shape contracts for the TinyAya tensor flow.

Documentation drifts from code silently; assertions do not. Every shape claimed
in docs/tensor_flow.md is encoded here and exercised by tests/test_shapes.py, so
a wrong number fails a test rather than misleading someone months later.

The checks are **duck-typed** — they read `.shape` and `.dtype` only. That means
the same contract validates a numpy fixture today and a real torch batch from
`TTSBatchCollator` tomorrow, with no changes and no import of either library.

Symbolic dimensions, used consistently throughout:

    B    batch                    S    main sequence length
    G    GRPO group size          P    prompt length
    L    generated length         R    reference length, in TOKENS
    T_r  reference length, in FRAMES (R and T_r differ by roughly Q)
    T    audio frames             Q    quantizers (8)
    C    codebook size (2048)     V    vocab (277404)
    H    hidden size (2048)       K    reference prefix tokens
    N    audio samples

`R` vs `T_r` is a genuine hazard: the upstream collator reuses one local name,
`ref_target_len`, for both quantities in the same function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

# --- codec / vocab constants ----------------------------------------------

#: Named constant in the training repo (scripts/pretokenize_emilia.py:24).
MIMI_ENCODE_DOWNSAMPLE = 1920
MIMI_SAMPLE_RATE = 24000
FRAMES_PER_SECOND = MIMI_SAMPLE_RATE / MIMI_ENCODE_DOWNSAMPLE   # 12.5
FRAME_SECONDS = MIMI_ENCODE_DOWNSAMPLE / MIMI_SAMPLE_RATE       # 0.08

DEFAULT_NUM_QUANTIZERS = 8
DEFAULT_CODEBOOK_SIZE = 2048

#: Label positions excluded from the loss.
IGNORE_INDEX = -100


def tokens_per_second(num_quantizers: int = DEFAULT_NUM_QUANTIZERS) -> float:
    """LM tokens per second of audio: 100 at Q=8."""
    return num_quantizers * FRAMES_PER_SECOND


def samples_from_frames(frames: int) -> int:
    """N = T * 1920."""
    return int(frames) * MIMI_ENCODE_DOWNSAMPLE


def frames_from_samples(samples: int) -> int:
    """T = ceil(N / 1920) — the encoder pads a partial trailing window."""
    return -(-int(samples) // MIMI_ENCODE_DOWNSAMPLE)


# --- audio vocab layout ----------------------------------------------------

@dataclass(slots=True, frozen=True)
class AudioVocabLayout:
    """Closed-form mapping between token ids and (code, quantizer).

    The vocab is built **code-major** — `for code: for q:` at
    torchtitan/train.py:96-98, and identically in pretokenize_emilia.py:244 and
    export_tinyaya_native_hf.py:244 (they must agree or checkpoints break):

        token_id = base_id + code * Q + q

    Do NOT trust this blindly. `expand_tokenizer_with_unit_tokens` skips tokens
    that already exist (train.py:100-102), so if any audio token pre-existed the
    mapping stops being affine. Call `verify_against()` with the vocab-derived
    map at startup and fall back to the dict if they disagree.
    """

    base_id: int
    num_quantizers: int = DEFAULT_NUM_QUANTIZERS
    codebook_size: int = DEFAULT_CODEBOOK_SIZE
    audio_start_id: int | None = None
    audio_end_id: int | None = None

    @property
    def num_audio_tokens(self) -> int:
        return self.codebook_size * self.num_quantizers

    def token_id(self, code: int, q: int) -> int:
        if not 0 <= code < self.codebook_size:
            raise ValueError(f"code {code} out of range")
        if not 0 <= q < self.num_quantizers:
            raise ValueError(f"quantizer {q} out of range")
        return self.base_id + code * self.num_quantizers + q

    def code_q(self, token_id: int) -> tuple[int, int]:
        offset = int(token_id) - self.base_id
        if not 0 <= offset < self.num_audio_tokens:
            raise ValueError(f"token id {token_id} is not an audio token")
        return divmod(offset, self.num_quantizers)

    def verify_against(self, audio_code_id_map: dict[int, tuple[int, int]]) -> bool:
        """True iff the closed form reproduces the vocab-derived map exactly."""
        for token_id, (code, q) in audio_code_id_map.items():
            try:
                if self.code_q(token_id) != (code, q):
                    return False
            except ValueError:
                return False
        return True

    @property
    def allowed_generation_ids(self) -> list[int]:
        """The support generation is restricted to: every audio token + EOS.

        Size is `C*Q + 1` = 16,385 at Q=8, C=2048 — about 17x smaller than the
        full 277,404 vocab. Since sampling was constrained to exactly this set,
        a GRPO logprob must be computed over it too: that is both correct and
        the single largest memory saving available.
        """
        ids = list(range(self.base_id, self.base_id + self.num_audio_tokens))
        if self.audio_end_id is not None:
            ids.append(int(self.audio_end_id))
        return ids


# --- shape specs -----------------------------------------------------------

class HasShape(Protocol):  # pragma: no cover - typing only
    shape: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class TensorSpec:
    name: str
    dims: tuple[str, ...]
    dtype: str = "int64"
    pad_value: Any = None
    required: bool = True
    note: str = ""


@dataclass(slots=True, frozen=True)
class BatchSpec:
    """A batch layout, plus the keys that must be ABSENT.

    `forbidden` matters because the two reference-conditioning modes produce
    mutually exclusive batches, and silently accepting the wrong one means the
    model receives no reference at all while still training happily.
    """

    name: str
    tensors: tuple[TensorSpec, ...]
    forbidden: tuple[str, ...] = ()
    note: str = ""

    def spec_for(self, key: str) -> TensorSpec | None:
        return next((t for t in self.tensors if t.name == key), None)


def _dtype_name(tensor: Any) -> str:
    dt = getattr(tensor, "dtype", None)
    if dt is None:
        raise AssertionError("object has no .dtype")
    # torch prints "torch.int64"; numpy prints "int64".
    return str(dt).replace("torch.", "")


def assert_batch(
    batch: dict[str, Any],
    spec: BatchSpec,
    *,
    dims: dict[str, int] | None = None,
) -> dict[str, int]:
    """Validate `batch` against `spec`, resolving symbolic dims consistently.

    Returns the resolved dimension map, so a caller can assert e.g. that `B`
    matched what it expected. Raises AssertionError with a specific message on
    the first violation — a shape bug should name the tensor and the axis.
    """
    resolved: dict[str, int] = dict(dims or {})

    for key in spec.forbidden:
        assert key not in batch, (
            f"{spec.name}: key {key!r} must be absent in this mode but was present"
        )

    for tspec in spec.tensors:
        if tspec.name not in batch:
            assert not tspec.required, f"{spec.name}: missing required key {tspec.name!r}"
            continue
        tensor = batch[tspec.name]
        shape = tuple(getattr(tensor, "shape"))
        assert len(shape) == len(tspec.dims), (
            f"{spec.name}.{tspec.name}: expected {len(tspec.dims)} dims "
            f"{tspec.dims}, got shape {shape}"
        )
        for axis, (dim_name, size) in enumerate(zip(tspec.dims, shape)):
            if dim_name in resolved:
                assert resolved[dim_name] == size, (
                    f"{spec.name}.{tspec.name}: axis {axis} is {dim_name}="
                    f"{size}, but {dim_name} was already {resolved[dim_name]}"
                )
            else:
                resolved[dim_name] = size
        actual = _dtype_name(tensor)
        assert actual == tspec.dtype, (
            f"{spec.name}.{tspec.name}: expected dtype {tspec.dtype}, got {actual}"
        )
    return resolved


# --- the two collated-batch variants --------------------------------------

_MAIN = (
    TensorSpec("input_ids", ("B", "S"), "int64", "pad_token_id"),
    TensorSpec("attention_mask", ("B", "S"), "int64", 0),
    TensorSpec("labels", ("B", "S"), "int64", IGNORE_INDEX,
               note="-100 outside the supervised span; note mask_text_loss "
                    "defaults to False, so text IS supervised by default"),
)

ICL_BATCH = BatchSpec(
    name="icl_batch",
    tensors=_MAIN + (
        TensorSpec("ref_length_hint", ("B",), "int64", 0,
                   note="count of real reference tokens; 0 when dropped"),
    ),
    forbidden=("ref_input_ids", "ref_attention_mask", "ref_audio_codes", "ref_audio_mask"),
    note="reference_conditioning_mode='icl'. _apply_icl_merge folds the "
         "reference into input_ids and POPS the four ref_* tensors.",
)

CODEBOOK_FRAME_BATCH = BatchSpec(
    name="codebook_frame_batch",
    tensors=_MAIN + (
        TensorSpec("ref_input_ids", ("B", "R"), "int64", "pad_token_id"),
        TensorSpec("ref_attention_mask", ("B", "R"), "int64", 0),
        TensorSpec("ref_audio_codes", ("B", "T_r", "Q"), "int64", "codebook_size",
                   note="FRAME-MAJOR (B,T_r,Q). The parser returns (B,Q,T) — "
                        "transposing one into the other silently yields garbage."),
        TensorSpec("ref_audio_mask", ("B", "T_r"), "int64", 0),
    ),
    forbidden=("ref_length_hint",),
    note="reference_conditioning_mode='codebook_frame' or 'legacy_token'.",
)

#: Output of extract_audio_codes_bqt_from_token_ids, and the input MimiModel
#: .decode expects. Quantizer-major, batch always 1.
DECODE_CODES = TensorSpec(
    "codes_bqt", ("B", "Q", "T"), "int64", None,
    note="(B=1, Q, T). Non-contiguous view (a transpose); CPU; move to device "
         "before decode. Callers own device placement and no_grad.",
)

#: MimiModel.decode(...).audio_values
DECODED_AUDIO = TensorSpec(
    "audio_values", ("B", "channels", "N"), "float32", None,
    note="(B, 1, N) with N = T * 1920. Callers strip [0, 0] to get 1-D.",
)
