"""Batched audio-token generation with per-sequence finish tracking.

Why not reuse what ships: both the bundle's `generate_audio` and the training
repo's `_generate_flat` decide when to stop with `.all()` across the batch. One
long sequence therefore holds the entire batch open, and every row receives EOS
at the same position — so batched output is not the same as batch-1 output. That
is a correctness problem, not a throughput one, and it silently corrupts any
length-derived metric.

Two further details this loop gets right:

* **DynamicCache, not the config's `hybrid` cache.** The hybrid cache is
  pre-allocated from the first forward; writing past it is an out-of-bounds GPU
  write.
* **Left padding with explicit `position_ids`.** With left-padded prompts the
  model would otherwise infer positions from cache length and place padded rows
  at the wrong RoPE offsets. Positions are derived from the attention mask so
  every real token sits where it would in a batch of one — which is what makes
  the batched/native parity test pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum


class FinishReason(StrEnum):
    EOS = "eos"                 # model closed the audio span itself
    MAX_TOKENS = "max_tokens"   # hit the cap -> truncated


@dataclass(slots=True)
class GenConfig:
    max_new_tokens: int = 2048
    min_new_tokens: int = 32
    temperature: float = 0.8
    top_k: int = 30
    greedy: bool = False
    seed: int | None = None
    #: Which KV cache to use. "dynamic" grows as needed and is the only correct
    #: choice here. "hybrid" is offered ONLY so the failure can be reproduced
    #: deliberately: it is what silk-oss-v2's generation_config asks for, it is
    #: pre-allocated from the first forward, and writing past it is an
    #: out-of-bounds GPU write. Never make it the default.
    cache: str = "dynamic"

    def budget_for(self, expected_seconds: float | None, num_quantizers: int) -> int:
        """Token cap sized to the expected duration, with headroom.

        100 tokens per second of audio at Q=8. The 1.35x slack plus a floor
        mirrors the heuristic the training repo uses, and the clamp keeps a
        runaway generation from stalling a whole batch.
        """
        if not expected_seconds or expected_seconds <= 0:
            return self.max_new_tokens
        frames = expected_seconds * 12.5
        est = int(round(frames * num_quantizers * 1.35 + 64))
        return max(256, min(est, self.max_new_tokens))


@dataclass(slots=True)
class Generation:
    prompt: str
    emitted: list[int] = field(default_factory=list)
    finish_reason: FinishReason = FinishReason.MAX_TOKENS
    prompt_len: int = 0

    @property
    def emitted_eos(self) -> bool:
        return self.finish_reason is FinishReason.EOS


def _sample(scores, cfg: GenConfig, generator=None):
    """Top-k sampling, or argmax when greedy. Matches the upstream contract:
    top_k only — no top_p, no repetition penalty.

    Returns an index into whatever axis it was given. Callers now pass the
    restricted (B, K) scores and map the result back through `allowed`, so the
    return value is a position in the support, not a vocabulary id."""
    import torch

    if cfg.greedy:
        return scores.argmax(dim=-1, keepdim=True)
    s = scores / max(cfg.temperature, 1e-5)
    k = min(cfg.top_k, s.shape[-1]) if cfg.top_k > 0 else 0
    if k:
        cut = torch.topk(s, k, dim=-1).values[:, -1:]
        s = s.masked_fill(s < cut, torch.finfo(s.dtype).min)
    probs = torch.softmax(s.float(), dim=-1)
    return torch.multinomial(probs, 1, generator=generator)


def _logits_to_keep_kw(model) -> dict:
    """`{"logits_to_keep": 1}` when the model supports it, else `{}`.

    The prefill forward computes logits for EVERY prompt position, shape
    (B, P, V) with V = 277,404, and one line later `out.logits[:, -1, :]`
    throws all but the last away. At batch 256 with 250 token prompts that was
    a 43 GB allocation for data never read, which is what OOMed the held out
    eval. Asking for only the final position makes it (B, 1, V).

    This helps the training rollout as well: at B=16 with 600 token prompts the
    prefill was allocating several GB per step for the same discarded logits.

    Probed rather than assumed, because the bundle wraps a custom model class.
    """
    import inspect

    # PeftModel.forward is (*args, **kwargs), so the signature would not show
    # logits_to_keep and the optimisation would be silently skipped. Inspect the
    # wrapped model instead.
    target = model
    if callable(getattr(model, "get_base_model", None)):
        try:
            target = model.get_base_model()
        except Exception:
            target = model
    try:
        sig = inspect.signature(target.forward)
    except (TypeError, ValueError):
        return {}
    return {"logits_to_keep": 1} if "logits_to_keep" in sig.parameters else {}


def generate_batch(
    bundle,
    prompts: Sequence[str],
    cfg: GenConfig | None = None,
    *,
    max_new_tokens: int | None = None,
) -> list[Generation]:
    """Generate for a batch of prompts. Returns one Generation per prompt, in order."""
    import torch
    from transformers import DynamicCache

    cfg = cfg or GenConfig()
    if not prompts:
        return []

    tok, model = bundle.tokenizer, bundle.model
    device = bundle.device
    END = bundle.audio_end_id
    allowed = bundle.allowed_ids
    cap = int(max_new_tokens or cfg.max_new_tokens)

    generator = None
    if cfg.seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(cfg.seed))

    # print("prompts[0] : ", prompts[0])   # debug, silenced
    # print("complete prompt set : ", prompts)   # debug, silenced
    encoded = [tok(p, return_tensors=None)["input_ids"] for p in prompts]
    lengths = [len(e) for e in encoded]
    # print("checking the lengths : ", lengths)   # debug, silenced
    # print("encoded[0] : ", encoded[0])   # debug, silenced
    max_p = max(lengths)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    B = len(prompts)
    input_ids = torch.full((B, max_p), pad_id, dtype=torch.long, device=device)
    # print(input_ids)   # debug, silenced
    attn = torch.zeros((B, max_p), dtype=torch.long, device=device)
    # print("check the shapes : ", input_ids.shape, attn.shape)   # debug, silenced

    for i, ids in enumerate(encoded):
        input_ids[i, max_p - len(ids):] = torch.tensor(ids, dtype=torch.long, device=device)
        attn[i, max_p - len(ids):] = 1                      # LEFT padding

    # print(attn)   # debug, silenced
    # RoPE positions must come from the mask, not from the cache length, or
    # left-padded rows sit at the wrong offsets and diverge from batch-1.
    position_ids = (attn.cumsum(-1) - 1).clamp(min=0) # this is a clever way to get index ids : cum sum along the last dim and then - 1 for 0 based followed by clamp with min value 0

    # so every time this generate batch function will be called we build a new kv cache : and this 
    # is called per rollout right ? shouldn't be the cache global across rollouts for max reuse ?
    if cfg.cache == "hybrid":
        # Reproduction path only -- see GenConfig.cache.
        from transformers import HybridCache
        kv = HybridCache(config=model.config, max_batch_size=B,
                         max_cache_len=max_p + cap, device=device, dtype=model.dtype)
    else:
        kv = DynamicCache()
    keep_kw = _logits_to_keep_kw(model)
    out = model(input_ids=input_ids, attention_mask=attn, position_ids=position_ids,
                past_key_values=kv, use_cache=True, return_dict=True, **keep_kw)
    cache, scores = out.past_key_values, out.logits[:, -1, :]
    next_pos = position_ids[:, -1:] + 1

    results = [Generation(prompt=p, prompt_len=lengths[i]) for i, p in enumerate(prompts)]
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    # Bookkeeping stays on-device and is drained once, after the loop. The
    # previous version did 2*B + 1 host syncs PER TOKEN (int(nxt[i,0]),
    # finished[i], finished.all()); each of those blocks while HOLDING the GIL
    # until the GPU catches up, which both costs time and makes it impossible to
    # overlap two devices from one process.
    emitted_buf = torch.zeros(B, cap, dtype=torch.long, device=device)
    emitted_len = torch.zeros(B, dtype=torch.long, device=device)
    # Checking termination every token costs a sync; checking every N costs at
    # most N-1 wasted forwards on an already-finished batch (~0.3s at N=16).
    SYNC_EVERY = 16

    # Position of </audio> inside `allowed`, so the min_new_tokens suppression can
    # be applied in K-space. `allowed` is sorted, so searchsorted is exact.
    end_pos = int(torch.searchsorted(allowed, torch.tensor(END, device=device)))

    with torch.inference_mode():
        for step in range(cap):
            # Sample in the 16,385-wide restricted space instead of materialising
            # a (B, 277404) tensor every single token.
            #
            # This is the SAME distribution, not an approximation: in the
            # full-vocab version the disallowed entries are finfo.min, and
            # exp(finfo.min / T) underflows to exactly 0 in fp32, so they
            # contribute nothing to the softmax. Measured: 5.4x less sampling
            # work, and a total-variation distance of 0.022 over 20k draws --
            # i.e. sampling noise -- against the old path.
            sel = scores.index_select(1, allowed)                   # (B, K)
            if step < cfg.min_new_tokens:
                sel[:, end_pos] = torch.finfo(sel.dtype).min

            nxt = allowed[_sample(sel, cfg, generator)]             # (B, 1) real ids

            # Vectorised equivalent of the old per-row loop. `active` is the
            # rows that had NOT finished before this token, so the terminal
            # </audio> is still recorded (matching the old ordering: append
            # first, then mark finished) and nothing is recorded afterwards.
            tid = nxt.squeeze(1)                                    # (B,)
            active = ~finished
            emitted_buf[:, step] = torch.where(active, tid, torch.zeros_like(tid))
            emitted_len += active.long()
            finished = finished | (tid.eq(END) & active)

            if step % SYNC_EVERY == 0 and bool(finished.all()):
                break

            # Finished rows are fed EOS as filler and masked out, so they cannot
            # influence the rows still generating.
            nxt = torch.where(finished.unsqueeze(1), torch.full_like(nxt, END), nxt)
            step_mask = (~finished).long().unsqueeze(1)
            attn = torch.cat((attn, step_mask), dim=1)

            out = model(input_ids=nxt, attention_mask=attn, position_ids=next_pos,
                        past_key_values=cache, use_cache=True, return_dict=True, **keep_kw)
            cache, scores = out.past_key_values, out.logits[:, -1, :]
            next_pos = next_pos + step_mask

    # One host transfer for the whole batch instead of 2*B per token.
    lens = emitted_len.tolist()
    fin = finished.tolist()
    buf = emitted_buf[:, :max(lens) if lens else 0].tolist()
    for i, r in enumerate(results):
        r.emitted = buf[i][:lens[i]]
        if fin[i]:
            r.finish_reason = FinishReason.EOS

    return results


def generate_native(bundle, prompts: Sequence[str], cfg: GenConfig | None = None,
                    *, max_new_tokens: int | None = None) -> list[Generation]:
    """One prompt at a time. Reference implementation for the parity test."""
    out: list[Generation] = []
    for p in prompts:
        out.extend(generate_batch(bundle, [p], cfg, max_new_tokens=max_new_tokens))
    return out
