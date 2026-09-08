"""Teacher forced CTC negative log likelihood from IndicConformer.

WHY THIS EXISTS
---------------
GCER is a discrete edit distance. Two candidates that both transcribe perfectly
score identically, so a group of siblings can tie at zero and produce no
gradient even though one of them is audibly better. CTC NLL is continuous: it is
the log likelihood of the reference under the acoustic model, so it separates
candidates that an edit distance calls equal.

WHAT MAKES IT POSSIBLE HERE
---------------------------
IndicConformer ships as ONNX, so there is no autograd through it. That does not
matter: a reward needs the NLL *value*, never a gradient through the judge. The
graph is decomposed into separate sessions and one of them, `ctc_decoder`,
returns per frame log probs directly.

    preprocessor (TorchScript, traced on CUDA)  wav -> mel (B, 80, T)
    encoder      (ONNX)                          mel -> (B, 1024, T')
    ctc_decoder  (ONNX)                          -> logprobs (B, T', 5633)
    [:, :, language_masks[lang]]                 -> (B, T', 257), blank last

The 257 wide masked output lines up one to one with `vocab[lang]`, so a vocab
index IS a target id. No offset arithmetic.

THE HONEST CAVEAT
-----------------
The repo ships no SentencePiece model, only `vocab.json`, so the exact
segmentation used in training cannot be recovered. This does greedy longest
match over the vocabulary strings, which is close to but not necessarily
identical to it. CTC marginalises over alignments but NOT over tokenisations, so
the absolute NLL is not the model's true likelihood of the reference.

That is acceptable for the use here and only for that reason: GRPO compares
siblings against the SAME reference, so every candidate in a group is scored
against an identical target sequence. A consistent but suboptimal tokenisation
still ranks correctly within a group. Do not compare these values across
different references, and do not report them as likelihoods.

Validate with `scripts/validate_nll.py`: NLL must correlate with GCER across a
batch. If it does not, the tokenisation is wrong and the signal is noise.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODEL = "ai4bharat/indic-conformer-600m-multilingual"
WORD_MARK = "▁"          # the SentencePiece word start marker


@dataclass(slots=True)
class NllResult:
    nll: float                # per target token, in nats
    nll_sum: float
    n_tokens: int
    n_frames: int
    ok: bool
    reason: str = "ok"


class CtcNll:
    """Scores a reference against generated audio with the CTC head.

    Lazily loaded, because it is the same 2.4 GB model the transcription judge
    already holds and most callers will want to share one instance.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, model=None):
        self.model_id = model_id
        self._m = model
        self._tok_cache: dict[str, list[tuple[str, int]]] = {}

    # -- model ------------------------------------------------------------
    def _load(self):
        if self._m is None:
            from transformers import AutoModel
            self._m = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        return self._m

    def supports(self, language: str) -> bool:
        m = self._load()
        return language in getattr(m, "language_masks", {})

    # -- tokenisation -----------------------------------------------------
    def _sorted_vocab(self, lang: str) -> list[tuple[str, int]]:
        """Vocab entries with their ids, longest first, for greedy matching."""
        if lang not in self._tok_cache:
            v = self._load().vocab[lang]
            pairs = [(t, i) for i, t in enumerate(v) if t and not t.startswith("<")]
            pairs.sort(key=lambda p: -len(p[0]))
            self._tok_cache[lang] = pairs
        return self._tok_cache[lang]

    def tokenize(self, text: str, lang: str) -> list[int]:
        """Greedy longest match into vocab ids. Unmatched characters are dropped.

        Spaces become the word start marker, matching SentencePiece convention,
        because the vocabulary is full of entries that begin with it.
        """
        pairs = self._sorted_vocab(lang)
        s = WORD_MARK + WORD_MARK.join(text.split())
        ids, i = [], 0
        while i < len(s):
            for tok, tid in pairs:
                if s.startswith(tok, i):
                    ids.append(tid)
                    i += len(tok)
                    break
            else:
                i += 1        # no entry covers this character; skip it
        return ids

    # -- scoring ----------------------------------------------------------
    def score(self, audio, sample_rate: int, reference: str, language: str) -> NllResult:
        import numpy as np
        import torch
        import torch.nn.functional as F

        m = self._load()
        if language not in getattr(m, "language_masks", {}):
            return NllResult(1e9, 1e9, 0, 0, False, f"unsupported language {language}")

        targets = self.tokenize(reference, language)
        if not targets:
            return NllResult(1e9, 1e9, 0, 0, False, "empty target after tokenisation")

        wav = torch.as_tensor(np.asarray(audio, dtype=np.float32)).reshape(1, -1)
        if sample_rate != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sample_rate, 16000)

        # The preprocessor is a TorchScript module traced on CUDA, so it carries
        # baked in device tensors and rejects CPU input outright.
        dev = getattr(m, "d", torch.device("cpu"))
        feats, flen = m.models["preprocessor"](
            input_signal=wav.to(dev),
            length=torch.tensor([wav.shape[-1]], device=dev))

        enc, _enc_len = m.models["encoder"].run(
            None, {"audio_signal": feats.detach().cpu().numpy().astype(np.float32),
                   "length": flen.detach().cpu().numpy().astype(np.int64)})
        raw = m.models["ctc_decoder"].run(["logprobs"], {"encoder_output": enc})[0]

        # Mask to the language block THEN log_softmax, exactly as the vendor's
        # own decode path does. Softmaxing over all 5633 first would normalise
        # across 22 languages and give a different distribution.
        lp = torch.from_numpy(np.asarray(raw)[:, :, m.language_masks[language]]).float()
        lp = lp.log_softmax(dim=-1)

        n_frames, n_vocab = lp.shape[1], lp.shape[2]
        blank = n_vocab - 1
        if len(targets) > n_frames:
            # CTC cannot align more labels than frames; the loss would be inf.
            return NllResult(1e9, 1e9, len(targets), n_frames, False,
                             f"target {len(targets)} exceeds {n_frames} frames")

        tgt = torch.tensor([targets], dtype=torch.long)
        total = F.ctc_loss(
            lp.transpose(0, 1),                       # (T, B, V)
            tgt,
            torch.tensor([n_frames], dtype=torch.long),
            torch.tensor([len(targets)], dtype=torch.long),
            blank=blank, reduction="sum", zero_infinity=True)
        total = float(total)
        if total != total or total >= 1e9:
            return NllResult(1e9, 1e9, len(targets), n_frames, False, "non finite loss")

        # Per token, not summed. Seed-TTS used the sum and reported a short
        # utterance bias for exactly that reason: a long reference accumulates
        # more loss regardless of how well it was spoken.
        return NllResult(total / len(targets), total, len(targets), n_frames, True)
