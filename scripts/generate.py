"""Generate audio with rumik-oss, using the path the published numbers came from.

    pip install -e ".[judges]"
    python scripts/generate.py --promptset prompts/indic22_bench.jsonl \
        --out audio/rumik_oss

The model is loaded from the Hugging Face Hub and nowhere else:

    rumik-ai/rumik-oss-1

No weights are vendored here. The Mimi codec ships in a `codec/` subfolder of
that repo, so there is nothing else to fetch.

WHY THIS DELEGATES TO rumik_oss_eval.rollout
--------------------------------------------
An earlier version of this script rolled its own decode loop, and a later one
called the model's `generate_audio()`. Neither reproduced `results/`: on four
Hindi prompts they gave 0.013-0.704 CER where the published run gave 0.027, and
generated audio ran 1.3-2.2x the expected duration.

The published numbers were produced by `rollout.generate.generate_batch` with
`rollout.loader.load_bundle` and `rollout.decode.decode_generation` -- a
different path with per-sequence finish tracking, explicit RoPE positions under
left padding, and a DynamicCache. That module is now part of this package and
this script simply calls it, so the public reproduction path and the path behind
the published table are the same code.

Settings default to the benchmark's own: temperature 0.8, top_k 30,
max_new_tokens 4096, sampling restricted to the audio token ids.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODEL_ID = "rumik-ai/rumik-oss-1"
LANG_NAME = {"as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri",
             "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada", "kok": "Konkani",
             "ks": "Kashmiri", "mai": "Maithili", "ml": "Malayalam",
             "mni": "Manipuri", "mr": "Marathi", "ne": "Nepali", "or": "Odia",
             "pa": "Punjabi", "sa": "Sanskrit", "sat": "Santali", "ta": "Tamil",
             "te": "Telugu", "ur": "Urdu"}


def load_prompts(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    import yaml
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    d = doc.get("defaults") or {}
    for p in doc["prompts"]:
        p.setdefault("speaker", d.get("speaker"))
        p["language"] = str(p.get("language", "")).split("-")[0].lower()
    return doc["prompts"]



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--promptset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=MODEL_ID,
                    help=f"Hugging Face repo id (default {MODEL_ID})")
    ap.add_argument("--lang", default="", help="restrict to one language")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=4,
                    help="prompts per generate_batch call")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--min-new-tokens", type=int, default=8)
    ap.add_argument("--no-budget", action="store_true",
                    help="use the flat --max-new-tokens instead of a "
                         "per-prompt budget derived from expected_seconds")
    ap.add_argument("--greedy", action="store_true",
                    help="argmax instead of sampling; the published numbers used sampling")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    import torch

    from rumik_oss_eval.rollout.decode import decode_generation, write_wav
    from rumik_oss_eval.rollout.generate import GenConfig, generate_batch
    from rumik_oss_eval.rollout.loader import load_bundle

    rows = load_prompts(a.promptset)
    if a.lang:
        rows = [p for p in rows if p.get("language") == a.lang]
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        sys.exit("no prompts selected")

    print(f"loading {a.model} \u2026", flush=True)
    bundle = load_bundle(a.model, device=a.device, codec_device=a.device)

    cfg = GenConfig(max_new_tokens=a.max_new_tokens, temperature=a.temperature,
                    top_k=a.top_k, greedy=a.greedy, seed=a.seed)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = skipped = failed = 0

    # Batched by language. generate_batch runs until its LONGEST sequence
    # finishes, so mixing lengths wastes the difference; the benchmark shards by
    # language for the same reason.
    for i in range(0, len(rows), a.batch):
        chunk = [p for p in rows[i:i + a.batch]
                 if not (out / f"{p['id']}.wav").exists()]
        skipped += len(rows[i:i + a.batch]) - len(chunk)
        if not chunk:
            continue
        # bundle.prompt() wraps the text into the wire format the model
        # was trained on: <text>{speaker}: {text}<audio>. Passing the
        # bare `text` field omits both the speaker prefix and the
        # trailing <audio> -- and without that trailing token the model
        # is never put into audio-emission mode, which is why an earlier
        # version of this script overran by up to 2.5x expected duration.
        rendered = [bundle.prompt(p["text"], speaker=p.get("speaker"))
                    for p in chunk]
        with torch.inference_mode():
            gens = generate_batch(bundle, rendered, cfg)
        for p, g in zip(chunk, gens):
            d = decode_generation(bundle, g, max_new_tokens=cfg.max_new_tokens)
            if not d.ok:
                failed += 1
                print(f"  {p['id']}: no decodable frame", flush=True)
                continue
            write_wav(d.audio, out / (p['id'] + ".wav"), d.sample_rate)
            done += 1
        print(f"  {min(i + a.batch, len(rows))}/{len(rows)}  written {done}  "
              f"skipped {skipped}  failed {failed}", flush=True)

    print(f"done: {done} written, {skipped} already present, {failed} failed -> {out}")
    print(f"next:  python scripts/score.py --audio {out} "
          f"--promptset {a.promptset} --judge indicconformer --out runs/rumik_oss")
    return 0


if __name__ == "__main__":
    sys.exit(main())
