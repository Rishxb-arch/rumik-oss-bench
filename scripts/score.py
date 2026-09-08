"""Score a directory of audio against a prompt set.

    python scripts/score.py --audio audio/my_system \\
        --promptset prompts/indic22_bench.jsonl \\
        --judge indicconformer --out runs/my_system

Audio files are matched to prompts by stem: `as_000001.wav` scores against the
prompt with id `as_000001`. Anything unmatched is reported, not silently ignored.

Writes `per_sample.tsv` in the same schema as `results/*.per_sample.tsv`, so the
output feeds straight into `scripts/aggregate.py` and is directly comparable
with our published numbers.

Needs the judge. IndicConformer is gated on Hugging Face -- request access, then
`huggingface-cli login`. Install the extra first:  pip install -e ".[judges]"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")


def load_prompts(path: Path) -> dict:
    out = {}
    if path.suffix == ".jsonl":
        for ln in path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                p = json.loads(ln)
                out[p["id"]] = p
    else:
        import yaml
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        d = doc.get("defaults") or {}
        for p in doc["prompts"]:
            p.setdefault("language", d.get("language", ""))
            p["language"] = str(p["language"]).split("-")[0].lower()
            out[p["id"]] = p
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--promptset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--judge", default="indicconformer",
                    choices=["indicconformer", "whisper", "routed"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    import soundfile as sf
    from rumik_oss_eval.asr.base import Clip
    from rumik_oss_eval.metrics.cer_wer import score_pair
    from rumik_oss_eval.normalize.pipeline import normalize_for_scoring

    prompts = load_prompts(a.promptset)
    files = sorted(p for p in a.audio.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if not files:
        sys.exit(f"no audio under {a.audio}")

    matched = [(f, prompts[f.stem]) for f in files if f.stem in prompts]
    missing = [f.name for f in files if f.stem not in prompts]
    print(f"{len(matched)} matched, {len(missing)} unmatched, "
          f"{len(prompts) - len(matched)} prompts with no audio")
    if missing[:5]:
        print(f"  unmatched examples: {missing[:5]}")
    if not matched:
        sys.exit("nothing to score: audio stems must equal prompt ids")

    if a.judge == "whisper":
        from rumik_oss_eval.asr.whisper_hf import WhisperAsr
        asr = WhisperAsr(device=a.device, batch_size=a.batch)
    else:
        from rumik_oss_eval.asr.indicconformer import IndicConformerAsr
        icf = IndicConformerAsr(decoder="rnnt")
        if a.judge == "routed":
            from rumik_oss_eval.asr.routed import ScriptRoutedAsr
            from rumik_oss_eval.asr.whisper_hf import WhisperAsr
            wh = WhisperAsr(device=a.device, batch_size=a.batch)
            icf.fallback = wh
            asr = ScriptRoutedAsr(indic=icf, latin=wh)
        else:
            asr = icf

    clips, meta = [], []
    for f, p in matched:
        audio, sr = sf.read(f, dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        clips.append(Clip(audio=audio, sample_rate=sr, clip_id=p["id"],
                          language=p["language"]))
        meta.append(p)

    print(f"transcribing {len(clips)} clips with {a.judge} ...", flush=True)
    results = asr.transcribe(clips)

    a.out.mkdir(parents=True, exist_ok=True)
    tsv = a.out / "per_sample.tsv"
    cols = ["id", "language", "gate", "asr_status", "cer", "wer", "valid"]
    scored = gated = 0
    with tsv.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for p, r in zip(meta, results):
            status = str(getattr(r, "status", "ok"))
            hyp = getattr(r, "text", "") or ""
            ref_n = normalize_for_scoring(p.get("spoken") or p["text"])
            hyp_n = normalize_for_scoring(hyp)
            s = score_pair(ref_n.folded, hyp_n.folded)
            gate = "" if status == "ok" else f"asr:{status}"
            if gate:
                gated += 1
            else:
                scored += 1
            fh.write("\t".join(str(x) for x in [
                p["id"], p["language"], gate, status,
                f"{s.cer:.6f}", f"{s.wer:.6f}", int(s.valid)]) + "\n")

    print(f"wrote {tsv}   scored {scored}, gated {gated}")
    print(f"next:  python scripts/aggregate.py {tsv} "
          f"--vendors results/vendor_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
