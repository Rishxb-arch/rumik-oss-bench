"""Rebuild the published tables from the published per-sample scores.

    python scripts/aggregate.py results/rumik_oss.per_sample.tsv

This is the check that needs nothing: no model, no GPU, no API key, no audio.
If the numbers it prints match README.txt, our arithmetic is right. If they do
not, one of us has a bug and it is worth knowing which.

    # add the vendors for the full comparison
    python scripts/aggregate.py results/rumik_oss.per_sample.tsv \\
        --vendors results/vendor_analysis.json

Rows with a gate set, or a non-ok ASR status, are excluded from the means and
counted separately -- a structural failure is not a character error rate, and
averaging the two together would understate both.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

NAME = {"as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri",
        "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada", "kok": "Konkani",
        "ks": "Kashmiri", "mai": "Maithili", "ml": "Malayalam", "mni": "Manipuri",
        "mr": "Marathi", "ne": "Nepali", "or": "Odia", "pa": "Punjabi",
        "sa": "Sanskrit", "sat": "Santali", "ta": "Tamil", "te": "Telugu",
        "ur": "Urdu"}
#: Run-to-run spread of one checkpoint. A per-language gap under this is noise.
NOISE = 0.006
#: Columns in vendor_analysis.json that are OUR models, not third-party APIs, and
#: so must be excluded when asking "what is the best API in this language".
#: `silk-oss-v2` is an earlier Rumik model that the published benchmark measured
#: alongside the vendors; it is kept under its original name because renaming a
#: record of what was measured would make the data say something untrue.
OURS = {"silk-oss-v2", "rumik-oss-1", "rumik-oss", "hosted_ira"}


def read(path: Path) -> tuple[list[dict], int]:
    lines = path.read_text(errors="ignore").splitlines()
    head = lines[0].split("\t")
    idx = {k: i for i, k in enumerate(head)}
    for need in ("id", "language", "cer", "wer"):
        if need not in idx:
            sys.exit(f"{path}: missing column {need!r}; found {head}")
    rows, gated = [], 0
    for ln in lines[1:]:
        f = ln.split("\t")
        if len(f) < len(head):
            continue
        if (idx.get("gate") is not None and f[idx["gate"]]) or \
           (idx.get("asr_status") is not None and f[idx["asr_status"]] not in ("", "ok")):
            gated += 1
            continue
        try:
            rows.append({"id": f[idx["id"]], "lang": f[idx["language"]],
                         "cer": float(f[idx["cer"]]), "wer": float(f[idx["wer"]])})
        except ValueError:
            gated += 1
    return rows, gated


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("per_sample", type=Path)
    ap.add_argument("--vendors", type=Path, default=None,
                    help="results/vendor_analysis.json, to compare against the APIs")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    a = ap.parse_args()

    rows, gated = read(a.per_sample)
    if not rows:
        sys.exit("no scoreable rows")
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["lang"], []).append(r)

    out = {"source": str(a.per_sample), "n_scored": len(rows), "n_gated": gated,
           "cer": st.mean(r["cer"] for r in rows),
           "wer": st.mean(r["wer"] for r in rows),
           "median_cer": st.median(r["cer"] for r in rows),
           "languages": {}}
    for lg, rs in sorted(by.items()):
        out["languages"][lg] = {"n": len(rs),
                                "cer": st.mean(r["cer"] for r in rs),
                                "wer": st.mean(r["wer"] for r in rs)}

    van = json.loads(a.vendors.read_text()) if a.vendors else None
    if van:
        for lg, d in van.get("languages", {}).items():
            if lg in out["languages"]:
                v = {k: (x or {}).get("cer") for k, x in (d.get("vendors") or {}).items()}
                v = {k: c for k, c in v.items() if c is not None and k not in OURS}
                if v:
                    best = min(v, key=v.get)
                    out["languages"][lg]["best_api"] = best
                    out["languages"][lg]["best_api_cer"] = v[best]

    if a.json:
        print(json.dumps(out, indent=1))
        return 0

    print(f"{a.per_sample}")
    print(f"  scored {out['n_scored']}   gated/failed {out['n_gated']}")
    print(f"  mean CER {out['cer']:.4f}   mean WER {out['wer']:.4f}"
          f"   median CER {out['median_cer']:.4f}\n")

    hdr = f"  {'lang':<11}{'n':>5}{'CER':>9}{'WER':>9}"
    if van:
        hdr += f"{'best API':>9}  {'which':<24}{'verdict':<12}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for lg, d in out["languages"].items():
        line = f"  {NAME.get(lg, lg):<11}{d['n']:>5}{d['cer']:>9.4f}{d['wer']:>9.4f}"
        if van and "best_api_cer" in d:
            delta = d["cer"] - d["best_api_cer"]
            verdict = ("ahead" if delta < -NOISE else
                       "behind" if delta > NOISE else "within noise")
            line += f"{d['best_api_cer']:>9.4f}  {d['best_api']:<24}{verdict:<12}"
        elif van:
            line += f"{'-':>9}  {'no API supports it':<24}{'':<12}"
        print(line)

    if van:
        cmp = [d for d in out["languages"].values() if "best_api_cer" in d]
        if cmp:
            print(f"\n  over the {len(cmp)} languages an API competes in:")
            print(f"    ours     {st.mean(d['cer'] for d in cmp):.4f}")
            print(f"    best API {st.mean(d['best_api_cer'] for d in cmp):.4f}"
                  f"   (per-language best, not one system)")
        print(f"\n  a gap under {NOISE} CER is run-to-run noise, not a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
