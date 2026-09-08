"""Record what a run actually used, so a result can never be misread later.

A promptset file is a moving target: it gets extended, reweighted, or a typo
gets fixed. A run record must not be. Without this, a reward number six weeks
old is uninterpretable — you cannot tell whether it came from the 34 prompts the
file holds today or the 26 it held then, and nothing in the summary would show
the difference.

The field that matters most is `rendered`: the exact string handed to the model,
including the `<text>`/`<audio>` wrapper `bundle.prompt()` adds. Everything else
can be reconstructed; that one cannot, and it is the one that decides what the
model actually saw.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def prompt_record(prompt, bundle=None) -> dict:
    """One prompt, as it was actually used."""
    rec = {
        "id": prompt.id,
        "language": prompt.language,
        "speaker": prompt.speaker or "",
        "script": prompt.script_key,
        "text": prompt.text,
        "reference": prompt.reference,
        "expected_seconds": prompt.expected_seconds,
        "tags": list(prompt.tags),
    }
    if bundle is not None:
        # The wrapper is applied at rollout time and is invisible in the
        # promptset, so it is recorded here rather than inferred.
        rec["rendered"] = bundle.prompt(prompt.text, speaker=prompt.speaker)
    return rec


def write_prompts_used(
    out_dir: str | Path,
    prompts: Sequence,
    *,
    bundle=None,
    promptset_path: str | Path | None = None,
    promptset=None,
    extra: dict | None = None,
) -> Path:
    """Write `prompts_used.jsonl` + `promptset.meta.json` into `out_dir`.

    Returns the path to the jsonl. Safe to call before generation — nothing here
    depends on results, so a crashed run still leaves a record of its inputs.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = [prompt_record(p, bundle) for p in prompts]
    jsonl = out / "prompts_used.jsonl"
    jsonl.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")

    meta = {
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_prompts": len(rows),
        "languages": dict(sorted(Counter(r["language"] for r in rows).items())),
        "scripts": dict(sorted(Counter(r["script"] for r in rows).items())),
        "tags": dict(sorted(Counter(t for r in rows for t in r["tags"]).items())),
        "with_expected_seconds": sum(1 for r in rows if r["expected_seconds"]),
    }
    if promptset_path is not None:
        p = Path(promptset_path)
        meta["promptset_path"] = str(p)
        meta["promptset_sha256"] = _sha256(p)
    if promptset is not None:
        meta["promptset_name"] = getattr(promptset, "name", None)
        meta["promptset_version"] = getattr(promptset, "version", None)
    if extra:
        meta.update(extra)
    (out / "promptset.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonl


def write_steps(out_dir: str | Path, reports: Iterable) -> Path:
    """Per-step prompt selection, so a reward traces back to its prompt.

    GRPO takes a slice of the promptset per step; without this the mapping from
    a reward in the report to the text that produced it is lost.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "steps.jsonl"
    lines = []
    for r in reports:
        lines.append(json.dumps({
            "step": r.step,
            "prompt_ids": list(getattr(r, "prompt_ids", []) or []),
            "group_size": getattr(r, "group_size", None),
            "n_groups": getattr(r, "n_groups", None),
            "dead_groups": getattr(r, "dead_groups", None),
            "reward_mean": getattr(r, "reward_mean", None),
            "within_group_std": getattr(r, "within_group_std", None),
        }, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path
