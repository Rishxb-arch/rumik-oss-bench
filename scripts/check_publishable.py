"""Fail if anything that must not be published has entered the tree.

Run before every push. The checks are deliberately dumb and fast -- this is a
last line of defence against an accident, not a substitute for reading
DATA_LICENSES.md.

Why each check exists:
  audio        vendor TTS output is not redistributable under most providers'
               terms, and our own is large and regenerable
  credentials  a live HF token was once baked into an image in a sibling repo
  private refs absolute paths and IPs from the machine this was developed on
               leak infrastructure and break for everyone else
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}

AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}
WEIGHTS = {".safetensors", ".bin", ".pt", ".ckpt", ".pth"}

PATTERNS = [
    ("hugging-face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("openai key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("private key block", re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY")),
    ("dev-machine path", re.compile(r"/root/(?:rl|models)\b")),
    ("bare IPv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("private gcs bucket", re.compile(r"gs://(?!<)[a-z0-9][a-z0-9._-]*")),
]

TEXT_SUFFIX = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".cfg",
               ".sh", ".html", ".tsv", ".csv", ".jsonl", ".cff"}


def walk():
    for p in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file():
            yield p


def main() -> int:
    problems: list[str] = []

    for p in walk():
        rel = p.relative_to(ROOT)
        if p.suffix.lower() in AUDIO:
            problems.append(f"AUDIO      {rel}  (not redistributable / regenerable)")
        if p.suffix.lower() in WEIGHTS:
            problems.append(f"WEIGHTS    {rel}  (belongs on a model hub)")

    for p in walk():
        if p.suffix.lower() not in TEXT_SUFFIX:
            continue
        rel = p.relative_to(ROOT)
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for name, rx in PATTERNS:
            for m in rx.finditer(text):
                # This file necessarily contains the patterns it searches for.
                if rel.name == "check_publishable.py":
                    continue
                line = text.count("\n", 0, m.start()) + 1
                problems.append(f"{name:<18} {rel}:{line}  {m.group(0)[:60]!r}")

    if problems:
        print(f"NOT PUBLISHABLE — {len(problems)} issue(s):\n")
        for x in sorted(set(problems)):
            print("  " + x)
        print("\nFix these, or add a deliberate exception, before pushing.")
        return 1

    n = sum(1 for _ in walk())
    print(f"OK — {n} files checked, nothing unpublishable found.")
    print("This checks for accidents only. Licence questions are in DATA_LICENSES.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
