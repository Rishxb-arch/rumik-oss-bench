"""Levenshtein distance, backed by rapidfuzz.

The training repo hand-rolls an O(n*m) full-matrix DP in pure Python
(`deepspeed_runner.py:6411-6439`, and again at `train.py:265`). That is fine for
ten eval samples and far too slow as an RL reward: character-level DP over a few
hundred characters, for thousands of rollouts per step, lands squarely on the
critical path. rapidfuzz is a C++ implementation of the same metric.

A pure-Python fallback is kept so the metric layer stays importable (and
testable) in an environment without rapidfuzz; it is exercised by the tests to
guarantee the two agree.
"""

from __future__ import annotations

from collections.abc import Sequence

try:  # pragma: no cover - trivial import branch
    from rapidfuzz.distance import Levenshtein as _RF

    _HAVE_RAPIDFUZZ = True
except ImportError:  # pragma: no cover
    _HAVE_RAPIDFUZZ = False


def _levenshtein_python(a: Sequence, b: Sequence) -> int:
    """Rolling-row DP. Reference implementation and fallback."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance between two sequences (strings or token lists)."""
    if _HAVE_RAPIDFUZZ:
        # rapidfuzz handles str directly and any sequence of hashables otherwise.
        return int(_RF.distance(a, b))
    return _levenshtein_python(a, b)
