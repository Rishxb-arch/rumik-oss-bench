"""Length prior: fitting, markup handling, and graceful degradation."""

from __future__ import annotations

import json

import pytest

from rumik_oss_eval.length_prior import (
    DEFAULT_CHARS_PER_SECOND,
    LengthPrior,
    fit_from_manifest,
    fit_prior,
    scoreable_length,
)


def test_description_is_not_counted_toward_duration():
    """The description is never voiced, so counting it inflates the estimate."""
    plain = scoreable_length("नमस्ते, आप कैसे हैं?")
    tagged = scoreable_length('<description="happy, indian accent, fast pace"> नमस्ते, आप कैसे हैं?')
    assert plain == tagged


def test_speaker_prefix_and_events_are_not_counted():
    assert scoreable_length("Silver: hello there <laugh>") == scoreable_length("hello there")


def test_empty_text_has_no_expectation():
    assert LengthPrior().expected_seconds("") is None
    assert LengthPrior().expected_seconds('<description="x">') is None


# --- fitting ---------------------------------------------------------------

def _rows(cps: float, n: int, lang: str = "hi", chars: int = 60):
    """n rows whose text/duration ratio is exactly `cps`."""
    return [("x" * chars, lang, chars / cps) for _ in range(n)]


def test_fit_recovers_a_known_rate():
    p = fit_prior(_rows(15.0, 100))
    assert p.cps("hi") == pytest.approx(15.0, abs=0.01)
    assert p.counts["hi"] == 100


def test_median_resists_outliers():
    """Corpora carry clips with trailing silence or mismatched transcripts."""
    rows = _rows(15.0, 100) + [("x" * 60, "hi", 600.0)] * 10   # absurdly slow
    assert fit_prior(rows).cps("hi") == pytest.approx(15.0, abs=0.1)


def test_languages_are_fitted_separately():
    p = fit_prior(_rows(12.0, 60, "hi") + _rows(20.0, 60, "ta"))
    assert p.cps("hi") == pytest.approx(12.0, abs=0.01)
    assert p.cps("ta") == pytest.approx(20.0, abs=0.01)


def test_thin_languages_fall_back_rather_than_fitting_noise():
    # 30 cps over 60 chars = 2.0 s clips, comfortably inside the duration filter,
    # so these rows are counted -- they are just too few to fit on.
    p = fit_prior(_rows(12.0, 100, "hi") + _rows(30.0, 3, "brx"))
    assert "brx" not in p.chars_per_second        # too few samples to trust
    assert p.counts["brx"] == 3
    assert p.cps("brx") == p.default_cps          # degrades, does not disable


def test_unknown_language_uses_the_default():
    p = fit_prior(_rows(12.0, 100, "hi"))
    assert p.cps("zz") == p.default_cps


def test_out_of_range_durations_are_dropped():
    rows = _rows(15.0, 100) + [("x" * 60, "hi", 0.1)] * 50 + [("x" * 60, "hi", 500.0)] * 50
    p = fit_prior(rows)
    assert p.counts["hi"] == 100
    assert p.cps("hi") == pytest.approx(15.0, abs=0.01)


def test_empty_input_yields_the_documented_default():
    p = fit_prior([])
    assert p.cps("hi") == DEFAULT_CHARS_PER_SECOND


# --- prediction and round trip --------------------------------------------

def test_expected_seconds_inverts_the_rate():
    p = LengthPrior(chars_per_second={"hi": 15.0})
    assert p.expected_seconds("x" * 45, "hi") == pytest.approx(3.0)


def test_json_round_trip(tmp_path):
    p = fit_prior(_rows(13.5, 80, "gu"))
    f = tmp_path / "prior.json"
    p.to_json(f)
    q = LengthPrior.from_json(f)
    assert q.chars_per_second == p.chars_per_second
    assert q.default_cps == p.default_cps


def test_fit_from_manifest(tmp_path):
    f = tmp_path / "m.jsonl"
    f.write_text("\n".join([
        *[json.dumps({"sample_id": str(i), "text": "[happy] " + "x" * 60, "duration_s": 4.0})
          for i in range(50)],
        json.dumps({"sample_id": "no-dur", "text": "x" * 60}),      # skipped
        "{bad json}",
    ]), encoding="utf-8")
    p = fit_from_manifest(f)
    # markup stripped -> 60 chars / 4.0 s = 15 cps, the "[happy] " must not count
    assert p.cps("hi") == pytest.approx(15.0, abs=0.01)
    assert p.counts["hi"] == 50
