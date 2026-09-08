"""Calibration harness invariants - chiefly the id -> filename mapping."""

from __future__ import annotations

import json

import pytest

from rumik_oss_eval.calibrate import CalibrationSample, load_manifest, summarize


@pytest.mark.parametrize("sid", ["7629", "3307", "005654", "8953", "1", "071056"])
def test_audio_name_uses_the_id_verbatim_regression(sid):
    """Zero-padding a short id fetches a DIFFERENT recording.

    The bucket has filenames at every width 1..6 and `7629.wav` is not
    `007629.wav`; 4,961 short ids have a zero-padded twin that is its own
    manifest row. Padding put 12.5% of a run at CER 1.0 and inflated the
    measured noise floor from ~0.08 to 0.19.
    """
    assert CalibrationSample(sample_id=sid, text="x").audio_name == f"{sid}.wav"


def test_padded_and_unpadded_ids_are_distinct_samples():
    short = CalibrationSample(sample_id="7629", text="a")
    padded = CalibrationSample(sample_id="007629", text="b")
    assert short.audio_name != padded.audio_name


def test_load_manifest_is_deterministic_and_bounded(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(
        json.dumps({"sample_id": str(i), "text": f"line {i}", "duration_s": 5.0})
        for i in range(200)
    ), encoding="utf-8")
    a = load_manifest(p, 20, seed=7)
    b = load_manifest(p, 20, seed=7)
    assert len(a) == 20
    assert [s.sample_id for s in a] == [s.sample_id for s in b]
    assert [s.sample_id for s in load_manifest(p, 20, seed=8)] != [s.sample_id for s in a]


def test_load_manifest_skips_short_clips_and_bad_rows(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join([
        json.dumps({"sample_id": "1", "text": "keep me", "duration_s": 5.0}),
        json.dumps({"sample_id": "2", "text": "too short", "duration_s": 0.2}),
        json.dumps({"sample_id": "3", "text": ""}),              # no text
        json.dumps({"sample_id": "abc", "text": "non numeric"}),  # bad id
        "{not json}",
    ]), encoding="utf-8")
    got = load_manifest(p, 10, seed=1)
    assert [s.sample_id for s in got] == ["1"]


def test_summarize_excludes_invalid_rows_only():
    from rumik_oss_eval.calibrate import CalibrationRow
    from rumik_oss_eval.metrics.cer_wer import score_pair

    rows = [
        CalibrationRow("a", "original", "ref", "ref", "ref", "ref",
                       score_pair("ref", "ref"), "ok"),
        CalibrationRow("b", "original", "", "x", "", "x",
                       score_pair("", "x"), "ok"),      # empty reference -> invalid
    ]
    s = summarize(rows)["original"]
    assert s["n"] == 1                    # the invalid row is excluded
    assert s["cer_mean"] == 0.0
