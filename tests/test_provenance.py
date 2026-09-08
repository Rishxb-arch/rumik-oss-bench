"""Run provenance: what a run actually used, recorded beside its results.

These assert the record is complete enough to reconstruct a run's inputs without
the promptset file, because the promptset file is the thing that changes.
"""

from __future__ import annotations

import json

import pytest

from rumik_oss_eval.promptset import Prompt, PromptSet
from rumik_oss_eval.provenance import prompt_record, write_prompts_used, write_steps


class Bundle:
    """Mirrors TinyAyaBundle.prompt()'s wrapper."""
    def prompt(self, text, speaker=None):
        return f"<text>{speaker}: {text}<audio>"


def prompts():
    return [
        Prompt(id="ml_000030", text='<description="professional, Malayalam accent"> നന്ദി',
               speaker="Ira", language="ml", expected_seconds=22.4,
               tags=("conv", "failure")),
        Prompt(id="kn_000093", text="ನನಗೆ ಚಿತ್ರಕಲೆ", speaker="Ira", language="kn",
               expected_seconds=17.3, tags=("happy", "control")),
    ]


def rows(tmp_path):
    return [json.loads(l) for l in (tmp_path / "prompts_used.jsonl").read_text(
        encoding="utf-8").splitlines()]


# --- the record itself ------------------------------------------------------

def test_rendered_captures_the_wrapper_the_promptset_does_not_hold():
    """`rendered` is the only record of what the model actually saw; the
    <text>/<audio> wrapper is added at rollout time and appears nowhere else."""
    r = prompt_record(prompts()[0], Bundle())
    assert r["rendered"] == '<text>Ira: <description="professional, Malayalam accent"> നന്ദി<audio>'
    assert r["text"] != r["rendered"]


def test_reference_is_recorded_separately_from_text():
    """CER scores against `reference` (markup stripped), not `text`."""
    r = prompt_record(prompts()[0], Bundle())
    assert "description" in r["text"]
    assert "description" not in r["reference"]


def test_record_survives_without_a_bundle():
    r = prompt_record(prompts()[0])
    assert "rendered" not in r and r["id"] == "ml_000030"


def test_indic_text_is_not_mangled(tmp_path):
    write_prompts_used(tmp_path, prompts(), bundle=Bundle())
    assert "നന്ദി" in rows(tmp_path)[0]["text"]


# --- the files --------------------------------------------------------------

def test_one_jsonl_row_per_prompt_in_order(tmp_path):
    write_prompts_used(tmp_path, prompts(), bundle=Bundle())
    assert [r["id"] for r in rows(tmp_path)] == ["ml_000030", "kn_000093"]


def test_meta_records_the_source_hash(tmp_path):
    src = tmp_path / "indic_v2.yaml"
    src.write_text("name: indic_v2\n", encoding="utf-8")
    write_prompts_used(tmp_path, prompts(), promptset_path=src,
                       promptset=PromptSet(name="indic_v2", prompts=list(prompts()), version=3))
    m = json.loads((tmp_path / "promptset.meta.json").read_text(encoding="utf-8"))
    assert len(m["promptset_sha256"]) == 64
    assert m["promptset_name"] == "indic_v2" and m["promptset_version"] == 3
    assert m["n_prompts"] == 2


def test_meta_summarises_languages_scripts_and_tags(tmp_path):
    write_prompts_used(tmp_path, prompts(), bundle=Bundle())
    m = json.loads((tmp_path / "promptset.meta.json").read_text(encoding="utf-8"))
    assert m["languages"] == {"kn": 1, "ml": 1}
    assert m["tags"]["failure"] == 1 and m["tags"]["control"] == 1
    assert m["with_expected_seconds"] == 2


def test_a_missing_source_file_does_not_raise(tmp_path):
    """Provenance must never be the thing that kills a run."""
    write_prompts_used(tmp_path, prompts(), promptset_path=tmp_path / "gone.yaml")
    m = json.loads((tmp_path / "promptset.meta.json").read_text(encoding="utf-8"))
    assert m["promptset_sha256"] is None


def test_out_dir_is_created(tmp_path):
    write_prompts_used(tmp_path / "runs" / "new", prompts())
    assert (tmp_path / "runs" / "new" / "prompts_used.jsonl").exists()


def test_extra_fields_land_in_meta(tmp_path):
    write_prompts_used(tmp_path, prompts(), extra={"command": "grpo-smoke", "seed": 11})
    m = json.loads((tmp_path / "promptset.meta.json").read_text(encoding="utf-8"))
    assert m["command"] == "grpo-smoke" and m["seed"] == 11


# --- per-step selection -----------------------------------------------------

class Report:
    def __init__(self, step, ids):
        self.step, self.prompt_ids = step, ids
        self.group_size, self.n_groups, self.dead_groups = 8, 2, 1
        self.reward_mean, self.within_group_std = 0.42, 0.031


def test_steps_map_rewards_back_to_prompts(tmp_path):
    write_steps(tmp_path, [Report(0, ["a", "b"]), Report(1, ["c", "d"])])
    got = [json.loads(l) for l in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert [g["step"] for g in got] == [0, 1]
    assert got[0]["prompt_ids"] == ["a", "b"]
    assert got[1]["group_size"] == 8 and got[1]["dead_groups"] == 1


def test_empty_run_writes_an_empty_file_not_a_crash(tmp_path):
    write_steps(tmp_path, [])
    assert (tmp_path / "steps.jsonl").read_text() == ""


def test_steps_tolerate_a_report_without_prompt_ids(tmp_path):
    class Old:
        step = 0
    write_steps(tmp_path, [Old()])
    assert json.loads((tmp_path / "steps.jsonl").read_text())["prompt_ids"] == []
