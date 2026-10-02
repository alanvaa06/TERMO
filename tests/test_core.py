"""The five stages end to end, on a synthetic curve written as a fake FRED snapshot."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.core import (
    HOLDOUT_JSON,
    INERTIA_TRIAL,
    SETUP_TRIAL,
    build_report,
    jump_trial_id,
    kmeans_trial_id,
    load_labels,
    register_trials,
    registered_columns,
    run_final_holdout,
    run_trials,
    save_labels,
    take_snapshot,
)
from termo.data.fred import FRED_URL, DataValidationError
from termo.data.loader import load_curve
from termo.data.snapshot import snapshot_hash
from termo.dataset import prepare
from termo.report import CoreReport, Verdict, write_report
from termo.validation.trials import TrialLog, TrialLogError


def fake_fred(curve: pd.DataFrame) -> Callable[[str], str]:
    def get(url: str) -> str:
        for series_id in curve.columns:
            if url == FRED_URL.format(series_id=series_id):
                lines = [f"observation_date,{series_id}"]
                lines += [f"{d:%Y-%m-%d},{v:.4f}" for d, v in curve[series_id].items()]
                return "\n".join(lines) + "\n"
        raise AssertionError(f"unexpected url {url}")

    return get


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("termo")


@pytest.fixture(scope="module")
def snapshot_dir(workspace: Path, curve: pd.DataFrame, config: CoreConfig) -> Path:
    target = workspace / "data" / "snapshots" / "2026-10-02"
    take_snapshot(config, target, fake_fred(curve), "2026-10-02T00:00:00+00:00")
    return target


@pytest.fixture(scope="module")
def log(workspace: Path) -> TrialLog:
    return TrialLog(workspace / "trials" / "trials.jsonl")


@pytest.fixture(scope="module")
def finished(workspace: Path, snapshot_dir: Path, log: TrialLog, config: CoreConfig) -> Path:
    """Register, run and report once for the whole module. Returns the reports directory."""
    curve = load_curve(snapshot_dir, config.series, config.start, config.holdout_start)
    register_trials(config, curve, log, snapshot_hash(snapshot_dir), "test-commit")
    messages: list[str] = []
    data = prepare(curve, config, registered_columns(log))
    run_trials(data, log, workspace / "trials", echo=messages.append)
    assert len(messages) == 7 and all(m.isascii() and m.startswith("[ok]") for m in messages)
    report = build_report(data, log, workspace / "trials")
    write_report(report, workspace / "reports")
    return workspace / "reports"


def test_snapshot_refuses_invalid_downloads(tmp_path: Path, config: CoreConfig) -> None:
    with pytest.raises(DataValidationError):
        take_snapshot(config, tmp_path / "snap", lambda url: "observation_date,X\n", "now")
    assert not (tmp_path / "snap").exists()


def test_everything_is_registered_before_any_result(
    finished: Path, log: TrialLog, config: CoreConfig
) -> None:
    records = log.records()
    kinds = [r["kind"] for r in records]
    first_result = kinds.index("result")
    assert set(kinds[:first_result]) == {"registered"}
    assert "registered" not in kinds[first_result:]
    expected = {SETUP_TRIAL, INERTIA_TRIAL}
    expected |= {jump_trial_id(k, lam) for k in config.k_values for lam in config.jump_penalties}
    expected |= {kmeans_trial_id(k) for k in config.k_values}
    assert set(log.registrations()) == expected
    assert set(log.results()) == expected - {SETUP_TRIAL}
    assert log.registrations()["jm_k2_lam50"]["code_commit"] == "test-commit"


def test_registration_cannot_be_repeated(
    finished: Path, snapshot_dir: Path, log: TrialLog, config: CoreConfig
) -> None:
    curve = load_curve(snapshot_dir, config.series, config.start, config.holdout_start)
    with pytest.raises(TrialLogError, match="already registered"):
        register_trials(config, curve, log, snapshot_hash(snapshot_dir), "test-commit")


def test_run_is_resumable_and_never_reruns_a_trial(
    finished: Path, workspace: Path, snapshot_dir: Path, log: TrialLog, config: CoreConfig
) -> None:
    curve = load_curve(snapshot_dir, config.series, config.start, config.holdout_start)
    before = log.path.read_text(encoding="utf-8")
    messages: list[str] = []
    run_trials(
        prepare(curve, config, registered_columns(log)), log, workspace / "trials", messages.append
    )
    assert messages == []
    assert log.path.read_text(encoding="utf-8") == before


def test_labels_round_trip(workspace: Path, finished: Path, tmp_path: Path) -> None:
    labels = load_labels(workspace / "trials" / "labels" / "jm_k2_lam50.csv")
    assert labels.index.name == "date" and labels.dtype.kind == "i"
    save_labels(labels, tmp_path / "copy.csv")
    pd.testing.assert_series_equal(load_labels(tmp_path / "copy.csv"), labels)


def test_report_reaches_a_verdict_with_all_criteria(finished: Path, config: CoreConfig) -> None:
    payload = json.loads((finished / "go_no_go.json").read_text(encoding="utf-8"))
    assert payload["verdict"] in {"go", "no-go"}
    assert payload["final_trial_id"] in {
        jump_trial_id(k, lam) for k in config.k_values for lam in config.jump_penalties
    }
    names = [c["name"] for c in payload["criteria"]]
    assert names == [
        "stability",
        "separation_vs_inertia_low95",
        "independence_p",
        "pbo",
        "s2_halves",
    ]
    blocking_failed = any(c["blocking"] and not c["passed"] for c in payload["criteria"])
    assert (payload["verdict"] == "no-go") == blocking_failed
    details = payload["details"]
    assert details["n_trials"] == 4 and 1 <= details["n_effective"] <= 4
    assert len(details["configurations"]) == 4
    assert set(details["baselines_separation"]) == {"inertia", "kmeans_k2", "kmeans_k3"}
    assert (finished / "go_no_go.md").read_text(encoding="utf-8").isascii()


def test_report_needs_every_result(tmp_path: Path, snapshot_dir: Path, config: CoreConfig) -> None:
    curve = load_curve(snapshot_dir, config.series, config.start, config.holdout_start)
    fresh = TrialLog(tmp_path / "trials.jsonl")
    columns = register_trials(config, curve, fresh, "hash", "commit")
    with pytest.raises(TrialLogError, match="without a result"):
        build_report(prepare(curve, config, columns), fresh, tmp_path)


def test_stages_need_a_registration_first(tmp_path: Path) -> None:
    with pytest.raises(TrialLogError, match="register stage first"):
        registered_columns(TrialLog(tmp_path / "trials.jsonl"))


def test_final_holdout_needs_a_go_verdict(
    finished: Path, tmp_path: Path, snapshot_dir: Path, log: TrialLog, config: CoreConfig
) -> None:
    reports = tmp_path / "reports"
    write_report(CoreReport(Verdict.NO_GO, "jm_k2_lam10", criteria=()), reports)
    private_log = TrialLog(tmp_path / "trials.jsonl")
    private_log.path.write_text(log.path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(TrialLogError, match="GO verdict"):
        run_final_holdout(config, snapshot_dir, private_log, reports)
    assert not private_log.holdout_opened()


def test_final_holdout_runs_exactly_once(
    finished: Path, tmp_path: Path, snapshot_dir: Path, log: TrialLog, config: CoreConfig
) -> None:
    reports = tmp_path / "reports"
    write_report(CoreReport(Verdict.GO, "jm_k2_lam10", criteria=()), reports)
    private_log = TrialLog(tmp_path / "trials.jsonl")
    private_log.path.write_text(log.path.read_text(encoding="utf-8"), encoding="utf-8")

    result = run_final_holdout(config, snapshot_dir, private_log, reports)

    assert private_log.holdout_opened()
    assert private_log.records()[-1]["final_trial_id"] == "jm_k2_lam10"
    saved = json.loads((reports / HOLDOUT_JSON).read_text(encoding="utf-8"))
    assert saved["final_trial_id"] == "jm_k2_lam10"
    assert saved["passed"] == result.passed and saved["n_weeks"] == result.n_weeks > 0
    with pytest.raises(TrialLogError, match="already been opened"):
        run_final_holdout(config, snapshot_dir, private_log, reports)
