"""Experiment 2 end to end on the synthetic curve: two families, a prior registry, disclosure."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from conftest import fake_fred
from termo.config import CoreConfig
from termo.core import (
    HYPOTHESES,
    INERTIA_TRIAL,
    MODEL_KMEANS_FROZEN,
    SETUP_TRIAL,
    build_report,
    frozen_trial_id,
    jump_trial_id,
    kmeans_trial_id,
    prior_log_summary,
    publish_report,
    register_trials,
    registered_columns,
    run_trials,
    take_snapshot,
)
from termo.data.loader import load_curve
from termo.data.snapshot import snapshot_hash
from termo.dataset import prepare
from termo.report import CoreReport, Verdict, report_payload
from termo.validation.trials import TrialLog, TrialLogError

COMMIT = "test-commit-2"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("termo-exp2")


@pytest.fixture(scope="module")
def prior_log(workspace: Path) -> Path:
    """A closed earlier registry: two trials and a no-go report."""
    log = TrialLog(workspace / "prior" / "trials.jsonl")
    log.register(SETUP_TRIAL, "setup", {}, "old-snapshot", "old-code")
    log.register("jm_k2_lam5", "h", {"model": "jump", "n_states": 2, "jump_penalty": 5}, "s", "c")
    log.register("inertia", "h", {"model": "inertia"}, "s", "c")
    log.record_report(report_payload(CoreReport(Verdict.NO_GO, "jm_k2_lam5", criteria=())))
    return log.path


@pytest.fixture(scope="module")
def exp2_config(tyccles_config: CoreConfig, prior_log: Path) -> CoreConfig:
    return replace(tyccles_config, prior_trial_logs=(prior_log.as_posix(),))


@pytest.fixture(scope="module")
def snapshot_dir(workspace: Path, curve: pd.DataFrame, exp2_config: CoreConfig) -> Path:
    target = workspace / "data" / "snapshots" / "2026-10-02"
    take_snapshot(exp2_config, target, fake_fred(curve), "2026-10-02T00:00:00+00:00")
    return target


@pytest.fixture(scope="module")
def log(workspace: Path) -> TrialLog:
    return TrialLog(workspace / "trials" / "exp2" / "trials.jsonl")


@pytest.fixture(scope="module")
def finished(
    workspace: Path, snapshot_dir: Path, log: TrialLog, exp2_config: CoreConfig, prior_log: Path
) -> Path:
    """Register, run and report once for the module. Returns the reports directory."""
    before = hashlib.sha256(prior_log.read_bytes()).hexdigest()
    data_hash = snapshot_hash(snapshot_dir)
    curve = load_curve(
        snapshot_dir, exp2_config.series, exp2_config.start, exp2_config.holdout_start
    )
    register_trials(exp2_config, curve, log, data_hash, COMMIT)
    messages: list[str] = []
    data = prepare(curve, exp2_config, registered_columns(log))
    trials_dir = workspace / "trials" / "exp2"
    run_trials(data, log, trials_dir, data_hash, COMMIT, echo=messages.append)
    assert len(messages) == 9 and all(m.isascii() and m.startswith("[ok]") for m in messages)
    report = build_report(data, log, trials_dir, data_hash, COMMIT)
    publish_report(report, log, workspace / "reports" / "exp2", data_hash, COMMIT)
    assert hashlib.sha256(prior_log.read_bytes()).hexdigest() == before  # never written
    return workspace / "reports" / "exp2"


def test_both_families_are_registered_with_the_hypotheses(
    finished: Path, log: TrialLog, exp2_config: CoreConfig
) -> None:
    registrations = log.registrations()
    expected = {SETUP_TRIAL, INERTIA_TRIAL}
    for k in exp2_config.k_values:
        expected |= {jump_trial_id(k, lam) for lam in exp2_config.jump_penalties}
        expected |= {kmeans_trial_id(k), frozen_trial_id(k)}
    assert set(registrations) == expected and len(registrations) == 10
    assert registrations["kmf_k2"]["config"] == {"model": MODEL_KMEANS_FROZEN, "n_states": 2}
    setup = registrations[SETUP_TRIAL]["config"]
    assert setup["hypotheses"] == list(HYPOTHESES) and len(HYPOTHESES) == 3
    assert setup["prior_trial_logs"] == [
        {"path": exp2_config.prior_trial_logs[0], "n_trials": 2, "verdict": "no-go"}
    ]
    assert len(setup["columns"]) == 139


def test_prior_logs_are_counted_at_registration(tmp_path: Path, tyccles_config: CoreConfig) -> None:
    assert prior_log_summary(tyccles_config) == []
    missing = replace(tyccles_config, prior_trial_logs=((tmp_path / "nope.jsonl").as_posix(),))
    with pytest.raises(TrialLogError, match="prior trial log not found"):
        prior_log_summary(missing)


def test_frozen_results_carry_the_family_markers(finished: Path, log: TrialLog) -> None:
    results = log.results()
    frozen = results["kmf_k2"]["metrics"]
    assert frozen["frozen"] is True and frozen["s1_mean"] is None
    assert frozen["stability"] == frozen["s2"]
    assert results["kmf_k2"]["status"] in {"kept", "discarded"}
    jump = results["jm_k2_lam0.5"]["metrics"]
    assert jump["frozen"] is False and jump["jump_penalty_effective"] == pytest.approx(69.5)
    assert results[kmeans_trial_id(2)]["reason"] == "baseline"
