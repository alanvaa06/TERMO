"""Spec 3 end to end on the synthetic curve: register, run, report, holdout."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import fake_fred, make_desc_config
from termo.config import CoreConfig
from termo.core import SETUP_TRIAL, registered_columns, take_snapshot
from termo.data.loader import load_curve
from termo.data.snapshot import snapshot_hash
from termo.dataset import ExperimentData, prepare
from termo.descriptive import stages
from termo.descriptive.criteria import APTO, NO_APTO
from termo.descriptive.stages import (
    FILES,
    HOLDOUT_DIR,
    PRE_HOLDOUT_DIR,
    analyse,
    desc_trial_id,
    holdout_desc,
    holdout_result,
    load_analysis,
    register_desc,
    report_desc,
    restrict,
    run_desc,
)
from termo.validation.trials import TrialLog, TrialLogError

COMMIT = "test-commit-3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("termo-desc")


@pytest.fixture(scope="module")
def prior_log(workspace: Path) -> Path:
    """A closed earlier registry: two trials and a no-go report, never to be written again."""
    log = TrialLog(workspace / "prior" / "trials.jsonl")
    log.register(SETUP_TRIAL, "setup", {}, "old-snapshot", "old-code")
    log.register("jm_k2_lam5", "h", {"model": "jump", "n_states": 2, "jump_penalty": 5}, "s", "c")
    log.register("inertia", "h", {"model": "inertia"}, "s", "c")
    log.record_report({"verdict": "no-go", "final_trial_id": "jm_k2_lam5"})
    return log.path


@pytest.fixture(scope="module")
def desc_config(prior_log: Path) -> CoreConfig:
    return replace(make_desc_config(), prior_trial_logs=(prior_log.as_posix(),))


@pytest.fixture(scope="module")
def snapshot_dir(workspace: Path, curve: pd.DataFrame, desc_config: CoreConfig) -> Path:
    target = workspace / "data" / "snapshots" / "2026-10-02"
    take_snapshot(desc_config, target, fake_fred(curve), "2026-10-02T00:00:00+00:00")
    return target


@pytest.fixture(scope="module")
def log(workspace: Path) -> TrialLog:
    return TrialLog(workspace / "trials" / "desc" / "trials.jsonl")


@pytest.fixture(scope="module")
def trials_dir(workspace: Path) -> Path:
    return workspace / "trials" / "desc"


@pytest.fixture(scope="module")
def data(
    snapshot_dir: Path, log: TrialLog, desc_config: CoreConfig, prior_log: Path
) -> ExperimentData:
    before = _sha256(prior_log)
    curve = load_curve(
        snapshot_dir, desc_config.series, desc_config.start, desc_config.holdout_start
    )
    register_desc(desc_config, curve, log, snapshot_hash(snapshot_dir), COMMIT)
    assert _sha256(prior_log) == before  # prior logs are read, never written
    return prepare(curve, desc_config, registered_columns(log))


@pytest.fixture(scope="module")
def finished(
    workspace: Path,
    data: ExperimentData,
    log: TrialLog,
    trials_dir: Path,
    snapshot_dir: Path,
    prior_log: Path,
) -> Path:
    before = _sha256(prior_log)
    data_hash = snapshot_hash(snapshot_dir)
    messages: list[str] = []
    run_desc(data, log, trials_dir, data_hash, COMMIT, echo=messages.append)
    assert len(messages) == 1 and messages[0].isascii() and messages[0].startswith("[ok]")
    reports = workspace / "reports" / "desc"
    report_desc(data, log, trials_dir, reports, data_hash, COMMIT)
    assert _sha256(prior_log) == before  # prior logs are read, never written
    return reports


def test_one_trial_is_registered_with_everything_it_is_bound_to(
    data: ExperimentData, log: TrialLog, desc_config: CoreConfig
) -> None:
    registrations = log.registrations()
    trial = desc_trial_id(desc_config)
    assert trial == "desc_k3" and set(registrations) == {SETUP_TRIAL, trial}
    assert registrations[trial]["config"] == {
        "model": "descriptive",
        "n_states": 3,
        "jump_penalty": 0.5,
    }
    setup = registrations[SETUP_TRIAL]["config"]
    assert len(setup["columns"]) == 139 and "xgboost" in setup["environment"]
    assert setup["config"]["descriptive"]["fidelity_min"] == 0.8
    assert setup["prior_trial_logs"] == [
        {"path": desc_config.prior_trial_logs[0], "n_trials": 2, "verdict": "no-go"}
    ]
    with pytest.raises(TrialLogError):
        register_desc(desc_config, data.curve, log, "x", COMMIT)


def test_blocks_that_miss_a_variable_fail_before_anything_is_registered(
    data: ExperimentData, desc_config: CoreConfig, tmp_path: Path
) -> None:
    assert desc_config.descriptive is not None
    blocks = tuple(
        (name, tuple(token for token in tokens if token != "d7"))
        for name, tokens in desc_config.descriptive.blocks
    )
    broken = replace(desc_config, descriptive=replace(desc_config.descriptive, blocks=blocks))
    empty = TrialLog(tmp_path / "trials.jsonl")
    with pytest.raises(ValueError, match="belongs to no block"):
        register_desc(broken, data.curve, empty, "x", COMMIT)
    assert empty.registrations() == {} and not empty.path.exists()


def test_analysis_covers_every_out_of_sample_day_without_look_ahead(
    data: ExperimentData, desc_config: CoreConfig
) -> None:
    analysis = analyse(data)
    days = analysis.labels.index
    assert days[0] > data.refits[0].cutoff and days[-1] == data.curve.index[-1]
    assert analysis.proba.index.equals(days) and analysis.shap_blocks.index.equals(days)
    assert analysis.shap_top.index.equals(days)
    assert list(analysis.proba.columns) == [0, 1, 2]
    assert list(analysis.shap_blocks.columns) == [
        "nivel corto",
        "nivel medio",
        "nivel largo",
        "pendientes",
        "curvatura",
        "volatilidad",
        "base",
    ]
    assert analysis.frozen_labels.index[0] > pd.Timestamp(desc_config.frozen_train_end)
    assert analysis.frozen_labels.index[-1] == days[-1]
    # no look-ahead: cutting the last 60 days leaves every earlier output unchanged
    shorter = analyse(prepare(data.curve.iloc[:-60], desc_config, data.columns))
    common = shorter.labels.index
    pd.testing.assert_series_equal(shorter.labels, analysis.labels.loc[common])
    pd.testing.assert_frame_equal(shorter.proba, analysis.proba.loc[common])
    pd.testing.assert_frame_equal(shorter.shap_blocks, analysis.shap_blocks.loc[common])
    pd.testing.assert_frame_equal(shorter.shap_top, analysis.shap_top.loc[common])
    frozen_days = shorter.frozen_labels.index
    assert len(frozen_days) == len(analysis.frozen_labels) - 60
    pd.testing.assert_series_equal(shorter.frozen_labels, analysis.frozen_labels.loc[frozen_days])


def test_run_writes_hashed_files_and_records_the_checks(
    finished: Path, log: TrialLog, trials_dir: Path, desc_config: CoreConfig
) -> None:
    result = log.results()[desc_trial_id(desc_config)]
    metrics = result["metrics"]
    assert set(metrics["files"]) == set(FILES)
    for name, expected in metrics["files"].items():
        path = trials_dir / PRE_HOLDOUT_DIR / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    assert result["labels_sha256"] == metrics["files"]["labels.csv"]
    assert [c["name"] for c in metrics["checks"]][0] == "D1_persistence"
    assert len(metrics["checks"]) == 7
    assert metrics["verdict"] in {APTO, NO_APTO}
    assert set(metrics["calibration"]) == {"brier", "reliability", "recall_by_phase"}
    json.dumps(metrics)


def test_run_never_repeats_and_files_round_trip(
    finished: Path, data: ExperimentData, log: TrialLog, trials_dir: Path, snapshot_dir: Path
) -> None:
    before = log.path.read_text(encoding="utf-8")
    messages: list[str] = []
    run_desc(data, log, trials_dir, snapshot_hash(snapshot_dir), COMMIT, echo=messages.append)
    assert messages == ["[ok] desc_k3 already has a result; nothing to run"]
    assert log.path.read_text(encoding="utf-8") == before
    loaded = load_analysis(
        trials_dir / PRE_HOLDOUT_DIR, log.results()["desc_k3"]["metrics"]["files"]
    )
    fresh = analyse(data)
    pd.testing.assert_series_equal(loaded.labels, fresh.labels, check_names=False, check_freq=False)
    assert np.allclose(loaded.proba.to_numpy(), fresh.proba.to_numpy())
    assert np.allclose(
        loaded.shap_blocks.to_numpy(), fresh.shap_blocks.to_numpy(), equal_nan=True, atol=1e-9
    )


def test_report_recomputes_from_the_files_and_is_logged(
    finished: Path, log: TrialLog, desc_config: CoreConfig
) -> None:
    payload = json.loads((finished / "diagnostic.json").read_text(encoding="utf-8"))
    logged = log.last_report()
    assert logged is not None and logged["stage"] == "diagnostic"
    assert logged["verdict"] == payload["verdict"] in {APTO, NO_APTO}
    assert payload["checks"] == log.results()["desc_k3"]["metrics"]["checks"]
    assert payload["final_trial_id"] == "desc_k3"
    assert payload["disclosure"]["n_trials_this_log"] == 1
    assert payload["disclosure"]["n_trials_total"] == 3  # two in the prior log, one here
    markdown = (finished / "diagnostic.md").read_text(encoding="utf-8")
    assert markdown.isascii() and "does not anticipate" in markdown
    assert "D6_fidelity" in markdown and "pre-holdout" in markdown
    assert "2 trials, verdict no-go" in markdown and "Total registered trials: 3" in markdown


def test_report_refuses_tampered_files_and_a_changed_configuration(
    finished: Path,
    data: ExperimentData,
    log: TrialLog,
    trials_dir: Path,
    snapshot_dir: Path,
    tmp_path: Path,
) -> None:
    data_hash = snapshot_hash(snapshot_dir)
    other = replace(data, config=replace(data.config, refit_weeks=13))
    with pytest.raises(TrialLogError, match="configuration"):
        report_desc(other, log, trials_dir, tmp_path, data_hash, COMMIT)
    path = trials_dir / PRE_HOLDOUT_DIR / "proba.csv"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        with pytest.raises(TrialLogError, match="not the ones recorded"):
            report_desc(data, log, trials_dir, tmp_path, data_hash, COMMIT)
    finally:
        path.write_bytes(original)


def test_run_and_report_refuse_data_with_holdout_rows(
    finished: Path,
    data: ExperimentData,
    log: TrialLog,
    trials_dir: Path,
    snapshot_dir: Path,
    desc_config: CoreConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_hash = snapshot_hash(snapshot_dir)
    full_curve = load_curve(
        snapshot_dir,
        desc_config.series,
        desc_config.start,
        desc_config.holdout_start,
        final_evaluation=True,
    )
    assert full_curve.index[-1] >= pd.Timestamp(desc_config.holdout_start)
    leaking = prepare(full_curve, desc_config, data.columns)

    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("nothing may be computed on data with holdout rows")

    monkeypatch.setattr(stages, "analyse", never)
    monkeypatch.setattr(stages, "load_analysis", never)
    # a log where the trial is registered but still pending, so run would really run
    pending = TrialLog(tmp_path / "trials.jsonl")
    lines = log.path.read_text(encoding="utf-8").splitlines(keepends=True)
    pending.path.write_text(
        "".join(line for line in lines if json.loads(line)["kind"] == "registered"),
        encoding="utf-8",
    )
    assert pending.is_registered("desc_k3") and not pending.has_result("desc_k3")
    before = pending.path.read_text(encoding="utf-8")
    messages: list[str] = []
    with pytest.raises(TrialLogError, match="pre-holdout"):
        run_desc(leaking, pending, tmp_path / "out", data_hash, COMMIT, echo=messages.append)
    assert messages == [] and pending.path.read_text(encoding="utf-8") == before
    assert not (tmp_path / "out").exists()

    before = log.path.read_text(encoding="utf-8")
    with pytest.raises(TrialLogError, match="pre-holdout"):
        report_desc(leaking, log, trials_dir, tmp_path / "reports", data_hash, COMMIT)
    assert log.path.read_text(encoding="utf-8") == before
    assert not (tmp_path / "reports").exists()


def _approved_copy(log: TrialLog, tmp_path: Path, verdict: str = APTO) -> TrialLog:
    """A private copy of the finished log whose last report says `verdict`."""
    copy = TrialLog(tmp_path / "trials.jsonl")
    copy.path.write_text(log.path.read_text(encoding="utf-8"), encoding="utf-8")
    last = log.last_report()
    assert last is not None
    payload = {k: v for k, v in last.items() if k not in {"kind", "trial_id", "at"}}
    copy.record_report({**payload, "verdict": verdict})
    return copy


def test_holdout_needs_an_apt_diagnostic(
    finished: Path,
    tmp_path: Path,
    snapshot_dir: Path,
    log: TrialLog,
    trials_dir: Path,
    desc_config: CoreConfig,
) -> None:
    empty = TrialLog(tmp_path / "empty.jsonl")
    with pytest.raises(TrialLogError):
        holdout_desc(desc_config, snapshot_dir, empty, trials_dir, tmp_path, COMMIT)
    refused = _approved_copy(log, tmp_path, NO_APTO)
    with pytest.raises(TrialLogError, match="diagnostic"):
        holdout_desc(desc_config, snapshot_dir, refused, trials_dir, tmp_path, COMMIT)
    assert not refused.holdout_opened()


def test_holdout_checks_everything_before_opening(
    finished: Path,
    tmp_path: Path,
    snapshot_dir: Path,
    log: TrialLog,
    trials_dir: Path,
    desc_config: CoreConfig,
) -> None:
    copy = _approved_copy(log, tmp_path)
    with pytest.raises(TrialLogError, match="code commit"):
        holdout_desc(desc_config, snapshot_dir, copy, trials_dir, tmp_path, "another-commit")
    changed = replace(desc_config, refit_weeks=13)
    with pytest.raises(TrialLogError, match="configuration"):
        holdout_desc(changed, snapshot_dir, copy, trials_dir, tmp_path, COMMIT)
    assert not copy.holdout_opened()


def test_holdout_checks_its_outputs_are_writable_before_opening(
    finished: Path,
    tmp_path: Path,
    snapshot_dir: Path,
    log: TrialLog,
    desc_config: CoreConfig,
) -> None:
    copy = _approved_copy(log, tmp_path)
    a_file = tmp_path / "a_file"
    a_file.write_text("not a directory", encoding="utf-8")
    good = tmp_path / "good"
    with pytest.raises(TrialLogError, match="cannot be written"):
        holdout_desc(desc_config, snapshot_dir, copy, a_file, good / "reports", COMMIT)
    assert not copy.holdout_opened()
    with pytest.raises(TrialLogError, match="cannot be written"):
        holdout_desc(desc_config, snapshot_dir, copy, good / "trials", a_file, COMMIT)
    assert not copy.holdout_opened()
    # an output path that cannot be opened as a file (here: it is a directory)
    (good / "trials" / HOLDOUT_DIR / "proba.csv").mkdir()
    with pytest.raises(TrialLogError, match="proba.csv"):
        holdout_desc(desc_config, snapshot_dir, copy, good / "trials", good / "reports", COMMIT)
    assert not copy.holdout_opened()
    assert a_file.read_text(encoding="utf-8") == "not a directory"
    # the probes leave nothing behind
    assert [p for p in good.rglob("*") if p.is_file()] == []


def test_holdout_logs_the_result_before_writing_any_file(
    finished: Path,
    tmp_path: Path,
    snapshot_dir: Path,
    log: TrialLog,
    desc_config: CoreConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copy = _approved_copy(log, tmp_path)

    def full_disk(path: Path, data: bytes) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(stages, "_write_bytes", full_disk)
    with pytest.raises(OSError, match="disk full"):
        holdout_desc(
            desc_config, snapshot_dir, copy, tmp_path / "trials", tmp_path / "reports", COMMIT
        )
    kinds = [r["kind"] for r in copy.records()]
    assert kinds[-2:] == ["holdout_opened", "holdout_result"]
    logged = copy.records()[-1]
    assert logged["verdict"] in {APTO, NO_APTO} and len(logged["checks"]) == 7
    assert set(logged["files"]) == set(FILES)
    assert all(len(digest) == 64 for digest in logged["files"].values())


@dataclass(frozen=True)
class Opened:
    """The holdout stage run once on a private copy of the finished log."""

    log: TrialLog
    trials: Path
    reports: Path
    result: str


@pytest.fixture(scope="module")
def opened(
    finished: Path,
    tmp_path_factory: pytest.TempPathFactory,
    snapshot_dir: Path,
    log: TrialLog,
    desc_config: CoreConfig,
    prior_log: Path,
) -> Opened:
    root = tmp_path_factory.mktemp("termo-desc-holdout")
    copy = _approved_copy(log, root)
    before = _sha256(prior_log)
    result = holdout_desc(
        desc_config, snapshot_dir, copy, root / "trials", root / "reports", COMMIT
    )
    assert _sha256(prior_log) == before  # prior logs are read, never written
    return Opened(copy, root / "trials", root / "reports", result)


def test_holdout_runs_once_on_holdout_days_only(
    opened: Opened,
    snapshot_dir: Path,
    log: TrialLog,
    desc_config: CoreConfig,
    curve: pd.DataFrame,
) -> None:
    copy, private_trials, reports, result = opened.log, opened.trials, opened.reports, opened.result
    start = pd.Timestamp(desc_config.holdout_start)
    kinds = [r["kind"] for r in copy.records()]
    assert kinds[-2:] == ["holdout_opened", "holdout_result"]
    logged = copy.records()[-1]
    assert logged["verdict"] == result in {APTO, NO_APTO} and len(logged["checks"]) == 7
    assert logged["days"] == int((curve.index >= start).sum())
    assert set(logged["files"]) == set(FILES)
    for name, expected in logged["files"].items():
        assert _sha256(private_trials / HOLDOUT_DIR / name) == expected
    # the recomputed past is the registered run: same labels before the holdout, byte for byte
    assert logged["pre_holdout_labels_sha256"] == log.results()["desc_k3"]["labels_sha256"]
    loaded = load_analysis(private_trials / HOLDOUT_DIR, logged["files"])
    assert loaded.labels.index[0] >= start and loaded.proba.index.equals(loaded.labels.index)
    # the frozen map of the holdout was fitted on everything before it, never on holdout days
    assert loaded.frozen_labels.index.equals(loaded.labels.index)
    full_curve = load_curve(
        snapshot_dir,
        desc_config.series,
        desc_config.start,
        desc_config.holdout_start,
        final_evaluation=True,
    )
    last_seen = (start - pd.Timedelta(days=1)).date()
    frozen_before = replace(desc_config, frozen_train_end=last_seen)
    expected_analysis = restrict(
        analyse(prepare(full_curve, frozen_before, registered_columns(copy))), start
    )
    pd.testing.assert_series_equal(
        loaded.frozen_labels,
        expected_analysis.frozen_labels,
        check_names=False,
        check_freq=False,
        check_dtype=False,
    )
    # that cutoff is not the registered frozen date: the stage moved it to the holdout's eve
    assert frozen_before.frozen_train_end != desc_config.frozen_train_end
    payload = json.loads((reports / "holdout.json").read_text(encoding="utf-8"))
    assert payload["verdict"] == result and payload["stage"] == "holdout"
    assert payload["pre_holdout_labels_sha256"] == logged["pre_holdout_labels_sha256"]
    assert (reports / "holdout.md").read_text(encoding="utf-8").isascii()
    # a second call evaluates nothing new: the log is untouched and the verdict is the logged one
    before = {path: _sha256(path) for path in (private_trials / HOLDOUT_DIR).iterdir()}
    before[copy.path] = _sha256(copy.path)
    assert holdout_desc(desc_config, snapshot_dir, copy, private_trials, reports, COMMIT) == result
    assert {path: _sha256(path) for path in before} == before


def test_holdout_files_are_restored_from_the_logged_result_only_when_they_match(
    opened: Opened, snapshot_dir: Path, desc_config: CoreConfig, tmp_path: Path
) -> None:
    copy, private_trials, reports = opened.log, opened.trials, opened.reports
    logged = holdout_result(copy)
    assert logged is not None
    log_before = copy.path.read_bytes()
    for name in FILES:
        (private_trials / HOLDOUT_DIR / name).unlink()
    assert holdout_desc(desc_config, snapshot_dir, copy, private_trials, reports, COMMIT) == (
        opened.result
    )
    for name, expected in logged["files"].items():
        assert _sha256(private_trials / HOLDOUT_DIR / name) == expected
    assert copy.path.read_bytes() == log_before
    # a log whose recorded hashes the recomputation cannot reproduce gets no files at all
    damaged = TrialLog(tmp_path / "trials.jsonl")
    lines = [json.loads(line) for line in copy.path.read_text(encoding="utf-8").splitlines()]
    lines[-1]["files"]["proba.csv"] = "0" * 64
    damaged.path.write_text(
        "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines), encoding="utf-8"
    )
    with pytest.raises(TrialLogError, match="does not match the logged result"):
        holdout_desc(
            desc_config, snapshot_dir, damaged, tmp_path / "trials", tmp_path / "reports", COMMIT
        )
    assert not (tmp_path / "trials" / HOLDOUT_DIR / "labels.csv").exists()
    assert not (tmp_path / "reports").exists() or not any((tmp_path / "reports").iterdir())
    assert damaged.path.read_text(encoding="utf-8") == "".join(
        json.dumps(line, sort_keys=True) + "\n" for line in lines
    )
