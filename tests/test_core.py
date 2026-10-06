"""The five stages end to end, on a synthetic curve written as a fake FRED snapshot."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from conftest import fake_fred
from termo.config import CoreConfig
from termo.core import (
    HOLDOUT_JSON,
    INERTIA_TRIAL,
    SETUP_TRIAL,
    build_report,
    config_fingerprint,
    environment_fingerprint,
    jump_trial_id,
    kmeans_trial_id,
    load_labels,
    publish_report,
    register_trials,
    registered_columns,
    run_final_holdout,
    run_trials,
    save_labels,
    take_snapshot,
    verify_binding,
    verify_configuration,
)
from termo.data.fred import DataValidationError
from termo.data.loader import load_curve
from termo.data.snapshot import SnapshotError, snapshot_hash
from termo.dataset import prepare
from termo.report import CoreReport, Verdict, report_payload, write_report
from termo.validation.trials import TrialLog, TrialLogError

COMMIT = "test-commit"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("termo")


@pytest.fixture(scope="module")
def snapshot_dir(workspace: Path, curve: pd.DataFrame, config: CoreConfig) -> Path:
    target = workspace / "data" / "snapshots" / "2026-10-02"
    take_snapshot(config, target, fake_fred(curve), "2026-10-02T00:00:00+00:00")
    return target


@pytest.fixture(scope="module")
def data_hash(snapshot_dir: Path) -> str:
    return snapshot_hash(snapshot_dir)


@pytest.fixture(scope="module")
def log(workspace: Path) -> TrialLog:
    return TrialLog(workspace / "trials" / "trials.jsonl")


@pytest.fixture(scope="module")
def snapshot_data(snapshot_dir: Path, config: CoreConfig) -> pd.DataFrame:
    return load_curve(snapshot_dir, config.series, config.start, config.holdout_start)


@pytest.fixture(scope="module")
def finished(
    workspace: Path,
    snapshot_data: pd.DataFrame,
    data_hash: str,
    log: TrialLog,
    config: CoreConfig,
) -> Path:
    """Register, run and report once for the whole module. Returns the reports directory."""
    register_trials(config, snapshot_data, log, data_hash, COMMIT)
    messages: list[str] = []
    data = prepare(snapshot_data, config, registered_columns(log))
    run_trials(data, log, workspace / "trials", data_hash, COMMIT, echo=messages.append)
    assert len(messages) == 7 and all(m.isascii() and m.startswith("[ok]") for m in messages)
    report = build_report(data, log, workspace / "trials", data_hash, COMMIT)
    publish_report(report, log, workspace / "reports", data_hash, COMMIT)
    return workspace / "reports"


def test_snapshot_refuses_a_hole_anywhere_in_the_download(
    tmp_path: Path, curve: pd.DataFrame, config: CoreConfig
) -> None:
    """Even a hole inside the holdout period is caught here, before anything is registered."""
    holed = curve.copy()
    holdout_days = holed.index[holed.index >= pd.Timestamp(config.holdout_start)]
    holed.loc[holdout_days[20:80], "DGS7"] = float("nan")

    def get(url: str) -> str:
        return fake_fred(holed)(url).replace(",nan\n", ",\n")

    with pytest.raises(DataValidationError, match="gap of"):
        take_snapshot(config, tmp_path / "snap", get, "now")
    assert not (tmp_path / "snap").exists()


def test_a_tampered_file_does_not_burn_the_holdout(
    finished: Path, tmp_path: Path, snapshot_dir: Path, log: TrialLog, config: CoreConfig
) -> None:
    """Same manifest, different bytes: the binding passes, the file check must stop it."""
    tampered = tmp_path / "snapshot"
    shutil.copytree(snapshot_dir, tampered)
    path = tampered / "DGS5.csv"
    path.write_bytes(path.read_bytes().replace(b"\n1990-01-02,", b"\n1990-01-02,9", 1))
    copy = private_copy(log, tmp_path)
    copy.record_report(go_report())
    with pytest.raises(SnapshotError, match="hash mismatch for DGS5"):
        run_final_holdout(config, tampered, copy, tmp_path, COMMIT)
    assert not copy.holdout_opened()


def private_copy(log: TrialLog, tmp_path: Path) -> TrialLog:
    """A log that tests may write to without touching the shared one."""
    copy = TrialLog(tmp_path / "trials.jsonl")
    copy.path.write_text(log.path.read_text(encoding="utf-8"), encoding="utf-8")
    return copy


def go_report(trial_id: str = "jm_k2_lam10") -> dict[str, object]:
    return report_payload(CoreReport(Verdict.GO, trial_id, criteria=()))


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
    assert log.registrations()["jm_k2_lam50"]["code_commit"] == COMMIT


def test_registration_stores_the_whole_configuration(
    finished: Path, log: TrialLog, config: CoreConfig, data_hash: str
) -> None:
    setup = log.registrations()[SETUP_TRIAL]
    stored = setup["config"]["config"]
    assert stored == config_fingerprint(config)
    assert stored["holdout_start"] == config.holdout_start.isoformat()
    assert stored["horizon_short_days"] == config.horizon_short_days
    assert stored["bootstrap"] == {"block_weeks": 8, "n_resamples": 200, "seed": 0}
    assert stored["thresholds"]["independence_p_max"] == 0.01
    assert setup["snapshot_hash"] == data_hash


def test_registration_needs_committed_code(
    tmp_path: Path, snapshot_data: pd.DataFrame, config: CoreConfig
) -> None:
    fresh = TrialLog(tmp_path / "trials.jsonl")
    with pytest.raises(TrialLogError, match="commit the code"):
        register_trials(config, snapshot_data, fresh, "hash", "abc123-dirty")
    assert fresh.records() == []


def test_registration_cannot_be_repeated(
    finished: Path, snapshot_data: pd.DataFrame, data_hash: str, log: TrialLog, config: CoreConfig
) -> None:
    with pytest.raises(TrialLogError, match="already registered"):
        register_trials(config, snapshot_data, log, data_hash, COMMIT)


@pytest.mark.parametrize(
    ("change", "named"),
    [
        ({"config": {"horizon_short_days": 10}}, "configuration"),
        ({"config": {"holdout_start": pd.Timestamp("1999-01-01").date()}}, "configuration"),
        ({"snapshot_hash": "another-snapshot"}, "data snapshot"),
        ({"code_commit": "another-commit"}, "code commit"),
    ],
)
def test_work_is_bound_to_what_was_registered(
    change: dict[str, object],
    named: str,
    finished: Path,
    log: TrialLog,
    config: CoreConfig,
    data_hash: str,
) -> None:
    current = replace(config, **change.get("config", {}))  # type: ignore[arg-type]
    current_hash = str(change.get("snapshot_hash", data_hash))
    current_commit = str(change.get("code_commit", COMMIT))
    with pytest.raises(TrialLogError, match=named):
        verify_binding(log, current, current_hash, current_commit)
    verify_binding(log, config, data_hash, COMMIT)  # the registered combination passes


def test_the_configuration_alone_binds_a_stage_that_takes_a_new_snapshot_every_time(
    finished: Path, log: TrialLog, config: CoreConfig
) -> None:
    """The shadow operation recomputes on a fresh snapshot each week: it binds the
    configuration, and only that; the snapshot and the code are not compared."""
    verify_configuration(log, config)
    with pytest.raises(TrialLogError, match="configuration"):
        verify_configuration(log, replace(config, refit_weeks=13))
    with pytest.raises(TrialLogError, match="no setup registration"):
        verify_configuration(TrialLog(log.path.parent / "missing.jsonl"), config)


def test_a_registration_that_predates_an_optional_key_reads_it_as_its_default(
    tmp_path: Path,
) -> None:
    """Later code may add optional keys; an earlier registry must still bind to its own
    configuration, and a value other than the default must still be refused."""
    from conftest import make_desc_config

    desc = make_desc_config()
    assert desc.descriptive is not None
    fingerprint = config_fingerprint(desc)
    assert fingerprint["descriptive"]["holdout_already_seen"] is False
    del fingerprint["descriptive"]["holdout_already_seen"]
    del fingerprint["descriptive"]["holdout_seen_by"]
    older = TrialLog(tmp_path / "trials.jsonl")
    older.register(
        SETUP_TRIAL,
        "setup",
        {"config": fingerprint, "environment": environment_fingerprint()},
        "hash",
        COMMIT,
    )
    verify_binding(older, desc, "hash", COMMIT)
    seen = replace(desc, descriptive=replace(desc.descriptive, holdout_already_seen=True))
    with pytest.raises(TrialLogError, match="configuration"):
        verify_binding(older, seen, "hash", COMMIT)
    with pytest.raises(TrialLogError, match="configuration"):
        verify_binding(older, replace(desc, prior_trial_logs=("a.jsonl",)), "hash", COMMIT)
    # a key that is NOT on the explicit list is never filled in, even with a default
    unlisted = dict(fingerprint)
    del unlisted["prior_trial_logs"]
    strict = TrialLog(tmp_path / "unlisted.jsonl")
    strict.register(
        SETUP_TRIAL,
        "setup",
        {"config": unlisted, "environment": environment_fingerprint()},
        "hash",
        COMMIT,
    )
    with pytest.raises(TrialLogError, match="configuration"):
        verify_binding(strict, desc, "hash", COMMIT)
    # a key the registration has but the code no longer knows is a difference too
    extra = dict(fingerprint)
    extra["descriptive"] = {**fingerprint["descriptive"], "removed_later": 42}
    stale = TrialLog(tmp_path / "stale.jsonl")
    stale.register(
        SETUP_TRIAL,
        "setup",
        {"config": extra, "environment": environment_fingerprint()},
        "hash",
        COMMIT,
    )
    with pytest.raises(TrialLogError, match="configuration"):
        verify_binding(stale, desc, "hash", COMMIT)
    # a required key the registration lacks is a difference, never filled in
    del fingerprint["refit_weeks"]
    lacking = TrialLog(tmp_path / "lacking.jsonl")
    lacking.register(
        SETUP_TRIAL,
        "setup",
        {"config": fingerprint, "environment": environment_fingerprint()},
        "hash",
        COMMIT,
    )
    with pytest.raises(TrialLogError, match="configuration"):
        verify_binding(lacking, desc, "hash", COMMIT)


def test_run_and_report_refuse_another_configuration(
    finished: Path,
    workspace: Path,
    snapshot_data: pd.DataFrame,
    data_hash: str,
    log: TrialLog,
    config: CoreConfig,
) -> None:
    other = replace(config, horizon_short_days=10)
    data = prepare(snapshot_data, other, registered_columns(log))
    before = log.path.read_text(encoding="utf-8")
    with pytest.raises(TrialLogError, match="configuration"):
        run_trials(data, log, workspace / "trials", data_hash, COMMIT)
    with pytest.raises(TrialLogError, match="configuration"):
        build_report(data, log, workspace / "trials", data_hash, COMMIT)
    assert log.path.read_text(encoding="utf-8") == before


def test_run_is_resumable_and_never_reruns_a_trial(
    finished: Path,
    workspace: Path,
    snapshot_data: pd.DataFrame,
    data_hash: str,
    log: TrialLog,
    config: CoreConfig,
) -> None:
    before = log.path.read_text(encoding="utf-8")
    messages: list[str] = []
    data = prepare(snapshot_data, config, registered_columns(log))
    run_trials(data, log, workspace / "trials", data_hash, COMMIT, messages.append)
    assert messages == []
    assert log.path.read_text(encoding="utf-8") == before


def test_labels_round_trip(workspace: Path, finished: Path, tmp_path: Path) -> None:
    labels = load_labels(workspace / "trials" / "labels" / "jm_k2_lam50.csv")
    assert labels.index.name == "date" and labels.dtype.kind == "i"
    digest = save_labels(labels, tmp_path / "copy.csv")
    pd.testing.assert_series_equal(load_labels(tmp_path / "copy.csv"), labels)
    assert digest == hashlib.sha256((tmp_path / "copy.csv").read_bytes()).hexdigest()


def test_every_result_records_the_hash_of_its_labels(finished: Path, log: TrialLog) -> None:
    for trial_id, result in log.results().items():
        data = Path(result["labels_path"]).read_bytes()
        assert result["labels_sha256"] == hashlib.sha256(data).hexdigest(), trial_id


@pytest.mark.parametrize("which", ["final model", "inertia"])
def test_report_refuses_labels_that_are_not_the_logged_ones(
    which: str,
    finished: Path,
    workspace: Path,
    tmp_path: Path,
    snapshot_data: pd.DataFrame,
    data_hash: str,
    log: TrialLog,
    config: CoreConfig,
) -> None:
    """The criteria are recomputed from the label files, so a changed file must be refused."""
    trials = tmp_path / "trials"
    shutil.copytree(workspace / "trials", trials)
    logged = log.last_report()
    assert logged is not None
    trial_id = logged["final_trial_id"] if which == "final model" else INERTIA_TRIAL
    path = trials / "labels" / f"{trial_id}.csv"
    save_labels(1 - load_labels(path).clip(upper=1), path)
    data = prepare(snapshot_data, config, registered_columns(log))
    with pytest.raises(TrialLogError, match=f"labels of {trial_id}"):
        build_report(data, log, trials, data_hash, COMMIT)


def test_a_crash_leaves_the_trial_pending(
    tmp_path: Path,
    snapshot_data: pd.DataFrame,
    config: CoreConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No result is written for a trial that did not finish, so the same command retries it."""
    fresh = TrialLog(tmp_path / "trials.jsonl")
    columns = register_trials(config, snapshot_data, fresh, "hash", "commit")
    data = prepare(snapshot_data, config, columns)

    def out_of_memory(*args: object, **kwargs: object) -> None:
        raise MemoryError("simulated")

    monkeypatch.setattr("termo.core.evaluate_config", out_of_memory)
    with pytest.raises(MemoryError):
        run_trials(data, fresh, tmp_path, "hash", "commit", echo=lambda message: None)
    assert fresh.results() == {}


def test_work_is_bound_to_the_library_versions(
    finished: Path,
    log: TrialLog,
    config: CoreConfig,
    data_hash: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered = log.registrations()[SETUP_TRIAL]["config"]["environment"]
    assert registered == environment_fingerprint()
    assert set(registered) == {
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "jumpmodels",
        "xgboost",
        "shap",
        "python",
    }
    other = {**registered, "numpy": "0.0.0"}
    monkeypatch.setattr("termo.core.environment_fingerprint", lambda: other)
    with pytest.raises(TrialLogError, match="environment"):
        verify_binding(log, config, data_hash, COMMIT)


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
    # The older measure is reported next to the deciding one, for every model and baseline.
    assert set(details["baselines_separation_shift"]) == set(details["baselines_separation"])
    assert all({"separation", "separation_shift"} <= set(row) for row in details["configurations"])
    final = details["final_model"]
    assert final["s1_min"] <= final["s1_mean"] <= 1.0
    assert set(final) == {
        "s1_mean",
        "s1_min",
        "s2",
        "separation_long",
        "median_duration_days",
        "days_by_decade",
    }
    assert set(final["days_by_decade"]["0"]) == {"1990"}
    assert (finished / "go_no_go.md").read_text(encoding="utf-8").isascii()


def test_every_report_is_logged(
    finished: Path,
    workspace: Path,
    tmp_path: Path,
    snapshot_data: pd.DataFrame,
    data_hash: str,
    log: TrialLog,
    config: CoreConfig,
) -> None:
    on_disk = json.loads((finished / "go_no_go.json").read_text(encoding="utf-8"))
    logged = log.last_report()
    assert logged is not None
    assert {key: logged[key] for key in on_disk} == on_disk
    assert logged["snapshot_hash"] == data_hash and logged["code_commit"] == COMMIT

    copy = private_copy(log, tmp_path)
    data = prepare(snapshot_data, config, registered_columns(copy))
    again = build_report(data, copy, workspace / "trials", data_hash, COMMIT)
    publish_report(again, copy, tmp_path / "reports", data_hash, COMMIT)
    reports = [r for r in copy.records() if r["kind"] == "report"]
    assert len(reports) == 2  # a second report is added, never substituted
    assert reports[0]["verdict"] == reports[1]["verdict"]


def test_report_needs_every_result(
    tmp_path: Path, snapshot_data: pd.DataFrame, config: CoreConfig
) -> None:
    fresh = TrialLog(tmp_path / "trials.jsonl")
    columns = register_trials(config, snapshot_data, fresh, "hash", "commit")
    with pytest.raises(TrialLogError, match="without a result"):
        build_report(prepare(snapshot_data, config, columns), fresh, tmp_path, "hash", "commit")


def test_stages_need_a_registration_first(tmp_path: Path, config: CoreConfig) -> None:
    empty = TrialLog(tmp_path / "trials.jsonl")
    with pytest.raises(TrialLogError, match="register stage first"):
        registered_columns(empty)
    with pytest.raises(TrialLogError, match="register stage first"):
        verify_binding(empty, config, "hash", "commit")


def test_final_holdout_needs_a_logged_go_verdict(
    finished: Path, tmp_path: Path, snapshot_dir: Path, log: TrialLog, config: CoreConfig
) -> None:
    reports = tmp_path / "reports"
    # A GO on disk proves nothing: only the log counts.
    write_report(CoreReport(Verdict.GO, "jm_k2_lam10", criteria=()), reports)
    without_report = TrialLog(tmp_path / "empty.jsonl")
    with pytest.raises(TrialLogError, match="report stage first"):
        run_final_holdout(config, snapshot_dir, without_report, reports, COMMIT)

    copy = private_copy(log, tmp_path)
    copy.record_report(report_payload(CoreReport(Verdict.NO_GO, "jm_k2_lam10", criteria=())))
    with pytest.raises(TrialLogError, match="GO verdict"):
        run_final_holdout(config, snapshot_dir, copy, reports, COMMIT)
    assert not copy.holdout_opened()


def test_a_wrong_snapshot_does_not_burn_the_holdout(
    finished: Path, tmp_path: Path, snapshot_dir: Path, log: TrialLog, config: CoreConfig
) -> None:
    copy = private_copy(log, tmp_path)
    copy.record_report(go_report())
    with pytest.raises(SnapshotError):
        run_final_holdout(config, tmp_path / "no-such-snapshot", copy, tmp_path, COMMIT)
    with pytest.raises(TrialLogError, match="code commit"):
        run_final_holdout(config, snapshot_dir, copy, tmp_path, "another-commit")
    with pytest.raises(TrialLogError, match="configuration"):
        other = replace(config, horizon_short_days=10)
        run_final_holdout(other, snapshot_dir, copy, tmp_path, COMMIT)
    assert not copy.holdout_opened()


def test_final_holdout_runs_exactly_once_and_is_logged(
    finished: Path, tmp_path: Path, snapshot_dir: Path, log: TrialLog, config: CoreConfig
) -> None:
    reports = tmp_path / "reports"
    copy = private_copy(log, tmp_path)
    copy.record_report(go_report())

    result = run_final_holdout(config, snapshot_dir, copy, reports, COMMIT)

    kinds = [r["kind"] for r in copy.records()]
    assert kinds[-2:] == ["holdout_opened", "holdout_result"]
    logged = copy.records()[-1]
    assert logged["final_trial_id"] == "jm_k2_lam10"
    assert logged["passed"] == result.passed and logged["n_weeks"] == result.n_weeks > 0
    saved = json.loads((reports / HOLDOUT_JSON).read_text(encoding="utf-8"))
    assert saved["excess_model"] == logged["excess_model"]
    with pytest.raises(TrialLogError, match="already been opened"):
        run_final_holdout(config, snapshot_dir, copy, reports, COMMIT)


def test_the_whole_snapshot_is_checked_before_the_holdout_opens(
    finished: Path,
    tmp_path: Path,
    snapshot_dir: Path,
    log: TrialLog,
    config: CoreConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hole inside the holdout must surface before the one opening is spent."""
    copy = private_copy(log, tmp_path)
    copy.record_report(go_report())

    def hole(*args: object, **kwargs: object) -> None:
        raise DataValidationError("gap of 85 days ending on 1998-09-01")

    monkeypatch.setattr("termo.core.snapshot_days", hole)
    with pytest.raises(DataValidationError, match="gap of 85 days"):
        run_final_holdout(config, snapshot_dir, copy, tmp_path, COMMIT)
    assert not copy.holdout_opened()


def test_a_holdout_that_is_too_short_is_refused_before_opening(
    finished: Path,
    tmp_path: Path,
    snapshot_dir: Path,
    log: TrialLog,
    config: CoreConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copy = private_copy(log, tmp_path)
    copy.record_report(go_report())
    thirty_days = pd.bdate_range(config.holdout_start, periods=30)
    monkeypatch.setattr("termo.core.snapshot_days", lambda *args, **kwargs: thirty_days)
    with pytest.raises(TrialLogError, match="30 holdout days"):
        run_final_holdout(config, snapshot_dir, copy, tmp_path, COMMIT)
    assert not copy.holdout_opened()
