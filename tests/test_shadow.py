"""The weekly shadow run: reproduce the registered history, read the last Friday, alert, log."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
import pytest

from conftest import SHADOW_DAYS, Desc2Registry, fake_fred_frames, make_macro
from termo.config import CoreConfig
from termo.core import SETUP_TRIAL
from termo.data.snapshot import snapshot_hash
from termo.descriptive.criteria import APTO, NO_APTO
from termo.descriptive.stages import (
    FILES,
    HOLDOUT_DIR,
    PRE_HOLDOUT_DIR,
    analysis_bytes,
    hashes_of,
    holdout_result,
    load_analysis,
)
from termo.operation.config import OperationConfig, load_operation_config
from termo.operation.shadow import (
    SHADOW_DIR,
    SHADOW_LOG_FILE,
    Alert,
    ShadowLog,
    ShadowRun,
    alert_for,
    history_check,
    reading_date,
    run_shadow,
    slice_between,
)
from termo.operation.snapshot import take_operation_snapshot
from termo.validation.trials import TrialLog, TrialLogError

REPO_OP = Path(__file__).resolve().parents[1] / "configs" / "operacion.yaml"
COMMIT = "test-commit-4"
NAMES = ("rally fuerte", "rally moderado", "venta")
REVISION = 0.05  # one yield moved 5 bp on one past day: a FRED revision
REVISED_DAYS_BEFORE_HOLDOUT = 300


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registered(registry: Desc2Registry) -> tuple[pd.Series, pd.Series]:
    """The registered pre-holdout and holdout labels, hash-verified."""
    log = registry.log
    holdout = holdout_result(log)
    assert holdout is not None
    pre = log.results()[str(holdout["final_trial_id"])]["metrics"]["files"]
    return (
        load_analysis(registry.trials_dir / PRE_HOLDOUT_DIR, pre).labels,
        load_analysis(registry.trials_dir / HOLDOUT_DIR, holdout["files"]).labels,
    )


def _shadow_log(registry: Desc2Registry) -> ShadowLog:
    return ShadowLog(registry.trials_dir / SHADOW_DIR / SHADOW_LOG_FILE)


def _take_op_snapshot(
    target: Path, curve: pd.DataFrame, config: CoreConfig, op: OperationConfig
) -> Path:
    get = fake_fred_frames([curve, make_macro(curve)])
    take_operation_snapshot(config, op, target, get, "1999-03-15T00:00:00+00:00")
    return target


@pytest.fixture(scope="module")
def op() -> OperationConfig:
    return load_operation_config(REPO_OP)


@pytest.fixture(scope="module")
def registry(
    desc2_registry: Desc2Registry, tmp_path_factory: pytest.TempPathFactory
) -> Desc2Registry:
    """A private copy of the finished registry: the shadow log written here stays here."""
    return desc2_registry.copy_to(tmp_path_factory.mktemp("termo-shadow"))


@pytest.fixture(scope="module")
def op_snapshot(
    tmp_path_factory: pytest.TempPathFactory,
    curve: pd.DataFrame,
    desc2_config: CoreConfig,
    op: OperationConfig,
) -> Path:
    """The operation snapshot: the FULL curve plus the macro series, 60 days past the registry."""
    return _take_op_snapshot(
        tmp_path_factory.mktemp("termo-shadow-snapshot") / "op", curve, desc2_config, op
    )


@pytest.fixture(scope="module")
def first(
    registry: Desc2Registry, op: OperationConfig, op_snapshot: Path
) -> tuple[ShadowRun, list[str]]:
    messages: list[str] = []
    run = run_shadow(
        op,
        registry.config,
        registry.log,
        op_snapshot,
        registry.trials_dir,
        registry.reports_dir,
        COMMIT,
        echo=messages.append,
    )
    return run, messages


@pytest.fixture(scope="module")
def second(
    first: tuple[ShadowRun, list[str]],
    registry: Desc2Registry,
    op: OperationConfig,
    op_snapshot: Path,
) -> ShadowRun:
    """The same snapshot run again: a repeated run is logged as such, never overwritten."""
    return run_shadow(
        op,
        registry.config,
        registry.log,
        op_snapshot,
        registry.trials_dir,
        registry.reports_dir,
        COMMIT,
        echo=lambda _: None,
    )


@dataclass(frozen=True)
class Revised:
    """A shadow run on a snapshot where one yield of the registered period was revised."""

    run: ShadowRun
    day: pd.Timestamp  # the revised day
    messages: list[str]
    registry: Desc2Registry  # its own private copy, with its own shadow log


@pytest.fixture(scope="module")
def revised(
    desc2_registry: Desc2Registry,
    tmp_path_factory: pytest.TempPathFactory,
    curve: pd.DataFrame,
    op: OperationConfig,
) -> Revised:
    root = tmp_path_factory.mktemp("termo-shadow-revised")
    private = desc2_registry.copy_to(root)
    pre, _ = _registered(private)
    day = pre.index[-REVISED_DAYS_BEFORE_HOLDOUT]
    altered = curve.copy()
    altered.loc[day, "DGS5"] += REVISION
    snapshot = _take_op_snapshot(root / "op", altered, private.config, op)
    messages: list[str] = []
    run = run_shadow(
        op,
        private.config,
        private.log,
        snapshot,
        private.trials_dir,
        private.reports_dir,
        COMMIT,
        echo=messages.append,
    )
    return Revised(run, day, messages, private)


def test_reading_date_is_the_last_friday_with_data() -> None:
    days = pd.bdate_range("2026-09-21", "2026-10-07")  # Monday 21 Sep .. Wednesday 7 Oct
    assert reading_date(days) == pd.Timestamp("2026-10-02")  # the Friday before a Wednesday
    assert reading_date(days).day_name() == "Friday"
    to_friday = pd.bdate_range("2026-09-21", "2026-10-02")
    assert reading_date(to_friday) == pd.Timestamp("2026-10-02")  # ends on Friday: that Friday
    holiday = days.drop(pd.Timestamp("2026-10-02"))
    assert reading_date(holiday) == pd.Timestamp("2026-10-01")  # Thursday of that week
    closed_week = days.drop(pd.bdate_range("2026-09-28", "2026-10-02"))
    assert reading_date(closed_week) == pd.Timestamp("2026-09-25")  # the previous week's Friday
    with pytest.raises(ValueError):
        reading_date(pd.DatetimeIndex([pd.Timestamp("2026-10-07")]))  # a Wednesday alone


def test_alert_rule_by_hand() -> None:
    reading = {"phase": 1, "confidence": 0.7, "phase_name": "rally moderado", "date": "2026-10-09"}
    assert alert_for(2, reading, NAMES, 0.6) == Alert("2026-10-09", "venta", "rally moderado", 0.7)
    assert alert_for(1, reading, NAMES, 0.6) is None  # same phase
    assert alert_for(2, {**reading, "confidence": 0.5}, NAMES, 0.6) is None  # not confident
    assert alert_for(2, {**reading, "confidence": 0.6}, NAMES, 0.6) is not None  # threshold
    assert alert_for(None, reading, NAMES, 0.6) is None  # nothing to compare with


def test_shadow_log_appends_dated_records_and_finds_the_last_reading(tmp_path: Path) -> None:
    log = ShadowLog(tmp_path / "sombra.jsonl", clock=lambda: "2026-10-09T21:30:00+00:00")
    assert log.records() == [] and log.last_reading() is None
    log.append({"kind": "reading", "reading_date": "2026-10-02", "reading": {"phase": 2}})
    log.append({"kind": "evaluation", "verdict": "apto"})
    log.append({"kind": "reading", "reading_date": "2026-10-09", "reading": {"phase": 1}})
    records = log.records()
    assert [r["kind"] for r in records] == ["reading", "evaluation", "reading"]
    assert all(r["at"] == "2026-10-09T21:30:00+00:00" for r in records)
    last = log.last_reading()
    assert last is not None and last["reading_date"] == "2026-10-09"
    assert [r["reading_date"] for r in log.readings()] == ["2026-10-02", "2026-10-09"]
    with pytest.raises(TrialLogError, match="kind"):
        log.append({"reading_date": "2026-10-16"})
    assert len(log.records()) == 3
    raw = log.path.read_bytes()
    assert b"\r\n" not in raw and raw.count(b"\n") == 3


def test_history_check_passes_when_the_past_is_reproduced_and_lists_days_when_not(
    first: tuple[ShadowRun, list[str]], revised: Revised, registry: Desc2Registry
) -> None:
    run, _ = first
    pre, hold = _registered(registry)
    pre_end, hold_end = pre.index[-1], hold.index[-1]
    assert pre_end < pd.Timestamp(registry.config.holdout_start) <= hold.index[0]
    check = history_check(run.analysis, registry.log, registry.trials_dir, pre_end, hold_end)
    assert check.reproduced is True and check.differing == {}
    assert run.record["history"] == {"reproduced": True, "differing": {}}
    # the slices are what the registry holds, byte for byte
    holdout = holdout_result(registry.log)
    assert holdout is not None
    assert (
        hashes_of(analysis_bytes(slice_between(run.analysis, end=pre_end)))
        == (registry.log.results()["desc_k3"]["metrics"]["files"])
    )
    assert hashes_of(analysis_bytes(slice_between(run.analysis, hold.index[0], hold_end))) == dict(
        holdout["files"]
    )
    sliced = slice_between(run.analysis, hold.index[0], hold_end)
    assert sliced.labels.index[0] == hold.index[0] and sliced.labels.index[-1] == hold_end
    assert sliced.proba.index.equals(sliced.labels.index)
    assert slice_between(run.analysis).labels.index.equals(run.analysis.labels.index)

    revised_run, day = revised.run, revised.day
    revised_check = history_check(
        revised_run.analysis, registry.log, registry.trials_dir, pre_end, hold_end
    )
    assert revised_check.reproduced is False and revised_check.differing
    assert set(revised_check.differing) <= set(FILES)
    dates = sorted({d for listed in revised_check.differing.values() for d in listed})
    assert dates == [d for d in dates if pd.Timestamp(d) in run.analysis.labels.index]
    # a revision cannot change the past before it (no look-ahead) and the check covers the
    # registered period only: the shadow days are not part of the history
    assert pd.Timestamp(dates[0]) >= day
    assert pd.Timestamp(dates[-1]) <= hold_end
    assert revised_run.record["history"] == {
        "reproduced": False,
        "differing": revised_check.differing,
    }


def test_history_check_refuses_registered_files_that_are_not_the_logged_ones(
    first: tuple[ShadowRun, list[str]], registry: Desc2Registry, tmp_path: Path
) -> None:
    run, _ = first
    pre, hold = _registered(registry)
    altered = replace(run.analysis, labels=run.analysis.labels.where(lambda s: s != 2, 0))
    path = registry.trials_dir / PRE_HOLDOUT_DIR / "labels.csv"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        with pytest.raises(TrialLogError, match="not the one recorded"):
            history_check(altered, registry.log, registry.trials_dir, pre.index[-1], hold.index[-1])
    finally:
        path.write_bytes(original)


def test_shadow_run_logs_reading_hashes_history_check_and_alert(
    first: tuple[ShadowRun, list[str]],
    registry: Desc2Registry,
    op: OperationConfig,
    op_snapshot: Path,
    curve: pd.DataFrame,
) -> None:
    run, messages = first
    log = _shadow_log(registry)
    records = log.records()
    assert len(records) == 1 and records[0]["kind"] == "reading"
    record = records[0]
    assert record == {**run.record, "at": record["at"]}
    assert set(record) == {
        "kind",
        "at",
        "run_at",
        "snapshot_hash",
        "reading_date",
        "reading",
        "files",
        "history",
        "alert",
        "macro",
        "code_commit",
        "environment",
    }
    assert record["snapshot_hash"] == snapshot_hash(op_snapshot)
    assert record["code_commit"] == COMMIT and "xgboost" in record["environment"]

    # the reading is the last Friday of the snapshot, built on the whole recomputed history
    labels = run.analysis.labels
    assert labels.index[-1] == curve.index[-1]
    day = reading_date(labels.index)
    assert record["reading_date"] == day.date().isoformat() == record["reading"]["date"]
    reading = record["reading"]
    assert reading["phase"] == int(labels.loc[day])
    assert reading["phase_name"] == NAMES[reading["phase"]]
    past = labels.loc[:day]
    changed = past.ne(past.iloc[-1]).to_numpy().nonzero()[0]
    episode = past.index[changed[-1] + 1] if len(changed) else past.index[0]
    assert reading["episode_start"] == episode.date().isoformat()
    assert reading["days_in_phase"] == len(past.loc[episode:])
    pre, hold = _registered(registry)
    # the spanned history: the episode may start before the shadow days, even in the holdout
    pd.testing.assert_series_equal(
        labels.loc[: hold.index[-1]],
        pd.concat([pre, hold]),
        check_names=False,
        check_freq=False,
        check_dtype=False,
    )
    assert reading["validation"]["sombra"] == {
        "semanas": 1,
        "historia_reproducida": True,
        "proxima_evaluacion_semanas": op.shadow_eval_min_weeks - 1,
    }
    assert reading["validation"]["holdout"] in {APTO, NO_APTO}
    assert reading["validation"]["holdout_seen_by"] == "desc_k3 (test)"
    assert reading["generated_with"] == {
        "code_commit": COMMIT,
        "registered_code_commit": registry.commit,
        "environment": record["environment"],
    }

    # the five files of the shadow days only, hashed in the record
    assert set(record["files"]) == set(FILES)
    directory = registry.trials_dir / SHADOW_DIR / record["reading_date"]
    for name, expected in record["files"].items():
        assert _sha256(directory / name) == expected
    saved = load_analysis(directory, record["files"])
    assert saved.labels.index[0] > hold.index[-1] and len(saved.labels) == SHADOW_DAYS
    assert saved.labels.index[-1] == curve.index[-1]
    pd.testing.assert_series_equal(
        saved.labels,
        labels.loc[saved.labels.index],
        check_names=False,
        check_freq=False,
        check_dtype=False,
    )

    # the first run compares with the last holdout day of the registered files
    previous = int(hold.iloc[-1])
    expected = alert_for(previous, reading, NAMES, op.alert_min_confidence)
    assert record["alert"] == (None if expected is None else asdict(expected))
    assert record["history"] == {"reproduced": True, "differing": {}}
    assert [row["serie"] for row in record["macro"]] == [m.name for m in op.macro][:1] + [
        "DGS2 - fed funds efectiva"
    ]
    assert all(row["fecha_valor"] <= record["reading_date"] for row in record["macro"])

    # the sheet, in Spanish, and the console, in ASCII
    assert run.sheet_path == registry.reports_dir / SHADOW_DIR / f"{record['reading_date']}.md"
    sheet = run.sheet_path.read_text(encoding="utf-8")
    assert f"hoja semanal del {record['reading_date']}" in sheet
    assert "HISTORIA REVISADA" not in sheet
    assert f"**{reading['phase_name']}** desde {reading['episode_start']}" in sheet
    payload = json.loads(run.sheet_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload == run.record
    assert all(m.isascii() for m in messages)
    assert messages[0] == "[ok] historia reproducida"
    assert messages[1] == (
        f"[ok] lectura {record['reading_date']} fase={reading['phase_name']} "
        f"conf={reading['confidence']:.2f}"
    )
    if expected is not None:
        assert messages[2] == f"[!] ALERTA: {expected.de} -> {expected.a}"
    assert messages[-1] == f"[ok] hoja -> {run.sheet_path.as_posix()}"
    assert len(messages) == 3 + (expected is not None)


def test_a_repeated_run_appends_an_identical_reading_without_overwriting(
    first: tuple[ShadowRun, list[str]], second: ShadowRun, registry: Desc2Registry
) -> None:
    run, _ = first
    records = _shadow_log(registry).records()
    assert len(records) == 2 and [r["kind"] for r in records] == ["reading", "reading"]
    assert records[0] == {**run.record, "at": records[0]["at"]}
    assert records[1] == {**second.record, "at": records[1]["at"]}
    assert second.record["reading"] == run.record["reading"]
    assert second.record["files"] == run.record["files"]
    assert second.record["reading_date"] == run.record["reading_date"]
    # the previous reading is now the first shadow one: the same phase, so no alert
    assert second.record["alert"] is None
    # the model trial log is read, never written
    assert all(r["kind"] != "reading" for r in registry.log.records())


def test_shadow_run_refuses_a_changed_model_configuration_or_columns(
    registry: Desc2Registry, op: OperationConfig, op_snapshot: Path, tmp_path: Path
) -> None:
    changed = replace(registry.config, refit_weeks=13)
    log_path = _shadow_log(registry).path
    before = log_path.read_bytes() if log_path.exists() else None
    trials, reports = tmp_path / "trials", tmp_path / "reports"
    with pytest.raises(TrialLogError, match="configuration"):
        run_shadow(op, changed, registry.log, op_snapshot, trials, reports, COMMIT)
    # a registry whose columns are not the ones this configuration builds
    lines = [
        json.loads(line) for line in registry.log.path.read_text(encoding="utf-8").splitlines()
    ]
    setup = next(r for r in lines if r["kind"] == "registered" and r["trial_id"] == SETUP_TRIAL)
    setup["config"]["columns"] = setup["config"]["columns"][:-1]
    forged = TrialLog(tmp_path / "forged.jsonl")
    forged.path.write_text(
        "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines), encoding="utf-8"
    )
    with pytest.raises(TrialLogError, match="columns"):
        run_shadow(op, registry.config, forged, op_snapshot, trials, reports, COMMIT)
    assert (log_path.read_bytes() if log_path.exists() else None) == before
    assert not trials.exists() and not reports.exists()


def test_shadow_run_marks_a_revised_history_but_still_produces_the_reading(
    revised: Revised,
) -> None:
    run, day, messages = revised.run, revised.day, revised.messages
    record = run.record
    assert record["history"]["reproduced"] is False
    differing = record["history"]["differing"]
    dates = sorted({d for listed in differing.values() for d in listed})
    assert dates and pd.Timestamp(dates[0]) >= day
    assert record["reading"]["validation"]["sombra"]["historia_reproducida"] is False
    assert record["reading"]["phase"] in {0, 1, 2} and record["reading_date"]
    sheet = run.sheet_path.read_text(encoding="utf-8")
    assert f"**HISTORIA REVISADA**: {len(dates)} días difieren" in sheet
    assert messages[0] == f"[x] HISTORIA REVISADA: {len(dates)} dias"
    assert all(m.isascii() for m in messages)
    log = _shadow_log(revised.registry)
    assert len(log.records()) == 1 and log.records()[0]["history"]["reproduced"] is False
    # the registered files are untouched and still the logged ones: declared, not repaired
    _, hold = _registered(revised.registry)
    assert hold.index[-1] < pd.Timestamp(record["reading_date"])
    directory = revised.registry.trials_dir / SHADOW_DIR / record["reading_date"]
    assert load_analysis(directory, record["files"]).labels.index[0] > hold.index[-1]
