"""The weekly shadow run (spec 4, section 3).

From a fresh snapshot: recompute every daily output, prove that the registered history
comes out byte for byte, read the last Friday, decide the alert, append to the shadow log,
save the files of the shadow days and write the Spanish sheet.

Nothing here predicts. A history that does not reproduce is declared in the record and on
the sheet, never repaired: the expected cause is a revision of FRED data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from termo.config import CoreConfig
from termo.core import (
    SETUP_TRIAL,
    environment_fingerprint,
    registered_columns,
    verify_configuration,
)
from termo.data.loader import load_curve
from termo.data.snapshot import snapshot_downloaded_at, snapshot_hash
from termo.dataset import ExperimentData, frozen_cutoff, prepare, recipe_for
from termo.descriptive.stages import (
    FILES,
    HOLDOUT_DIR,
    PRE_HOLDOUT_DIR,
    Analysis,
    analyse,
    analysis_bytes,
    ensure_writable,
    frozen_map,
    hashes_of,
    holdout_result,
    load_analysis,
    restrict,
    validation_from_log,
    write_files,
)
from termo.operation.config import OperationConfig
from termo.operation.macro import macro_frame, macro_panel
from termo.operation.sheet_es import write_sheet
from termo.reading import build_reading
from termo.validation.trials import TrialLog, TrialLogError
from termo.validation.walkforward import build_refits

SHADOW_DIR = "sombra"  # trials/<registry>/sombra/sombra.jsonl and trials/<registry>/sombra/<date>/
SHADOW_LOG_FILE = "sombra.jsonl"
READING = "reading"
FRIDAY = 4
ONE_DAY = pd.Timedelta(days=1)
ONE_WEEK = pd.Timedelta(days=7)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ShadowLog:
    """Append-only JSONL, one record per run; a repeated run is a new record, never a rewrite."""

    path: Path
    clock: Callable[[], str] = _utc_now

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def append(self, record: Mapping[str, Any]) -> None:
        """Every record says what kind it is and gets the time it was appended."""
        if "kind" not in record:
            raise TrialLogError("a shadow record needs a kind")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # newline fixed so the file is byte-identical on every platform
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({**record, "at": self.clock()}, sort_keys=True) + "\n")

    def readings(self) -> list[dict[str, Any]]:
        return [r for r in self.records() if r["kind"] == READING]

    def last_reading(self) -> dict[str, Any] | None:
        found = self.readings()
        return found[-1] if found else None

    def last_reading_before(self, stamp: str) -> dict[str, Any] | None:
        """The reading of the latest date strictly earlier than `stamp` (ISO dates compare).

        Latest by reading date, not by position in the log: an old week run again after a
        newer one is appended last but is not the previous week. Among the runs of that
        date, the last one appended.
        """
        found = [r for r in self.readings() if str(r["reading_date"]) < stamp]
        if not found:
            return None
        return sorted(found, key=lambda r: str(r["reading_date"]))[-1]  # stable: ties by position


def reading_date(index: pd.DatetimeIndex) -> pd.Timestamp:
    """The last Friday with data on or before the last day of `index`.

    When that Friday has no data (a holiday) the last day of its Monday-Friday week does;
    a week without any data steps back to the previous Friday.
    """
    last = index[-1]
    friday = last - pd.Timedelta(days=(last.weekday() - FRIDAY) % 7)
    while friday >= index[0]:
        monday = friday - pd.Timedelta(days=FRIDAY)
        week = index[(index >= monday) & (index <= friday)]
        if len(week):
            return pd.Timestamp(week[-1])
        friday -= ONE_WEEK
    raise ValueError(f"no Friday with data on or before {last.date().isoformat()}")


def slice_between(
    analysis: Analysis, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
) -> Analysis:
    """The same outputs between `start` and `end`, both inclusive, either optional."""
    return Analysis(
        labels=analysis.labels.loc[start:end],
        frozen_labels=analysis.frozen_labels.loc[start:end],
        proba=analysis.proba.loc[start:end],
        shap_blocks=analysis.shap_blocks.loc[start:end],
        shap_top=analysis.shap_top.loc[start:end],
    )


def frozen_as_registered(analysis: Analysis, data: ExperimentData) -> Analysis:
    """The frozen map the way the registry holds it.

    Before the holdout it is the registered cutoff; from the holdout on, the holdout stage
    fitted it once more through the holdout's eve (D5 wants a model that saw everything
    before the holdout). Reproducing the registered files byte for byte takes both; the
    online outputs depend on neither cutoff.

    Only that one refit is built here (the pipeline at the eve, as `prepare` would build
    it with `frozen_train_end` moved there): preparing the whole curve again would refit
    every cutoff a second time for one map.
    """
    config = data.config
    start = pd.Timestamp(config.holdout_start)
    feature_dates = data.curve.index[config.burn_in_days :]
    eve = frozen_cutoff(feature_dates, pd.Timestamp(config.holdout_start - timedelta(days=1)))
    refits = build_refits(
        data.curve, [eve], data.columns, config.burn_in_days, recipe=recipe_for(config)
    )
    through_eve = frozen_map(replace(data, frozen_refits=tuple(refits)))
    before = analysis.frozen_labels.loc[analysis.frozen_labels.index < start]
    return replace(analysis, frozen_labels=pd.concat([before, through_eve.loc[start:]]))


@dataclass(frozen=True)
class HistoryCheck:
    reproduced: bool
    differing: dict[str, list[str]]  # file -> ISO dates whose rows differ

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rows(data: bytes) -> dict[bytes, bytes]:
    """The CSV lines keyed by their date (the first column), header excluded."""
    lines = data.split(b"\n")[1:]
    return {line.split(b",", 1)[0]: line for line in lines if line}


def _differing_dates(recomputed: bytes, registered: bytes) -> list[str]:
    """Dates whose line differs, or exists on one side only (byte comparison, like the hash)."""
    ours, theirs = _rows(recomputed), _rows(registered)
    dates = ours.keys() | theirs.keys()
    return sorted(key.decode() for key in dates if ours.get(key) != theirs.get(key))


def _registered_bytes(directory: Path, name: str, expected: str) -> bytes:
    data = (directory / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise TrialLogError(f"{(directory / name).as_posix()}: the file is not the one recorded")
    return data


def _compare(
    contents: Mapping[str, bytes], logged: Mapping[str, str], directory: Path
) -> dict[str, list[str]]:
    found = hashes_of(contents)
    differing: dict[str, list[str]] = {}
    for name in FILES:
        if found[name] != str(logged[name]):
            registered = _registered_bytes(directory, name, str(logged[name]))
            differing[name] = _differing_dates(contents[name], registered)
    return differing


def _registered_files(log: TrialLog) -> tuple[dict[str, str], dict[str, str]]:
    """The logged hashes of the pre-holdout files and of the holdout files."""
    holdout = holdout_result(log)
    if holdout is None:
        raise TrialLogError("no holdout result in the log: the shadow operates a finished registry")
    trial = str(holdout["final_trial_id"])
    return dict(log.results()[trial]["metrics"]["files"]), dict(holdout["files"])


def history_check(
    analysis: Analysis,
    log: TrialLog,
    trials_dir: Path,
    pre_end: pd.Timestamp,
    hold_end: pd.Timestamp,
) -> HistoryCheck:
    """Does the recomputation give the registered files, byte for byte?

    Dates up to `pre_end` against the logged pre-holdout hashes, dates after it up to
    `hold_end` against the logged holdout hashes. On a mismatch the registered files are
    read (refused if they are not the logged ones) and the differing dates listed per file.
    """
    pre_hashes, hold_hashes = _registered_files(log)
    differing = _compare(
        analysis_bytes(slice_between(analysis, end=pre_end)),
        pre_hashes,
        trials_dir / PRE_HOLDOUT_DIR,
    )
    later = _compare(
        analysis_bytes(slice_between(analysis, pre_end + ONE_DAY, hold_end)),
        hold_hashes,
        trials_dir / HOLDOUT_DIR,
    )
    for name, dates in later.items():
        differing[name] = sorted(differing.get(name, []) + dates)
    return HistoryCheck(reproduced=not differing, differing=differing)


@dataclass(frozen=True)
class Alert:
    fecha: str
    de: str
    a: str
    confianza: float


def alert_for(
    previous_phase: int | None,
    reading: Mapping[str, Any],
    names: Sequence[str],
    min_confidence: float,
) -> Alert | None:
    """A phase other than the previous reading's, read with enough confidence."""
    phase, confidence = int(reading["phase"]), float(reading["confidence"])
    if previous_phase is None or phase == previous_phase or confidence < min_confidence:
        return None
    return Alert(str(reading["date"]), names[previous_phase], names[phase], confidence)


def previous_phase(shadow_log: ShadowLog, stamp: str, holdout_last: int) -> int:
    """The phase the reading of `stamp` is compared with for the alert.

    The last shadow reading of an earlier date; before any, the last registered holdout
    day. A run repeated for the same date is never compared with itself: it gives the
    same record, alert included.
    """
    last = shadow_log.last_reading_before(stamp)
    return holdout_last if last is None else int(last["reading"]["phase"])


@dataclass(frozen=True, eq=False)
class ShadowRun:
    record: dict[str, Any]
    curve: pd.DataFrame  # the whole curve of the snapshot, shadow days included
    analysis: Analysis  # the whole recomputed history, shadow days included
    sheet_path: Path


def _registered_recipe(model_log: TrialLog, config: CoreConfig) -> tuple[str, ...]:
    """The registered columns, after refusing any other configuration or column set."""
    verify_configuration(model_log, config)
    columns = registered_columns(model_log)
    if tuple(recipe_for(config).names) != columns:
        raise TrialLogError("the registered columns are not the ones this configuration builds")
    return columns


def _recompute(
    config: CoreConfig, columns: tuple[str, ...], snapshot_dir: Path
) -> tuple[pd.DataFrame, Analysis]:
    """Every daily output on the whole curve of `snapshot_dir`, the frozen map as registered."""
    curve = load_curve(
        snapshot_dir, config.series, config.start, config.holdout_start, final_evaluation=True
    )
    data = prepare(curve, config, columns)
    return curve, frozen_as_registered(analyse(data), data)


def latest_analysis(
    config: CoreConfig, model_log: TrialLog, snapshot_dir: Path
) -> tuple[pd.DataFrame, Analysis]:
    """The curve and the recomputed history of a snapshot, as a shadow run computes them.

    The stages that need the latest outputs without logging a reading (the monthly report,
    the shadow evaluation, the exports) recompute them this way: minutes on real data,
    and the same code path as the shadow run, so there is one set of outputs per snapshot.
    """
    return _recompute(config, _registered_recipe(model_log, config), snapshot_dir)


def _registered_labels(log: TrialLog, trials_dir: Path) -> tuple[pd.Series, pd.Series]:
    """The registered pre-holdout and holdout labels, refused if not the logged files."""
    pre_hashes, hold_hashes = _registered_files(log)
    return (
        load_analysis(trials_dir / PRE_HOLDOUT_DIR, pre_hashes).labels,
        load_analysis(trials_dir / HOLDOUT_DIR, hold_hashes).labels,
    )


def _all_dates(differing: Mapping[str, Sequence[str]]) -> set[str]:
    return {d for dates in differing.values() for d in dates}


def run_shadow(
    op: OperationConfig,
    config: CoreConfig,
    model_log: TrialLog,
    snapshot_dir: Path,
    trials_dir: Path,
    reports_dir: Path,
    code_commit: str,
    echo: Callable[[str], None] = print,
) -> ShadowRun:
    """One shadow run. Everything is computed before anything is appended or written.

    The configuration and the registered columns are checked first; the snapshot is new
    by design. The reading is built on the whole recomputed history, so an episode that
    began before the shadow days (even before the holdout) keeps its start and length.
    """
    desc = config.descriptive
    if desc is None:
        raise TrialLogError("this configuration has no descriptive section")
    columns = _registered_recipe(model_log, config)
    validation = validation_from_log(model_log, config)
    shadow_log = ShadowLog(trials_dir / SHADOW_DIR / SHADOW_LOG_FILE)
    run_at = shadow_log.clock()
    pre_labels, hold_labels = _registered_labels(model_log, trials_dir)
    pre_end, hold_end = pre_labels.index[-1], hold_labels.index[-1]

    curve, analysis = _recompute(config, columns, snapshot_dir)
    history = history_check(analysis, model_log, trials_dir, pre_end, hold_end)

    day = reading_date(analysis.labels.index)
    stamp = day.date().isoformat()
    # a run repeated for the same reading date is logged again but is the same week
    weeks = len({str(r["reading_date"]) for r in shadow_log.readings()} | {stamp})
    validation["sombra"] = {
        "semanas": weeks,
        "historia_reproducida": history.reproduced,
        "proxima_evaluacion_semanas": max(0, op.shadow_eval_min_weeks - weeks),
    }
    environment = environment_fingerprint()
    generated_with = {
        "code_commit": code_commit,
        "registered_code_commit": str(model_log.registrations()[SETUP_TRIAL]["code_commit"]),
        "environment": environment,
    }
    reading = build_reading(analysis, day.date(), config, validation, generated_with)
    previous = previous_phase(shadow_log, stamp, int(hold_labels.iloc[-1]))
    alert = alert_for(previous, reading, desc.phase_names, op.alert_min_confidence)
    panel = macro_panel(macro_frame(snapshot_dir, config, op), op, day.date())
    contents = analysis_bytes(restrict(analysis, hold_end + ONE_DAY))  # the shadow days only
    record: dict[str, Any] = {
        "kind": READING,
        "run_at": run_at,
        "snapshot_hash": snapshot_hash(snapshot_dir),
        "snapshot_downloaded_at": snapshot_downloaded_at(snapshot_dir),
        "reading_date": stamp,
        "reading": reading,
        "files": hashes_of(contents),
        "history": history.as_dict(),
        "alert": None if alert is None else asdict(alert),
        "macro": panel,
        "code_commit": code_commit,
        "environment": environment,
    }

    files_dir = trials_dir / SHADOW_DIR / stamp
    sheets_dir = reports_dir / SHADOW_DIR
    ensure_writable(
        [shadow_log.path]
        + [files_dir / name for name in FILES]
        + [sheets_dir / f"{stamp}.md", sheets_dir / f"{stamp}.json"]
    )
    shadow_log.append(record)  # the record is in the log before any file is written
    if history.reproduced:
        echo("[ok] historia reproducida")
    else:
        echo(f"[x] HISTORIA REVISADA: {len(_all_dates(history.differing))} dias")
    echo(f"[ok] lectura {stamp} fase={reading['phase_name']} conf={reading['confidence']:.2f}")
    if alert is not None:
        echo(f"[!] ALERTA: {alert.de} -> {alert.a}")
    write_files(contents, files_dir)
    sheet_path = write_sheet(record, op, sheets_dir)
    echo(f"[ok] hoja -> {sheet_path.as_posix()}")
    return ShadowRun(record=record, curve=curve, analysis=analysis, sheet_path=sheet_path)
