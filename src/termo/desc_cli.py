"""Command line for spec 3 (descriptive tool). Console output is ASCII only."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from termo.cli import code_identity
from termo.config import load_config
from termo.core import registered_columns, verify_binding
from termo.data.loader import load_curve
from termo.data.snapshot import snapshot_hash
from termo.dataset import prepare
from termo.descriptive.stages import (
    HOLDOUT_DIR,
    PRE_HOLDOUT_DIR,
    desc_trial_id,
    holdout_desc,
    load_analysis,
    register_desc,
    report_desc,
    run_desc,
)
from termo.reading import build_reading, write_reading
from termo.validation.trials import RecordKind, TrialLog, TrialLogError

TRIALS_DIR = Path("trials") / "desc"
REPORTS_DIR = Path("reports") / "desc"
TRIALS_FILE = "trials.jsonl"
APPROVAL_FLAG = "--i-approve-opening-the-holdout"


def _holdout_result(log: TrialLog) -> dict[str, object] | None:
    found = [r for r in log.records() if r["kind"] == RecordKind.HOLDOUT_RESULT.value]
    return found[-1] if found else None


def _read(args: argparse.Namespace, log: TrialLog, data_hash: str, commit: str) -> Path:
    config = load_config(args.config)
    verify_binding(log, config, data_hash, commit)
    day = date.fromisoformat(args.date)
    report = log.last_report()
    trial = desc_trial_id(config)
    if report is None or trial not in log.results():
        raise TrialLogError("no validated reading: run the run and report stages first")
    holdout = _holdout_result(log)
    status = {
        "diagnostic": str(report["verdict"]),
        "holdout": None if holdout is None else str(holdout["verdict"]),
    }
    if pd.Timestamp(day) < pd.Timestamp(config.holdout_start):
        files = log.results()[trial]["metrics"]["files"]
        analysis = load_analysis(TRIALS_DIR / PRE_HOLDOUT_DIR, files)
    elif holdout is not None:
        analysis = load_analysis(TRIALS_DIR / HOLDOUT_DIR, holdout["files"])  # type: ignore[arg-type]
    else:
        raise TrialLogError("no validated reading for that date: the holdout has no result")
    try:
        reading = build_reading(analysis, day, config, status)
    except KeyError as error:
        raise TrialLogError(f"no validated reading for that date: {error.args[0]}") from error
    out = REPORTS_DIR / "readings"
    write_reading(reading, out)
    return out / f"{reading['date']}.md"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="termo-desc", description="TERMO: descriptive tool")
    parser.add_argument("stage", choices=["register", "run", "report", "holdout", "read"])
    parser.add_argument("--config", type=Path, default=Path("configs/desc.yaml"))
    parser.add_argument("--snapshot", type=Path, required=True, help="snapshot directory")
    parser.add_argument("--date", help="reading date, YYYY-MM-DD (stage read)")
    parser.add_argument(
        APPROVAL_FLAG,
        dest="approved",
        action="store_true",
        help="required by the holdout stage: it runs once and cannot be repeated",
    )
    args = parser.parse_args(argv)
    if args.stage == "read" and args.date is None:
        parser.error("--date is required for the read stage")
    if args.stage == "holdout" and not args.approved:
        parser.error(f"the holdout runs once and cannot be repeated: pass {APPROVAL_FLAG}")

    config = load_config(args.config)
    log = TrialLog(TRIALS_DIR / TRIALS_FILE)
    data_hash, commit = snapshot_hash(args.snapshot), code_identity()

    if args.stage == "read":
        path = _read(args, log, data_hash, commit)
        print(f"[ok] reading -> {path.as_posix()}")
        return 0
    if args.stage == "holdout":
        result = holdout_desc(config, args.snapshot, log, TRIALS_DIR, REPORTS_DIR, commit)
        print(f"[ok] holdout verdict: {result} -> {REPORTS_DIR.as_posix()}/holdout.md")
        return 0

    curve = load_curve(args.snapshot, config.series, config.start, config.holdout_start)
    if args.stage == "register":
        columns = register_desc(config, curve, log, data_hash, commit)
        print(f"[ok] registered {desc_trial_id(config)} with {len(columns)} features")
        return 0
    verify_binding(log, config, data_hash, commit)  # before any computation
    data = prepare(curve, config, registered_columns(log))
    if args.stage == "run":
        run_desc(data, log, TRIALS_DIR, data_hash, commit)
        print("[ok] the registered trial has a result")
    else:
        result = report_desc(data, log, TRIALS_DIR, REPORTS_DIR, data_hash, commit)
        print(f"[ok] diagnostic verdict: {result} -> {REPORTS_DIR.as_posix()}/diagnostic.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
