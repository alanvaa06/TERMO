"""Command line for spec 3 (descriptive tool). Console output is ASCII only."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from termo.cli import code_identity
from termo.config import load_config
from termo.core import (
    SETUP_TRIAL,
    environment_fingerprint,
    registered_columns,
    verify_binding,
    verify_data_binding,
)
from termo.data.loader import load_curve
from termo.data.snapshot import snapshot_hash
from termo.dataset import prepare
from termo.descriptive.stages import (
    HOLDOUT_DIR,
    PRE_HOLDOUT_DIR,
    desc_trial_id,
    diagnostic_report,
    failed_checks,
    holdout_desc,
    holdout_result,
    load_analysis,
    register_desc,
    report_desc,
    run_desc,
)
from termo.reading import (
    FIDELITY_CHECK,
    LOST_HOLDOUT,
    build_reading,
    span_periods,
    write_reading,
)
from termo.validation.trials import TrialLog, TrialLogError

TRIALS_DIR = Path("trials") / "desc"
REPORTS_DIR = Path("reports") / "desc"
TRIALS_FILE = "trials.jsonl"
APPROVAL_FLAG = "--i-approve-opening-the-holdout"


def iso_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not a date in YYYY-MM-DD form") from error


def _read(args: argparse.Namespace, log: TrialLog, data_hash: str, commit: str) -> Path:
    config = load_config(args.config)
    # the reading binds the data, not the code: later code must still read this log
    verify_data_binding(log, config, data_hash)
    day: date = args.date
    report = diagnostic_report(log)
    trial = desc_trial_id(config)
    if report is None or trial not in log.results():
        raise TrialLogError("no reading: run the run and report stages first")
    holdout = holdout_result(log)
    lost = holdout is None and log.holdout_opened()
    holdout_status: str | None = None if holdout is None else str(holdout["verdict"])
    if lost:
        holdout_status = LOST_HOLDOUT
    governing = report if holdout is None else holdout
    failed = failed_checks(governing)
    validation = {
        "diagnostic": str(report["verdict"]),
        "holdout": holdout_status,
        "failed_checks": failed,
        "fidelity_failed": FIDELITY_CHECK in failed,
    }
    in_holdout = pd.Timestamp(day) >= pd.Timestamp(config.holdout_start)
    if in_holdout and holdout is None:
        raise TrialLogError(
            "no reading for that date: the holdout " + (LOST_HOLDOUT if lost else "has no result")
        )
    files = log.results()[trial]["metrics"]["files"]
    analysis = load_analysis(TRIALS_DIR / PRE_HOLDOUT_DIR, files)
    if in_holdout and holdout is not None:
        # the holdout outputs, with the labels before them so an episode keeps its start
        analysis = span_periods(analysis, load_analysis(TRIALS_DIR / HOLDOUT_DIR, holdout["files"]))
    generated_with = {
        "code_commit": commit,
        "registered_code_commit": str(log.registrations()[SETUP_TRIAL]["code_commit"]),
        "environment": environment_fingerprint(),
    }
    try:
        reading = build_reading(analysis, day, config, validation, generated_with)
    except KeyError as error:
        raise TrialLogError(f"no reading for that date: {error.args[0]}") from error
    out = REPORTS_DIR / "readings"
    write_reading(reading, out)
    return out / f"{reading['date']}.md"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="termo-desc", description="TERMO: descriptive tool", allow_abbrev=False
    )
    parser.add_argument("stage", choices=["register", "run", "report", "holdout", "read"])
    parser.add_argument("--config", type=Path, default=Path("configs/desc.yaml"))
    parser.add_argument("--snapshot", type=Path, required=True, help="snapshot directory")
    parser.add_argument("--date", type=iso_date, help="reading date, YYYY-MM-DD (stage read)")
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
