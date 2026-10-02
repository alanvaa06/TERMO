"""Command line for spec 1. Console output is ASCII only."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from termo.config import CoreConfig, load_config
from termo.core import (
    DIRTY_SUFFIX,
    build_report,
    publish_report,
    register_trials,
    registered_columns,
    run_final_holdout,
    run_trials,
    take_snapshot,
    verify_binding,
)
from termo.data.fred import http_get
from termo.data.loader import load_curve
from termo.data.snapshot import snapshot_hash
from termo.dataset import prepare
from termo.validation.trials import TrialLog

TRIALS_DIR = Path("trials")
REPORTS_DIR = Path("reports")
SNAPSHOTS_DIR = Path("data") / "snapshots"
TRIALS_FILE = "trials.jsonl"
CODE_PATHS = ("src", "configs", "pyproject.toml")


def code_commit() -> str:
    """Last commit that touched code or configuration, marked dirty if they have changed since.

    Only the paths that decide the results count. Snapshots, the trial log, labels and
    reports are committed while the experiment runs; they must not move this value.
    """
    head = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *CODE_PATHS],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *CODE_PATHS],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{head}{DIRTY_SUFFIX}" if status else head


def _pre_holdout_curve(config: CoreConfig, snapshot_dir: Path) -> pd.DataFrame:
    return load_curve(snapshot_dir, config.series, config.start, config.holdout_start)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="termo", description="TERMO spec 1: core go/no-go")
    parser.add_argument("stage", choices=["snapshot", "register", "run", "report", "final-holdout"])
    parser.add_argument("--config", type=Path, default=Path("configs/core.yaml"))
    parser.add_argument(
        "--snapshot", type=Path, help="snapshot directory (all stages but snapshot)"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    log = TrialLog(TRIALS_DIR / TRIALS_FILE)

    if args.stage == "snapshot":
        now = datetime.now(UTC)
        target = SNAPSHOTS_DIR / now.date().isoformat()
        take_snapshot(config, target, http_get, now.isoformat(timespec="seconds"))
        print(f"[ok] snapshot written to {target.as_posix()}")
        return 0

    if args.snapshot is None:
        parser.error("--snapshot is required for this stage")
    snapshot_dir: Path = args.snapshot

    data_hash, commit = snapshot_hash(snapshot_dir), code_commit()

    if args.stage == "register":
        curve = _pre_holdout_curve(config, snapshot_dir)
        columns = register_trials(config, curve, log, data_hash, commit)
        print(f"[ok] registered {len(log.registrations())} trials; features: {', '.join(columns)}")
    elif args.stage == "run":
        verify_binding(log, config, data_hash, commit)  # before any computation
        curve = _pre_holdout_curve(config, snapshot_dir)
        data = prepare(curve, config, registered_columns(log))
        run_trials(data, log, TRIALS_DIR, data_hash, commit)
        print("[ok] all registered trials have a result")
    elif args.stage == "report":
        verify_binding(log, config, data_hash, commit)
        curve = _pre_holdout_curve(config, snapshot_dir)
        data = prepare(curve, config, registered_columns(log))
        report = build_report(data, log, TRIALS_DIR, data_hash, commit)
        publish_report(report, log, REPORTS_DIR, data_hash, commit)
        print(f"[ok] verdict: {report.verdict.value} -> {REPORTS_DIR.as_posix()}/go_no_go.md")
    else:
        result = run_final_holdout(config, snapshot_dir, log, REPORTS_DIR, commit)
        mark = "[ok]" if result.passed else "[x]"
        print(
            f"{mark} holdout: excess_model={result.excess_model:.4f} "
            f"excess_inertia={result.excess_inertia:.4f} weeks={result.n_weeks} "
            f"episodes={result.n_episodes}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
