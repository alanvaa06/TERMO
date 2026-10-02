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
    build_report,
    register_trials,
    registered_columns,
    run_final_holdout,
    run_trials,
    take_snapshot,
)
from termo.data.fred import http_get
from termo.data.loader import load_curve
from termo.data.snapshot import snapshot_hash
from termo.dataset import prepare
from termo.report import write_report
from termo.validation.trials import TrialLog

TRIALS_DIR = Path("trials")
REPORTS_DIR = Path("reports")
SNAPSHOTS_DIR = Path("data") / "snapshots"
TRIALS_FILE = "trials.jsonl"


def code_commit() -> str:
    """Current commit, marked dirty when there are uncommitted changes."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return f"{head}-dirty" if status else head


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

    if args.stage == "register":
        curve = _pre_holdout_curve(config, snapshot_dir)
        columns = register_trials(config, curve, log, snapshot_hash(snapshot_dir), code_commit())
        print(f"[ok] registered {len(log.registrations())} trials; features: {', '.join(columns)}")
    elif args.stage == "run":
        curve = _pre_holdout_curve(config, snapshot_dir)
        run_trials(prepare(curve, config, registered_columns(log)), log, TRIALS_DIR)
        print("[ok] all registered trials have a result")
    elif args.stage == "report":
        curve = _pre_holdout_curve(config, snapshot_dir)
        data = prepare(curve, config, registered_columns(log))
        report = build_report(data, log, TRIALS_DIR)
        write_report(report, REPORTS_DIR)
        print(f"[ok] verdict: {report.verdict.value} -> {REPORTS_DIR.as_posix()}/go_no_go.md")
    else:
        result = run_final_holdout(config, snapshot_dir, log, REPORTS_DIR)
        mark = "[ok]" if result.passed else "[x]"
        print(
            f"{mark} holdout: excess_model={result.excess_model:.4f} "
            f"excess_inertia={result.excess_inertia:.4f} weeks={result.n_weeks} "
            f"episodes={result.n_episodes}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
