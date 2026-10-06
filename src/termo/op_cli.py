"""Command line for spec 4 (operation in shadow mode). Console output is ASCII only.

Stages: `sombra` (the weekly run, then the CSV), `ficha --mes AAAA-MM` (the monthly
report), `evaluar-sombra` (the registered criteria on the shadow days) and `exportar`
(the five CSV). Every stage but `sombra --download` works on a given snapshot; the three
that do not log a reading recompute the latest outputs from it the way the shadow run does.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from termo.cli import code_identity
from termo.config import CoreConfig, load_config
from termo.data.fred import http_get
from termo.data.snapshot import snapshot_hash
from termo.desc_cli import REGISTRY_PATTERN, TRIALS_FILE, reports_dir, trials_dir
from termo.descriptive.stages import Analysis
from termo.operation.config import OperationConfig, load_operation_config
from termo.operation.exports import export_all
from termo.operation.macro import macro_frame, macro_panel
from termo.operation.monthly import build_monthly, write_monthly
from termo.operation.shadow import (
    SHADOW_DIR,
    SHADOW_LOG_FILE,
    ShadowLog,
    latest_analysis,
    run_shadow,
)
from termo.operation.shadow_eval import REPORT_PREFIX, evaluate_shadow
from termo.operation.snapshot import take_operation_snapshot
from termo.validation.trials import TrialLog

STAGES = ("sombra", "ficha", "evaluar-sombra", "exportar")
OPERATION_CONFIG = Path("configs") / "operacion.yaml"
SNAPSHOTS_DIR = Path("data") / "snapshots"
CSV_DIR = "csv"  # reports/<registry>/csv
MONTHLY_DIR = "fichas"  # reports/<registry>/fichas/<AAAA-MM>.md
MONTH_PATTERN = r"^\d{4}-\d{2}$"


def iso_month(text: str) -> str:
    """AAAA-MM, refused with a message rather than a traceback."""
    try:
        if not re.fullmatch(MONTH_PATTERN, text):
            raise ValueError(text)
        date.fromisoformat(f"{text}-01")
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not a month in AAAA-MM form") from error
    return text


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _export(
    op: OperationConfig,
    config: CoreConfig,
    model_log: TrialLog,
    shadow_log: ShadowLog,
    curve: pd.DataFrame,
    analysis: Analysis,
    snapshot_dir: Path,
    reports: Path,
) -> Path:
    out = reports / CSV_DIR
    export_all(
        shadow_log.records(),
        model_log,
        analysis,
        curve,
        config,
        op,
        macro_frame(snapshot_dir, config, op),
        out,
        snapshot_hash(snapshot_dir),
        _utc_now().isoformat(timespec="seconds"),
    )
    return out


def _month_close(analysis: Analysis, month: str) -> date:
    """The last labelled day of the month: the day the monthly report describes."""
    period = pd.Period(month, freq="M")
    in_month = analysis.labels.loc[period.start_time : period.end_time]
    if in_month.empty:
        raise ValueError(f"no labelled days in {month}")
    return pd.Timestamp(in_month.index[-1]).date()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="termo-op", description="TERMO: operation in shadow mode", allow_abbrev=False
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument(
        "--operacion",
        type=Path,
        default=OPERATION_CONFIG,
        help=f"operation configuration (default: {OPERATION_CONFIG.as_posix()})",
    )
    parser.add_argument("--snapshot", type=Path, help="snapshot directory")
    parser.add_argument(
        "--download",
        action="store_true",
        help="stage sombra: take a new operation snapshot into data/snapshots/<today>/ first",
    )
    parser.add_argument("--mes", type=iso_month, help="month AAAA-MM (stage ficha)")
    parser.add_argument(
        "--min-weeks",
        type=int,
        help="stage evaluar-sombra: weeks of shadow readings required (default: configured)",
    )
    args = parser.parse_args(argv)
    if args.download and args.stage != "sombra":
        parser.error("--download belongs to the sombra stage")
    if args.download and args.snapshot is not None:
        parser.error("give --snapshot or --download, not both")
    if args.snapshot is None and not args.download:
        parser.error("--snapshot is required (the sombra stage may use --download instead)")
    if args.stage == "ficha" and args.mes is None:
        parser.error("--mes is required for the ficha stage")
    if args.min_weeks is not None and args.stage != "evaluar-sombra":
        parser.error("--min-weeks belongs to the evaluar-sombra stage")
    if args.min_weeks is not None and args.min_weeks < 1:
        parser.error("--min-weeks must be a positive number of weeks")

    op = load_operation_config(args.operacion)
    if not re.fullmatch(REGISTRY_PATTERN, op.registry):
        parser.error("registry must be a plain name: lowercase letters, digits, underscore")
    trials, reports = trials_dir(op.registry), reports_dir(op.registry)
    config = load_config(op.model_config)
    model_log = TrialLog(trials / TRIALS_FILE)
    shadow_log = ShadowLog(trials / SHADOW_DIR / SHADOW_LOG_FILE)
    commit = code_identity()

    snapshot_dir: Path
    if args.download:
        now = _utc_now()
        snapshot_dir = SNAPSHOTS_DIR / now.date().isoformat()
        if snapshot_dir.exists():  # checked before anything is fetched
            parser.error(
                f"{snapshot_dir.as_posix()} exists: use --snapshot {snapshot_dir.as_posix()}"
            )
        take_operation_snapshot(
            config, op, snapshot_dir, http_get, now.isoformat(timespec="seconds")
        )
        print(f"[ok] snapshot -> {snapshot_dir.as_posix()}")
    else:
        snapshot_dir = args.snapshot

    if args.stage == "sombra":
        run = run_shadow(op, config, model_log, snapshot_dir, trials, reports, commit)
        curve, analysis = run.curve, run.analysis
    else:
        curve, analysis = latest_analysis(config, model_log, snapshot_dir)

    if args.stage == "ficha":
        month: str = args.mes
        rows = macro_panel(macro_frame(snapshot_dir, config, op), op, _month_close(analysis, month))
        payload = build_monthly(
            month, shadow_log.records(), model_log, analysis, curve, config, op, rows, commit
        )
        path = write_monthly(payload, op, reports / MONTHLY_DIR)
        print(f"[ok] ficha -> {path.as_posix()}")
        return 0
    if args.stage == "evaluar-sombra":
        result = evaluate_shadow(
            op,
            config,
            model_log,
            shadow_log,
            trials,
            reports,
            curve,
            analysis,
            commit,
            min_weeks=args.min_weeks,
        )
        path = reports / f"{REPORT_PREFIX}{result['period']['hasta']}.md"
        print(f"[ok] evaluacion de sombra: {result['verdict']} -> {path.as_posix()}")
        return 0
    out = _export(op, config, model_log, shadow_log, curve, analysis, snapshot_dir, reports)
    print(f"[ok] csv -> {out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
