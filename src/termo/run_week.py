"""The week in one command: the shadow run, the monthly report if asked, the visual report.

Everything the operation produces stays where `termo.op_cli` puts it (the shadow log, the
sheet, the CSV, the ficha); this module runs those stages, copies their files under
`output/<reading date>/` and builds `reporte.html` from those copies and the snapshot.
`--solo-reporte` rebuilds only the report, without running the model. A week run again
overwrites that directory; the shadow log keeps both runs, as always. Console output is
ASCII only.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from termo.op_cli import (
    OPERATION_CONFIG,
    Operation,
    download_snapshot,
    iso_month,
    open_operation,
    run_ficha,
    run_sombra,
)
from termo.operation.visual.pagina import ARCHIVO, construir

OUTPUT_DIR = Path("output")
REPORT_FILE = ARCHIVO
SHEET_STEM = "hoja"  # output/<date>/hoja.md and hoja.json
FICHA_PREFIX = "ficha-"  # output/<date>/ficha-<AAAA-MM>.md and .json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_termo.py",
        description=(
            "TERMO: the weekly run in one command. Run it from the repository root: it "
            "reads configs/ and the registry under trials/ and reports/, and writes the "
            "deliverable under output/<reading date>/."
        ),
        allow_abbrev=False,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--download",
        action="store_true",
        help="take today's operation snapshot into data/snapshots/<today>/ and run on it",
    )
    source.add_argument("--snapshot", type=Path, help="run on an existing operation snapshot")
    parser.add_argument(
        "--mes", type=iso_month, help="also build the monthly report of this month (AAAA-MM)"
    )
    parser.add_argument(
        "--solo-reporte",
        type=Path,
        metavar="DIR",
        help=(
            "only rebuild DIR/reporte.html from the files in DIR (an output/<date>/ "
            "directory) and --snapshot; the model does not run"
        ),
    )
    parser.add_argument(
        "--operacion",
        type=Path,
        default=OPERATION_CONFIG,
        help=f"operation configuration (default: {OPERATION_CONFIG.as_posix()})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help=f"where <reading date>/ is created (default: {OUTPUT_DIR.as_posix()})",
    )
    return parser


def _report(
    parser: argparse.ArgumentParser, ctx: Operation, out_dir: Path, snapshot_dir: Path
) -> Path:
    try:
        return construir(
            out_dir,
            snapshot_dir,
            ctx.config,
            ctx.op,
            datetime.now(UTC).isoformat(timespec="seconds"),
        )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.solo_reporte is not None and (args.download or args.mes is not None):
        parser.error("--solo-reporte takes --snapshot only: no --download, no --mes")
    try:
        ctx = open_operation(args.operacion)
    except ValueError as error:
        parser.error(str(error))

    if args.solo_reporte is not None:
        report = _report(parser, ctx, args.solo_reporte, args.snapshot)
        print(f"[ok] reporte -> {report.as_posix()}")
        return 0

    snapshot_dir: Path
    if args.download:
        try:
            snapshot_dir = download_snapshot(ctx)
        except FileExistsError as error:
            parser.error(str(error))
    else:
        snapshot_dir = args.snapshot

    run, csv_paths = run_sombra(ctx, snapshot_dir)
    stamp = str(run.record["reading_date"])
    copies: list[tuple[Path, str]] = [
        (run.sheet_path, f"{SHEET_STEM}.md"),
        (run.sheet_path.with_suffix(".json"), f"{SHEET_STEM}.json"),
        *((path, name) for name, path in csv_paths.items()),
    ]
    if args.mes is not None:
        _, ficha_path = run_ficha(ctx, snapshot_dir, args.mes, run.curve, run.analysis)
        copies += [
            (ficha_path, f"{FICHA_PREFIX}{args.mes}.md"),
            (ficha_path.with_suffix(".json"), f"{FICHA_PREFIX}{args.mes}.json"),
        ]

    out_dir: Path = args.output / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    for source_path, name in copies:
        shutil.copyfile(source_path, out_dir / name)
    report = _report(parser, ctx, out_dir, snapshot_dir)
    print(f"[ok] reporte -> {report.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
