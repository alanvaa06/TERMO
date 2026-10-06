"""The week in one command: the shadow run, the monthly report if asked, one HTML page.

Everything the operation produces stays where `termo.op_cli` puts it (the shadow log, the
sheet, the CSV, the ficha); this module runs those stages and leaves the deliverable under
`output/<reading date>/`: `reporte.html` plus copies of the files it was built from. A
week run again overwrites that directory; the shadow log keeps both runs, as always.
Console output is ASCII only.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from termo.op_cli import (
    OPERATION_CONFIG,
    download_snapshot,
    iso_month,
    open_operation,
    run_ficha,
    run_sombra,
)
from termo.operation.html_report import render_html, write_html
from termo.operation.monthly import render_monthly
from termo.operation.sheet_es import render_sheet

OUTPUT_DIR = Path("output")
REPORT_FILE = "reporte.html"
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        ctx = open_operation(args.operacion)
    except ValueError as error:
        parser.error(str(error))

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
    sections = [("Hoja semanal", render_sheet(run.record, ctx.op))]
    copies: list[tuple[Path, str]] = [
        (run.sheet_path, f"{SHEET_STEM}.md"),
        (run.sheet_path.with_suffix(".json"), f"{SHEET_STEM}.json"),
        *((path, name) for name, path in csv_paths.items()),
    ]
    if args.mes is not None:
        payload, ficha_path = run_ficha(ctx, snapshot_dir, args.mes, run.curve, run.analysis)
        sections.append((f"Ficha mensual {args.mes}", render_monthly(payload, ctx.op)))
        copies += [
            (ficha_path, f"{FICHA_PREFIX}{args.mes}.md"),
            (ficha_path.with_suffix(".json"), f"{FICHA_PREFIX}{args.mes}.json"),
        ]

    out_dir: Path = args.output / stamp
    page = render_html(
        sections,
        title=f"TERMO — semana del {stamp}",
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    report = write_html(page, out_dir / REPORT_FILE)
    for source_path, name in copies:
        shutil.copyfile(source_path, out_dir / name)
    print(f"[ok] reporte -> {report.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
