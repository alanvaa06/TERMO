"""The one-command weekly run: the sombra stage, the ficha, and the deliverable under output/."""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest

from conftest import DESC2_COMMIT, Desc2Registry
from termo.desc_cli import TRIALS_FILE
from termo.op_cli import CSV_DIR, MONTHLY_DIR
from termo.operation.config import load_operation_config
from termo.operation.shadow import SHADOW_DIR
from termo.run_week import REPORT_FILE, main
from test_op_cli import (
    CSV_FILES,
    REPORTS_DIR,
    SNAPSHOT,
    TRIALS_DIR,
    _console_ok,
    _shadow_log,
    make_workspace,
)
from visual_fixture import PLOTLY_INICIO, sin_plotlyjs

RECURSO_EXTERNO = re.compile(r"""(?:src|href)\s*=\s*["']?https?:""", re.IGNORECASE)


@pytest.fixture(scope="module")
def workspace(
    desc2_registry: Desc2Registry, tmp_path_factory: pytest.TempPathFactory, curve: pd.DataFrame
) -> Path:
    """This module's own registry copy: its shadow log starts empty."""
    return make_workspace(desc2_registry, tmp_path_factory.mktemp("termo-run-week"), curve)


@pytest.fixture
def here(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(workspace)
    monkeypatch.setattr("termo.op_cli.code_identity", lambda: DESC2_COMMIT)
    return workspace


def test_the_week_in_one_command_with_the_monthly_report(
    here: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_log = TRIALS_DIR / TRIALS_FILE
    model_before = model_log.read_bytes()
    # the month is the reading's: not known before the run, so run once to learn it
    assert main(["--snapshot", str(SNAPSHOT)]) == 0
    lines = _console_ok(capsys.readouterr().out)
    (record,) = _shadow_log().readings()
    day = str(record["reading_date"])
    out = Path("output") / day
    assert lines[-1] == f"[ok] reporte -> {out.as_posix()}/{REPORT_FILE}"
    assert not (out / f"ficha-{day[:7]}.md").exists()
    month = day[:7]

    # the same week again, with the ficha: the output directory is overwritten
    assert main(["--snapshot", str(SNAPSHOT), "--mes", month]) == 0
    lines = _console_ok(capsys.readouterr().out)
    assert lines[0] == "[ok] historia reproducida"
    assert lines[1].startswith(f"[ok] lectura {day} ")
    assert f"[ok] csv -> {REPORTS_DIR.as_posix()}/{CSV_DIR}" in lines
    ficha = REPORTS_DIR / MONTHLY_DIR / f"{month}.md"
    assert f"[ok] ficha -> {ficha.as_posix()}" in lines
    assert lines[-1] == f"[ok] reporte -> {out.as_posix()}/{REPORT_FILE}"
    readings = _shadow_log().readings()
    assert len(readings) == 2 and readings[1]["reading_date"] == day

    report = out / REPORT_FILE
    raw = report.read_bytes()
    page = raw.decode("utf-8")
    assert b"\r\n" not in raw
    assert page.lower().startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in page
    assert '<section id="portada"' in page and '<section id="validacion"' in page
    assert record["reading"]["phase_name"] in page
    assert page.count(PLOTLY_INICIO) == 1
    sheet_json = json.loads((out / "hoja.json").read_text(encoding="utf-8"))
    assert sheet_json["reading_date"] == day
    descargo = load_operation_config(Path("configs") / "operacion.yaml").texts["descargo"]
    assert not descargo.isascii() and html.escape(descargo) in page
    assert not RECURSO_EXTERNO.search(sin_plotlyjs(page))

    # byte-for-byte copies of the sheet, the ficha and the five CSV
    sheet = REPORTS_DIR / SHADOW_DIR / f"{day}.md"
    assert (out / "hoja.md").read_bytes() == sheet.read_bytes()
    assert (out / "hoja.json").read_bytes() == sheet.with_suffix(".json").read_bytes()
    assert (out / f"ficha-{month}.md").read_bytes() == ficha.read_bytes()
    assert (out / f"ficha-{month}.json").read_bytes() == ficha.with_suffix(".json").read_bytes()
    for name in CSV_FILES:
        assert (out / name).read_bytes() == (REPORTS_DIR / CSV_DIR / name).read_bytes()
    expected = [REPORT_FILE, "hoja.md", "hoja.json", f"ficha-{month}.md", f"ficha-{month}.json"]
    assert sorted(p.name for p in out.iterdir()) == sorted([*expected, *CSV_FILES])
    assert model_log.read_bytes() == model_before


def test_output_directory_can_be_chosen(here: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--snapshot", str(SNAPSHOT), "--output", "entregas"]) == 0
    lines = _console_ok(capsys.readouterr().out)
    day = str(_shadow_log().readings()[-1]["reading_date"])
    assert lines[-1] == f"[ok] reporte -> entregas/{day}/{REPORT_FILE}"
    assert (Path("entregas") / day / REPORT_FILE).exists()


def test_snapshot_and_download_are_exclusive_and_one_is_required(
    here: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_path = _shadow_log().path
    before = log_path.read_bytes() if log_path.exists() else None
    with pytest.raises(SystemExit):
        main(["--snapshot", str(SNAPSHOT), "--download"])
    error = capsys.readouterr().err
    assert "--download" in error and "--snapshot" in error and error.isascii()
    with pytest.raises(SystemExit):
        main([])
    assert "--snapshot" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--mes", "1999-03"])
    with pytest.raises(SystemExit):
        main(["--snapshot", str(SNAPSHOT), "--mes", "1999-3"])
    assert "AAAA-MM" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # no abbreviations
        main(["--snap", str(SNAPSHOT)])
    assert (log_path.read_bytes() if log_path.exists() else None) == before


def test_help_says_to_run_from_the_repository_root(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.isascii() and "repository root" in out
    assert "--download" in out and "--snapshot" in out and "--mes" in out
    assert "--solo-reporte" in out


def test_the_report_alone_is_rebuilt_from_the_files(
    here: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--snapshot", str(SNAPSHOT)]) == 0
    capsys.readouterr()
    day = str(_shadow_log().readings()[-1]["reading_date"])
    out = Path("output") / day
    log_before = _shadow_log().path.read_bytes()
    (out / REPORT_FILE).unlink()
    (out / "comentario.md").write_text("## Lectura\n\nNota del analista.", encoding="utf-8")

    assert main(["--solo-reporte", str(out), "--snapshot", str(SNAPSHOT)]) == 0
    lines = _console_ok(capsys.readouterr().out)
    assert lines == [f"[ok] reporte -> {out.as_posix()}/{REPORT_FILE}"]
    page = (out / REPORT_FILE).read_text(encoding="utf-8")
    assert '<section id="comentario"' in page and "Nota del analista." in page
    assert _shadow_log().path.read_bytes() == log_before  # no model run, no new reading


def test_the_report_alone_refuses_download_mes_and_a_foreign_snapshot(
    here: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["--snapshot", str(SNAPSHOT)]) == 0
    capsys.readouterr()
    out = Path("output") / str(_shadow_log().readings()[-1]["reading_date"])
    with pytest.raises(SystemExit):
        main(["--solo-reporte", str(out), "--download"])
    assert "--solo-reporte" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--solo-reporte", str(out), "--snapshot", str(SNAPSHOT), "--mes", "1999-03"])
    assert "--solo-reporte" in capsys.readouterr().err

    foreign = tmp_path / "otra-semana"
    shutil.copytree(out, foreign)
    sheet = json.loads((foreign / "hoja.json").read_text(encoding="utf-8"))
    sheet["snapshot_hash"] = "f" * 64
    (foreign / "hoja.json").write_text(json.dumps(sheet), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--solo-reporte", str(foreign), "--snapshot", str(SNAPSHOT)])
    error = capsys.readouterr().err
    assert "snapshot" in error and error.isascii()
