"""The spec 4 command line: the four stages on the synthetic desc2 registry."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from conftest import DESC2_COMMIT, Desc2Registry, fake_fred_frames, make_macro
from termo.config import load_config
from termo.core import config_fingerprint
from termo.desc_cli import TRIALS_FILE
from termo.op_cli import CSV_DIR, MONTHLY_DIR, main
from termo.operation.config import load_operation_config
from termo.operation.shadow import SHADOW_DIR, SHADOW_LOG_FILE, ShadowLog
from termo.operation.shadow_eval import REPORT_PREFIX
from termo.operation.snapshot import take_operation_snapshot
from test_desc_cli import DESC_SMALL_CONFIG

REPO_OP = Path(__file__).resolve().parents[1] / "configs" / "operacion.yaml"
REGISTRY = "desc2"
TRIALS_DIR = Path("trials") / REGISTRY
REPORTS_DIR = Path("reports") / REGISTRY
OP_CONFIG = Path("configs") / "operacion.yaml"
MODEL_CONFIG = Path("configs") / "desc2.yaml"
SNAPSHOT = Path("data") / "snapshots" / "op"
CSV_FILES = ("lecturas.csv", "historia_diaria.csv", "episodios.csv", "macro.csv", "alertas.csv")
PREFIXES = ("[ok] ", "[x] ", "[!] ")

# The registry fixture's configuration as YAML: desc.yaml's small variant with the desc2
# names and bookkeeping, without the curve-shape claim. The fixture proves it round-trips.
DESC2_SMALL_CONFIG = (
    DESC_SMALL_CONFIG.replace(
        "phase_names: [rally de la parte corta, rally de la parte larga, venta]",
        "phase_names: [rally fuerte, rally moderado, venta]",
    )
    .replace(
        "  short_led_phase: 0\n  long_led_phase: 1\n",
        '  holdout_already_seen: true\n  holdout_seen_by: "desc_k3 (test)"\n',
    )
    .replace("  slope_long: DGS10\n  slope_short: DGS2\n", "")
)


def _lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line]


def _console_ok(out: str) -> list[str]:
    """Console lines must be ASCII and each one tagged; return them."""
    assert out.isascii()
    lines = _lines(out)
    assert lines and all(line.startswith(PREFIXES) for line in lines)
    return lines


@pytest.fixture(scope="module")
def workspace(
    desc2_registry: Desc2Registry, tmp_path_factory: pytest.TempPathFactory, curve: pd.DataFrame
) -> Path:
    """A private registry copy, the two configurations and an operation snapshot of the FULL
    curve plus the macro series, laid out as the repository is (trials/, reports/, configs/)."""
    assert "rally fuerte" in DESC2_SMALL_CONFIG and "slope_long" not in DESC2_SMALL_CONFIG
    root = tmp_path_factory.mktemp("termo-op-cli")
    registry = desc2_registry.copy_to(root)
    (root / "configs").mkdir()
    (root / MODEL_CONFIG).write_text(DESC2_SMALL_CONFIG, encoding="utf-8")
    assert config_fingerprint(load_config(root / MODEL_CONFIG)) == config_fingerprint(
        registry.config
    )
    shutil.copy(REPO_OP, root / OP_CONFIG)
    op = load_operation_config(root / OP_CONFIG)
    assert op.registry == REGISTRY and op.model_config == MODEL_CONFIG
    get = fake_fred_frames([curve, make_macro(curve)])
    take_operation_snapshot(registry.config, op, root / SNAPSHOT, get, "1999-03-15T00:00:00+00:00")
    return root


@pytest.fixture
def here(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(workspace)
    monkeypatch.setattr("termo.op_cli.code_identity", lambda: DESC2_COMMIT)
    return workspace


def _shadow_log() -> ShadowLog:
    return ShadowLog(TRIALS_DIR / SHADOW_DIR / SHADOW_LOG_FILE)


def test_the_four_stages_through_the_command_line(
    here: Path, capsys: pytest.CaptureFixture[str], curve: pd.DataFrame
) -> None:
    model_log = TRIALS_DIR / TRIALS_FILE
    model_before = model_log.read_bytes()
    stage = ["--snapshot", str(SNAPSHOT)]

    # sombra, twice: two readings of the same week, the CSV refreshed after each
    assert main(["sombra", *stage]) == 0
    lines = _console_ok(capsys.readouterr().out)
    assert lines[0] == "[ok] historia reproducida"
    assert lines[1].startswith("[ok] lectura ")
    assert lines[-2].startswith(f"[ok] hoja -> {REPORTS_DIR.as_posix()}/{SHADOW_DIR}/")
    assert lines[-1] == f"[ok] csv -> {REPORTS_DIR.as_posix()}/{CSV_DIR}"
    (record,) = _shadow_log().readings()
    day = str(record["reading_date"])
    assert record["code_commit"] == DESC2_COMMIT
    sheet = REPORTS_DIR / SHADOW_DIR / f"{day}.md"
    assert sheet.exists() and sheet.with_suffix(".json").exists()
    text = sheet.read_text(encoding="utf-8")
    assert "días" in text and "## Fase actual" in text and not text.isascii()
    assert f"hoja semanal del {day}" in text
    for name in CSV_FILES:
        assert (REPORTS_DIR / CSV_DIR / name).exists()
    assert model_log.read_bytes() == model_before

    assert main(["sombra", *stage]) == 0
    _console_ok(capsys.readouterr().out)
    readings = _shadow_log().readings()
    assert len(readings) == 2 and readings[1]["reading_date"] == day
    assert readings[1]["alert"] is None  # the same phase as the previous shadow reading
    assert model_log.read_bytes() == model_before

    # exportar: the five CSV again, one row per reading
    assert main(["exportar", *stage]) == 0
    lines = _console_ok(capsys.readouterr().out)
    assert lines == [f"[ok] csv -> {REPORTS_DIR.as_posix()}/{CSV_DIR}"]
    lecturas = pd.read_csv(REPORTS_DIR / CSV_DIR / "lecturas.csv", comment="#")
    assert lecturas["fecha"].tolist() == [day, day]
    historia = pd.read_csv(REPORTS_DIR / CSV_DIR / "historia_diaria.csv", comment="#")
    assert historia["fecha"].iloc[-1] == curve.index[-1].date().isoformat()
    assert "sombra" in set(historia["periodo"])
    assert model_log.read_bytes() == model_before

    # ficha for the month of the reading
    month = day[:7]
    assert main(["ficha", *stage, "--mes", month]) == 0
    lines = _console_ok(capsys.readouterr().out)
    ficha = REPORTS_DIR / MONTHLY_DIR / f"{month}.md"
    assert lines == [f"[ok] ficha -> {ficha.as_posix()}"]
    assert ficha.exists() and ficha.with_suffix(".json").exists()
    markdown = ficha.read_text(encoding="utf-8")
    assert f"ficha mensual {month}" in markdown and "Duraciones históricas" in markdown
    payload = json.loads(ficha.with_suffix(".json").read_text(encoding="utf-8"))
    assert [r["fecha"] for r in payload["lecturas"]] == [day, day]
    assert payload["ultimo_dia"] == curve.index[-1].date().isoformat()
    assert all(row["fecha_valor"] <= payload["ultimo_dia"] for row in payload["macro"])
    assert payload["generado_con"] == {"code_commit": DESC2_COMMIT}
    assert model_log.read_bytes() == model_before

    # evaluar-sombra, with the minimum lowered to the one week there is
    assert main(["evaluar-sombra", *stage, "--min-weeks", "1"]) == 0
    lines = _console_ok(capsys.readouterr().out)
    records = _shadow_log().records()
    assert [r["kind"] for r in records] == ["reading", "reading", "evaluation"]
    evaluation = records[-1]
    report = REPORTS_DIR / f"{REPORT_PREFIX}{evaluation['period']['hasta']}.md"
    assert lines == [f"[ok] evaluacion de sombra: {evaluation['verdict']} -> {report.as_posix()}"]
    assert report.exists() and report.with_suffix(".json").exists()
    assert evaluation["weeks"] == 1 and evaluation["code_commit"] == DESC2_COMMIT
    assert model_log.read_bytes() == model_before


def test_arguments_are_checked_before_anything_runs(
    here: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_path = TRIALS_DIR / SHADOW_DIR / SHADOW_LOG_FILE
    before = log_path.read_bytes() if log_path.exists() else None
    stage = ["--snapshot", str(SNAPSHOT)]
    # sombra needs a snapshot or the download; never both
    with pytest.raises(SystemExit):
        main(["sombra"])
    assert "--snapshot" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["sombra", *stage, "--download"])
    # the download belongs to sombra only
    with pytest.raises(SystemExit):
        main(["exportar", "--download"])
    # ficha needs a month in AAAA-MM form
    with pytest.raises(SystemExit):
        main(["ficha", *stage])
    assert "--mes" in capsys.readouterr().err
    for bad in ("1999-13", "1999-3", "03/1999"):
        with pytest.raises(SystemExit):
            main(["ficha", *stage, "--mes", bad])
        assert "AAAA-MM" in capsys.readouterr().err
    # --min-weeks belongs to evaluar-sombra and is a positive count
    with pytest.raises(SystemExit):
        main(["sombra", *stage, "--min-weeks", "1"])
    with pytest.raises(SystemExit):
        main(["evaluar-sombra", *stage, "--min-weeks", "0"])
    # no abbreviations
    with pytest.raises(SystemExit):
        main(["sombra", "--snap", str(SNAPSHOT)])
    assert (log_path.read_bytes() if log_path.exists() else None) == before
    assert not (Path("data") / "snapshots").joinpath("1999-03-15").exists()


def test_a_registry_name_that_is_not_plain_is_refused(
    here: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = REPO_OP.read_text(encoding="utf-8").replace("registry: desc2", "registry: ../desc2")
    bad = tmp_path / "operacion.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["exportar", "--operacion", str(bad), "--snapshot", str(SNAPSHOT)])
    error = capsys.readouterr().err
    assert "registry" in error and error.isascii()
    assert not (here / "desc2").exists()  # where trials/../desc2 would have landed
