"""Loading the week for the visual report: files checked, snapshot matched, dates aligned."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_desc2_config
from termo.operation.config import load_operation_config
from termo.operation.visual.datos import DatosReporte, cargar
from visual_fixture import BLOQUES, NOMBRES, OP_CONFIG, en_disco


def _cargar(salida: Path, snapshot: Path) -> DatosReporte:
    return cargar(salida, snapshot, make_desc2_config(), load_operation_config(OP_CONFIG))


def test_the_week_round_trips_from_disk(tmp_path: Path) -> None:
    datos, salida, snapshot = en_disco(tmp_path, comentario="## Nota\n\nTexto del analista.")
    cargado = _cargar(salida, snapshot)
    assert cargado.fecha == datos.fecha
    assert cargado.hoja == datos.hoja
    assert cargado.nombres == NOMBRES and cargado.bloques == BLOQUES
    assert list(cargado.historia.index) == list(datos.historia.index)
    assert list(cargado.historia.columns) == list(datos.historia.columns)
    assert np.allclose(cargado.historia[list(BLOQUES)], datos.historia[list(BLOQUES)])
    assert cargado.historia["periodo"].tolist() == datos.historia["periodo"].tolist()
    assert cargado.episodios["inicio"].tolist() == datos.episodios["inicio"].tolist()
    assert cargado.episodios["dias"].tolist() == datos.episodios["dias"].tolist()
    assert list(cargado.curva.columns) == list(datos.curva.columns)
    assert np.allclose(cargado.curva.to_numpy(), datos.curva.to_numpy())
    assert [s.columna for s in cargado.series_macro] == ["THREEFYTP10", "DGS2_menos_DFF"]
    assert cargado.series_macro[1].etiqueta == "DGS2 - fed funds efectiva"
    assert cargado.comentario == "## Nota\n\nTexto del analista."
    assert cargado.ventana_percentil == 2520
    assert cargado.lectura["phase_name"] == "venta"


def test_without_a_comment_there_is_none(tmp_path: Path) -> None:
    _, salida, snapshot = en_disco(tmp_path)
    assert _cargar(salida, snapshot).comentario is None


def test_a_snapshot_that_is_not_the_sheets_is_refused(tmp_path: Path) -> None:
    _, salida, snapshot = en_disco(tmp_path)
    hoja = json.loads((salida / "hoja.json").read_text(encoding="utf-8"))
    hoja["snapshot_hash"] = "f" * 64
    (salida / "hoja.json").write_text(json.dumps(hoja), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot") as error:
        _cargar(salida, snapshot)
    assert str(error.value).isascii() and "ffffffffffff" in str(error.value)


@pytest.mark.parametrize(
    "archivo", ["hoja.json", "historia_diaria.csv", "episodios.csv", "macro.csv"]
)
def test_a_missing_required_file_is_named(tmp_path: Path, archivo: str) -> None:
    _, salida, snapshot = en_disco(tmp_path)
    (salida / archivo).unlink()
    with pytest.raises(FileNotFoundError, match=archivo.replace(".", r"\.")) as error:
        _cargar(salida, snapshot)
    assert str(error.value).isascii()


def test_a_history_that_does_not_end_on_the_reading_date_is_refused(tmp_path: Path) -> None:
    _, salida, snapshot = en_disco(tmp_path)
    hoja = json.loads((salida / "hoja.json").read_text(encoding="utf-8"))
    hoja["reading_date"] = (pd.Timestamp(hoja["reading_date"]) - pd.Timedelta(days=7)).strftime(
        "%Y-%m-%d"
    )
    (salida / "hoja.json").write_text(json.dumps(hoja), encoding="utf-8")
    with pytest.raises(ValueError, match="la historia termina") as error:
        _cargar(salida, snapshot)
    assert str(error.value).isascii()


def test_a_csv_from_another_run_is_refused(tmp_path: Path) -> None:
    _, salida, snapshot = en_disco(tmp_path)
    path = salida / "episodios.csv"
    first, rest = path.read_bytes().split(b"\n", 1)
    assert first.startswith(b"# snapshot_hash=")
    path.write_bytes(b"# snapshot_hash=" + b"e" * 64 + b"\n" + rest)
    with pytest.raises(ValueError, match="otra corrida") as error:
        _cargar(salida, snapshot)
    message = str(error.value)
    assert message.isascii() and "episodios.csv" in message and "eeeeeeeeeeee" in message
