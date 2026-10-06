"""What the visual report reads: the week's output directory and its snapshot, checked.

The report is rebuilt from files, never from a model run: the CSV and hoja.json are the
public contract of the deliverable. The snapshot must be the one the sheet names.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from termo.config import CoreConfig
from termo.data.loader import load_curve
from termo.data.snapshot import snapshot_hash
from termo.operation.config import OperationConfig
from termo.operation.macro import _column_name, _label

HOJA = "hoja.json"
HISTORIA = "historia_diaria.csv"
EPISODIOS = "episodios.csv"
MACRO = "macro.csv"
COMENTARIO = "comentario.md"
REQUERIDOS = (HOJA, HISTORIA, EPISODIOS, MACRO)


@dataclass(frozen=True)
class SerieMacro:
    columna: str  # column of macro.csv
    etiqueta: str  # Spanish label, as the sheet shows it


@dataclass(frozen=True, eq=False)
class DatosReporte:
    fecha: pd.Timestamp  # the reading date: the last row of `historia`
    hoja: Mapping[str, Any]  # hoja.json as written by the shadow stage
    historia: pd.DataFrame  # one row per labelled day, index fecha (ns)
    episodios: pd.DataFrame  # one row per episode; the last one is the current, still open
    macro: pd.DataFrame  # index fecha (ns)
    curva: pd.DataFrame  # yields in percent, one column per tenor, index ns
    nombres: tuple[str, ...]  # phase names, by phase number
    bloques: tuple[str, ...]  # SHAP block names, as registered
    series_macro: tuple[SerieMacro, ...]
    textos: Mapping[str, str]  # operacion.yaml texts: descargo, nota_confianza, ...
    ventana_percentil: int  # known values in the macro percentile window
    comentario: str | None  # comentario.md, when the analyst wrote one

    @property
    def lectura(self) -> Mapping[str, Any]:
        lectura: Mapping[str, Any] = self.hoja["reading"]
        return lectura


HUELLA = "# snapshot_hash="


def _csv(path: Path, esperada: str, fechas: Sequence[str]) -> pd.DataFrame:
    """A CSV of the export, read only when its first line names the sheet's snapshot."""
    with path.open(encoding="utf-8") as handle:
        primera = handle.readline().strip()
    if primera != f"{HUELLA}{esperada}":
        encontrada = primera.removeprefix(HUELLA)
        raise ValueError(
            f"{path.name} es de otra corrida (snapshot {encontrada[:12]}, la hoja dice "
            f"{esperada[:12]})"
        )
    table = pd.read_csv(path, skiprows=2, parse_dates=list(fechas))
    for column in fechas:
        table[column] = table[column].dt.as_unit("ns")
    return table


def cargar(
    dir_salida: Path, dir_snapshot: Path, config: CoreConfig, op: OperationConfig
) -> DatosReporte:
    """Every input of the report; refused when a file is missing or the snapshot differs."""
    faltan = [name for name in REQUERIDOS if not (dir_salida / name).exists()]
    if faltan:
        raise FileNotFoundError(f"faltan {faltan} en {dir_salida.as_posix()}")
    desc = config.descriptive
    if desc is None:
        raise ValueError("la configuracion del modelo no es descriptiva")
    hoja: dict[str, Any] = json.loads((dir_salida / HOJA).read_text(encoding="utf-8"))
    huella = snapshot_hash(dir_snapshot)
    esperada = str(hoja["snapshot_hash"])
    if huella != esperada:
        raise ValueError(
            f"el snapshot {dir_snapshot.as_posix()} ({huella[:12]}) no es el de la hoja "
            f"({esperada[:12]})"
        )
    historia = _csv(dir_salida / HISTORIA, esperada, ["fecha"]).set_index("fecha")
    fecha = pd.Timestamp(hoja["reading_date"]).as_unit("ns")
    if historia.index[-1] != fecha:
        raise ValueError(
            f"la historia termina el {historia.index[-1].date()} y la lectura es del "
            f"{fecha.date()}"
        )
    episodios = _csv(dir_salida / EPISODIOS, esperada, ["inicio", "fin"])
    ultima, leida = int(episodios.iloc[-1]["fase"]), int(hoja["reading"]["phase"])
    if ultima != leida:
        raise ValueError(
            f"el ultimo episodio no es la fase de la lectura (episodio {ultima}, lectura {leida})"
        )
    curva = load_curve(
        dir_snapshot, config.series, config.start, config.holdout_start, final_evaluation=True
    )
    curva.index = pd.DatetimeIndex(curva.index).as_unit("ns")
    comentario = dir_salida / COMENTARIO
    return DatosReporte(
        fecha=fecha,
        hoja=hoja,
        historia=historia,
        episodios=episodios,
        macro=_csv(dir_salida / MACRO, esperada, ["fecha"]).set_index("fecha"),
        curva=curva,
        nombres=tuple(desc.phase_names),
        bloques=tuple(name for name, _ in desc.blocks),
        series_macro=tuple(SerieMacro(_column_name(m), _label(m)) for m in op.macro),
        textos=dict(op.texts),
        ventana_percentil=op.percentile_window_days,
        comentario=comentario.read_text(encoding="utf-8") if comentario.exists() else None,
    )
