"""A small, consistent input for the visual report tests: in memory and on disk.

700 labelled business days with seven episodes, the last one an open 'venta'; the curve
and the macro series start PRE_HISTORIA days before the history, as the real ones start
years before it; the surrogate disagrees with the jump model one day in fifty; the last
200 days are the holdout and the last 2 the shadow, as in the real export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from conftest import FRED_URL_TEMPLATE, TENORS, fake_fred, make_curve, make_macro
from termo.data.snapshot import snapshot_hash, write_snapshot
from termo.operation.config import OperationConfig, load_operation_config
from termo.operation.exports import _macro
from termo.operation.macro import macro_panel
from termo.operation.monthly import episodes
from termo.operation.visual.calculos import columna_probabilidad
from termo.operation.visual.datos import DatosReporte, SerieMacro

REPO = Path(__file__).resolve().parents[1]
OP_CONFIG = REPO / "configs" / "operacion.yaml"
NOMBRES = ("rally fuerte", "rally moderado", "venta")
BLOQUES = ("nivel corto", "nivel medio", "nivel largo", "pendientes", "curvatura", "volatilidad")
N_DIAS = 700  # labelled days
PRE_HISTORIA = 60  # curve (and macro) days before the first labelled day
RACHAS = ((0, 60), (2, 90), (1, 70), (2, 120), (0, 40), (1, 100))  # then 'venta' to the end
VENTA = 2
HOLDOUT_DIAS, SOMBRA_DIAS = 200, 2
HUELLA_FALSA = "0" * 64
GENERADO = "1992-09-01T00:00:00+00:00"
PLOTLY_INICIO = get_plotlyjs()[:120]


def sin_plotlyjs(page: str) -> str:
    """The page without the embedded plotly.js bundle (which mentions URLs in its code)."""
    return page.replace(get_plotlyjs(), "")


def _iso(index: pd.Index) -> list[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in index]


def curva_sintetica() -> pd.DataFrame:
    curve, _ = make_curve(N_DIAS + PRE_HISTORIA, seed=1)
    curva = curve.round(4)
    curva.index = pd.DatetimeIndex(curva.index).as_unit("ns")
    return curva


def fases_sinteticas(index: pd.DatetimeIndex) -> pd.Series:
    valores: list[int] = []
    for fase, dias in RACHAS:
        valores += [fase] * dias
    valores += [VENTA] * (len(index) - len(valores))
    return pd.Series(valores[: len(index)], index=index, name="fase")


def historia_sintetica(curva: pd.DataFrame) -> pd.DataFrame:
    index = pd.DatetimeIndex(curva.index[-N_DIAS:], name="fecha")
    fases = fases_sinteticas(index)
    n = len(index)
    posicion = np.arange(n)
    proba = np.full((n, len(NOMBRES)), 0.05)
    proba[posicion, fases.to_numpy()] = 0.9
    desacuerdo = posicion % 50 == 7  # the surrogate names another phase one day in fifty
    otra = (fases.to_numpy() + 1) % len(NOMBRES)
    proba[desacuerdo] = 0.05
    proba[desacuerdo, otra[desacuerdo]] = 0.9
    table = pd.DataFrame(
        {"fase": fases.to_numpy(), "nombre": [NOMBRES[f] for f in fases]}, index=index
    )
    for i, nombre in enumerate(NOMBRES):
        table[columna_probabilidad(nombre)] = proba[:, i]
    rng = np.random.default_rng(3)
    for bloque in BLOQUES:
        table[bloque] = rng.normal(0.0, 0.5, n).round(4)
    table["base"] = 0.3
    periodo = np.full(n, "pre_holdout", dtype=object)
    periodo[n - HOLDOUT_DIAS - SOMBRA_DIAS : n - SOMBRA_DIAS] = "holdout"
    periodo[n - SOMBRA_DIAS :] = "sombra"
    table["periodo"] = periodo.astype(str)
    return table


def episodios_sinteticos(historia: pd.DataFrame, curva: pd.DataFrame) -> pd.DataFrame:
    table = episodes(historia["fase"], curva, NOMBRES)
    table["inicio"] = pd.DatetimeIndex(table["inicio"]).as_unit("ns")
    table["fin"] = pd.DatetimeIndex(table["fin"]).as_unit("ns")
    table["periodo"] = historia["periodo"].reindex(table["inicio"]).to_numpy()
    return table


def _macro_valores(curva: pd.DataFrame) -> pd.DataFrame:
    """The two macro series on the curve's business days, holes where the source has none."""
    crudo = make_macro(curva)
    prima = crudo["THREEFYTP10"].reindex(curva.index)
    prima[prima.index.dayofweek != 4] = np.nan  # weekly, as Kim-Wright arrives
    spread = curva["DGS2"] - crudo["DFF"].reindex(curva.index)
    table = pd.DataFrame({"THREEFYTP10": prima, "DGS2_menos_DFF": spread})
    table.index = pd.DatetimeIndex(curva.index, name="fecha")
    return table


def macro_sintetica(curva: pd.DataFrame, op: OperationConfig | None = None) -> pd.DataFrame:
    """macro.csv as the export builds it: value, 10y percentile and 21d change per series."""
    op = op or load_operation_config(OP_CONFIG)
    table = _macro(_macro_valores(curva), op)
    table["fecha"] = pd.DatetimeIndex(pd.to_datetime(table["fecha"])).as_unit("ns")
    return table.set_index("fecha")


def hoja_sintetica(
    historia: pd.DataFrame,
    episodios: pd.DataFrame,
    huella: str,
    macro: pd.DataFrame | None = None,
    op: OperationConfig | None = None,
) -> dict[str, Any]:
    fecha = _iso(historia.index[-1:])[0]
    actual = episodios.iloc[-1]
    fila = historia.iloc[-1]
    op = op or load_operation_config(OP_CONFIG)
    macro = macro if macro is not None else macro_sintetica(curva_sintetica(), op)
    drivers = sorted(  # by absolute contribution, as reading.py does (stable)
        ({"block": b, "contribution": float(fila[b])} for b in BLOQUES),
        key=lambda d: -abs(float(d["contribution"])),
    )[:3]
    return {
        "kind": "reading",
        "reading_date": fecha,
        "snapshot_hash": huella,
        "snapshot_downloaded_at": GENERADO,
        "run_at": GENERADO,
        "code_commit": "test-commit-visual",
        "alert": None,
        "macro": macro_panel(macro, op, historia.index[-1].date()),
        "reading": {
            "date": fecha,
            "phase": VENTA,
            "phase_name": NOMBRES[VENTA],
            "confidence": 0.9,
            "low_confidence": False,
            "days_in_phase": int(actual["dias"]),
            "episode_start": _iso(pd.DatetimeIndex([actual["inicio"]]))[0],
            "probabilities": {n: float(fila[columna_probabilidad(n)]) for n in NOMBRES},
            "drivers": drivers,
            "drivers_validated": True,
            "surrogate_agrees": True,
            "top_variables": [
                {"variable": "d5_63_r252", "contribution": 0.91},
                {"variable": "d7_84_r252", "contribution": 0.5},
                {"variable": "c5_189_r252", "contribution": 0.3},
                {"variable": "vol10_r252", "contribution": -0.12},
                {"variable": "s10s30_63_r252", "contribution": 0.24},
            ],
            "validation": {
                "by_phase": [
                    {
                        "phase": i,
                        "name": nombre,
                        "days": 100,
                        "evaluable": i != 0,
                        "median_duration_days": 50.0,
                        "direction_share": 0.8,
                        "recall": 0.9,
                        "episodes": 2,
                    }
                    for i, nombre in enumerate(NOMBRES)
                ],
                "registered_verdicts": [
                    {"stage": "diagnostic", "verdict": "apto"},
                    {"stage": "holdout", "verdict": "apto"},
                ],
                "diagnostic": "apto",
                "holdout": "apto",
                "holdout_seen_by": "desc_k3 (test)",
                "failed_checks": [],
                "fidelity_failed": False,
                "sombra": {
                    "semanas": 1,
                    "proxima_evaluacion_semanas": 25,
                    "historia_reproducida": True,
                },
            },
        },
    }


def make_datos(huella: str = HUELLA_FALSA, comentario: str | None = None) -> DatosReporte:
    op = load_operation_config(OP_CONFIG)
    curva = curva_sintetica()
    historia = historia_sintetica(curva)
    episodios = episodios_sinteticos(historia, curva)
    macro = macro_sintetica(curva, op)
    return DatosReporte(
        fecha=pd.Timestamp(historia.index[-1]),
        hoja=hoja_sintetica(historia, episodios, huella, macro, op),
        historia=historia,
        episodios=episodios,
        macro=macro,
        curva=curva,
        nombres=NOMBRES,
        bloques=BLOQUES,
        series_macro=(
            SerieMacro("THREEFYTP10", "prima por plazo 10 anos (Kim-Wright)"),
            SerieMacro("DGS2_menos_DFF", "DGS2 - fed funds efectiva"),
        ),
        textos=dict(op.texts),
        ventana_percentil=op.percentile_window_days,
        comentario=comentario,
    )


def escribir_snapshot(dir_snapshot: Path, curva: pd.DataFrame) -> str:
    """The seven tenors as FRED serves them; returns the snapshot hash."""
    get = fake_fred(curva)
    texts = {s: get(FRED_URL_TEMPLATE.format(series_id=s)) for s in TENORS}
    write_snapshot(dir_snapshot, texts, GENERADO)
    return snapshot_hash(dir_snapshot)


def _escribir_csv(path: Path, table: pd.DataFrame, huella: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# snapshot_hash={huella}\n# generado={GENERADO}\n")
        table.to_csv(handle, index=False, lineterminator="\n")


def escribir_salida(dir_salida: Path, datos: DatosReporte) -> None:
    """hoja.json and the CSV in the export's format (dates as ISO strings)."""
    dir_salida.mkdir(parents=True, exist_ok=True)
    huella = str(datos.hoja["snapshot_hash"])
    (dir_salida / "hoja.json").write_text(
        json.dumps(datos.hoja, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    historia = datos.historia.reset_index()
    historia["fecha"] = _iso(pd.DatetimeIndex(historia["fecha"]))
    _escribir_csv(dir_salida / "historia_diaria.csv", historia, huella)
    episodios = datos.episodios.copy()
    episodios["inicio"] = _iso(pd.DatetimeIndex(episodios["inicio"]))
    episodios["fin"] = _iso(pd.DatetimeIndex(episodios["fin"]))
    _escribir_csv(dir_salida / "episodios.csv", episodios, huella)
    macro = datos.macro.reset_index()
    macro["fecha"] = _iso(pd.DatetimeIndex(macro["fecha"]))
    _escribir_csv(dir_salida / "macro.csv", macro, huella)


def en_disco(raiz: Path, comentario: str | None = None) -> tuple[DatosReporte, Path, Path]:
    """The synthetic week written to `raiz`: (datos with the real hash, output dir, snapshot)."""
    dir_snapshot = raiz / "snapshot"
    huella = escribir_snapshot(dir_snapshot, curva_sintetica())
    datos = make_datos(huella, comentario)
    dir_salida = raiz / "salida"
    escribir_salida(dir_salida, datos)
    if comentario is not None:
        (dir_salida / "comentario.md").write_text(comentario, encoding="utf-8", newline="\n")
    return datos, dir_salida, dir_snapshot
