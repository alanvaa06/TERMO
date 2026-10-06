"""The descriptive numbers the visual report adds; each day uses only itself and the past."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from termo.operation.visual.calculos import (
    Cuadrante,
    acuerdo,
    banda_macro,
    bloque_dominante,
    cambio_pasado,
    columna_probabilidad,
    cuadrante,
    curvas_pasadas,
    fase_imitador,
    firma_por_fase,
    posicion_duracion,
)

NOMBRES = ("rally fuerte", "rally moderado", "venta")
DIAS = pd.bdate_range("2020-01-01", periods=60, name="fecha")


def _curva(valores: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(valores, index=DIAS)


def test_probability_column_names_follow_the_export() -> None:
    assert columna_probabilidad("rally fuerte") == "p_rally_fuerte"


def test_trailing_change_uses_only_the_past() -> None:
    curva = _curva({"DGS2": np.arange(60) * 0.01, "DGS10": np.zeros(60)})
    cambio = cambio_pasado(curva, dias=21)
    assert cambio["DGS2"].iloc[:21].isna().all()
    assert cambio["DGS2"].iloc[30] == pytest.approx(21.0)  # 21 days x 1 bp
    futuro = curva.copy()
    futuro.iloc[31:] = 99.0  # rewrite everything after day 30
    assert cambio_pasado(futuro, dias=21)["DGS2"].iloc[30] == pytest.approx(21.0)


def test_signature_is_the_median_trailing_change_per_phase_and_tenor() -> None:
    curva = _curva({"DGS2": np.arange(60) * 0.01, "DGS10": np.arange(60) * -0.02})
    fases = pd.Series([0] * 30 + [2] * 30, index=DIAS)
    firma = firma_por_fase(curva, fases, NOMBRES, dias=21)
    assert list(firma.index) == list(NOMBRES)
    assert list(firma.columns) == ["DGS2", "DGS10"]
    assert firma.loc["rally fuerte", "DGS2"] == pytest.approx(21.0)
    assert firma.loc["venta", "DGS10"] == pytest.approx(-42.0)
    assert firma.loc["rally moderado"].isna().all()  # no day in that phase


def test_surrogate_phase_and_agreement_over_the_last_days() -> None:
    historia = pd.DataFrame(
        {
            "fase": [0, 0, 2, 2],
            "p_rally_fuerte": [0.9, 0.2, 0.1, 0.1],
            "p_rally_moderado": [0.05, 0.7, 0.1, 0.1],
            "p_venta": [0.05, 0.1, 0.8, 0.8],
        },
        index=DIAS[:4],
    )
    assert fase_imitador(historia, NOMBRES).tolist() == [0, 1, 2, 2]
    assert acuerdo(historia, NOMBRES, dias=4) == pytest.approx(0.75)
    assert acuerdo(historia, NOMBRES, dias=2) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("d2", "d10", "esperado"),
    [
        (137.0, 116.0, Cuadrante.BEAR_FLATTENER),
        (20.0, 60.0, Cuadrante.BEAR_STEEPENER),
        (-30.0, -50.0, Cuadrante.BULL_FLATTENER),
        (-80.0, -20.0, Cuadrante.BULL_STEEPENER),
        (10.0, 10.5, Cuadrante.PARALELO),
        (15.0, -10.0, Cuadrante.MIXTO),
    ],
)
def test_quadrant_of_an_episode(d2: float, d10: float, esperado: Cuadrante) -> None:
    assert cuadrante(d2, d10) is esperado


def test_past_curves_count_rows_back_from_the_last_day_on_or_before_the_date() -> None:
    curva = _curva({"DGS2": np.arange(60, dtype=float), "DGS10": np.zeros(60)})
    sabado = DIAS[40] + pd.Timedelta(days=(5 - DIAS[40].dayofweek) % 7 or 7)
    hasta = curva.loc[:sabado]
    curvas = curvas_pasadas(curva, sabado)
    assert list(curvas.index) == ["hoy", "hace 1m"]  # no 63 or 252 rows behind
    assert curvas.loc["hoy", "DGS2"] == hasta["DGS2"].iloc[-1]
    assert curvas.loc["hace 1m", "DGS2"] == hasta["DGS2"].iloc[-22]


def test_duration_position_uses_finished_episodes_only() -> None:
    episodios = pd.DataFrame(
        {"fase": [2, 0, 2, 1, 2, 2, 2], "dias": [40, 10, 80, 30, 120, 160, 500]}
    )  # the last row is the open, current episode: its 500 days must not count
    posicion = posicion_duracion(episodios, fase=2, dias=100)
    assert posicion.episodios == 4  # 40, 80, 120, 160
    assert posicion.mediana == pytest.approx(100.0)
    assert posicion.p25 == pytest.approx(70.0)
    assert posicion.p75 == pytest.approx(130.0)
    assert posicion.relativa == "dentro"
    assert posicion_duracion(episodios, fase=2, dias=200).relativa == "por encima"
    assert posicion_duracion(episodios, fase=2, dias=20).relativa == "por debajo"
    vacia = posicion_duracion(episodios, fase=1, dias=5)
    assert vacia.episodios == 1 and vacia.relativa == "por debajo"
    sin_historia = posicion_duracion(episodios.iloc[:2], fase=0, dias=5)
    assert sin_historia.episodios == 0 and sin_historia.mediana is None
    assert sin_historia.relativa is None


def test_dominant_block_is_the_largest_mean_since_a_date() -> None:
    historia = pd.DataFrame(
        {"nivel medio": [5.0, 0.0, 0.1, 0.1], "pendientes": [0.0, 1.0, 0.5, 0.5]},
        index=DIAS[:4],
    )
    assert bloque_dominante(historia, ("nivel medio", "pendientes"), DIAS[0]) == (
        "nivel medio",
        pytest.approx(1.3),
    )
    nombre, media = bloque_dominante(historia, ("nivel medio", "pendientes"), DIAS[1])
    assert nombre == "pendientes" and media == pytest.approx(2.0 / 3.0)


def test_macro_band_is_trailing_over_known_values() -> None:
    serie = pd.Series(np.arange(10, dtype=float), index=DIAS[:10])
    serie.iloc[3] = np.nan
    banda = banda_macro(serie, ventana=4, minimo=4)
    assert list(banda.columns) == ["p10", "p90"]
    assert DIAS[3] not in banda.index  # holes are not filled
    assert banda["p10"].iloc[:3].isna().all()
    # known values 0,1,2,4 at DIAS[4]: linear quantiles
    assert banda.loc[DIAS[4], "p10"] == pytest.approx(0.3)
    assert banda.loc[DIAS[4], "p90"] == pytest.approx(3.4)
