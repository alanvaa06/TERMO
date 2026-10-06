"""Descriptive numbers the visual report adds (spec 5, section 5).

A value assigned to a day uses only that day and earlier ones. The pooled frequencies
(signature, durations) describe the whole past and the report labels them as such.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from termo.validation.metrics import BP_PER_PERCENT

DIAS_MES = 21
DIAS_ANIO = 252
TOLERANCIA_PARALELO_PB = 1.0
ATRAS: dict[str, int] = {"hoy": 0, "hace 1m": 21, "hace 3m": 63, "hace 1a": 252}


class Cuadrante(StrEnum):
    BEAR_FLATTENER = "bear flattener"
    BEAR_STEEPENER = "bear steepener"
    BULL_FLATTENER = "bull flattener"
    BULL_STEEPENER = "bull steepener"
    PARALELO = "paralelo"
    MIXTO = "mixto"


@dataclass(frozen=True)
class Posicion:
    """The current episode's length against the finished episodes of its phase."""

    dias: int
    episodios: int
    p25: float | None
    mediana: float | None
    p75: float | None

    @property
    def relativa(self) -> str | None:
        if self.p25 is None or self.p75 is None:
            return None
        if self.dias > self.p75:
            return "por encima"
        if self.dias < self.p25:
            return "por debajo"
        return "dentro"


def columna_probabilidad(nombre: str) -> str:
    """'rally fuerte' -> 'p_rally_fuerte', as the CSV export names it."""
    return "p_" + nombre.replace(" ", "_")


def cambio_pasado(curva: pd.DataFrame, dias: int = DIAS_MES) -> pd.DataFrame:
    """Change in bp over the `dias` rows that END on each row; NaN for the first rows."""
    return (curva - curva.shift(dias)) * BP_PER_PERCENT


def firma_por_fase(
    curva: pd.DataFrame, fases: pd.Series, nombres: Sequence[str], dias: int = DIAS_MES
) -> pd.DataFrame:
    """Median trailing change per phase (rows, in `nombres` order) and tenor (columns)."""
    cambio = cambio_pasado(curva, dias)
    cambio["_fase"] = fases.reindex(cambio.index).to_numpy()
    juntos = cambio.dropna()
    juntos = juntos.astype({"_fase": int})
    mediana = juntos.groupby("_fase").median().reindex(range(len(nombres)))
    mediana.index = pd.Index(list(nombres), name="fase")
    return mediana


def fase_imitador(historia: pd.DataFrame, nombres: Sequence[str]) -> pd.Series:
    """The surrogate's most probable phase, day by day."""
    probas = historia[[columna_probabilidad(n) for n in nombres]].to_numpy()
    return pd.Series(probas.argmax(axis=1), index=historia.index, name="fase_imitador")


def acuerdo(historia: pd.DataFrame, nombres: Sequence[str], dias: int = DIAS_ANIO) -> float:
    """Share of the last `dias` rows on which the surrogate names the jump model's phase."""
    ventana = historia.tail(dias)
    return float((fase_imitador(ventana, nombres) == ventana["fase"]).mean())


def cuadrante(
    cambio_2y: float, cambio_10y: float, tolerancia: float = TOLERANCIA_PARALELO_PB
) -> Cuadrante:
    """Bear/bull by the sign of both moves; flattener when the 2Y moved up more."""
    if abs(cambio_2y - cambio_10y) < tolerancia:
        return Cuadrante.PARALELO
    aplana = cambio_2y > cambio_10y
    if cambio_2y > 0 and cambio_10y > 0:
        return Cuadrante.BEAR_FLATTENER if aplana else Cuadrante.BEAR_STEEPENER
    if cambio_2y < 0 and cambio_10y < 0:
        return Cuadrante.BULL_FLATTENER if aplana else Cuadrante.BULL_STEEPENER
    return Cuadrante.MIXTO


def cuadrantes(episodios: pd.DataFrame) -> pd.Series:
    """The quadrant of every episode, from its 2Y and 10Y change inside the episode."""
    return pd.Series(
        [
            cuadrante(float(d2), float(d10))
            for d2, d10 in zip(episodios["cambio_2y_pb"], episodios["cambio_10y_pb"], strict=True)
        ],
        index=episodios.index,
        name="cuadrante",
    )


def curvas_pasadas(curva: pd.DataFrame, fecha: pd.Timestamp) -> pd.DataFrame:
    """The curve on the last row on or before `fecha` and 21, 63 and 252 rows earlier."""
    hasta = curva.loc[:fecha]
    ultimo = len(hasta) - 1
    filas = {
        etiqueta: hasta.iloc[ultimo - atras]
        for etiqueta, atras in ATRAS.items()
        if ultimo - atras >= 0
    }
    return pd.DataFrame(filas).T


def posicion_duracion(episodios: pd.DataFrame, fase: int, dias: int) -> Posicion:
    """The last episode is the current one, still open: only the others are the past."""
    terminados = episodios.iloc[:-1]
    largos = terminados.loc[terminados["fase"] == fase, "dias"]
    if largos.empty:
        return Posicion(dias, 0, None, None, None)
    return Posicion(
        dias=dias,
        episodios=int(len(largos)),
        p25=float(largos.quantile(0.25)),
        mediana=float(largos.median()),
        p75=float(largos.quantile(0.75)),
    )


def bloque_dominante(
    historia: pd.DataFrame, bloques: Sequence[str], desde: pd.Timestamp
) -> tuple[str, float]:
    """The block with the largest mean contribution from `desde` to the end of `historia`."""
    medias = historia.loc[desde:, list(bloques)].mean()
    nombre = str(medias.idxmax())
    return nombre, float(medias[nombre])


def banda_macro(serie: pd.Series, ventana: int, minimo: int = DIAS_ANIO) -> pd.DataFrame:
    """Trailing 10th and 90th percentiles over the last `ventana` known values."""
    conocida = serie.dropna()
    rodante = conocida.rolling(ventana, min_periods=minimo)
    return pd.DataFrame({"p10": rodante.quantile(0.1), "p90": rodante.quantile(0.9)})
