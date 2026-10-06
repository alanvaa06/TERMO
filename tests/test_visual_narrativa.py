"""Headlines and bullets: exact texts from closed templates, never future or conditional."""

from __future__ import annotations

import re

import pytest

from termo.operation.visual.calculos import Cuadrante, Posicion
from termo.operation.visual.narrativa import (
    CLAVES,
    titular_10a,
    titular_acuerdo,
    titular_curva,
    titular_episodios,
    titular_firma,
    titular_macro,
    titular_motor,
    titular_motores_tiempo,
    titular_portada,
    titular_transicion,
    titulares,
)
from visual_fixture import make_datos

PROHIBIDO = re.compile(
    r"\b\w+(?:rá|rán|ría|rían)\b|\b(?:pronto|esperamos|probable|anticipa)\b", re.IGNORECASE
)


def test_cover_headline_places_the_episode_against_the_quartiles() -> None:
    posicion = Posicion(dias=199, episodios=55, p25=41.0, mediana=85.0, p75=114.0)
    assert titular_portada("venta", posicion) == (
        "La curva está en venta: 199 días, por encima del rango intercuartil histórico"
    )
    sin_pasado = Posicion(dias=12, episodios=0, p25=None, mediana=None, p75=None)
    assert titular_portada("rally fuerte", sin_pasado) == "La curva está en rally fuerte: 12 días"


def test_ten_year_headline() -> None:
    assert titular_10a("venta", "2025-12-18", 116.0) == (
        "El 10A subió 116 pb desde que empezó el episodio de venta (2025-12-18)"
    )
    assert titular_10a("rally moderado", "2025-03-05", -12.4) == (
        "El 10A bajó 12 pb desde que empezó el episodio de rally moderado (2025-03-05)"
    )
    assert titular_10a("venta", "2025-12-18", 0.3) == (
        "El 10A no cambió desde que empezó el episodio de venta (2025-12-18)"
    )


def test_agreement_driver_and_dominant_block() -> None:
    assert titular_acuerdo(0.934) == (
        "Imitador y fase coinciden en el 93% de los días del último año"
    )
    assert titular_motor("nivel medio", 2.6979) == (
        "Movimiento tramo medio (5A-7A) es el bloque de mayor aporte a la lectura (+2.70)"
    )
    assert titular_motores_tiempo("pendientes", "2025-12-18") == (
        "Desde 2025-12-18, el bloque de mayor aporte medio es pendientes"
    )


def test_curve_headline_names_the_reshaping() -> None:
    assert titular_curva(40.0, 12.0) == (
        "En 3 meses el 2A subió 40 pb y el 10A subió 12 pb: la curva se aplanó"
    )
    assert titular_curva(-30.0, -5.0) == (
        "En 3 meses el 2A bajó 30 pb y el 10A bajó 5 pb: la curva se empinó"
    )
    assert titular_curva(5.0, 5.4) == (
        "En 3 meses el 2A subió 5 pb y el 10A subió 5 pb: la curva se movió en paralelo"
    )


def test_signature_episodes_transition_and_macro() -> None:
    assert titular_firma("venta", "5A", 18.4) == (
        "En venta, el plazo que más se mueve en un mes (mediana) es el 5A (+18 pb)"
    )
    assert titular_episodios("venta", Cuadrante.BEAR_FLATTENER, 31, 55) == (
        "31 de 55 episodios de venta fueron bear flattener"
    )
    assert titular_transicion("venta", "rally moderado", 0.87, 54) == (
        "Tras venta, el 87% de los episodios siguió con rally moderado"
    )
    assert titular_transicion("rally fuerte", None, None, 0) == (
        "No hay episodios de rally fuerte con sucesor"
    )
    assert titular_macro("prima por plazo 10 anos (Kim-Wright)", 0.9996) == (
        "Prima por plazo 10 anos (Kim-Wright) está en el percentil 100 a 10 años"
    )


def test_every_block_gets_a_headline_and_the_cover_three_bullets() -> None:
    textos = titulares(make_datos())
    assert set(textos) == set(CLAVES)
    assert len(textos["portada"].vinetas) == 3
    assert textos["portada"].texto.startswith("La curva está en venta: ")
    assert textos["episodios"].texto.endswith(
        tuple(f"fueron {c.value}" for c in Cuadrante)
    )


def test_no_generated_text_speaks_of_the_future() -> None:
    textos = titulares(make_datos())
    for titular in textos.values():
        for frase in (titular.texto, *titular.vinetas):
            assert not PROHIBIDO.search(frase), frase


@pytest.mark.parametrize("frase", ["La tasa subirá", "podría bajar", "pronto cambia"])
def test_the_forbidden_pattern_catches_future_and_conditional(frase: str) -> None:
    assert PROHIBIDO.search(frase)
