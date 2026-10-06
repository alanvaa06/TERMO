"""Headlines and bullets: exact texts from closed templates, never future or conditional."""

from __future__ import annotations

import dataclasses
import re
from collections import Counter

import pytest

from termo.operation.visual.calculos import Cuadrante, Posicion
from termo.operation.visual.datos import DatosReporte
from termo.operation.visual.etiquetas import bloque_es
from termo.operation.visual.narrativa import (
    CLAVES,
    CURVA_CORTA,
    NO_VALIDADO,
    PLAZOS,
    Titular,
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

PERMITIDAS = re.compile(r"\b(?:mayoría|Tesorería|categoría)\b", re.IGNORECASE)
PROHIBIDO = re.compile(
    r"\b\w+(?:rá|rán|rás|remos|réis|ría|rían|rías|ríamos)\b"
    r"|\bva(?:n)? a \w+r\b"
    r"|\b(?:pronto|esperamos|puede|pueden|podr\w*|tiende\w*|espera\w*|prev\w*|anticip\w*"
    r"|probab\w*|pr[oó]xim\w*|futur\w*|habr\w*|ir[aá]\w*)\b",
    re.IGNORECASE,
)


def _habla_del_futuro(frase: str) -> bool:
    return PROHIBIDO.search(PERMITIDAS.sub("", frase)) is not None


def _sin_validar(datos: DatosReporte) -> DatosReporte:
    lectura = {**datos.lectura, "drivers_validated": False}
    return dataclasses.replace(datos, hoja={**datos.hoja, "reading": lectura})


def _frases(textos: dict[str, Titular]) -> list[str]:
    return [frase for t in textos.values() for frase in (t.texto, *t.vinetas)]


def _a(*proporciones: float | None) -> dict[str, float | None]:
    """Transition shares in phase order, as transitions() gives them."""
    return dict(zip(("rally fuerte", "rally moderado", "venta"), proporciones, strict=True))


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
        "Movimiento tramo medio (5A-7A) es el bloque que más pesa en la lectura (+2.70)"
    )
    assert titular_motor("nivel largo", -1.1722) == (
        "Movimiento tramo largo (10A-30A) es el bloque que más pesa en la lectura (-1.17)"
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
    # a 1 bp gap with float noise is a reshaping, as in the quadrants
    assert titular_curva((4.56 - 4.53) * 100, (3.22 - 3.20) * 100).endswith("se aplanó")


def test_signature_and_macro() -> None:
    assert titular_firma("venta", "5A", 18.4) == (
        "En venta, el plazo que más se mueve en un mes (mediana) es el 5A (+18 pb)"
    )
    assert titular_firma("rally fuerte", None, None) == (
        "Sin cambios a 1 mes registrados para rally fuerte"
    )
    assert titular_macro("prima por plazo 10 anos (Kim-Wright)", 0.9996) == (
        "Prima por plazo 10 anos (Kim-Wright) está en el percentil 100 a 10 años"
    )


def test_episodes_headline_counts_closed_episodes() -> None:
    assert titular_episodios("venta", Cuadrante.BEAR_FLATTENER, 31, 55) == (
        "31 de 55 episodios de venta ya cerrados fueron bear flattener"
    )
    assert titular_episodios("venta", Cuadrante.MIXTO, 1, 1) == (
        "El único episodio de venta ya cerrado fue mixto"
    )
    assert titular_episodios("rally fuerte", None, 0, 0) == (
        "Es el primer episodio de rally fuerte en la historia"
    )


def test_transition_headline_is_a_past_frequency_with_counts() -> None:
    assert titular_transicion("venta", 54, _a(7 / 54, 47 / 54, 0.0)) == (
        "De 54 episodios de venta ya cerrados, 47 (87%) dieron paso a rally moderado"
    )
    assert titular_transicion("venta", 3, _a(1 / 3, 2 / 3, 0.0)) == (
        "De 3 episodios de venta ya cerrados, 2 (67%) dieron paso a rally moderado"
    )
    assert titular_transicion("rally fuerte", 1, _a(0.0, 0.0, 1.0)) == (
        "El único episodio de rally fuerte ya cerrado dio paso a venta"
    )
    assert titular_transicion("rally fuerte", 0, _a(None, None, None)) == (
        "No hay episodios de rally fuerte ya cerrados"
    )


def test_a_tie_in_transitions_names_every_tied_phase_in_phase_order() -> None:
    assert titular_transicion("venta", 2, _a(0.5, 0.5, 0.0)) == (
        "De 2 episodios de venta ya cerrados, 1 (50%) dio paso a rally fuerte "
        "y 1 (50%) a rally moderado"
    )
    assert titular_transicion("venta", 4, _a(0.5, 0.5, 0.0)) == (
        "De 4 episodios de venta ya cerrados, 2 (50%) dieron paso a rally fuerte "
        "y 2 (50%) a rally moderado"
    )


def test_every_block_gets_a_headline_and_the_cover_three_bullets() -> None:
    textos = titulares(make_datos())
    assert set(textos) == set(CLAVES)
    assert len(textos["portada"].vinetas) == 3
    assert textos["portada"].texto.startswith("La curva está en venta: ")
    assert textos["episodios"].texto.endswith(
        tuple(f"ya cerrados fueron {c.value}" for c in Cuadrante)
    )


# --- integration: every pick recomputed independently on the synthetic week ---


def test_main_driver_is_the_first_driver_of_the_sheet() -> None:
    datos = make_datos()
    primero = datos.lectura["drivers"][0]
    fila = datos.historia.iloc[-1]
    assert primero["block"] == max(datos.bloques, key=lambda b: abs(float(fila[b])))
    esperado = titular_motor(primero["block"], float(primero["contribution"]))
    textos = titulares(datos)
    assert textos["motores"].texto == esperado
    assert textos["portada"].vinetas[0] == esperado
    assert bloque_es(str(primero["block"]))[1:] in esperado


def test_signature_tenor_is_the_largest_absolute_median_of_the_phase() -> None:
    datos = make_datos()
    dias_venta = datos.historia.index[datos.historia["fase"] == 2]
    mediana = (datos.curva.diff(21) * 100.0).loc[dias_venta].median()
    plazo = str(mediana.abs().idxmax())
    assert titulares(datos)["firma"].texto == titular_firma(
        "venta", PLAZOS[plazo], float(mediana[plazo])
    )


def test_macro_bullet_is_the_series_farthest_from_the_median() -> None:
    datos = make_datos()
    lejana = sorted(datos.hoja["macro"], key=lambda m: -abs(m["percentil_10a"] - 0.5))[0]
    esperado = titular_macro(str(lejana["serie"]), float(lejana["percentil_10a"]))
    textos = titulares(datos)
    assert textos["macro"].texto == esperado
    assert textos["portada"].vinetas[2] == esperado


def test_transition_targets_are_the_largest_share_of_closed_venta_episodes() -> None:
    datos = make_datos()
    fases = datos.episodios["fase"].tolist()
    destinos = Counter(b for a, b in zip(fases[:-1], fases[1:], strict=True) if a == 2)
    maximo = max(destinos.values())
    ganadores = [datos.nombres[f] for f in sorted(destinos) if destinos[f] == maximo]
    texto = titulares(datos)["transiciones"].texto
    total = sum(destinos.values())
    assert texto.startswith(f"De {total} episodios de venta ya cerrados, {maximo} ")
    for nombre in ganadores:
        assert f"a {nombre}" in texto
    # the synthetic week: venta -> rally moderado, then venta -> rally fuerte
    assert texto == (
        "De 2 episodios de venta ya cerrados, 1 (50%) dio paso a rally fuerte "
        "y 1 (50%) a rally moderado"
    )


def test_episodes_headline_ignores_the_open_episode() -> None:
    datos = make_datos()
    cerrados = datos.episodios.iloc[:-1]
    assert int((cerrados["fase"] == 2).sum()) == 2
    # closed venta episodes: 2Y +90.54/10Y +82.83 (bear flattener) and 2Y -183.96/10Y
    # -185.33 (bull flattener); the open one does not count
    assert titulares(datos)["episodios"].texto == (
        "1 de 2 episodios de venta ya cerrados fueron bear flattener"
    )
    solo_actual = dataclasses.replace(datos, episodios=datos.episodios.iloc[-1:])
    textos = titulares(solo_actual)
    assert textos["episodios"].texto == "Es el primer episodio de venta en la historia"
    assert textos["transiciones"].texto == "No hay episodios de venta ya cerrados"


def test_drivers_that_failed_fidelity_are_not_presented_as_validated() -> None:
    textos = titulares(_sin_validar(make_datos()))
    assert textos["portada"].vinetas[0] == NO_VALIDADO
    assert textos["motores"].texto == NO_VALIDADO
    assert textos["motores-tiempo"].texto == NO_VALIDADO
    assert NO_VALIDADO == (
        "Las explicaciones de esta semana no pasaron la validación de fidelidad del imitador"
    )


def test_a_short_curve_and_a_curve_without_one_month_changes() -> None:
    datos = make_datos()
    corta = titulares(dataclasses.replace(datos, curva=datos.curva.tail(50)))
    assert corta["curva"].texto == CURVA_CORTA == "La curva tiene menos de 3 meses de historia"
    assert corta["firma"].texto.startswith("En venta, el plazo que más se mueve")
    vacia = titulares(dataclasses.replace(datos, curva=datos.curva.tail(15)))
    assert vacia["firma"].texto == "Sin cambios a 1 mes registrados para venta"


# --- no sentence speaks of the future, in any branch ---


def _todas_las_frases() -> list[str]:
    datos = make_datos()
    frases = _frases(titulares(datos))
    frases += _frases(titulares(_sin_validar(datos)))
    frases += _frases(titulares(dataclasses.replace(datos, episodios=datos.episodios.iloc[-1:])))
    frases += _frases(titulares(dataclasses.replace(datos, curva=datos.curva.tail(50))))
    frases += _frases(titulares(dataclasses.replace(datos, curva=datos.curva.tail(15))))
    frases += [
        titular_transicion("venta", 2, _a(0.5, 0.5, 0.0)),
        titular_transicion("venta", 0, _a(None, None, None)),
        titular_episodios("venta", None, 0, 0),
        titular_episodios("venta", Cuadrante.MIXTO, 1, 1),
        titular_transicion("venta", 1, _a(0.0, 1.0, 0.0)),
        titular_firma("venta", None, None),
        titular_portada("venta", Posicion(5, 0, None, None, None)),
        titular_curva(5.0, 5.4),
        titular_10a("venta", "2025-12-18", 0.0),
    ]
    return frases


def test_no_generated_text_speaks_of_the_future() -> None:
    for frase in _todas_las_frases():
        assert not _habla_del_futuro(frase), frase


@pytest.mark.parametrize(
    "frase",
    [
        "La tasa subirá",
        "podría bajar",
        "pronto cambia",
        "va a subir",
        "puede bajar",
        "se espera un alza",
        "probablemente siga",
        "habrá cambios",
        "irá al alza",
        "la próxima semana",
        "subiremos",
    ],
)
def test_the_forbidden_pattern_catches_future_and_conditional(frase: str) -> None:
    assert _habla_del_futuro(frase)


@pytest.mark.parametrize(
    "frase", ["La mayoría de los días", "bonos de la Tesorería", "una categoría de bloque"]
)
def test_the_forbidden_pattern_lets_legit_words_through(frase: str) -> None:
    assert PROHIBIDO.search(frase)  # they look like a conditional...
    assert not _habla_del_futuro(frase)  # ...and the allow-list lets them pass
