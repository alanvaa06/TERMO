"""The whole page: self-contained, Plotly once, every block, escaped, comment optional."""

from __future__ import annotations

import html
import re
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import make_desc2_config
from termo.operation.config import load_operation_config
from termo.operation.sheet_es import FIDELITY_WARNING
from termo.operation.visual.calculos import Posicion
from termo.operation.visual.datos import DatosReporte
from termo.operation.visual.narrativa import titulares
from termo.operation.visual.pagina import (
    ANCLAS,
    ARCHIVO,
    _num,
    _termometro,
    construir,
    render,
)
from visual_fixture import OP_CONFIG, PLOTLY_INICIO, en_disco, make_datos, sin_plotlyjs

RECURSO_EXTERNO = re.compile(r"""(?:src|href)\s*=\s*["']?https?:""", re.IGNORECASE)
GENERADO = "1992-09-01T10:00:00+00:00"


@pytest.fixture(scope="module")
def datos() -> DatosReporte:
    return make_datos()


@pytest.fixture(scope="module")
def page(datos: DatosReporte) -> str:
    return render(datos, GENERADO)


def _seccion(page: str, ancla: str) -> str:
    inicio = page.index(f'<section id="{ancla}"')
    return page[inicio : page.index("</section>", inicio)]


def test_a_self_contained_spanish_page(page: str) -> None:
    assert page.lower().startswith("<!doctype html>")
    assert '<html lang="es"' in page and '<meta charset="utf-8">' in page
    assert page.count(PLOTLY_INICIO) == 1
    assert not RECURSO_EXTERNO.search(sin_plotlyjs(page))
    assert "prefers-color-scheme: dark" in page and "@media print" in page


def test_every_block_is_there_once_and_in_the_index(page: str) -> None:
    for ancla in ANCLAS:
        assert page.count(f'<section id="{ancla}"') == 1, ancla
        assert f'href="#{ancla}"' in page
    assert 'id="comentario"' not in page


def test_charts_are_numbered_and_sourced(page: str) -> None:
    numeros = [int(n) for n in re.findall(r"<figcaption>(\d+)\. ", page)]
    assert numeros == list(range(1, len(numeros) + 1)) and len(numeros) >= 12
    assert page.count('class="fuente"') == len(numeros)
    assert page.count("Frecuencias del pasado, no pronóstico.") >= 3


def test_the_cover_and_the_validation(datos: DatosReporte, page: str) -> None:
    assert "<h1>Venta</h1>" in page
    assert "La curva está en venta" in page
    assert datos.textos["descargo"] in page
    assert "Nombres de fase elegidos tras el holdout; los valida solo la sombra." in page
    assert str(datos.hoja["snapshot_hash"])[:12] in page
    assert GENERADO in page


def test_text_is_escaped(datos: DatosReporte) -> None:
    textos = {**datos.textos, "descargo": "a <b> & c"}
    page = render(replace(datos, textos=textos), GENERADO)
    assert "a &lt;b&gt; &amp; c" in page and "a <b> & c" not in page


def test_the_analyst_comment_appears_only_when_written(datos: DatosReporte) -> None:
    page = render(replace(datos, comentario="## Lectura\n\nLa curva <sube>."), GENERADO)
    assert page.count('<section id="comentario"') == 1
    comentario = _seccion(page, "comentario")
    assert "<h2>Comentario del analista</h2>" in comentario
    assert "(no generado por TERMO)" in comentario
    assert "<h2>Lectura</h2>" in page and "La curva &lt;sube&gt;." in page


def test_unvalidated_drivers_carry_the_fidelity_warning(
    datos: DatosReporte, page: str
) -> None:
    assert FIDELITY_WARNING not in page
    lectura = {**datos.lectura, "drivers_validated": False}
    fallida = render(replace(datos, hoja={**datos.hoja, "reading": lectura}), GENERADO)
    motores = _seccion(fallida, "motores")
    assert motores.index(FIDELITY_WARNING) < motores.index("<figure")
    assert fallida.count(FIDELITY_WARNING) == 1


def test_build_writes_the_report_next_to_the_files(tmp_path: Path) -> None:
    _, salida, snapshot = en_disco(tmp_path)
    path = construir(
        salida, snapshot, make_desc2_config(), load_operation_config(OP_CONFIG), GENERADO
    )
    assert path == salida / ARCHIVO
    raw = path.read_bytes()
    assert b"\r\n" not in raw and raw.decode("utf-8").count(PLOTLY_INICIO) == 1


def test_every_headline_is_on_the_page(datos: DatosReporte, page: str) -> None:
    for clave, titular in titulares(datos).items():
        assert html.escape(titular.texto, quote=True) in page, clave
    assert '<p class="bajada">' in page  # the signature chart's headline


def test_accent_text_is_dark_enough_on_the_light_paper(page: str) -> None:
    # 'venta': the base hue on the dark paper, the darker tone on the light one
    assert '--acento:#e34948;--acento-texto-claro:#cc4241"' in page
    aviso = re.search(r"\.aviso\{[^}]*\}", page)
    assert aviso and "color:var(--tinta)" in aviso.group(0)
    assert "border:1px solid var(--acento-texto)" in aviso.group(0)
    assert re.search(r"h1\{[^}]*color:var\(--acento-texto\)", page)


def test_the_page_script_recolours_every_y_axis_and_follows_x(page: str) -> None:
    assert r"/^yaxis\d*$/" in page  # yaxis, yaxis2, yaxis3 ... all recoloured
    assert "addListener" in page  # older Safari has no addEventListener on MediaQueryList
    assert "plotly_relayout" in page and "ajustar_y" in page
    assert "beforeprint" in page and "afterprint" in page and "Plotly.Plots.resize" in page


def test_print_uses_the_light_palette_without_buttons(page: str) -> None:
    impresion = page[page.index("@media print") :]
    assert "--papel:#fbfaf7" in impresion and ".rangeselector{display:none}" in impresion
    assert "print-color-adjust:exact" in page


def test_navigation_and_mobile_anchors(page: str) -> None:
    assert '<nav aria-label="Índice">' in page
    assert "scroll-margin-top:48px" in page


def test_cover_notices(datos: DatosReporte) -> None:
    lectura = {**datos.lectura, "low_confidence": True}
    alerta = {"fecha": "1992-08-28", "de": "rally <fuerte>", "a": "venta", "confianza": 0.7}
    page = render(replace(datos, hoja={**datos.hoja, "reading": lectura, "alert": alerta}),
                  GENERADO)
    portada = _seccion(page, "portada")
    assert '<span class="aviso">confianza baja</span>' in portada
    assert "cambio de fase: rally &lt;fuerte&gt; a venta" in portada


def test_validation_escapes_at_the_sink(datos: DatosReporte) -> None:
    validacion = {
        **datos.lectura["validation"],
        "registered_verdicts": [{"stage": "<x>", "verdict": "apto & listo"}],
    }
    lectura = {**datos.lectura, "validation": validacion}
    page = render(replace(datos, hoja={**datos.hoja, "reading": lectura}), GENERADO)
    caja = _seccion(page, "validacion")
    assert "&lt;x&gt;: APTO &amp; LISTO" in caja and "<x>" not in caja


def test_missing_statistics_show_as_a_dash() -> None:
    assert _num(None) == "-" and _num(0.5) == "0.50" and _num(49.6, 0) == "50"


def _izquierda(termometro: str, clase: str) -> float:
    encontrado = re.search(rf'class="{clase}" style="left:([\d.]+)%', termometro)
    assert encontrado, clase
    return float(encontrado.group(1))


def test_thermometer_without_history() -> None:
    texto = _termometro(Posicion(40, 0, None, None, None))
    assert "Sin episodios terminados" in texto and 'class="termometro"' not in texto


def test_thermometer_with_few_episodes_shows_only_the_median() -> None:
    texto = _termometro(Posicion(40, 1, 50.0, 50.0, 50.0))
    assert 'class="banda"' not in texto and 'class="mediana"' in texto
    assert "1 episodio terminado de" in texto
    assert "menos de 4" in texto
    tres = _termometro(Posicion(40, 3, 30.0, 50.0, 70.0))
    assert "3 episodios terminados" in tres and 'class="banda"' not in tres


def test_thermometer_positions() -> None:
    debajo = _termometro(Posicion(10, 6, 30.0, 50.0, 70.0))
    assert _izquierda(debajo, "hoy") < _izquierda(debajo, "banda")
    assert "6 episodios terminados" in debajo
    # the right end of the scale is its maximum, not today's count
    assert debajo.endswith("<span>80 días</span></div></div>")  # 70 * 1.15
    encima = _termometro(Posicion(100, 6, 30.0, 50.0, 70.0))
    banda = _izquierda(encima, "banda")
    ancho = float(re.search(r"width:([\d.]+)%", encima).group(1))  # type: ignore[union-attr]
    assert _izquierda(encima, "hoy") > banda + ancho
    assert _izquierda(encima, "hoy") <= 100.0
    assert "<span>115 días</span>" in encima
