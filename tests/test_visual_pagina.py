"""The whole page: self-contained, Plotly once, every block, escaped, comment optional."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import make_desc2_config
from termo.operation.config import load_operation_config
from termo.operation.sheet_es import FIDELITY_WARNING
from termo.operation.visual.datos import DatosReporte
from termo.operation.visual.pagina import ANCLAS, ARCHIVO, construir, render
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
    assert "Comentario del analista (no generado por TERMO)" in page
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
