"""One Plotly figure per chart: traces, phase colours, data lengths, initial view."""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable, Iterable, Iterator

import pandas as pd
import plotly.graph_objects as go
import pytest

from termo.operation.visual import calculos
from termo.operation.visual import graficas as g
from termo.operation.visual.datos import DatosReporte
from termo.operation.visual.etiquetas import bloque_es
from visual_fixture import BLOQUES, NOMBRES, make_datos

PAPELES = ("#fbfaf7", "#161615")  # the page background, light and dark
FECHA_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")


@pytest.fixture(scope="module")
def datos() -> DatosReporte:
    return make_datos()


def _con_fase_final(datos: DatosReporte, fase: int) -> DatosReporte:
    """The fixture with the reading day's phase changed (the sheet still says 'venta')."""
    historia = datos.historia.copy()
    historia.iloc[-1, historia.columns.get_loc("fase")] = fase
    return dataclasses.replace(datos, historia=historia)


def _luminancia(color: str) -> float:
    canales = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lineal = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in canales]
    return 0.2126 * lineal[0] + 0.7152 * lineal[1] + 0.0722 * lineal[2]


def _contraste(a: str, b: str) -> float:
    claro, oscuro = sorted((_luminancia(a), _luminancia(b)), reverse=True)
    return (claro + 0.05) / (oscuro + 0.05)


def _legible(color: str) -> bool:
    """WCAG 3:1 for graphical marks, on both the light and the dark page."""
    return all(_contraste(color, papel) >= 3.0 for papel in PAPELES)


def _semanas(datos: DatosReporte) -> pd.DatetimeIndex:
    """The last labelled day of each W-FRI week, up to and including the reading date."""
    dias = datos.historia.loc[: datos.fecha].index
    ultimo = ~dias.to_period("W-FRI").duplicated(keep="last")
    return pd.DatetimeIndex(dias[ultimo])


def _transparent(fig: go.Figure) -> bool:
    tema = fig.layout.template.layout
    return tema.paper_bgcolor == "rgba(0,0,0,0)" and tema.plot_bgcolor == "rgba(0,0,0,0)"


def test_phase_colours_are_fixed_by_phase_number() -> None:
    assert [g.color_fase(i) for i in range(3)] == ["#2a78d6", "#1baf7a", "#e34948"]


def test_ten_year_by_phase(datos: DatosReporte) -> None:
    fig = g.fig_10a(datos)
    assert _transparent(fig)
    assert [t.name for t in fig.data] == list(NOMBRES)
    assert [t.line.color for t in fig.data] == [g.color_fase(i) for i in range(3)]
    n = len(datos.historia)
    assert all(len(t.x) == n and len(t.y) == n for t in fig.data)
    # every day is drawn by the trace of its phase
    for fase, valor in zip(datos.historia["fase"], zip(*(t.y for t in fig.data), strict=True),
                           strict=True):
        assert not math.isnan(valor[fase])
    inicio = (datos.fecha - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    assert list(fig.layout.xaxis.range) == [inicio, datos.fecha.strftime("%Y-%m-%d")]
    assert fig.layout.xaxis.rangeselector.buttons[-1].label == "todo"
    assert any(s.type == "rect" for s in fig.layout.shapes)  # the holdout band


def test_ten_year_reading_day_in_its_own_colour(datos: DatosReporte) -> None:
    # a segment takes the colour of the day it ends on: a phase that starts on the
    # reading day still draws the last segment
    fig = g.fig_10a(_con_fase_final(datos, 0))
    assert not any(math.isnan(v) for v in fig.data[0].y[-2:])


def test_surrogate_against_phase(datos: DatosReporte) -> None:
    fig = g.fig_imitador(datos)
    franjas = [t for t in fig.data if isinstance(t, go.Heatmap)]
    areas = [t for t in fig.data if isinstance(t, go.Scatter)]
    assert len(franjas) == 2 and len(areas) == 3
    hoy = datos.fecha.strftime("%Y-%m-%d")
    # the strips are daily: a one-day disagreement is never sampled away ...
    for franja in franjas:
        assert len(franja.x) == len(datos.historia) and franja.x[-1] == hoy
    assert list(franjas[0].z[0]) == datos.historia["fase"].tolist()
    imitador = calculos.fase_imitador(datos.historia, NOMBRES)
    assert list(franjas[1].z[0]) == imitador.tolist()
    assert list(franjas[0].z[0]) != list(franjas[1].z[0])  # the planted disagreements
    # ... the probability areas are weekly
    for area in areas:
        assert len(area.x) == len(_semanas(datos)) and area.x[-1] == hoy
    assert {t.stackgroup for t in areas} == {"p"}


def test_drivers_of_the_reading(datos: DatosReporte) -> None:
    fig = g.fig_motores_hoy(datos)
    (cascada,) = fig.data
    assert isinstance(cascada, go.Waterfall)
    fila = datos.historia.iloc[-1]
    assert list(cascada.y) == ["base", *(bloque_es(b) for b in BLOQUES), "lectura"]
    assert list(cascada.x[1:-1]) == pytest.approx([float(fila[b]) for b in BLOQUES])
    assert cascada.measure[0] == "absolute" and cascada.measure[-1] == "total"
    assert cascada.totals.marker.color != cascada.decreasing.marker.color


def test_drivers_of_the_reading_take_the_phase_of_their_row(datos: DatosReporte) -> None:
    fig = g.fig_motores_hoy(_con_fase_final(datos, 0))
    assert fig.data[0].increasing.marker.color == g.color_fase(0)
    assert fig.layout.xaxis.title.text == f"log-odds de {NOMBRES[0]}"


def test_top_variables_are_translated(datos: DatosReporte) -> None:
    (barras,) = g.fig_variables_hoy(datos).data
    assert barras.y[0] == "5A · cambio 3m · rango 1a"
    assert barras.marker.color[3] == g.GRIS  # the negative one


def test_drivers_over_time(datos: DatosReporte) -> None:
    fig = g.fig_motores_tiempo(datos)
    assert len(fig.data) == 2 * len(BLOQUES)
    assert sum(bool(t.showlegend) for t in fig.data) == len(BLOQUES)
    assert {t.stackgroup for t in fig.data} == {"pos", "neg"}
    assert len(fig.layout.shapes) >= len(datos.episodios)  # phase shading
    semanas = _semanas(datos)
    for traza in fig.data:
        assert len(traza.x) == len(semanas)
        assert traza.x[-1] == datos.fecha.strftime("%Y-%m-%d")


def test_drivers_over_time_parts_add_up_and_missing_stays_missing(datos: DatosReporte) -> None:
    historia = datos.historia.copy()
    semanas = _semanas(datos)
    historia.loc[semanas[-1], BLOQUES[0]] = float("nan")  # a missing SHAP value
    fig = g.fig_motores_tiempo(dataclasses.replace(datos, historia=historia))
    for posicion, bloque in enumerate(BLOQUES):
        pos, neg = fig.data[2 * posicion], fig.data[2 * posicion + 1]
        assert pos.stackgroup == "pos" and neg.stackgroup == "neg"
        esperado = historia[bloque].loc[semanas].fillna(0.0)
        suma = [a + b for a, b in zip(pos.y, neg.y, strict=True)]
        assert suma == pytest.approx(esperado.tolist(), abs=1e-3)
        assert len(pos.customdata) == len(semanas)
        assert neg.customdata is None and neg.hoverinfo == "skip"
    assert math.isnan(fig.data[0].customdata[-1])  # shown as missing, not +0.00


def test_curve_today_and_signature(datos: DatosReporte) -> None:
    curva = g.fig_curva(datos)
    assert [t.name for t in curva.data] == ["hoy", "hace 1m", "hace 3m", "hace 1a"]
    assert list(curva.data[0].x) == ["1A", "2A", "3A", "5A", "7A", "10A", "30A"]
    referencias = curva.data[1:]
    assert all(_legible(t.line.color) for t in referencias)
    assert len({(t.line.dash, t.line.width) for t in referencias}) == len(referencias)
    firma = g.fig_firma(datos)
    assert [t.name for t in firma.data] == list(NOMBRES)


def test_episodes_strip_and_scatter(datos: DatosReporte) -> None:
    (franja,) = g.fig_franja(datos).data
    assert len(franja.x) == len(datos.historia)
    fig = g.fig_dispersion(datos)
    assert [t.name for t in fig.data] == [*NOMBRES, "episodio actual"]
    actual = datos.episodios.iloc[-1]
    assert fig.data[-1].x[0] == pytest.approx(actual["cambio_2y_pb"])
    assert fig.data[-1].y[0] == pytest.approx(actual["cambio_10y_pb"])
    textos = {a.text for a in fig.layout.annotations}
    assert {"bear flattener", "bear steepener", "bull flattener", "bull steepener"} <= textos
    nombres_cuadrante = {c.value for c in calculos.Cuadrante}
    for nota in fig.layout.annotations:
        if nota.text in nombres_cuadrante:  # each label sits in its own quadrant
            assert calculos.cuadrante(nota.x, nota.y).value == nota.text


def test_transitions_and_durations(datos: DatosReporte) -> None:
    transiciones = g.fig_transiciones(datos)
    (mapa,) = transiciones.data
    assert len(mapa.z) == 3 and all(len(fila) == 3 for fila in mapa.z)
    # a neutral ramp (blue means 'rally fuerte' elsewhere) ...
    assert all(str(c).startswith("rgba(137,135,129,") for _, c in mapa.colorscale)
    # ... and the cell text as annotations, which follow the page's font colour
    assert mapa.texttemplate is None
    notas = transiciones.layout.annotations
    assert len(notas) == 9 and all(n.font.color is None for n in notas)
    fig = g.fig_duraciones(datos)
    assert [t.name for t in fig.data] == [*NOMBRES, "episodio actual"]
    assert fig.data[-1].y[0] == datos.episodios.iloc[-1]["dias"]


def test_macro_has_one_axis_and_a_band(datos: DatosReporte) -> None:
    for serie in datos.series_macro:
        fig = g.fig_macro(datos, serie)
        assert len(fig.data) == 3
        assert "yaxis2" not in fig.to_dict()["layout"]  # never a dual axis
        assert fig.data[1].fill == "tonexty"
        assert fig.data[1].name == "P10-P90 a 10 años"
    cinco = dataclasses.replace(datos, ventana_percentil=5 * calculos.DIAS_ANIO)
    assert g.fig_macro(cinco, datos.series_macro[0]).data[1].name == "P10-P90 a 5 años"


def test_marks_and_reference_lines_read_on_both_papers(datos: DatosReporte) -> None:
    cascada = g.fig_motores_hoy(datos)
    colores = [
        cascada.data[0].totals.marker.color,
        cascada.data[0].connector.line.color,
        cascada.layout.template.layout.xaxis.linecolor,  # every figure's x axis line
        g.fig_dispersion(datos).data[-1].marker.color,  # the current episode's ring
        g.fig_duraciones(datos).data[-1].marker.color,  # the current episode's diamond
    ]
    for fig in (g.fig_dispersion(datos), g.fig_firma(datos)):
        colores += [forma.line.color for forma in fig.layout.shapes]  # diagonal, zero lines
    assert len(colores) == 9 and all(_legible(c) for c in colores)


def _todas(datos: DatosReporte) -> Iterator[tuple[str, go.Figure]]:
    figuras: tuple[Callable[[DatosReporte], go.Figure], ...] = (
        g.fig_10a, g.fig_imitador, g.fig_motores_hoy, g.fig_variables_hoy,
        g.fig_motores_tiempo, g.fig_curva, g.fig_firma, g.fig_franja, g.fig_dispersion,
        g.fig_transiciones, g.fig_duraciones,
    )
    for figura in figuras:
        yield figura.__name__, figura(datos)
    for serie in datos.series_macro:
        yield f"fig_macro {serie.columna}", g.fig_macro(datos, serie)


def _fechas(valores: Iterable[object] | None) -> list[pd.Timestamp]:
    """The ISO dates among a trace's x values or a shape's x0/x1 (others are not dates)."""
    return [pd.Timestamp(str(v)[:10]) for v in valores or () if FECHA_ISO.match(str(v))]


def _con_dias_despues(datos: DatosReporte, n: int = 10) -> DatosReporte:
    """Curve and macro running `n` business days past the reading date, as a later download."""
    despues = pd.bdate_range(datos.fecha + pd.offsets.BDay(1), periods=n).as_unit("ns")

    def alargar(table: pd.DataFrame) -> pd.DataFrame:
        cola = pd.DataFrame(
            [table.ffill().iloc[-1].to_numpy() + 1.0] * n,
            index=pd.DatetimeIndex(despues, name=table.index.name),
            columns=table.columns,
        )
        return pd.concat([table, cola])

    return dataclasses.replace(datos, curva=alargar(datos.curva), macro=alargar(datos.macro))


def test_no_figure_shows_a_day_after_the_reading(datos: DatosReporte) -> None:
    futuro = _con_dias_despues(datos)
    assert futuro.curva.index[-1] > datos.fecha and futuro.macro.index[-1] > datos.fecha
    for nombre, fig in _todas(futuro):
        fechas = [f for traza in fig.data for f in _fechas(traza.x)]
        fechas += [f for forma in fig.layout.shapes for f in _fechas([forma.x0, forma.x1])]
        assert all(f <= datos.fecha for f in fechas), nombre
