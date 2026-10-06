"""One Plotly figure per chart: traces, phase colours, data lengths, initial view."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import pytest

from termo.operation.visual import graficas as g
from termo.operation.visual.datos import DatosReporte
from termo.operation.visual.etiquetas import bloque_es
from visual_fixture import BLOQUES, NOMBRES, make_datos


@pytest.fixture(scope="module")
def datos() -> DatosReporte:
    return make_datos()


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


def test_surrogate_against_phase(datos: DatosReporte) -> None:
    fig = g.fig_imitador(datos)
    franjas = [t for t in fig.data if isinstance(t, go.Heatmap)]
    areas = [t for t in fig.data if isinstance(t, go.Scatter)]
    assert len(franjas) == 2 and len(areas) == 3
    assert list(franjas[0].z[0]) == datos.historia["fase"].tolist()
    assert list(franjas[0].z[0]) != list(franjas[1].z[0])  # the planted disagreements
    assert {t.stackgroup for t in areas} == {"p"}


def test_drivers_of_the_reading(datos: DatosReporte) -> None:
    fig = g.fig_motores_hoy(datos)
    (cascada,) = fig.data
    assert isinstance(cascada, go.Waterfall)
    fila = datos.historia.iloc[-1]
    assert list(cascada.y) == ["base", *(bloque_es(b) for b in BLOQUES), "lectura"]
    assert list(cascada.x[1:-1]) == pytest.approx([float(fila[b]) for b in BLOQUES])
    assert cascada.measure[0] == "absolute" and cascada.measure[-1] == "total"


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


def test_curve_today_and_signature(datos: DatosReporte) -> None:
    curva = g.fig_curva(datos)
    assert [t.name for t in curva.data] == ["hoy", "hace 1m", "hace 3m", "hace 1a"]
    assert list(curva.data[0].x) == ["1A", "2A", "3A", "5A", "7A", "10A", "30A"]
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


def test_transitions_and_durations(datos: DatosReporte) -> None:
    (mapa,) = g.fig_transiciones(datos).data
    assert len(mapa.z) == 3 and all(len(fila) == 3 for fila in mapa.z)
    fig = g.fig_duraciones(datos)
    assert [t.name for t in fig.data] == [*NOMBRES, "episodio actual"]
    assert fig.data[-1].y[0] == datos.episodios.iloc[-1]["dias"]


def test_macro_has_one_axis_and_a_band(datos: DatosReporte) -> None:
    for serie in datos.series_macro:
        fig = g.fig_macro(datos, serie)
        assert len(fig.data) == 3
        assert "yaxis2" not in fig.to_dict()["layout"]  # never a dual axis
        assert fig.data[1].fill == "tonexty"
