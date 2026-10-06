"""One Plotly figure per chart of the visual report (spec 5, section 3).

Figures have transparent backgrounds: the page's CSS owns the paper colour, and a short
script recolours fonts and gridlines in dark mode. Phase colours are fixed by phase
number, block colours avoid the phase hues, and no chart has two y axes.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from termo.operation.monthly import transitions
from termo.operation.visual import calculos
from termo.operation.visual.datos import DatosReporte, SerieMacro
from termo.operation.visual.etiquetas import bloque_es, variable_es

COLORES_FASE = ("#2a78d6", "#1baf7a", "#e34948")
COLORES_BLOQUE = ("#6250d6", "#eb6834", "#eda100", "#e87ba4", "#008300", "#888780")
# Marks and reference lines keep a fixed colour in both modes, so they are mid-tones that
# clear WCAG 3:1 on the light (#fbfaf7) and the dark (#161615) page alike.
GRIS = "#888780"  # 3.45 on light, 5.02 on dark
MARCA = "#77756f"  # 4.41 on light, 3.93 on dark: totals and the current episode
TINTA = "#4a4843"  # the light-mode font only: the page script recolours it in dark mode
EJE = GRIS  # x axis line, ticks, waterfall connector (#c3c2b7 was 1.72 on light)
RETICULA = "rgba(137,135,129,0.18)"
COLOR_MACRO = "#6250d6"
GRISES_CURVA = (("hace 1m", "dot", 2.4), ("hace 3m", "dash", 1.6),
                ("hace 1a", "longdash", 1.0))  # one grey, told apart by dash and width
# Neutral (blue is 'rally fuerte'). Capped at 0.6 so the cell text, in the page's font
# colour, keeps >= 3.8:1 on the darkest cell in both modes (0.85 drops to 2.5 in dark).
RAMPA_TRANSICION = [[0.0, "rgba(137,135,129,0.08)"], [1.0, "rgba(137,135,129,0.6)"]]
PLAZOS = {"DGS1": "1A", "DGS2": "2A", "DGS3": "3A", "DGS5": "5A", "DGS7": "7A",
          "DGS10": "10A", "DGS30": "30A"}
ALTO = 380
TRANSPARENTE = "rgba(0,0,0,0)"

TEMA = go.layout.Template(
    layout=go.Layout(
        font={"family": "'Segoe UI', Helvetica, Arial, sans-serif", "size": 12, "color": TINTA},
        paper_bgcolor=TRANSPARENTE,
        plot_bgcolor=TRANSPARENTE,
        margin={"l": 56, "r": 16, "t": 24, "b": 40},
        height=ALTO,
        xaxis={"showgrid": False, "linecolor": EJE, "ticks": "outside", "tickcolor": EJE},
        yaxis={"gridcolor": RETICULA, "zeroline": False},
        legend={"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"},
        hovermode="x unified",
    )
)


def color_fase(fase: int) -> str:
    return COLORES_FASE[fase % len(COLORES_FASE)]


def _color_bloque(posicion: int) -> str:
    return COLORES_BLOQUE[posicion % len(COLORES_BLOQUE)]


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _dias(index: pd.Index) -> list[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in index]


def _valores(serie: pd.Series, decimales: int = 3) -> list[float]:
    return [float("nan") if math.isnan(v) else round(float(v), decimales) for v in serie]


def _historia(datos: DatosReporte) -> pd.DataFrame:
    return datos.historia.loc[: datos.fecha]


def _semanal(historia: pd.DataFrame) -> pd.DataFrame:
    """The last labelled day of each W-FRI week: past days dropped, the reading day kept."""
    ultimo = ~historia.index.to_period("W-FRI").duplicated(keep="last")
    return historia[ultimo]


def _selector_rango(fig: go.Figure, fecha: pd.Timestamp, anios: int = 3) -> None:
    botones = [
        {"count": n, "label": f"{n}A", "step": "year", "stepmode": "backward"}
        for n in (1, 3, 5, 10)
    ]
    botones.append({"step": "all", "label": "todo"})
    fig.update_layout(xaxis_rangeselector={"buttons": botones, "x": 0, "y": 1.12})
    fig.update_xaxes(
        range=[(fecha - pd.DateOffset(years=anios)).strftime("%Y-%m-%d"),
               fecha.strftime("%Y-%m-%d")]
    )


def _escala_fases(n: int) -> list[list[float | str]]:
    escala: list[list[float | str]] = []
    for fase in range(n):
        escala += [[fase / n, color_fase(fase)], [(fase + 1) / n, color_fase(fase)]]
    return escala


def _franja(fases: pd.Series, nombres: tuple[str, ...], etiqueta: str) -> go.Heatmap:
    valores = [int(f) for f in fases]
    return go.Heatmap(
        x=_dias(fases.index),
        y=[etiqueta],
        z=[valores],
        text=[[nombres[f] for f in valores]],
        colorscale=_escala_fases(len(nombres)),
        zmin=-0.5,
        zmax=len(nombres) - 0.5,
        showscale=False,
        hovertemplate="%{x}: %{text}<extra>" + etiqueta + "</extra>",
    )


def _sombrear_holdout(fig: go.Figure, historia: pd.DataFrame) -> None:
    dias = historia.index[historia["periodo"] == "holdout"]
    if len(dias):
        fig.add_vrect(
            x0=_dias(dias[:1])[0], x1=_dias(dias[-1:])[0], fillcolor=GRIS, opacity=0.10,
            line_width=0, layer="below", annotation_text="holdout",
            annotation_position="top left",
        )


def _sombrear_fases(fig: go.Figure, episodios: pd.DataFrame) -> None:
    for inicio, fin, fase in zip(episodios["inicio"], episodios["fin"], episodios["fase"],
                                 strict=True):
        fig.add_vrect(
            x0=pd.Timestamp(inicio).strftime("%Y-%m-%d"),
            x1=pd.Timestamp(fin).strftime("%Y-%m-%d"),
            fillcolor=color_fase(int(fase)), opacity=0.07, line_width=0, layer="below",
        )


def fig_10a(datos: DatosReporte) -> go.Figure:
    """The 10Y coloured by the jump model's phase; a segment takes the colour of its end day.

    So the reading day is always drawn in its own phase, even on the day a phase starts.
    """
    historia = _historia(datos)
    fases = historia["fase"]
    tasa = datos.curva["DGS10"].reindex(historia.index)
    dias = _dias(historia.index)
    fig = go.Figure()
    for fase, nombre in enumerate(datos.nombres):
        dentro = (fases == fase) | (fases.shift(-1) == fase)
        fig.add_trace(
            go.Scatter(
                x=dias, y=_valores(tasa.where(dentro)), name=nombre, mode="lines",
                line={"color": color_fase(fase), "width": 1.6}, connectgaps=False,
                hovertemplate="%{y:.2f}%<extra>" + nombre + "</extra>",
            )
        )
    _sombrear_holdout(fig, historia)
    fig.update_layout(template=TEMA, yaxis_title="10A (%)")
    _selector_rango(fig, datos.fecha)
    return fig


def fig_imitador(datos: DatosReporte) -> go.Figure:
    """Jump-model phase and surrogate phase as strips, the surrogate's probabilities below.

    The strips are daily, so a one-day disagreement is never sampled away; the probability
    areas are sampled weekly (the last labelled day of each week) to keep the page light.
    """
    historia = _historia(datos)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.08, 0.08, 0.84],
                        vertical_spacing=0.02)
    fig.add_trace(_franja(historia["fase"], datos.nombres, "fase"), row=1, col=1)
    imitador = calculos.fase_imitador(historia, datos.nombres)
    fig.add_trace(_franja(imitador, datos.nombres, "imitador"), row=2, col=1)
    semanal = _semanal(historia)
    dias = _dias(semanal.index)
    for fase, nombre in enumerate(datos.nombres):
        fig.add_trace(
            go.Scatter(
                x=dias, y=_valores(semanal[calculos.columna_probabilidad(nombre)]),
                name=nombre, stackgroup="p", mode="lines", line={"width": 0},
                fillcolor=_rgba(color_fase(fase), 0.8),
                hovertemplate="%{y:.2f}<extra>" + nombre + "</extra>",
            ),
            row=3, col=1,
        )
    fig.update_layout(template=TEMA, height=440, yaxis3={"range": [0, 1], "title": "probabilidad"})
    _selector_rango(fig, datos.fecha)
    return fig


def fig_motores_hoy(datos: DatosReporte) -> go.Figure:
    """Base value, then each block's SHAP contribution, then the reading (log-odds)."""
    fila = datos.historia.loc[datos.fecha]
    fase = int(fila["fase"])  # the phase of the row the bars come from
    valores = [float(fila[b]) for b in datos.bloques]
    base = float(fila["base"])
    total = base + sum(valores)
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["absolute", *["relative"] * len(valores), "total"],
            y=["base", *(bloque_es(b) for b in datos.bloques), "lectura"],
            x=[base, *valores, 0.0],
            text=[f"{base:+.2f}", *(f"{v:+.2f}" for v in valores), f"{total:+.2f}"],
            textposition="outside",
            increasing={"marker": {"color": color_fase(fase)}},
            # negatives hollow-ish (light fill, grey outline), the reading solid MARCA
            decreasing={"marker": {"color": _rgba(GRIS, 0.3),
                                   "line": {"color": GRIS, "width": 1.5}}},
            totals={"marker": {"color": MARCA}},
            connector={"line": {"color": EJE}},
        )
    )
    fig.update_layout(template=TEMA, hovermode="closest", yaxis={"autorange": "reversed"},
                      xaxis_title=f"log-odds de {datos.nombres[fase]}", margin={"l": 230})
    return fig


def fig_variables_hoy(datos: DatosReporte) -> go.Figure:
    variables = list(datos.lectura["top_variables"])
    aportes = [float(v["contribution"]) for v in variables]
    fase = int(datos.lectura["phase"])
    fig = go.Figure(
        go.Bar(
            orientation="h",
            x=aportes,
            y=[variable_es(str(v["variable"])) for v in variables],
            marker={"color": [color_fase(fase) if a >= 0 else GRIS for a in aportes]},
            text=[f"{a:+.2f}" for a in aportes],
            textposition="outside",
            hovertemplate="%{y}: %{x:+.2f}<extra></extra>",
        )
    )
    fig.update_layout(template=TEMA, hovermode="closest", height=300,
                      yaxis={"autorange": "reversed"}, margin={"l": 260},
                      xaxis_title="contribución SHAP (log-odds)")
    return fig


def fig_motores_tiempo(datos: DatosReporte) -> go.Figure:
    """Each block's contribution to the day's own phase, stacked by sign; sampled weekly.

    A missing contribution stacks as 0 but shows as missing in the hover.
    """
    historia = _semanal(_historia(datos))
    dias = _dias(historia.index)
    fig = go.Figure()
    for posicion, bloque in enumerate(datos.bloques):
        valor = historia[bloque]
        apilado = valor.fillna(0.0)
        nombre = bloque_es(bloque)
        relleno = _rgba(_color_bloque(posicion), 0.75)
        fig.add_trace(
            go.Scatter(
                x=dias, y=_valores(apilado.clip(lower=0.0)), customdata=_valores(valor),
                name=nombre, legendgroup=bloque, showlegend=True, stackgroup="pos", mode="lines",
                line={"width": 0}, fillcolor=relleno,
                hovertemplate="%{customdata:+.2f}<extra>" + nombre + "</extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dias, y=_valores(apilado.clip(upper=0.0)), name=nombre, legendgroup=bloque,
                showlegend=False, stackgroup="neg", mode="lines", line={"width": 0},
                fillcolor=relleno, hoverinfo="skip",
            )
        )
    _sombrear_fases(fig, datos.episodios)
    fig.update_layout(template=TEMA, height=420, yaxis_title="log-odds de la fase del día")
    _selector_rango(fig, datos.fecha)
    return fig


def fig_curva(datos: DatosReporte) -> go.Figure:
    curvas = calculos.curvas_pasadas(datos.curva, datos.fecha)
    plazos = [PLAZOS.get(c, c) for c in curvas.columns]
    fase = int(datos.lectura["phase"])
    estilos = {"hoy": (color_fase(fase), "solid", 3.0)}
    estilos.update({nombre: (GRIS, guion, ancho) for nombre, guion, ancho in GRISES_CURVA})
    fig = go.Figure()
    for etiqueta, fila in curvas.iterrows():
        color, guion, ancho = estilos[str(etiqueta)]
        fig.add_trace(
            go.Scatter(x=plazos, y=_valores(fila, 2), name=str(etiqueta), mode="lines+markers",
                       line={"color": color, "dash": guion, "width": ancho},
                       hovertemplate="%{y:.2f}%<extra>" + str(etiqueta) + "</extra>")
        )
    fig.update_layout(template=TEMA, yaxis_title="rendimiento (%)", xaxis_type="category")
    return fig


def fig_firma(datos: DatosReporte) -> go.Figure:
    firma = calculos.firma_por_fase(datos.curva, _historia(datos)["fase"], datos.nombres)
    plazos = [PLAZOS.get(c, c) for c in firma.columns]
    fig = go.Figure()
    for fase, (nombre, fila) in enumerate(firma.iterrows()):
        fig.add_trace(
            go.Scatter(x=plazos, y=_valores(fila, 1), name=str(nombre), mode="lines+markers",
                       line={"color": color_fase(fase), "width": 2},
                       hovertemplate="%{y:+.1f} pb<extra>" + str(nombre) + "</extra>")
        )
    fig.add_hline(y=0, line_color=GRIS, line_width=1)
    fig.update_layout(template=TEMA, xaxis_type="category",
                      yaxis_title="cambio mediano a 1 mes (pb)")
    return fig


def fig_franja(datos: DatosReporte) -> go.Figure:
    historia = _historia(datos)
    fig = go.Figure(_franja(historia["fase"], datos.nombres, "fase"))
    fig.update_layout(template=TEMA, height=110, margin={"t": 8, "b": 28},
                      yaxis={"showticklabels": False}, hovermode="closest")
    return fig


def fig_dispersion(datos: DatosReporte) -> go.Figure:
    """Each episode's 2Y change (x) against its 10Y change (y), quadrants named."""
    episodios = datos.episodios
    cuadrantes = calculos.cuadrantes(episodios)
    fig = go.Figure()
    for fase, nombre in enumerate(datos.nombres):
        suyos = episodios[episodios["fase"] == fase]
        fig.add_trace(
            go.Scatter(
                x=_valores(suyos["cambio_2y_pb"], 1), y=_valores(suyos["cambio_10y_pb"], 1),
                name=nombre, mode="markers",
                marker={"color": color_fase(fase), "opacity": 0.75,
                        "size": [6 + math.sqrt(float(d)) for d in suyos["dias"]]},
                text=[
                    f"{pd.Timestamp(i):%Y-%m-%d} a {pd.Timestamp(f):%Y-%m-%d} · {d} días · {c}"
                    for i, f, d, c in zip(suyos["inicio"], suyos["fin"], suyos["dias"],
                                          cuadrantes[suyos.index], strict=True)
                ],
                hovertemplate="%{text}<br>Δ2A %{x:+.0f} pb · Δ10A %{y:+.0f} pb<extra></extra>",
            )
        )
    actual = episodios.iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=[float(actual["cambio_2y_pb"])], y=[float(actual["cambio_10y_pb"])],
            name="episodio actual", mode="markers",
            marker={"symbol": "circle-open", "size": 22, "line": {"width": 3}, "color": MARCA},
            hovertemplate="episodio actual<extra></extra>",
        )
    )
    tope = 1.05 * max(
        1.0, float(episodios["cambio_2y_pb"].abs().max()),
        float(episodios["cambio_10y_pb"].abs().max()),
    )
    fig.add_shape(type="line", x0=-tope, y0=-tope, x1=tope, y1=tope,
                  line={"color": GRIS, "dash": "dot", "width": 1})
    fig.add_hline(y=0, line_color=GRIS, line_width=1)
    fig.add_vline(x=0, line_color=GRIS, line_width=1)
    for texto, x, y in (("bear steepener", 0.3, 0.9), ("bear flattener", 0.9, 0.3),
                        ("bull steepener", -0.9, -0.3), ("bull flattener", -0.3, -0.9)):
        fig.add_annotation(x=x * tope, y=y * tope, text=texto, showarrow=False,
                           font={"color": GRIS, "size": 12})
    fig.update_layout(template=TEMA, height=520, hovermode="closest",
                      xaxis={"title": "Δ2A en el episodio (pb)", "range": [-tope, tope],
                             "zeroline": False},
                      yaxis={"title": "Δ10A en el episodio (pb)", "range": [-tope, tope],
                             "scaleanchor": "x", "zeroline": False})
    return fig


def fig_transiciones(datos: DatosReporte) -> go.Figure:
    """Share of episodes of each phase followed by each phase.

    The cell text is drawn as annotations: they take the layout font colour, which the
    page script recolours in dark mode (a heatmap's own text would not follow it).
    """
    filas = transitions(datos.episodios, datos.nombres)
    z: list[list[float | None]] = []
    texto: list[list[str]] = []
    for fila in filas:
        total = int(fila["total"])
        proporciones = [fila["a"][nombre] for nombre in datos.nombres]
        z.append([None if p is None else 100.0 * p for p in proporciones])
        texto.append(["-" if p is None else f"{p:.0%} ({round(p * total)})" for p in proporciones])
    fig = go.Figure(
        go.Heatmap(
            z=z, x=list(datos.nombres), y=list(datos.nombres), text=texto,
            colorscale=RAMPA_TRANSICION, zmin=0, zmax=100, showscale=False,
            hovertemplate="de %{y} a %{x}: %{text}<extra></extra>",
        )
    )
    for desde, fila_texto in zip(datos.nombres, texto, strict=True):
        for hacia, celda in zip(datos.nombres, fila_texto, strict=True):
            fig.add_annotation(x=hacia, y=desde, text=celda, showarrow=False)
    fig.update_layout(template=TEMA, height=340, hovermode="closest",
                      xaxis={"title": "fase siguiente", "side": "top"},
                      yaxis={"title": "desde", "autorange": "reversed"})
    return fig


def fig_duraciones(datos: DatosReporte) -> go.Figure:
    terminados = datos.episodios.iloc[:-1]
    fig = go.Figure()
    for fase, nombre in enumerate(datos.nombres):
        fig.add_trace(
            go.Box(y=terminados.loc[terminados["fase"] == fase, "dias"].tolist(), name=nombre,
                   marker_color=color_fase(fase), boxpoints="all", jitter=0.4, pointpos=0)
        )
    actual = datos.episodios.iloc[-1]
    fig.add_trace(
        go.Scatter(x=[datos.nombres[int(actual["fase"])]], y=[int(actual["dias"])],
                   name="episodio actual", mode="markers",
                   marker={"symbol": "diamond", "size": 14, "color": MARCA},
                   hovertemplate="episodio actual: %{y} días<extra></extra>")
    )
    fig.update_layout(template=TEMA, hovermode="closest", yaxis_title="días hábiles")
    return fig


def fig_macro(datos: DatosReporte, serie: SerieMacro) -> go.Figure:
    """The series with its trailing P10-P90 band over the percentile window; one axis."""
    valores = datos.macro[serie.columna].loc[: datos.fecha]
    conocida = valores.dropna()
    banda = calculos.banda_macro(valores, datos.ventana_percentil)
    dias = _dias(conocida.index)
    anios = round(datos.ventana_percentil / calculos.DIAS_ANIO)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dias, y=_valores(banda["p90"]), mode="lines",
                             line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dias, y=_valores(banda["p10"]), mode="lines", line={"width": 0},
                             fill="tonexty", fillcolor=_rgba(GRIS, 0.18),
                             name=f"P10-P90 a {anios} años", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dias, y=_valores(conocida), mode="lines", name=serie.etiqueta,
                             line={"color": COLOR_MACRO, "width": 1.6},
                             hovertemplate="%{y:.2f}<extra></extra>"))
    fig.update_layout(template=TEMA, height=320)
    _selector_rango(fig, datos.fecha, anios=10)
    return fig
