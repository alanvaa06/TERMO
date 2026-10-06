"""The visual report as one self-contained HTML page (spec 5).

plotly.js is embedded once, in the head; every figure is a div plus its own small
script. No network: no CDN, no web fonts. The page works in light and dark mode and
prints to PDF from the browser.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from termo.config import CoreConfig
from termo.operation.config import OperationConfig
from termo.operation.html_report import md_to_html, write_html
from termo.operation.sheet_es import FIDELITY_WARNING, SEEN_BY_NOTE
from termo.operation.visual import graficas as g
from termo.operation.visual.calculos import Posicion, posicion_duracion
from termo.operation.visual.datos import DatosReporte, cargar
from termo.operation.visual.narrativa import Titular, titulares

ARCHIVO = "reporte.html"
CONFIG_PLOTLY = {"displayModeBar": False, "responsive": True}
FUENTE = "Fuente: FRED, TERMO."
FUENTE_MACRO = "Fuente: FRED (Kim-Wright, fed funds efectiva), TERMO."
LEYENDA_PASADO = "Frecuencias del pasado, no pronóstico. Incluye el periodo holdout, ya abierto."
NOTA_HOLDOUT = "La banda gris marca el periodo holdout, ya abierto."
NOTA_MOTORES = (
    "Cada día explica su propia fase (log-odds); cuando cambia la fase, cambia lo que se explica."
)
GUION = "-"
MIN_EPISODIOS_BANDA = 4  # below this, quartiles of the finished episodes say little
NOTA_POCOS = (
    "Con menos de 4 episodios terminados no se muestra el rango intercuartil, solo la mediana."
)
ANCLAS = (
    "portada", "tasa-10a", "imitador", "motores", "motores-tiempo", "curva", "episodios",
    "transiciones", "macro", "validacion",
)

# The light palette, also forced when printing. The <html> element carries --acento (the
# phase hue) and --acento-texto-claro (its darker tone for text on the light paper).
PALETA_CLARA = (
    "--papel:#fbfaf7;--tinta:#1d1c1a;--tinta-2:#4a4843;--tinta-3:#77756e;--filete:#d9d6cc;"
    "--tarjeta:#f3f1ea;--acento-texto:var(--acento-texto-claro)"
)
ESTILO = """
:root{{claro}}
@media (prefers-color-scheme: dark){:root{--papel:#161615;--tinta:#f0efec;--tinta-2:#c3c2b7;
--tinta-3:#898781;--filete:#383835;--tarjeta:#1f1f1d;--acento-texto:var(--acento)}}
*{box-sizing:border-box}
body{margin:0;background:var(--papel);color:var(--tinta);
font:16px/1.6 "Segoe UI",Helvetica,Arial,sans-serif}
.filete{height:6px;background:var(--acento);print-color-adjust:exact;
-webkit-print-color-adjust:exact}
.marco{display:grid;grid-template-columns:200px minmax(0,1fr);gap:48px;max-width:1220px;
margin:0 auto;padding:32px 24px}
nav{position:sticky;top:24px;align-self:start;font-size:14px}
nav a{display:block;color:var(--tinta-2);text-decoration:none;padding:4px 0 4px 10px;
border-left:2px solid var(--filete)}
nav a:hover{color:var(--tinta);border-left-color:var(--acento)}
main{min-width:0;max-width:920px}
h1,h2{font-family:Georgia,Cambria,"Times New Roman",serif;font-weight:400;line-height:1.2}
h1{font-size:56px;margin:8px 0 4px;color:var(--acento-texto)}
h2{font-size:26px;margin:4px 0 12px}
.meta,.kicker{font-size:13px;color:var(--tinta-3);margin:0}
.titular{font-family:Georgia,Cambria,serif;font-size:28px;line-height:1.25;margin:8px 0 16px}
section{padding:40px 0;border-top:1px solid var(--filete)}
section#portada{border-top:0;padding-top:8px}
ul.vinetas{list-style:none;padding:0;margin:0 0 16px}
ul.vinetas li{padding:2px 0}
ul.vinetas li::before{content:"\\25C6";color:var(--acento-texto);margin-right:10px;
font-size:12px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px;
margin:24px 0}
.kpi{border-top:2px solid var(--tinta);padding-top:8px}
.kpi .valor{font-family:Georgia,Cambria,serif;font-size:32px;line-height:1.1}
.kpi .etiqueta,.kpi .sub{font-size:13px;color:var(--tinta-3)}
.termometro{margin:8px 0 0}
.termometro .pista{position:relative;height:14px;background:var(--tarjeta);border-radius:4px}
.termometro .banda{position:absolute;top:0;height:100%;background:var(--acento);opacity:.28}
.termometro .mediana{position:absolute;top:0;width:2px;height:100%;background:var(--tinta-2)}
.termometro .hoy{position:absolute;top:-4px;width:10px;height:22px;
background:var(--acento-texto);
border-radius:3px;transform:translateX(-50%)}
.termometro .escala{display:flex;justify-content:space-between;font-size:12px;
color:var(--tinta-3);margin-top:6px}
.aviso{display:inline-block;font-size:13px;padding:2px 8px;
border:1px solid var(--acento-texto);border-radius:4px;color:var(--tinta);margin-right:8px}
figure.grafica{margin:24px 0}
figcaption{font-weight:600;font-size:15px;margin-bottom:6px}
.bajada{font-size:14px;color:var(--tinta-2);margin:0 0 6px}
.fuente,.leyenda,.nota{font-size:12px;color:var(--tinta-3);margin:4px 0}
.leyenda{font-style:italic}
.par{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:24px}
.tarjetas{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;
margin:16px 0}
.tarjeta{background:var(--tarjeta);border-radius:8px;padding:14px 16px}
.tarjeta .valor{font-family:Georgia,Cambria,serif;font-size:24px}
table{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0}
th,td{padding:6px 10px;border-bottom:1px solid var(--filete);text-align:left}
.descargo{border-left:3px solid var(--acento);padding:10px 16px;background:var(--tarjeta);
margin:16px 0}
footer{font-size:12px;color:var(--tinta-3);padding:32px 0;border-top:1px solid var(--filete)}
@media (max-width:768px){.marco{grid-template-columns:minmax(0,1fr);gap:16px;padding:16px}
nav{top:0;z-index:2;background:var(--papel);display:flex;overflow-x:auto;gap:14px;
padding:8px 0}nav a{border-left:0;white-space:nowrap;padding:4px 0}h1{font-size:40px}
.titular{font-size:22px}section{scroll-margin-top:48px}}
@media print{:root{{claro}}nav{display:none}.rangeselector{display:none}
.marco{display:block;padding:0}figure.grafica,.kpis,.tarjetas{break-inside:avoid}}
""".replace("{claro}", PALETA_CLARA)

# Three jobs, no dependencies beyond plotly.js:
# - dark mode: recolour the font and every y axis's grid (the figures ship light colours,
#   so a light page needs no relayout on load); printing forces the light colours;
# - y follows x: in a figure marked layout.meta.ajustar_y (graficas._selector_rango), when
#   the x range changes (buttons, zoom, autorange) refit yaxis to the points inside it,
#   stacked areas by their total and with 0, as graficas._extremos does in Python.
SCRIPT_PAGINA = r"""
(function(){var q=window.matchMedia('(prefers-color-scheme: dark)');
function graficas(){
return Array.prototype.slice.call(document.querySelectorAll('.js-plotly-plot'));}
function colorear(oscuro){var t=oscuro?'#c3c2b7':'#4a4843',
r=oscuro?'rgba(255,255,255,0.10)':'rgba(137,135,129,0.18)';
graficas().forEach(function(el){var c={'font.color':t};
Object.keys(el.layout).forEach(function(k){if(/^yaxis\d*$/.test(k))c[k+'.gridcolor']=r;});
Plotly.relayout(el,c);});}
function dia(v){
return typeof v==='number'?new Date(v).toISOString().slice(0,10):String(v).slice(0,10);}
function ajustarY(el){var r=el.layout.xaxis.range;if(!r)return;
var a=dia(r[0]),b=dia(r[1]),lo=Infinity,hi=-Infinity,pilas={},hay=false;
function ver(y){if(y<lo)lo=y;if(y>hi)hi=y;}
(el._fullData||el.data).forEach(function(t){if(t.visible===false||!t.x||!t.y)return;
for(var i=0;i<t.x.length;i++){var x=String(t.x[i]).slice(0,10),y=t.y[i];
if(x<a||x>b||y===null||y===undefined||isNaN(y))continue;
if(t.stackgroup){var k=t.stackgroup+' '+x;pilas[k]=(pilas[k]||0)+y;hay=true;}else ver(y);}});
for(var k in pilas)ver(pilas[k]);if(hay)ver(0);if(lo>hi)return;
var m=(hi-lo)*0.05||Math.abs(hi)*0.05||1;Plotly.relayout(el,{'yaxis.range':[lo-m,hi+m]});}
function seguirX(el){if(!(el.layout.meta&&el.layout.meta.ajustar_y))return;
el.on('plotly_relayout',function(e){for(var k in e){
if(k.indexOf('xaxis.')===0||k==='yaxis.autorange'){ajustarY(el);return;}}});}
function redibujar(){graficas().forEach(function(el){Plotly.Plots.resize(el);});}
function cambio(){colorear(q.matches);}
window.addEventListener('load',function(){graficas().forEach(seguirX);if(q.matches)colorear(true);});
if(q.addEventListener)q.addEventListener('change',cambio);else q.addListener(cambio);
window.addEventListener('beforeprint',function(){colorear(false);redibujar();});
window.addEventListener('afterprint',function(){cambio();redibujar();});})();
"""


@dataclass(frozen=True)
class Grafica:
    titulo: str
    figura: go.Figure
    fuente: str = FUENTE
    leyenda: str = ""
    bajada: str = ""  # the chart's own headline, under its title


@dataclass(frozen=True)
class Seccion:
    ancla: str
    indice: str  # its label in the side index
    titular: Titular
    graficas: tuple[Grafica, ...]
    nota: str = ""
    par: bool = False  # two charts side by side on wide screens
    aviso: str = ""  # a warning right under the heading, before any chart


def _e(texto: object) -> str:
    return html.escape(str(texto), quote=True)


def _num(valor: float | None, decimales: int = 2) -> str:
    """Missing statistics (a phase without evaluable days) show as a dash, as in the sheet."""
    return GUION if valor is None else f"{float(valor):.{decimales}f}"


def _secciones(datos: DatosReporte, textos: dict[str, Titular]) -> list[Seccion]:
    macro = tuple(
        Grafica(s.etiqueta, g.fig_macro(datos, s), FUENTE_MACRO) for s in datos.series_macro
    )
    aviso_motores = "" if datos.lectura["drivers_validated"] else FIDELITY_WARNING
    return [
        Seccion("tasa-10a", "10A por fase", textos["tasa-10a"],
                (Grafica("Rendimiento del bono a 10 años, coloreado por fase (%)",
                         g.fig_10a(datos), leyenda=NOTA_HOLDOUT),)),
        Seccion("imitador", "Fase e imitador", textos["imitador"],
                (Grafica("Fase del modelo, fase del imitador y sus probabilidades",
                         g.fig_imitador(datos)),)),
        Seccion("motores", "Motores de hoy", textos["motores"],
                (Grafica("Contribución SHAP por bloque a la lectura de hoy",
                         g.fig_motores_hoy(datos)),
                 Grafica("Las cinco variables de mayor contribución",
                         g.fig_variables_hoy(datos))),
                aviso=aviso_motores),
        Seccion("motores-tiempo", "Motores en el tiempo", textos["motores-tiempo"],
                (Grafica("Contribución SHAP por bloque, día por día",
                         g.fig_motores_tiempo(datos)),),
                nota=NOTA_MOTORES),
        Seccion("curva", "Curva", textos["curva"],
                (Grafica("Curva de Treasuries: hoy y hace 1 mes, 3 meses y 1 año",
                         g.fig_curva(datos)),
                 Grafica("Firma de cada fase: cambio mediano a 1 mes por plazo",
                         g.fig_firma(datos), leyenda=LEYENDA_PASADO,
                         bajada=textos["firma"].texto)),
                par=True),
        Seccion("episodios", "Episodios", textos["episodios"],
                (Grafica("Fases desde el inicio de la historia", g.fig_franja(datos)),
                 Grafica("Cada episodio: cambio del 2A contra cambio del 10A",
                         g.fig_dispersion(datos), leyenda=LEYENDA_PASADO))),
        Seccion("transiciones", "Transiciones", textos["transiciones"],
                (Grafica("Fase siguiente, por fase de origen", g.fig_transiciones(datos),
                         leyenda=LEYENDA_PASADO),
                 Grafica("Duración de los episodios terminados", g.fig_duraciones(datos),
                         leyenda=LEYENDA_PASADO)),
                par=True),
        Seccion("macro", "Contexto macro", textos["macro"], macro),
    ]


def _grafica(numero: int, grafica: Grafica) -> str:
    div = pio.to_html(grafica.figura, full_html=False, include_plotlyjs=False,
                      div_id=f"fig-{numero}", config=CONFIG_PLOTLY)
    leyenda = f'<p class="leyenda">{_e(grafica.leyenda)}</p>' if grafica.leyenda else ""
    bajada = f'<p class="bajada">{_e(grafica.bajada)}</p>' if grafica.bajada else ""
    return (
        f'<figure class="grafica"><figcaption>{numero}. {_e(grafica.titulo)}</figcaption>'
        f'{bajada}{div}{leyenda}<p class="fuente">{_e(grafica.fuente)}</p></figure>'
    )


def _termometro(posicion: Posicion) -> str:
    """Today's length on a 0..max scale; the P25-P75 band only with 4+ finished episodes."""
    if posicion.p25 is None or posicion.p75 is None or posicion.mediana is None:
        return (
            '<p class="nota">Sin episodios terminados de esta fase para comparar la duración.</p>'
        )
    con_banda = posicion.episodios >= MIN_EPISODIOS_BANDA
    tope = max(float(posicion.dias), posicion.p75 if con_banda else posicion.mediana) * 1.15

    def pct(valor: float) -> str:
        return f"{100.0 * valor / tope:.1f}%"

    terminados = "episodio terminado" if posicion.episodios == 1 else "episodios terminados"
    if con_banda:
        banda = (
            f'<div class="banda" style="left:{pct(posicion.p25)};'
            f'width:{pct(posicion.p75 - posicion.p25)}"></div>'
        )
        resumen = (
            f"hoy {posicion.dias} · P25 {posicion.p25:.0f} · mediana {posicion.mediana:.0f}"
            f" · P75 {posicion.p75:.0f}"
        )
        nota = ""
    else:
        banda = ""
        resumen = f"hoy {posicion.dias} · mediana {posicion.mediana:.0f}"
        nota = f'<p class="nota">{_e(NOTA_POCOS)}</p>'
    return (
        '<div class="termometro"><p class="meta">Duración del episodio contra '
        f"{posicion.episodios} {terminados} de la misma fase</p>"
        f'<div class="pista">{banda}'
        f'<div class="mediana" style="left:{pct(posicion.mediana)}"></div>'
        f'<div class="hoy" style="left:{pct(posicion.dias)}"></div></div>'
        f'<div class="escala"><span>0</span><span>{_e(resumen)}</span>'
        f"<span>{tope:.0f} días</span></div>{nota}</div>"
    )


def _portada(datos: DatosReporte, titular: Titular) -> str:
    lectura = datos.lectura
    fase_n = int(lectura["phase"])
    fase = datos.nombres[fase_n]
    actual = datos.episodios.iloc[-1]
    inicio = actual["inicio"].strftime("%Y-%m-%d")
    posicion = posicion_duracion(datos.episodios, fase_n, int(lectura["days_in_phase"]))
    avisos = ""
    if lectura.get("low_confidence"):
        avisos += '<span class="aviso">confianza baja</span>'
    alerta = datos.hoja.get("alert")
    if alerta:
        avisos += (
            f'<span class="aviso">cambio de fase: {_e(alerta["de"])} a {_e(alerta["a"])}</span>'
        )
    kpis = (
        ("Confianza del imitador", f"{float(lectura['confidence']):.2f}",
         "su probabilidad, no calibrada"),
        ("Días en la fase", str(int(lectura["days_in_phase"])), f"desde {inicio}"),
        ("Δ10A en el episodio", f"{float(actual['cambio_10y_pb']):+.0f} pb", "fin menos inicio"),
        ("Δ2A en el episodio", f"{float(actual['cambio_2y_pb']):+.0f} pb", "fin menos inicio"),
    )
    tarjetas = "".join(
        f'<div class="kpi"><div class="etiqueta">{_e(e)}</div><div class="valor">{_e(v)}</div>'
        f'<div class="sub">{_e(s)}</div></div>'
        for e, v, s in kpis
    )
    vinetas = "".join(f"<li>{_e(v)}</li>" for v in titular.vinetas)
    return (
        '<section id="portada">'
        f'<p class="meta">TERMO · Monitor semanal de Treasuries · lectura '
        f"{datos.fecha:%Y-%m-%d}</p>"
        f"<h1>{_e(fase[:1].upper() + fase[1:])}</h1>"
        f'<p class="titular">{_e(titular.texto)}</p>{avisos}'
        f'<ul class="vinetas">{vinetas}</ul>'
        f'<div class="kpis">{tarjetas}</div>{_termometro(posicion)}'
        f'<p class="nota">{_e(datos.textos["nota_confianza"])}</p></section>'
    )


def _seccion(seccion: Seccion, primero: int) -> tuple[str, int]:
    numero = primero
    figuras: list[str] = []
    for grafica in seccion.graficas:
        figuras.append(_grafica(numero, grafica))
        numero += 1
    cuerpo = "".join(figuras)
    if seccion.par:
        cuerpo = f'<div class="par">{cuerpo}</div>'
    aviso = f'<p class="aviso">{_e(seccion.aviso)}</p>' if seccion.aviso else ""
    nota = f'<p class="nota">{_e(seccion.nota)}</p>' if seccion.nota else ""
    return (
        f'<section id="{seccion.ancla}"><p class="kicker">{_e(seccion.indice)}</p>'
        f"<h2>{_e(seccion.titular.texto)}</h2>{aviso}{cuerpo}{nota}</section>",
        numero,
    )


def _validacion(datos: DatosReporte) -> str:
    validacion = datos.lectura["validation"]
    veredictos = " · ".join(
        f"{v['stage']}: {str(v['verdict']).upper()}" for v in validacion["registered_verdicts"]
    )
    sombra = validacion["sombra"]
    faltan = int(sombra["proxima_evaluacion_semanas"])
    proxima = "ya puede correrse" if faltan == 0 else f"en {faltan} semanas"
    tarjetas = (
        ("Veredictos registrados", veredictos),
        ("Semanas de sombra", f"{int(sombra['semanas'])}"),
        ("Próxima evaluación de sombra", proxima),
    )
    cajas = "".join(
        f'<div class="tarjeta"><div class="meta">{_e(e)}</div>'
        f'<div class="valor">{_e(v)}</div></div>'
        for e, v in tarjetas
    )
    filas = "".join(
        f"<tr><td>{_e(f['name'])}</td><td>{int(f['days'])}</td>"
        f"<td>{'sí' if f['evaluable'] else 'no'}</td><td>{_num(f['recall'])}</td>"
        f"<td>{_num(f['median_duration_days'], 0)}</td></tr>"
        for f in validacion["by_phase"]
    )
    vista = f'<p class="nota">{_e(SEEN_BY_NOTE)}</p>' if "holdout_seen_by" in validacion else ""
    return (
        '<section id="validacion"><p class="kicker">Validación</p>'
        "<h2>Qué tan confiable es la lectura</h2>"
        f'<div class="tarjetas">{cajas}</div>'
        "<table><thead><tr><th>Fase</th><th>Días</th><th>Evaluable</th><th>Recall</th>"
        f"<th>Duración mediana (días)</th></tr></thead><tbody>{filas}</tbody></table>"
        f'<p class="descargo">{_e(datos.textos["descargo"])}</p>'
        f"{vista}"
        f'<p class="nota">{_e(datos.textos["no_dice"])}</p></section>'
    )


def _comentario(datos: DatosReporte) -> str:
    if datos.comentario is None:
        return ""
    return (
        '<section id="comentario"><p class="kicker">Comentario (no generado por TERMO)</p>'
        f"<h2>Comentario del analista</h2>{md_to_html(datos.comentario)}</section>"
    )


def render(datos: DatosReporte, generado: str) -> str:
    """The page as one string: plotly.js once, then the cover and every block."""
    textos = titulares(datos)
    fase = int(datos.lectura["phase"])
    acento = f"--acento:{g.color_fase(fase)};--acento-texto-claro:{g.color_texto_fase(fase)}"
    numero = 1
    cuerpos: list[str] = [_portada(datos, textos["portada"])]
    indice = [("portada", "Portada")]
    for seccion in _secciones(datos, textos):
        cuerpo, numero = _seccion(seccion, numero)
        cuerpos.append(cuerpo)
        indice.append((seccion.ancla, seccion.indice))
    cuerpos.append(_validacion(datos))
    indice.append(("validacion", "Validación"))
    comentario = _comentario(datos)
    if comentario:
        cuerpos.append(comentario)
        indice.append(("comentario", "Comentario"))
    nav = "".join(f'<a href="#{a}">{_e(t)}</a>' for a, t in indice)
    huella = str(datos.hoja["snapshot_hash"])
    pie = (
        f"<footer>Generado el {_e(generado)} · snapshot {_e(huella[:12])} · código "
        f"{_e(str(datos.hoja.get('code_commit', ''))[:12])} · {_e(FUENTE)}</footer>"
    )
    return "\n".join(
        [
            "<!doctype html>",
            f'<html lang="es" style="{acento}">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>TERMO · {datos.fecha:%Y-%m-%d}</title>",
            f"<style>{ESTILO}</style>",
            f"<script>{get_plotlyjs()}</script>",
            "</head>",
            "<body>",
            '<div class="filete"></div>',
            '<div class="marco">',
            f'<nav aria-label="Índice">{nav}</nav>',
            f"<main>{''.join(cuerpos)}{pie}</main>",
            "</div>",
            f"<script>{SCRIPT_PAGINA}</script>",
            "</body>",
            "</html>",
            "",
        ]
    )


def construir(
    dir_salida: Path, dir_snapshot: Path, config: CoreConfig, op: OperationConfig, generado: str
) -> Path:
    """Load the week from `dir_salida` and its snapshot; write `dir_salida/reporte.html`."""
    datos = cargar(dir_salida, dir_snapshot, config, op)
    return write_html(render(datos, generado), dir_salida / ARCHIVO)
