---
Writer: Claude
---

# TERMO — Spec 5: reporte visual semanal

**Fecha:** 2026-10-05
**Estado:** implementado en `feat/reporte-visual` (2026-10-06); §3, §7 y §8 ajustados a lo implementado.
**Documentos base:** [spec 4](2026-10-06-operacion-sombra-design.md) (operación en sombra, CSV y hoja), `src/termo/operation/html_report.py` (reporte actual), referencia visual HSBC *TYCCLES says UST max sell-off* (3 jun 2026), `docs/context/memory.md` (decisión de no publicar desempeño condicionado a que la fase sobreviva).

---

## 1. Objetivo

Reemplazar el `reporte.html` actual (Markdown convertido a tablas, sin gráficas) por un reporte de research **sumamente visual** para el comité: titulares con la conclusión, gráficas numeradas e interactivas, y la misma trazabilidad y prudencia de siempre (describe, no predice).

### Dentro

1. Página HTML autocontenida, sin red: Plotly con su JS embebido una vez.
2. Once bloques (§3) con gráficas, titulares y viñetas generados por reglas.
3. Etiquetas en español para bloques SHAP y para las 139 variables de la receta.
4. Cálculos descriptivos nuevos (§5): firma por fase y plazo, acuerdo imitador–fase, cuadrante de cada episodio, curvas en fechas pasadas.
5. Regenerar el reporte sin correr el modelo: `--solo-reporte`.
6. Comentario opcional del analista (`output/<fecha>/comentario.md`).

### Fuera

- Gráficas de desempeño del 10Y **después** del inicio de una fase, condicionadas a que la fase sobreviva (HSBC charts 9–12). Sigue vigente la decisión de `memory.md`.
- Cualquier estadística de movimiento **posterior** a una fecha de lectura (se mantiene de spec 4). Todo cambio que se grafica termina en el día al que se asigna.
- Correo/Slack, PDF generado por el sistema (el navegador imprime), cambios al modelo, a la receta, a `configs/desc2.yaml` o al registro de trials.
- El peso semanal de `historia_diaria.csv` en git (pendiente aparte).

### Cambio respecto de spec 4

Spec 4 dejó las gráficas fuera. Este spec las añade. No cambia ningún criterio, umbral ni veredicto: es presentación más cálculos descriptivos que se rotulan como tales.

---

## 2. Decisiones de brainstorming

| Decisión | Elegido | Razón |
|---|---|---|
| Tecnología de gráficas | Plotly, JS embebido en el HTML (sin CDN), template propio | Zoom y selector de rango sobre la historia de 1988 a hoy con poco código; el comité abre el reporte en navegador |
| Inventario | Los 11 bloques de §3 | Todo sale de datos que ya se producen cada semana |
| Etiquetas | Renombrar solo en el reporte | No tocar la config registrada (amarrada a hash y commit) |
| Texto | Titulares y viñetas por reglas + `comentario.md` opcional | Reporte completo sin trabajo manual; criterio humano opcional |
| Identidad | Editorial research: papel cálido, titulares serif, gráficas numeradas con "Fuente:" | Es para un comité de inversión, se imprime bien a PDF |
| Arquitectura | El reporte se construye desde archivos (`output/<fecha>/` + snapshot) | Iterar el diseño sin reentrenar; regenerar semanas pasadas |
| Git | `reporte.html` fuera de git | Es derivado (~6–7 MB); se reconstruye desde CSV y snapshot versionados |

---

## 3. La página

Scroll largo, una gráfica por fila (sin gráficas lado a lado: Plotly las dibuja a ancho completo y se enciman). Índice lateral fijo en escritorio, barra superior en móvil (< 768 px). Siempre en tema claro, aunque el sistema del lector esté en oscuro (decisión del usuario, 2026-10-06). CSS de impresión: sin índice, una gráfica por bloque sin cortes de página a la mitad.

Colores de fase, fijos en todo el reporte: **rally fuerte** azul `#2a78d6`, **rally moderado** verde agua `#1baf7a`, **venta** rojo `#e34948`. Cada gráfica lleva número ("1.", "2.", …), título y "Fuente: FRED, TERMO" (más "Kim-Wright vía FRED" en la prima por plazo).

| # | Bloque | Contenido | Titular (plantilla de ejemplo) |
|---|---|---|---|
| 0 | Portada | Fase en serif grande con el color de la fase; fecha de lectura, snapshot (hash corto) y commit; 3 viñetas; 4 KPI: confianza, días en fase, Δ10Y y Δ2Y del episodio (pb); termómetro de duración: banda P25–P75, mediana y marca del episodio actual, contra los episodios terminados de esa fase | "La curva lleva {días} días en {fase}, más que el P75 de los episodios de {fase} cerrados ({p75} días)" (o "menos que el P25" / "entre el P25 y el P75"; sin cuartiles con menos de 4 episodios cerrados) |
| 1 | 10Y por fase | Línea del DGS10 coloreada por la fase del JM, desde 1988. Selector 1A/3A/5A/10A/todo, vista inicial de 3A. Banda gris en los días con `periodo == "holdout"` de `historia_diaria.csv` | "El 10Y {subió / bajó} {x} pb desde que empezó el episodio de {fase}" |
| 2 | Fase contra imitador | Dos franjas diarias alineadas (fase del JM y fase de mayor probabilidad del imitador) sobre el área apilada de las 3 probabilidades (muestreo semanal): los desacuerdos se ven como diferencias entre franjas | "Imitador y fase coinciden en el {n}% de los días del último año" |
| 3 | Motores de hoy | Cascada SHAP: columna `base` → los 6 bloques → valor de la lectura (log-odds de la fase actual), de la fila de la fecha de lectura en `historia_diaria.csv`. Las top 5 variables salen de `hoja.json`. Barras con las 5 variables de mayor contribución, traducidas | "{bloque} explica la mayor parte de la lectura ({aporte:+.2f})" |
| 4 | Motores en el tiempo | Área apilada con signo de la contribución de los 6 bloques, mismo selector de rango, fondo sombreado con el color de la fase de cada día. Nota fija: "Cada día muestra la contribución a la fase de ese día, en log-odds. Cuando la fase cambia, la gráfica pasa a explicar otra probabilidad." | "Desde el {inicio}, {bloque} es el bloque con mayor peso, {media} de promedio" |
| 5a | Curva hoy | Curva de 7 plazos hoy, hace 21, 63 y 252 días hábiles | "En 3 meses el 2A {subió / bajó} {x} pb y el 10A {subió / bajó} {y} pb; la pendiente 2A-10A se {empinó / aplanó} {z} pb" |
| 5b | Firma por fase | Una línea por fase: mediana, en todos los días de esa fase, del cambio de 21 días hábiles **que termina** en ese día, por plazo (1A…30A) | "En {fase}, el {plazo} tiene el mayor cambio mediano a un mes ({valor} pb)" |
| 6 | Episodios | Franja tipo código de barras de los episodios desde 1988; dispersión Δ2Y (x) contra Δ10Y (y) por episodio, coloreada por fase, con la diagonal y los 4 cuadrantes nombrados (bear flattener, bear steepener, bull flattener, bull steepener) y el episodio actual resaltado | "{n} de {N} episodios de venta fueron bear flattener" |
| 7 | Transiciones | Heatmap 3×3 (desde la fase → fase siguiente, % y número de episodios); caja de duraciones por fase con un punto en el episodio actual | "Tras {fase}, el {p}% de los episodios siguió {fase siguiente}" |
| 8 | Macro | Una gráfica por serie (nunca doble eje): prima por plazo 10A (Kim-Wright) y 2Y − fed funds, con banda P10–P90 a 10 años móviles | "La prima por plazo está en el percentil {p} a 10 años" |
| 9 | Validación | Titular con los veredictos y semanas de sombra ("Diagnóstico APTO y holdout APTO; 1 semana de sombra"). Tarjetas: veredictos registrados, recall y duración mediana por fase, semanas de sombra y próxima evaluación. Descargo fijo y visible: "TERMO describe la fase actual de la curva. No anticipa la tasa a 10 años: en 62 pruebas registradas las fases no superaron a la inercia." También "Nombres de fase elegidos tras el holdout; los valida solo la sombra." | — |
| 10 | Comentario del analista | Solo si existe `output/<fecha>/comentario.md`, convertido con `md_to_html`. Rotulado "Comentario del analista (no generado por TERMO)" | — |

Leyenda obligatoria en 5b, 6 y 7: **"Frecuencias del pasado, no pronóstico. Incluye el periodo holdout, ya abierto."**

---

## 4. Etiquetas

### Bloques (solo en el reporte)

| Config (`desc2.yaml`) | Reporte |
|---|---|
| nivel corto | movimiento tramo corto (1A–3A) |
| nivel medio | movimiento tramo medio (5A–7A) |
| nivel largo | movimiento tramo largo (10A–30A) |
| pendientes | pendientes |
| curvatura | curvatura |
| volatilidad | volatilidad |

Razón: los bloques "nivel" miden **cambios** recientes en rango, no el nivel de las tasas.

### Variables

Patrones de `TycclesRecipe.names`:

- `d{plazo}_{h}_r{w}` → "{plazo}A · cambio {h} · rango {w}"
- `{medida}_{h}_r{w}` → "{medida} · cambio {h} · rango {w}", con medidas 1s5s (`s12m5s`), 3s10s, 10s30s y curvatura 2·5A−2A−10A (`c5`)
- `vol{plazo}_r{w}` → "{plazo}A · volatilidad 21d · rango {w}"

Horizontes en meses hábiles: 21 → 1m, 42 → 2m, 63 → 3m, 84 → 4m, 126 → 6m, 189 → 9m. Ventanas: 126 → 6m, 252 → 1a. Si un nombre no coincide con ningún patrón, error (el test cubre las 139).

---

## 5. Cálculos descriptivos nuevos (`calculos.py`)

Todos usan solo datos ya publicados (CSV, snapshot) y ninguno mira después de la fecha a la que se asigna.

- **Firma por fase**: para cada día *t* con fase *f*, el cambio `y(t) − y(t−21)` en pb por plazo; mediana por fase y plazo. Días sin los 21 previos se omiten.
- **Acuerdo imitador–fase**: % de días de la ventana (último año = 252 días hábiles) en que `argmax(p)` coincide con la fase del JM.
- **Cuadrante del episodio**: con Δ2Y y Δ10Y del episodio (`episodios.csv`). Si los dos suben → bear; si los dos bajan → bull; si se mueven en sentidos opuestos → "mixto". Aplanamiento si Δ2Y > Δ10Y, empinamiento si no. Si |Δ2Y − Δ10Y| < 1 pb → "paralelo".
- **Curvas pasadas**: la curva en la fecha de lectura y en las fechas hábiles 21, 63 y 252 días antes (última observación disponible en o antes de esa fecha).
- **Posición en la duración**: los días del episodio actual contra P25, mediana y P75 de los episodios **terminados** de la misma fase.
- **Duraciones y transiciones**: se reutilizan `monthly.durations_by_phase` y `monthly.transitions` sobre `episodios.csv`.

---

## 6. Narrativa (`narrativa.py`)

- Una función por bloque devuelve `Titular(texto, viñetas)`. Las plantillas son cerradas: solo se insertan números y nombres que vienen de `DatosReporte` o de `calculos`.
- La portada lleva 3 viñetas: el motor principal, el acuerdo imitador–fase y el macro más extremo por percentil.
- Ningún texto usa futuro ni condicional. Un test los rechaza con un regex sobre todos los textos generados con fixtures: palabras terminadas en -rá/-rán/-ría/-rían, más "pronto", "esperamos", "probable", "anticipa" (salvo el descargo fijo, que es texto constante).

---

## 7. Arquitectura

Módulos nuevos en `src/termo/operation/visual/`:

| Módulo | Responsabilidad | Interfaz |
|---|---|---|
| `datos.py` | Carga y valida | `cargar(dir_salida, dir_snapshot, config: CoreConfig, op: OperationConfig) -> DatosReporte` (dataclass inmutable: `hoja`, `historia`, `episodios`, `macro`, `curva`, nombres, bloques, series macro, textos, `comentario: str \| None`). No lee `lecturas.csv` ni `alertas.csv`: la alerta vigente sale de `hoja.json` |
| `etiquetas.py` | Nombres en español | `bloque_es(nombre) -> str`, `variable_es(nombre) -> str` |
| `calculos.py` | §5 | funciones puras sobre DataFrames |
| `graficas.py` | Una figura por gráfica | `fig_<bloque>(datos, …) -> go.Figure`, template `TERMO`, `COLOR_FASE` |
| `narrativa.py` | §6 | `titulares(datos) -> dict[str, Titular]` |
| `pagina.py` | HTML final | `construir(dir_salida, dir_snapshot, config, op, generado: str) -> Path` (escribe `reporte.html`) |

`pagina.py`:
- `plotly.offline.get_plotlyjs()` en un solo `<script>` al inicio.
- Cada figura con `to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})`.
- Fondos de figura transparentes. Un script corto aplica `Plotly.relayout` (color de texto y de la retícula) según `prefers-color-scheme` y cuando cambia.
- Fuentes del sistema (serif: Georgia/Cambria; sans: Segoe UI/Helvetica). Sin fuentes externas.

`html_report.py`: se queda `md_to_html` (lo usa el comentario) y `write_html`. Se retira `render_html`.

### Flujo

1. `run_week.main` igual que hoy hasta copiar los archivos a `output/<fecha>/`.
2. Después: `visual.pagina.construir(out_dir, snapshot_dir, config, op, generado)`.
3. Flag nuevo `--solo-reporte DIR`: exige `--snapshot` y es incompatible con `--download` y `--mes`. Solo ejecuta el paso 2.
4. El reporte ya no depende de `--mes`; la ficha sigue saliendo como archivo aparte cuando se pide.

### Dependencias y repo

- `plotly` en `dependencies` de `pyproject.toml`, con la versión fijada a la instalada al implementar.
- `.gitignore`: `output/**/reporte.html`; `git rm --cached output/2026-10-02/reporte.html`.

---

## 8. Errores (solo en la frontera)

- `snapshot_hash(dir_snapshot)` ≠ `hoja.json["snapshot_hash"]` → `ValueError` con las dos huellas cortas. `run_week` lo convierte en `parser.error` (ASCII, salida ≠ 0).
- Falta `hoja.json`, `historia_diaria.csv`, `episodios.csv` o `macro.csv` → `FileNotFoundError` con el nombre.
- Un CSV cuya línea `# snapshot_hash=` no es la de `hoja.json`, una historia que no termina en la fecha de lectura, o un último episodio que no es la fase de la lectura → `ValueError` ASCII.
- Sin alerta, sin `comentario.md`, serie macro con huecos → el bloque muestra "sin datos" o anota el último dato disponible (como la hoja actual). No es error.
- Salida de consola ASCII: `[ok] reporte -> output/<fecha>/reporte.html`.

---

## 9. Tests (pytest)

| Archivo | Verifica |
|---|---|
| `test_visual_etiquetas.py` | Las 139 variables de `TycclesRecipe` (config desc2) se traducen; un nombre desconocido lanza error; los 6 bloques se traducen |
| `test_visual_calculos.py` | Series sintéticas con respuesta conocida: firma (incluido que solo usa el pasado: cambiar *y* después de *t* no cambia el valor en *t*), acuerdo, los cuadrantes y "paralelo", curvas pasadas en días no hábiles, posición en la duración solo con episodios terminados |
| `test_visual_graficas.py` | Por figura: número de trazos, color por fase, longitud de los datos, vista inicial de 3A en el bloque 1, sin doble eje en macro |
| `test_visual_narrativa.py` | Titulares con fixtures (texto exacto); el regex de futuro/condicional sobre todos los textos; las 3 viñetas de portada |
| `test_visual_pagina.py` | plotly.js aparece exactamente una vez; ningún `src="http` ni `href="http` a recursos; los 11 bloques presentes con su id; títulos escapados; el comentario aparece solo si existe |
| `test_visual_datos.py` | Hash que no coincide → error; falta un CSV obligatorio → error con el nombre |
| `test_run_week.py` (actualizar) | La corrida completa produce el reporte nuevo; `--solo-reporte` regenera sin tocar la bitácora de sombra; `--solo-reporte` con `--download` se rechaza |

---

## 10. Documentación

- `docs/context/memory.md`: decisión de reporte visual (Plotly embebido, construido desde archivos, fuera de git).
- `README.md`: uso de `--solo-reporte` y de `comentario.md`.
