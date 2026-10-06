# Reporte visual semanal — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reemplazar `output/<fecha>/reporte.html` (Markdown convertido a tablas) por un reporte de research visual con Plotly embebido, construido desde los archivos de la semana (spec 5).

**Architecture:** Paquete nuevo `src/termo/operation/visual/` con seis módulos de una sola tarea cada uno: `etiquetas` (nombres en español), `datos` (carga y valida `output/<fecha>/` + snapshot), `calculos` (números descriptivos nuevos), `graficas` (una figura Plotly por gráfica), `narrativa` (titulares por plantillas cerradas) y `pagina` (HTML autocontenido). `run_week.py` llama `pagina.construir` al final de la corrida y gana `--solo-reporte`.

**Tech Stack:** Python 3.14, pandas 3, plotly 7.1.0 (JS embebido con `get_plotlyjs`), pytest, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-10-05-reporte-visual-design.md`

---

## Reglas para quien ejecute

- Todo comando corre desde la raíz del repo `C:\Proyectos\TERMO` con el venv: `.venv/Scripts/python -m pytest ...`.
- Salida de consola **solo ASCII** (CLAUDE.md global). El texto dentro del HTML sí lleva acentos.
- Sin look-ahead: ningún cálculo asignado a un día usa datos posteriores a ese día.
- El reporte **describe, no predice**: ningún texto en futuro o condicional.
- Commits con el trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Después de cada tarea: `.venv/Scripts/python -m ruff check src tests` y `.venv/Scripts/python -m mypy` deben quedar limpios.

## Mapa de archivos

| Archivo | Acción | Responsabilidad |
|---|---|---|
| `pyproject.toml` | Modificar | Dependencia `plotly==7.1.0`; override de mypy si hace falta |
| `.gitignore` | Modificar | `output/**/reporte.html` |
| `src/termo/operation/visual/__init__.py` | Crear | Paquete vacío |
| `src/termo/operation/visual/etiquetas.py` | Crear | Bloques y variables en español |
| `src/termo/operation/visual/datos.py` | Crear | `DatosReporte`, `SerieMacro`, `cargar` |
| `src/termo/operation/visual/calculos.py` | Crear | Firma, acuerdo, cuadrantes, curvas pasadas, posición, bloque dominante, banda macro |
| `src/termo/operation/visual/graficas.py` | Crear | Template `TERMO`, colores, 12 funciones `fig_*` |
| `src/termo/operation/visual/narrativa.py` | Crear | `Titular`, plantillas, `titulares` |
| `src/termo/operation/visual/pagina.py` | Crear | `render`, `construir`, `ARCHIVO` |
| `src/termo/operation/html_report.py` | Modificar | Retirar `render_html` y `STYLE`; quedan `md_to_html`, `write_html` |
| `src/termo/run_week.py` | Modificar | Llamar `construir`; flag `--solo-reporte` |
| `tests/visual_fixture.py` | Crear | Datos sintéticos coherentes, en memoria y en disco |
| `tests/test_visual_*.py` | Crear | Una por módulo |
| `tests/test_html_report.py` | Modificar | Quitar pruebas de `render_html` |
| `tests/test_run_week.py` | Modificar | Aserciones del reporte nuevo; `--solo-reporte` |
| `README.md`, `docs/context/*`, spec 5 | Modificar | Documentación |

---

### Task 1: Dependencia Plotly y reporte fuera de git

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `src/termo/operation/visual/__init__.py`

- [ ] **Step 1: Agregar la dependencia**

En `pyproject.toml`, dentro de `dependencies = [...]`, después de `"shap==0.52.0",` agregar:

```toml
    "plotly==7.1.0",
```

- [ ] **Step 2: Instalar**

Run: `.venv/Scripts/python -m pip install -e ".[dev]"`
Expected: termina con `Successfully installed ... plotly-7.1.0 ...` (o "Requirement already satisfied" para el resto).

- [ ] **Step 3: Verificar que el JS embebible existe**

Run: `.venv/Scripts/python -c "from plotly.offline import get_plotlyjs; import plotly; print(plotly.__version__, len(get_plotlyjs()) > 4000000)"`
Expected: `7.1.0 True`

- [ ] **Step 4: Crear el paquete**

`src/termo/operation/visual/__init__.py`:

```python
"""The visual weekly report (spec 5): built from the week's output files and snapshot."""
```

- [ ] **Step 5: Sacar el reporte de git**

En `.gitignore`, después de la línea `!output/**/*.csv`, agregar:

```gitignore

# The visual report is derived (~6-7 MB with Plotly embedded): rebuild it with
# `python run_termo.py --solo-reporte output/<fecha> --snapshot data/snapshots/<fecha>`
output/**/reporte.html
```

Run: `git rm --cached output/2026-10-02/reporte.html`
Expected: `rm 'output/2026-10-02/reporte.html'` (el archivo sigue en disco).

- [ ] **Step 6: Comprobar mypy con plotly**

Run: `.venv/Scripts/python -m mypy`
Expected: `Success: no issues found` (aún no se importa plotly en `src`). El override se decide en la Task 6.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore src/termo/operation/visual/__init__.py
git commit -m "build: plotly dependency; visual report kept out of git

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Etiquetas en español

**Files:**
- Create: `src/termo/operation/visual/etiquetas.py`
- Test: `tests/test_visual_etiquetas.py`

- [ ] **Step 1: Escribir las pruebas**

`tests/test_visual_etiquetas.py`:

```python
"""Spanish labels: the six SHAP blocks and every variable of the registered recipe."""

from __future__ import annotations

from pathlib import Path

import pytest

from termo.config import load_config
from termo.dataset import recipe_for
from termo.features.tyccles import TycclesRecipe
from termo.operation.visual.etiquetas import bloque_es, variable_es

REPO = Path(__file__).resolve().parents[1]


def test_the_level_blocks_say_movement_not_level() -> None:
    assert bloque_es("nivel corto") == "movimiento tramo corto (1A-3A)"
    assert bloque_es("nivel medio") == "movimiento tramo medio (5A-7A)"
    assert bloque_es("nivel largo") == "movimiento tramo largo (10A-30A)"
    assert bloque_es("pendientes") == "pendientes"
    assert bloque_es("curvatura") == "curvatura"
    assert bloque_es("volatilidad") == "volatilidad"


def test_an_unknown_block_is_refused() -> None:
    with pytest.raises(ValueError, match="nivel ultra"):
        bloque_es("nivel ultra")


@pytest.mark.parametrize(
    ("nombre", "esperado"),
    [
        ("d5_63_r252", "5A · cambio 3m · rango 1a"),
        ("d1_21_r126", "1A · cambio 1m · rango 6m"),
        ("d30_189_r252", "30A · cambio 9m · rango 1a"),
        ("s12m5s_42_r126", "pendiente 1s5s · cambio 2m · rango 6m"),
        ("s3s10_84_r252", "pendiente 3s10s · cambio 4m · rango 1a"),
        ("s10s30_63_r252", "pendiente 10s30s · cambio 3m · rango 1a"),
        ("c5_189_r252", "curvatura 2·5A−2A−10A · cambio 9m · rango 1a"),
        ("vol10_r252", "10A · volatilidad 21d · rango 1a"),
    ],
)
def test_variables_are_translated(nombre: str, esperado: str) -> None:
    assert variable_es(nombre) == esperado


def test_every_variable_of_the_registered_recipe_is_translated() -> None:
    recipe = recipe_for(load_config(REPO / "configs" / "desc2.yaml"))
    assert isinstance(recipe, TycclesRecipe)
    assert len(recipe.names) == 139
    traducidas = {variable_es(name) for name in recipe.names}
    assert len(traducidas) == 139  # no two variables share a label


@pytest.mark.parametrize("nombre", ["d5_64_r252", "d5_63_r100", "x5_63_r252", "vol10_r126x"])
def test_an_unknown_variable_is_refused(nombre: str) -> None:
    with pytest.raises(ValueError, match="variable"):
        variable_es(nombre)
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `.venv/Scripts/python -m pytest tests/test_visual_etiquetas.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'termo.operation.visual.etiquetas'`

- [ ] **Step 3: Implementar**

`src/termo/operation/visual/etiquetas.py`:

```python
"""Spanish labels for the report: the SHAP blocks and the variables of the TYCCLES recipe.

Only the report renames; configs/desc2.yaml keeps its registered names. The three
"nivel" blocks measure recent CHANGES of the yields, ranked, not the level of rates,
so the report calls them "movimiento".
"""

from __future__ import annotations

import re

BLOQUES: dict[str, str] = {
    "nivel corto": "movimiento tramo corto (1A-3A)",
    "nivel medio": "movimiento tramo medio (5A-7A)",
    "nivel largo": "movimiento tramo largo (10A-30A)",
    "pendientes": "pendientes",
    "curvatura": "curvatura",
    "volatilidad": "volatilidad",
}
HORIZONTES: dict[str, str] = {
    "21": "1m",
    "42": "2m",
    "63": "3m",
    "84": "4m",
    "126": "6m",
    "189": "9m",
}
VENTANAS: dict[str, str] = {"126": "6m", "252": "1a"}
MEDIDAS: dict[str, str] = {
    "s12m5s": "pendiente 1s5s",
    "s3s10": "pendiente 3s10s",
    "s10s30": "pendiente 10s30s",
    "c5": "curvatura 2·5A−2A−10A",
}
CAMBIO = re.compile(r"^(?P<serie>d\d+|s12m5s|s3s10|s10s30|c5)_(?P<h>\d+)_r(?P<w>\d+)$")
VOLATILIDAD = re.compile(r"^vol(?P<plazo>\d+)_r(?P<w>\d+)$")


def bloque_es(nombre: str) -> str:
    """'nivel medio' -> 'movimiento tramo medio (5A-7A)'."""
    try:
        return BLOQUES[nombre]
    except KeyError:
        raise ValueError(f"bloque desconocido: {nombre}") from None


def variable_es(nombre: str) -> str:
    """'d5_63_r252' -> '5A · cambio 3m · rango 1a'; an unknown pattern is an error."""
    if found := CAMBIO.match(nombre):
        serie = found["serie"]
        sujeto = f"{serie[1:]}A" if serie.startswith("d") else MEDIDAS[serie]
        horizonte = _buscar(HORIZONTES, found["h"], nombre)
        ventana = _buscar(VENTANAS, found["w"], nombre)
        return f"{sujeto} · cambio {horizonte} · rango {ventana}"
    if found := VOLATILIDAD.match(nombre):
        ventana = _buscar(VENTANAS, found["w"], nombre)
        return f"{found['plazo']}A · volatilidad 21d · rango {ventana}"
    raise ValueError(f"variable desconocida: {nombre}")


def _buscar(tabla: dict[str, str], clave: str, nombre: str) -> str:
    try:
        return tabla[clave]
    except KeyError:
        raise ValueError(f"variable desconocida: {nombre}") from None
```

- [ ] **Step 4: Correr las pruebas**

Run: `.venv/Scripts/python -m pytest tests/test_visual_etiquetas.py -v`
Expected: PASS (todas).

- [ ] **Step 5: Lint y commit**

Run: `.venv/Scripts/python -m ruff check src tests && .venv/Scripts/python -m mypy`
Expected: limpio.

```bash
git add src/termo/operation/visual/etiquetas.py tests/test_visual_etiquetas.py
git commit -m "feat(visual): Spanish labels for SHAP blocks and recipe variables

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Cálculos descriptivos

**Files:**
- Create: `src/termo/operation/visual/calculos.py`
- Test: `tests/test_visual_calculos.py`

- [ ] **Step 1: Escribir las pruebas**

`tests/test_visual_calculos.py`:

```python
"""The descriptive numbers the visual report adds; each day uses only itself and the past."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from termo.operation.visual.calculos import (
    Cuadrante,
    acuerdo,
    banda_macro,
    bloque_dominante,
    cambio_pasado,
    columna_probabilidad,
    cuadrante,
    curvas_pasadas,
    fase_imitador,
    firma_por_fase,
    posicion_duracion,
)

NOMBRES = ("rally fuerte", "rally moderado", "venta")
DIAS = pd.bdate_range("2020-01-01", periods=60, name="fecha")


def _curva(valores: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(valores, index=DIAS)


def test_probability_column_names_follow_the_export() -> None:
    assert columna_probabilidad("rally fuerte") == "p_rally_fuerte"


def test_trailing_change_uses_only_the_past() -> None:
    curva = _curva({"DGS2": np.arange(60) * 0.01, "DGS10": np.zeros(60)})
    cambio = cambio_pasado(curva, dias=21)
    assert cambio["DGS2"].iloc[:21].isna().all()
    assert cambio["DGS2"].iloc[30] == pytest.approx(21.0)  # 21 days x 1 bp
    futuro = curva.copy()
    futuro.iloc[31:] = 99.0  # rewrite everything after day 30
    assert cambio_pasado(futuro, dias=21)["DGS2"].iloc[30] == pytest.approx(21.0)


def test_signature_is_the_median_trailing_change_per_phase_and_tenor() -> None:
    curva = _curva({"DGS2": np.arange(60) * 0.01, "DGS10": np.arange(60) * -0.02})
    fases = pd.Series([0] * 30 + [2] * 30, index=DIAS)
    firma = firma_por_fase(curva, fases, NOMBRES, dias=21)
    assert list(firma.index) == list(NOMBRES)
    assert list(firma.columns) == ["DGS2", "DGS10"]
    assert firma.loc["rally fuerte", "DGS2"] == pytest.approx(21.0)
    assert firma.loc["venta", "DGS10"] == pytest.approx(-42.0)
    assert firma.loc["rally moderado"].isna().all()  # no day in that phase


def test_surrogate_phase_and_agreement_over_the_last_days() -> None:
    historia = pd.DataFrame(
        {
            "fase": [0, 0, 2, 2],
            "p_rally_fuerte": [0.9, 0.2, 0.1, 0.1],
            "p_rally_moderado": [0.05, 0.7, 0.1, 0.1],
            "p_venta": [0.05, 0.1, 0.8, 0.8],
        },
        index=DIAS[:4],
    )
    assert fase_imitador(historia, NOMBRES).tolist() == [0, 1, 2, 2]
    assert acuerdo(historia, NOMBRES, dias=4) == pytest.approx(0.75)
    assert acuerdo(historia, NOMBRES, dias=2) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("d2", "d10", "esperado"),
    [
        (137.0, 116.0, Cuadrante.BEAR_FLATTENER),
        (20.0, 60.0, Cuadrante.BEAR_STEEPENER),
        (-30.0, -50.0, Cuadrante.BULL_FLATTENER),
        (-80.0, -20.0, Cuadrante.BULL_STEEPENER),
        (10.0, 10.5, Cuadrante.PARALELO),
        (15.0, -10.0, Cuadrante.MIXTO),
    ],
)
def test_quadrant_of_an_episode(d2: float, d10: float, esperado: Cuadrante) -> None:
    assert cuadrante(d2, d10) is esperado


def test_past_curves_count_rows_back_from_the_last_day_on_or_before_the_date() -> None:
    curva = _curva({"DGS2": np.arange(60, dtype=float), "DGS10": np.zeros(60)})
    sabado = DIAS[40] + pd.Timedelta(days=(5 - DIAS[40].dayofweek) % 7 or 7)
    hasta = curva.loc[:sabado]
    curvas = curvas_pasadas(curva, sabado)
    assert list(curvas.index) == ["hoy", "hace 1m"]  # no 63 or 252 rows behind
    assert curvas.loc["hoy", "DGS2"] == hasta["DGS2"].iloc[-1]
    assert curvas.loc["hace 1m", "DGS2"] == hasta["DGS2"].iloc[-22]


def test_duration_position_uses_finished_episodes_only() -> None:
    episodios = pd.DataFrame(
        {"fase": [2, 0, 2, 1, 2, 2, 2], "dias": [40, 10, 80, 30, 120, 160, 500]}
    )  # the last row is the open, current episode: its 500 days must not count
    posicion = posicion_duracion(episodios, fase=2, dias=100)
    assert posicion.episodios == 4  # 40, 80, 120, 160
    assert posicion.mediana == pytest.approx(100.0)
    assert posicion.p25 == pytest.approx(70.0)
    assert posicion.p75 == pytest.approx(130.0)
    assert posicion.relativa == "dentro"
    assert posicion_duracion(episodios, fase=2, dias=200).relativa == "por encima"
    assert posicion_duracion(episodios, fase=2, dias=20).relativa == "por debajo"
    vacia = posicion_duracion(episodios, fase=1, dias=5)
    assert vacia.episodios == 1 and vacia.relativa == "por debajo"
    sin_historia = posicion_duracion(episodios.iloc[:2], fase=0, dias=5)
    assert sin_historia.episodios == 0 and sin_historia.mediana is None
    assert sin_historia.relativa is None


def test_dominant_block_is_the_largest_mean_since_a_date() -> None:
    historia = pd.DataFrame(
        {"nivel medio": [5.0, 0.0, 0.1, 0.1], "pendientes": [0.0, 1.0, 0.5, 0.5]},
        index=DIAS[:4],
    )
    assert bloque_dominante(historia, ("nivel medio", "pendientes"), DIAS[0]) == (
        "nivel medio",
        pytest.approx(1.3),
    )
    nombre, media = bloque_dominante(historia, ("nivel medio", "pendientes"), DIAS[1])
    assert nombre == "pendientes" and media == pytest.approx(2.0 / 3.0)


def test_macro_band_is_trailing_over_known_values() -> None:
    serie = pd.Series(np.arange(10, dtype=float), index=DIAS[:10])
    serie.iloc[3] = np.nan
    banda = banda_macro(serie, ventana=4, minimo=4)
    assert list(banda.columns) == ["p10", "p90"]
    assert DIAS[3] not in banda.index  # holes are not filled
    assert banda["p10"].iloc[:3].isna().all()
    # known values 0,1,2,4 at DIAS[4]: linear quantiles
    assert banda.loc[DIAS[4], "p10"] == pytest.approx(0.3)
    assert banda.loc[DIAS[4], "p90"] == pytest.approx(3.4)
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `.venv/Scripts/python -m pytest tests/test_visual_calculos.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'termo.operation.visual.calculos'`

- [ ] **Step 3: Implementar**

`src/termo/operation/visual/calculos.py`:

```python
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
```

- [ ] **Step 4: Correr las pruebas**

Run: `.venv/Scripts/python -m pytest tests/test_visual_calculos.py -v`
Expected: PASS. Si `test_past_curves...` falla por la fecha del sábado, revisar que `DIAS[40]` cae entre semana (bdate_range) y que el sábado calculado es posterior a él: la prueba exige que la curva use la última fila en o antes de esa fecha.

- [ ] **Step 5: Lint y commit**

Run: `.venv/Scripts/python -m ruff check src tests && .venv/Scripts/python -m mypy`
Expected: limpio.

```bash
git add src/termo/operation/visual/calculos.py tests/test_visual_calculos.py
git commit -m "feat(visual): descriptive calculations (signature, agreement, quadrants)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `DatosReporte` y el fixture sintético

**Files:**
- Create: `src/termo/operation/visual/datos.py`
- Create: `tests/visual_fixture.py`
- Test: `tests/test_visual_datos.py`

- [ ] **Step 1: Escribir `datos.py` (solo la dataclass, para que el fixture pueda importarla)**

`src/termo/operation/visual/datos.py`:

```python
"""What the visual report reads: the week's output directory and its snapshot, checked.

The report is rebuilt from files, never from a model run: the CSV and hoja.json are the
public contract of the deliverable. The snapshot must be the one the sheet names.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from termo.config import CoreConfig
from termo.data.loader import load_curve
from termo.data.snapshot import snapshot_hash
from termo.operation.config import OperationConfig
from termo.operation.macro import _column_name, _label

HOJA = "hoja.json"
HISTORIA = "historia_diaria.csv"
EPISODIOS = "episodios.csv"
MACRO = "macro.csv"
COMENTARIO = "comentario.md"
REQUERIDOS = (HOJA, HISTORIA, EPISODIOS, MACRO)


@dataclass(frozen=True)
class SerieMacro:
    columna: str  # column of macro.csv
    etiqueta: str  # Spanish label, as the sheet shows it


@dataclass(frozen=True, eq=False)
class DatosReporte:
    fecha: pd.Timestamp  # the reading date: the last row of `historia`
    hoja: Mapping[str, Any]  # hoja.json as written by the shadow stage
    historia: pd.DataFrame  # one row per labelled day, index fecha (ns)
    episodios: pd.DataFrame  # one row per episode; the last one is the current, still open
    macro: pd.DataFrame  # index fecha (ns)
    curva: pd.DataFrame  # yields in percent, one column per tenor, index ns
    nombres: tuple[str, ...]  # phase names, by phase number
    bloques: tuple[str, ...]  # SHAP block names, as registered
    series_macro: tuple[SerieMacro, ...]
    textos: Mapping[str, str]  # operacion.yaml texts: descargo, nota_confianza, ...
    ventana_percentil: int  # known values in the macro percentile window
    comentario: str | None  # comentario.md, when the analyst wrote one

    @property
    def lectura(self) -> Mapping[str, Any]:
        lectura: Mapping[str, Any] = self.hoja["reading"]
        return lectura


def _csv(path: Path, fechas: Sequence[str]) -> pd.DataFrame:
    table = pd.read_csv(path, comment="#", parse_dates=list(fechas))
    for column in fechas:
        table[column] = table[column].dt.as_unit("ns")
    return table


def cargar(
    dir_salida: Path, dir_snapshot: Path, config: CoreConfig, op: OperationConfig
) -> DatosReporte:
    """Every input of the report; refused when a file is missing or the snapshot differs."""
    faltan = [name for name in REQUERIDOS if not (dir_salida / name).exists()]
    if faltan:
        raise FileNotFoundError(f"faltan {faltan} en {dir_salida.as_posix()}")
    hoja: dict[str, Any] = json.loads((dir_salida / HOJA).read_text(encoding="utf-8"))
    huella = snapshot_hash(dir_snapshot)
    esperada = str(hoja["snapshot_hash"])
    if huella != esperada:
        raise ValueError(
            f"el snapshot {dir_snapshot.as_posix()} ({huella[:12]}) no es el de la hoja "
            f"({esperada[:12]})"
        )
    desc = config.descriptive
    if desc is None:
        raise ValueError("la configuracion del modelo no es descriptiva")
    historia = _csv(dir_salida / HISTORIA, ["fecha"]).set_index("fecha")
    fecha = pd.Timestamp(hoja["reading_date"]).as_unit("ns")
    if historia.index[-1] != fecha:
        raise ValueError(
            f"la historia termina el {historia.index[-1].date()} y la lectura es del "
            f"{fecha.date()}"
        )
    curva = load_curve(
        dir_snapshot, config.series, config.start, config.holdout_start, final_evaluation=True
    )
    curva.index = pd.DatetimeIndex(curva.index).as_unit("ns")
    comentario = dir_salida / COMENTARIO
    return DatosReporte(
        fecha=fecha,
        hoja=hoja,
        historia=historia,
        episodios=_csv(dir_salida / EPISODIOS, ["inicio", "fin"]),
        macro=_csv(dir_salida / MACRO, ["fecha"]).set_index("fecha"),
        curva=curva,
        nombres=tuple(desc.phase_names),
        bloques=tuple(name for name, _ in desc.blocks),
        series_macro=tuple(SerieMacro(_column_name(m), _label(m)) for m in op.macro),
        textos=dict(op.texts),
        ventana_percentil=op.percentile_window_days,
        comentario=comentario.read_text(encoding="utf-8") if comentario.exists() else None,
    )
```

Nota: `_column_name` y `_label` son privadas en `macro.py`; `tests/test_exports.py` ya importa `_column_name` igual. Se importan en lugar de duplicar la regla de nombres.

- [ ] **Step 2: Escribir el fixture sintético**

`tests/visual_fixture.py`:

```python
"""A small, consistent input for the visual report tests: in memory and on disk.

700 business days from 1990 with seven episodes, the last one an open 'venta'; the
surrogate disagrees with the jump model one day in fifty; the last 200 days are the
holdout and the last 2 the shadow, as in the real export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from conftest import FRED_URL_TEMPLATE, TENORS, fake_fred, make_curve, make_macro
from termo.data.snapshot import snapshot_hash, write_snapshot
from termo.operation.config import load_operation_config
from termo.operation.monthly import episodes
from termo.operation.visual.calculos import columna_probabilidad
from termo.operation.visual.datos import DatosReporte, SerieMacro

REPO = Path(__file__).resolve().parents[1]
OP_CONFIG = REPO / "configs" / "operacion.yaml"
NOMBRES = ("rally fuerte", "rally moderado", "venta")
BLOQUES = ("nivel corto", "nivel medio", "nivel largo", "pendientes", "curvatura", "volatilidad")
N_DIAS = 700
RACHAS = ((0, 60), (2, 90), (1, 70), (2, 120), (0, 40), (1, 100))  # then 'venta' to the end
VENTA = 2
HOLDOUT_DIAS, SOMBRA_DIAS = 200, 2
HUELLA_FALSA = "0" * 64
GENERADO = "1992-09-01T00:00:00+00:00"
PLOTLY_INICIO = get_plotlyjs()[:120]


def sin_plotlyjs(page: str) -> str:
    """The page without the embedded plotly.js bundle (which mentions URLs in its code)."""
    return page.replace(get_plotlyjs(), "")


def _iso(index: pd.Index) -> list[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in index]


def curva_sintetica() -> pd.DataFrame:
    curve, _ = make_curve(N_DIAS, seed=1)
    curva = curve.round(4)
    curva.index = pd.DatetimeIndex(curva.index).as_unit("ns")
    return curva


def fases_sinteticas(index: pd.DatetimeIndex) -> pd.Series:
    valores: list[int] = []
    for fase, dias in RACHAS:
        valores += [fase] * dias
    valores += [VENTA] * (len(index) - len(valores))
    return pd.Series(valores[: len(index)], index=index, name="fase")


def historia_sintetica(curva: pd.DataFrame) -> pd.DataFrame:
    index = pd.DatetimeIndex(curva.index, name="fecha")
    fases = fases_sinteticas(index)
    n = len(index)
    posicion = np.arange(n)
    proba = np.full((n, len(NOMBRES)), 0.05)
    proba[posicion, fases.to_numpy()] = 0.9
    desacuerdo = posicion % 50 == 7  # the surrogate names another phase one day in fifty
    otra = (fases.to_numpy() + 1) % len(NOMBRES)
    proba[desacuerdo] = 0.05
    proba[desacuerdo, otra[desacuerdo]] = 0.9
    table = pd.DataFrame(
        {"fase": fases.to_numpy(), "nombre": [NOMBRES[f] for f in fases]}, index=index
    )
    for i, nombre in enumerate(NOMBRES):
        table[columna_probabilidad(nombre)] = proba[:, i]
    rng = np.random.default_rng(3)
    for bloque in BLOQUES:
        table[bloque] = rng.normal(0.0, 0.5, n).round(4)
    table["base"] = 0.3
    periodo = np.full(n, "pre_holdout", dtype=object)
    periodo[n - HOLDOUT_DIAS - SOMBRA_DIAS : n - SOMBRA_DIAS] = "holdout"
    periodo[n - SOMBRA_DIAS :] = "sombra"
    table["periodo"] = periodo.astype(str)
    return table


def episodios_sinteticos(historia: pd.DataFrame, curva: pd.DataFrame) -> pd.DataFrame:
    table = episodes(historia["fase"], curva, NOMBRES)
    table["inicio"] = pd.DatetimeIndex(table["inicio"]).as_unit("ns")
    table["fin"] = pd.DatetimeIndex(table["fin"]).as_unit("ns")
    table["periodo"] = historia["periodo"].reindex(table["inicio"]).to_numpy()
    return table


def macro_sintetica(curva: pd.DataFrame) -> pd.DataFrame:
    crudo = make_macro(curva)
    prima = crudo["THREEFYTP10"].reindex(curva.index)
    prima[prima.index.dayofweek != 4] = np.nan  # weekly, as Kim-Wright arrives
    spread = curva["DGS2"] - crudo["DFF"].reindex(curva.index)
    table = pd.DataFrame({"THREEFYTP10": prima, "DGS2_menos_DFF": spread})
    table.index = pd.DatetimeIndex(curva.index, name="fecha")
    return table


def hoja_sintetica(
    historia: pd.DataFrame, episodios: pd.DataFrame, huella: str
) -> dict[str, Any]:
    fecha = _iso(historia.index[-1:])[0]
    actual = episodios.iloc[-1]
    fila = historia.iloc[-1]
    drivers = sorted(
        ({"block": b, "contribution": float(fila[b])} for b in BLOQUES),
        key=lambda d: -float(d["contribution"]),
    )[:3]
    return {
        "kind": "reading",
        "reading_date": fecha,
        "snapshot_hash": huella,
        "snapshot_downloaded_at": GENERADO,
        "run_at": GENERADO,
        "code_commit": "test-commit-visual",
        "alert": None,
        "macro": [
            {
                "serie": "prima por plazo 10 anos (Kim-Wright)",
                "valor": 1.02,
                "fecha_valor": fecha,
                "percentil_10a": 0.97,
                "cambio_21d": 0.18,
                "dias_de_ventana": 2520,
                "nota": "",
            },
            {
                "serie": "DGS2 - fed funds efectiva",
                "valor": 0.5,
                "fecha_valor": fecha,
                "percentil_10a": 0.4,
                "cambio_21d": -0.05,
                "dias_de_ventana": 2520,
                "nota": "",
            },
        ],
        "reading": {
            "date": fecha,
            "phase": VENTA,
            "phase_name": NOMBRES[VENTA],
            "confidence": 0.9,
            "low_confidence": False,
            "days_in_phase": int(actual["dias"]),
            "episode_start": _iso(pd.DatetimeIndex([actual["inicio"]]))[0],
            "probabilities": {n: float(fila[columna_probabilidad(n)]) for n in NOMBRES},
            "drivers": drivers,
            "drivers_validated": True,
            "surrogate_agrees": True,
            "top_variables": [
                {"variable": "d5_63_r252", "contribution": 0.91},
                {"variable": "d7_84_r252", "contribution": 0.5},
                {"variable": "c5_189_r252", "contribution": 0.3},
                {"variable": "vol10_r252", "contribution": -0.12},
                {"variable": "s10s30_63_r252", "contribution": 0.24},
            ],
            "validation": {
                "by_phase": [
                    {
                        "phase": i,
                        "name": nombre,
                        "days": 100,
                        "evaluable": i != 0,
                        "median_duration_days": 50.0,
                        "direction_share": 0.8,
                        "recall": 0.9,
                        "episodes": 2,
                    }
                    for i, nombre in enumerate(NOMBRES)
                ],
                "registered_verdicts": [
                    {"stage": "diagnostic", "verdict": "apto"},
                    {"stage": "holdout", "verdict": "apto"},
                ],
                "diagnostic": "apto",
                "holdout": "apto",
                "holdout_seen_by": "desc_k3 (test)",
                "failed_checks": [],
                "fidelity_failed": False,
                "sombra": {
                    "semanas": 1,
                    "proxima_evaluacion_semanas": 25,
                    "historia_reproducida": True,
                },
            },
        },
    }


def make_datos(huella: str = HUELLA_FALSA, comentario: str | None = None) -> DatosReporte:
    op = load_operation_config(OP_CONFIG)
    curva = curva_sintetica()
    historia = historia_sintetica(curva)
    episodios = episodios_sinteticos(historia, curva)
    return DatosReporte(
        fecha=pd.Timestamp(historia.index[-1]),
        hoja=hoja_sintetica(historia, episodios, huella),
        historia=historia,
        episodios=episodios,
        macro=macro_sintetica(curva),
        curva=curva,
        nombres=NOMBRES,
        bloques=BLOQUES,
        series_macro=(
            SerieMacro("THREEFYTP10", "prima por plazo 10 anos (Kim-Wright)"),
            SerieMacro("DGS2_menos_DFF", "DGS2 - fed funds efectiva"),
        ),
        textos=dict(op.texts),
        ventana_percentil=op.percentile_window_days,
        comentario=comentario,
    )


def escribir_snapshot(dir_snapshot: Path, curva: pd.DataFrame) -> str:
    """The seven tenors as FRED serves them; returns the snapshot hash."""
    get = fake_fred(curva)
    texts = {s: get(FRED_URL_TEMPLATE.format(series_id=s)) for s in TENORS}
    write_snapshot(dir_snapshot, texts, GENERADO)
    return snapshot_hash(dir_snapshot)


def _escribir_csv(path: Path, table: pd.DataFrame, huella: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# snapshot_hash={huella}\n# generado={GENERADO}\n")
        table.to_csv(handle, index=False, lineterminator="\n")


def escribir_salida(dir_salida: Path, datos: DatosReporte) -> None:
    """hoja.json and the CSV in the export's format (dates as ISO strings)."""
    dir_salida.mkdir(parents=True, exist_ok=True)
    huella = str(datos.hoja["snapshot_hash"])
    (dir_salida / "hoja.json").write_text(
        json.dumps(datos.hoja, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    historia = datos.historia.reset_index()
    historia["fecha"] = _iso(pd.DatetimeIndex(historia["fecha"]))
    _escribir_csv(dir_salida / "historia_diaria.csv", historia, huella)
    episodios = datos.episodios.copy()
    episodios["inicio"] = _iso(pd.DatetimeIndex(episodios["inicio"]))
    episodios["fin"] = _iso(pd.DatetimeIndex(episodios["fin"]))
    _escribir_csv(dir_salida / "episodios.csv", episodios, huella)
    macro = datos.macro.reset_index()
    macro["fecha"] = _iso(pd.DatetimeIndex(macro["fecha"]))
    _escribir_csv(dir_salida / "macro.csv", macro, huella)


def en_disco(raiz: Path, comentario: str | None = None) -> tuple[DatosReporte, Path, Path]:
    """The synthetic week written to `raiz`: (datos with the real hash, output dir, snapshot)."""
    dir_snapshot = raiz / "snapshot"
    huella = escribir_snapshot(dir_snapshot, curva_sintetica())
    datos = make_datos(huella, comentario)
    dir_salida = raiz / "salida"
    escribir_salida(dir_salida, datos)
    if comentario is not None:
        (dir_salida / "comentario.md").write_text(comentario, encoding="utf-8", newline="\n")
    return datos, dir_salida, dir_snapshot
```

- [ ] **Step 3: Escribir las pruebas de `cargar`**

`tests/test_visual_datos.py`:

```python
"""Loading the week for the visual report: files checked, snapshot matched, dates aligned."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_desc2_config
from termo.operation.config import load_operation_config
from termo.operation.visual.datos import DatosReporte, cargar
from visual_fixture import BLOQUES, NOMBRES, OP_CONFIG, en_disco


def _cargar(salida: Path, snapshot: Path) -> DatosReporte:
    return cargar(salida, snapshot, make_desc2_config(), load_operation_config(OP_CONFIG))


def test_the_week_round_trips_from_disk(tmp_path: Path) -> None:
    datos, salida, snapshot = en_disco(tmp_path, comentario="## Nota\n\nTexto del analista.")
    cargado = _cargar(salida, snapshot)
    assert cargado.fecha == datos.fecha
    assert cargado.hoja == datos.hoja
    assert cargado.nombres == NOMBRES and cargado.bloques == BLOQUES
    assert list(cargado.historia.index) == list(datos.historia.index)
    assert list(cargado.historia.columns) == list(datos.historia.columns)
    assert np.allclose(cargado.historia[list(BLOQUES)], datos.historia[list(BLOQUES)])
    assert cargado.historia["periodo"].tolist() == datos.historia["periodo"].tolist()
    assert cargado.episodios["inicio"].tolist() == datos.episodios["inicio"].tolist()
    assert cargado.episodios["dias"].tolist() == datos.episodios["dias"].tolist()
    assert list(cargado.curva.columns) == list(datos.curva.columns)
    assert np.allclose(cargado.curva.to_numpy(), datos.curva.to_numpy())
    assert [s.columna for s in cargado.series_macro] == ["THREEFYTP10", "DGS2_menos_DFF"]
    assert cargado.series_macro[1].etiqueta == "DGS2 - fed funds efectiva"
    assert cargado.comentario == "## Nota\n\nTexto del analista."
    assert cargado.ventana_percentil == 2520
    assert cargado.lectura["phase_name"] == "venta"


def test_without_a_comment_there_is_none(tmp_path: Path) -> None:
    _, salida, snapshot = en_disco(tmp_path)
    assert _cargar(salida, snapshot).comentario is None


def test_a_snapshot_that_is_not_the_sheets_is_refused(tmp_path: Path) -> None:
    _, salida, snapshot = en_disco(tmp_path)
    hoja = json.loads((salida / "hoja.json").read_text(encoding="utf-8"))
    hoja["snapshot_hash"] = "f" * 64
    (salida / "hoja.json").write_text(json.dumps(hoja), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot") as error:
        _cargar(salida, snapshot)
    assert str(error.value).isascii() and "ffffffffffff" in str(error.value)


@pytest.mark.parametrize(
    "archivo", ["hoja.json", "historia_diaria.csv", "episodios.csv", "macro.csv"]
)
def test_a_missing_required_file_is_named(tmp_path: Path, archivo: str) -> None:
    _, salida, snapshot = en_disco(tmp_path)
    (salida / archivo).unlink()
    with pytest.raises(FileNotFoundError, match=archivo.replace(".", r"\.")):
        _cargar(salida, snapshot)


def test_a_history_that_does_not_end_on_the_reading_date_is_refused(tmp_path: Path) -> None:
    _, salida, snapshot = en_disco(tmp_path)
    hoja = json.loads((salida / "hoja.json").read_text(encoding="utf-8"))
    hoja["reading_date"] = (pd.Timestamp(hoja["reading_date"]) - pd.Timedelta(days=7)).strftime(
        "%Y-%m-%d"
    )
    (salida / "hoja.json").write_text(json.dumps(hoja), encoding="utf-8")
    with pytest.raises(ValueError, match="la historia termina"):
        _cargar(salida, snapshot)
```

- [ ] **Step 4: Correr las pruebas**

Run: `.venv/Scripts/python -m pytest tests/test_visual_datos.py -v`
Expected: PASS. Si `test_the_week_round_trips_from_disk` falla en `cargado.hoja == datos.hoja`, comparar las claves: el fixture escribe la hoja con `json.dumps` y la vuelve a leer, así que los floats deben coincidir; un `numpy.float64` dentro de `hoja_sintetica` sí se serializa igual. Si falla en el índice de la historia, verificar que `_csv` convierte a `ns` y que el fixture también usa `ns`.

- [ ] **Step 5: Lint y commit**

Run: `.venv/Scripts/python -m ruff check src tests && .venv/Scripts/python -m mypy`
Expected: limpio.

```bash
git add src/termo/operation/visual/datos.py tests/visual_fixture.py tests/test_visual_datos.py
git commit -m "feat(visual): load and check the week's files for the report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Narrativa por reglas

**Files:**
- Create: `src/termo/operation/visual/narrativa.py`
- Test: `tests/test_visual_narrativa.py`

- [ ] **Step 1: Escribir las pruebas**

`tests/test_visual_narrativa.py`:

```python
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
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `.venv/Scripts/python -m pytest tests/test_visual_narrativa.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'termo.operation.visual.narrativa'`

- [ ] **Step 3: Implementar**

`src/termo/operation/visual/narrativa.py`:

```python
"""Headlines and bullets of the visual report: closed templates filled with the data.

Every sentence describes the past or the present; none uses the future or the
conditional (tests/test_visual_narrativa.py enforces it). Numbers come only from
DatosReporte and calculos.
"""

from __future__ import annotations

from dataclasses import dataclass

from termo.operation.monthly import transitions
from termo.operation.visual import calculos
from termo.operation.visual.calculos import Cuadrante, Posicion
from termo.operation.visual.datos import DatosReporte
from termo.operation.visual.etiquetas import bloque_es

CLAVES = (
    "portada",
    "tasa-10a",
    "imitador",
    "motores",
    "motores-tiempo",
    "curva",
    "firma",
    "episodios",
    "transiciones",
    "macro",
)
PLAZOS = {"DGS1": "1A", "DGS2": "2A", "DGS3": "3A", "DGS5": "5A", "DGS7": "7A", "DGS10": "10A",
          "DGS30": "30A"}


@dataclass(frozen=True)
class Titular:
    texto: str
    vinetas: tuple[str, ...] = ()


def _capital(texto: str) -> str:
    return texto[:1].upper() + texto[1:]


def _movimiento(pb: float) -> str:
    if round(pb) == 0:
        return "no cambió"
    return f"{'subió' if pb > 0 else 'bajó'} {abs(pb):.0f} pb"


def titular_portada(fase: str, posicion: Posicion) -> str:
    base = f"La curva está en {fase}: {posicion.dias} días"
    if posicion.relativa is None:
        return base
    return f"{base}, {posicion.relativa} del rango intercuartil histórico"


def titular_10a(fase: str, inicio: str, cambio_pb: float) -> str:
    return f"El 10A {_movimiento(cambio_pb)} desde que empezó el episodio de {fase} ({inicio})"


def titular_acuerdo(acuerdo: float) -> str:
    return f"Imitador y fase coinciden en el {acuerdo:.0%} de los días del último año"


def titular_motor(bloque: str, aporte: float) -> str:
    return f"{_capital(bloque_es(bloque))} es el bloque de mayor aporte a la lectura ({aporte:+.2f})"


def titular_motores_tiempo(bloque: str, inicio: str) -> str:
    return f"Desde {inicio}, el bloque de mayor aporte medio es {bloque_es(bloque)}"


def titular_curva(cambio_2y: float, cambio_10y: float) -> str:
    diferencia = cambio_2y - cambio_10y
    if abs(diferencia) < calculos.TOLERANCIA_PARALELO_PB:
        forma = "se movió en paralelo"
    else:
        forma = "se aplanó" if diferencia > 0 else "se empinó"
    return (
        f"En 3 meses el 2A {_movimiento(cambio_2y)} y el 10A {_movimiento(cambio_10y)}: "
        f"la curva {forma}"
    )


def titular_firma(fase: str, plazo: str, valor_pb: float) -> str:
    return (
        f"En {fase}, el plazo que más se mueve en un mes (mediana) es el {plazo} "
        f"({valor_pb:+.0f} pb)"
    )


def titular_episodios(fase: str, cuadrante: Cuadrante, n: int, total: int) -> str:
    return f"{n} de {total} episodios de {fase} fueron {cuadrante.value}"


def titular_transicion(
    fase: str, siguiente: str | None, proporcion: float | None, total: int
) -> str:
    if total == 0 or siguiente is None or proporcion is None:
        return f"No hay episodios de {fase} con sucesor"
    return f"Tras {fase}, el {proporcion:.0%} de los episodios siguió con {siguiente}"


def titular_macro(etiqueta: str, percentil: float) -> str:
    return f"{_capital(etiqueta)} está en el percentil {percentil * 100:.0f} a 10 años"


def titulares(datos: DatosReporte) -> dict[str, Titular]:
    """One headline per block of the page (keys in CLAVES); the cover has three bullets."""
    lectura = datos.lectura
    fase_n = int(lectura["phase"])
    fase = datos.nombres[fase_n]
    actual = datos.episodios.iloc[-1]
    inicio = actual["inicio"].strftime("%Y-%m-%d")
    historia = datos.historia.loc[: datos.fecha]

    posicion = calculos.posicion_duracion(datos.episodios, fase_n, int(lectura["days_in_phase"]))
    coincidencia = calculos.acuerdo(historia, datos.nombres)
    fila = historia.iloc[-1]
    motor = max(datos.bloques, key=lambda b: float(fila[b]))
    dominante, _ = calculos.bloque_dominante(historia, datos.bloques, actual["inicio"])

    curvas = calculos.curvas_pasadas(datos.curva, datos.fecha)
    if "hace 3m" in curvas.index:
        cambio = (curvas.loc["hoy"] - curvas.loc["hace 3m"]) * 100.0
        curva = titular_curva(float(cambio["DGS2"]), float(cambio["DGS10"]))
    else:
        curva = "La curva aún no tiene 3 meses de historia"

    firma = calculos.firma_por_fase(datos.curva, historia["fase"], datos.nombres).loc[fase]
    plazo = str(firma.abs().idxmax())

    cuadrantes = calculos.cuadrantes(datos.episodios)[datos.episodios["fase"] == fase_n]
    moda = Cuadrante(cuadrantes.mode().iloc[0])
    total_episodios = int(len(cuadrantes))
    n_moda = int((cuadrantes == moda).sum())

    salida = transitions(datos.episodios, datos.nombres)[fase_n]
    validos = {k: v for k, v in salida["a"].items() if v is not None}
    siguiente = max(validos, key=lambda k: validos[k]) if validos else None

    macro = max(datos.hoja["macro"], key=lambda m: abs(float(m["percentil_10a"]) - 0.5))
    texto_macro = titular_macro(str(macro["serie"]), float(macro["percentil_10a"]))
    texto_acuerdo = titular_acuerdo(coincidencia)

    return {
        "portada": Titular(
            titular_portada(fase, posicion),
            (titular_motor(motor, float(fila[motor])), texto_acuerdo, texto_macro),
        ),
        "tasa-10a": Titular(titular_10a(fase, inicio, float(actual["cambio_10y_pb"]))),
        "imitador": Titular(texto_acuerdo),
        "motores": Titular(titular_motor(motor, float(fila[motor]))),
        "motores-tiempo": Titular(titular_motores_tiempo(dominante, inicio)),
        "curva": Titular(curva),
        "firma": Titular(titular_firma(fase, PLAZOS.get(plazo, plazo), float(firma[plazo]))),
        "episodios": Titular(titular_episodios(fase, moda, n_moda, total_episodios)),
        "transiciones": Titular(
            titular_transicion(
                fase,
                siguiente,
                None if siguiente is None else validos[siguiente],
                int(salida["total"]),
            )
        ),
        "macro": Titular(texto_macro),
    }
```

Nota: "La curva aún no tiene 3 meses de historia" usa presente; "aún" no dispara el regex.

- [ ] **Step 4: Correr las pruebas**

Run: `.venv/Scripts/python -m pytest tests/test_visual_narrativa.py -v`
Expected: PASS.

- [ ] **Step 5: Lint y commit**

Run: `.venv/Scripts/python -m ruff check src tests && .venv/Scripts/python -m mypy`
Expected: limpio. (`ruff format` no se exige en el repo; si `E501` se queja del dict `PLAZOS`, partirlo en una entrada por línea.)

```bash
git add src/termo/operation/visual/narrativa.py tests/test_visual_narrativa.py
git commit -m "feat(visual): rule-based headlines and bullets, present tense only

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Gráficas Plotly

**Files:**
- Create: `src/termo/operation/visual/graficas.py`
- Modify (si mypy lo pide): `pyproject.toml`
- Test: `tests/test_visual_graficas.py`

- [ ] **Step 1: Escribir las pruebas**

`tests/test_visual_graficas.py`:

```python
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
        assert fig.layout.yaxis2.overlaying is None  # never a dual axis
        assert fig.data[1].fill == "tonexty"
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `.venv/Scripts/python -m pytest tests/test_visual_graficas.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'termo.operation.visual.graficas'`

- [ ] **Step 3: Implementar**

`src/termo/operation/visual/graficas.py`:

```python
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
GRIS = "#888780"
TINTA = "#4a4843"
EJE = "#c3c2b7"
RETICULA = "rgba(137,135,129,0.18)"
COLOR_MACRO = "#6250d6"
GRISES_CURVA = (("hace 1m", "#888780", "dot"), ("hace 3m", "#b4b2a9", "dash"),
                ("hace 1a", "#d3d1c7", "longdash"))
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
    """The 10Y coloured by the jump model's phase; a run reaches into the next day."""
    historia = _historia(datos)
    fases = historia["fase"]
    tasa = datos.curva["DGS10"].reindex(historia.index)
    dias = _dias(historia.index)
    fig = go.Figure()
    for fase, nombre in enumerate(datos.nombres):
        dentro = (fases == fase) | (fases.shift(1) == fase)
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
    """Jump-model phase and surrogate phase as strips, the surrogate's probabilities below."""
    historia = _historia(datos)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.08, 0.08, 0.84],
                        vertical_spacing=0.02)
    fig.add_trace(_franja(historia["fase"], datos.nombres, "fase"), row=1, col=1)
    imitador = calculos.fase_imitador(historia, datos.nombres)
    fig.add_trace(_franja(imitador, datos.nombres, "imitador"), row=2, col=1)
    dias = _dias(historia.index)
    for fase, nombre in enumerate(datos.nombres):
        fig.add_trace(
            go.Scatter(
                x=dias, y=_valores(historia[calculos.columna_probabilidad(nombre)]),
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
    fase = int(datos.lectura["phase"])
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
            decreasing={"marker": {"color": GRIS}},
            totals={"marker": {"color": TINTA}},
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
    """Each block's contribution to the day's own phase, stacked by sign."""
    historia = _historia(datos)
    dias = _dias(historia.index)
    fig = go.Figure()
    for posicion, bloque in enumerate(datos.bloques):
        valor = historia[bloque].fillna(0.0)
        nombre = bloque_es(bloque)
        for grupo, parte in (("pos", valor.clip(lower=0.0)), ("neg", valor.clip(upper=0.0))):
            fig.add_trace(
                go.Scatter(
                    x=dias, y=_valores(parte), customdata=_valores(valor), name=nombre,
                    legendgroup=bloque, showlegend=grupo == "pos", stackgroup=grupo,
                    mode="lines", line={"width": 0},
                    fillcolor=_rgba(_color_bloque(posicion), 0.75),
                    hovertemplate="%{customdata:+.2f}<extra>" + nombre + "</extra>"
                    if grupo == "pos" else None,
                    hoverinfo=None if grupo == "pos" else "skip",
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
    estilos.update({nombre: (color, guion, 1.8) for nombre, color, guion in GRISES_CURVA})
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
    fig.add_hline(y=0, line_color=EJE, line_width=1)
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
            marker={"symbol": "circle-open", "size": 22, "line": {"width": 3}, "color": TINTA},
            hovertemplate="episodio actual<extra></extra>",
        )
    )
    tope = 1.05 * max(
        1.0, float(episodios["cambio_2y_pb"].abs().max()),
        float(episodios["cambio_10y_pb"].abs().max()),
    )
    fig.add_shape(type="line", x0=-tope, y0=-tope, x1=tope, y1=tope,
                  line={"color": EJE, "dash": "dot", "width": 1})
    fig.add_hline(y=0, line_color=EJE, line_width=1)
    fig.add_vline(x=0, line_color=EJE, line_width=1)
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
            texttemplate="%{text}", colorscale=[[0.0, "#f0efec"], [1.0, "#2a78d6"]],
            zmin=0, zmax=100, showscale=False,
            hovertemplate="de %{y} a %{x}: %{text}<extra></extra>",
        )
    )
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
                   marker={"symbol": "diamond", "size": 14, "color": TINTA},
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dias, y=_valores(banda["p90"]), mode="lines",
                             line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dias, y=_valores(banda["p10"]), mode="lines", line={"width": 0},
                             fill="tonexty", fillcolor=_rgba(GRIS, 0.18),
                             name="P10-P90 a 10 años", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dias, y=_valores(conocida), mode="lines", name=serie.etiqueta,
                             line={"color": COLOR_MACRO, "width": 1.6},
                             hovertemplate="%{y:.2f}<extra></extra>"))
    fig.update_layout(template=TEMA, height=320)
    _selector_rango(fig, datos.fecha, anios=10)
    return fig
```

- [ ] **Step 4: Correr las pruebas**

Run: `.venv/Scripts/python -m pytest tests/test_visual_graficas.py -v`
Expected: PASS. Puntos a revisar si algo falla:
- `test_macro_has_one_axis_and_a_band`: en Plotly, `fig.layout.yaxis2` de una figura sin segundo eje devuelve un objeto vacío con `overlaying is None`. Si la versión instalada lanza error al acceder, cambiar la aserción por `assert "yaxis2" not in fig.to_dict()["layout"]`.
- `test_drivers_over_time`: si `hovertemplate=None` es rechazado en las trazas negativas, quitar ese argumento en lugar de pasar `None` (con `hoverinfo="skip"` basta).

- [ ] **Step 5: mypy con plotly**

Run: `.venv/Scripts/python -m mypy`
Expected: limpio. Si sale `error: Skipping analyzing "plotly...": module is installed, but missing library stubs or py.typed marker`, agregar `"plotly.*"` a la lista `module = [...]` del override existente en `pyproject.toml`:

```toml
module = ["jumpmodels.*", "sklearn.*", "scipy.*", "pandas.*", "plotly.*"]
```

y volver a correr mypy hasta que quede limpio.

- [ ] **Step 6: Lint y commit**

Run: `.venv/Scripts/python -m ruff check src tests`
Expected: limpio (partir líneas > 100 si `E501` aparece).

```bash
git add src/termo/operation/visual/graficas.py tests/test_visual_graficas.py pyproject.toml
git commit -m "feat(visual): Plotly figures with the TERMO template

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: La página

**Files:**
- Create: `src/termo/operation/visual/pagina.py`
- Test: `tests/test_visual_pagina.py`

- [ ] **Step 1: Escribir las pruebas**

`tests/test_visual_pagina.py`:

```python
"""The whole page: self-contained, Plotly once, every block, escaped, comment optional."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from conftest import make_desc2_config
from termo.operation.config import load_operation_config
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
    from dataclasses import replace

    textos = {**datos.textos, "descargo": "a <b> & c"}
    page = render(replace(datos, textos=textos), GENERADO)
    assert "a &lt;b&gt; &amp; c" in page and "a <b> & c" not in page


def test_the_analyst_comment_appears_only_when_written(datos: DatosReporte) -> None:
    from dataclasses import replace

    page = render(replace(datos, comentario="## Lectura\n\nLa curva <sube>."), GENERADO)
    assert page.count('<section id="comentario"') == 1
    assert "Comentario del analista (no generado por TERMO)" in page
    assert "<h2>Lectura</h2>" in page and "La curva &lt;sube&gt;." in page


def test_build_writes_the_report_next_to_the_files(tmp_path: Path) -> None:
    _, salida, snapshot = en_disco(tmp_path)
    path = construir(
        salida, snapshot, make_desc2_config(), load_operation_config(OP_CONFIG), GENERADO
    )
    assert path == salida / ARCHIVO
    raw = path.read_bytes()
    assert b"\r\n" not in raw and raw.decode("utf-8").count(PLOTLY_INICIO) == 1
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `.venv/Scripts/python -m pytest tests/test_visual_pagina.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'termo.operation.visual.pagina'`

- [ ] **Step 3: Implementar**

`src/termo/operation/visual/pagina.py`:

```python
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
from termo.operation.sheet_es import SEEN_BY_NOTE
from termo.operation.visual import graficas as g
from termo.operation.visual.calculos import Posicion, posicion_duracion
from termo.operation.visual.datos import DatosReporte, cargar
from termo.operation.visual.narrativa import Titular, titulares

ARCHIVO = "reporte.html"
CONFIG_PLOTLY = {"displayModeBar": False, "responsive": True}
FUENTE = "Fuente: FRED, TERMO."
FUENTE_MACRO = "Fuente: FRED (Kim-Wright, fed funds efectiva), TERMO."
LEYENDA_PASADO = "Frecuencias del pasado, no pronóstico. Incluye el periodo holdout, ya abierto."
NOTA_MOTORES = (
    "Cada día explica su propia fase (log-odds); cuando cambia la fase, cambia lo que se explica."
)
ANCLAS = (
    "portada", "tasa-10a", "imitador", "motores", "motores-tiempo", "curva", "episodios",
    "transiciones", "macro", "validacion",
)

ESTILO = """
:root{--papel:#fbfaf7;--tinta:#1d1c1a;--tinta-2:#4a4843;--tinta-3:#77756e;--filete:#d9d6cc;
--tarjeta:#f3f1ea}
@media (prefers-color-scheme: dark){:root{--papel:#161615;--tinta:#f0efec;--tinta-2:#c3c2b7;
--tinta-3:#898781;--filete:#383835;--tarjeta:#1f1f1d}}
*{box-sizing:border-box}
body{margin:0;background:var(--papel);color:var(--tinta);
font:16px/1.6 "Segoe UI",Helvetica,Arial,sans-serif}
.filete{height:6px;background:var(--acento)}
.marco{display:grid;grid-template-columns:200px minmax(0,1fr);gap:48px;max-width:1220px;
margin:0 auto;padding:32px 24px}
nav{position:sticky;top:24px;align-self:start;font-size:14px}
nav a{display:block;color:var(--tinta-2);text-decoration:none;padding:4px 0 4px 10px;
border-left:2px solid var(--filete)}
nav a:hover{color:var(--tinta);border-left-color:var(--acento)}
main{min-width:0;max-width:920px}
h1,h2{font-family:Georgia,Cambria,"Times New Roman",serif;font-weight:400;line-height:1.2}
h1{font-size:56px;margin:8px 0 4px;color:var(--acento)}
h2{font-size:26px;margin:4px 0 12px}
.meta,.kicker{font-size:13px;color:var(--tinta-3);margin:0}
.titular{font-family:Georgia,Cambria,serif;font-size:28px;line-height:1.25;margin:8px 0 16px}
section{padding:40px 0;border-top:1px solid var(--filete)}
section#portada{border-top:0;padding-top:8px}
ul.vinetas{list-style:none;padding:0;margin:0 0 16px}
ul.vinetas li{padding:2px 0}
ul.vinetas li::before{content:"\\25C6";color:var(--acento);margin-right:10px;font-size:12px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px;margin:24px 0}
.kpi{border-top:2px solid var(--tinta);padding-top:8px}
.kpi .valor{font-family:Georgia,Cambria,serif;font-size:32px;line-height:1.1}
.kpi .etiqueta,.kpi .sub{font-size:13px;color:var(--tinta-3)}
.termometro{margin:8px 0 0}
.termometro .pista{position:relative;height:14px;background:var(--tarjeta);border-radius:4px}
.termometro .banda{position:absolute;top:0;height:100%;background:var(--acento);opacity:.28}
.termometro .mediana{position:absolute;top:0;width:2px;height:100%;background:var(--tinta-2)}
.termometro .hoy{position:absolute;top:-4px;width:10px;height:22px;background:var(--acento);
border-radius:3px;transform:translateX(-50%)}
.termometro .escala{display:flex;justify-content:space-between;font-size:12px;
color:var(--tinta-3);margin-top:6px}
.aviso{display:inline-block;font-size:13px;padding:2px 8px;border:1px solid var(--acento);
border-radius:4px;color:var(--acento);margin-right:8px}
figure.grafica{margin:24px 0}
figcaption{font-weight:600;font-size:15px;margin-bottom:6px}
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
.titular{font-size:22px}}
@media print{nav{display:none}.marco{display:block;padding:0}
figure.grafica,.kpis,.tarjetas{break-inside:avoid}}
"""

MODO_OSCURO = """
(function(){var q=window.matchMedia('(prefers-color-scheme: dark)');
function aplicar(){var o=q.matches,t=o?'#c3c2b7':'#4a4843',
r=o?'rgba(255,255,255,0.10)':'rgba(137,135,129,0.18)';
document.querySelectorAll('.js-plotly-plot').forEach(function(el){
Plotly.relayout(el,{'font.color':t,'yaxis.gridcolor':r});});}
window.addEventListener('load',aplicar);q.addEventListener('change',aplicar);})();
"""


@dataclass(frozen=True)
class Grafica:
    titulo: str
    figura: go.Figure
    fuente: str = FUENTE
    leyenda: str = ""


@dataclass(frozen=True)
class Seccion:
    ancla: str
    indice: str  # its label in the side index
    titular: Titular
    graficas: tuple[Grafica, ...]
    nota: str = ""
    par: bool = False  # two charts side by side on wide screens


def _e(texto: object) -> str:
    return html.escape(str(texto), quote=True)


def _secciones(datos: DatosReporte, textos: dict[str, Titular]) -> list[Seccion]:
    macro = tuple(
        Grafica(s.etiqueta, g.fig_macro(datos, s), FUENTE_MACRO) for s in datos.series_macro
    )
    return [
        Seccion("tasa-10a", "10A por fase", textos["tasa-10a"],
                (Grafica("Rendimiento del bono a 10 años, coloreado por fase (%)",
                         g.fig_10a(datos)),)),
        Seccion("imitador", "Fase e imitador", textos["imitador"],
                (Grafica("Fase del modelo, fase del imitador y sus probabilidades",
                         g.fig_imitador(datos)),)),
        Seccion("motores", "Motores de hoy", textos["motores"],
                (Grafica("Contribución SHAP por bloque a la lectura de hoy",
                         g.fig_motores_hoy(datos)),
                 Grafica("Las cinco variables de mayor contribución",
                         g.fig_variables_hoy(datos)))),
        Seccion("motores-tiempo", "Motores en el tiempo", textos["motores-tiempo"],
                (Grafica("Contribución SHAP por bloque, día por día",
                         g.fig_motores_tiempo(datos)),),
                nota=NOTA_MOTORES),
        Seccion("curva", "Curva", textos["curva"],
                (Grafica("Curva de Treasuries: hoy y hace 1 mes, 3 meses y 1 año",
                         g.fig_curva(datos)),
                 Grafica("Firma de cada fase: cambio mediano a 1 mes por plazo",
                         g.fig_firma(datos), leyenda=LEYENDA_PASADO)),
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
    return (
        f'<figure class="grafica"><figcaption>{numero}. {_e(grafica.titulo)}</figcaption>'
        f'{div}{leyenda}<p class="fuente">{_e(grafica.fuente)}</p></figure>'
    )


def _termometro(posicion: Posicion) -> str:
    if posicion.p25 is None or posicion.p75 is None or posicion.mediana is None:
        return (
            '<p class="nota">Sin episodios terminados de esta fase para comparar la duración.</p>'
        )
    tope = max(float(posicion.dias), posicion.p75) * 1.15

    def pct(valor: float) -> str:
        return f"{100.0 * valor / tope:.1f}%"

    return (
        '<div class="termometro"><p class="meta">Duración del episodio contra los '
        f"{posicion.episodios} episodios terminados de la misma fase</p>"
        '<div class="pista">'
        f'<div class="banda" style="left:{pct(posicion.p25)};'
        f'width:{pct(posicion.p75 - posicion.p25)}"></div>'
        f'<div class="mediana" style="left:{pct(posicion.mediana)}"></div>'
        f'<div class="hoy" style="left:{pct(posicion.dias)}"></div></div>'
        f'<div class="escala"><span>0</span><span>P25 {posicion.p25:.0f} · mediana '
        f"{posicion.mediana:.0f} · P75 {posicion.p75:.0f}</span>"
        f"<span>hoy {posicion.dias}</span></div></div>"
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
        avisos += f'<span class="aviso">cambio de fase: {_e(alerta["de"])} a {_e(alerta["a"])}</span>'
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
    nota = f'<p class="nota">{_e(seccion.nota)}</p>' if seccion.nota else ""
    return (
        f'<section id="{seccion.ancla}"><p class="kicker">{_e(seccion.indice)}</p>'
        f"<h2>{_e(seccion.titular.texto)}</h2>{cuerpo}{nota}</section>",
        numero,
    )


def _validacion(datos: DatosReporte) -> str:
    validacion = datos.lectura["validation"]
    veredictos = " · ".join(
        f"{_e(v['stage'])}: {_e(str(v['verdict']).upper())}"
        for v in validacion["registered_verdicts"]
    )
    sombra = validacion["sombra"]
    tarjetas = (
        ("Veredictos registrados", veredictos),
        ("Semanas de sombra", f"{int(sombra['semanas'])}"),
        ("Próxima evaluación de sombra", f"en {int(sombra['proxima_evaluacion_semanas'])} semanas"),
    )
    cajas = "".join(
        f'<div class="tarjeta"><div class="meta">{_e(e)}</div><div class="valor">{v}</div></div>'
        for e, v in tarjetas
    )
    filas = "".join(
        f"<tr><td>{_e(f['name'])}</td><td>{int(f['days'])}</td>"
        f"<td>{'sí' if f['evaluable'] else 'no'}</td><td>{float(f['recall']):.2f}</td>"
        f"<td>{float(f['median_duration_days']):.0f}</td></tr>"
        for f in validacion["by_phase"]
    )
    return (
        '<section id="validacion"><p class="kicker">Validación</p>'
        "<h2>Qué tan confiable es la lectura</h2>"
        f'<div class="tarjetas">{cajas}</div>'
        "<table><thead><tr><th>Fase</th><th>Días</th><th>Evaluable</th><th>Recall</th>"
        f"<th>Duración mediana (días)</th></tr></thead><tbody>{filas}</tbody></table>"
        f'<p class="descargo">{_e(datos.textos["descargo"])}</p>'
        f'<p class="nota">{_e(SEEN_BY_NOTE)}</p>'
        f'<p class="nota">{_e(datos.textos["no_dice"])}</p></section>'
    )


def _comentario(datos: DatosReporte) -> str:
    if datos.comentario is None:
        return ""
    return (
        '<section id="comentario"><p class="kicker">Comentario del analista '
        "(no generado por TERMO)</p>"
        f"{md_to_html(datos.comentario)}</section>"
    )


def render(datos: DatosReporte, generado: str) -> str:
    """The page as one string: plotly.js once, then the cover and every block."""
    textos = titulares(datos)
    acento = g.color_fase(int(datos.lectura["phase"]))
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
            f'<html lang="es" style="--acento:{acento}">',
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
            f"<nav>{nav}</nav>",
            f"<main>{''.join(cuerpos)}{pie}</main>",
            "</div>",
            f"<script>{MODO_OSCURO}</script>",
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
```

- [ ] **Step 4: Correr las pruebas**

Run: `.venv/Scripts/python -m pytest tests/test_visual_pagina.py -v`
Expected: PASS. Si `test_charts_are_numbered_and_sourced` cuenta menos de 12 gráficas: son 1 + 1 + 2 + 1 + 2 + 2 + 2 + 2 (macro) = 13; revisar que `_secciones` incluye todas.

- [ ] **Step 5: Lint y commit**

Run: `.venv/Scripts/python -m ruff check src tests && .venv/Scripts/python -m mypy`
Expected: limpio (partir líneas largas si `E501`).

```bash
git add src/termo/operation/visual/pagina.py tests/test_visual_pagina.py
git commit -m "feat(visual): self-contained report page with Plotly embedded once

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: `run_week` usa el reporte nuevo; `--solo-reporte`

**Files:**
- Modify: `src/termo/run_week.py`
- Modify: `src/termo/operation/html_report.py`
- Modify: `tests/test_html_report.py`
- Modify: `tests/test_run_week.py`

- [ ] **Step 1: Actualizar las pruebas de `run_week`**

En `tests/test_run_week.py`:

1. Agregar, **después** del bloque `from test_op_cli import (...)` (orden alfabético de ruff `I`):

```python
from visual_fixture import PLOTLY_INICIO, sin_plotlyjs
```

y agregar debajo de los imports:

```python
import re

RECURSO_EXTERNO = re.compile(r"""(?:src|href)\s*=\s*["']?https?:""", re.IGNORECASE)
```

(mover `import re` junto a los otros imports estándar para que ruff `I` quede limpio).

2. En `test_the_week_in_one_command_with_the_monthly_report`, reemplazar el bloque que va desde `assert "Hoja semanal" in page and "Fase actual" in page` hasta `assert "http" not in page.lower() and "<script" not in page.lower()` por:

```python
    assert '<section id="portada"' in page and '<section id="validacion"' in page
    assert record["reading"]["phase_name"] in page
    assert page.count(PLOTLY_INICIO) == 1
    sheet_json = json.loads((out / "hoja.json").read_text(encoding="utf-8"))
    assert sheet_json["reading_date"] == day
    descargo = load_operation_config(Path("configs") / "operacion.yaml").texts["descargo"]
    assert not descargo.isascii() and html.escape(descargo) in page
    assert not RECURSO_EXTERNO.search(sin_plotlyjs(page))
```

3. Agregar al final del archivo:

```python
def test_the_report_alone_is_rebuilt_from_the_files(
    here: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--snapshot", str(SNAPSHOT)]) == 0
    capsys.readouterr()
    day = str(_shadow_log().readings()[-1]["reading_date"])
    out = Path("output") / day
    log_before = _shadow_log().path.read_bytes()
    (out / REPORT_FILE).unlink()
    (out / "comentario.md").write_text("## Lectura\n\nNota del analista.", encoding="utf-8")

    assert main(["--solo-reporte", str(out), "--snapshot", str(SNAPSHOT)]) == 0
    lines = _console_ok(capsys.readouterr().out)
    assert lines == [f"[ok] reporte -> {out.as_posix()}/{REPORT_FILE}"]
    page = (out / REPORT_FILE).read_text(encoding="utf-8")
    assert '<section id="comentario"' in page and "Nota del analista." in page
    assert _shadow_log().path.read_bytes() == log_before  # no model run, no new reading


def test_the_report_alone_refuses_download_mes_and_a_foreign_snapshot(
    here: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["--snapshot", str(SNAPSHOT)]) == 0
    capsys.readouterr()
    out = Path("output") / str(_shadow_log().readings()[-1]["reading_date"])
    with pytest.raises(SystemExit):
        main(["--solo-reporte", str(out), "--download"])
    assert "--solo-reporte" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--solo-reporte", str(out), "--snapshot", str(SNAPSHOT), "--mes", "1999-03"])
    assert "--solo-reporte" in capsys.readouterr().err

    foreign = tmp_path / "otra-semana"
    shutil.copytree(out, foreign)
    sheet = json.loads((foreign / "hoja.json").read_text(encoding="utf-8"))
    sheet["snapshot_hash"] = "f" * 64
    (foreign / "hoja.json").write_text(json.dumps(sheet), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--solo-reporte", str(foreign), "--snapshot", str(SNAPSHOT)])
    error = capsys.readouterr().err
    assert "snapshot" in error and error.isascii()
```

y agregar `import shutil` a los imports estándar.

4. En `test_help_says_to_run_from_the_repository_root`, agregar al final:

```python
    assert "--solo-reporte" in out
```

- [ ] **Step 2: Quitar `render_html` de sus pruebas**

En `tests/test_html_report.py`:
- Cambiar el import a `from termo.operation.html_report import md_to_html, write_html`.
- Borrar las funciones `test_render_html_is_a_self_contained_spanish_page_with_the_sections_in_order` y `test_render_html_escapes_the_title_and_section_headings` completas.
- Cambiar el docstring del módulo a `"""The Markdown subset as HTML (the analyst's comment) and the UTF-8 writer."""`.

- [ ] **Step 3: Correr y ver que fallan**

Run: `.venv/Scripts/python -m pytest tests/test_run_week.py tests/test_html_report.py -v`
Expected: FAIL en `test_run_week.py` (el reporte aún es el viejo; `--solo-reporte` no existe). `test_html_report.py` pasa.

- [ ] **Step 4: Retirar `render_html` de `html_report.py`**

En `src/termo/operation/html_report.py`:
- Borrar la constante `STYLE` y la función `render_html` completas.
- Cambiar el docstring del módulo a:

```python
"""A small Markdown subset as HTML, and the UTF-8 writer of the report.

Used for the analyst's comment in the visual report. The subset: `#`/`##` headings,
`**bold**`, pipe tables with a `|---|` separator row, `- ` bullets and blank-line
separated paragraphs; any other line is a paragraph. Text is escaped, so a `<` in a
cell stays a `<`; accents and the `→` are kept.
"""
```

- Borrar `from collections.abc import Sequence` solo si queda sin uso (ruff `F401` lo indica; `_table` sigue usando `Sequence`, así que probablemente se queda).

- [ ] **Step 5: Reescribir `run_week.py`**

Reemplazar el contenido de `src/termo/run_week.py` por:

```python
"""The week in one command: the shadow run, the monthly report if asked, the visual report.

Everything the operation produces stays where `termo.op_cli` puts it (the shadow log, the
sheet, the CSV, the ficha); this module runs those stages, copies their files under
`output/<reading date>/` and builds `reporte.html` from those copies and the snapshot.
`--solo-reporte` rebuilds only the report, without running the model. A week run again
overwrites that directory; the shadow log keeps both runs, as always. Console output is
ASCII only.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from termo.op_cli import (
    OPERATION_CONFIG,
    Operation,
    download_snapshot,
    iso_month,
    open_operation,
    run_ficha,
    run_sombra,
)
from termo.operation.visual.pagina import ARCHIVO, construir

OUTPUT_DIR = Path("output")
REPORT_FILE = ARCHIVO
SHEET_STEM = "hoja"  # output/<date>/hoja.md and hoja.json
FICHA_PREFIX = "ficha-"  # output/<date>/ficha-<AAAA-MM>.md and .json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_termo.py",
        description=(
            "TERMO: the weekly run in one command. Run it from the repository root: it "
            "reads configs/ and the registry under trials/ and reports/, and writes the "
            "deliverable under output/<reading date>/."
        ),
        allow_abbrev=False,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--download",
        action="store_true",
        help="take today's operation snapshot into data/snapshots/<today>/ and run on it",
    )
    source.add_argument("--snapshot", type=Path, help="run on an existing operation snapshot")
    parser.add_argument(
        "--mes", type=iso_month, help="also build the monthly report of this month (AAAA-MM)"
    )
    parser.add_argument(
        "--solo-reporte",
        type=Path,
        metavar="DIR",
        help=(
            "only rebuild DIR/reporte.html from the files in DIR (an output/<date>/ "
            "directory) and --snapshot; the model does not run"
        ),
    )
    parser.add_argument(
        "--operacion",
        type=Path,
        default=OPERATION_CONFIG,
        help=f"operation configuration (default: {OPERATION_CONFIG.as_posix()})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help=f"where <reading date>/ is created (default: {OUTPUT_DIR.as_posix()})",
    )
    return parser


def _report(
    parser: argparse.ArgumentParser, ctx: Operation, out_dir: Path, snapshot_dir: Path
) -> Path:
    try:
        return construir(
            out_dir,
            snapshot_dir,
            ctx.config,
            ctx.op,
            datetime.now(UTC).isoformat(timespec="seconds"),
        )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.solo_reporte is not None and (args.download or args.mes is not None):
        parser.error("--solo-reporte takes --snapshot only: no --download, no --mes")
    try:
        ctx = open_operation(args.operacion)
    except ValueError as error:
        parser.error(str(error))

    if args.solo_reporte is not None:
        report = _report(parser, ctx, args.solo_reporte, args.snapshot)
        print(f"[ok] reporte -> {report.as_posix()}")
        return 0

    snapshot_dir: Path
    if args.download:
        try:
            snapshot_dir = download_snapshot(ctx)
        except FileExistsError as error:
            parser.error(str(error))
    else:
        snapshot_dir = args.snapshot

    run, csv_paths = run_sombra(ctx, snapshot_dir)
    stamp = str(run.record["reading_date"])
    copies: list[tuple[Path, str]] = [
        (run.sheet_path, f"{SHEET_STEM}.md"),
        (run.sheet_path.with_suffix(".json"), f"{SHEET_STEM}.json"),
        *((path, name) for name, path in csv_paths.items()),
    ]
    if args.mes is not None:
        _, ficha_path = run_ficha(ctx, snapshot_dir, args.mes, run.curve, run.analysis)
        copies += [
            (ficha_path, f"{FICHA_PREFIX}{args.mes}.md"),
            (ficha_path.with_suffix(".json"), f"{FICHA_PREFIX}{args.mes}.json"),
        ]

    out_dir: Path = args.output / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    for source_path, name in copies:
        shutil.copyfile(source_path, out_dir / name)
    report = _report(parser, ctx, out_dir, snapshot_dir)
    print(f"[ok] reporte -> {report.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Verificar que `Operation` está exportado en `termo.op_cli` (es la dataclass definida ahí, línea ~67): `from termo.op_cli import Operation` funciona sin cambios.

- [ ] **Step 6: Correr las pruebas**

Run: `.venv/Scripts/python -m pytest tests/test_run_week.py tests/test_html_report.py -v`
Expected: PASS.

Si `cargar` rechaza con "la historia termina el ... y la lectura es del ...": la corrida de sombra exporta más días que la fecha de lectura. En ese caso NO quitar la validación; en `datos.cargar` recortar `historia = historia.loc[:fecha]` antes de validar que `fecha` existe (`if fecha not in historia.index: raise ValueError(...)`), ajustar el test `test_a_history_that_does_not_end_on_the_reading_date_is_refused` para mover la fecha a un día que no existe (p. ej. un sábado), y anotarlo en el commit.

- [ ] **Step 7: Suite completa, lint y commit**

Run: `.venv/Scripts/python -m pytest -q`
Expected: todo PASS (anotar el número de pruebas).

Run: `.venv/Scripts/python -m ruff check src tests && .venv/Scripts/python -m mypy`
Expected: limpio.

```bash
git add src/termo/run_week.py src/termo/operation/html_report.py tests/test_run_week.py tests/test_html_report.py
git commit -m "feat: weekly run builds the visual report; --solo-reporte rebuilds it

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Verificación sobre datos reales

**Files:** ninguno (solo verificación; el HTML no se commitea).

- [ ] **Step 1: Regenerar el reporte de la semana real**

Run: `.venv/Scripts/python run_termo.py --solo-reporte output/2026-10-02 --snapshot data/snapshots/2026-10-06`
Expected: `[ok] reporte -> output/2026-10-02/reporte.html`

- [ ] **Step 2: Medir el tamaño**

Run: `ls -la output/2026-10-02/reporte.html`
Expected: entre 5 y 12 MB. Si pasa de 12 MB, reportarlo (no optimizar sin decidirlo con el usuario).

- [ ] **Step 3: Verificar que git lo ignora**

Run: `git status --short output/`
Expected: no aparece `reporte.html`.

- [ ] **Step 4: Revisión visual en el navegador**

Abrir `file:///C:/Proyectos/TERMO/output/2026-10-02/reporte.html` en el navegador integrado. Revisar en escritorio y en móvil (375 px):
- Portada: "Venta" en rojo, titular, 3 viñetas, 4 KPI (+116 pb y +137 pb), termómetro.
- Las 13 gráficas se dibujan; el selector 1A/3A/5A/10A/todo funciona en la gráfica 1.
- La dispersión marca el episodio actual en el cuadrante bear flattener.
- No hay errores en la consola.
- Índice lateral en escritorio; barra superior en móvil sin scroll horizontal de la página.

Anotar cualquier defecto visual y corregirlo en `pagina.py` o `graficas.py` con su prueba antes de seguir.

---

### Task 10: Documentación y contexto

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-10-05-reporte-visual-design.md`
- Modify: `docs/context/memory.md`, `docs/context/results.md`, `docs/context/todo.md`, `docs/context/sesion-log.md`

- [ ] **Step 1: README**

Agregar al final de `README.md`:

```markdown
## Weekly run

```bash
python run_termo.py --download            # snapshot of today + shadow reading + report
python run_termo.py --snapshot data/snapshots/<fecha> --mes AAAA-MM   # also the monthly ficha
python run_termo.py --solo-reporte output/<fecha> --snapshot data/snapshots/<fecha>   # report only
```

The deliverable lands in `output/<reading date>/`: `reporte.html` (visual report, Plotly
embedded, works offline; not versioned, rebuild it with `--solo-reporte`), `hoja.md/json`,
the five CSV and, with `--mes`, the ficha. An optional `output/<fecha>/comentario.md`
written by the analyst appears in the report as "Comentario del analista".
```

- [ ] **Step 2: Ajustar el spec a lo implementado**

En `docs/superpowers/specs/2026-10-05-reporte-visual-design.md`, §7:
- La interfaz de `datos.py` pasa a `cargar(dir_salida, dir_snapshot, config: CoreConfig, op: OperationConfig) -> DatosReporte`, sin `lecturas` ni `alertas` (el reporte no los usa; la alerta vigente sale de `hoja.json`).
- `construir(dir_salida, dir_snapshot, config, op, generado) -> Path`.
- §3 bloque 2: "dos franjas alineadas (fase del JM y fase del imitador) sobre el área de probabilidades: los desacuerdos se ven como diferencias entre franjas".
- §8: `hoja.json`, `historia_diaria.csv`, `episodios.csv`, `macro.csv` son los obligatorios.

- [ ] **Step 3: Contexto**

`docs/context/memory.md`, agregar una línea:

```markdown
- # decision: (2026-10-05) reporte semanal = reporte visual (spec 5): Plotly embebido sin red, construido desde output/<fecha>/ + snapshot (`--solo-reporte`), fuera de git; bloques SHAP rotulados "movimiento tramo ..." solo en el reporte; titulares por plantillas cerradas sin futuro/condicional.
```

`docs/context/results.md`, agregar:

```markdown
- [2026-10-05]: spec 5 implementado: reporte visual con 13 gráficas Plotly, narrativa por reglas, `--solo-reporte`; <N> pruebas, ruff y mypy limpios; reporte real de 2026-10-02 regenerado (<tamaño> MB) y revisado en navegador.
```

(con `<N>` y `<tamaño>` reales de las Tasks 8 y 9).

`docs/context/todo.md`, agregar:

```markdown
- pending: historia_diaria.csv (~2.3 MB) se repite completo cada semana en git; decidir si se versiona solo el delta o se deja fuera.
- pending: etiqueta macro "prima por plazo 10 anos" en configs/operacion.yaml sin ñ (se ve en el reporte); corregir a "años" si la consola no la imprime.
```

`docs/context/sesion-log.md`, agregar:

```markdown
[2026-10-05]: reporte visual (spec 5) diseñado con referencia HSBC TYCCLES, planificado e implementado; HTML fuera de git.
```

- [ ] **Step 4: Commit**

```bash
git add README.md docs/superpowers/specs/2026-10-05-reporte-visual-design.md docs/context/memory.md docs/context/results.md docs/context/todo.md docs/context/sesion-log.md
git commit -m "docs: visual report usage, spec adjusted to the implementation, context

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
