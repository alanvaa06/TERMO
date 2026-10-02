# Núcleo go/no-go — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir el núcleo de TERMO (datos de FRED, 12 variables diarias, jump model walk-forward y sus pruebas) y producir un veredicto go/no-go sobre si existen fases estables y distintas en la curva del Tesoro.

**Architecture:** Paquete Python `termo` en `src/`, con un módulo por responsabilidad: datos, variables, modelo, validación. Cinco etapas por línea de comandos: `snapshot`, `register`, `run`, `report`, `final-holdout`. Una bitácora de solo-agregar obliga a registrar cada configuración antes de correrla, y el cargador de datos es la única puerta al holdout.

**Tech Stack:** Python 3.14, numpy, pandas, scipy, scikit-learn, `jumpmodels` 0.1.1, pyyaml; pytest, hypothesis, ruff, mypy.

**Spec:** [`docs/superpowers/specs/2026-10-01-nucleo-go-no-go-design.md`](../specs/2026-10-01-nucleo-go-no-go-design.md)

---

## Reglas para quien ejecuta

1. **Sin look-ahead.** Ninguna variable, escala o parámetro usa datos posteriores a la fecha de lectura. Una fuga es un bug.
2. **Registrar antes de mirar.** No se corre nada sobre datos reales antes de la Tarea 24. Todas las pruebas usan una curva sintética.
3. **El holdout no se toca.** La única puerta es la etapa `final-holdout` (Tarea 25), que requiere aprobación explícita del usuario y corre una sola vez.
4. **No ajustar una prueba para que pase.** Si un resultado "Esperado" no coincide, detenerse y depurar la causa.
5. **Consola solo ASCII:** `->`, `[ok]`, `[x]`. Las consolas de Windows fallan con otros caracteres.
6. **Comandos.** Todos usan `.venv/Scripts/python` y funcionan igual en Git Bash y PowerShell.
7. **Commits.** Un commit por tarea. Terminar cada mensaje con la línea `Co-Authored-By` que indique la sesión.

## Lo que ya se verificó al escribir este plan

Todo el código de este plan se prototipó fuera del repositorio, con datos sintéticos:

- 140 pruebas pasan; `ruff` y `mypy` sin observaciones.
- El orden TDD se reprodujo tarea por tarea sobre una copia limpia: cada prueba falla antes de su implementación y pasa después. Los mensajes "Esperado" de cada paso salen de esa corrida.
- `jumpmodels` 0.1.1 funciona con Python 3.14, numpy 2.5, pandas 3.0 y scikit-learn 1.9. Su lectura en línea es causal.
- Un ajuste con K=5 y 12,000 días tarda cerca de 1 segundo con multa 50, pero 84 segundos con multa cero.
- La descarga sin clave de FRED funciona con un encabezado `User-Agent`.

## Tres correcciones al spec encontradas al prototipar

El spec ya quedó actualizado con ellas.

| # | Qué decía el diseño | Qué se midió | Qué hace el plan |
|---|---|---|---|
| 1 | La separación se mide con η² crudo | Cinco fases de puro ruido le "ganaban" a dos fases de puro ruido el 25% de las veces (lo nominal es 2.5%) | La separación es η² **en exceso**: η² menos el promedio que logra la misma serie de fases desplazada en el tiempo. Con eso baja a 1.7% |
| 2 | La prueba de independencia desplaza las fases un mínimo de 26 semanas | Con esa zona de exclusión rechaza el 4% de las veces cuando debería rechazar el 1% | Se usan todos los desplazamientos; así rechaza el 0.7% |
| 3 | La línea base K-means es el jump model con multa cero | 84 segundos por ajuste con `jumpmodels` | `sklearn.KMeans`, que minimiza el mismo objetivo |

PBO sigue ordenando las configuraciones por η² crudo: la corrección de azar no se puede recalcular en cada una de las 12,870 particiones a un costo razonable. Queda anotado como limitación en el reporte.

## Mapa de archivos

| Archivo | Responsabilidad |
|---|---|
| `configs/core.yaml` | Todos los valores registrados: series, fechas, grilla, umbrales |
| `src/termo/config.py` | Lee y valida la configuración |
| `src/termo/data/fred.py` | Descarga y valida una serie |
| `src/termo/data/snapshot.py` | Escribe y verifica snapshots con hash |
| `src/termo/data/loader.py` | Entrega la curva; guardia de holdout |
| `src/termo/features/pca.py` | PCA de la curva con signo fijo |
| `src/termo/features/velocity.py` | Cambio suavizado con ventanas vecinas |
| `src/termo/features/volatility.py` | Volatilidad exponencial |
| `src/termo/features/pipeline.py` | Las 12 variables, recorte y z-score |
| `src/termo/regime/model.py` | Interfaz de modelo; jump model y K-means |
| `src/termo/regime/align.py` | Nombres de fase estables |
| `src/termo/validation/metrics.py` | ARI, η², separación en exceso, duraciones |
| `src/termo/validation/bootstrap.py` | Bloques, diferencia pareada, independencia |
| `src/termo/validation/walkforward.py` | Reentrenar y leer en línea |
| `src/termo/validation/stability.py` | S2 y puntaje de estabilidad |
| `src/termo/validation/ftic.py` | Criterio FTIC |
| `src/termo/validation/pbo.py` | PBO y N efectivo |
| `src/termo/validation/trials.py` | Bitácora |
| `src/termo/dataset.py` | Datos compartidos por todas las configuraciones |
| `src/termo/experiment.py` | Evaluación de una configuración |
| `src/termo/selection.py` | Elección de K y λ |
| `src/termo/report.py` | Criterios, veredicto y archivos de reporte |
| `src/termo/core.py` | Las cinco etapas |
| `src/termo/cli.py` | Línea de comandos |
| `tests/conftest.py` | Curva sintética con fases plantadas |

---

### Tarea 1: Entorno y prueba de humo de `jumpmodels`

Deja el proyecto instalable y convierte el spike de compatibilidad en una prueba permanente.
El spike ya se corrió al escribir este plan: `jumpmodels` 0.1.1 funciona con Python 3.14,
numpy 2.5, pandas 3.0 y scikit-learn 1.9.

**Archivos:**
- Crear: `pyproject.toml`, `.gitattributes`
- Crear (vacíos): `src/termo/data/__init__.py`, `src/termo/features/__init__.py`, `src/termo/regime/__init__.py`, `src/termo/validation/__init__.py`
- Crear: `src/termo/__init__.py`
- Modificar: `.gitignore`
- Prueba: `tests/test_smoke_jumpmodels.py`

- [ ] **Paso 1: Crear la rama de trabajo**

````bash
git checkout spec/nucleo-go-no-go
git checkout -b feat/nucleo-go-no-go
````

- [ ] **Paso 2: Escribir `pyproject.toml`**

`pyproject.toml`

````toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "termo"
version = "0.1.0"
description = "TERMO: Treasury Environment & Regime MOnitor"
requires-python = ">=3.12"
dependencies = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "jumpmodels==0.1.1",
    "pyyaml",
]

[project.optional-dependencies]
dev = ["pytest", "hypothesis", "ruff", "mypy", "types-PyYAML"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.mypy]
files = ["src"]
disallow_untyped_defs = true
no_implicit_optional = true

[[tool.mypy.overrides]]
module = ["jumpmodels.*", "sklearn.*", "scipy.*", "pandas.*"]
ignore_missing_imports = true
````

- [ ] **Paso 3: Crear el paquete**

`src/termo/__init__.py`

````python
"""TERMO: Treasury Environment & Regime MOnitor."""
````

Crear vacíos: `src/termo/data/__init__.py`, `src/termo/features/__init__.py`,
`src/termo/regime/__init__.py`, `src/termo/validation/__init__.py`.

- [ ] **Paso 4: Proteger los artefactos reproducibles**

El `.gitignore` actual ignora todos los `*.csv`. Los snapshots y las etiquetas de cada trial
deben versionarse. Agregar al final de `.gitignore`:

````
# Tooling caches
.pytest_cache/
.mypy_cache/
.ruff_cache/
.hypothesis/
*.egg-info/

# Reproducibility artifacts are versioned on purpose
!data/snapshots/**/*.csv
!trials/labels/*.csv
````

Crear `.gitattributes` para que git no cambie los saltos de línea (cambiaria los hashes):

````
data/snapshots/** -text
trials/** -text
````

- [ ] **Paso 5: Crear el entorno e instalar**

````bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
````

Esperado: termina sin errores e instala `jumpmodels 0.1.1`.

- [ ] **Paso 6: Escribir la prueba de humo**

`tests/test_smoke_jumpmodels.py`

````python
"""The properties of `jumpmodels` that the rest of the project relies on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from jumpmodels.jump import JumpModel
from sklearn.metrics import adjusted_rand_score


@pytest.fixture(scope="module")
def planted() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    truth = np.repeat([0, 1, 0, 2, 1, 0], 300)
    centers = np.array([[0.0] * 6, [2.0] * 6, [-2.0] * 6])
    values = centers[truth] + rng.normal(size=(len(truth), 6))
    index = pd.bdate_range("2000-01-03", periods=len(truth))
    return pd.DataFrame(values, index=index, columns=[f"f{i}" for i in range(6)]), truth


def test_recovers_planted_regimes(planted: tuple[pd.DataFrame, np.ndarray]) -> None:
    features, truth = planted
    model = JumpModel(n_components=3, jump_penalty=50.0, random_state=0).fit(features)
    assert adjusted_rand_score(truth, np.asarray(model.labels_)) > 0.95


def test_fit_is_deterministic(planted: tuple[pd.DataFrame, np.ndarray]) -> None:
    features, _ = planted
    first = JumpModel(n_components=3, jump_penalty=50.0, random_state=0).fit(features)
    second = JumpModel(n_components=3, jump_penalty=50.0, random_state=0).fit(features)
    assert (np.asarray(first.labels_) == np.asarray(second.labels_)).all()


def test_online_prediction_ignores_the_future(planted: tuple[pd.DataFrame, np.ndarray]) -> None:
    features, _ = planted
    model = JumpModel(n_components=3, jump_penalty=50.0, random_state=0).fit(features)
    altered = features.copy()
    altered.iloc[1000:] = 0.0
    before = np.asarray(model.predict_online(features))[:1000]
    after = np.asarray(model.predict_online(altered))[:1000]
    assert (before == after).all()


def test_rejects_missing_values(planted: tuple[pd.DataFrame, np.ndarray]) -> None:
    features, _ = planted
    broken = features.copy()
    broken.iloc[5, 0] = np.nan
    with pytest.raises(AssertionError):
        JumpModel(n_components=3, jump_penalty=50.0, random_state=0).fit(broken)
````

- [ ] **Paso 7: Correr la prueba**

Correr: `.venv/Scripts/python -m pytest tests/test_smoke_jumpmodels.py -q`

Esperado: `4 passed`. Esta prueba no empieza en rojo: verifica una dependencia, no código nuestro.

Si falla, DETENERSE y avisar al usuario: el resto del plan asume esta libreria.

- [ ] **Paso 8: Commit**

````bash
git add pyproject.toml .gitignore .gitattributes src/termo tests/test_smoke_jumpmodels.py
git commit -m "chore: set up termo package and jumpmodels smoke test"
````

### Tarea 2: Configuración registrada

Todos los números del spec (series, fechas, grilla, umbrales) viven en un solo archivo. El código los lee tipados y rechaza combinaciones inválidas. `conftest.py` trae la curva sintética con fases plantadas que usan casi todas las pruebas; ningún test usa datos reales.

**Archivos:**
- Crear: `configs/core.yaml`
- Crear: `src/termo/config.py`
- Prueba: `tests/conftest.py`
- Prueba: `tests/test_config.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/conftest.py`

````python
"""Synthetic yield curves with planted regimes. No real market data in the tests."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from termo.config import BootstrapConfig, CoreConfig, FticConfig, Thresholds

if TYPE_CHECKING:
    from termo.dataset import ExperimentData

TENORS = ("DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS30")
MATURITIES = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0])
MIN_REGIME_DAYS, MAX_REGIME_DAYS = 60, 200


def planted_regimes(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Alternating 0/1 runs of random length, so the series never repeats with a fixed period."""
    regimes = np.empty(n_days, dtype=int)
    position, state = 0, 0
    while position < n_days:
        length = int(rng.integers(MIN_REGIME_DAYS, MAX_REGIME_DAYS + 1))
        regimes[position : position + length] = state
        position, state = position + length, 1 - state
    return regimes


def make_curve(n_days: int = 2400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Two alternating regimes: calm rally (0) and volatile sell-off (1)."""
    rng = np.random.default_rng(seed)
    regimes = planted_regimes(n_days, rng)
    drift = np.where(regimes == 1, 0.015, -0.015)
    vol = np.where(regimes == 1, 0.06, 0.02)
    level = 10.0 + np.cumsum(drift + vol * rng.normal(size=n_days))
    slope = np.cumsum(0.01 * rng.normal(size=n_days))
    curvature = np.cumsum(0.004 * rng.normal(size=n_days))
    x = np.log(MATURITIES)
    x = (x - x.mean()) / x.std()
    belly = -(x**2 - (x**2).mean())
    noise = 0.003 * rng.normal(size=(n_days, len(TENORS)))
    values = level[:, None] + slope[:, None] * x[None, :] + curvature[:, None] * belly[None, :]
    index = pd.bdate_range("1990-01-01", periods=n_days, name="date")
    curve = pd.DataFrame(values + noise, index=index, columns=list(TENORS))
    assert (curve > 0).all().all(), "synthetic yields must stay positive"
    return curve, regimes


def make_config() -> CoreConfig:
    return CoreConfig(
        series=TENORS,
        start=date(1990, 1, 1),
        holdout_start=date(1998, 4, 1),
        first_train_end=date(1994, 6, 30),
        refit_weeks=26,
        burn_in_days=252,
        k_values=(2, 3),
        jump_penalties=(10.0, 50.0),
        horizon_short_days=20,
        horizon_long_days=65,
        pbo_blocks=4,
        effective_n_cut=0.2,
        thresholds=Thresholds(
            stability_min=0.6,
            independence_p_max=0.01,
            min_median_duration_days=20,
            pbo_max=0.05,
            collinearity_max=0.8,
        ),
        bootstrap=BootstrapConfig(block_weeks=8, n_resamples=200, seed=0),
        ftic=FticConfig(k0=3, mean_phase_days=40, saturated_k=6, max_jump_fraction=0.4),
    )


@pytest.fixture(scope="session")
def curve_and_regimes() -> tuple[pd.DataFrame, np.ndarray]:
    return make_curve()


@pytest.fixture(scope="session")
def curve(curve_and_regimes: tuple[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    return curve_and_regimes[0]


@pytest.fixture(scope="session")
def config() -> CoreConfig:
    return make_config()


@pytest.fixture(scope="session")
def pre_holdout(curve: pd.DataFrame, config: CoreConfig) -> pd.DataFrame:
    return curve.loc[curve.index < pd.Timestamp(config.holdout_start)]


@pytest.fixture(scope="session")
def data(pre_holdout: pd.DataFrame, config: CoreConfig) -> ExperimentData:
    # Imported here so that the fixtures above work before these modules exist.
    from termo.dataset import prepare
    from termo.features.pipeline import FEATURE_NAMES

    return prepare(pre_holdout, config, FEATURE_NAMES)
````

`tests/test_config.py`

````python
from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from termo.config import CoreConfig, load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "core.yaml"


def test_loads_the_registered_configuration() -> None:
    config = load_config(REPO_CONFIG)
    assert config.series == ("DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS30")
    assert config.start == date(1977, 2, 15)
    assert config.holdout_start == date(2024, 10, 1)
    assert len(config.k_values) * len(config.jump_penalties) == 24
    assert config.thresholds.min_median_duration_days == 20
    assert config.ftic.saturated_k == 6


def test_rejects_training_window_inside_the_holdout(config: CoreConfig) -> None:
    with pytest.raises(ValueError, match="start < first_train_end < holdout_start"):
        replace(config, first_train_end=config.holdout_start)


def test_rejects_odd_number_of_pbo_blocks(config: CoreConfig) -> None:
    with pytest.raises(ValueError, match="pbo_blocks"):
        replace(config, pbo_blocks=5)


def test_rejects_single_state_models(config: CoreConfig) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        replace(config, k_values=(1, 2))
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_config.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.config'`

- [ ] **Paso 3: Escribir la implementación**

`configs/core.yaml`

````yaml
# TERMO spec 1 (nucleo go/no-go). Every value here is pre-registered:
# changing one after a run is a new trial, not an edit.
series: [DGS1, DGS2, DGS3, DGS5, DGS7, DGS10, DGS30]
start: 1977-02-15
holdout_start: 2024-10-01
first_train_end: 1987-12-31
refit_weeks: 26
burn_in_days: 252

grid:
  k: [2, 3, 4, 5]
  jump_penalty: [5, 12, 30, 80, 200, 500]

horizons_days:
  short: 20
  long: 65

thresholds:
  stability_min: 0.6
  independence_p_max: 0.01
  min_median_duration_days: 20
  pbo_max: 0.05
  collinearity_max: 0.8

bootstrap:
  block_weeks: 26
  n_resamples: 2000
  seed: 0

ftic:
  k0: 3
  mean_phase_days: 40
  saturated_k: 6
  max_jump_fraction: 0.4

pbo_blocks: 16
effective_n_cut: 0.2
````

`src/termo/config.py`

````python
"""Typed, validated view of configs/core.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Thresholds:
    stability_min: float
    independence_p_max: float
    min_median_duration_days: int
    pbo_max: float
    collinearity_max: float


@dataclass(frozen=True)
class BootstrapConfig:
    block_weeks: int
    n_resamples: int
    seed: int


@dataclass(frozen=True)
class FticConfig:
    k0: int
    mean_phase_days: int
    saturated_k: int
    max_jump_fraction: float


@dataclass(frozen=True)
class CoreConfig:
    series: tuple[str, ...]
    start: date
    holdout_start: date
    first_train_end: date
    refit_weeks: int
    burn_in_days: int
    k_values: tuple[int, ...]
    jump_penalties: tuple[float, ...]
    horizon_short_days: int
    horizon_long_days: int
    pbo_blocks: int
    effective_n_cut: float
    thresholds: Thresholds
    bootstrap: BootstrapConfig
    ftic: FticConfig

    def __post_init__(self) -> None:
        if not self.start < self.first_train_end < self.holdout_start:
            raise ValueError("dates must satisfy start < first_train_end < holdout_start")
        if not self.k_values or min(self.k_values) < 2:
            raise ValueError("every K in the grid must be at least 2")
        if not self.jump_penalties or min(self.jump_penalties) <= 0:
            raise ValueError("every jump penalty in the grid must be positive")
        if self.pbo_blocks < 2 or self.pbo_blocks % 2 != 0:
            raise ValueError("pbo_blocks must be an even number >= 2")
        if self.refit_weeks < 1 or self.burn_in_days < 0:
            raise ValueError("refit_weeks must be >= 1 and burn_in_days >= 0")


def load_config(path: Path) -> CoreConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return CoreConfig(
        series=tuple(raw["series"]),
        start=raw["start"],
        holdout_start=raw["holdout_start"],
        first_train_end=raw["first_train_end"],
        refit_weeks=int(raw["refit_weeks"]),
        burn_in_days=int(raw["burn_in_days"]),
        k_values=tuple(int(k) for k in raw["grid"]["k"]),
        jump_penalties=tuple(float(x) for x in raw["grid"]["jump_penalty"]),
        horizon_short_days=int(raw["horizons_days"]["short"]),
        horizon_long_days=int(raw["horizons_days"]["long"]),
        pbo_blocks=int(raw["pbo_blocks"]),
        effective_n_cut=float(raw["effective_n_cut"]),
        thresholds=Thresholds(**raw["thresholds"]),
        bootstrap=BootstrapConfig(**raw["bootstrap"]),
        ftic=FticConfig(**raw["ftic"]),
    )
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_config.py -q`

Esperado: `4 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/conftest.py tests/test_config.py configs/core.yaml src/termo/config.py
git commit -m "feat: add typed core configuration"
````

### Tarea 3: Descarga y validación de una serie de FRED

Una serie por petición (con varias, FRED ignora el rango de fechas). El archivo se valida antes de confiar en el: columnas, fechas crecientes, valores entre 0 y 25.

**Archivos:**
- Crear: `src/termo/data/fred.py`
- Prueba: `tests/test_fred.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_fred.py`

````python
from __future__ import annotations

import math

import pytest

from termo.data.fred import FRED_URL, DataValidationError, fetch_series_csv, parse_series_csv

GOOD = "observation_date,DGS1\n1977-02-15,5.39\n1977-02-16,5.40\n1977-02-21,\n1977-02-22,5.46\n"


def test_fetch_asks_for_one_series() -> None:
    seen: list[str] = []

    def fake_get(url: str) -> str:
        seen.append(url)
        return GOOD

    assert fetch_series_csv("DGS1", fake_get) == GOOD
    assert seen == [FRED_URL.format(series_id="DGS1")]


def test_parse_keeps_holidays_as_missing() -> None:
    series = parse_series_csv(GOOD, "DGS1")
    assert series.name == "DGS1"
    assert list(series.index.strftime("%Y-%m-%d")) == [
        "1977-02-15",
        "1977-02-16",
        "1977-02-21",
        "1977-02-22",
    ]
    assert series.iloc[0] == pytest.approx(5.39)
    assert math.isnan(series.iloc[2])


def test_parse_accepts_dot_as_missing() -> None:
    series = parse_series_csv("observation_date,DGS1\n1977-02-15,5.39\n1977-02-16,.\n", "DGS1")
    assert math.isnan(series.iloc[1])


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("date,DGS1\n1977-02-15,5.39\n", "expected columns"),
        ("observation_date,DGS2\n1977-02-15,5.39\n", "expected columns"),
        ("observation_date,DGS1\n", "no rows"),
        ("observation_date,DGS1\n1977-02-16,5.4\n1977-02-15,5.3\n", "strictly increasing"),
        ("observation_date,DGS1\n1977-02-15,5.4\n1977-02-15,5.3\n", "strictly increasing"),
        ("observation_date,DGS1\nnot-a-date,5.4\n", "unparseable dates"),
        ("observation_date,DGS1\n1977-02-15,abc\n", "non-numeric"),
        ("observation_date,DGS1\n1977-02-15,\n", "no observed values"),
        ("observation_date,DGS1\n1977-02-15,539\n", "outside"),
        ("observation_date,DGS1\n1977-02-15,-0.5\n", "outside"),
    ],
)
def test_parse_rejects_malformed_files(text: str, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        parse_series_csv(text, "DGS1")
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_fred.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.data.fred'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/data/fred.py`

````python
"""Download and validate one FRED series at a time."""

from __future__ import annotations

import io
import urllib.request
from collections.abc import Callable

import pandas as pd

# One series per request: with several ids FRED ignores the date range for all but the first.
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
MIN_YIELD = 0.0
MAX_YIELD = 25.0


class DataValidationError(ValueError):
    """A downloaded series does not look like a Treasury yield series."""


def http_get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "termo/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body: bytes = response.read()
    return body.decode("utf-8")


def fetch_series_csv(series_id: str, get: Callable[[str], str] = http_get) -> str:
    return get(FRED_URL.format(series_id=series_id))


def parse_series_csv(text: str, series_id: str) -> pd.Series:
    """Return the series in percent, indexed by date, with NaN where FRED has no value."""
    frame = pd.read_csv(io.StringIO(text), na_values=["."])
    expected = ["observation_date", series_id]
    if list(frame.columns) != expected:
        raise DataValidationError(
            f"{series_id}: expected columns {expected}, got {list(frame.columns)}"
        )
    if frame.empty:
        raise DataValidationError(f"{series_id}: no rows")

    dates = pd.to_datetime(frame["observation_date"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise DataValidationError(f"{series_id}: unparseable dates")
    if not (dates.is_monotonic_increasing and dates.is_unique):
        raise DataValidationError(f"{series_id}: dates must be strictly increasing")

    values = pd.to_numeric(frame[series_id], errors="coerce")
    if (frame[series_id].notna() & values.isna()).any():
        raise DataValidationError(f"{series_id}: non-numeric values")
    observed = values.dropna()
    if observed.empty:
        raise DataValidationError(f"{series_id}: no observed values")
    if ((observed < MIN_YIELD) | (observed > MAX_YIELD)).any():
        raise DataValidationError(f"{series_id}: values outside [{MIN_YIELD}, {MAX_YIELD}]")

    index = pd.DatetimeIndex(dates, name="date")
    return pd.Series(values.to_numpy(dtype=float), index=index, name=series_id)
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_fred.py -q`

Esperado: `13 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_fred.py src/termo/data/fred.py
git commit -m "feat: fetch and validate FRED series"
````

### Tarea 4: Snapshot con hash

Cada corrida cita un snapshot inmutable. Se escribe en bytes para que el hash no cambie con los saltos de línea de Windows.

**Archivos:**
- Crear: `src/termo/data/snapshot.py`
- Prueba: `tests/test_snapshot.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_snapshot.py`

````python
from __future__ import annotations

from pathlib import Path

import pytest

from termo.data.snapshot import SnapshotError, read_snapshot, snapshot_hash, write_snapshot

FILES = {
    "DGS1": "observation_date,DGS1\n1977-02-15,5.39\n",
    "DGS2": "observation_date,DGS2\n1977-02-15,5.95\n",
}


def test_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "2026-10-02"
    write_snapshot(target, FILES, "2026-10-02T00:00:00+00:00")
    assert read_snapshot(target) == FILES


def test_refuses_to_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "2026-10-02"
    write_snapshot(target, FILES, "2026-10-02T00:00:00+00:00")
    with pytest.raises(FileExistsError):
        write_snapshot(target, FILES, "2026-10-03T00:00:00+00:00")


def test_detects_a_modified_file(tmp_path: Path) -> None:
    target = tmp_path / "snap"
    write_snapshot(target, FILES, "2026-10-02T00:00:00+00:00")
    (target / "DGS1.csv").write_bytes(b"observation_date,DGS1\n1977-02-15,9.99\n")
    with pytest.raises(SnapshotError, match="hash mismatch for DGS1"):
        read_snapshot(target)


def test_detects_a_missing_file(tmp_path: Path) -> None:
    target = tmp_path / "snap"
    write_snapshot(target, FILES, "2026-10-02T00:00:00+00:00")
    (target / "DGS2.csv").unlink()
    with pytest.raises(SnapshotError, match="missing file for DGS2"):
        read_snapshot(target)


def test_hash_depends_on_content_not_on_download_time(tmp_path: Path) -> None:
    write_snapshot(tmp_path / "a", FILES, "2026-10-02T00:00:00+00:00")
    write_snapshot(tmp_path / "b", FILES, "2026-11-30T00:00:00+00:00")
    write_snapshot(tmp_path / "c", {**FILES, "DGS1": FILES["DGS1"] + "1977-02-16,5.40\n"}, "x")
    assert snapshot_hash(tmp_path / "a") == snapshot_hash(tmp_path / "b")
    assert snapshot_hash(tmp_path / "a") != snapshot_hash(tmp_path / "c")
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_snapshot.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.data.snapshot'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/data/snapshot.py`

````python
"""Immutable, hash-checked copies of the downloaded CSV files."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

MANIFEST = "manifest.json"


class SnapshotError(RuntimeError):
    """The snapshot on disk is incomplete or does not match its manifest."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_manifest(snapshot_dir: Path) -> dict[str, str]:
    path = snapshot_dir / MANIFEST
    if not path.exists():
        raise SnapshotError(f"no manifest in {snapshot_dir}")
    files: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))["files"]
    return files


def write_snapshot(
    snapshot_dir: Path, csv_by_series: Mapping[str, str], downloaded_at: str
) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    files: dict[str, str] = {}
    for series_id, text in sorted(csv_by_series.items()):
        data = text.encode("utf-8")
        # Bytes, not text: Windows newline translation would change the hash.
        (snapshot_dir / f"{series_id}.csv").write_bytes(data)
        files[series_id] = _sha256(data)
    manifest = {"downloaded_at": downloaded_at, "files": files}
    (snapshot_dir / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )


def read_snapshot(snapshot_dir: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    for series_id, expected in _read_manifest(snapshot_dir).items():
        path = snapshot_dir / f"{series_id}.csv"
        if not path.exists():
            raise SnapshotError(f"missing file for {series_id}")
        data = path.read_bytes()
        if _sha256(data) != expected:
            raise SnapshotError(f"hash mismatch for {series_id}")
        texts[series_id] = data.decode("utf-8")
    return texts


def snapshot_hash(snapshot_dir: Path) -> str:
    files = _read_manifest(snapshot_dir)
    joined = "".join(f"{series_id}:{digest};" for series_id, digest in sorted(files.items()))
    return _sha256(joined.encode("utf-8"))
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_snapshot.py -q`

Esperado: `5 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_snapshot.py src/termo/data/snapshot.py
git commit -m "feat: add hash-checked data snapshots"
````

### Tarea 5: Cargador con guardia de holdout

Es la única puerta a los datos. Sin la bandera `final_evaluation` nunca devuelve fechas del holdout, y si se le piden explícitamente, falla.

**Archivos:**
- Crear: `src/termo/data/loader.py`
- Prueba: `tests/test_loader.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_loader.py`

````python
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from termo.data.loader import HoldoutAccessError, load_curve
from termo.data.snapshot import SnapshotError, write_snapshot

START = date(2024, 9, 25)
HOLDOUT = date(2024, 10, 1)
SERIES = ("DGS1", "DGS2")
FILES = {
    "DGS1": (
        "observation_date,DGS1\n2024-09-24,4.00\n2024-09-25,4.01\n2024-09-26,4.02\n"
        "2024-09-27,\n2024-09-30,4.04\n2024-10-01,4.05\n2024-10-02,4.06\n"
    ),
    "DGS2": (
        "observation_date,DGS2\n2024-09-24,3.50\n2024-09-25,3.51\n2024-09-26,3.52\n"
        "2024-09-27,3.53\n2024-09-30,3.54\n2024-10-01,3.55\n2024-10-02,3.56\n"
    ),
}


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    target = tmp_path / "snap"
    write_snapshot(target, FILES, "2026-10-02T00:00:00+00:00")
    return target


def test_default_load_stops_before_the_holdout(snapshot: Path) -> None:
    frame = load_curve(snapshot, SERIES, START, HOLDOUT)
    assert list(frame.columns) == list(SERIES)
    # 09-24 is before the start; 09-27 lacks DGS1; 10-01 onwards is holdout.
    assert list(frame.index.strftime("%Y-%m-%d")) == ["2024-09-25", "2024-09-26", "2024-09-30"]


def test_requesting_holdout_dates_is_refused(snapshot: Path) -> None:
    with pytest.raises(HoldoutAccessError):
        load_curve(snapshot, SERIES, START, HOLDOUT, end=date(2024, 10, 1))


def test_final_evaluation_opens_the_holdout(snapshot: Path) -> None:
    frame = load_curve(snapshot, SERIES, START, HOLDOUT, final_evaluation=True)
    assert frame.index[-1].strftime("%Y-%m-%d") == "2024-10-02"


def test_missing_series_is_reported(snapshot: Path) -> None:
    with pytest.raises(SnapshotError, match="DGS10"):
        load_curve(snapshot, ("DGS1", "DGS10"), START, HOLDOUT)
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_loader.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.data.loader'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/data/loader.py`

````python
"""The only door to the yield curve, and the only door to the holdout."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from termo.data.fred import parse_series_csv
from termo.data.snapshot import SnapshotError, read_snapshot


class HoldoutAccessError(RuntimeError):
    """Holdout dates were requested without the final-evaluation flag."""


def load_curve(
    snapshot_dir: Path,
    series: Sequence[str],
    start: date,
    holdout_start: date,
    *,
    end: date | None = None,
    final_evaluation: bool = False,
) -> pd.DataFrame:
    """Yields in percent, one column per series, only days where every series has a value.

    Without `final_evaluation` the frame stops the day before `holdout_start`.
    """
    if not final_evaluation and end is not None and end >= holdout_start:
        raise HoldoutAccessError(
            f"end={end} reaches the holdout ({holdout_start}); pass final_evaluation=True"
        )
    texts = read_snapshot(snapshot_dir)
    missing = [name for name in series if name not in texts]
    if missing:
        raise SnapshotError(f"snapshot lacks series: {missing}")

    columns = [parse_series_csv(texts[name], name) for name in series]
    frame = pd.concat(columns, axis=1, join="outer").sort_index().dropna(how="any")
    frame = frame.loc[frame.index >= pd.Timestamp(start)]

    limit: pd.Timestamp | None
    if end is not None:
        limit = pd.Timestamp(end)
    elif final_evaluation:
        limit = None
    else:
        limit = pd.Timestamp(holdout_start) - pd.Timedelta(days=1)
    if limit is not None:
        frame = frame.loc[frame.index <= limit]
    return frame
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_loader.py -q`

Esperado: `4 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_loader.py src/termo/data/loader.py
git commit -m "feat: add curve loader with holdout guard"
````

### Tarea 6: PCA de la curva con signo fijo

Nivel, pendiente y curvatura salen de un PCA sobre cambios diarios. El signo se fija después de cada ajuste; si no, un reentrenamiento puede voltearlo.

**Archivos:**
- Crear: `src/termo/features/pca.py`
- Prueba: `tests/test_pca.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_pca.py`

````python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from termo.features.pca import CurvePCA


def test_three_factors_explain_the_synthetic_curve(curve: pd.DataFrame) -> None:
    pca = CurvePCA.fit(curve)
    assert pca.loadings.shape == (3, 7)
    assert pca.explained[0] > pca.explained[1] > pca.explained[2]
    assert pca.explained.sum() > 0.95


def test_sign_convention(curve: pd.DataFrame) -> None:
    pca = CurvePCA.fit(curve)
    tenors = list(curve.columns)
    short, belly = tenors.index("DGS1"), tenors.index("DGS5")
    ten, long_ = tenors.index("DGS10"), tenors.index("DGS30")
    level, slope, curvature = pca.loadings
    assert level[ten] > 0
    assert slope[long_] - slope[short] > 0
    assert curvature[belly] - (curvature[short] + curvature[long_]) / 2 > 0


def test_sign_convention_survives_a_mirrored_sample(curve: pd.DataFrame) -> None:
    # Mirroring the data flips nothing in the covariance; the loadings must come out the same.
    mirrored = 2 * curve.iloc[0] - curve
    original, flipped = CurvePCA.fit(curve), CurvePCA.fit(mirrored)
    assert np.allclose(original.loadings, flipped.loadings, atol=1e-8)


def test_parallel_shift_moves_the_level_only_upwards(curve: pd.DataFrame) -> None:
    pca = CurvePCA.fit(curve)
    shifted = curve.iloc[-5:] + 0.10  # +10 bp on every tenor
    delta = pca.scores(shifted) - pca.scores(curve.iloc[-5:])
    assert (delta["L"] > 0).all()


def test_scores_are_centered_on_the_training_mean(curve: pd.DataFrame) -> None:
    pca = CurvePCA.fit(curve)
    assert np.allclose(pca.scores(curve).mean().to_numpy(), 0.0, atol=1e-8)


def test_rejects_other_tenors(curve: pd.DataFrame) -> None:
    pca = CurvePCA.fit(curve)
    with pytest.raises(ValueError, match="do not match"):
        pca.scores(curve[["DGS1", "DGS2"]])
    with pytest.raises(ValueError, match="sign convention"):
        CurvePCA.fit(curve.drop(columns=["DGS30"]))
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_pca.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.features.pca'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/features/pca.py`

````python
"""Level, slope and curvature of the curve from a PCA of daily yield changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BP_PER_PERCENT = 100.0
FACTORS = ("L", "S", "C")
SHORT, BELLY, TEN_YEAR, LONG = "DGS1", "DGS5", "DGS10", "DGS30"


@dataclass(frozen=True, eq=False)
class CurvePCA:
    tenors: tuple[str, ...]
    loadings: np.ndarray  # (3, n_tenors); rows are L, S, C
    mean_level: np.ndarray  # (n_tenors,), percent
    explained: np.ndarray  # (3,), share of variance of daily changes

    @classmethod
    def fit(cls, curve_train: pd.DataFrame) -> CurvePCA:
        tenors = tuple(str(name) for name in curve_train.columns)
        missing = [name for name in (SHORT, BELLY, TEN_YEAR, LONG) if name not in tenors]
        if missing:
            raise ValueError(f"sign convention needs tenors {missing}")
        changes = curve_train.diff().dropna().to_numpy() * BP_PER_PERCENT
        if len(changes) <= len(tenors):
            raise ValueError("not enough observations to fit the PCA")

        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(changes, rowvar=False))
        top = np.argsort(eigenvalues)[::-1][: len(FACTORS)]
        loadings = eigenvectors[:, top].T.copy()

        short, belly = tenors.index(SHORT), tenors.index(BELLY)
        ten, long_ = tenors.index(TEN_YEAR), tenors.index(LONG)
        # L up when the 10Y rises; S up when the curve steepens; C up when the belly cheapens.
        orientation = (
            loadings[0, ten],
            loadings[1, long_] - loadings[1, short],
            loadings[2, belly] - (loadings[2, short] + loadings[2, long_]) / 2.0,
        )
        for row, value in enumerate(orientation):
            if value < 0:
                loadings[row] *= -1.0

        return cls(
            tenors=tenors,
            loadings=loadings,
            mean_level=curve_train.mean().to_numpy(),
            explained=eigenvalues[top] / eigenvalues.sum(),
        )

    def scores(self, curve: pd.DataFrame) -> pd.DataFrame:
        """L, S, C in basis points for every date in `curve`."""
        if tuple(str(name) for name in curve.columns) != self.tenors:
            raise ValueError("curve columns do not match the fitted tenors")
        centered = (curve.to_numpy() - self.mean_level) * BP_PER_PERCENT
        return pd.DataFrame(centered @ self.loadings.T, index=curve.index, columns=list(FACTORS))
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_pca.py -q`

Esperado: `6 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_pca.py src/termo/features/pca.py
git commit -m "feat: add curve PCA with sign convention"
````

### Tarea 7: Velocidad suavizada y volatilidad

La velocidad promedia tres ventanas que terminan hoy. La volatilidad es una media exponencial de cambios al cuadrado. Ambas usan solo el pasado.

**Archivos:**
- Crear: `src/termo/features/velocity.py`
- Crear: `src/termo/features/volatility.py`
- Prueba: `tests/test_velocity_volatility.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_velocity_volatility.py`

````python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from termo.features.velocity import smoothed_change
from termo.features.volatility import ewm_vol


def test_smoothed_change_of_a_straight_line() -> None:
    line = pd.Series(np.arange(100, dtype=float) * 2.0)  # +2 per day
    change = smoothed_change(line, horizon=21, delta=5)
    # mean of 2*16, 2*21 and 2*26
    assert change.iloc[-1] == pytest.approx(42.0)
    assert change.iloc[:26].isna().all()
    assert change.iloc[26:].notna().all()


def test_old_move_leaving_one_window_is_diluted() -> None:
    step = pd.Series(np.where(np.arange(80) >= 30, 30.0, 0.0))  # +30 on day 30
    change = smoothed_change(step, horizon=21, delta=5)
    assert change.iloc[30] == pytest.approx(30.0)  # today's move counts in full
    assert change.iloc[45] == pytest.approx(30.0)  # still inside the three windows
    assert change.iloc[48] == pytest.approx(20.0)  # left the 16-day window only
    assert change.iloc[53] == pytest.approx(10.0)  # left the 21-day window too
    assert change.iloc[60] == pytest.approx(0.0)  # left all of them


def test_smoothed_change_rejects_bad_windows() -> None:
    with pytest.raises(ValueError):
        smoothed_change(pd.Series([1.0, 2.0]), horizon=5, delta=5)


@given(st.floats(min_value=0.1, max_value=50.0), st.floats(min_value=1.0, max_value=200.0))
def test_vol_of_constant_absolute_moves_is_that_move(move: float, halflife: float) -> None:
    changes = pd.Series(np.tile([move, -move], 50))
    assert ewm_vol(changes, halflife).iloc[-1] == pytest.approx(move)


def test_vol_reacts_faster_with_a_short_halflife() -> None:
    changes = pd.Series([1.0] * 200 + [10.0] * 5)
    assert ewm_vol(changes, 20).iloc[-1] > ewm_vol(changes, 120).iloc[-1]


def test_vol_uses_the_past_only() -> None:
    rng = np.random.default_rng(1)
    changes = pd.Series(rng.normal(size=300))
    altered = changes.copy()
    altered.iloc[200:] = 99.0
    assert np.allclose(ewm_vol(changes, 60).iloc[:200], ewm_vol(altered, 60).iloc[:200])


def test_vol_rejects_non_positive_halflife() -> None:
    with pytest.raises(ValueError):
        ewm_vol(pd.Series([1.0, 2.0]), 0)
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_velocity_volatility.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.features.velocity'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/features/velocity.py`

````python
"""Change of a factor over a horizon, smoothed over neighbouring windows."""

from __future__ import annotations

import pandas as pd


def smoothed_change(series: pd.Series, horizon: int, delta: int) -> pd.Series:
    """Average of the changes over horizon-delta, horizon and horizon+delta days.

    All three windows end today, so an old move leaving one window is diluted
    while today's move counts in full.
    """
    if not 0 < delta < horizon:
        raise ValueError("need 0 < delta < horizon")
    windows = (horizon - delta, horizon, horizon + delta)
    return sum(series.diff(window) for window in windows) / float(len(windows))
````

`src/termo/features/volatility.py`

````python
"""Exponentially weighted volatility of daily yield changes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewm_vol(changes: pd.Series, halflife: float) -> pd.Series:
    """Root of the exponentially weighted mean of squared changes (weights sum to one)."""
    if halflife <= 0:
        raise ValueError("halflife must be positive")
    return np.sqrt((changes**2).ewm(halflife=halflife, adjust=True).mean())
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_velocity_volatility.py -q`

Esperado: `7 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_velocity_volatility.py src/termo/features/velocity.py src/termo/features/volatility.py
git commit -m "feat: add smoothed velocity and EWM volatility"
````

### Tarea 8: Pipeline de las 12 variables

Une PCA, velocidades y volatilidades en logaritmos; recorta a 3 desviaciones y estandariza con estadísticas de la ventana de entrenamiento. Aquí viven las dos pruebas más importantes del proyecto: sin look-ahead y ajuste solo en entrenamiento.

**Archivos:**
- Crear: `src/termo/features/pipeline.py`
- Prueba: `tests/test_pipeline.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_pipeline.py`

````python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from termo.features.pipeline import (
    FEATURE_NAMES,
    fit_pipeline,
    raw_features,
    select_columns,
)

BURN_IN = 252


@pytest.fixture(scope="module")
def train_end(curve: pd.DataFrame) -> pd.Timestamp:
    return curve.index[999]


def test_twelve_features_without_the_level(curve: pd.DataFrame, train_end: pd.Timestamp) -> None:
    pipeline = fit_pipeline(curve, train_end, FEATURE_NAMES, BURN_IN)
    features = pipeline.transform(curve)
    assert tuple(features.columns) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 12
    assert "L" not in features.columns
    assert features.index[0] == curve.index[BURN_IN]
    assert np.isfinite(features.to_numpy()).all()


def test_training_rows_are_standardized_and_clipped(
    curve: pd.DataFrame, train_end: pd.Timestamp
) -> None:
    shocked = curve.copy()
    shocked.iloc[600:] += 4.0  # a one-day jump of 400 bp: an extreme velocity and volatility
    pipeline = fit_pipeline(shocked, train_end, FEATURE_NAMES, BURN_IN)
    train = pipeline.transform(shocked).loc[:train_end]
    assert np.allclose(train.mean().to_numpy(), 0.0, atol=1e-8)
    assert np.allclose(train.std(ddof=0).to_numpy(), 1.0, atol=1e-8)
    raw = pipeline.raw(shocked).loc[:train_end]
    mean, std = raw.mean(), raw.std(ddof=0)
    assert (((raw - mean) / std).abs() > 3.0).to_numpy().any(), "the shock must create extremes"
    clipped = pipeline.clipper.transform(raw)
    assert ((clipped - mean).abs() <= 3.0 * std + 1e-9).to_numpy().all()
    assert not clipped.equals(raw)


def test_no_look_ahead_in_the_features(curve: pd.DataFrame, train_end: pd.Timestamp) -> None:
    """Changing the future must not change any feature of the past."""
    pipeline = fit_pipeline(curve, train_end, FEATURE_NAMES, BURN_IN)
    cut = 1200
    altered = curve.copy()
    altered.iloc[cut:] = altered.iloc[cut:] + 1.5
    before = pipeline.transform(curve).iloc[: cut - BURN_IN]
    after = pipeline.transform(altered).iloc[: cut - BURN_IN]
    pd.testing.assert_frame_equal(before, after)


def test_fit_uses_the_training_window_only(curve: pd.DataFrame, train_end: pd.Timestamp) -> None:
    """Changing data after the training window must not change the fitted pipeline."""
    altered = curve.copy()
    altered.loc[altered.index > train_end] += 2.0
    original = fit_pipeline(curve, train_end, FEATURE_NAMES, BURN_IN)
    refit = fit_pipeline(altered, train_end, FEATURE_NAMES, BURN_IN)
    assert np.array_equal(original.pca.loadings, refit.pca.loadings)
    assert np.array_equal(original.clipper.lb, refit.clipper.lb)
    assert np.array_equal(original.scaler.scaler.mean_, refit.scaler.scaler.mean_)
    pd.testing.assert_frame_equal(
        original.transform(curve.loc[:train_end]), refit.transform(altered.loc[:train_end])
    )


def test_training_window_can_start_late(curve: pd.DataFrame) -> None:
    start, end = curve.index[800], curve.index[-1]
    late = fit_pipeline(curve, end, FEATURE_NAMES, BURN_IN, train_start=start)
    window = late.transform(curve).loc[start:]
    assert np.allclose(window.mean().to_numpy(), 0.0, atol=1e-8)
    early = fit_pipeline(curve, end, FEATURE_NAMES, BURN_IN)
    assert not np.allclose(late.pca.mean_level, early.pca.mean_level)


def test_vol_features_are_logs(curve: pd.DataFrame, train_end: pd.Timestamp) -> None:
    pipeline = fit_pipeline(curve, train_end, FEATURE_NAMES, BURN_IN)
    raw = raw_features(curve, pipeline.pca)
    changes = curve["DGS5"].diff() * 100.0
    vol60 = np.sqrt((changes**2).ewm(halflife=60).mean())
    vol20 = np.sqrt((changes**2).ewm(halflife=20).mean())
    assert np.allclose(raw["logvol5_60"].iloc[BURN_IN:], np.log(vol60).iloc[BURN_IN:])
    assert np.allclose(raw["logr5_20_60"].iloc[BURN_IN:], np.log(vol20 / vol60).iloc[BURN_IN:])


def test_non_finite_features_are_refused(curve: pd.DataFrame, train_end: pd.Timestamp) -> None:
    pipeline = fit_pipeline(curve, train_end, FEATURE_NAMES, BURN_IN)
    flat = curve.copy()
    flat.iloc[:] = 5.0  # no changes at all: zero volatility, log of zero
    with pytest.raises(ValueError, match="non-finite"):
        pipeline.transform(flat)


def test_select_columns_drops_the_later_of_a_correlated_pair() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    b = rng.normal(size=500)
    frame = pd.DataFrame({"a": a, "b": b, "a_copy": a + 0.01 * rng.normal(size=500), "c": -b})
    assert select_columns(frame, 0.8) == ("a", "b")
    assert select_columns(frame, 1.0) == ("a", "b", "a_copy", "c")
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_pipeline.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.features.pipeline'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/features/pipeline.py`

````python
"""From the yield curve to the standardized daily feature vector."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD

from termo.features.pca import BP_PER_PERCENT, CurvePCA
from termo.features.velocity import smoothed_change
from termo.features.volatility import ewm_vol

FEATURE_NAMES: tuple[str, ...] = (
    "S",
    "C",
    "dL21",
    "dL63",
    "dS21",
    "dS63",
    "dC63",
    "logvol5_60",
    "logr5_20_60",
    "logr5_60_120",
    "logr2_20_60",
    "logr10_20_60",
)
LEVEL_CHANGE = "dL63"
# name, factor, horizon, delta
VELOCITIES: tuple[tuple[str, str, int, int], ...] = (
    ("dL21", "L", 21, 5),
    ("dL63", "L", 63, 10),
    ("dS21", "S", 21, 5),
    ("dS63", "S", 63, 10),
    ("dC63", "C", 63, 10),
)
CLIP_STD = 3.0


def raw_features(curve: pd.DataFrame, pca: CurvePCA) -> pd.DataFrame:
    """The 12 features before clipping and scaling. Row t uses data up to t only."""
    scores = pca.scores(curve)
    changes = curve.diff() * BP_PER_PERCENT
    out = pd.DataFrame(index=curve.index)
    out["S"] = scores["S"]
    out["C"] = scores["C"]
    for name, factor, horizon, delta in VELOCITIES:
        out[name] = smoothed_change(scores[factor], horizon, delta)
    vol5 = {halflife: ewm_vol(changes["DGS5"], halflife) for halflife in (20, 60, 120)}
    out["logvol5_60"] = np.log(vol5[60])
    out["logr5_20_60"] = np.log(vol5[20] / vol5[60])
    out["logr5_60_120"] = np.log(vol5[60] / vol5[120])
    for name, tenor in (("logr2_20_60", "DGS2"), ("logr10_20_60", "DGS10")):
        out[name] = np.log(ewm_vol(changes[tenor], 20) / ewm_vol(changes[tenor], 60))
    return out[list(FEATURE_NAMES)]


@dataclass(frozen=True, eq=False)
class FittedPipeline:
    pca: CurvePCA
    columns: tuple[str, ...]
    burn_in: int
    clipper: DataClipperStd
    scaler: StandardScalerPD

    def raw(self, curve: pd.DataFrame) -> pd.DataFrame:
        """All 12 raw features after the burn-in. `curve` must start at the sample start."""
        raw = raw_features(curve, self.pca).iloc[self.burn_in :]
        if not np.isfinite(raw.to_numpy()).all():
            raise ValueError("non-finite feature values after the burn-in")
        return raw

    def transform(self, curve: pd.DataFrame) -> pd.DataFrame:
        selected = self.raw(curve)[list(self.columns)]
        return self.scaler.transform(self.clipper.transform(selected))


def fit_pipeline(
    curve: pd.DataFrame,
    train_end: pd.Timestamp,
    columns: Sequence[str],
    burn_in: int,
    train_start: pd.Timestamp | None = None,
) -> FittedPipeline:
    """Fit PCA, clipping bounds and z-score on the training window only."""
    window = curve.loc[:train_end] if train_start is None else curve.loc[train_start:train_end]
    pca = CurvePCA.fit(window)
    raw = raw_features(curve.loc[:train_end], pca).iloc[burn_in:]
    train = raw.loc[window.index[0] :, list(columns)]
    if train.empty:
        raise ValueError("training window is empty after the burn-in")
    clipper = DataClipperStd(mul=CLIP_STD).fit(train)
    scaler = StandardScalerPD().fit(clipper.transform(train))
    return FittedPipeline(
        pca=pca, columns=tuple(columns), burn_in=burn_in, clipper=clipper, scaler=scaler
    )


def select_columns(standardized_train: pd.DataFrame, max_abs_corr: float) -> tuple[str, ...]:
    """Keep each column unless it correlates above the limit with one already kept."""
    corr = standardized_train.corr().abs()
    kept: list[str] = []
    for name in standardized_train.columns:
        if all(corr.loc[name, other] <= max_abs_corr for other in kept):
            kept.append(str(name))
    return tuple(kept)
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_pipeline.py -q`

Esperado: `8 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_pipeline.py src/termo/features/pipeline.py
git commit -m "feat: add feature pipeline without look-ahead"
````

### Tarea 9: Métricas básicas y separación en exceso

ARI, eta-cuadrado, cambio futuro, lectura semanal y duraciones. La separación se mide en exceso: eta-cuadrado menos lo que la misma serie de fases logra desplazada en el tiempo. Sin esa resta, un modelo con más fases gana por pura suerte.

**Archivos:**
- Crear: `src/termo/validation/metrics.py`
- Prueba: `tests/test_metrics.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_metrics.py`

````python
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from termo.validation.metrics import (
    adjusted_rand,
    chance_eta_squared,
    count_jumps,
    eta_squared,
    excess_eta_squared,
    forward_change_bp,
    median_durations,
    run_lengths,
    weekly_last,
)


def test_adjusted_rand_ignores_the_names() -> None:
    assert adjusted_rand(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])) == pytest.approx(1.0)


def test_eta_squared_known_value() -> None:
    values = np.array([1.0, 3.0, 5.0, 7.0])
    groups = np.array([0, 0, 1, 1])
    # group means 2 and 6, grand mean 4: between = 16, total = 20
    assert eta_squared(values, groups) == pytest.approx(0.8)


def test_eta_squared_edge_cases() -> None:
    assert eta_squared(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0])) == 0.0
    assert eta_squared(np.array([2.0, 2.0]), np.array([0, 1])) == 0.0
    assert eta_squared(np.array([]), np.array([])) == 0.0
    assert eta_squared(np.array([1.0, 1.0, 9.0, 9.0]), np.array([0, 0, 1, 1])) == pytest.approx(1.0)


@given(
    arrays(np.float64, 30, elements=st.floats(min_value=-100.0, max_value=100.0)),
    arrays(np.int64, 30, elements=st.integers(min_value=0, max_value=3)),
)
def test_eta_squared_is_a_share(values: np.ndarray, groups: np.ndarray) -> None:
    assert -1e-9 <= eta_squared(values, groups) <= 1.0 + 1e-9


def persistent(rng: np.random.Generator, n_rows: int, n_states: int) -> np.ndarray:
    return (rng.integers(0, n_states) + np.cumsum(rng.random(n_rows) < 0.05)) % n_states


def test_chance_eta_grows_with_the_number_of_states() -> None:
    """Unrelated persistent groups explain variance by luck, and more groups explain more."""
    two, five = [], []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        noise = rng.normal(size=403)
        values = (
            noise[3:] + noise[2:-1] + noise[1:-2] + noise[:-3]
        )  # overlapping, like 4-week moves
        two.append(chance_eta_squared(values, persistent(rng, 400, 2)))
        five.append(chance_eta_squared(values, persistent(rng, 400, 5)))
    assert 0.0 < np.mean(two) < np.mean(five)


def test_excess_eta_is_zero_on_average_without_a_real_link() -> None:
    excess = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        values = rng.normal(size=300)
        excess.append(excess_eta_squared(values, persistent(rng, 300, 5)))
    assert abs(float(np.mean(excess))) < 0.01


def test_excess_eta_keeps_a_real_link() -> None:
    rng = np.random.default_rng(0)
    groups = persistent(rng, 400, 2)
    values = np.where(groups == 1, 3.0, -3.0) + rng.normal(size=400)
    assert eta_squared(values, groups) > 0.8
    assert excess_eta_squared(values, groups) > 0.7
    assert chance_eta_squared(np.array([1.0]), np.array([0])) == 0.0


def test_forward_change_is_in_basis_points_and_blank_at_the_end() -> None:
    yields = pd.Series([4.00, 4.10, 4.05, 4.30])
    change = forward_change_bp(yields, 2)
    assert change.iloc[0] == pytest.approx(5.0)
    assert change.iloc[1] == pytest.approx(20.0)
    assert change.iloc[2:].isna().all()


def test_weekly_last_takes_the_last_available_day() -> None:
    index = pd.bdate_range("2024-01-01", periods=12)  # Mon 1 Jan .. Tue 16 Jan
    series = pd.Series(range(12), index=index).drop(pd.Timestamp("2024-01-12"))  # no Friday
    weekly = weekly_last(series)
    assert list(weekly.index.strftime("%Y-%m-%d")) == ["2024-01-05", "2024-01-11", "2024-01-16"]
    assert weekly.tolist() == [4, 8, 11]


def test_run_lengths_and_durations() -> None:
    labels = np.array([0, 0, 0, 1, 1, 0, 2, 2, 2, 2])
    assert run_lengths(labels) == [(0, 3), (1, 2), (0, 1), (2, 4)]
    assert median_durations(labels, 4) == {
        0: 2.0,
        1: 2.0,
        2: 4.0,
        3: pytest.approx(math.nan, nan_ok=True),
    }
    assert count_jumps(labels) == 3
    assert run_lengths(np.array([], dtype=int)) == []


@given(st.lists(st.integers(min_value=0, max_value=2), min_size=1, max_size=100))
def test_run_lengths_rebuild_the_series(raw: list[int]) -> None:
    labels = np.array(raw)
    runs = run_lengths(labels)
    rebuilt = np.concatenate([np.full(length, state) for state, length in runs])
    assert (rebuilt == labels).all()
    assert len(runs) == count_jumps(labels) + 1
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_metrics.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.validation.metrics'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/validation/metrics.py`

````python
"""Small, model-free measurements used by every test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

BP_PER_PERCENT = 100.0
VARIANCE_FLOOR = 1e-12  # relative; below this the values are constant up to rounding


def adjusted_rand(first: np.ndarray, second: np.ndarray) -> float:
    return float(adjusted_rand_score(first, second))


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Share of the variance of `values` explained by the group means (0 to 1)."""
    if len(values) == 0:
        return 0.0
    grand_mean = values.mean()
    total = float(((values - grand_mean) ** 2).sum())
    if total <= VARIANCE_FLOOR * max(1.0, float((values**2).sum())):
        return 0.0  # constant values: nothing to explain
    between = 0.0
    for group in np.unique(groups):
        member = values[groups == group]
        between += len(member) * float((member.mean() - grand_mean) ** 2)
    return between / total


def chance_eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Eta-squared that the same group series gets by luck: its mean over every time shift.

    Persistent groups explain some variance of an autocorrelated series by chance,
    and more groups explain more. Shifting the groups in time keeps their number and
    their persistence and breaks any real link with `values`.
    """
    n_rows = len(values)
    if n_rows < 2:
        return 0.0
    shifted = [eta_squared(values, np.roll(groups, shift)) for shift in range(1, n_rows)]
    return float(np.mean(shifted))


def excess_eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Eta-squared above what luck alone gives. This is the separation measure."""
    return eta_squared(values, groups) - chance_eta_squared(values, groups)


def forward_change_bp(yields: pd.Series, horizon: int) -> pd.Series:
    """Change from t to t+horizon rows, in basis points; NaN where the future is not in the data."""
    return (yields.shift(-horizon) - yields) * BP_PER_PERCENT


def weekly_last(series: pd.Series) -> pd.Series:
    """The last available row of each Monday-to-Friday week."""
    return series.groupby(series.index.to_period("W-FRI")).tail(1)


def run_lengths(labels: np.ndarray) -> list[tuple[int, int]]:
    """Consecutive runs as (state, length), in order."""
    runs: list[tuple[int, int]] = []
    if len(labels) == 0:
        return runs
    change_points = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [len(labels)]))
    for start, end in zip(starts, ends, strict=True):
        runs.append((int(labels[start]), int(end - start)))
    return runs


def median_durations(labels: np.ndarray, n_states: int) -> dict[int, float]:
    """Median run length per state; NaN for a state that never appears."""
    by_state: dict[int, list[int]] = {state: [] for state in range(n_states)}
    for state, length in run_lengths(labels):
        by_state[state].append(length)
    return {
        state: float(np.median(lengths)) if lengths else float("nan")
        for state, lengths in by_state.items()
    }


def count_jumps(labels: np.ndarray) -> int:
    return int((labels[1:] != labels[:-1]).sum())
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_metrics.py -q`

Esperado: `11 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_metrics.py src/termo/validation/metrics.py
git commit -m "feat: add validation metrics with chance-corrected separation"
````

### Tarea 10: Modelos de régimen detrás de una interfaz

El jump model y la línea base K-means exponen los mismos tres métodos. K-means usa scikit-learn: es el mismo objetivo que el jump model con multa cero, pero el ajuste con `jumpmodels` tarda más de un minuto por corrida.

**Archivos:**
- Crear: `src/termo/regime/model.py`
- Prueba: `tests/test_regime_model.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_regime_model.py`

````python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from termo.regime.model import RegimeFitter, jump_fitter, kmeans_fitter
from termo.validation.metrics import count_jumps


@pytest.fixture(scope="module")
def planted() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(3)
    truth = np.repeat([0, 1, 0, 1, 0], 200)
    centers = np.array([[-1.0] * 4, [1.0] * 4])
    values = centers[truth] + rng.normal(size=(len(truth), 4))
    index = pd.bdate_range("2000-01-03", periods=len(truth))
    return pd.DataFrame(values, index=index, columns=list("abcd")), truth


@pytest.mark.parametrize("fitter", [jump_fitter(2, 30.0), kmeans_fitter(2)])
def test_models_share_one_interface(
    fitter: RegimeFitter, planted: tuple[pd.DataFrame, np.ndarray]
) -> None:
    features, truth = planted
    model = fitter(features)
    assert model.n_states == 2
    for labels in (
        model.insample_labels(),
        model.online_labels(features),
        model.full_labels(features),
    ):
        assert labels.shape == (len(features),)
        assert labels.dtype.kind == "i"
        assert set(np.unique(labels)) <= {0, 1}
    assert adjusted_rand_score(truth, model.insample_labels()) > 0.5


def test_jump_penalty_makes_regimes_persistent(planted: tuple[pd.DataFrame, np.ndarray]) -> None:
    features, truth = planted
    jump = jump_fitter(2, 30.0)(features).insample_labels()
    kmeans = kmeans_fitter(2)(features).insample_labels()
    assert count_jumps(jump) == count_jumps(truth) == 4
    assert count_jumps(kmeans) > 10 * count_jumps(jump)


@pytest.mark.parametrize("fitter", [jump_fitter(2, 30.0), kmeans_fitter(2)])
def test_online_labels_use_the_past_only(
    fitter: RegimeFitter, planted: tuple[pd.DataFrame, np.ndarray]
) -> None:
    features, _ = planted
    model = fitter(features.iloc[:600])
    altered = features.copy()
    altered.iloc[800:] = 5.0
    assert (model.online_labels(features)[:800] == model.online_labels(altered)[:800]).all()
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_regime_model.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.regime.model'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/regime/model.py`

````python
"""Regime models behind one small interface: the jump model and the K-means baseline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from sklearn.cluster import KMeans

N_INIT = 10
RANDOM_STATE = 0


class RegimeModel(Protocol):
    @property
    def n_states(self) -> int: ...

    def insample_labels(self) -> np.ndarray:
        """Labels of the training rows, as fitted."""
        ...

    def online_labels(self, features: pd.DataFrame) -> np.ndarray:
        """Label of row t using rows up to t only."""
        ...

    def full_labels(self, features: pd.DataFrame) -> np.ndarray:
        """Labels decoded with the whole sequence in view."""
        ...


RegimeFitter = Callable[[pd.DataFrame], RegimeModel]


@dataclass(frozen=True, eq=False)
class JumpRegimeModel:
    model: JumpModel
    n_states: int

    def insample_labels(self) -> np.ndarray:
        return np.asarray(self.model.labels_, dtype=int)

    def online_labels(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict_online(features), dtype=int)

    def full_labels(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(features), dtype=int)


@dataclass(frozen=True, eq=False)
class KMeansRegimeModel:
    model: KMeans
    n_states: int

    def insample_labels(self) -> np.ndarray:
        return np.asarray(self.model.labels_, dtype=int)

    def online_labels(self, features: pd.DataFrame) -> np.ndarray:
        # No memory: the nearest centroid today is the same with or without the future.
        return np.asarray(self.model.predict(features.to_numpy()), dtype=int)

    def full_labels(self, features: pd.DataFrame) -> np.ndarray:
        return self.online_labels(features)


def jump_fitter(n_states: int, jump_penalty: float) -> RegimeFitter:
    def fit(features: pd.DataFrame) -> RegimeModel:
        model = JumpModel(
            n_components=n_states,
            jump_penalty=jump_penalty,
            cont=False,
            n_init=N_INIT,
            random_state=RANDOM_STATE,
        )
        model.fit(features)
        return JumpRegimeModel(model=model, n_states=n_states)

    return fit


def kmeans_fitter(n_states: int) -> RegimeFitter:
    """The jump model with zero penalty, fitted with scikit-learn because it is far faster."""

    def fit(features: pd.DataFrame) -> RegimeModel:
        model = KMeans(n_clusters=n_states, n_init=N_INIT, random_state=RANDOM_STATE)
        model.fit(features.to_numpy())
        return KMeansRegimeModel(model=model, n_states=n_states)

    return fit
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_regime_model.py -q`

Esperado: `5 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_regime_model.py src/termo/regime/model.py
git commit -m "feat: add regime model interface with jump model and k-means"
````

### Tarea 11: Nombres de fase estables

Cada reentrenamiento puede permutar las etiquetas. Se renombran por coincidencia con el modelo anterior (algoritmo húngaro).

**Archivos:**
- Crear: `src/termo/regime/align.py`
- Prueba: `tests/test_align.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_align.py`

````python
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from termo.regime.align import apply_permutation, match_to_reference, order_by_target


def test_order_by_target_names_the_lowest_mean_zero() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    target = np.array([5.0, 5.0, -3.0, -3.0, 1.0, 1.0])
    permutation = order_by_target(labels, target, 3)
    assert permutation.tolist() == [2, 0, 1]
    assert apply_permutation(labels, permutation).tolist() == [2, 2, 0, 0, 1, 1]


def test_order_by_target_puts_empty_states_last() -> None:
    labels = np.array([2, 2, 0, 0])
    target = np.array([1.0, 1.0, 4.0, 4.0])
    assert order_by_target(labels, target, 3).tolist() == [1, 2, 0]


def test_match_rejects_different_lengths() -> None:
    with pytest.raises(ValueError):
        match_to_reference(np.array([0, 1]), np.array([0, 1, 1]), 2)


@given(
    st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=200),
    st.permutations([0, 1, 2, 3]),
)
def test_match_undoes_any_renaming(reference: list[int], renaming: list[int]) -> None:
    truth = np.array(reference)
    scrambled = np.array(renaming)[truth]
    permutation = match_to_reference(truth, scrambled, 4)
    assert sorted(permutation.tolist()) == [0, 1, 2, 3]
    assert (apply_permutation(scrambled, permutation) == truth).all()


def test_match_tolerates_some_disagreement() -> None:
    reference = np.array([0] * 50 + [1] * 50)
    labels = np.array([1] * 45 + [0] * 55)  # swapped names, boundary moved by 5 rows
    renamed = apply_permutation(labels, match_to_reference(reference, labels, 2))
    assert (renamed == reference).mean() == pytest.approx(0.95)
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_align.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.regime.align'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/regime/align.py`

````python
"""Keep regime names stable across refits. A permutation maps old label -> new label."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def order_by_target(labels: np.ndarray, target: np.ndarray, n_states: int) -> np.ndarray:
    """Name states by the mean of `target` inside each one, lowest first; empty states last."""
    means = np.full(n_states, np.inf)
    for state in range(n_states):
        mask = labels == state
        if mask.any():
            means[state] = float(target[mask].mean())
    permutation = np.empty(n_states, dtype=int)
    permutation[np.argsort(means, kind="stable")] = np.arange(n_states)
    return permutation


def match_to_reference(reference: np.ndarray, labels: np.ndarray, n_states: int) -> np.ndarray:
    """Rename `labels` so that they agree with `reference` on as many rows as possible."""
    if reference.shape != labels.shape:
        raise ValueError("reference and labels must cover the same rows")
    overlap = np.zeros((n_states, n_states), dtype=int)
    np.add.at(overlap, (labels, reference), 1)
    old, new = linear_sum_assignment(-overlap)
    permutation = np.empty(n_states, dtype=int)
    permutation[old] = new
    return permutation


def apply_permutation(labels: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    renamed: np.ndarray = permutation[labels]
    return renamed
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_align.py -q`

Esperado: `5 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_align.py src/termo/regime/align.py
git commit -m "feat: add regime label alignment"
````

### Tarea 12: Remuestreo por bloques y prueba de independencia

Los intervalos salen de bloques, no de semanas sueltas. La prueba de independencia desplaza las fases en el tiempo usando TODOS los desplazamientos: dejar fuera los pequeños hace que rechace de más (medido: 4% en lugar de 1%).

**Archivos:**
- Crear: `src/termo/validation/bootstrap.py`
- Prueba: `tests/test_bootstrap.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_bootstrap.py`

````python
from __future__ import annotations

import numpy as np
import pytest

from termo.validation.bootstrap import (
    chi2_statistic,
    circular_shift_test,
    moving_block_indices,
    paired_excess_eta_difference,
)
from termo.validation.metrics import eta_squared


def persistent_groups(n_rows: int, run: int) -> np.ndarray:
    return (np.arange(n_rows) // run) % 2


def test_block_indices_are_contiguous_blocks() -> None:
    indices = moving_block_indices(100, 10, np.random.default_rng(0))
    assert indices.shape == (100,)
    assert indices.min() >= 0 and indices.max() < 100
    assert (np.diff(indices.reshape(10, 10), axis=1) == 1).all()


def test_block_indices_reject_bad_block() -> None:
    with pytest.raises(ValueError):
        moving_block_indices(10, 11, np.random.default_rng(0))


def test_paired_difference_detects_the_informative_grouping() -> None:
    rng = np.random.default_rng(0)
    informative = persistent_groups(400, 20)
    noise_groups = persistent_groups(400, 7)
    values = np.where(informative == 1, 10.0, -10.0) + rng.normal(scale=5.0, size=400)
    result = paired_excess_eta_difference(values, informative, noise_groups, 26, 300, seed=0)
    assert result.point > 0.5
    assert 0.0 < result.low < result.point < result.high


def test_paired_difference_of_a_grouping_with_itself_is_zero() -> None:
    rng = np.random.default_rng(0)
    groups = persistent_groups(300, 15)
    values = rng.normal(size=300)
    result = paired_excess_eta_difference(values, groups, groups, 26, 100, seed=0)
    assert (result.point, result.low, result.high) == (0.0, 0.0, 0.0)


def test_more_states_get_no_head_start() -> None:
    """Five unrelated states against two unrelated states: no advantage after the correction."""
    raw, corrected = [], []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        values = rng.normal(size=300)
        five = (rng.integers(0, 5) + np.cumsum(rng.random(300) < 0.05)) % 5
        two = (rng.integers(0, 2) + np.cumsum(rng.random(300) < 0.05)) % 2
        raw.append(eta_squared(values, five) - eta_squared(values, two))
        corrected.append(paired_excess_eta_difference(values, five, two, 26, 20, seed).point)
    assert np.mean(raw) > 0.01
    assert abs(float(np.mean(corrected))) < 0.01


def test_paired_difference_is_reproducible() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    a, b = persistent_groups(200, 10), persistent_groups(200, 25)
    assert paired_excess_eta_difference(
        values, a, b, 20, 50, seed=7
    ) == paired_excess_eta_difference(values, a, b, 20, 50, seed=7)


def test_chi2_known_value() -> None:
    groups = np.array([0] * 20 + [1] * 20)
    signs = np.array([1] * 15 + [0] * 5 + [1] * 5 + [0] * 15)
    # expected 10 in every cell; (5^2 / 10) * 4 = 10
    assert chi2_statistic(groups, signs) == pytest.approx(10.0)
    assert chi2_statistic(groups, np.ones(40, dtype=int)) == 0.0


def test_circular_shift_finds_real_dependence() -> None:
    rng = np.random.default_rng(0)
    # Runs of random length: a strictly periodic series would match itself after a shift.
    groups = (np.cumsum(rng.random(400) < 0.05) % 2).astype(int)
    signs = np.where(rng.random(400) < 0.9, groups, 1 - groups)
    result = circular_shift_test(groups, signs)
    assert result.p_value < 0.01


def test_circular_shift_is_not_fooled_by_persistence_alone() -> None:
    """Two persistent but unrelated series: the test must keep its nominal error rate.

    A naive chi-square test rejects most of these pairs.
    """
    rejections, naive_rejections = 0, 0
    for seed in range(300):
        rng = np.random.default_rng(seed)
        # Random starting state: otherwise both series begin aligned at zero.
        groups = ((rng.integers(0, 2) + np.cumsum(rng.random(300) < 0.03)) % 2).astype(int)
        signs = ((rng.integers(0, 2) + np.cumsum(rng.random(300) < 0.03)) % 2).astype(int)
        result = circular_shift_test(groups, signs)
        rejections += result.p_value < 0.01
        naive_rejections += result.statistic > 6.63  # 1% point of chi-square with 1 d.o.f.
    # Nominal: 3 of 300. Leaving out the small shifts gives about 15.
    assert rejections <= 7
    assert naive_rejections > 100


def test_circular_shift_smallest_p_value_is_one_over_n() -> None:
    groups = np.array([0] * 60 + [1] * 60)
    result = circular_shift_test(groups, groups)
    assert result.p_value == pytest.approx(2 / 120)  # the mirror shift ties with the observed one


def test_circular_shift_needs_enough_rows() -> None:
    with pytest.raises(ValueError):
        circular_shift_test(np.zeros(50, dtype=int), np.zeros(50, dtype=int))
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_bootstrap.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.validation.bootstrap'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/validation/bootstrap.py`

````python
"""Resampling that respects autocorrelation: blocks, never single weeks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from termo.validation.metrics import chance_eta_squared, eta_squared

CONFIDENCE_TAILS = (2.5, 97.5)
MIN_ROWS_INDEPENDENCE = 100  # so that p < 0.01 is reachable


@dataclass(frozen=True)
class EtaDifference:
    point: float
    low: float
    high: float


@dataclass(frozen=True)
class IndependenceTest:
    statistic: float
    p_value: float


def moving_block_indices(n_rows: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one moving-block bootstrap resample of length n_rows."""
    if not 1 <= block <= n_rows:
        raise ValueError("need 1 <= block <= n_rows")
    n_blocks = -(-n_rows // block)
    starts = rng.integers(0, n_rows - block + 1, size=n_blocks)
    indices: np.ndarray = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n_rows]
    return indices


def paired_excess_eta_difference(
    values: np.ndarray,
    groups_a: np.ndarray,
    groups_b: np.ndarray,
    block: int,
    n_resamples: int,
    seed: int,
) -> EtaDifference:
    """Excess eta2 of A minus excess eta2 of B, with a 95% interval.

    The interval comes from resampling the same blocks of rows for both groupings.
    It is then shifted by the difference of their chance levels, so that a grouping
    with more states gets no head start.
    """
    rng = np.random.default_rng(seed)
    raw_point = eta_squared(values, groups_a) - eta_squared(values, groups_b)
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        rows = moving_block_indices(len(values), block, rng)
        draws[i] = eta_squared(values[rows], groups_a[rows]) - eta_squared(
            values[rows], groups_b[rows]
        )
    low, high = np.percentile(draws, CONFIDENCE_TAILS)
    head_start = chance_eta_squared(values, groups_a) - chance_eta_squared(values, groups_b)
    return EtaDifference(
        point=raw_point - head_start, low=float(low) - head_start, high=float(high) - head_start
    )


def chi2_statistic(groups: np.ndarray, signs: np.ndarray) -> float:
    """Pearson chi-square of the groups x signs table; 0 when the table is degenerate."""
    _, group_codes = np.unique(groups, return_inverse=True)
    _, sign_codes = np.unique(signs, return_inverse=True)
    table = np.zeros((group_codes.max() + 1, sign_codes.max() + 1))
    np.add.at(table, (group_codes, sign_codes), 1.0)
    if min(table.shape) < 2:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    return float(((table - expected) ** 2 / expected).sum())


def circular_shift_test(groups: np.ndarray, signs: np.ndarray) -> IndependenceTest:
    """Is the group/sign association larger than under every other time offset?

    Shifting the group series circularly keeps its autocorrelation and breaks its
    alignment with the signs. Every non-zero shift is used: leaving out the small
    ones makes the test reject too often, because those are the shifts most likely
    to match a large observed statistic. The smallest possible p-value is 1 / n_rows.
    """
    n_rows = len(groups)
    if n_rows < MIN_ROWS_INDEPENDENCE:
        raise ValueError(f"need at least {MIN_ROWS_INDEPENDENCE} rows")
    observed = chi2_statistic(groups, signs)
    exceed = sum(
        chi2_statistic(np.roll(groups, shift), signs) >= observed for shift in range(1, n_rows)
    )
    return IndependenceTest(statistic=observed, p_value=(1 + exceed) / n_rows)
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_bootstrap.py -q`

Esperado: `11 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_bootstrap.py src/termo/validation/bootstrap.py
git commit -m "feat: add block bootstrap and circular-shift independence test"
````

### Tarea 13: Walk-forward y datos del experimento

Reentrenar con el pasado, leer el bloque siguiente en línea y mantener los nombres. `dataset.py` prepara una sola vez lo que comparten todas las configuraciones.

**Archivos:**
- Crear: `src/termo/validation/walkforward.py`
- Crear: `src/termo/dataset.py`
- Prueba: `tests/test_walkforward.py`
- Prueba: `tests/test_dataset.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_walkforward.py`

````python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.dataset import ExperimentData
from termo.regime.model import jump_fitter
from termo.validation.metrics import adjusted_rand
from termo.validation.walkforward import inertia_labels, refit_cutoffs, run_walkforward


def test_cutoffs_step_every_refit_weeks() -> None:
    dates = pd.bdate_range("2000-01-03", "2001-12-31")
    cutoffs = refit_cutoffs(dates, pd.Timestamp("2000-12-31"), 26)
    assert [c.strftime("%Y-%m-%d") for c in cutoffs] == ["2000-12-29", "2001-06-29", "2001-12-28"]


def test_cutoffs_need_out_of_sample_dates() -> None:
    dates = pd.bdate_range("2000-01-03", "2000-06-30")
    with pytest.raises(ValueError, match="no out-of-sample"):
        refit_cutoffs(dates, pd.Timestamp("2000-12-31"), 26)
    with pytest.raises(ValueError, match="before the first"):
        refit_cutoffs(dates, pd.Timestamp("1999-12-31"), 26)


def test_blocks_tile_the_out_of_sample_period(data: ExperimentData, config: CoreConfig) -> None:
    refits = data.refits
    assert len(refits) == 8
    assert refits[0].cutoff == pd.Timestamp("1994-06-30")
    for earlier, later in zip(refits[:-1], refits[1:], strict=True):
        assert earlier.block_end == later.cutoff
    assert refits[-1].block_end == data.curve.index[-1]
    assert refits[-1].block_end < pd.Timestamp(config.holdout_start)
    for refit in refits:
        assert refit.features.index[-1] == refit.block_end
        assert refit.features.index.equals(refit.level_change.index)


def test_walkforward_reads_every_out_of_sample_day_once(data: ExperimentData) -> None:
    result = run_walkforward(data.refits, jump_fitter(2, 50.0), 2, data.daily_change_10y)
    expected = data.curve.index[data.curve.index > data.refits[0].cutoff]
    assert result.oos_labels.index.equals(expected)
    assert len(result.consecutive_ari) == len(data.refits) - 1
    assert all(-1.0 <= value <= 1.0 for value in result.consecutive_ari)


def test_walkforward_finds_the_planted_regimes_with_stable_names(
    data: ExperimentData, curve_and_regimes: tuple[pd.DataFrame, np.ndarray]
) -> None:
    curve, regimes = curve_and_regimes
    truth = pd.Series(regimes, index=curve.index)
    result = run_walkforward(data.refits, jump_fitter(2, 10.0), 2, data.daily_change_10y)
    labels = result.oos_labels
    # Online reading recognizes a new regime with a lag, so agreement is high but not perfect.
    assert adjusted_rand(truth.loc[labels.index].to_numpy(), labels.to_numpy()) > 0.4
    # State 0 is the rally (yields falling), state 1 the sell-off, in every block.
    change = data.daily_change_10y.loc[labels.index]
    assert change[labels == 0].mean() < 0 < change[labels == 1].mean()
    assert (labels == truth.loc[labels.index]).mean() > 0.8


def test_walkforward_has_no_look_ahead(data: ExperimentData) -> None:
    """Labels of a block must not change when later blocks are altered."""
    first_two = data.refits[:2]
    result = run_walkforward(data.refits, jump_fitter(2, 50.0), 2, data.daily_change_10y)
    partial = run_walkforward(first_two, jump_fitter(2, 50.0), 2, data.daily_change_10y)
    pd.testing.assert_series_equal(
        result.oos_labels.loc[partial.oos_labels.index], partial.oos_labels
    )


def test_inertia_labels_follow_the_sign_of_the_level_change(data: ExperimentData) -> None:
    labels = inertia_labels(data.refits)
    assert set(labels.unique()) <= {0, 1}
    first = data.refits[0]
    block = first.level_change.loc[first.level_change.index > first.cutoff]
    assert (labels.loc[block.index] == (block > 0).astype(int)).all()
````

`tests/test_dataset.py`

````python
from __future__ import annotations

import numpy as np
import pandas as pd

from termo.config import CoreConfig
from termo.dataset import ExperimentData, first_window_columns
from termo.features.pipeline import FEATURE_NAMES


def test_first_window_columns_are_a_subset_in_order(
    pre_holdout: pd.DataFrame, config: CoreConfig
) -> None:
    columns = first_window_columns(pre_holdout, config)
    assert columns[0] == "S"
    assert [name for name in FEATURE_NAMES if name in columns] == list(columns)


def test_pre_holdout_data_never_reaches_the_holdout(data: ExperimentData) -> None:
    holdout_start = pd.Timestamp(data.config.holdout_start)
    assert data.curve.index[-1] < holdout_start
    assert all(refit.features.index[-1] < holdout_start for refit in data.refits)
    assert np.isfinite(data.full_features.to_numpy()).all()


def test_ten_year_series_are_exposed(data: ExperimentData) -> None:
    assert data.yields_10y.equals(data.curve["DGS10"])
    change = data.daily_change_10y
    assert np.isnan(change.iloc[0])
    expected = (data.curve["DGS10"].iloc[5] - data.curve["DGS10"].iloc[4]) * 100.0
    assert np.isclose(change.iloc[5], expected)
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_walkforward.py tests/test_dataset.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.dataset'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/validation/walkforward.py`

````python
"""Refit on the past, read the next block online, keep regime names stable."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from termo.features.pipeline import LEVEL_CHANGE, fit_pipeline
from termo.regime.align import apply_permutation, match_to_reference, order_by_target
from termo.regime.model import RegimeFitter
from termo.validation.metrics import adjusted_rand


@dataclass(frozen=True, eq=False)
class RefitData:
    cutoff: pd.Timestamp  # last training day
    block_end: pd.Timestamp  # last day read with this fit
    features: pd.DataFrame  # standardized, every feature date up to block_end
    level_change: pd.Series  # raw smoothed 63-day change of L, same dates


@dataclass(frozen=True, eq=False)
class WalkForwardResult:
    oos_labels: pd.Series  # daily out-of-sample labels with stable names
    consecutive_ari: tuple[float, ...]  # one per pair of consecutive refits


def refit_cutoffs(
    feature_dates: pd.DatetimeIndex, first_train_end: pd.Timestamp, refit_weeks: int
) -> list[pd.Timestamp]:
    """Last training day of each refit: first_train_end, then every refit_weeks weeks."""
    last = feature_dates[-1]
    cutoffs: list[pd.Timestamp] = []
    target = first_train_end
    while True:
        eligible = feature_dates[feature_dates <= target]
        if len(eligible) == 0:
            raise ValueError("first_train_end is before the first feature date")
        cutoff = eligible[-1]
        if cutoff >= last:
            break
        if not cutoffs or cutoff > cutoffs[-1]:
            cutoffs.append(cutoff)
        target = target + pd.Timedelta(weeks=refit_weeks)
    if not cutoffs:
        raise ValueError("no out-of-sample dates after first_train_end")
    return cutoffs


def build_refits(
    curve: pd.DataFrame, cutoffs: Sequence[pd.Timestamp], columns: Sequence[str], burn_in: int
) -> list[RefitData]:
    """Fit the feature pipeline at every cutoff. Shared by all model configurations."""
    block_ends = [*cutoffs[1:], curve.index[-1]]
    refits: list[RefitData] = []
    for cutoff, block_end in zip(cutoffs, block_ends, strict=True):
        pipeline = fit_pipeline(curve, cutoff, columns, burn_in)
        visible = curve.loc[:block_end]
        refits.append(
            RefitData(
                cutoff=cutoff,
                block_end=block_end,
                features=pipeline.transform(visible),
                level_change=pipeline.raw(visible)[LEVEL_CHANGE],
            )
        )
    return refits


def run_walkforward(
    refits: Sequence[RefitData], fitter: RegimeFitter, n_states: int, target: pd.Series
) -> WalkForwardResult:
    """`target` names the states of the first fit (lowest in-state mean becomes state 0)."""
    previous: pd.Series | None = None
    blocks: list[pd.Series] = []
    consecutive: list[float] = []
    for refit in refits:
        train = refit.features.loc[: refit.cutoff]
        model = fitter(train)
        fitted = model.insample_labels()
        if previous is None:
            permutation = order_by_target(fitted, target.reindex(train.index).to_numpy(), n_states)
        else:
            shared = len(previous)
            if not train.index[:shared].equals(previous.index):
                raise ValueError("training windows must be nested")
            consecutive.append(adjusted_rand(previous.to_numpy(), fitted[:shared]))
            permutation = match_to_reference(previous.to_numpy(), fitted[:shared], n_states)
        previous = pd.Series(apply_permutation(fitted, permutation), index=train.index)
        online = pd.Series(
            apply_permutation(model.online_labels(refit.features), permutation),
            index=refit.features.index,
        )
        blocks.append(online.loc[online.index > refit.cutoff])
    return WalkForwardResult(oos_labels=pd.concat(blocks), consecutive_ari=tuple(consecutive))


def inertia_labels(refits: Sequence[RefitData]) -> pd.Series:
    """Baseline: 1 when the smoothed 63-day change of the level is positive, else 0."""
    blocks = [
        (refit.level_change.loc[refit.level_change.index > refit.cutoff] > 0).astype(int)
        for refit in refits
    ]
    return pd.concat(blocks)
````

`src/termo/dataset.py`

````python
"""Everything a model configuration needs, prepared once and shared by all of them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from termo.config import CoreConfig
from termo.features.pipeline import FEATURE_NAMES, fit_pipeline, select_columns
from termo.validation.metrics import BP_PER_PERCENT
from termo.validation.walkforward import RefitData, build_refits, refit_cutoffs

TEN_YEAR = "DGS10"


@dataclass(frozen=True, eq=False)
class ExperimentData:
    config: CoreConfig
    curve: pd.DataFrame
    columns: tuple[str, ...]
    refits: tuple[RefitData, ...]
    full_features: pd.DataFrame  # pipeline fitted on the whole curve; used for FTIC only

    @property
    def yields_10y(self) -> pd.Series:
        return self.curve[TEN_YEAR]

    @property
    def daily_change_10y(self) -> pd.Series:
        return self.curve[TEN_YEAR].diff() * BP_PER_PERCENT


def first_window_columns(curve: pd.DataFrame, config: CoreConfig) -> tuple[str, ...]:
    """Apply the collinearity rule once, on the first training window."""
    first_end = pd.Timestamp(config.first_train_end)
    pipeline = fit_pipeline(curve, first_end, FEATURE_NAMES, config.burn_in_days)
    train = pipeline.transform(curve.loc[:first_end])
    return select_columns(train, config.thresholds.collinearity_max)


def prepare(curve: pd.DataFrame, config: CoreConfig, columns: Sequence[str]) -> ExperimentData:
    """Fit the feature pipeline at every refit date. `curve` must start at the sample start."""
    feature_dates = curve.index[config.burn_in_days :]
    cutoffs = refit_cutoffs(feature_dates, pd.Timestamp(config.first_train_end), config.refit_weeks)
    refits = build_refits(curve, cutoffs, columns, config.burn_in_days)
    full = fit_pipeline(curve, curve.index[-1], columns, config.burn_in_days).transform(curve)
    return ExperimentData(
        config=config,
        curve=curve,
        columns=tuple(columns),
        refits=tuple(refits),
        full_features=full,
    )
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_walkforward.py tests/test_dataset.py -q`

Esperado: `10 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_walkforward.py tests/test_dataset.py src/termo/validation/walkforward.py src/termo/dataset.py
git commit -m "feat: add walk-forward runner and shared experiment data"
````

### Tarea 14: Estabilidad

S2 entrena un modelo en cada mitad de la muestra y compara cómo etiquetan toda la historia. La estabilidad es el promedio de S1 y S2.

**Archivos:**
- Crear: `src/termo/validation/stability.py`
- Prueba: `tests/test_stability.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_stability.py`

````python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.features.pipeline import FEATURE_NAMES
from termo.regime.model import jump_fitter
from termo.validation.stability import halves_ari, stability_score


def test_halves_agree_on_planted_regimes(pre_holdout: pd.DataFrame, config: CoreConfig) -> None:
    value = halves_ari(pre_holdout, FEATURE_NAMES, config.burn_in_days, jump_fitter(2, 50.0))
    assert 0.5 < value <= 1.0


def test_halves_disagree_on_noise(config: CoreConfig) -> None:
    rng = np.random.default_rng(0)
    index = pd.bdate_range("1990-01-01", periods=1400)
    tenors = ["DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS30"]
    walk = 6.0 + np.cumsum(0.03 * rng.normal(size=(1400, 7)), axis=0)
    noise = pd.DataFrame(walk, index=index, columns=tenors)
    value = halves_ari(noise, FEATURE_NAMES, config.burn_in_days, jump_fitter(2, 50.0))
    assert value < 0.5


def test_stability_score_averages_s1_and_s2() -> None:
    assert stability_score([0.9, 0.7], 0.4) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        stability_score([], 0.4)
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_stability.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.validation.stability'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/validation/stability.py`

````python
"""Do the same regimes come out when the model sees different years?"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from termo.features.pipeline import fit_pipeline
from termo.regime.model import RegimeFitter
from termo.validation.metrics import adjusted_rand


def halves_ari(
    curve: pd.DataFrame, columns: Sequence[str], burn_in: int, fitter: RegimeFitter
) -> float:
    """S2: fit one model per half of the sample, let both label the whole sample, compare.

    Each half has its own PCA, clipping and scaling, so nothing is shared but the method.
    """
    dates = curve.index[burn_in:]
    middle = len(dates) // 2
    first_end, second_start, last = dates[middle - 1], dates[middle], dates[-1]
    windows = (
        (fit_pipeline(curve, first_end, columns, burn_in), dates[0], first_end),
        (fit_pipeline(curve, last, columns, burn_in, train_start=second_start), second_start, last),
    )
    labelings: list[np.ndarray] = []
    for pipeline, train_start, train_end in windows:
        features = pipeline.transform(curve)
        model = fitter(features.loc[train_start:train_end])
        labelings.append(model.full_labels(features))
    return adjusted_rand(labelings[0], labelings[1])


def stability_score(consecutive_ari: Sequence[float], halves: float) -> float:
    """Average of S1 (mean ARI between consecutive refits) and S2 (halves)."""
    if not consecutive_ari:
        raise ValueError("stability needs at least two refits")
    return (float(np.mean(consecutive_ari)) + halves) / 2.0
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_stability.py -q`

Esperado: `3 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_stability.py src/termo/validation/stability.py
git commit -m "feat: add stability measures"
````

### Tarea 15: FTIC

Criterio de Fan-Tang para jump models, solo como segunda opinión sobre K.

**Archivos:**
- Crear: `src/termo/validation/ftic.py`
- Prueba: `tests/test_ftic.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_ftic.py`

````python
from __future__ import annotations

import math

import numpy as np
import pytest

from termo.validation.ftic import ftic, within_cluster_ss

COMMON = {
    "wcss_saturated": 200.0,
    "saturated_states": 6,
    "n_obs": 100,
    "n_features": 4,
    "prior_states": 3,
    "prior_jumps": 2.5,
}


def test_wcss_known_value() -> None:
    features = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 1.0], [10.0, 3.0]])
    labels = np.array([0, 0, 1, 1])
    # state 0: mean (1, 0) -> 1 + 1 ; state 1: mean (10, 2) -> 1 + 1
    assert within_cluster_ss(features, labels) == pytest.approx(4.0)


def test_ftic_hand_computed() -> None:
    # a_T = log(log 100) * log 4 = 1.527180 * 1.386294 = 2.117121
    # M   = 2 * (4 + 2.5) + 3 * (5 - 2.5) = 20.5
    # (300 - 200 + 2.117121 * 20.5) / 100 = 1.434010
    # 2 * (log 2 - log 6) = -2.197225
    assert ftic(2, 300.0, 5, **COMMON) == pytest.approx(-0.763215, abs=1e-5)


def test_ftic_penalizes_jumps_and_states() -> None:
    base = ftic(3, 250.0, 5, **COMMON)
    assert ftic(3, 250.0, 6, **COMMON) > base
    assert ftic(4, 250.0, 5, **COMMON) > base
    assert ftic(3, 240.0, 5, **COMMON) < base


def test_ftic_trades_one_jump_for_a_fixed_amount_of_fit() -> None:
    # One more jump costs a_T * K0 in WCSS units, whatever K is.
    cost = math.log(math.log(100)) * math.log(4) * 3
    for n_states in (2, 3, 4):
        assert ftic(n_states, 250.0 - cost, 6, **COMMON) == pytest.approx(
            ftic(n_states, 250.0, 5, **COMMON)
        )
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_ftic.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.validation.ftic'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/validation/ftic.py`

````python
"""Fan-Tang information criterion for jump models (Cortese, Kolm, Lindstrom, AStA 2026).

Used only as a second opinion on the number of states. Lower is better.
"""

from __future__ import annotations

import math

import numpy as np


def within_cluster_ss(features: np.ndarray, labels: np.ndarray) -> float:
    """Sum of squared distances from each row to the mean of its state."""
    total = 0.0
    for state in np.unique(labels):
        member = features[labels == state]
        total += float(((member - member.mean(axis=0)) ** 2).sum())
    return total


def ftic(
    n_states: int,
    wcss: float,
    jumps: int,
    *,
    wcss_saturated: float,
    saturated_states: int,
    n_obs: int,
    n_features: int,
    prior_states: int,
    prior_jumps: float,
) -> float:
    """Eq. (12) with the linearized complexity of eq. (14) and every feature active.

    FTIC = [WCSS - WCSS_sat + a_T * M] / T + 2 * [log K - log K_sat]
    a_T  = log(log T) * log p
    M    = K * (p + jumps0) + K0 * (jumps - jumps0)
    """
    penalty = math.log(math.log(n_obs)) * math.log(n_features)
    complexity = n_states * (n_features + prior_jumps) + prior_states * (jumps - prior_jumps)
    fit_gap = wcss - wcss_saturated
    return (fit_gap + penalty * complexity) / n_obs + 2.0 * (
        math.log(n_states) - math.log(saturated_states)
    )
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_ftic.py -q`

Esperado: `4 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_ftic.py src/termo/validation/ftic.py
git commit -m "feat: add FTIC for jump models"
````

### Tarea 16: PBO y N efectivo

PBO: en cuántas particiones de bloques el ganador de un lado cae en la mitad inferior del otro. N efectivo: cuántas configuraciones son realmente distintas.

**Archivos:**
- Crear: `src/termo/validation/pbo.py`
- Prueba: `tests/test_pbo.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_pbo.py`

````python
from __future__ import annotations

import numpy as np
import pytest

from termo.validation.metrics import eta_squared
from termo.validation.pbo import block_stats, effective_n, eta_from_stats, pbo_cscv


def test_eta_from_stats_matches_the_direct_computation() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=240)
    groups = rng.integers(0, 3, size=240)
    stats = block_stats(values, groups, n_states=5, n_blocks=8)  # two states never used
    assert stats.shape == (8, 5, 3)
    assert stats[..., 0].sum() == 240
    assert eta_from_stats(stats.sum(axis=0)) == pytest.approx(eta_squared(values, groups))
    first_half = eta_from_stats(stats[:4].sum(axis=0))
    assert first_half == pytest.approx(eta_squared(values[:120], groups[:120]))


def test_pbo_is_low_when_one_configuration_is_really_better() -> None:
    rng = np.random.default_rng(0)
    truth = (np.arange(480) // 20) % 2
    values = np.where(truth == 1, 5.0, -5.0) + rng.normal(scale=3.0, size=480)
    good = block_stats(values, truth, 2, 8)
    noise = [block_stats(values, rng.integers(0, 2, size=480), 2, 8) for _ in range(9)]
    assert pbo_cscv(np.stack([good, *noise])) == 0.0


def test_pbo_is_near_half_when_every_configuration_is_noise() -> None:
    estimates = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        values = rng.normal(size=480)
        stats = [block_stats(values, rng.integers(0, 2, size=480), 2, 8) for _ in range(10)]
        estimates.append(pbo_cscv(np.stack(stats)))
    assert 0.3 < float(np.mean(estimates)) < 0.7


def test_pbo_needs_an_even_number_of_blocks() -> None:
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((3, 5, 2, 3)))


def test_effective_n_counts_distinct_labelings() -> None:
    rng = np.random.default_rng(0)
    a = (np.arange(300) // 30) % 2
    b = (np.arange(300) // 50) % 3
    near_a = a.copy()
    near_a[:3] = 1 - near_a[:3]
    relabelled_a = 1 - a
    noise = rng.integers(0, 2, size=300)
    assert effective_n([a, near_a, relabelled_a], cut=0.2) == 1
    assert effective_n([a, near_a, b, noise], cut=0.2) == 3
    assert effective_n([a], cut=0.2) == 1
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_pbo.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.validation.pbo'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/validation/pbo.py`

````python
"""Probability of backtest overfitting (CSCV) and the effective number of trials."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from termo.validation.metrics import VARIANCE_FLOOR, adjusted_rand


def block_stats(values: np.ndarray, groups: np.ndarray, n_states: int, n_blocks: int) -> np.ndarray:
    """Per contiguous block and state: count, sum and sum of squares of `values`.

    Shape (n_blocks, n_states, 3). These add up across blocks, so eta-squared of any
    set of blocks can be computed without touching the rows again.
    """
    edges = np.linspace(0, len(values), n_blocks + 1).astype(int)
    stats = np.zeros((n_blocks, n_states, 3))
    for block in range(n_blocks):
        block_values = values[edges[block] : edges[block + 1]]
        block_groups = groups[edges[block] : edges[block + 1]]
        for state in range(n_states):
            member = block_values[block_groups == state]
            stats[block, state] = (len(member), member.sum(), (member**2).sum())
    return stats


def eta_from_stats(stats: np.ndarray) -> np.ndarray:
    """Eta-squared from stats of shape (..., n_states, 3), summed over the blocks in use."""
    count, total, squares = stats[..., 0], stats[..., 1], stats[..., 2]
    grand = total.sum(axis=-1) ** 2 / count.sum(axis=-1)
    ss_total = squares.sum(axis=-1) - grand
    with np.errstate(divide="ignore", invalid="ignore"):
        per_state = np.where(count > 0, total**2 / count, 0.0)
        eta = (per_state.sum(axis=-1) - grand) / ss_total
    constant = ss_total <= VARIANCE_FLOOR * np.maximum(1.0, squares.sum(axis=-1))
    result: np.ndarray = np.where(constant, 0.0, eta)
    return result


def pbo_cscv(stats: np.ndarray) -> float:
    """Share of block splits where the in-sample winner lands in the bottom half out of sample.

    `stats` has shape (n_configs, n_blocks, n_states, 3) from `block_stats`.
    """
    n_configs, n_blocks = stats.shape[0], stats.shape[1]
    if n_blocks % 2 != 0:
        raise ValueError("the number of blocks must be even")
    splits = list(combinations(range(n_blocks), n_blocks // 2))
    inside = np.zeros((len(splits), n_blocks))
    for row, split in enumerate(splits):
        inside[row, list(split)] = 1.0
    eta_in = eta_from_stats(np.einsum("cb,nbsk->cnsk", inside, stats))
    eta_out = eta_from_stats(np.einsum("cb,nbsk->cnsk", 1.0 - inside, stats))
    winner = eta_in.argmax(axis=1)
    winner_out = eta_out[np.arange(len(splits)), winner]
    rank = (eta_out <= winner_out[:, None]).sum(axis=1)  # 1 = worst, n_configs = best
    omega = rank / (n_configs + 1.0)
    logit = np.log(omega / (1.0 - omega))
    return float((logit <= 0.0).mean())


def effective_n(labelings: Sequence[np.ndarray], cut: float) -> int:
    """Number of clusters of label series under distance 1 - ARI (average linkage)."""
    if len(labelings) < 2:
        return len(labelings)
    distances = [
        max(0.0, 1.0 - adjusted_rand(labelings[i], labelings[j]))
        for i in range(len(labelings))
        for j in range(i + 1, len(labelings))
    ]
    clusters = fcluster(linkage(np.array(distances), method="average"), t=cut, criterion="distance")
    return int(len(set(clusters.tolist())))
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_pbo.py -q`

Esperado: `5 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_pbo.py src/termo/validation/pbo.py
git commit -m "feat: add PBO (CSCV) and effective number of trials"
````

### Tarea 17: Bitácora

Registrar antes de mirar: el código se niega a guardar un resultado sin registro previo, un segundo resultado, o una segunda apertura del holdout.

**Archivos:**
- Crear: `src/termo/validation/trials.py`
- Prueba: `tests/test_trials.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_trials.py`

````python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from termo.validation.trials import TrialLog, TrialLogError, TrialStatus


@pytest.fixture
def log(tmp_path: Path) -> TrialLog:
    return TrialLog(tmp_path / "trials" / "trials.jsonl", clock=lambda: "2026-10-02T12:00:00+00:00")


def register(log: TrialLog, trial_id: str = "jm_k2_lam50") -> None:
    log.register(trial_id, "hypothesis", {"n_states": 2, "jump_penalty": 50.0}, "hash", "commit")


def test_result_without_registration_is_refused(log: TrialLog) -> None:
    with pytest.raises(TrialLogError, match="no prior registration"):
        log.record_result("jm_k2_lam50", {"score": 0.1}, TrialStatus.KEPT, "", None)
    assert log.records() == []


def test_second_result_is_refused(log: TrialLog) -> None:
    register(log)
    log.record_result("jm_k2_lam50", {"score": 0.1}, TrialStatus.KEPT, "", "labels.csv")
    with pytest.raises(TrialLogError, match="already has a result"):
        log.record_result("jm_k2_lam50", {"score": 0.9}, TrialStatus.KEPT, "", "labels.csv")


def test_second_registration_is_refused(log: TrialLog) -> None:
    register(log)
    with pytest.raises(TrialLogError, match="already registered"):
        register(log)


def test_log_is_append_only_json_lines(log: TrialLog) -> None:
    register(log)
    before = log.path.read_text(encoding="utf-8")
    log.record_result("jm_k2_lam50", {"score": 0.1}, TrialStatus.DISCARDED, "short", None)
    after = log.path.read_text(encoding="utf-8")
    assert after.startswith(before)
    first, second = (json.loads(line) for line in after.splitlines())
    assert first["kind"] == "registered" and first["at"] == "2026-10-02T12:00:00+00:00"
    assert first["config"] == {"n_states": 2, "jump_penalty": 50.0}
    assert first["snapshot_hash"] == "hash" and first["code_commit"] == "commit"
    assert second["kind"] == "result" and second["status"] == "discarded"
    assert log.is_registered("jm_k2_lam50") and log.has_result("jm_k2_lam50")
    assert set(log.registrations()) == set(log.results()) == {"jm_k2_lam50"}


def test_holdout_opens_once_and_only_for_a_finished_trial(log: TrialLog) -> None:
    register(log)
    with pytest.raises(TrialLogError, match="no recorded result"):
        log.open_holdout("jm_k2_lam50")
    log.record_result("jm_k2_lam50", {"score": 0.1}, TrialStatus.KEPT, "", None)
    assert not log.holdout_opened()
    log.open_holdout("jm_k2_lam50")
    assert log.holdout_opened()
    with pytest.raises(TrialLogError, match="already been opened"):
        log.open_holdout("jm_k2_lam50")
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_trials.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.validation.trials'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/validation/trials.py`

````python
"""Append-only trial log. A result without a prior registration is refused."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

HOLDOUT_TRIAL = "holdout"


class TrialLogError(RuntimeError):
    """The log-before-you-look rule would be broken."""


class RecordKind(Enum):
    REGISTERED = "registered"
    RESULT = "result"
    HOLDOUT_OPENED = "holdout_opened"


class TrialStatus(Enum):
    KEPT = "kept"
    DISCARDED = "discarded"
    FAILED = "failed"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class TrialLog:
    path: Path
    clock: Callable[[], str] = _utc_now

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def _ids(self, kind: RecordKind) -> set[str]:
        return {r["trial_id"] for r in self.records() if r["kind"] == kind.value}

    def _append(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # newline fixed so the file is byte-identical on every platform
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def is_registered(self, trial_id: str) -> bool:
        return trial_id in self._ids(RecordKind.REGISTERED)

    def has_result(self, trial_id: str) -> bool:
        return trial_id in self._ids(RecordKind.RESULT)

    def register(
        self,
        trial_id: str,
        hypothesis: str,
        config: Mapping[str, Any],
        snapshot_hash: str,
        code_commit: str,
    ) -> None:
        if self.is_registered(trial_id):
            raise TrialLogError(f"trial {trial_id} is already registered")
        self._append(
            {
                "kind": RecordKind.REGISTERED.value,
                "trial_id": trial_id,
                "at": self.clock(),
                "hypothesis": hypothesis,
                "config": dict(config),
                "snapshot_hash": snapshot_hash,
                "code_commit": code_commit,
            }
        )

    def record_result(
        self,
        trial_id: str,
        metrics: Mapping[str, Any],
        status: TrialStatus,
        reason: str,
        labels_path: str | None,
    ) -> None:
        if not self.is_registered(trial_id):
            raise TrialLogError(f"trial {trial_id} has no prior registration")
        if self.has_result(trial_id):
            raise TrialLogError(f"trial {trial_id} already has a result")
        self._append(
            {
                "kind": RecordKind.RESULT.value,
                "trial_id": trial_id,
                "at": self.clock(),
                "metrics": dict(metrics),
                "status": status.value,
                "reason": reason,
                "labels_path": labels_path,
            }
        )

    def results(self) -> dict[str, dict[str, Any]]:
        return {r["trial_id"]: r for r in self.records() if r["kind"] == RecordKind.RESULT.value}

    def registrations(self) -> dict[str, dict[str, Any]]:
        return {
            r["trial_id"]: r for r in self.records() if r["kind"] == RecordKind.REGISTERED.value
        }

    def holdout_opened(self) -> bool:
        return HOLDOUT_TRIAL in self._ids(RecordKind.HOLDOUT_OPENED)

    def open_holdout(self, final_trial_id: str) -> None:
        """Record, once and for ever, that the holdout has been looked at."""
        if self.holdout_opened():
            raise TrialLogError("the holdout has already been opened")
        if not self.has_result(final_trial_id):
            raise TrialLogError(f"final model {final_trial_id} has no recorded result")
        self._append(
            {
                "kind": RecordKind.HOLDOUT_OPENED.value,
                "trial_id": HOLDOUT_TRIAL,
                "at": self.clock(),
                "final_trial_id": final_trial_id,
            }
        )
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_trials.py -q`

Esperado: `5 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_trials.py src/termo/validation/trials.py
git commit -m "feat: add append-only trial log"
````

### Tarea 18: Evaluación de una configuración

Junta todo para un par (K, lambda): walk-forward, estabilidad, separación, duraciones, y los insumos de FTIC. También las pruebas de criterio y la evaluación de holdout.

**Archivos:**
- Crear: `src/termo/experiment.py`
- Prueba: `tests/test_experiment.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_experiment.py`

````python
from __future__ import annotations

import json

import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.dataset import ExperimentData, prepare
from termo.experiment import (
    ConfigEvaluation,
    evaluate_config,
    gate_tests,
    holdout_evaluation,
    pbo_of,
    saturated_wcss,
    separation_frame,
)
from termo.features.pipeline import FEATURE_NAMES
from termo.validation.walkforward import inertia_labels


@pytest.fixture(scope="module")
def jump(data: ExperimentData) -> ConfigEvaluation:
    return evaluate_config(data, 2, 50.0)


@pytest.fixture(scope="module")
def kmeans(data: ExperimentData) -> ConfigEvaluation:
    return evaluate_config(data, 2, None)


def test_separation_frame_is_weekly_and_drops_open_horizons(data: ExperimentData) -> None:
    labels = inertia_labels(data.refits)
    frame = separation_frame(labels, data.yields_10y, 20)
    assert list(frame.columns) == ["label", "change"]
    assert frame.index.to_period("W-FRI").is_unique
    assert frame.notna().all().all()
    # The last 20 days have no 20-day future inside the data.
    assert frame.index[-1] <= data.curve.index[-21]
    row = frame.index[3]
    position = data.curve.index.get_loc(row)
    expected = (data.yields_10y.iloc[position + 20] - data.yields_10y.iloc[position]) * 100.0
    assert frame.loc[row, "change"] == pytest.approx(expected)


def test_jump_model_beats_kmeans_on_persistence(
    jump: ConfigEvaluation, kmeans: ConfigEvaluation
) -> None:
    assert jump.passes_duration
    assert min(jump.durations.values()) >= 20
    assert jump.jumps < kmeans.jumps
    assert jump.wcss >= kmeans.wcss  # the penalty buys persistence with fit


def test_evaluation_metrics_are_consistent(jump: ConfigEvaluation) -> None:
    assert jump.s1_min <= jump.s1_mean <= 1.0
    assert jump.stability == pytest.approx((jump.s1_mean + jump.s2) / 2.0)
    assert jump.score == pytest.approx(jump.stability * jump.excess_short)
    assert 0.0 < jump.excess_short < jump.eta_short <= 1.0
    metrics = jump.metrics()
    assert json.loads(json.dumps(metrics)) == metrics
    assert set(metrics["durations"]) == {"0", "1"}


def test_gate_tests_are_well_formed(jump: ConfigEvaluation, data: ExperimentData) -> None:
    gate = gate_tests(jump.oos_labels, inertia_labels(data.refits), data.yields_10y, data.config)
    assert gate.n_weeks > 150
    assert gate.eta_difference.low <= gate.eta_difference.point <= gate.eta_difference.high
    assert 0.0 < gate.independence.p_value <= 1.0


def test_gate_tests_pass_an_oracle_and_fail_the_baseline_itself(data: ExperimentData) -> None:
    inertia = inertia_labels(data.refits)
    # An oracle that knows the sign of the next 20-day move (look-ahead on purpose).
    future = data.yields_10y.shift(-data.config.horizon_short_days) - data.yields_10y
    oracle = (future.loc[inertia.index] > 0).astype(int)
    passed = gate_tests(oracle, inertia, data.yields_10y, data.config)
    assert passed.eta_difference.low > 0.0
    assert passed.independence.p_value < 0.01
    same = gate_tests(inertia, inertia, data.yields_10y, data.config)
    assert same.eta_difference.point == 0.0 and same.eta_difference.low == 0.0


def test_pbo_runs_on_label_series(
    jump: ConfigEvaluation, kmeans: ConfigEvaluation, data: ExperimentData
) -> None:
    value = pbo_of([jump.oos_labels, kmeans.oos_labels], data.yields_10y, data.config)
    assert 0.0 <= value <= 1.0


def test_saturated_model_fits_better_than_any_candidate(
    jump: ConfigEvaluation, data: ExperimentData
) -> None:
    assert 0.0 < saturated_wcss(data) < jump.wcss


def test_holdout_evaluation_reads_only_holdout_weeks(
    curve: pd.DataFrame, config: CoreConfig
) -> None:
    full = prepare(curve, config, FEATURE_NAMES)
    result = holdout_evaluation(full, 2, 50.0)
    holdout_days = int((curve.index >= pd.Timestamp(config.holdout_start)).sum())
    assert 0 < result.n_weeks <= holdout_days // 5 + 1
    assert result.n_episodes >= 1
    assert -1.0 <= result.excess_model <= 1.0 and -1.0 <= result.excess_inertia <= 1.0
    assert result.passed == (result.excess_model >= result.excess_inertia)
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_experiment.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.experiment'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/experiment.py`

````python
"""Evaluate one model configuration end to end, without look-ahead."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from termo.config import CoreConfig
from termo.dataset import ExperimentData
from termo.regime.model import RegimeFitter, jump_fitter, kmeans_fitter
from termo.validation.bootstrap import (
    EtaDifference,
    IndependenceTest,
    circular_shift_test,
    paired_excess_eta_difference,
)
from termo.validation.ftic import within_cluster_ss
from termo.validation.metrics import (
    count_jumps,
    eta_squared,
    excess_eta_squared,
    forward_change_bp,
    median_durations,
    run_lengths,
    weekly_last,
)
from termo.validation.pbo import block_stats, pbo_cscv
from termo.validation.stability import halves_ari, stability_score
from termo.validation.walkforward import inertia_labels, run_walkforward


@dataclass(frozen=True, eq=False)
class ConfigEvaluation:
    n_states: int
    jump_penalty: float | None  # None is the K-means baseline
    oos_labels: pd.Series
    s1_mean: float
    s1_min: float
    s2: float
    stability: float
    eta_short: float  # raw eta-squared of the forward 10Y change, short horizon
    excess_short: float  # the same above its chance level: the separation measure
    excess_long: float
    durations: dict[int, float]
    passes_duration: bool
    wcss: float
    jumps: int

    @property
    def score(self) -> float:
        return self.stability * self.excess_short

    def metrics(self) -> dict[str, Any]:
        return {
            "s1_mean": self.s1_mean,
            "s1_min": self.s1_min,
            "s2": self.s2,
            "stability": self.stability,
            "eta_short": self.eta_short,
            "excess_short": self.excess_short,
            "excess_long": self.excess_long,
            "score": self.score,
            "durations": {str(state): value for state, value in self.durations.items()},
            "passes_duration": self.passes_duration,
            "wcss": self.wcss,
            "jumps": self.jumps,
        }


@dataclass(frozen=True)
class GateResult:
    eta_difference: EtaDifference
    independence: IndependenceTest
    n_weeks: int


@dataclass(frozen=True)
class HoldoutResult:
    excess_model: float
    excess_inertia: float
    n_weeks: int
    n_episodes: int

    @property
    def passed(self) -> bool:
        return self.excess_model >= self.excess_inertia


def separation_frame(labels: pd.Series, yields: pd.Series, horizon: int) -> pd.DataFrame:
    """Weekly readings with the change of `yields` over the next `horizon` rows.

    Readings whose horizon runs past the end of `yields` are dropped.
    """
    weekly = weekly_last(labels)
    change = forward_change_bp(yields, horizon).reindex(weekly.index)
    return pd.DataFrame({"label": weekly, "change": change}).dropna()


def separation(labels: pd.Series, yields: pd.Series, horizon: int) -> tuple[float, float]:
    """(raw eta-squared, excess eta-squared) of the forward change grouped by label."""
    frame = separation_frame(labels, yields, horizon)
    values = frame["change"].to_numpy()
    groups = frame["label"].to_numpy().astype(int)
    return eta_squared(values, groups), excess_eta_squared(values, groups)


def make_fitter(n_states: int, jump_penalty: float | None) -> RegimeFitter:
    return kmeans_fitter(n_states) if jump_penalty is None else jump_fitter(n_states, jump_penalty)


def evaluate_config(
    data: ExperimentData, n_states: int, jump_penalty: float | None
) -> ConfigEvaluation:
    config = data.config
    fitter = make_fitter(n_states, jump_penalty)
    walk = run_walkforward(data.refits, fitter, n_states, data.daily_change_10y)
    s2 = halves_ari(data.curve, data.columns, config.burn_in_days, fitter)
    durations = median_durations(walk.oos_labels.to_numpy(), n_states)
    minimum = config.thresholds.min_median_duration_days
    passes = all(not math.isnan(value) and value >= minimum for value in durations.values())
    fitted = fitter(data.full_features).insample_labels()
    eta_short, excess_short = separation(
        walk.oos_labels, data.yields_10y, config.horizon_short_days
    )
    _, excess_long = separation(walk.oos_labels, data.yields_10y, config.horizon_long_days)
    return ConfigEvaluation(
        n_states=n_states,
        jump_penalty=jump_penalty,
        oos_labels=walk.oos_labels,
        s1_mean=float(np.mean(walk.consecutive_ari)),
        s1_min=float(np.min(walk.consecutive_ari)),
        s2=s2,
        stability=stability_score(walk.consecutive_ari, s2),
        eta_short=eta_short,
        excess_short=excess_short,
        excess_long=excess_long,
        durations=durations,
        passes_duration=passes,
        wcss=within_cluster_ss(data.full_features.to_numpy(), fitted),
        jumps=count_jumps(fitted),
    )


def saturated_wcss(data: ExperimentData) -> float:
    """WCSS of the saturated model of the FTIC: no jump penalty, saturated_k states."""
    labels = kmeans_fitter(data.config.ftic.saturated_k)(data.full_features).insample_labels()
    return within_cluster_ss(data.full_features.to_numpy(), labels)


def gate_tests(
    labels: pd.Series, inertia: pd.Series, yields: pd.Series, config: CoreConfig
) -> GateResult:
    """Excess separation against the inertia baseline, and independence, weekly."""
    frame = separation_frame(labels, yields, config.horizon_short_days)
    values = frame["change"].to_numpy()
    groups = frame["label"].to_numpy().astype(int)
    baseline = weekly_last(inertia).reindex(frame.index).to_numpy().astype(int)
    boot = config.bootstrap
    return GateResult(
        eta_difference=paired_excess_eta_difference(
            values, groups, baseline, boot.block_weeks, boot.n_resamples, boot.seed
        ),
        independence=circular_shift_test(groups, (values > 0).astype(int)),
        n_weeks=len(frame),
    )


def pbo_of(labelings: Sequence[pd.Series], yields: pd.Series, config: CoreConfig) -> float:
    max_states = max(config.k_values)
    stats = []
    for labels in labelings:
        frame = separation_frame(labels, yields, config.horizon_short_days)
        stats.append(
            block_stats(
                frame["change"].to_numpy(),
                frame["label"].to_numpy().astype(int),
                max_states,
                config.pbo_blocks,
            )
        )
    return pbo_cscv(np.stack(stats))


def holdout_evaluation(data: ExperimentData, n_states: int, jump_penalty: float) -> HoldoutResult:
    """`data` must come from a curve loaded with final_evaluation=True."""
    config = data.config
    start = pd.Timestamp(config.holdout_start)
    walk = run_walkforward(
        data.refits, jump_fitter(n_states, jump_penalty), n_states, data.daily_change_10y
    )
    labels = walk.oos_labels.loc[start:]
    frame = separation_frame(labels, data.yields_10y, config.horizon_short_days)
    baseline = weekly_last(inertia_labels(data.refits)).reindex(frame.index)
    values = frame["change"].to_numpy()
    return HoldoutResult(
        excess_model=excess_eta_squared(values, frame["label"].to_numpy().astype(int)),
        excess_inertia=excess_eta_squared(values, baseline.to_numpy().astype(int)),
        n_weeks=len(frame),
        n_episodes=len(run_lengths(labels.to_numpy())),
    )
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_experiment.py -q`

Esperado: `8 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_experiment.py src/termo/experiment.py
git commit -m "feat: evaluate one model configuration end to end"
````

### Tarea 19: Selección de K y lambda

Gana el mayor puntaje entre las que pasan la duración mínima. FTIC opina sobre K a la lambda ganadora; si prefiere menos fases, se propone el modelo más simple.

**Archivos:**
- Crear: `src/termo/selection.py`
- Prueba: `tests/test_selection.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_selection.py`

````python
from __future__ import annotations

from termo.config import FticConfig
from termo.selection import (
    Candidate,
    candidate_from_record,
    ftic_states,
    pick_winner,
    simpler_alternative,
)

FTIC = FticConfig(k0=3, mean_phase_days=40, saturated_k=6, max_jump_fraction=0.4)


def candidate(
    n_states: int,
    jump_penalty: float,
    stability: float = 0.8,
    separation: float = 0.1,
    passes: bool = True,
    wcss: float = 5000.0,
    jumps: int = 20,
) -> Candidate:
    return Candidate(
        trial_id=f"jm_k{n_states}_lam{jump_penalty:g}",
        n_states=n_states,
        jump_penalty=jump_penalty,
        stability=stability,
        separation=separation,
        passes_duration=passes,
        wcss=wcss,
        jumps=jumps,
    )


def test_candidate_from_record() -> None:
    built = candidate_from_record(
        "jm_k3_lam80",
        {"model": "jump", "n_states": 3, "jump_penalty": 80.0},
        {
            "stability": 0.7,
            "excess_short": 0.2,
            "passes_duration": True,
            "wcss": 10.0,
            "jumps": 4,
        },
    )
    assert built == candidate(3, 80.0, 0.7, 0.2, True, 10.0, 4)
    assert built.score == 0.7 * 0.2


def test_winner_has_the_best_score_among_those_passing_duration() -> None:
    flickering = candidate(4, 5.0, stability=0.9, separation=0.5, passes=False)
    solid = candidate(3, 80.0, stability=0.8, separation=0.2)
    weak = candidate(2, 80.0, stability=0.9, separation=0.1)
    assert pick_winner([flickering, solid, weak]) == solid
    assert pick_winner([flickering]) is None
    assert pick_winner([]) is None


def test_ftic_compares_only_the_winning_penalty() -> None:
    candidates = [
        candidate(2, 80.0, wcss=9000.0),
        candidate(3, 80.0, wcss=5000.0),
        candidate(4, 80.0, wcss=4990.0),
        candidate(5, 12.0, wcss=10.0),  # other penalty: ignored
    ]
    kwargs = {"wcss_saturated": 3000.0, "n_obs": 2000, "n_features": 12, "config": FTIC}
    assert ftic_states(candidates, 80.0, **kwargs) == 3
    assert ftic_states(candidates, 999.0, **kwargs) is None


def test_ftic_excludes_models_that_jump_too_often() -> None:
    candidates = [candidate(2, 80.0, wcss=9000.0), candidate(3, 80.0, wcss=10.0, jumps=900)]
    kwargs = {"wcss_saturated": 3000.0, "n_obs": 2000, "n_features": 12, "config": FTIC}
    assert ftic_states(candidates, 80.0, **kwargs) == 2


def test_simpler_alternative_only_when_ftic_wants_fewer_states() -> None:
    two, three, four = candidate(2, 80.0), candidate(3, 80.0), candidate(4, 80.0)
    other_penalty = candidate(2, 12.0)
    pool = [other_penalty, two, three, four]
    assert simpler_alternative(pool, three, 2) == two
    assert simpler_alternative(pool, three, 3) is None
    assert simpler_alternative(pool, three, 4) is None
    assert simpler_alternative(pool, three, None) is None
    assert simpler_alternative([three, four], three, 2) is None
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_selection.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.selection'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/selection.py`

````python
"""Pick K and lambda by walk-forward score; FTIC gives a second opinion on K."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from termo.config import FticConfig
from termo.validation.ftic import ftic


@dataclass(frozen=True)
class Candidate:
    trial_id: str
    n_states: int
    jump_penalty: float
    stability: float
    separation: float  # excess eta-squared at the short horizon
    passes_duration: bool
    wcss: float
    jumps: int

    @property
    def score(self) -> float:
        return self.stability * self.separation


def candidate_from_record(
    trial_id: str, config: Mapping[str, Any], metrics: Mapping[str, Any]
) -> Candidate:
    return Candidate(
        trial_id=trial_id,
        n_states=int(config["n_states"]),
        jump_penalty=float(config["jump_penalty"]),
        stability=float(metrics["stability"]),
        separation=float(metrics["excess_short"]),
        passes_duration=bool(metrics["passes_duration"]),
        wcss=float(metrics["wcss"]),
        jumps=int(metrics["jumps"]),
    )


def pick_winner(candidates: Sequence[Candidate]) -> Candidate | None:
    """Highest stability x separation among configurations that pass the duration filter."""
    eligible = [c for c in candidates if c.passes_duration]
    if not eligible:
        return None
    return max(eligible, key=lambda c: c.score)


def ftic_states(
    candidates: Sequence[Candidate],
    jump_penalty: float,
    *,
    wcss_saturated: float,
    n_obs: int,
    n_features: int,
    config: FticConfig,
) -> int | None:
    """K with the lowest FTIC among the configurations that share `jump_penalty`."""
    prior_jumps = n_obs / config.mean_phase_days
    values: dict[int, float] = {}
    for c in candidates:
        if c.jump_penalty != jump_penalty or c.jumps > config.max_jump_fraction * n_obs:
            continue
        values[c.n_states] = ftic(
            c.n_states,
            c.wcss,
            c.jumps,
            wcss_saturated=wcss_saturated,
            saturated_states=config.saturated_k,
            n_obs=n_obs,
            n_features=n_features,
            prior_states=config.k0,
            prior_jumps=prior_jumps,
        )
    if not values:
        return None
    return min(values, key=lambda k: values[k])


def simpler_alternative(
    candidates: Sequence[Candidate], winner: Candidate, ftic_k: int | None
) -> Candidate | None:
    """When FTIC prefers fewer states than the winner: same lambda, the smaller K."""
    if ftic_k is None or ftic_k >= winner.n_states:
        return None
    for c in candidates:
        if c.n_states == ftic_k and c.jump_penalty == winner.jump_penalty:
            return c
    return None
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_selection.py -q`

Esperado: `5 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_selection.py src/termo/selection.py
git commit -m "feat: add model selection with FTIC second opinión"
````

### Tarea 20: Veredicto y reporte

Tres criterios bloquean (estabilidad, separación contra la inercia, independencia). PBO y S2 se reportan sin bloquear.

**Archivos:**
- Crear: `src/termo/report.py`
- Prueba: `tests/test_report.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_report.py`

````python
from __future__ import annotations

import json
from pathlib import Path

from termo.config import Thresholds
from termo.report import (
    REPORT_JSON,
    REPORT_MD,
    CoreReport,
    Criterion,
    Verdict,
    blocking_criteria,
    decide,
    render_markdown,
    write_report,
)
from termo.validation.bootstrap import EtaDifference

THRESHOLDS = Thresholds(
    stability_min=0.6,
    independence_p_max=0.01,
    min_median_duration_days=20,
    pbo_max=0.05,
    collinearity_max=0.8,
)
GOOD_DIFFERENCE = EtaDifference(point=0.08, low=0.02, high=0.15)


def test_all_blocking_criteria_pass() -> None:
    criteria = blocking_criteria(0.7, GOOD_DIFFERENCE, 0.001, THRESHOLDS)
    assert [c.name for c in criteria] == [
        "stability",
        "separation_vs_inertia_low95",
        "independence_p",
    ]
    assert all(c.passed and c.blocking for c in criteria)
    assert decide(criteria) is Verdict.GO


def test_each_blocking_criterion_can_veto() -> None:
    overlapping = EtaDifference(point=0.08, low=-0.01, high=0.15)
    cases = [
        blocking_criteria(0.59, GOOD_DIFFERENCE, 0.001, THRESHOLDS),
        blocking_criteria(0.7, overlapping, 0.001, THRESHOLDS),
        blocking_criteria(0.7, GOOD_DIFFERENCE, 0.01, THRESHOLDS),
    ]
    for criteria in cases:
        assert sum(not c.passed for c in criteria) == 1
        assert decide(criteria) is Verdict.NO_GO


def test_non_blocking_failure_does_not_veto() -> None:
    criteria = (
        *blocking_criteria(0.7, GOOD_DIFFERENCE, 0.001, THRESHOLDS),
        Criterion("pbo", 0.4, "<= 0.05", passed=False, blocking=False),
    )
    assert decide(criteria) is Verdict.GO


def test_report_files(tmp_path: Path) -> None:
    criteria = blocking_criteria(0.7, GOOD_DIFFERENCE, 0.001, THRESHOLDS)
    report = CoreReport(
        verdict=decide(criteria),
        final_trial_id="jm_k3_lam80",
        criteria=criteria,
        notes=("FTIC prefers K=2; the score prefers K=3.",),
        details={"n_trials": 24},
    )
    write_report(report, tmp_path / "reports")
    markdown = (tmp_path / "reports" / REPORT_MD).read_text(encoding="utf-8")
    assert markdown == render_markdown(report)
    assert "**Verdict: GO**" in markdown
    assert "| stability | 0.7000 | >= 0.6 | pass | yes |" in markdown
    assert "FTIC prefers K=2" in markdown
    assert markdown.isascii()
    payload = json.loads((tmp_path / "reports" / REPORT_JSON).read_text(encoding="utf-8"))
    assert payload["verdict"] == "go"
    assert payload["final_trial_id"] == "jm_k3_lam80"
    assert payload["criteria"][0] == {
        "name": "stability",
        "value": 0.7,
        "requirement": ">= 0.6",
        "passed": True,
        "blocking": True,
    }
    assert payload["details"] == {"n_trials": 24}
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_report.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.report'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/report.py`

````python
"""Turn the measurements into a go/no-go verdict and write it down."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from termo.config import Thresholds
from termo.validation.bootstrap import EtaDifference

REPORT_MD = "go_no_go.md"
REPORT_JSON = "go_no_go.json"


class Verdict(Enum):
    GO = "go"
    NO_GO = "no-go"


@dataclass(frozen=True)
class Criterion:
    name: str
    value: float
    requirement: str
    passed: bool
    blocking: bool  # a failed blocking criterion means no-go


@dataclass(frozen=True)
class CoreReport:
    verdict: Verdict
    final_trial_id: str | None
    criteria: tuple[Criterion, ...]
    notes: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


def blocking_criteria(
    stability: float,
    eta_difference: EtaDifference,
    independence_p: float,
    thresholds: Thresholds,
) -> tuple[Criterion, ...]:
    return (
        Criterion(
            name="stability",
            value=stability,
            requirement=f">= {thresholds.stability_min}",
            passed=stability >= thresholds.stability_min,
            blocking=True,
        ),
        Criterion(
            name="separation_vs_inertia_low95",
            value=eta_difference.low,
            requirement="> 0",
            passed=eta_difference.low > 0.0,
            blocking=True,
        ),
        Criterion(
            name="independence_p",
            value=independence_p,
            requirement=f"< {thresholds.independence_p_max}",
            passed=independence_p < thresholds.independence_p_max,
            blocking=True,
        ),
    )


def decide(criteria: Sequence[Criterion]) -> Verdict:
    if any(c.blocking and not c.passed for c in criteria):
        return Verdict.NO_GO
    return Verdict.GO


def render_markdown(report: CoreReport) -> str:
    lines = [
        "# TERMO - go/no-go",
        "",
        f"**Verdict: {report.verdict.value.upper()}**",
        "",
        f"Final model: `{report.final_trial_id}`",
        "",
        "| Criterion | Value | Requirement | Result | Blocking |",
        "|---|---|---|---|---|",
    ]
    for c in report.criteria:
        result = "pass" if c.passed else "FAIL"
        blocking = "yes" if c.blocking else "no"
        lines.append(f"| {c.name} | {c.value:.4f} | {c.requirement} | {result} | {blocking} |")
    if report.notes:
        lines += ["", "## Notes", ""] + [f"- {note}" for note in report.notes]
    lines += ["", "## Details", "", "```json", json.dumps(report.details, indent=2), "```", ""]
    return "\n".join(lines)


def write_report(report: CoreReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REPORT_MD).write_text(render_markdown(report), encoding="utf-8", newline="\n")
    payload = {
        "verdict": report.verdict.value,
        "final_trial_id": report.final_trial_id,
        "criteria": [asdict(c) for c in report.criteria],
        "notes": list(report.notes),
        "details": report.details,
    }
    (out_dir / REPORT_JSON).write_text(
        json.dumps(payload, indent=2), encoding="utf-8", newline="\n"
    )
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_report.py -q`

Esperado: `4 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_report.py src/termo/report.py
git commit -m "feat: add go/no-go criteria and report writer"
````

### Tarea 21: Las cinco etapas

snapshot, register, run, report y final-holdout. La prueba recorre el flujo completo sobre un FRED falso construido con la curva sintética.

**Archivos:**
- Crear: `src/termo/core.py`
- Prueba: `tests/test_core.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_core.py`

````python
"""The five stages end to end, on a synthetic curve written as a fake FRED snapshot."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.core import (
    HOLDOUT_JSON,
    INERTIA_TRIAL,
    SETUP_TRIAL,
    build_report,
    jump_trial_id,
    kmeans_trial_id,
    load_labels,
    register_trials,
    registered_columns,
    run_final_holdout,
    run_trials,
    save_labels,
    take_snapshot,
)
from termo.data.fred import FRED_URL, DataValidationError
from termo.data.loader import load_curve
from termo.data.snapshot import snapshot_hash
from termo.dataset import prepare
from termo.report import CoreReport, Verdict, write_report
from termo.validation.trials import TrialLog, TrialLogError


def fake_fred(curve: pd.DataFrame) -> Callable[[str], str]:
    def get(url: str) -> str:
        for series_id in curve.columns:
            if url == FRED_URL.format(series_id=series_id):
                lines = [f"observation_date,{series_id}"]
                lines += [f"{d:%Y-%m-%d},{v:.4f}" for d, v in curve[series_id].items()]
                return "\n".join(lines) + "\n"
        raise AssertionError(f"unexpected url {url}")

    return get


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("termo")


@pytest.fixture(scope="module")
def snapshot_dir(workspace: Path, curve: pd.DataFrame, config: CoreConfig) -> Path:
    target = workspace / "data" / "snapshots" / "2026-10-02"
    take_snapshot(config, target, fake_fred(curve), "2026-10-02T00:00:00+00:00")
    return target


@pytest.fixture(scope="module")
def log(workspace: Path) -> TrialLog:
    return TrialLog(workspace / "trials" / "trials.jsonl")


@pytest.fixture(scope="module")
def finished(workspace: Path, snapshot_dir: Path, log: TrialLog, config: CoreConfig) -> Path:
    """Register, run and report once for the whole module. Returns the reports directory."""
    curve = load_curve(snapshot_dir, config.series, config.start, config.holdout_start)
    register_trials(config, curve, log, snapshot_hash(snapshot_dir), "test-commit")
    messages: list[str] = []
    data = prepare(curve, config, registered_columns(log))
    run_trials(data, log, workspace / "trials", echo=messages.append)
    assert len(messages) == 7 and all(m.isascii() and m.startswith("[ok]") for m in messages)
    report = build_report(data, log, workspace / "trials")
    write_report(report, workspace / "reports")
    return workspace / "reports"


def test_snapshot_refuses_invalid_downloads(tmp_path: Path, config: CoreConfig) -> None:
    with pytest.raises(DataValidationError):
        take_snapshot(config, tmp_path / "snap", lambda url: "observation_date,X\n", "now")
    assert not (tmp_path / "snap").exists()


def test_everything_is_registered_before_any_result(
    finished: Path, log: TrialLog, config: CoreConfig
) -> None:
    records = log.records()
    kinds = [r["kind"] for r in records]
    first_result = kinds.index("result")
    assert set(kinds[:first_result]) == {"registered"}
    assert "registered" not in kinds[first_result:]
    expected = {SETUP_TRIAL, INERTIA_TRIAL}
    expected |= {jump_trial_id(k, lam) for k in config.k_values for lam in config.jump_penalties}
    expected |= {kmeans_trial_id(k) for k in config.k_values}
    assert set(log.registrations()) == expected
    assert set(log.results()) == expected - {SETUP_TRIAL}
    assert log.registrations()["jm_k2_lam50"]["code_commit"] == "test-commit"


def test_registration_cannot_be_repeated(
    finished: Path, snapshot_dir: Path, log: TrialLog, config: CoreConfig
) -> None:
    curve = load_curve(snapshot_dir, config.series, config.start, config.holdout_start)
    with pytest.raises(TrialLogError, match="already registered"):
        register_trials(config, curve, log, snapshot_hash(snapshot_dir), "test-commit")


def test_run_is_resumable_and_never_reruns_a_trial(
    finished: Path, workspace: Path, snapshot_dir: Path, log: TrialLog, config: CoreConfig
) -> None:
    curve = load_curve(snapshot_dir, config.series, config.start, config.holdout_start)
    before = log.path.read_text(encoding="utf-8")
    messages: list[str] = []
    run_trials(
        prepare(curve, config, registered_columns(log)), log, workspace / "trials", messages.append
    )
    assert messages == []
    assert log.path.read_text(encoding="utf-8") == before


def test_labels_round_trip(workspace: Path, finished: Path, tmp_path: Path) -> None:
    labels = load_labels(workspace / "trials" / "labels" / "jm_k2_lam50.csv")
    assert labels.index.name == "date" and labels.dtype.kind == "i"
    save_labels(labels, tmp_path / "copy.csv")
    pd.testing.assert_series_equal(load_labels(tmp_path / "copy.csv"), labels)


def test_report_reaches_a_verdict_with_all_criteria(finished: Path, config: CoreConfig) -> None:
    payload = json.loads((finished / "go_no_go.json").read_text(encoding="utf-8"))
    assert payload["verdict"] in {"go", "no-go"}
    assert payload["final_trial_id"] in {
        jump_trial_id(k, lam) for k in config.k_values for lam in config.jump_penalties
    }
    names = [c["name"] for c in payload["criteria"]]
    assert names == [
        "stability",
        "separation_vs_inertia_low95",
        "independence_p",
        "pbo",
        "s2_halves",
    ]
    blocking_failed = any(c["blocking"] and not c["passed"] for c in payload["criteria"])
    assert (payload["verdict"] == "no-go") == blocking_failed
    details = payload["details"]
    assert details["n_trials"] == 4 and 1 <= details["n_effective"] <= 4
    assert len(details["configurations"]) == 4
    assert set(details["baselines_separation"]) == {"inertia", "kmeans_k2", "kmeans_k3"}
    assert (finished / "go_no_go.md").read_text(encoding="utf-8").isascii()


def test_report_needs_every_result(tmp_path: Path, snapshot_dir: Path, config: CoreConfig) -> None:
    curve = load_curve(snapshot_dir, config.series, config.start, config.holdout_start)
    fresh = TrialLog(tmp_path / "trials.jsonl")
    columns = register_trials(config, curve, fresh, "hash", "commit")
    with pytest.raises(TrialLogError, match="without a result"):
        build_report(prepare(curve, config, columns), fresh, tmp_path)


def test_stages_need_a_registration_first(tmp_path: Path) -> None:
    with pytest.raises(TrialLogError, match="register stage first"):
        registered_columns(TrialLog(tmp_path / "trials.jsonl"))


def test_final_holdout_needs_a_go_verdict(
    finished: Path, tmp_path: Path, snapshot_dir: Path, log: TrialLog, config: CoreConfig
) -> None:
    reports = tmp_path / "reports"
    write_report(CoreReport(Verdict.NO_GO, "jm_k2_lam10", criteria=()), reports)
    private_log = TrialLog(tmp_path / "trials.jsonl")
    private_log.path.write_text(log.path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(TrialLogError, match="GO verdict"):
        run_final_holdout(config, snapshot_dir, private_log, reports)
    assert not private_log.holdout_opened()


def test_final_holdout_runs_exactly_once(
    finished: Path, tmp_path: Path, snapshot_dir: Path, log: TrialLog, config: CoreConfig
) -> None:
    reports = tmp_path / "reports"
    write_report(CoreReport(Verdict.GO, "jm_k2_lam10", criteria=()), reports)
    private_log = TrialLog(tmp_path / "trials.jsonl")
    private_log.path.write_text(log.path.read_text(encoding="utf-8"), encoding="utf-8")

    result = run_final_holdout(config, snapshot_dir, private_log, reports)

    assert private_log.holdout_opened()
    assert private_log.records()[-1]["final_trial_id"] == "jm_k2_lam10"
    saved = json.loads((reports / HOLDOUT_JSON).read_text(encoding="utf-8"))
    assert saved["final_trial_id"] == "jm_k2_lam10"
    assert saved["passed"] == result.passed and saved["n_weeks"] == result.n_weeks > 0
    with pytest.raises(TrialLogError, match="already been opened"):
        run_final_holdout(config, snapshot_dir, private_log, reports)
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_core.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.core'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/core.py`

````python
"""The stages of spec 1: snapshot, register, run, report, final holdout."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from termo.config import CoreConfig
from termo.data.fred import fetch_series_csv, parse_series_csv
from termo.data.loader import load_curve
from termo.data.snapshot import write_snapshot
from termo.dataset import ExperimentData, first_window_columns, prepare
from termo.experiment import (
    HoldoutResult,
    evaluate_config,
    gate_tests,
    holdout_evaluation,
    pbo_of,
    saturated_wcss,
    separation,
)
from termo.report import (
    REPORT_JSON,
    CoreReport,
    Criterion,
    Verdict,
    blocking_criteria,
    decide,
)
from termo.selection import (
    Candidate,
    candidate_from_record,
    ftic_states,
    pick_winner,
    simpler_alternative,
)
from termo.validation.pbo import effective_n
from termo.validation.trials import TrialLog, TrialLogError, TrialStatus
from termo.validation.walkforward import inertia_labels

SETUP_TRIAL = "setup"
INERTIA_TRIAL = "inertia"
LABELS_DIR = "labels"
HOLDOUT_JSON = "holdout.json"
MODEL_JUMP, MODEL_KMEANS, MODEL_INERTIA = "jump", "kmeans", "inertia"
KNOWN_LIMITATIONS = (
    "DGS30 between 2002-02-19 and 2006-02-08 is built differently from the rest of the series.",
    "S1 is high almost by construction: consecutive windows share most of their data.",
    "PBO ranks configurations by raw eta-squared, without the chance correction.",
    "Velocities are in basis points, so the 1980s can dominate the extreme regimes.",
    "Evidence for jump models comes from equities; these tests are the criterion for rates.",
)


def jump_trial_id(n_states: int, jump_penalty: float) -> str:
    return f"jm_k{n_states}_lam{jump_penalty:g}"


def kmeans_trial_id(n_states: int) -> str:
    return f"kmeans_k{n_states}"


def take_snapshot(
    config: CoreConfig, snapshot_dir: Path, get: Callable[[str], str], downloaded_at: str
) -> None:
    texts: dict[str, str] = {}
    for series_id in config.series:
        text = fetch_series_csv(series_id, get)
        parse_series_csv(text, series_id)  # refuse to store a file that does not validate
        texts[series_id] = text
    write_snapshot(snapshot_dir, texts, downloaded_at)


def register_trials(
    config: CoreConfig, curve: pd.DataFrame, log: TrialLog, snapshot_hash: str, code_commit: str
) -> tuple[str, ...]:
    """Write every trial to the log before anything is run. Returns the fixed feature set."""
    columns = first_window_columns(curve, config)
    log.register(
        SETUP_TRIAL,
        "Feature set fixed on the first training window; FTIC assumptions fixed in advance.",
        {"columns": list(columns), "ftic": asdict(config.ftic)},
        snapshot_hash,
        code_commit,
    )
    for n_states in config.k_values:
        for jump_penalty in config.jump_penalties:
            log.register(
                jump_trial_id(n_states, jump_penalty),
                f"A jump model with {n_states} states and penalty {jump_penalty:g} "
                "gives stable regimes that separate the forward 10Y move.",
                {"model": MODEL_JUMP, "n_states": n_states, "jump_penalty": jump_penalty},
                snapshot_hash,
                code_commit,
            )
        log.register(
            kmeans_trial_id(n_states),
            "Baseline: the same model without a jump penalty.",
            {"model": MODEL_KMEANS, "n_states": n_states},
            snapshot_hash,
            code_commit,
        )
    log.register(
        INERTIA_TRIAL,
        "Baseline: sign of the smoothed 63-day change of the level.",
        {"model": MODEL_INERTIA},
        snapshot_hash,
        code_commit,
    )
    return columns


def registered_columns(log: TrialLog) -> tuple[str, ...]:
    registrations = log.registrations()
    if SETUP_TRIAL not in registrations:
        raise TrialLogError("no setup registration: run the register stage first")
    return tuple(registrations[SETUP_TRIAL]["config"]["columns"])


def save_labels(labels: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.rename("label").to_csv(path, index_label="date", lineterminator="\n")


def load_labels(path: Path) -> pd.Series:
    return pd.read_csv(path, parse_dates=["date"], index_col="date")["label"]


def run_trials(
    data: ExperimentData, log: TrialLog, trials_dir: Path, echo: Callable[[str], None] = print
) -> None:
    """Run every registered trial that has no result yet. Safe to re-run after an interruption."""
    config = data.config
    for trial_id, registration in log.registrations().items():
        if trial_id == SETUP_TRIAL or log.has_result(trial_id):
            continue
        spec = registration["config"]
        labels_path = trials_dir / LABELS_DIR / f"{trial_id}.csv"
        if spec["model"] == MODEL_INERTIA:
            labels = inertia_labels(data.refits)
            eta, excess = separation(labels, data.yields_10y, config.horizon_short_days)
            save_labels(labels, labels_path)
            log.record_result(
                trial_id,
                {"eta_short": eta, "excess_short": excess},
                TrialStatus.KEPT,
                "baseline",
                str(labels_path),
            )
            echo(f"[ok] {trial_id} excess_short={excess:.4f}")
            continue
        try:
            evaluation = evaluate_config(data, int(spec["n_states"]), spec.get("jump_penalty"))
        except Exception as error:
            log.record_result(trial_id, {}, TrialStatus.FAILED, repr(error), None)
            raise
        save_labels(evaluation.oos_labels, labels_path)
        if spec["model"] == MODEL_KMEANS:
            status, reason = TrialStatus.KEPT, "baseline"
        elif evaluation.passes_duration:
            status, reason = TrialStatus.KEPT, ""
        else:
            status, reason = TrialStatus.DISCARDED, "fails the minimum median duration"
        log.record_result(trial_id, evaluation.metrics(), status, reason, str(labels_path))
        echo(f"[ok] {trial_id} score={evaluation.score:.4f} status={status.value}")


def build_report(data: ExperimentData, log: TrialLog, trials_dir: Path) -> CoreReport:
    config = data.config
    registrations, results = log.registrations(), log.results()
    pending = [t for t in registrations if t != SETUP_TRIAL and t not in results]
    if pending:
        raise TrialLogError(f"trials without a result: {pending}")

    def is_model(trial_id: str, model: str) -> bool:
        ok = results[trial_id]["status"] != TrialStatus.FAILED.value
        return ok and registrations[trial_id]["config"].get("model") == model

    candidates = [
        candidate_from_record(t, registrations[t]["config"], results[t]["metrics"])
        for t in registrations
        if t != SETUP_TRIAL and is_model(t, MODEL_JUMP)
    ]
    details: dict[str, object] = {
        "columns": list(data.columns),
        "configurations": [
            {
                "trial_id": c.trial_id,
                "stability": c.stability,
                "separation": c.separation,
                "score": c.score,
                "passes_duration": c.passes_duration,
            }
            for c in candidates
        ],
        "baselines_separation": {
            t: results[t]["metrics"]["excess_short"]
            for t in registrations
            if t != SETUP_TRIAL and (is_model(t, MODEL_KMEANS) or is_model(t, MODEL_INERTIA))
        },
        "limitations": list(KNOWN_LIMITATIONS),
    }

    winner = pick_winner(candidates)
    if winner is None:
        return CoreReport(
            verdict=Verdict.NO_GO,
            final_trial_id=None,
            criteria=(),
            notes=("No configuration passes the minimum median duration.",),
            details=details,
        )

    labels_dir = trials_dir / LABELS_DIR
    inertia = load_labels(labels_dir / f"{INERTIA_TRIAL}.csv")

    def gates(candidate: Candidate) -> tuple[tuple[Criterion, ...], dict[str, float]]:
        labels = load_labels(labels_dir / f"{candidate.trial_id}.csv")
        gate = gate_tests(labels, inertia, data.yields_10y, config)
        criteria = blocking_criteria(
            candidate.stability, gate.eta_difference, gate.independence.p_value, config.thresholds
        )
        summary = {
            "eta_difference_point": gate.eta_difference.point,
            "eta_difference_low": gate.eta_difference.low,
            "eta_difference_high": gate.eta_difference.high,
            "independence_statistic": gate.independence.statistic,
            "n_weeks": float(gate.n_weeks),
        }
        return criteria, summary

    final = winner
    criteria, gate_summary = gates(winner)
    notes: list[str] = []
    n_obs, n_features = data.full_features.shape
    ftic_k = ftic_states(
        candidates,
        winner.jump_penalty,
        wcss_saturated=saturated_wcss(data),
        n_obs=n_obs,
        n_features=n_features,
        config=config.ftic,
    )
    if ftic_k is not None and ftic_k != winner.n_states:
        notes.append(f"FTIC prefers K={ftic_k}; the score prefers K={winner.n_states}.")
    alternative = simpler_alternative(candidates, winner, ftic_k)
    if alternative is not None and alternative.passes_duration:
        alt_criteria, alt_summary = gates(alternative)
        if decide(alt_criteria) is Verdict.GO:
            final, criteria, gate_summary = alternative, alt_criteria, alt_summary
            notes.append(
                "The simpler model passes every blocking criterion: it is the final model."
            )

    labelings = [load_labels(labels_dir / f"{c.trial_id}.csv") for c in candidates]
    pbo = pbo_of(labelings, data.yields_10y, config)
    s2 = float(results[final.trial_id]["metrics"]["s2"])
    thresholds = config.thresholds
    extra = (
        Criterion("pbo", pbo, f"<= {thresholds.pbo_max}", pbo <= thresholds.pbo_max, False),
        Criterion(
            "s2_halves",
            s2,
            f">= {thresholds.stability_min}",
            s2 >= thresholds.stability_min,
            False,
        ),
    )
    if pbo > thresholds.pbo_max:
        notes.append("PBO above the limit: prune the grid and repeat as new trials.")
    if s2 < thresholds.stability_min:
        notes.append("S2 (halves) is below the threshold: review even though the average passes.")
    details.update(
        {
            "score_winner": winner.trial_id,
            "ftic_states": ftic_k,
            "gate": gate_summary,
            "n_trials": len(candidates),
            "n_effective": effective_n(
                [labels.to_numpy() for labels in labelings], config.effective_n_cut
            ),
        }
    )
    all_criteria = criteria + extra
    return CoreReport(
        verdict=decide(all_criteria),
        final_trial_id=final.trial_id,
        criteria=all_criteria,
        notes=tuple(notes),
        details=details,
    )


def run_final_holdout(
    config: CoreConfig, snapshot_dir: Path, log: TrialLog, reports_dir: Path
) -> HoldoutResult:
    """One-shot evaluation on the holdout. The log refuses a second run."""
    report = json.loads((reports_dir / REPORT_JSON).read_text(encoding="utf-8"))
    final_trial_id = report["final_trial_id"]
    if report["verdict"] != Verdict.GO.value or final_trial_id is None:
        raise TrialLogError("the holdout is only opened for a model with a GO verdict")
    spec = log.registrations()[final_trial_id]["config"]
    columns = registered_columns(log)
    log.open_holdout(final_trial_id)  # recorded before any holdout row is read
    curve = load_curve(
        snapshot_dir, config.series, config.start, config.holdout_start, final_evaluation=True
    )
    result = holdout_evaluation(
        prepare(curve, config, columns), int(spec["n_states"]), float(spec["jump_penalty"])
    )
    payload = {**asdict(result), "passed": result.passed, "final_trial_id": final_trial_id}
    (reports_dir / HOLDOUT_JSON).write_text(
        json.dumps(payload, indent=2), encoding="utf-8", newline="\n"
    )
    return result
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_core.py -q`

Esperado: `10 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_core.py src/termo/core.py
git commit -m "feat: add core stages with log-before-look enforcement"
````

### Tarea 22: Línea de comandos

Envoltura delgada sobre las etapas. Salida solo ASCII.

**Archivos:**
- Crear: `src/termo/cli.py`
- Prueba: `tests/test_cli.py`

- [ ] **Paso 1: Escribir la prueba que falla**

`tests/test_cli.py`

````python
from __future__ import annotations

from pathlib import Path

import pytest

from termo.cli import code_commit, main

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "core.yaml"


def test_stages_other_than_snapshot_need_a_snapshot_directory(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(REPO_CONFIG)])
    assert error.value.code == 2
    assert "--snapshot is required" in capsys.readouterr().err


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(SystemExit) as error:
        main(["tune"])
    assert error.value.code == 2


def test_code_commit_is_a_git_hash() -> None:
    commit = code_commit().removesuffix("-dirty")
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
````

- [ ] **Paso 2: Correr la prueba y ver que falla**

Correr: `.venv/Scripts/python -m pytest tests/test_cli.py -q`

Esperado: FALLA con `ModuleNotFoundError: No module named 'termo.cli'`

- [ ] **Paso 3: Escribir la implementación**

`src/termo/cli.py`

````python
"""Command line for spec 1. Console output is ASCII only."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from termo.config import CoreConfig, load_config
from termo.core import (
    build_report,
    register_trials,
    registered_columns,
    run_final_holdout,
    run_trials,
    take_snapshot,
)
from termo.data.fred import http_get
from termo.data.loader import load_curve
from termo.data.snapshot import snapshot_hash
from termo.dataset import prepare
from termo.report import write_report
from termo.validation.trials import TrialLog

TRIALS_DIR = Path("trials")
REPORTS_DIR = Path("reports")
SNAPSHOTS_DIR = Path("data") / "snapshots"
TRIALS_FILE = "trials.jsonl"


def code_commit() -> str:
    """Current commit, marked dirty when there are uncommitted changes."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return f"{head}-dirty" if status else head


def _pre_holdout_curve(config: CoreConfig, snapshot_dir: Path) -> pd.DataFrame:
    return load_curve(snapshot_dir, config.series, config.start, config.holdout_start)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="termo", description="TERMO spec 1: core go/no-go")
    parser.add_argument("stage", choices=["snapshot", "register", "run", "report", "final-holdout"])
    parser.add_argument("--config", type=Path, default=Path("configs/core.yaml"))
    parser.add_argument(
        "--snapshot", type=Path, help="snapshot directory (all stages but snapshot)"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    log = TrialLog(TRIALS_DIR / TRIALS_FILE)

    if args.stage == "snapshot":
        now = datetime.now(UTC)
        target = SNAPSHOTS_DIR / now.date().isoformat()
        take_snapshot(config, target, http_get, now.isoformat(timespec="seconds"))
        print(f"[ok] snapshot written to {target.as_posix()}")
        return 0

    if args.snapshot is None:
        parser.error("--snapshot is required for this stage")
    snapshot_dir: Path = args.snapshot

    if args.stage == "register":
        curve = _pre_holdout_curve(config, snapshot_dir)
        columns = register_trials(config, curve, log, snapshot_hash(snapshot_dir), code_commit())
        print(f"[ok] registered {len(log.registrations())} trials; features: {', '.join(columns)}")
    elif args.stage == "run":
        curve = _pre_holdout_curve(config, snapshot_dir)
        run_trials(prepare(curve, config, registered_columns(log)), log, TRIALS_DIR)
        print("[ok] all registered trials have a result")
    elif args.stage == "report":
        curve = _pre_holdout_curve(config, snapshot_dir)
        data = prepare(curve, config, registered_columns(log))
        report = build_report(data, log, TRIALS_DIR)
        write_report(report, REPORTS_DIR)
        print(f"[ok] verdict: {report.verdict.value} -> {REPORTS_DIR.as_posix()}/go_no_go.md")
    else:
        result = run_final_holdout(config, snapshot_dir, log, REPORTS_DIR)
        mark = "[ok]" if result.passed else "[x]"
        print(
            f"{mark} holdout: excess_model={result.excess_model:.4f} "
            f"excess_inertia={result.excess_inertia:.4f} weeks={result.n_weeks} "
            f"episodes={result.n_episodes}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````

- [ ] **Paso 4: Correr la prueba y ver que pasa**

Correr: `.venv/Scripts/python -m pytest tests/test_cli.py -q`

Esperado: `3 passed`

- [ ] **Paso 5: Commit**

````bash
git add tests/test_cli.py src/termo/cli.py
git commit -m "feat: add command line for the core stages"
````

### Tarea 23: Calidad: linter, tipos y prueba de las pruebas

**Archivos:** ninguno nuevo.

- [ ] **Paso 1: Formato y linter**

Correr: `.venv/Scripts/python -m ruff format .` y luego `.venv/Scripts/python -m ruff check .`

Esperado: `All checks passed!`

- [ ] **Paso 2: Tipos**

Correr: `.venv/Scripts/python -m mypy`

Esperado: `Success: no issues found in 28 source files`

- [ ] **Paso 3: Suite completa**

Correr: `.venv/Scripts/python -m pytest -q`

Esperado: `140 passed` en cerca de un minuto.

- [ ] **Paso 4: Probar las pruebas con tres errores a mano**

Una suite que no detecta un error sembrado no protege nada. Para cada fila: hacer el cambio, correr la prueba indicada, confirmar que **falla**, y deshacer con `git checkout -- <archivo>`.

| Archivo | Cambiar | Por | Debe fallar |
|---|---|---|---|
| `src/termo/features/pipeline.py` | `raw_features(curve.loc[:train_end], pca)` | `raw_features(curve, pca)` | `tests/test_pipeline.py` |
| `src/termo/validation/walkforward.py` | `online.index > refit.cutoff` | `online.index >= refit.cutoff` | `tests/test_walkforward.py` |
| `src/termo/validation/bootstrap.py` | `range(1, n_rows)` | `range(26, n_rows - 26)` | `tests/test_bootstrap.py` |

Si alguna prueba **no** falla, detenerse y avisar: hay un hueco en la suite.

- [ ] **Paso 5: Commit (solo si el formateador cambió algo)**

````bash
git add -A
git commit -m "style: apply formatter"
````

---

### Tarea 24: Corrida real sobre datos de FRED

Esta es la primera vez que el código toca datos reales. El orden importa: primero el snapshot, luego el registro, luego la corrida. No se cambia ningún valor de `configs/core.yaml` entre pasos.

**Archivos:**
- Crear: `data/snapshots/<fecha>/` (7 CSV y `manifest.json`)
- Crear: `trials/trials.jsonl`, `trials/labels/*.csv`
- Crear: `reports/go_no_go.md`, `reports/go_no_go.json`

- [ ] **Paso 1: Snapshot**

Correr: `.venv/Scripts/python -m termo.cli snapshot`

Esperado: `[ok] snapshot written to data/snapshots/<fecha>`

Si aparece `DataValidationError`, detenerse y avisar al usuario con el mensaje completo. No se verificó el inicio de `DGS3`, `DGS5` ni `DGS10`; esta validación es la que lo confirma.

````bash
git add data/snapshots
git commit -m "data: add FRED snapshot"
````

- [ ] **Paso 2: Confirmar que el árbol está limpio**

Correr: `git status --porcelain`

Esperado: sin salida. La bitácora guarda el commit del código; si hay cambios sin commitear lo marca como `-dirty` y la corrida deja de ser reproducible.

- [ ] **Paso 3: Registrar todos los trials**

Correr: `.venv/Scripts/python -m termo.cli register --snapshot data/snapshots/<fecha>`

Esperado: `[ok] registered 30 trials; features: ...` (24 del jump model, 4 de K-means, la inercia y el registro de configuración).

Si la lista de variables tiene menos de 12, la regla de colinealidad eliminó alguna en la primera ventana. Es el comportamiento diseñado; anotarlo en el Paso 7.

````bash
git add trials/trials.jsonl
git commit -m "trials: register the core grid before any run"
````

- [ ] **Paso 4: Correr los trials**

Correr en segundo plano: `.venv/Scripts/python -m termo.cli run --snapshot data/snapshots/<fecha>`

Esperado: una línea `[ok] <trial> ...` por trial (29) y al final `[ok] all registered trials have a result`.

El tiempo no está medido; la estimación gruesa es de una a dos horas. Si se interrumpe, volver a correr el mismo comando: retoma donde quedó y nunca repite un trial.

Si un trial termina con error, queda registrado como `failed` y la corrida se detiene. No reintentar: detenerse y avisar al usuario. Un reintento es un trial nuevo y lo decide el usuario.

````bash
git add trials
git commit -m "trials: record core grid results"
````

- [ ] **Paso 5: Reporte**

Correr: `.venv/Scripts/python -m termo.cli report --snapshot data/snapshots/<fecha>`

Esperado: `[ok] verdict: go -> reports/go_no_go.md` o `[ok] verdict: no-go -> reports/go_no_go.md`

````bash
git add reports
git commit -m "report: core go/no-go verdict"
````

- [ ] **Paso 6: DETENERSE**

Mostrar `reports/go_no_go.md` al usuario. No ajustar nada, no agregar configuraciones, no volver a correr. Cualquier configuración nueva es un trial nuevo que el usuario debe decidir y registrar antes.

- [ ] **Paso 7: Anotar en los archivos de contexto**

Agregar una línea a `docs/context/results.md` con: veredicto, modelo final, valor de cada criterio, variables usadas y si FTIC discrepó. Actualizar `docs/context/todo.md` y `docs/context/sesion-log.md`.

````bash
git add docs/context
git commit -m "docs: log core go/no-go results"
````

---

### Tarea 25: Evaluación final en holdout — BLOQUEADA

**No ejecutar sin aprobación explícita del usuario en la conversación.** Solo aplica si el veredicto fue `go`. Corre una sola vez: la bitácora rechaza un segundo intento, y no hay forma de deshacerlo.

- [ ] **Paso 1: Pedir aprobación**

Preguntar al usuario si autoriza abrir el holdout para el modelo final de `reports/go_no_go.json`. Esperar un sí claro.

- [ ] **Paso 2: Correr**

Correr: `.venv/Scripts/python -m termo.cli final-holdout --snapshot data/snapshots/<fecha>`

Esperado: `[ok] holdout: excess_model=... excess_inertia=... weeks=... episodes=...` si pasa, o la misma línea con `[x]` si no pasa.

Con unas 104 semanas habrá pocos episodios; el resultado es ruidoso. Si no pasa, lleva a revisión, no a cancelar.

- [ ] **Paso 3: Commit y reporte al usuario**

````bash
git add trials reports
git commit -m "report: final holdout evaluation"
````

---

## Cobertura del spec

| Sección del spec | Tarea |
|---|---|
| §3 Datos: series, fuente, validación | 3 |
| §3 Snapshot con hash | 4 |
| §3 Guardia de holdout | 5, 21 |
| §4.1 PCA y convención de signo | 6 |
| §4.2 Velocidad, §4.3 Volatilidad | 7 |
| §4.4 Preproceso, §4.5 Arranque y colinealidad, §4.6 Reajuste | 8, 13 |
| §5 Motor y lectura en línea | 10, 13 |
| §5.1 Nombres de fase | 11, 13 |
| §5.2 Grilla | 2, 21 |
| §6 Bitácora | 17, 21 |
| §7 Selección y FTIC | 15, 19 |
| §8.1 Estabilidad | 13, 14 |
| §8.2 Separación | 9, 18 |
| §8.3 Independencia | 12 |
| §8.4 Duración | 9, 18 |
| §8.5 Líneas base | 10, 13, 21 |
| §8.6 PBO y N efectivo | 16 |
| §8.7 Criterios | 20, 21 |
| §8.8 Holdout | 18, 21, 25 |
| §9 Salidas | 20, 22 |
| §11 Pruebas del código | 2 a 23 |
| §12 Riesgo de compatibilidad de `jumpmodels` | 1 |
