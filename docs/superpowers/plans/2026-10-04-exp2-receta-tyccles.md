# Experimento 2 (receta TYCCLES) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the pre-registered experiment 2 of TERMO: TYCCLES-style rank features (139 causal ranks), two engines (jump model walk-forward and K-means frozen in 1997), the same tests and thresholds as spec 1, in a new trial log, and publish the per-family verdict.

**Architecture:** The spec 1 code stays; it is generalized at three seams. (1) The feature pipeline takes a *recipe* object (`PcaRecipe` = spec 1, `TycclesRecipe` = new) that computes raw causal features and the level change for the inertia baseline. (2) `CoreConfig` gains optional fields with defaults so `configs/core.yaml` keeps loading; `configs/exp2.yaml` turns the new behaviour on. (3) `evaluate_config` learns a *frozen* K-means family (one fit, no refits, stability = S2) and a per-feature jump penalty; registration, run and report handle the families separately and the report discloses every trial ever registered.

**Tech Stack:** Python 3.14, pandas 3, numpy 2, scikit-learn (KMeans), jumpmodels 0.1.1, pytest, ruff, mypy. Commands run from the repo root `C:\Proyectos\TERMO` with the project venv (`.venv\Scripts\python.exe -m pytest`).

**Spec:** [`docs/superpowers/specs/2026-10-04-exp2-receta-tyccles-design.md`](../specs/2026-10-04-exp2-receta-tyccles-design.md). Read §4 (features), §5 (engines), §7 (criteria per family) before touching the corresponding task.

**Rules that bind every task:**
- No look-ahead: every feature row `t` uses data up to `t`; every scaler and model uses the training window only.
- Console output ASCII only (`[ok]`, `[x]`, `->`).
- Do not touch `trials/trials.jsonl`, `trials/labels/`, `reports/go_no_go.*`, `data/snapshots/` (spec 1 artefacts, closed).
- The holdout (dates >= 2024-10-01) is never loaded outside `final-holdout`, which is not run in this plan.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- After each task: `.venv\Scripts\python.exe -m pytest -q` (all green), `.venv\Scripts\python.exe -m ruff check src tests`, `.venv\Scripts\python.exe -m mypy src`.

---

## File structure

| File | Responsibility |
|---|---|
| `src/termo/features/recipes.py` (new) | Protocols `Recipe` / `FittedRecipe`: what a feature recipe must offer. No imports from the pipeline. |
| `src/termo/features/pipeline.py` (modify) | `PcaRecipe` (spec 1 features wrapped as a recipe), `FittedPipeline` holds a fitted recipe, `fit_pipeline(..., recipe=)`. |
| `src/termo/features/tyccles.py` (new) | `causal_rank`, `curve_measures`, `TycclesRecipe` (139 named ranks, level proxy). |
| `src/termo/validation/walkforward.py` (modify) | `build_refits(..., recipe=)`; level change comes from the pipeline. |
| `src/termo/validation/stability.py` (modify) | `halves_ari(..., recipe=)`. |
| `src/termo/config.py` (modify) | `TycclesConfig`, new optional `CoreConfig` fields, loader. |
| `configs/exp2.yaml` (new) | The registered configuration of experiment 2. |
| `src/termo/dataset.py` (modify) | `recipe_for(config)`, collinearity switch, `frozen_refits`. |
| `src/termo/experiment.py` (modify) | `effective_penalty`, frozen evaluation, `jump_penalty_effective` in metrics, frozen holdout. |
| `src/termo/core.py` (modify) | `kmeans_frozen` family in register/run, hypotheses and prior-log counts in `setup`, report per family, disclosure. |
| `src/termo/report.py` (modify) | Markdown sections for families and disclosure. |
| `src/termo/cli.py` (modify) | `--trials-dir`, `--reports-dir`; final holdout for any family. |
| `tests/conftest.py` (modify) | `make_tyccles_config`, fixtures `tyccles_config`, `tyccles_data`. |
| `tests/test_recipes.py`, `tests/test_tyccles.py`, `tests/test_core_exp2.py` (new); `tests/test_pipeline.py`, `tests/test_config.py`, `tests/test_dataset.py`, `tests/test_experiment.py`, `tests/test_cli.py` (modify) | Tests. |

---

### Task 1: Feature recipe seam (PCA recipe = spec 1, unchanged numbers)

**Files:**
- Create: `src/termo/features/recipes.py`
- Modify: `src/termo/features/pipeline.py`
- Modify: `src/termo/validation/walkforward.py:10,52-69`
- Modify: `src/termo/validation/stability.py:15-34`
- Modify: `tests/test_pipeline.py:66,80,85`
- Create: `tests/test_recipes.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_recipes.py`:

```python
"""The pipeline computes raw features through a recipe; spec 1 is the PCA recipe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from conftest import TENORS
from termo.config import CoreConfig
from termo.features.pca import CurvePCA
from termo.features.pipeline import (
    FEATURE_NAMES,
    LEVEL_CHANGE,
    PCA_RECIPE,
    fit_pipeline,
    raw_features,
)
from termo.features.velocity import smoothed_change
from termo.regime.model import kmeans_fitter
from termo.validation.stability import halves_ari
from termo.validation.walkforward import build_refits

BURN_IN = 252


@dataclass(frozen=True)
class MeanRecipe:
    """Two features from the mean yield: its level in bp and its 5-day change. Stateless."""

    @property
    def names(self) -> tuple[str, ...]:
        return ("mean", "dmean5")

    def fit(self, window: pd.DataFrame) -> MeanRecipe:
        return self

    def raw(self, curve: pd.DataFrame) -> pd.DataFrame:
        mean = curve[list(TENORS)].mean(axis=1) * 100.0
        return pd.DataFrame({"mean": mean, "dmean5": mean.diff(5)})

    def level_change(self, curve: pd.DataFrame) -> pd.Series:
        return smoothed_change(curve[list(TENORS)].mean(axis=1) * 100.0, 63, 10)


def test_pca_recipe_reproduces_the_spec_1_features(curve: pd.DataFrame) -> None:
    train_end = curve.index[999]
    pipeline = fit_pipeline(curve, train_end, FEATURE_NAMES, BURN_IN)
    expected = raw_features(curve, CurvePCA.fit(curve.loc[:train_end])).iloc[BURN_IN:]
    pd.testing.assert_frame_equal(pipeline.raw(curve), expected)
    pd.testing.assert_series_equal(pipeline.level_change(curve), expected[LEVEL_CHANGE])
    assert PCA_RECIPE.names == FEATURE_NAMES


def test_the_pipeline_uses_the_recipe_it_is_given(curve: pd.DataFrame) -> None:
    train_end = curve.index[999]
    recipe = MeanRecipe()
    pipeline = fit_pipeline(curve, train_end, recipe.names, 80, recipe=recipe)
    features = pipeline.transform(curve)
    assert tuple(features.columns) == ("mean", "dmean5")
    assert features.index[0] == curve.index[80]
    assert np.isfinite(features.to_numpy()).all()
    # standardized on the training window only (tolerance covers ddof and the +/-3 sigma clip)
    train = features.loc[:train_end]
    assert np.allclose(train.mean(), 0.0, atol=1e-6)
    assert np.allclose(train.std(ddof=0), 1.0, atol=0.01)


def test_refits_and_halves_take_the_recipe(curve: pd.DataFrame, config: CoreConfig) -> None:
    recipe = MeanRecipe()
    cutoffs = [curve.index[999], curve.index[1299]]
    refits = build_refits(curve, cutoffs, recipe.names, 80, recipe=recipe)
    assert [tuple(r.features.columns) for r in refits] == [("mean", "dmean5")] * 2
    expected = recipe.level_change(curve.loc[: refits[0].block_end]).iloc[80:]
    pd.testing.assert_series_equal(refits[0].level_change, expected)
    value = halves_ari(curve, recipe.names, 80, kmeans_fitter(2), recipe=recipe)
    assert -0.5 <= value <= 1.0
```

In `tests/test_pipeline.py` replace the three uses of `.pca` on a fitted pipeline:

```python
    assert np.array_equal(original.recipe.pca.loadings, refit.recipe.pca.loadings)
```
```python
    assert not np.allclose(late.recipe.pca.mean_level, early.recipe.pca.mean_level)
```
```python
    raw = raw_features(curve, pipeline.recipe.pca)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_recipes.py tests/test_pipeline.py -q`
Expected: FAIL with `ImportError: cannot import name 'PCA_RECIPE'` (and `AttributeError: 'FittedPipeline' object has no attribute 'recipe'`).

- [ ] **Step 3: Create the protocols**

`src/termo/features/recipes.py`:

```python
"""What a feature recipe must offer. The recipe computes raw features; the pipeline scales them."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class FittedRecipe(Protocol):
    """Raw features with whatever the recipe estimates fixed on a training window."""

    @property
    def names(self) -> tuple[str, ...]: ...

    def raw(self, curve: pd.DataFrame) -> pd.DataFrame:
        """Every raw feature for every row of `curve`. Row t uses data up to t only."""
        ...

    def level_change(self, curve: pd.DataFrame) -> pd.Series:
        """Smoothed 63-day change of the level, same rows. The inertia baseline reads its sign."""
        ...


class Recipe(Protocol):
    @property
    def names(self) -> tuple[str, ...]: ...

    def fit(self, window: pd.DataFrame) -> FittedRecipe:
        """Fix what the recipe estimates (PCA loadings, nothing for ranks) on `window` only."""
        ...
```

- [ ] **Step 4: Wrap spec 1 as the PCA recipe and route the pipeline through recipes**

`src/termo/features/pipeline.py` — replace everything from `@dataclass(frozen=True, eq=False) class FittedPipeline` to the end of `fit_pipeline` with:

```python
@dataclass(frozen=True, eq=False)
class FittedPcaRecipe:
    """Spec 1: PCA scores, velocities and log volatilities, loadings fixed on the window."""

    pca: CurvePCA

    @property
    def names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    def raw(self, curve: pd.DataFrame) -> pd.DataFrame:
        return raw_features(curve, self.pca)

    def level_change(self, curve: pd.DataFrame) -> pd.Series:
        return raw_features(curve, self.pca)[LEVEL_CHANGE]


@dataclass(frozen=True)
class PcaRecipe:
    @property
    def names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    def fit(self, window: pd.DataFrame) -> FittedPcaRecipe:
        return FittedPcaRecipe(pca=CurvePCA.fit(window))


PCA_RECIPE: Recipe = PcaRecipe()


@dataclass(frozen=True, eq=False)
class FittedPipeline:
    recipe: FittedRecipe
    columns: tuple[str, ...]
    burn_in: int
    clipper: DataClipperStd
    scaler: StandardScalerPD

    def raw(self, curve: pd.DataFrame) -> pd.DataFrame:
        """All raw features after the burn-in. `curve` must start at the sample start."""
        raw = self.recipe.raw(curve).iloc[self.burn_in :]
        if not np.isfinite(raw.to_numpy()).all():
            raise ValueError("non-finite feature values after the burn-in")
        return raw

    def level_change(self, curve: pd.DataFrame) -> pd.Series:
        """Raw smoothed 63-day change of the level after the burn-in, for the inertia baseline."""
        return self.recipe.level_change(curve).iloc[self.burn_in :]

    def transform(self, curve: pd.DataFrame) -> pd.DataFrame:
        selected = self.raw(curve)[list(self.columns)]
        return self.scaler.transform(self.clipper.transform(selected))


def fit_pipeline(
    curve: pd.DataFrame,
    train_end: pd.Timestamp,
    columns: Sequence[str],
    burn_in: int,
    train_start: pd.Timestamp | None = None,
    recipe: Recipe = PCA_RECIPE,
) -> FittedPipeline:
    """Fit the recipe, clipping bounds and z-score on the training window only."""
    window = curve.loc[:train_end] if train_start is None else curve.loc[train_start:train_end]
    fitted = recipe.fit(window)
    raw = fitted.raw(curve.loc[:train_end]).iloc[burn_in:]
    train = raw.loc[window.index[0] :, list(columns)]
    if train.empty:
        raise ValueError("training window is empty after the burn-in")
    clipper = DataClipperStd(mul=CLIP_STD).fit(train)
    scaler = StandardScalerPD().fit(clipper.transform(train))
    return FittedPipeline(
        recipe=fitted, columns=tuple(columns), burn_in=burn_in, clipper=clipper, scaler=scaler
    )
```

Add the import near the other `termo.features` imports:

```python
from termo.features.recipes import FittedRecipe, Recipe
```

`src/termo/validation/walkforward.py` — change the import and `build_refits`:

```python
from termo.features.pipeline import PCA_RECIPE, fit_pipeline
from termo.features.recipes import Recipe
```

```python
def build_refits(
    curve: pd.DataFrame,
    cutoffs: Sequence[pd.Timestamp],
    columns: Sequence[str],
    burn_in: int,
    recipe: Recipe = PCA_RECIPE,
) -> list[RefitData]:
    """Fit the feature pipeline at every cutoff. Shared by all model configurations."""
    block_ends = [*cutoffs[1:], curve.index[-1]]
    refits: list[RefitData] = []
    for cutoff, block_end in zip(cutoffs, block_ends, strict=True):
        pipeline = fit_pipeline(curve, cutoff, columns, burn_in, recipe=recipe)
        visible = curve.loc[:block_end]
        refits.append(
            RefitData(
                cutoff=cutoff,
                block_end=block_end,
                features=pipeline.transform(visible),
                level_change=pipeline.level_change(visible),
            )
        )
    return refits
```

Update the `RefitData.level_change` comment to `# raw smoothed 63-day change of the level, same dates`.

`src/termo/validation/stability.py` — `halves_ari` takes the recipe:

```python
from termo.features.pipeline import PCA_RECIPE, fit_pipeline
from termo.features.recipes import Recipe
```

```python
def halves_ari(
    curve: pd.DataFrame,
    columns: Sequence[str],
    burn_in: int,
    fitter: RegimeFitter,
    recipe: Recipe = PCA_RECIPE,
) -> float:
    """S2: fit one model per half of the sample, let both label the whole sample, compare.

    Each half has its own recipe fit, clipping and scaling, so nothing is shared but the method.
    """
    dates = curve.index[burn_in:]
    middle = len(dates) // 2
    first_end, second_start, last = dates[middle - 1], dates[middle], dates[-1]
    windows = (
        (fit_pipeline(curve, first_end, columns, burn_in, recipe=recipe), dates[0], first_end),
        (
            fit_pipeline(curve, last, columns, burn_in, train_start=second_start, recipe=recipe),
            second_start,
            last,
        ),
    )
```
(the loop below the `windows` tuple is unchanged).

- [ ] **Step 5: Run the whole suite, lint, types**

Run: `.venv\Scripts\python.exe -m pytest -q && .venv\Scripts\python.exe -m ruff check src tests && .venv\Scripts\python.exe -m mypy src`
Expected: all tests pass (184 + 3 new), ruff and mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/termo/features/recipes.py src/termo/features/pipeline.py src/termo/validation/walkforward.py src/termo/validation/stability.py tests/test_recipes.py tests/test_pipeline.py
git commit -m "feat: feature recipe seam; spec 1 becomes the PCA recipe

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: The TYCCLES recipe (139 causal ranks)

**Files:**
- Create: `src/termo/features/tyccles.py`
- Create: `tests/test_tyccles.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_tyccles.py`:

```python
"""TYCCLES data recipe: smoothed changes over 1-9 months as causal ranks, curve, volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import TENORS
from termo.features.pipeline import fit_pipeline
from termo.features.tyccles import TycclesRecipe, causal_rank, curve_measures, smoothing_delta
from termo.features.velocity import smoothed_change

RECIPE = TycclesRecipe(
    tenors=TENORS,
    horizons=(21, 42, 63, 84, 126, 189),
    rank_windows=(126, 252),
    vol_window=21,
    vol_rank_window=252,
)


def test_causal_rank_by_hand() -> None:
    rank = causal_rank(pd.Series([3.0, 1.0, 2.0, 5.0, 4.0]), 3)
    assert rank.iloc[:2].isna().all()
    assert rank.iloc[2] == pytest.approx(2 / 3)  # 2 among (3, 1, 2)
    assert rank.iloc[3] == pytest.approx(1.0)  # 5 among (1, 2, 5)
    assert rank.iloc[4] == pytest.approx(2 / 3)  # 4 among (2, 5, 4)


def test_rank_ignores_the_future() -> None:
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(size=600))
    altered = series.copy()
    altered.iloc[400:] += 100.0
    pd.testing.assert_series_equal(
        causal_rank(series, 126).iloc[:400], causal_rank(altered, 126).iloc[:400]
    )


def test_smoothing_delta_follows_the_spec() -> None:
    assert [smoothing_delta(h) for h in (21, 42, 63, 84, 126, 189)] == [5, 10, 15, 21, 31, 47]


def test_one_hundred_thirty_nine_named_features(curve: pd.DataFrame) -> None:
    names = RECIPE.names
    assert len(names) == 139 and len(set(names)) == 139
    assert names[0] == "d1_21_r126" and names[-1] == "vol30_r252"
    assert "s3s10_63_r252" in names and "c5_189_r126" in names and "d30_189_r252" in names
    raw = RECIPE.raw(curve)
    assert tuple(raw.columns) == names
    assert RECIPE.burn_in_needed == 189 + 47 + 252 - 1
    assert raw.iloc[RECIPE.burn_in_needed :].notna().all().all()
    assert raw.iloc[RECIPE.burn_in_needed - 1].isna().any()
    finite = raw.dropna()
    assert ((finite > 0.0) & (finite <= 1.0)).all().all()


def test_features_are_the_ranked_smoothed_changes(curve: pd.DataFrame) -> None:
    raw = RECIPE.raw(curve)
    ten_year = curve["DGS10"] * 100.0
    expected = causal_rank(smoothed_change(ten_year, 63, 15), 252)
    pd.testing.assert_series_equal(raw["d10_63_r252"], expected, check_names=False)
    slope = (curve["DGS10"] - curve["DGS3"]) * 100.0
    expected = causal_rank(smoothed_change(slope, 21, 5), 126)
    pd.testing.assert_series_equal(raw["s3s10_21_r126"], expected, check_names=False)
    vol = (curve["DGS2"] * 100.0).diff().rolling(21).std()
    pd.testing.assert_series_equal(raw["vol2_r252"], causal_rank(vol, 252), check_names=False)


def test_curve_measures_have_the_right_sign() -> None:
    index = pd.bdate_range("2000-01-03", periods=3)
    flat = dict.fromkeys(TENORS, 5.0)
    steep = {**flat, "DGS1": 4.0, "DGS30": 6.0}
    humped = {**flat, "DGS5": 5.5}
    measures = curve_measures(pd.DataFrame([flat, steep, humped], index=index))
    assert measures.loc[index[0]].eq(0.0).all()
    assert measures.loc[index[1], "s12m5s"] == 1.0
    assert measures.loc[index[1], "s3s10"] == 0.0
    assert measures.loc[index[1], "s10s30"] == 1.0
    assert measures.loc[index[2], "c5"] == 1.0  # 2 * 5.5 - 5 - 5


def test_level_change_is_the_smoothed_change_of_the_mean_yield(curve: pd.DataFrame) -> None:
    expected = smoothed_change(curve[list(TENORS)].mean(axis=1) * 100.0, 63, 10)
    pd.testing.assert_series_equal(RECIPE.level_change(curve), expected)


def test_recipe_needs_the_tenors_of_the_curve_measures() -> None:
    with pytest.raises(ValueError, match="curve measures need"):
        TycclesRecipe(
            tenors=("DGS1", "DGS10"),
            horizons=(21,),
            rank_windows=(126,),
            vol_window=21,
            vol_rank_window=252,
        )


def test_the_pipeline_runs_the_recipe(curve: pd.DataFrame) -> None:
    pipeline = fit_pipeline(curve, curve.index[1199], RECIPE.names, 504, recipe=RECIPE)
    features = pipeline.transform(curve)
    assert features.shape[1] == 139
    assert features.index[0] == curve.index[504]
    assert np.isfinite(features.to_numpy()).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tyccles.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'termo.features.tyccles'`.

- [ ] **Step 3: Write the recipe**

`src/termo/features/tyccles.py`:

```python
"""The TYCCLES data recipe: changes over 1-9 months as causal ranks, curve shape, volatility.

HSBC (TYCCLES, 2026) feeds K-means with yield changes over 1m-9m horizons turned into
six-month and one-year ranks, curve slopes and curvature with their changes, and realised
volatility; every change is smoothed against base effects. Spec 2 §4 fixes our adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from termo.features.pca import BP_PER_PERCENT
from termo.features.velocity import smoothed_change

LEVEL_HORIZON, LEVEL_DELTA = 63, 10  # the inertia baseline, as in spec 1
SMOOTHING_DIVISOR = 4  # delta = horizon // 4: 21 -> 5, 63 -> 15, 189 -> 47
MIN_HORIZON = 2 * SMOOTHING_DIVISOR  # so that 0 < delta < horizon

# name -> (tenors added, tenors subtracted): slope = long - short, curvature = 2 * belly - wings
CURVE_MEASURES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "s12m5s": (("DGS5",), ("DGS1",)),
    "s3s10": (("DGS10",), ("DGS3",)),
    "s10s30": (("DGS30",), ("DGS10",)),
    "c5": (("DGS5", "DGS5"), ("DGS2", "DGS10")),
}


def short_tenor(series_id: str) -> str:
    """DGS10 -> 10."""
    return series_id.removeprefix("DGS")


def smoothing_delta(horizon: int) -> int:
    return horizon // SMOOTHING_DIVISOR


def causal_rank(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of today within the last `window` days, today included. Past only."""
    return series.rolling(window, min_periods=window).rank(pct=True)


def curve_measures(curve: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=curve.index)
    for name, (plus, minus) in CURVE_MEASURES.items():
        out[name] = sum(curve[t] for t in plus) - sum(curve[t] for t in minus)
    return out


@dataclass(frozen=True)
class TycclesRecipe:
    """Stateless: nothing is estimated, so the fitted recipe is the recipe itself."""

    tenors: tuple[str, ...]
    horizons: tuple[int, ...]
    rank_windows: tuple[int, ...]
    vol_window: int
    vol_rank_window: int

    def __post_init__(self) -> None:
        needed = {t for plus, minus in CURVE_MEASURES.values() for t in plus + minus}
        missing = sorted(needed - set(self.tenors))
        if missing:
            raise ValueError(f"curve measures need the tenors {missing}")
        if min(self.horizons) < MIN_HORIZON:
            raise ValueError(f"every horizon must be at least {MIN_HORIZON} days")
        if min(self.rank_windows) < 2 or self.vol_window < 2 or self.vol_rank_window < 2:
            raise ValueError("rank and volatility windows must be at least 2 days")

    @property
    def names(self) -> tuple[str, ...]:
        names: list[str] = []
        for tenor in self.tenors:
            names += self._change_names(f"d{short_tenor(tenor)}")
        for measure in CURVE_MEASURES:
            names += self._change_names(measure)
        names += [f"vol{short_tenor(tenor)}_r{self.vol_rank_window}" for tenor in self.tenors]
        return tuple(names)

    @property
    def burn_in_needed(self) -> int:
        """Rows before the first finite feature: the longest smoothed change, then the longest rank."""
        horizon = max(self.horizons)
        return horizon + smoothing_delta(horizon) + max(self.rank_windows) - 1

    def fit(self, window: pd.DataFrame) -> TycclesRecipe:
        return self

    def raw(self, curve: pd.DataFrame) -> pd.DataFrame:
        yields_bp = curve[list(self.tenors)] * BP_PER_PERCENT
        measures_bp = curve_measures(curve) * BP_PER_PERCENT
        columns: dict[str, pd.Series] = {}
        for tenor in self.tenors:
            columns.update(self._ranked_changes(f"d{short_tenor(tenor)}", yields_bp[tenor]))
        for measure in CURVE_MEASURES:
            columns.update(self._ranked_changes(measure, measures_bp[measure]))
        for tenor in self.tenors:
            vol = yields_bp[tenor].diff().rolling(self.vol_window, min_periods=self.vol_window).std()
            columns[f"vol{short_tenor(tenor)}_r{self.vol_rank_window}"] = causal_rank(
                vol, self.vol_rank_window
            )
        return pd.DataFrame(columns, index=curve.index)[list(self.names)]

    def level_change(self, curve: pd.DataFrame) -> pd.Series:
        """Level proxy: the mean of the tenors. Spec 1 used the PCA level, absent here."""
        level = curve[list(self.tenors)].mean(axis=1) * BP_PER_PERCENT
        return smoothed_change(level, LEVEL_HORIZON, LEVEL_DELTA)

    def _change_names(self, prefix: str) -> list[str]:
        return [f"{prefix}_{h}_r{w}" for h in self.horizons for w in self.rank_windows]

    def _ranked_changes(self, prefix: str, series: pd.Series) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for horizon in self.horizons:
            change = smoothed_change(series, horizon, smoothing_delta(horizon))
            for window in self.rank_windows:
                out[f"{prefix}_{horizon}_r{window}"] = causal_rank(change, window)
        return out
```

- [ ] **Step 4: Run the tests, lint, types**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tyccles.py -q && .venv\Scripts\python.exe -m ruff check src tests && .venv\Scripts\python.exe -m mypy src`
Expected: 9 passed; ruff and mypy clean. If mypy complains that `TycclesRecipe.fit` returns `TycclesRecipe` where `FittedRecipe` is expected, it is fine: the class satisfies the protocol structurally (names, raw, level_change).

- [ ] **Step 5: Commit**

```bash
git add src/termo/features/tyccles.py tests/test_tyccles.py
git commit -m "feat: TYCCLES recipe with 139 causal rank features

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Configuration for experiment 2

**Files:**
- Modify: `src/termo/config.py`
- Create: `configs/exp2.yaml`
- Modify: `tests/test_config.py`
- Modify: `tests/conftest.py` (add `make_tyccles_config` and two fixtures)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
EXP2_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "exp2.yaml"


def test_spec_1_configuration_loads_with_the_defaults_of_spec_2() -> None:
    config = load_config(REPO_CONFIG)
    assert config.feature_set == "pca" and config.tyccles is None
    assert config.apply_collinearity_rule is True
    assert config.jump_penalty_per_feature is False
    assert config.frozen_train_end is None and config.prior_trial_logs == ()


def test_loads_the_experiment_2_configuration() -> None:
    config = load_config(EXP2_CONFIG)
    assert config.feature_set == "tyccles" and config.tyccles is not None
    assert config.tyccles.change_horizons_days == (21, 42, 63, 84, 126, 189)
    assert config.tyccles.rank_windows_days == (126, 252)
    assert config.tyccles.vol_window_days == 21 and config.tyccles.vol_rank_window_days == 252
    assert config.burn_in_days == 504
    assert config.apply_collinearity_rule is False
    assert config.jump_penalty_per_feature is True
    assert config.jump_penalties == (0.5, 1.2, 3.0, 8.0, 20.0, 50.0)
    assert config.frozen_train_end == date(1997, 12, 31)
    assert config.prior_trial_logs == ("trials/trials.jsonl",)
    # everything else is spec 1, untouched
    base = load_config(REPO_CONFIG)
    assert (config.series, config.start, config.holdout_start) == (
        base.series,
        base.start,
        base.holdout_start,
    )
    assert config.first_train_end == base.first_train_end
    assert config.k_values == base.k_values and config.thresholds == base.thresholds
    assert config.bootstrap == base.bootstrap and config.ftic == base.ftic
    assert (config.horizon_short_days, config.horizon_long_days) == (20, 65)


def test_tyccles_parameters_go_with_the_tyccles_feature_set(config: CoreConfig) -> None:
    with pytest.raises(ValueError, match="tyccles"):
        replace(config, feature_set="tyccles")
    with pytest.raises(ValueError, match="feature_set"):
        replace(config, feature_set="wavelets")


def test_frozen_date_must_lie_inside_the_pre_holdout_sample(config: CoreConfig) -> None:
    with pytest.raises(ValueError, match="frozen_train_end"):
        replace(config, frozen_train_end=config.holdout_start)
```

Append to `tests/conftest.py` (after `make_config`; add `from dataclasses import replace` and `TycclesConfig` to the imports at the top):

```python
def make_tyccles_config() -> CoreConfig:
    """Spec 2 on the synthetic curve: ranks, no collinearity rule, a frozen K-means in 1995."""
    return replace(
        make_config(),
        burn_in_days=504,
        jump_penalties=(0.5, 3.0),
        feature_set="tyccles",
        tyccles=TycclesConfig(
            change_horizons_days=(21, 42, 63, 84, 126, 189),
            rank_windows_days=(126, 252),
            vol_window_days=21,
            vol_rank_window_days=252,
        ),
        apply_collinearity_rule=False,
        jump_penalty_per_feature=True,
        frozen_train_end=date(1995, 12, 31),
    )


@pytest.fixture(scope="session")
def tyccles_config() -> CoreConfig:
    return make_tyccles_config()


@pytest.fixture(scope="session")
def tyccles_data(pre_holdout: pd.DataFrame, tyccles_config: CoreConfig) -> ExperimentData:
    from termo.dataset import prepare, recipe_for

    return prepare(pre_holdout, tyccles_config, recipe_for(tyccles_config).names)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -q`
Expected: FAIL with `ImportError: cannot import name 'TycclesConfig'` (conftest) — fix the import error first by adding the class, then the remaining tests fail on missing attributes / missing `configs/exp2.yaml`.

- [ ] **Step 3: Extend the configuration**

`src/termo/config.py` — add after `FticConfig`:

```python
FEATURE_SETS = ("pca", "tyccles")


@dataclass(frozen=True)
class TycclesConfig:
    change_horizons_days: tuple[int, ...]
    rank_windows_days: tuple[int, ...]
    vol_window_days: int
    vol_rank_window_days: int
```

In `CoreConfig`, after `ftic: FticConfig`, add the optional fields (defaults keep `configs/core.yaml` loading):

```python
    feature_set: str = "pca"
    tyccles: TycclesConfig | None = None
    apply_collinearity_rule: bool = True
    jump_penalty_per_feature: bool = False  # lambda = value * number of features
    frozen_train_end: date | None = None  # K-means fitted once through this date, never refitted
    prior_trial_logs: tuple[str, ...] = ()  # earlier registries the report must count
```

Append to `__post_init__`:

```python
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
        if (self.feature_set == "tyccles") != (self.tyccles is not None):
            raise ValueError("the tyccles feature set needs its parameters, and only that set")
        if self.frozen_train_end is not None and not (
            self.start < self.frozen_train_end < self.holdout_start
        ):
            raise ValueError("frozen_train_end must satisfy start < frozen_train_end < holdout_start")
```

In `load_config`, before `return CoreConfig(`:

```python
    tyccles = raw.get("tyccles")
```

and add to the constructor call:

```python
        feature_set=str(raw.get("feature_set", "pca")),
        tyccles=None
        if tyccles is None
        else TycclesConfig(
            change_horizons_days=tuple(int(h) for h in tyccles["change_horizons_days"]),
            rank_windows_days=tuple(int(w) for w in tyccles["rank_windows_days"]),
            vol_window_days=int(tyccles["vol_window_days"]),
            vol_rank_window_days=int(tyccles["vol_rank_window_days"]),
        ),
        apply_collinearity_rule=bool(raw.get("apply_collinearity_rule", True)),
        jump_penalty_per_feature=bool(raw.get("jump_penalty_per_feature", False)),
        frozen_train_end=raw.get("frozen_train_end"),
        prior_trial_logs=tuple(str(p) for p in raw.get("prior_trial_logs", ())),
```

`configs/exp2.yaml`:

```yaml
# TERMO spec 2 (experiment 2: TYCCLES data recipe). Every value here is pre-registered:
# changing one after a run is a new trial, not an edit. Everything not listed under
# "spec 2" is identical to configs/core.yaml.
series: [DGS1, DGS2, DGS3, DGS5, DGS7, DGS10, DGS30]
start: 1977-02-15
holdout_start: 2024-10-01
first_train_end: 1987-12-31
refit_weeks: 26

# --- spec 2 ---
feature_set: tyccles
tyccles:
  change_horizons_days: [21, 42, 63, 84, 126, 189]
  rank_windows_days: [126, 252]
  vol_window_days: 21
  vol_rank_window_days: 252
burn_in_days: 504             # longest smoothed change (189 + 47) plus the 252-day rank
apply_collinearity_rule: false  # HSBC fed all 102 correlated inputs; we feed all 139
jump_penalty_per_feature: true  # lambda = value * 139; with 10 features this is spec 1's grid
frozen_train_end: 1997-12-31  # K-means fitted once through here, read 1998-01 -> 2024-09
prior_trial_logs: [trials/trials.jsonl]  # counted in the report: 30 trials, verdict no-go
# --- end spec 2 ---

grid:
  k: [2, 3, 4, 5]
  jump_penalty: [0.5, 1.2, 3, 8, 20, 50]

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
```

- [ ] **Step 4: Run the tests, lint, types**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -q && .venv\Scripts\python.exe -m ruff check src tests && .venv\Scripts\python.exe -m mypy src`
Expected: all pass (`tyccles_data` is not used yet; `recipe_for` arrives in Task 4). ruff and mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/termo/config.py configs/exp2.yaml tests/test_config.py tests/conftest.py
git commit -m "feat: experiment 2 configuration with spec 1 defaults

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Dataset: recipe from config, collinearity switch, frozen refit

**Files:**
- Modify: `src/termo/dataset.py`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dataset.py` (add `from termo.dataset import recipe_for` and `from termo.features.pipeline import PCA_RECIPE, fit_pipeline` to the imports; `CoreConfig` and `pd` are already imported there — check and add if not):

```python
def test_recipe_follows_the_feature_set(config: CoreConfig, tyccles_config: CoreConfig) -> None:
    assert recipe_for(config) is PCA_RECIPE
    recipe = recipe_for(tyccles_config)
    assert len(recipe.names) == 139
    with pytest.raises(ValueError, match="burn_in_days"):
        recipe_for(replace(tyccles_config, burn_in_days=400))


def test_collinearity_rule_can_be_switched_off(
    pre_holdout: pd.DataFrame, tyccles_config: CoreConfig
) -> None:
    columns = first_window_columns(pre_holdout, tyccles_config)
    assert columns == recipe_for(tyccles_config).names
    pruned = first_window_columns(pre_holdout, replace(tyccles_config, apply_collinearity_rule=True))
    assert 0 < len(pruned) < 139 and set(pruned) <= set(columns)


def test_frozen_refit_is_one_fit_on_the_past_until_the_frozen_date(
    tyccles_data: ExperimentData, pre_holdout: pd.DataFrame, tyccles_config: CoreConfig
) -> None:
    assert len(tyccles_data.frozen_refits) == 1
    (frozen,) = tyccles_data.frozen_refits
    assert frozen.cutoff <= pd.Timestamp(tyccles_config.frozen_train_end)
    assert frozen.block_end == pre_holdout.index[-1]
    assert frozen.features.index[-1] == pre_holdout.index[-1]
    assert tuple(frozen.features.columns) == tyccles_data.columns
    pipeline = fit_pipeline(
        pre_holdout,
        frozen.cutoff,
        tyccles_data.columns,
        tyccles_config.burn_in_days,
        recipe=recipe_for(tyccles_config),
    )
    pd.testing.assert_frame_equal(
        frozen.features.loc[: frozen.cutoff], pipeline.transform(pre_holdout.loc[: frozen.cutoff])
    )


def test_no_frozen_refit_without_a_frozen_date(data: ExperimentData) -> None:
    assert data.frozen_refits == ()
```

Add `from dataclasses import replace` and `import pytest` to the test module imports if missing.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_dataset.py -q`
Expected: FAIL with `ImportError: cannot import name 'recipe_for'`.

- [ ] **Step 3: Implement**

`src/termo/dataset.py` — full new content:

```python
"""Everything a model configuration needs, prepared once and shared by all of them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from termo.config import CoreConfig
from termo.features.pipeline import PCA_RECIPE, fit_pipeline, select_columns
from termo.features.recipes import Recipe
from termo.features.tyccles import TycclesRecipe
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
    frozen_refits: tuple[RefitData, ...] = ()  # one refit, never updated: the frozen K-means

    @property
    def yields_10y(self) -> pd.Series:
        return self.curve[TEN_YEAR]

    @property
    def daily_change_10y(self) -> pd.Series:
        return self.curve[TEN_YEAR].diff() * BP_PER_PERCENT


def recipe_for(config: CoreConfig) -> Recipe:
    """The feature recipe the configuration names, checked against its burn-in."""
    if config.feature_set != "tyccles":
        return PCA_RECIPE
    assert config.tyccles is not None  # CoreConfig validates the pair
    recipe = TycclesRecipe(
        tenors=config.series,
        horizons=config.tyccles.change_horizons_days,
        rank_windows=config.tyccles.rank_windows_days,
        vol_window=config.tyccles.vol_window_days,
        vol_rank_window=config.tyccles.vol_rank_window_days,
    )
    if config.burn_in_days < recipe.burn_in_needed:
        raise ValueError(
            f"burn_in_days must be at least {recipe.burn_in_needed} for these horizons and ranks"
        )
    return recipe


def first_window_columns(curve: pd.DataFrame, config: CoreConfig) -> tuple[str, ...]:
    """Apply the collinearity rule once, on the first training window, when the config says so."""
    recipe = recipe_for(config)
    if not config.apply_collinearity_rule:
        return recipe.names
    first_end = pd.Timestamp(config.first_train_end)
    pipeline = fit_pipeline(curve, first_end, recipe.names, config.burn_in_days, recipe=recipe)
    train = pipeline.transform(curve.loc[:first_end])
    return select_columns(train, config.thresholds.collinearity_max)


def frozen_cutoff(feature_dates: pd.DatetimeIndex, frozen_train_end: pd.Timestamp) -> pd.Timestamp:
    """Last feature date on or before the frozen date; there must be dates after it."""
    eligible = feature_dates[feature_dates <= frozen_train_end]
    if len(eligible) == 0 or eligible[-1] >= feature_dates[-1]:
        raise ValueError("frozen_train_end leaves no out-of-sample dates")
    return eligible[-1]


def prepare(curve: pd.DataFrame, config: CoreConfig, columns: Sequence[str]) -> ExperimentData:
    """Fit the feature pipeline at every refit date. `curve` must start at the sample start."""
    recipe = recipe_for(config)
    burn_in = config.burn_in_days
    feature_dates = curve.index[burn_in:]
    cutoffs = refit_cutoffs(feature_dates, pd.Timestamp(config.first_train_end), config.refit_weeks)
    refits = build_refits(curve, cutoffs, columns, burn_in, recipe=recipe)
    frozen: tuple[RefitData, ...] = ()
    if config.frozen_train_end is not None:
        cutoff = frozen_cutoff(feature_dates, pd.Timestamp(config.frozen_train_end))
        frozen = tuple(build_refits(curve, [cutoff], columns, burn_in, recipe=recipe))
    full = fit_pipeline(curve, curve.index[-1], columns, burn_in, recipe=recipe).transform(curve)
    return ExperimentData(
        config=config,
        curve=curve,
        columns=tuple(columns),
        refits=tuple(refits),
        full_features=full,
        frozen_refits=frozen,
    )
```

- [ ] **Step 4: Run the suite, lint, types**

Run: `.venv\Scripts\python.exe -m pytest -q && .venv\Scripts\python.exe -m ruff check src tests && .venv\Scripts\python.exe -m mypy src`
Expected: all pass. (`tyccles_data` now builds: ~139 columns, refits every 26 weeks from 1994-06-30 plus one frozen refit at 1995-12-29.)

- [ ] **Step 5: Commit**

```bash
git add src/termo/dataset.py tests/test_dataset.py
git commit -m "feat: dataset picks the recipe, optional collinearity rule, frozen refit

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Experiment: per-feature penalty and the frozen K-means family

**Files:**
- Modify: `src/termo/experiment.py`
- Modify: `tests/test_experiment.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_experiment.py` (add `from dataclasses import replace`, `from termo.dataset import recipe_for`, `from termo.experiment import effective_penalty` to the imports):

```python
@pytest.fixture(scope="module")
def frozen(tyccles_data: ExperimentData) -> ConfigEvaluation:
    return evaluate_config(tyccles_data, 2, None, frozen=True)


def test_penalty_per_feature_scales_with_the_number_of_columns(
    config: CoreConfig, tyccles_config: CoreConfig
) -> None:
    assert effective_penalty(config, 0.5, 139) == 0.5
    assert effective_penalty(tyccles_config, 0.5, 139) == pytest.approx(69.5)
    per_feature = replace(config, jump_penalty_per_feature=True)
    grid = (0.5, 1.2, 3.0, 8.0, 20.0, 50.0)
    assert [effective_penalty(per_feature, c, 10) for c in grid] == pytest.approx(
        [5.0, 12.0, 30.0, 80.0, 200.0, 500.0]
    )


def test_the_jump_model_sees_the_effective_penalty(
    tyccles_data: ExperimentData, jump: ConfigEvaluation
) -> None:
    evaluation = evaluate_config(tyccles_data, 2, 0.5)
    assert evaluation.jump_penalty_effective == pytest.approx(0.5 * 139)
    assert evaluation.metrics()["jump_penalty_effective"] == pytest.approx(69.5)
    assert jump.jump_penalty_effective == 50.0  # spec 1 configuration: absolute


def test_frozen_engine_has_no_s1_and_uses_s2(
    frozen: ConfigEvaluation, tyccles_data: ExperimentData
) -> None:
    assert frozen.frozen and frozen.jump_penalty is None
    assert frozen.s1_mean is None and frozen.s1_min is None
    assert frozen.stability == frozen.s2
    (refit,) = tyccles_data.frozen_refits
    assert frozen.oos_labels.index[0] > refit.cutoff
    assert frozen.oos_labels.index[-1] == tyccles_data.curve.index[-1]
    metrics = frozen.metrics()
    json.dumps(metrics)
    assert metrics["s1_mean"] is None and metrics["frozen"] is True


def test_frozen_labels_do_not_change_when_data_are_appended(
    pre_holdout: pd.DataFrame, tyccles_config: CoreConfig
) -> None:
    columns = recipe_for(tyccles_config).names
    shorter = prepare(pre_holdout.iloc[:-60], tyccles_config, columns)
    longer = prepare(pre_holdout, tyccles_config, columns)
    first = evaluate_config(shorter, 2, None, frozen=True).oos_labels
    second = evaluate_config(longer, 2, None, frozen=True).oos_labels
    pd.testing.assert_series_equal(first, second.loc[first.index])


def test_frozen_engine_refuses_a_penalty_or_a_missing_refit(
    tyccles_data: ExperimentData, data: ExperimentData
) -> None:
    with pytest.raises(ValueError, match="no jump penalty"):
        evaluate_config(tyccles_data, 2, 3.0, frozen=True)
    with pytest.raises(ValueError, match="frozen_train_end"):
        evaluate_config(data, 2, None, frozen=True)


def test_holdout_evaluation_runs_the_frozen_engine(
    curve: pd.DataFrame, tyccles_config: CoreConfig
) -> None:
    full = prepare(curve, tyccles_config, recipe_for(tyccles_config).names)
    result = holdout_evaluation(full, 2, None, frozen=True)
    holdout_days = int((curve.index >= pd.Timestamp(tyccles_config.holdout_start)).sum())
    assert 0 < result.n_weeks <= holdout_days // 5 + 1
    assert result.passed == (result.excess_model >= result.excess_inertia)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_experiment.py -q`
Expected: FAIL with `ImportError: cannot import name 'effective_penalty'`.

- [ ] **Step 3: Implement**

`src/termo/experiment.py` — changes:

Imports: add `from termo.dataset import ExperimentData, recipe_for` (replacing the existing `ExperimentData` import) and `kmeans_fitter` is already imported.

`ConfigEvaluation` becomes:

```python
@dataclass(frozen=True, eq=False)
class ConfigEvaluation:
    n_states: int
    jump_penalty: float | None  # None is K-means (the refit baseline or the frozen family)
    jump_penalty_effective: float | None  # lambda as the jump model saw it
    frozen: bool  # one K-means fit, never refitted: stability is S2 alone
    oos_labels: pd.Series
    s1_mean: float | None  # None for the frozen family: there are no refits
    s1_min: float | None
    s2: float
    stability: float
    eta_short: float  # raw eta-squared of the forward 10Y change, short horizon
    excess_short: float  # the same above its chance level: the separation measure
    excess_long: float
    excess_short_shift: float  # sensitivity check, never used to decide
    durations: dict[int, float]
    passes_duration: bool
    wcss: float
    jumps: int

    @property
    def score(self) -> float:
        return self.stability * self.excess_short

    def metrics(self) -> dict[str, Any]:
        return {
            "jump_penalty_effective": self.jump_penalty_effective,
            "frozen": self.frozen,
            "s1_mean": self.s1_mean,
            "s1_min": self.s1_min,
            "s2": self.s2,
            "stability": self.stability,
            "eta_short": self.eta_short,
            "excess_short": self.excess_short,
            "excess_long": self.excess_long,
            "excess_short_shift": self.excess_short_shift,
            "score": self.score,
            "durations": {str(state): value for state, value in self.durations.items()},
            "passes_duration": self.passes_duration,
            "wcss": self.wcss,
            "jumps": self.jumps,
        }
```

Add after `make_fitter`:

```python
def effective_penalty(config: CoreConfig, jump_penalty: float, n_features: int) -> float:
    """Lambda as the jump model sees it. Per feature, the registered value is multiplied by p."""
    return jump_penalty * n_features if config.jump_penalty_per_feature else jump_penalty


def _engine(
    data: ExperimentData, n_states: int, jump_penalty: float | None, frozen: bool
) -> tuple[RegimeFitter, tuple[RefitData, ...], float | None]:
    """Fitter, refits and effective penalty of one configuration."""
    if frozen:
        if jump_penalty is not None:
            raise ValueError("the frozen family is K-means: no jump penalty")
        if not data.frozen_refits:
            raise ValueError("no frozen refit: set frozen_train_end in the configuration")
        return kmeans_fitter(n_states), data.frozen_refits, None
    penalty = (
        None
        if jump_penalty is None
        else effective_penalty(data.config, jump_penalty, len(data.columns))
    )
    return make_fitter(n_states, penalty), data.refits, penalty
```

Add `from termo.validation.walkforward import RefitData, inertia_labels, run_walkforward` (extend the existing import).

Replace `evaluate_config`:

```python
def evaluate_config(
    data: ExperimentData, n_states: int, jump_penalty: float | None, frozen: bool = False
) -> ConfigEvaluation:
    config = data.config
    fitter, refits, penalty = _engine(data, n_states, jump_penalty, frozen)
    walk = run_walkforward(refits, fitter, n_states, data.daily_change_10y)
    s2 = halves_ari(
        data.curve, data.columns, config.burn_in_days, fitter, recipe=recipe_for(config)
    )
    durations = median_durations(walk.oos_labels.to_numpy(), n_states)
    minimum = config.thresholds.min_median_duration_days
    passes = all(not math.isnan(value) and value >= minimum for value in durations.values())
    fitted = fitter(data.full_features).insample_labels()
    short = separation(walk.oos_labels, data.yields_10y, config.horizon_short_days, config)
    long_ = separation(walk.oos_labels, data.yields_10y, config.horizon_long_days, config)
    if frozen:
        s1_mean: float | None = None
        s1_min: float | None = None
        stability = s2
    else:
        s1_mean = float(np.mean(walk.consecutive_ari))
        s1_min = float(np.min(walk.consecutive_ari))
        stability = stability_score(walk.consecutive_ari, s2)
    return ConfigEvaluation(
        n_states=n_states,
        jump_penalty=jump_penalty,
        jump_penalty_effective=penalty,
        frozen=frozen,
        oos_labels=walk.oos_labels,
        s1_mean=s1_mean,
        s1_min=s1_min,
        s2=s2,
        stability=stability,
        eta_short=short.raw,
        excess_short=short.excess,
        excess_long=long_.excess,
        excess_short_shift=short.excess_shift,
        durations=durations,
        passes_duration=passes,
        wcss=within_cluster_ss(data.full_features.to_numpy(), fitted),
        jumps=count_jumps(fitted),
    )
```

Replace `holdout_evaluation`:

```python
def holdout_evaluation(
    data: ExperimentData, n_states: int, jump_penalty: float | None, frozen: bool = False
) -> HoldoutResult:
    """`data` must come from a curve loaded with final_evaluation=True."""
    config = data.config
    start = pd.Timestamp(config.holdout_start)
    fitter, refits, _ = _engine(data, n_states, jump_penalty, frozen)
    walk = run_walkforward(refits, fitter, n_states, data.daily_change_10y)
    labels = walk.oos_labels.loc[start:]
    frame = separation_frame(labels, data.yields_10y, config.horizon_short_days)
    baseline = weekly_last(inertia_labels(refits)).reindex(frame.index)
    values = frame["change"].to_numpy()
    return HoldoutResult(
        excess_model=_excess(values, frame["label"].to_numpy().astype(int), config),
        excess_inertia=_excess(values, baseline.to_numpy().astype(int), config),
        n_weeks=len(frame),
        n_episodes=len(run_lengths(labels.to_numpy())),
    )
```

Note: `holdout_evaluation` previously used `jump_fitter` directly; `_engine` with `frozen=False` and a penalty gives the same model (via `make_fitter`). `jump_fitter` may now be unused in this module: remove it from the import if ruff flags it.

- [ ] **Step 4: Run the suite, lint, types**

Run: `.venv\Scripts\python.exe -m pytest -q && .venv\Scripts\python.exe -m ruff check src tests && .venv\Scripts\python.exe -m mypy src`
Expected: all pass. `test_evaluation_metrics_are_consistent` (existing) still passes: it reads `s1_mean`/`s1_min` of a jump evaluation, which are floats.

- [ ] **Step 5: Commit**

```bash
git add src/termo/experiment.py tests/test_experiment.py
git commit -m "feat: per-feature jump penalty and the frozen K-means family

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Register and run the frozen family, hypotheses and prior logs

**Files:**
- Modify: `src/termo/core.py:52-69,122-168,224-274`
- Create: `tests/test_core_exp2.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_core_exp2.py`:

```python
"""Experiment 2 end to end on the synthetic curve: two families, a prior registry, disclosure."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from conftest import fake_fred
from termo.config import CoreConfig
from termo.core import (
    HYPOTHESES,
    INERTIA_TRIAL,
    MODEL_KMEANS_FROZEN,
    SETUP_TRIAL,
    build_report,
    frozen_trial_id,
    jump_trial_id,
    kmeans_trial_id,
    prior_log_summary,
    publish_report,
    register_trials,
    registered_columns,
    run_trials,
    take_snapshot,
)
from termo.data.loader import load_curve
from termo.data.snapshot import snapshot_hash
from termo.dataset import prepare
from termo.report import CoreReport, Verdict, report_payload
from termo.validation.trials import TrialLog, TrialLogError

COMMIT = "test-commit-2"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("termo-exp2")


@pytest.fixture(scope="module")
def prior_log(workspace: Path) -> Path:
    """A closed earlier registry: two trials and a no-go report."""
    log = TrialLog(workspace / "prior" / "trials.jsonl")
    log.register(SETUP_TRIAL, "setup", {}, "old-snapshot", "old-code")
    log.register("jm_k2_lam5", "h", {"model": "jump", "n_states": 2, "jump_penalty": 5}, "s", "c")
    log.register("inertia", "h", {"model": "inertia"}, "s", "c")
    log.record_report(report_payload(CoreReport(Verdict.NO_GO, "jm_k2_lam5", criteria=())))
    return log.path


@pytest.fixture(scope="module")
def exp2_config(tyccles_config: CoreConfig, prior_log: Path) -> CoreConfig:
    return replace(tyccles_config, prior_trial_logs=(prior_log.as_posix(),))


@pytest.fixture(scope="module")
def snapshot_dir(workspace: Path, curve: pd.DataFrame, exp2_config: CoreConfig) -> Path:
    target = workspace / "data" / "snapshots" / "2026-10-02"
    take_snapshot(exp2_config, target, fake_fred(curve), "2026-10-02T00:00:00+00:00")
    return target


@pytest.fixture(scope="module")
def log(workspace: Path) -> TrialLog:
    return TrialLog(workspace / "trials" / "exp2" / "trials.jsonl")


@pytest.fixture(scope="module")
def finished(
    workspace: Path, snapshot_dir: Path, log: TrialLog, exp2_config: CoreConfig, prior_log: Path
) -> Path:
    """Register, run and report once for the module. Returns the reports directory."""
    before = hashlib.sha256(prior_log.read_bytes()).hexdigest()
    data_hash = snapshot_hash(snapshot_dir)
    curve = load_curve(snapshot_dir, exp2_config.series, exp2_config.start, exp2_config.holdout_start)
    register_trials(exp2_config, curve, log, data_hash, COMMIT)
    messages: list[str] = []
    data = prepare(curve, exp2_config, registered_columns(log))
    trials_dir = workspace / "trials" / "exp2"
    run_trials(data, log, trials_dir, data_hash, COMMIT, echo=messages.append)
    assert len(messages) == 9 and all(m.isascii() and m.startswith("[ok]") for m in messages)
    report = build_report(data, log, trials_dir, data_hash, COMMIT)
    publish_report(report, log, workspace / "reports" / "exp2", data_hash, COMMIT)
    assert hashlib.sha256(prior_log.read_bytes()).hexdigest() == before  # never written
    return workspace / "reports" / "exp2"


def test_both_families_are_registered_with_the_hypotheses(
    finished: Path, log: TrialLog, exp2_config: CoreConfig
) -> None:
    registrations = log.registrations()
    expected = {SETUP_TRIAL, INERTIA_TRIAL}
    for k in exp2_config.k_values:
        expected |= {jump_trial_id(k, lam) for lam in exp2_config.jump_penalties}
        expected |= {kmeans_trial_id(k), frozen_trial_id(k)}
    assert set(registrations) == expected and len(registrations) == 10
    assert registrations["kmf_k2"]["config"] == {"model": MODEL_KMEANS_FROZEN, "n_states": 2}
    setup = registrations[SETUP_TRIAL]["config"]
    assert setup["hypotheses"] == list(HYPOTHESES) and len(HYPOTHESES) == 3
    assert setup["prior_trial_logs"] == [
        {"path": exp2_config.prior_trial_logs[0], "n_trials": 2, "verdict": "no-go"}
    ]
    assert len(setup["columns"]) == 139


def test_prior_logs_are_counted_at_registration(tmp_path: Path, tyccles_config: CoreConfig) -> None:
    assert prior_log_summary(tyccles_config) == []
    missing = replace(tyccles_config, prior_trial_logs=((tmp_path / "nope.jsonl").as_posix(),))
    with pytest.raises(TrialLogError, match="prior trial log not found"):
        prior_log_summary(missing)


def test_frozen_results_carry_the_family_markers(finished: Path, log: TrialLog) -> None:
    results = log.results()
    frozen = results["kmf_k2"]["metrics"]
    assert frozen["frozen"] is True and frozen["s1_mean"] is None
    assert frozen["stability"] == frozen["s2"]
    assert results["kmf_k2"]["status"] in {"kept", "discarded"}
    jump = results["jm_k2_lam0.5"]["metrics"]
    assert jump["frozen"] is False and jump["jump_penalty_effective"] == pytest.approx(69.5)
    assert results[kmeans_trial_id(2)]["reason"] == "baseline"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_core_exp2.py -q`
Expected: FAIL with `ImportError: cannot import name 'HYPOTHESES'`.

- [ ] **Step 3: Implement registration and run**

`src/termo/core.py` — add constants after `MODEL_JUMP, MODEL_KMEANS, MODEL_INERTIA = ...`:

```python
MODEL_KMEANS_FROZEN = "kmeans_frozen"
FAMILIES = (MODEL_JUMP, MODEL_KMEANS_FROZEN)  # each has its own winner and criteria
HYPOTHESES = (
    "H1 (diagnostic, not a criterion): with rank features the refit K-means baselines pass "
    "the minimum median duration.",
    "H2: at least one jump model configuration passes every blocking criterion.",
    "H3: the frozen K-means passes every blocking criterion.",
)
TYCCLES_LIMITATIONS = (
    "Ranks of multi-month changes are smooth by construction: stability and duration can pass "
    "without any information about the future; separation and independence decide.",
    "139 correlated inputs without pruning: the distance is dominated by level changes.",
    "The 102-input TYCCLES recipe is reconstructed from the text, not reproduced.",
    "The frozen K-means reads 1998-2024 with centroids from 1979-1997; HSBC fitted on 50 years.",
    "PBO over four frozen configurations is nearly blind.",
    "Second look at the same pre-holdout data: every test family has had two chances.",
)
```

Add next to `kmeans_trial_id`:

```python
def frozen_trial_id(n_states: int) -> str:
    return f"kmf_k{n_states}"
```

Add before `register_trials`:

```python
def prior_log_summary(config: CoreConfig) -> list[dict[str, Any]]:
    """What earlier registries hold, counted now so that the report cannot forget them."""
    summary: list[dict[str, Any]] = []
    for path in config.prior_trial_logs:
        log = TrialLog(Path(path))
        if not log.path.exists():
            raise TrialLogError(f"prior trial log not found: {path}")
        trials = [t for t in log.registrations() if t != SETUP_TRIAL]
        report = log.last_report()
        summary.append(
            {
                "path": path,
                "n_trials": len(trials),
                "verdict": None if report is None else report["verdict"],
            }
        )
    return summary
```

In `register_trials`, the setup registration becomes:

```python
    log.register(
        SETUP_TRIAL,
        "Feature set fixed on the first training window; every parameter fixed in advance.",
        {
            "columns": list(columns),
            "config": config_fingerprint(config),
            "environment": environment_fingerprint(),
            "hypotheses": list(HYPOTHESES),
            "prior_trial_logs": prior_log_summary(config),
        },
        snapshot_hash,
        code_commit,
    )
```

and inside the `for n_states in config.k_values:` loop, after the `kmeans_trial_id` registration:

```python
        if config.frozen_train_end is not None:
            log.register(
                frozen_trial_id(n_states),
                f"K-means fitted once through {config.frozen_train_end.isoformat()} with "
                f"{n_states} states gives stable regimes that separate the forward 10Y move.",
                {"model": MODEL_KMEANS_FROZEN, "n_states": n_states},
                snapshot_hash,
                code_commit,
            )
```

In `run_trials`, replace the line `evaluation = evaluate_config(data, int(spec["n_states"]), spec.get("jump_penalty"))` with:

```python
        evaluation = evaluate_config(
            data,
            int(spec["n_states"]),
            spec.get("jump_penalty"),
            frozen=spec["model"] == MODEL_KMEANS_FROZEN,
        )
```

- [ ] **Step 4: Run the suite, lint, types**

Run: `.venv\Scripts\python.exe -m pytest -q && .venv\Scripts\python.exe -m ruff check src tests && .venv\Scripts\python.exe -m mypy src`
Expected: the three new tests pass; the existing `tests/test_core.py` still passes (spec 1 config has no frozen date and no prior logs, so its setup record gains `hypotheses` and an empty `prior_trial_logs` only; `test_registration_stores_the_whole_configuration` compares `setup["config"]["config"]`, which is unchanged). `build_report` is not yet family-aware: the `finished` fixture of the new module may fail at `build_report` if a `kmf_*` trial is not a jump candidate — Task 7 fixes it. If so, mark Task 6 done when the registration/run tests pass with `-k "registered or prior_logs"` and finish the module in Task 7.

- [ ] **Step 5: Commit**

```bash
git add src/termo/core.py tests/test_core_exp2.py
git commit -m "feat: register and run the frozen K-means family with hypotheses and prior logs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Report per family with disclosure

**Files:**
- Modify: `src/termo/core.py:276-439`
- Modify: `src/termo/selection.py:29-41`
- Modify: `src/termo/report.py:79-97`
- Modify: `tests/test_core_exp2.py` (append), `tests/test_report.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_core_exp2.py`:

```python
def test_the_report_judges_each_family_and_discloses_every_trial(
    finished: Path, log: TrialLog, exp2_config: CoreConfig
) -> None:
    payload = json.loads((finished / "go_no_go.json").read_text(encoding="utf-8"))
    details = payload["details"]
    families = details["families"]
    assert set(families) <= {"jump", "kmeans_frozen"} and families
    for family, outcome in families.items():
        assert outcome["verdict"] in {"go", "no-go"}
        assert [c["name"] for c in outcome["criteria"]] == [
            "stability",
            "separation_vs_inertia_low95",
            "independence_p",
            "pbo",
            "s2_halves",
        ]
        assert outcome["final_trial_id"].startswith("jm_" if family == "jump" else "kmf_")
    chosen = details["chosen_family"]
    assert chosen in families
    assert payload["final_trial_id"] == families[chosen]["final_trial_id"]
    assert payload["verdict"] == families[chosen]["verdict"]
    assert (payload["verdict"] == "go") == any(f["verdict"] == "go" for f in families.values())
    if "kmeans_frozen" in families:
        assert families["kmeans_frozen"]["final_model"]["s1_mean"] is None
        assert families["kmeans_frozen"]["ftic_states"] is None
    disclosure = details["disclosure"]
    assert disclosure["n_trials_this_log"] == 9
    assert disclosure["prior_logs"] == [
        {"path": exp2_config.prior_trial_logs[0], "n_trials": 2, "verdict": "no-go"}
    ]
    assert disclosure["n_trials_total"] == 11
    assert set(details["baselines_separation"]) == {"inertia", "kmeans_k2", "kmeans_k3"}
    assert any("reconstructed" in line for line in details["limitations"])
    markdown = (finished / "go_no_go.md").read_text(encoding="utf-8")
    assert markdown.isascii()
    assert "## Families" in markdown and "## Disclosure" in markdown
    assert "Total registered trials: 11" in markdown
    assert log.last_report() is not None and log.last_report()["verdict"] == payload["verdict"]
```

Append to `tests/test_report.py` (it already imports `CoreReport`, `render_markdown` or `write_report`; add `render_markdown` to the import if absent):

```python
def test_markdown_lists_families_and_disclosure_when_present() -> None:
    report = CoreReport(
        Verdict.NO_GO,
        "kmf_k3",
        criteria=(),
        details={
            "chosen_family": "kmeans_frozen",
            "families": {
                "jump": {"verdict": "no-go", "final_trial_id": "jm_k2_lam0.5", "gate": {"eta_difference_low": -0.01}},
                "kmeans_frozen": {"verdict": "no-go", "final_trial_id": "kmf_k3", "gate": {"eta_difference_low": -0.002}},
            },
            "disclosure": {
                "n_trials_this_log": 33,
                "prior_logs": [{"path": "trials/trials.jsonl", "n_trials": 30, "verdict": "no-go"}],
                "n_trials_total": 63,
                "note": "The pre-holdout data were already examined.",
            },
        },
    )
    text = render_markdown(report)
    assert "## Families" in text and "| kmeans_frozen | NO-GO | `kmf_k3` | -0.0020 | chosen |" in text
    assert "## Disclosure" in text
    assert "- Trials in this log: 33" in text
    assert "- trials/trials.jsonl: 30 trials, verdict no-go" in text
    assert "- Total registered trials: 63" in text
    assert text.isascii()
```

(`Verdict` must be imported in `tests/test_report.py`; add it if missing.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_core_exp2.py tests/test_report.py -q`
Expected: FAIL (`KeyError: 'families'`, `KeyError: 'jump_penalty'` for `kmf_*` candidates, missing "## Families").

- [ ] **Step 3: Candidates without a penalty**

`src/termo/selection.py` — `candidate_from_record` reads the penalty with a default (K-means is the jump model with penalty 0):

```python
        jump_penalty=float(config.get("jump_penalty", 0.0)),
```

- [ ] **Step 4: Markdown sections**

`src/termo/report.py` — in `render_markdown`, after the `if report.notes:` block and before the Details block, add:

```python
    families = report.details.get("families")
    if families:
        chosen = report.details.get("chosen_family")
        lines += [
            "",
            "## Families",
            "",
            "| Family | Verdict | Final model | Separation vs inertia (low 95%) | |",
            "|---|---|---|---|---|",
        ]
        for name, outcome in families.items():
            low = outcome["gate"]["eta_difference_low"]
            mark = "chosen" if name == chosen else ""
            lines.append(
                f"| {name} | {str(outcome['verdict']).upper()} | `{outcome['final_trial_id']}` "
                f"| {low:.4f} | {mark} |"
            )
    disclosure = report.details.get("disclosure")
    if disclosure:
        lines += ["", "## Disclosure", ""]
        lines.append(f"- Trials in this log: {disclosure['n_trials_this_log']}")
        for prior in disclosure["prior_logs"]:
            lines.append(
                f"- {prior['path']}: {prior['n_trials']} trials, verdict {prior['verdict']}"
            )
        lines.append(f"- Total registered trials: {disclosure['n_trials_total']}")
        lines.append(f"- {disclosure['note']}")
```

- [ ] **Step 5: Report per family in core**

`src/termo/core.py` — add the import `from dataclasses import asdict, dataclass` (extend the existing `asdict` import) and, before `build_report`, the family outcome:

```python
@dataclass(frozen=True)
class FamilyOutcome:
    family: str
    score_winner: Candidate
    final: Candidate
    criteria: tuple[Criterion, ...]
    notes: tuple[str, ...]
    gate: dict[str, float]
    ftic_states: int | None
    pbo: float
    n_candidates: int
    n_effective: int
    final_model: dict[str, Any]

    @property
    def verdict(self) -> Verdict:
        return decide(self.criteria)

    @property
    def separation_low(self) -> float:
        return self.gate["eta_difference_low"]

    def payload(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "score_winner": self.score_winner.trial_id,
            "final_trial_id": self.final.trial_id,
            "criteria": [asdict(c) for c in self.criteria],
            "notes": list(self.notes),
            "gate": self.gate,
            "ftic_states": self.ftic_states,
            "pbo": self.pbo,
            "n_trials": self.n_candidates,
            "n_effective": self.n_effective,
            "final_model": self.final_model,
        }


def _family_outcome(
    family: str,
    candidates: list[Candidate],
    data: ExperimentData,
    log: TrialLog,
    trials_dir: Path,
    inertia: pd.Series,
) -> FamilyOutcome | None:
    """Winner, gates, FTIC (jump family only), PBO and S2 of one family. None if nothing is eligible."""
    config = data.config
    results = log.results()
    winner = pick_winner(candidates)
    if winner is None:
        return None

    def gates(candidate: Candidate) -> tuple[tuple[Criterion, ...], dict[str, float]]:
        labels = logged_labels(log, trials_dir, candidate.trial_id)
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
    ftic_k: int | None = None
    if family == MODEL_JUMP:
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
        if alternative is not None and is_eligible(alternative):
            alt_criteria, alt_summary = gates(alternative)
            if decide(alt_criteria) is Verdict.GO:
                final, criteria, gate_summary = alternative, alt_criteria, alt_summary
                notes.append(
                    "The simpler model passes every blocking criterion: it is the final model."
                )

    labelings = [logged_labels(log, trials_dir, c.trial_id) for c in candidates]
    pbo = pbo_of(labelings, data.yields_10y, config)
    final_metrics = results[final.trial_id]["metrics"]
    s2 = float(final_metrics["s2"])
    thresholds = config.thresholds
    extra = (
        Criterion("pbo", pbo, f"<= {thresholds.pbo_max}", pbo <= thresholds.pbo_max, False),
        Criterion(
            "s2_halves", s2, f">= {thresholds.stability_min}", s2 >= thresholds.stability_min, False
        ),
    )
    if pbo > thresholds.pbo_max:
        notes.append("PBO above the limit: prune the grid and repeat as new trials.")
    if s2 < thresholds.stability_min <= final.stability:
        notes.append("S2 (halves) is below the threshold even though the average passes.")
    final_model = {
        "s1_mean": final_metrics["s1_mean"],
        "s1_min": final_metrics["s1_min"],
        "s2": s2,
        "separation_long": final_metrics["excess_long"],
        "median_duration_days": final_metrics["durations"],
        "days_by_decade": _days_by_decade(logged_labels(log, trials_dir, final.trial_id)),
    }
    return FamilyOutcome(
        family=family,
        score_winner=winner,
        final=final,
        criteria=criteria + extra,
        notes=tuple(notes),
        gate=gate_summary,
        ftic_states=ftic_k,
        pbo=pbo,
        n_candidates=len(candidates),
        n_effective=effective_n(
            [labels.to_numpy() for labels in labelings], config.effective_n_cut
        ),
        final_model=final_model,
    )


def disclosure(log: TrialLog) -> dict[str, Any]:
    """How many trials this project has registered, here and in every earlier registry."""
    prior = list(_setup(log)["config"].get("prior_trial_logs", []))
    here = len([t for t in log.registrations() if t != SETUP_TRIAL])
    return {
        "n_trials_this_log": here,
        "prior_logs": prior,
        "n_trials_total": here + sum(int(p["n_trials"]) for p in prior),
        "note": "The pre-holdout data were already examined by every prior log listed here. "
        "The holdout stays closed until a GO verdict and the user's explicit approval.",
    }
```

Then replace `build_report` entirely:

```python
def build_report(
    data: ExperimentData, log: TrialLog, trials_dir: Path, snapshot_hash: str, code_commit: str
) -> CoreReport:
    config = data.config
    verify_binding(log, config, snapshot_hash, code_commit)
    registrations, results = log.registrations(), log.results()
    trial_ids = [t for t in registrations if t != SETUP_TRIAL]
    pending = [t for t in trial_ids if t not in results]
    if pending:
        raise TrialLogError(f"trials without a result: {pending}")

    def is_model(trial_id: str, model: str) -> bool:
        return bool(registrations[trial_id]["config"].get("model") == model)

    def candidates_of(family: str) -> list[Candidate]:
        return [
            candidate_from_record(t, registrations[t]["config"], results[t]["metrics"])
            for t in trial_ids
            if is_model(t, family)
        ]

    families = {family: candidates_of(family) for family in FAMILIES}
    limitations = list(KNOWN_LIMITATIONS)
    if config.feature_set == "tyccles":
        limitations += list(TYCCLES_LIMITATIONS)
    details: dict[str, object] = {
        "columns": list(data.columns),
        "configurations": [
            {
                "trial_id": c.trial_id,
                "family": family,
                "stability": c.stability,
                "separation": c.separation,
                "separation_shift": results[c.trial_id]["metrics"]["excess_short_shift"],
                "score": c.score,
                "passes_duration": c.passes_duration,
            }
            for family, candidates in families.items()
            for c in candidates
        ],
        "baselines_separation": {
            t: results[t]["metrics"]["excess_short"]
            for t in trial_ids
            if is_model(t, MODEL_KMEANS) or is_model(t, MODEL_INERTIA)
        },
        "baselines_separation_shift": {
            t: results[t]["metrics"]["excess_short_shift"]
            for t in trial_ids
            if is_model(t, MODEL_KMEANS) or is_model(t, MODEL_INERTIA)
        },
        "limitations": limitations,
        "disclosure": disclosure(log),
    }

    if not any(families.values()):
        raise TrialLogError("no candidate trials in the log")
    inertia = logged_labels(log, trials_dir, INERTIA_TRIAL)
    outcomes: dict[str, FamilyOutcome] = {}
    for family, candidates in families.items():
        if candidates:
            outcome = _family_outcome(family, candidates, data, log, trials_dir, inertia)
            if outcome is not None:
                outcomes[family] = outcome
    details["families"] = {name: o.payload() for name, o in outcomes.items()}
    if not outcomes:
        return CoreReport(
            verdict=Verdict.NO_GO,
            final_trial_id=None,
            criteria=(),
            notes=(
                "No configuration passes the minimum median duration "
                "with positive stability and positive separation.",
            ),
            details=details,
        )

    passing = [o for o in outcomes.values() if o.verdict is Verdict.GO]
    chosen = max(passing or list(outcomes.values()), key=lambda o: o.separation_low)
    notes = list(chosen.notes)
    if len(outcomes) > 1:
        notes.append(
            f"Family {chosen.family} decides: "
            + ("it passes every blocking criterion" if passing else "no family passes")
            + "; the others are reported above."
        )
    details.update(
        {
            "chosen_family": chosen.family,
            "score_winner": chosen.score_winner.trial_id,
            "ftic_states": chosen.ftic_states,
            "gate": chosen.gate,
            "n_trials": chosen.n_candidates,
            "n_effective": chosen.n_effective,
            "final_model": chosen.final_model,
        }
    )
    return CoreReport(
        verdict=chosen.verdict,
        final_trial_id=chosen.final.trial_id,
        criteria=chosen.criteria,
        notes=tuple(notes),
        details=details,
    )
```

Keep the existing `publish_report` and `run_final_holdout` for now (Task 8 touches the holdout). Delete nothing else; `_days_by_decade` stays above.

- [ ] **Step 6: Run the suite, lint, types**

Run: `.venv\Scripts\python.exe -m pytest -q && .venv\Scripts\python.exe -m ruff check src tests && .venv\Scripts\python.exe -m mypy src`
Expected: all pass. In particular `tests/test_core.py::test_report_reaches_a_verdict_with_all_criteria` still passes: with the spec 1 config only the jump family exists, `details["n_trials"] == 4`, `final_model` keeps exactly its six keys, and `configurations` rows gained a `family` key, which the test does not forbid.

- [ ] **Step 7: Commit**

```bash
git add src/termo/core.py src/termo/selection.py src/termo/report.py tests/test_core_exp2.py tests/test_report.py
git commit -m "feat: verdict per family, chosen family decides, disclosure of every trial

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: CLI directories and the holdout for any family

**Files:**
- Modify: `src/termo/cli.py`
- Modify: `src/termo/core.py` (`run_final_holdout`)
- Modify: `tests/test_cli.py` (append), `tests/test_core_exp2.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
EXP2_SMALL_CONFIG = SMALL_CONFIG + """\
feature_set: tyccles
tyccles:
  change_horizons_days: [21, 42, 63, 84, 126, 189]
  rank_windows_days: [126, 252]
  vol_window_days: 21
  vol_rank_window_days: 252
apply_collinearity_rule: false
jump_penalty_per_feature: true
frozen_train_end: 1995-12-31
""".replace("burn_in_days: 252", "burn_in_days: 504").replace(
    "jump_penalty: [10, 50]", "jump_penalty: [0.5, 3]"
)


def test_trials_and_reports_directories_are_arguments(
    tmp_path: Path,
    curve: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "exp2.yaml"
    config_path.write_text(EXP2_SMALL_CONFIG, encoding="utf-8")
    monkeypatch.setattr("termo.cli.http_get", fake_fred(curve))
    monkeypatch.setattr("termo.cli.code_identity", lambda: "code-2")
    base = ["--config", str(config_path)]
    assert main(["snapshot", *base]) == 0
    (snapshot,) = (tmp_path / "data" / "snapshots").iterdir()
    stage = [
        *base,
        "--snapshot",
        str(snapshot),
        "--trials-dir",
        "trials/exp2",
        "--reports-dir",
        "reports/exp2",
    ]
    assert main(["register", *stage]) == 0
    assert (tmp_path / "trials" / "exp2" / "trials.jsonl").exists()
    assert not (tmp_path / "trials" / "trials.jsonl").exists()
    assert main(["run", *stage]) == 0
    assert (tmp_path / "trials" / "exp2" / "labels" / "kmf_k2.csv").exists()
    assert main(["report", *stage]) == 0
    out = capsys.readouterr().out
    assert out.isascii() and "reports/exp2/go_no_go.md" in out
    assert (tmp_path / "reports" / "exp2" / "go_no_go.json").exists()
```

Append to `tests/test_core_exp2.py`:

```python
def test_final_holdout_runs_the_frozen_family(
    finished: Path, tmp_path: Path, snapshot_dir: Path, log: TrialLog, exp2_config: CoreConfig
) -> None:
    copy = TrialLog(tmp_path / "trials.jsonl")
    copy.path.write_text(log.path.read_text(encoding="utf-8"), encoding="utf-8")
    copy.record_report(report_payload(CoreReport(Verdict.GO, "kmf_k2", criteria=())))
    result = run_final_holdout(exp2_config, snapshot_dir, copy, tmp_path / "reports", COMMIT)
    assert result.n_weeks > 0 and copy.records()[-1]["final_trial_id"] == "kmf_k2"
    with pytest.raises(TrialLogError, match="already been opened"):
        run_final_holdout(exp2_config, snapshot_dir, copy, tmp_path / "reports", COMMIT)
```

(add `run_final_holdout` to the `termo.core` import of that module.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py tests/test_core_exp2.py -q`
Expected: FAIL: `main` rejects `--trials-dir` (`argparse` error → `SystemExit`), and the holdout test fails with `TypeError: float() argument must be ... not 'NoneType'` (no `jump_penalty` for `kmf_k2`).

- [ ] **Step 3: Implement**

`src/termo/core.py` — in `run_final_holdout`, replace the `holdout_evaluation(...)` call:

```python
    result = holdout_evaluation(
        prepare(curve, config, columns),
        int(spec["n_states"]),
        None if spec.get("jump_penalty") is None else float(spec["jump_penalty"]),
        frozen=spec["model"] == MODEL_KMEANS_FROZEN,
    )
```

`src/termo/cli.py` — add the two arguments after `--snapshot`:

```python
    parser.add_argument(
        "--trials-dir",
        type=Path,
        default=TRIALS_DIR,
        help="trial log and labels directory (default: trials)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="report directory (default: reports)",
    )
```

and use them: `log = TrialLog(args.trials_dir / TRIALS_FILE)`; every `TRIALS_DIR` in the stages becomes `args.trials_dir` and every `REPORTS_DIR` becomes `args.reports_dir` (the `print` of the report path uses `args.reports_dir.as_posix()`). Update the parser description to `"TERMO: core go/no-go experiments"`.

- [ ] **Step 4: Run the suite, lint, types**

Run: `.venv\Scripts\python.exe -m pytest -q && .venv\Scripts\python.exe -m ruff check src tests && .venv\Scripts\python.exe -m mypy src`
Expected: all pass. The CLI test on the synthetic curve with 139 features and bootstrap 200 should take well under two minutes.

- [ ] **Step 5: Commit**

```bash
git add src/termo/cli.py src/termo/core.py tests/test_cli.py tests/test_core_exp2.py
git commit -m "feat: trials and reports directories as CLI arguments; holdout for any family

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Timing on synthetic data of the real size (no real data)

**Files:**
- Create: `<scratchpad>/timing_exp2.py` (outside the repo; nothing committed)

- [ ] **Step 1: Write the timing script**

```python
"""How long does one jump model fit take with 139 features and ~11,500 rows? Synthetic only."""

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Proyectos\TERMO\tests")
from conftest import make_curve  # noqa: E402

from termo.features.pipeline import fit_pipeline  # noqa: E402
from termo.features.tyccles import TycclesRecipe  # noqa: E402
from termo.regime.model import jump_fitter, kmeans_fitter  # noqa: E402

curve, _ = make_curve(n_days=12000, seed=1)
recipe = TycclesRecipe(("DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS30"),
                       (21, 42, 63, 84, 126, 189), (126, 252), 21, 252)
t0 = time.time()
features = fit_pipeline(curve, curve.index[-1], recipe.names, 504, recipe=recipe).transform(curve)
print(f"features {features.shape} in {time.time() - t0:.1f}s")
for k in (2, 5):
    for lam in (0.5 * 139, 50 * 139):
        t0 = time.time()
        jump_fitter(k, lam)(features)
        print(f"jump K={k} lambda={lam:g}: {time.time() - t0:.1f}s")
    t0 = time.time()
    kmeans_fitter(k)(features)
    print(f"kmeans K={k}: {time.time() - t0:.1f}s")
```

- [ ] **Step 2: Run it and extrapolate**

Run: `set PYTHONPATH=C:\Proyectos\TERMO\src && .venv\Scripts\python.exe <scratchpad>\timing_exp2.py`
Expected: prints seconds per fit. Estimate for the real run: 24 jump configurations x (72 walk-forward refits + 2 halves + 1 full fit) + 4 refit K-means x 75 + 4 frozen x 3, plus bootstrap (2000 draws x 2 horizons x 33 trials, cheap). Write the estimate in `docs/context/results.md`. If the estimate exceeds ~6 h, the run still goes ahead (spec §8), in the user's terminal tab with a monitor, as in spec 1.

---

### Task 10: Register experiment 2 (log before you look)

**Files:**
- Create: `trials/exp2/trials.jsonl` (by the CLI)

- [ ] **Step 1: Everything committed, suite green**

Run: `git status --porcelain -- src configs pyproject.toml` → empty. `.venv\Scripts\python.exe -m pytest -q` → all pass.

- [ ] **Step 2: Register**

Run from the repo root:

```bash
.venv/Scripts/python.exe -m termo.cli register --config configs/exp2.yaml --snapshot data/snapshots/2026-10-02 --trials-dir trials/exp2 --reports-dir reports/exp2
```

Expected: `[ok] registered 34 trials; features: d1_21_r126, ...` (139 names). Verify: `grep -c '"kind": "registered"' trials/exp2/trials.jsonl` → 34; `trials/trials.jsonl` unchanged (`git status` shows nothing under `trials/trials.jsonl`).

- [ ] **Step 3: Commit the registration**

```bash
git add trials/exp2/trials.jsonl
git commit -m "trials: register experiment 2 before any run

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Run and report

- [ ] **Step 1: Run** (in the user's terminal tab, monitored; resumable if interrupted)

```bash
.venv/Scripts/python.exe -m termo.cli run --config configs/exp2.yaml --snapshot data/snapshots/2026-10-02 --trials-dir trials/exp2 --reports-dir reports/exp2
```

Expected: 33 `[ok]` lines, then `[ok] all registered trials have a result`.

- [ ] **Step 2: Commit results**

```bash
git add trials/exp2
git commit -m "trials: record experiment 2 results

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 3: Report**

```bash
.venv/Scripts/python.exe -m termo.cli report --config configs/exp2.yaml --snapshot data/snapshots/2026-10-02 --trials-dir trials/exp2 --reports-dir reports/exp2
```

Expected: `[ok] verdict: go|no-go -> reports/exp2/go_no_go.md`. Check the markdown has `## Families` and `## Disclosure` with `Total registered trials: 63`.

- [ ] **Step 4: Commit the report and the context files**

Update `docs/context/results.md` (verdict line per family), `memory.md` (decisions: recipe seam, per-feature penalty, per-family verdict), `todo.md` (item → done / next decision), `sesion-log.md`.

```bash
git add reports/exp2 trials/exp2 docs/context
git commit -m "report: experiment 2 verdict

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

The holdout is **not** opened in this plan.

---

## Self-review

- **Spec coverage.** §3 data: same snapshot (Task 10 uses it). §4.2 features, smoothing, ranks, no collinearity, preprocessing, burn-in, inertia proxy, state naming: Tasks 2–4. §5.1 per-feature penalty: Tasks 3, 5. §5.2 frozen K-means, S2-only stability: Tasks 4–6. §5.3 baselines: existing + Task 6. §5.4 34 registrations: Task 6/10. §6 separate registry, disclosure, hypotheses: Tasks 6–8. §7 per-family winner, FTIC only for jump, same criteria, verdict rule: Task 7. §7.4 power: already in the spec. §8 timing: Task 9. §9 tests: each task. §10 limitations: `TYCCLES_LIMITATIONS` in Task 6, reported in Task 7.
- **Placeholders.** None: every test and implementation step carries its code.
- **Type consistency.** `Recipe.fit -> FittedRecipe`; `TycclesRecipe.fit -> TycclesRecipe` (structural match); `fit_pipeline(..., recipe=)`, `build_refits(..., recipe=)`, `halves_ari(..., recipe=)` all keyword with default `PCA_RECIPE`; `ExperimentData.frozen_refits: tuple[RefitData, ...] = ()`; `evaluate_config(data, n_states, jump_penalty, frozen=False)` and `holdout_evaluation(data, n_states, jump_penalty, frozen=False)`; `ConfigEvaluation.s1_mean/s1_min: float | None`; `candidate_from_record` default penalty 0.0; `FamilyOutcome.payload()` keys match the Task 7 test (`verdict`, `final_trial_id`, `criteria`, `gate`, `ftic_states`, `final_model`); `disclosure(log)` keys match the Task 7 tests and the Markdown renderer.
