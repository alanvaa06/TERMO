# TERMO descriptivo (spec 3) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 3-phase jump model of experiment 2 into a descriptive tool: an XGBoost surrogate gives a confidence and a SHAP explanation by blocks, six descriptive criteria (D1-D6) are registered and evaluated on pre-holdout data, then once on the holdout, and a `read` command produces the weekly reading with a fixed disclaimer.

**Architecture:** Everything from specs 1-2 is reused (rank recipe, pipeline, walk-forward, trial log, bindings, holdout lock). New, small modules: `surrogate/model.py` (fit, probabilities, walk-forward), `surrogate/explain.py` (SHAP per row, block sums), `descriptive/criteria.py` (pure D1-D6 checks), `descriptive/stages.py` (register / run / report / holdout for this spec), `reading.py` (lookup of hashed outputs into Markdown + JSON), `desc_cli.py`. The `run` stage writes labels, probabilities and SHAP for every out-of-sample day to hashed CSV files; `report`, `holdout` and `read` only read those files.

**Tech Stack:** Python 3.14, pandas 3, numpy 2.5, scikit-learn 1.9, jumpmodels 0.1.1, xgboost 3.4.1, shap 0.52.0 (both already installed in `.venv` and smoke-tested on synthetic data: deterministic multiclass fit, SHAP additive in log-odds with error ~3e-6, `shap_values` shape `(rows, features, classes)`).

**Spec:** [`docs/superpowers/specs/2026-10-05-termo-descriptivo-design.md`](../specs/2026-10-05-termo-descriptivo-design.md). Read §3 and §4 before Tasks 3-6.

**Rules that bind every task:**
- No look-ahead: a surrogate fitted at a cutoff sees rows up to the cutoff only; a reading for a date uses data up to that date only.
- Nothing about moves AFTER the reading date is ever computed or printed. The criteria are contemporaneous (changes over the PAST 63 days).
- Console output ASCII only (`[ok]`, `[x]`, `->`). Markdown reports ASCII only too (write "anos" not accented text in generated files).
- Do not touch `trials/trials.jsonl`, `trials/labels/`, `trials/exp2/`, `reports/go_no_go.*`, `reports/exp2/`, `data/snapshots/`.
- The holdout (dates >= 2024-10-01) is loaded only by the `holdout` stage, which this plan does NOT run.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- After each task: `.venv\Scripts\python.exe -m pytest -q` all green, `.venv\Scripts\python.exe -m ruff check src tests`, `.venv\Scripts\python.exe -m ruff format --check src tests`, `.venv\Scripts\python.exe -m mypy src`.

---

## File structure

| File | Responsibility |
|---|---|
| `pyproject.toml` (modify) | add `xgboost==3.4.1`, `shap==0.52.0` |
| `src/termo/core.py` (modify) | `BOUND_PACKAGES` gains xgboost and shap |
| `src/termo/validation/walkforward.py` (modify) | `WalkForwardResult.fits`: aligned in-sample and online labels of every refit |
| `src/termo/surrogate/__init__.py`, `model.py` (new) | `SurrogateParams`, `Surrogate`, `fit_surrogate`, `surrogate_walk` |
| `src/termo/surrogate/explain.py` (new) | `block_map`, `explain`, `block_sums`, `top_variables` |
| `src/termo/config.py` (modify) | `SurrogateConfig`, `DescriptiveConfig`, `CoreConfig.descriptive` |
| `configs/desc.yaml` (new) | the registered configuration of spec 3 |
| `src/termo/descriptive/__init__.py`, `criteria.py` (new) | `Check`, `evaluate`, `verdict`, `calibration` |
| `src/termo/descriptive/stages.py` (new) | `analyse`, `save_analysis`, `load_analysis`, `register_desc`, `run_desc`, `report_desc`, `holdout_desc` |
| `src/termo/reading.py` (new) | `build_reading`, `render_reading`, `write_reading` |
| `src/termo/desc_cli.py` (new) | stages `register`, `run`, `report`, `holdout`, `read` |
| `tests/test_smoke_surrogate.py`, `test_surrogate.py`, `test_explain.py`, `test_criteria.py`, `test_desc_stages.py`, `test_reading.py`, `test_desc_cli.py` (new); `tests/conftest.py`, `tests/test_walkforward.py`, `tests/test_config.py` (modify) | tests |

---

### Task 1: Dependencies bound and smoke-tested

**Files:**
- Modify: `pyproject.toml` (dependencies list)
- Modify: `src/termo/core.py` (`BOUND_PACKAGES`)
- Create: `tests/test_smoke_surrogate.py`

- [ ] **Step 1: Write the smoke test**

`tests/test_smoke_surrogate.py`:

```python
"""xgboost and shap behave as spec 3 assumes, in this environment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from termo.core import BOUND_PACKAGES, environment_fingerprint

PARAMS = {
    "n_estimators": 30,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": 0,
    "n_jobs": 1,
}


def _data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    features = pd.DataFrame(rng.normal(size=(600, 12)), columns=[f"f{i}" for i in range(12)])
    labels = (features["f0"] > 0).astype(int) + (features["f1"] > 1).astype(int)
    return features, labels.to_numpy()


def test_multiclass_fit_is_deterministic_and_gives_one_column_per_class() -> None:
    features, labels = _data()
    first = XGBClassifier(**PARAMS).fit(features, labels)
    second = XGBClassifier(**PARAMS).fit(features, labels)
    proba = first.predict_proba(features)
    assert proba.shape == (600, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.array_equal(proba, second.predict_proba(features))


def test_shap_values_are_additive_in_log_odds() -> None:
    features, labels = _data()
    model = XGBClassifier(**PARAMS).fit(features, labels)
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(features.iloc[:5]))
    assert values.shape == (5, 12, 3)  # rows, features, classes
    margin = model.predict(features.iloc[:5], output_margin=True)
    rebuilt = values.sum(axis=1) + np.asarray(explainer.expected_value)
    assert np.abs(rebuilt - margin).max() < 1e-3


def test_the_surrogate_libraries_are_part_of_the_bound_environment() -> None:
    assert {"xgboost", "shap"} <= set(BOUND_PACKAGES)
    assert {"xgboost", "shap"} <= set(environment_fingerprint())
```

- [ ] **Step 2: Run it to see the binding test fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_smoke_surrogate.py -q`
Expected: the first two tests pass (libraries are installed), the third FAILS (`xgboost` not in `BOUND_PACKAGES`).

- [ ] **Step 3: Bind the libraries**

`pyproject.toml` — the dependencies list becomes:

```toml
dependencies = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "jumpmodels==0.1.1",
    "pyyaml",
    "xgboost==3.4.1",
    "shap==0.52.0",
]
```

`src/termo/core.py`:

```python
BOUND_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "jumpmodels", "xgboost", "shap")
```

If mypy reports missing stubs for `xgboost` or `shap` in later tasks, add to `pyproject.toml` (create the section if absent, or extend the existing `[[tool.mypy.overrides]]`):

```toml
[[tool.mypy.overrides]]
module = ["xgboost.*", "shap.*"]
ignore_missing_imports = true
```

- [ ] **Step 4: Run the suite, lint, types**

Run: `.venv\Scripts\python.exe -m pytest -q && .venv\Scripts\python.exe -m ruff check src tests && .venv\Scripts\python.exe -m mypy src`
Expected: all pass (233 + 3).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/termo/core.py tests/test_smoke_surrogate.py
git commit -m "build: bind xgboost and shap; smoke tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: The walk-forward returns the labels of every refit

**Files:**
- Modify: `src/termo/validation/walkforward.py`
- Modify: `tests/test_walkforward.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_walkforward.py` (the module already imports `run_walkforward`, `ExperimentData`; add `from termo.regime.model import kmeans_fitter` if absent):

```python
def test_walkforward_returns_the_aligned_labels_of_every_refit(data: ExperimentData) -> None:
    walk = run_walkforward(data.refits, kmeans_fitter(2), 2, data.daily_change_10y)
    assert len(walk.fits) == len(data.refits)
    blocks = []
    for refit, fit in zip(data.refits, walk.fits, strict=True):
        assert fit.cutoff == refit.cutoff
        assert fit.insample.index.equals(refit.features.loc[: refit.cutoff].index)
        assert fit.online.index.equals(refit.features.index)
        assert set(fit.insample.unique()) <= {0, 1} and set(fit.online.unique()) <= {0, 1}
        blocks.append(fit.online.loc[fit.online.index > refit.cutoff])
    # the out-of-sample series is exactly the online labels after each cutoff, same names
    pd.testing.assert_series_equal(pd.concat(blocks), walk.oos_labels)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_walkforward.py -q`
Expected: FAIL with `AttributeError: 'WalkForwardResult' object has no attribute 'fits'`.

- [ ] **Step 3: Implement**

`src/termo/validation/walkforward.py` — add the dataclass before `WalkForwardResult` and extend the result:

```python
@dataclass(frozen=True, eq=False)
class RefitLabels:
    cutoff: pd.Timestamp
    insample: pd.Series  # labels of the training rows as fitted, with the stable names
    online: pd.Series  # online labels of every row this refit can see, with the stable names


@dataclass(frozen=True, eq=False)
class WalkForwardResult:
    oos_labels: pd.Series  # daily out-of-sample labels with stable names
    consecutive_ari: tuple[float, ...]  # one per pair of consecutive refits
    fits: tuple[RefitLabels, ...] = ()  # one per refit: what a surrogate learns from
```

In `run_walkforward`, add `fits: list[RefitLabels] = []` next to `blocks`, and after `online` is built inside the loop:

```python
        fits.append(RefitLabels(cutoff=refit.cutoff, insample=previous, online=online))
```

and return `WalkForwardResult(oos_labels=pd.concat(blocks), consecutive_ari=tuple(consecutive), fits=tuple(fits))`.

- [ ] **Step 4: Run the suite, lint, types** — all pass.

- [ ] **Step 5: Commit**

```bash
git add src/termo/validation/walkforward.py tests/test_walkforward.py
git commit -m "feat: walk-forward exposes the aligned labels of every refit

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: The surrogate

**Files:**
- Create: `src/termo/surrogate/__init__.py` (empty), `src/termo/surrogate/model.py`
- Create: `tests/test_surrogate.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_surrogate.py`:

```python
"""The surrogate imitates the regime labels from the same-day features, past only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score

from termo.dataset import ExperimentData
from termo.regime.model import kmeans_fitter
from termo.surrogate.model import SurrogateParams, fit_surrogate, surrogate_walk
from termo.validation.walkforward import run_walkforward

PARAMS = SurrogateParams(
    n_estimators=30, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, seed=0
)


def _toy(n: int = 500) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(1)
    index = pd.bdate_range("2000-01-03", periods=n)
    features = pd.DataFrame(rng.normal(size=(n, 6)), index=index, columns=list("abcdef"))
    labels = (features["a"] > 0).astype(int) + (features["b"] > 1).astype(int)
    return features, labels


def test_probabilities_have_one_column_per_phase_and_sum_to_one() -> None:
    features, labels = _toy()
    surrogate = fit_surrogate(features, labels, 3, PARAMS)
    proba = surrogate.proba(features)
    assert list(proba.columns) == [0, 1, 2] and proba.index.equals(features.index)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert balanced_accuracy_score(labels, proba.idxmax(axis=1)) > 0.9


def test_a_phase_missing_from_the_training_window_gets_probability_zero() -> None:
    features, labels = _toy()
    two = labels.clip(upper=1) * 2  # phases 0 and 2 only
    surrogate = fit_surrogate(features, two, 3, PARAMS)
    proba = surrogate.proba(features)
    assert surrogate.classes == (0, 2)
    assert (proba[1] == 0.0).all() and np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_one_phase_only_cannot_be_imitated() -> None:
    features, labels = _toy()
    with pytest.raises(ValueError, match="at least two phases"):
        fit_surrogate(features, labels * 0, 3, PARAMS)


def test_the_fit_is_deterministic() -> None:
    features, labels = _toy()
    first = fit_surrogate(features, labels, 3, PARAMS).proba(features)
    second = fit_surrogate(features, labels, 3, PARAMS).proba(features)
    pd.testing.assert_frame_equal(first, second)


def test_surrogate_walk_reads_each_block_with_the_fit_of_its_cutoff(data: ExperimentData) -> None:
    walk = run_walkforward(data.refits, kmeans_fitter(2), 2, data.daily_change_10y)
    result = surrogate_walk(data.refits, walk.fits, 2, PARAMS)
    assert result.proba.index.equals(walk.oos_labels.index)
    assert len(result.surrogates) == len(data.refits)
    # a memoryless clustering is easy to imitate from the same features
    assert balanced_accuracy_score(walk.oos_labels, result.proba.idxmax(axis=1)) > 0.9
    # no look-ahead: the first block equals a surrogate fitted by hand on the first window
    refit, fit = data.refits[0], walk.fits[0]
    by_hand = fit_surrogate(refit.features.loc[: refit.cutoff], fit.insample, 2, PARAMS)
    later = refit.features.loc[refit.features.index > refit.cutoff]
    pd.testing.assert_frame_equal(result.proba.loc[later.index], by_hand.proba(later))


def test_shuffled_labels_cannot_be_imitated() -> None:
    features, labels = _toy(800)
    rng = np.random.default_rng(2)
    shuffled = pd.Series(rng.permutation(labels.to_numpy()), index=labels.index)
    surrogate = fit_surrogate(features.iloc[:500], shuffled.iloc[:500], 3, PARAMS)
    guess = surrogate.proba(features.iloc[500:]).idxmax(axis=1)
    assert balanced_accuracy_score(shuffled.iloc[500:], guess) < 0.5
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_surrogate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'termo.surrogate'`.

- [ ] **Step 3: Implement**

`src/termo/surrogate/__init__.py`: empty file.

`src/termo/surrogate/model.py`:

```python
"""An XGBoost classifier that imitates the regime labels from the same-day features.

It gives what the jump model does not: a probability per phase. It never sees
yesterday's phase, so it disagrees with the jump model mostly around transitions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from termo.validation.walkforward import RefitData, RefitLabels


@dataclass(frozen=True)
class SurrogateParams:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    seed: int


@dataclass(frozen=True, eq=False)
class Surrogate:
    model: XGBClassifier
    classes: tuple[int, ...]  # phases present in the training window, in the model's order
    n_states: int
    columns: tuple[str, ...]

    def proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """One column per phase 0..n_states-1; a phase never seen in training gets 0."""
        raw = self.model.predict_proba(features[list(self.columns)])
        out = pd.DataFrame(0.0, index=features.index, columns=list(range(self.n_states)))
        out[list(self.classes)] = raw
        return out


def fit_surrogate(
    features: pd.DataFrame, labels: pd.Series, n_states: int, params: SurrogateParams
) -> Surrogate:
    """Fit on one training window. `labels` are the regime labels of exactly those rows."""
    if not features.index.equals(labels.index):
        raise ValueError("features and labels must cover the same rows")
    classes = tuple(sorted(int(c) for c in np.unique(labels.to_numpy())))
    if len(classes) < 2:
        raise ValueError("the surrogate needs at least two phases in the training window")
    encoded = np.searchsorted(np.asarray(classes), labels.to_numpy())
    model = XGBClassifier(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        tree_method="hist",
        random_state=params.seed,
        n_jobs=1,
    )
    model.fit(features, encoded)
    return Surrogate(
        model=model, classes=classes, n_states=n_states, columns=tuple(features.columns)
    )


@dataclass(frozen=True, eq=False)
class SurrogateWalk:
    proba: pd.DataFrame  # out-of-sample probabilities, one row per out-of-sample day
    surrogates: tuple[Surrogate, ...]  # one per refit, same order


def surrogate_walk(
    refits: Sequence[RefitData],
    fits: Sequence[RefitLabels],
    n_states: int,
    params: SurrogateParams,
) -> SurrogateWalk:
    """Refit the surrogate at every cutoff on the labels fitted there; read the next block."""
    blocks: list[pd.DataFrame] = []
    surrogates: list[Surrogate] = []
    for refit, fit in zip(refits, fits, strict=True):
        if fit.cutoff != refit.cutoff:
            raise ValueError("refits and fitted labels are out of step")
        train = refit.features.loc[: refit.cutoff]
        surrogate = fit_surrogate(train, fit.insample, n_states, params)
        later = refit.features.loc[refit.features.index > refit.cutoff]
        blocks.append(surrogate.proba(later))
        surrogates.append(surrogate)
    return SurrogateWalk(proba=pd.concat(blocks), surrogates=tuple(surrogates))
```

- [ ] **Step 4: Run the suite, lint, types** — all pass.

- [ ] **Step 5: Commit**

```bash
git add src/termo/surrogate tests/test_surrogate.py
git commit -m "feat: XGBoost surrogate with walk-forward probabilities

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: SHAP by blocks

**Files:**
- Create: `src/termo/surrogate/explain.py`
- Create: `tests/test_explain.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_explain.py`:

```python
"""SHAP of the surrogate for the phase of each day, summed in blocks of variables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from termo.surrogate.explain import block_map, block_sums, explain, top_variables
from termo.surrogate.model import SurrogateParams, fit_surrogate

PARAMS = SurrogateParams(
    n_estimators=30, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, seed=0
)
BLOCKS = (("left", ("a", "b")), ("right", ("c", "d")))
COLUMNS = ("a_1", "a_2", "b_1", "c_1", "d_1", "d_2")


def _fitted() -> tuple[pd.DataFrame, pd.Series, object]:
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2000-01-03", periods=400)
    features = pd.DataFrame(rng.normal(size=(400, 6)), index=index, columns=list(COLUMNS))
    labels = (features["a_1"] > 0).astype(int) + (features["c_1"] > 1).astype(int)
    return features, labels, fit_surrogate(features, labels, 3, PARAMS)


def test_every_variable_belongs_to_exactly_one_block() -> None:
    mapping = block_map(COLUMNS, BLOCKS)
    assert mapping == {
        "a_1": "left", "a_2": "left", "b_1": "left", "c_1": "right", "d_1": "right", "d_2": "right"
    }
    with pytest.raises(ValueError, match="no block"):
        block_map((*COLUMNS, "z_1"), BLOCKS)
    with pytest.raises(ValueError, match="more than one block"):
        block_map(COLUMNS, (*BLOCKS, ("again", ("a",))))


def test_values_plus_base_rebuild_the_margin_of_the_phase() -> None:
    features, labels, surrogate = _fitted()
    rows = features.iloc[:20]
    explanation = explain(surrogate, rows, labels.iloc[:20])
    assert explanation.values.shape == (20, 6)
    assert list(explanation.values.columns) == list(COLUMNS)
    margin = surrogate.model.predict(rows, output_margin=True)
    expected = margin[np.arange(20), labels.iloc[:20].to_numpy()]
    rebuilt = explanation.values.sum(axis=1) + explanation.base
    assert np.abs(rebuilt.to_numpy() - expected).max() < 1e-3


def test_block_sums_keep_the_total() -> None:
    features, labels, surrogate = _fitted()
    explanation = explain(surrogate, features.iloc[:20], labels.iloc[:20])
    sums = block_sums(explanation.values, BLOCKS)
    assert list(sums.columns) == ["left", "right"]
    assert np.allclose(sums.sum(axis=1), explanation.values.sum(axis=1))
    # the label depends on a_1 and c_1: both blocks carry weight somewhere
    assert (sums.abs().mean() > 0.01).all()


def test_top_variables_are_ordered_by_absolute_contribution() -> None:
    row = pd.Series({"a_1": 0.1, "a_2": -0.9, "b_1": 0.5, "c_1": 0.0, "d_1": -0.2, "d_2": 0.3})
    assert top_variables(row, 3) == [("a_2", -0.9), ("b_1", 0.5), ("d_2", 0.3)]


def test_a_phase_the_surrogate_never_saw_has_no_explanation() -> None:
    features, labels, _ = _fitted()
    two = labels.clip(upper=1) * 2  # phases 0 and 2 only
    surrogate = fit_surrogate(features, two, 3, PARAMS)
    phases = pd.Series([0, 1, 2], index=features.index[:3])
    explanation = explain(surrogate, features.iloc[:3], phases)
    assert explanation.values.iloc[1].isna().all() and np.isnan(explanation.base.iloc[1])
    assert explanation.values.iloc[[0, 2]].notna().all().all()
    margin = surrogate.model.predict(features.iloc[:3], output_margin=True)  # margin of phase 2
    rebuilt = explanation.values.sum(axis=1) + explanation.base
    assert rebuilt.iloc[2] == pytest.approx(float(np.ravel(margin)[2]), abs=1e-3)
    assert rebuilt.iloc[0] == pytest.approx(-float(np.ravel(margin)[0]), abs=1e-3)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_explain.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'termo.surrogate.explain'`.

- [ ] **Step 3: Implement**

`src/termo/surrogate/explain.py`:

```python
"""Why the surrogate gives today's phase: SHAP values in log-odds, summed by block.

SHAP explains the surrogate, not the market and not the jump model. It is only
worth reading when the surrogate imitates the jump model well (criterion D6).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from termo.surrogate.model import Surrogate

Blocks = Sequence[tuple[str, Sequence[str]]]  # (block name, first tokens of its variables)


def first_token(name: str) -> str:
    """d10_63_r252 -> d10; vol2_r252 -> vol2."""
    return name.split("_", 1)[0]


def block_map(columns: Sequence[str], blocks: Blocks) -> dict[str, str]:
    """Variable -> block. Every variable must fall in exactly one block."""
    owners: dict[str, list[str]] = {}
    for block, tokens in blocks:
        for token in tokens:
            owners.setdefault(token, []).append(block)
    mapping: dict[str, str] = {}
    for column in columns:
        found = owners.get(first_token(column), [])
        if not found:
            raise ValueError(f"variable {column} belongs to no block")
        if len(found) > 1:
            raise ValueError(f"variable {column} belongs to more than one block: {found}")
        mapping[column] = found[0]
    return mapping


@dataclass(frozen=True, eq=False)
class Explanation:
    values: pd.DataFrame  # rows x variables: contribution to the log-odds of the row's phase
    base: pd.Series  # the surrogate's base value for that phase


def explain(surrogate: Surrogate, features: pd.DataFrame, phases: pd.Series) -> Explanation:
    """SHAP values of each row for ITS phase. NaN where the surrogate never saw that phase."""
    if not features.index.equals(phases.index):
        raise ValueError("features and phases must cover the same rows")
    rows = features[list(surrogate.columns)]
    explainer = shap.TreeExplainer(surrogate.model)
    raw = np.asarray(explainer.shap_values(rows), dtype=float)
    base = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
    if raw.ndim == 2:  # two classes: a single margin, that of the second class
        raw = np.stack([-raw, raw], axis=2)
        base = np.array([-base[0], base[0]])
    position = {phase: i for i, phase in enumerate(surrogate.classes)}
    values = np.full(rows.shape, np.nan)
    bases = np.full(len(rows), np.nan)
    for row, phase in enumerate(phases.to_numpy()):
        where = position.get(int(phase))
        if where is not None:
            values[row] = raw[row, :, where]
            bases[row] = base[where]
    return Explanation(
        values=pd.DataFrame(values, index=rows.index, columns=list(surrogate.columns)),
        base=pd.Series(bases, index=rows.index),
    )


def block_sums(values: pd.DataFrame, blocks: Blocks) -> pd.DataFrame:
    """Sum the contributions of each block. Exact: SHAP values are additive."""
    mapping = block_map(list(values.columns), blocks)
    out = pd.DataFrame(index=values.index)
    for block, _ in blocks:
        members = [column for column in values.columns if mapping[column] == block]
        out[block] = values[members].sum(axis=1, min_count=1) if members else 0.0
    return out


def top_variables(row: pd.Series, count: int) -> list[tuple[str, float]]:
    """The `count` variables with the largest absolute contribution, largest first."""
    ordered = row.dropna().abs().sort_values(ascending=False, kind="stable").index[:count]
    return [(str(name), float(row[name])) for name in ordered]
```

Note on `min_count=1`: a row of NaN (phase never seen) sums to NaN, not 0.

- [ ] **Step 4: Run the suite, lint, types** — all pass. If the two-class branch test fails because `shap_values` returns shape `(rows, features, 1)` or `(rows, features, 2)` for a binary model in this shap version, adapt the branch to the shape actually returned (print `raw.shape` once) and keep the test's meaning: values + base rebuild the margin of the row's own phase.

- [ ] **Step 5: Commit**

```bash
git add src/termo/surrogate/explain.py tests/test_explain.py
git commit -m "feat: SHAP of the surrogate for each day's phase, summed by block

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Configuration of spec 3

**Files:**
- Modify: `src/termo/config.py`
- Create: `configs/desc.yaml`
- Modify: `tests/test_config.py` (append), `tests/conftest.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
DESC_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "desc.yaml"


def test_loads_the_descriptive_configuration() -> None:
    config = load_config(DESC_CONFIG)
    desc = config.descriptive
    assert desc is not None
    assert config.k_values == (3,) and config.jump_penalties == (3.0,)
    assert config.jump_penalty_per_feature is True and config.feature_set == "tyccles"
    assert config.frozen_train_end == date(2014, 12, 31)
    assert config.prior_trial_logs == ("trials/trials.jsonl", "trials/exp2/trials.jsonl")
    assert desc.phase_names == ("rally de la parte corta", "rally de la parte larga", "venta")
    assert (desc.sell_phase, desc.short_led_phase, desc.long_led_phase) == (2, 0, 1)
    assert (desc.slope_long, desc.slope_short, desc.level_series) == ("DGS10", "DGS2", "DGS10")
    assert desc.change_days == 63 and desc.min_days_evaluable == 40
    assert desc.coherence_share_min == 0.6 and desc.map_ari_min == 0.6
    assert desc.fidelity_min == 0.8 and desc.low_confidence_below == 0.6
    assert desc.surrogate.n_estimators == 300 and desc.surrogate.max_depth == 4
    assert desc.surrogate.learning_rate == 0.05 and desc.surrogate.seed == 0
    assert [name for name, _ in desc.blocks] == [
        "nivel corto", "nivel medio", "nivel largo", "pendientes", "curvatura", "volatilidad"
    ]
    assert dict(desc.blocks)["nivel corto"] == ("d1", "d2", "d3")
    assert dict(desc.blocks)["volatilidad"] == (
        "vol1", "vol2", "vol3", "vol5", "vol7", "vol10", "vol30"
    )


def test_descriptive_configuration_is_experiment_2_with_one_model() -> None:
    exp2, config = load_config(EXP2_CONFIG), load_config(DESC_CONFIG)
    aligned = replace(
        config,
        k_values=exp2.k_values,
        jump_penalties=exp2.jump_penalties,
        frozen_train_end=exp2.frozen_train_end,
        prior_trial_logs=exp2.prior_trial_logs,
        descriptive=None,
    )
    assert aligned == exp2


def test_descriptive_configuration_needs_one_model_and_a_name_per_phase() -> None:
    config = load_config(DESC_CONFIG)
    with pytest.raises(ValueError, match="one K and one jump penalty"):
        replace(config, k_values=(2, 3))
    assert config.descriptive is not None
    with pytest.raises(ValueError, match="one name per phase"):
        replace(config, descriptive=replace(config.descriptive, phase_names=("a", "b")))
    with pytest.raises(ValueError, match="frozen_train_end"):
        replace(config, frozen_train_end=None)
```

Append to `tests/conftest.py` (add `DescriptiveConfig`, `SurrogateConfig` to the `termo.config` import):

```python
def make_desc_config() -> CoreConfig:
    """Spec 3 on the synthetic curve: one 3-phase model, a small surrogate, a frozen fit in 1996."""
    return replace(
        make_tyccles_config(),
        k_values=(3,),
        jump_penalties=(0.5,),
        frozen_train_end=date(1996, 6, 28),
        descriptive=DescriptiveConfig(
            phase_names=("rally de la parte corta", "rally de la parte larga", "venta"),
            sell_phase=2,
            short_led_phase=0,
            long_led_phase=1,
            level_series="DGS10",
            slope_long="DGS10",
            slope_short="DGS2",
            change_days=63,
            coherence_share_min=0.6,
            min_days_evaluable=40,
            map_ari_min=0.6,
            fidelity_min=0.8,
            low_confidence_below=0.6,
            surrogate=SurrogateConfig(
                n_estimators=20,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                seed=0,
            ),
            blocks=(
                ("nivel corto", ("d1", "d2", "d3")),
                ("nivel medio", ("d5", "d7")),
                ("nivel largo", ("d10", "d30")),
                ("pendientes", ("s12m5s", "s3s10", "s10s30")),
                ("curvatura", ("c5",)),
                ("volatilidad", ("vol1", "vol2", "vol3", "vol5", "vol7", "vol10", "vol30")),
            ),
        ),
    )


@pytest.fixture(scope="session")
def desc_config() -> CoreConfig:
    return make_desc_config()
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -q`
Expected: FAIL with `ImportError: cannot import name 'DescriptiveConfig'`.

- [ ] **Step 3: Implement**

`src/termo/config.py` — add after `TycclesConfig`:

```python
@dataclass(frozen=True)
class SurrogateConfig:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    seed: int


@dataclass(frozen=True)
class DescriptiveConfig:
    """Spec 3: one fixed model read as a description, never as a forecast."""

    phase_names: tuple[str, ...]
    sell_phase: int
    short_led_phase: int  # the rally led by the short end: the curve steepens
    long_led_phase: int  # the rally led by the long end: the curve flattens
    level_series: str
    slope_long: str
    slope_short: str
    change_days: int  # contemporaneous changes are taken over the PAST this many days
    coherence_share_min: float
    min_days_evaluable: int
    map_ari_min: float
    fidelity_min: float
    low_confidence_below: float
    surrogate: SurrogateConfig
    blocks: tuple[tuple[str, tuple[str, ...]], ...]  # (block, first tokens of its variables)
```

In `CoreConfig`, after `prior_trial_logs`:

```python
    descriptive: DescriptiveConfig | None = None
```

Append to `__post_init__`:

```python
        if self.descriptive is not None:
            desc = self.descriptive
            if len(self.k_values) != 1 or len(self.jump_penalties) != 1:
                raise ValueError("the descriptive tool is one model: one K and one jump penalty")
            if len(desc.phase_names) != self.k_values[0]:
                raise ValueError("the descriptive tool needs one name per phase")
            phases = {desc.sell_phase, desc.short_led_phase, desc.long_led_phase}
            if len(phases) != 3 or not phases <= set(range(self.k_values[0])):
                raise ValueError("sell, short-led and long-led phases must be three valid phases")
            if self.frozen_train_end is None:
                raise ValueError("the descriptive tool needs frozen_train_end for criterion D5")
```

In `load_config`, before `return CoreConfig(`:

```python
    desc = raw.get("descriptive")
```

and add to the constructor call:

```python
        descriptive=None
        if desc is None
        else DescriptiveConfig(
            phase_names=tuple(str(name) for name in desc["phase_names"]),
            sell_phase=int(desc["sell_phase"]),
            short_led_phase=int(desc["short_led_phase"]),
            long_led_phase=int(desc["long_led_phase"]),
            level_series=str(desc["level_series"]),
            slope_long=str(desc["slope_long"]),
            slope_short=str(desc["slope_short"]),
            change_days=int(desc["change_days"]),
            coherence_share_min=float(desc["coherence_share_min"]),
            min_days_evaluable=int(desc["min_days_evaluable"]),
            map_ari_min=float(desc["map_ari_min"]),
            fidelity_min=float(desc["fidelity_min"]),
            low_confidence_below=float(desc["low_confidence_below"]),
            surrogate=SurrogateConfig(
                n_estimators=int(desc["surrogate"]["n_estimators"]),
                max_depth=int(desc["surrogate"]["max_depth"]),
                learning_rate=float(desc["surrogate"]["learning_rate"]),
                subsample=float(desc["surrogate"]["subsample"]),
                colsample_bytree=float(desc["surrogate"]["colsample_bytree"]),
                seed=int(desc["surrogate"]["seed"]),
            ),
            blocks=tuple(
                (str(name), tuple(str(token) for token in tokens))
                for name, tokens in desc["blocks"].items()
            ),
        ),
```

`configs/desc.yaml`:

```yaml
# TERMO spec 3 (descriptive tool). Every value here is pre-registered. The model, its
# penalty and the phase names were chosen LOOKING at pre-holdout data (experiment 2);
# only the holdout is clean evidence. Everything not under "spec 3" equals configs/exp2.yaml.
series: [DGS1, DGS2, DGS3, DGS5, DGS7, DGS10, DGS30]
start: 1977-02-15
holdout_start: 2024-10-01
first_train_end: 1987-12-31
refit_weeks: 26

feature_set: tyccles
tyccles:
  change_horizons_days: [21, 42, 63, 84, 126, 189]
  rank_windows_days: [126, 252]
  vol_window_days: 21
  vol_rank_window_days: 252
burn_in_days: 504
apply_collinearity_rule: false
jump_penalty_per_feature: true

# --- spec 3 ---
frozen_train_end: 2014-12-31  # D5 on pre-holdout: a model fitted once through here
prior_trial_logs: [trials/trials.jsonl, trials/exp2/trials.jsonl]  # 29 + 33 trials, both no-go
grid:
  k: [3]             # the trial jm_k3_lam3 of experiment 2
  jump_penalty: [3]  # per feature: lambda = 3 * 139 = 417
descriptive:
  phase_names: [rally de la parte corta, rally de la parte larga, venta]
  sell_phase: 2
  short_led_phase: 0
  long_led_phase: 1
  level_series: DGS10
  slope_long: DGS10
  slope_short: DGS2
  change_days: 63
  coherence_share_min: 0.6
  min_days_evaluable: 40
  map_ari_min: 0.6
  fidelity_min: 0.8
  low_confidence_below: 0.6
  surrogate:
    n_estimators: 300
    max_depth: 4
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
    seed: 0
  blocks:
    nivel corto: [d1, d2, d3]
    nivel medio: [d5, d7]
    nivel largo: [d10, d30]
    pendientes: [s12m5s, s3s10, s10s30]
    curvatura: [c5]
    volatilidad: [vol1, vol2, vol3, vol5, vol7, vol10, vol30]
# --- end spec 3 ---

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

- [ ] **Step 4: Run the suite, lint, types** — all pass.

- [ ] **Step 5: Commit**

```bash
git add src/termo/config.py configs/desc.yaml tests/test_config.py tests/conftest.py
git commit -m "feat: descriptive configuration (one model, surrogate, criteria, blocks)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: The descriptive criteria D1-D6

**Files:**
- Create: `src/termo/descriptive/__init__.py` (empty), `src/termo/descriptive/criteria.py`
- Create: `tests/test_criteria.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_criteria.py`:

```python
"""D1-D6 by hand: contemporaneous checks, never a look at what came after."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.descriptive.criteria import APTO, NO_APTO, calibration, evaluate, verdict

DAYS = 600


def _case() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
    """200 days of each phase, in order 2 (sell), 1 (long-led rally), 0 (short-led rally).

    Yields move so that every phase means what its name says, with a 63-day lead-in.
    """
    index = pd.bdate_range("2001-01-01", periods=DAYS + 63)
    step10 = np.r_[np.full(63, 0.02), np.full(200, 0.02), np.full(200, -0.02), np.full(200, -0.02)]
    step2 = np.r_[np.full(63, 0.02), np.full(200, 0.02), np.full(200, -0.01), np.full(200, -0.04)]
    curve = pd.DataFrame(
        {"DGS10": 20.0 + np.cumsum(step10), "DGS2": 20.0 + np.cumsum(step2)}, index=index
    )
    days = index[63:]
    labels = pd.Series(np.repeat([2, 1, 0], 200), index=days)
    proba = pd.DataFrame(0.05, index=days, columns=[0, 1, 2])
    for phase in (0, 1, 2):
        proba.loc[labels == phase, phase] = 0.9
    return labels, curve, proba, labels.copy()


def _by_name(checks: tuple, name: str):  # type: ignore[no-untyped-def]
    (found,) = [c for c in checks if c.name == name]
    return found


def test_a_coherent_map_passes_every_check(desc_config: CoreConfig) -> None:
    labels, curve, proba, frozen = _case()
    checks = evaluate(labels, curve, proba, frozen, desc_config)
    assert [c.name for c in checks] == [
        "D1_persistence",
        "D2_sell_coherence",
        "D3_rally_coherence",
        "D4_short_led_steepens",
        "D4_long_led_flattens",
        "D5_map_stable",
        "D6_fidelity",
    ]
    assert all(c.passed is True for c in checks)
    assert _by_name(checks, "D1_persistence").value == 200.0
    assert _by_name(checks, "D5_map_stable").value == pytest.approx(1.0)
    assert _by_name(checks, "D6_fidelity").value == pytest.approx(1.0)
    assert verdict(checks) == APTO


def test_each_check_can_fail_on_its_own(desc_config: CoreConfig) -> None:
    labels, curve, proba, frozen = _case()

    flicker = labels.copy()
    flicker.iloc[::7] = (flicker.iloc[::7] + 1) % 3  # runs of at most 6 days
    assert _by_name(evaluate(flicker, curve, proba, frozen, desc_config), "D1_persistence").passed is False

    falling = curve.copy()
    falling["DGS10"] = 20.0 - 0.02 * np.arange(len(falling))
    checks = evaluate(labels, falling, proba, frozen, desc_config)
    assert _by_name(checks, "D2_sell_coherence").passed is False

    rising = curve.copy()
    rising["DGS10"] = 20.0 + 0.02 * np.arange(len(rising))
    assert _by_name(evaluate(labels, rising, proba, frozen, desc_config), "D3_rally_coherence").passed is False

    swapped = labels.replace({0: 1, 1: 0})  # the names of the two rallies exchanged
    checks = evaluate(swapped, curve, proba.rename(columns={0: 1, 1: 0})[[0, 1, 2]], swapped, desc_config)
    assert _by_name(checks, "D4_short_led_steepens").passed is False
    assert _by_name(checks, "D4_long_led_flattens").passed is False

    rng = np.random.default_rng(0)
    other_map = pd.Series(rng.integers(0, 3, size=len(labels)), index=labels.index)
    assert _by_name(evaluate(labels, curve, proba, other_map, desc_config), "D5_map_stable").passed is False

    wrong = proba[[1, 2, 0]].set_axis([0, 1, 2], axis=1)  # the surrogate points at another phase
    assert _by_name(evaluate(labels, curve, wrong, frozen, desc_config), "D6_fidelity").passed is False
    assert verdict(evaluate(labels, curve, wrong, frozen, desc_config)) == NO_APTO


def test_a_phase_with_few_days_is_not_evaluable_and_does_not_fail(desc_config: CoreConfig) -> None:
    labels, curve, proba, frozen = _case()
    keep = labels.index[:430]  # 200 sell, 200 long-led rally, 30 short-led rally (< 40)
    checks = evaluate(labels.loc[keep], curve, proba.loc[keep], frozen.loc[keep], desc_config)
    assert _by_name(checks, "D4_short_led_steepens").passed is None
    assert _by_name(checks, "D4_short_led_steepens").value is None
    assert _by_name(checks, "D3_rally_coherence").passed is True  # judged on the long-led rally
    assert verdict(checks) == APTO


def test_no_evaluable_phase_means_not_fit(desc_config: CoreConfig) -> None:
    labels, curve, proba, frozen = _case()
    short = pd.Index(np.r_[labels.index[:30], labels.index[200:230], labels.index[400:430]])
    checks = evaluate(labels.loc[short], curve, proba.loc[short], frozen.loc[short], desc_config)
    assert all(_by_name(checks, n).passed is None for n in ("D1_persistence", "D2_sell_coherence"))
    assert verdict(checks) == NO_APTO


def test_the_frozen_map_may_cover_fewer_days(desc_config: CoreConfig) -> None:
    labels, curve, proba, frozen = _case()
    checks = evaluate(labels, curve, proba, frozen.iloc[300:], desc_config)
    assert _by_name(checks, "D5_map_stable").value == pytest.approx(1.0)
    few = evaluate(labels, curve, proba, frozen.iloc[:10], desc_config)
    assert _by_name(few, "D5_map_stable").passed is None


def test_checks_use_the_configured_thresholds(desc_config: CoreConfig) -> None:
    labels, curve, proba, frozen = _case()
    assert desc_config.descriptive is not None
    strict = replace(desc_config, descriptive=replace(desc_config.descriptive, fidelity_min=1.01))
    assert _by_name(evaluate(labels, curve, proba, frozen, strict), "D6_fidelity").passed is False


def test_calibration_reports_brier_and_reliability() -> None:
    labels, _, proba, _ = _case()
    report = calibration(labels, proba, bins=5)
    assert set(report) == {"brier", "reliability"}
    assert set(report["brier"]) == {"0", "1", "2"}
    assert all(0.0 <= value <= 1.0 for value in report["brier"].values())
    assert all({"bin", "mean_probability", "share_correct", "days"} == set(row) for row in report["reliability"])
```

(Wrap the long lines to 100 characters with `ruff format` after pasting.)

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_criteria.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'termo.descriptive'`.

- [ ] **Step 3: Implement**

`src/termo/descriptive/__init__.py`: empty file.

`src/termo/descriptive/criteria.py`:

```python
"""The descriptive criteria of spec 3 (D1-D6). All contemporaneous.

Every change is taken over the PAST `change_days` days: nothing here looks at what
happened after a reading. D2-D4 are partly true by construction (the phases are built
from ranks of these same changes): they check that the names still hold on new data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from termo.config import CoreConfig
from termo.validation.metrics import BP_PER_PERCENT, adjusted_rand, median_durations

APTO, NO_APTO = "apto", "no-apto"
PHASE_CHECKS = (
    "D1_persistence",
    "D2_sell_coherence",
    "D3_rally_coherence",
    "D4_short_led_steepens",
    "D4_long_led_flattens",
)


@dataclass(frozen=True)
class Check:
    name: str
    value: float | None  # None: not evaluable on this period
    requirement: str
    passed: bool | None  # None: not evaluable; it is reported and does not fail


def _check(name: str, value: float | None, requirement: str, passed: bool | None) -> Check:
    return Check(name, None if value is None else float(value), requirement, passed)


def evaluate(
    labels: pd.Series,
    curve: pd.DataFrame,
    proba: pd.DataFrame,
    frozen_labels: pd.Series,
    config: CoreConfig,
) -> tuple[Check, ...]:
    """The seven checks on the days of `labels`.

    `curve` must reach back at least `change_days` rows before the first labelled day.
    `frozen_labels` are the labels of a model fitted once and never refitted; they may
    cover only some of the days.
    """
    desc = config.descriptive
    if desc is None:
        raise ValueError("the configuration has no descriptive section")
    if not proba.index.equals(labels.index):
        raise ValueError("labels and probabilities must cover the same days")
    n_states = config.k_values[0]
    change = (curve[desc.level_series].diff(desc.change_days) * BP_PER_PERCENT).reindex(labels.index)
    slope = curve[desc.slope_long] - curve[desc.slope_short]
    slope_change = (slope.diff(desc.change_days) * BP_PER_PERCENT).reindex(labels.index)
    if change.isna().any() or slope_change.isna().any():
        raise ValueError("the curve does not reach back far enough for the changes")

    days = labels.value_counts()
    evaluable = {p for p in range(n_states) if int(days.get(p, 0)) >= desc.min_days_evaluable}
    minimum = config.thresholds.min_median_duration_days
    share = desc.coherence_share_min

    durations = median_durations(labels.to_numpy(), n_states)
    kept = [durations[p] for p in sorted(evaluable)]
    d1 = (
        _check("D1_persistence", min(kept), f">= {minimum} days", min(kept) >= minimum)
        if kept
        else _check("D1_persistence", None, f">= {minimum} days", None)
    )

    def share_check(name: str, phases: Sequence[int], rising: bool) -> Check:
        requirement = f">= {share}"
        shares = []
        for phase in phases:
            if phase in evaluable:
                moves = change[labels == phase]
                shares.append(float((moves > 0).mean() if rising else (moves < 0).mean()))
        if not shares:
            return _check(name, None, requirement, None)
        return _check(name, min(shares), requirement, min(shares) >= share)

    def lead_check(name: str, phase: int, steepens: bool) -> Check:
        requirement = "> 0 bp" if steepens else "< 0 bp"
        if phase not in evaluable:
            return _check(name, None, requirement, None)
        mean = float(slope_change[labels == phase].mean())
        return _check(name, mean, requirement, mean > 0 if steepens else mean < 0)

    rallies = [p for p in range(n_states) if p != desc.sell_phase]
    shared = frozen_labels.index.intersection(labels.index)
    if len(shared) >= desc.min_days_evaluable:
        ari = adjusted_rand(frozen_labels.loc[shared].to_numpy(), labels.loc[shared].to_numpy())
        d5 = _check("D5_map_stable", ari, f">= {desc.map_ari_min}", ari >= desc.map_ari_min)
    else:
        d5 = _check("D5_map_stable", None, f">= {desc.map_ari_min}", None)

    guess = proba.idxmax(axis=1).to_numpy().astype(int)
    fidelity = float(balanced_accuracy_score(labels.to_numpy().astype(int), guess))
    d6 = _check(
        "D6_fidelity", fidelity, f">= {desc.fidelity_min}", fidelity >= desc.fidelity_min
    )
    return (
        d1,
        share_check("D2_sell_coherence", [desc.sell_phase], rising=True),
        share_check("D3_rally_coherence", rallies, rising=False),
        lead_check("D4_short_led_steepens", desc.short_led_phase, steepens=True),
        lead_check("D4_long_led_flattens", desc.long_led_phase, steepens=False),
        d5,
        d6,
    )


def verdict(checks: Sequence[Check]) -> str:
    """APTO when every evaluable check passes and at least one phase could be judged."""
    by_name = {c.name: c for c in checks}
    if all(by_name[name].passed is None for name in PHASE_CHECKS):
        return NO_APTO  # no phase had enough days: no evidence is not a pass
    return NO_APTO if any(c.passed is False for c in checks) else APTO


def calibration(labels: pd.Series, proba: pd.DataFrame, bins: int) -> dict[str, Any]:
    """Brier score per phase and a reliability table of the top probability. Reported only."""
    truth = labels.to_numpy().astype(int)
    brier = {
        str(phase): float(np.mean((proba[phase].to_numpy() - (truth == phase)) ** 2))
        for phase in proba.columns
    }
    top = proba.max(axis=1).to_numpy()
    correct = proba.idxmax(axis=1).to_numpy().astype(int) == truth
    edges = np.linspace(0.0, 1.0, bins + 1)
    where = np.clip(np.digitize(top, edges[1:-1]), 0, bins - 1)
    reliability = [
        {
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "mean_probability": float(top[where == b].mean()) if (where == b).any() else None,
            "share_correct": float(correct[where == b].mean()) if (where == b).any() else None,
            "days": int((where == b).sum()),
        }
        for b in range(bins)
    ]
    return {"brier": brier, "reliability": reliability}
```

- [ ] **Step 4: Run the suite, lint, types** — all pass. If `balanced_accuracy_score` warns when a class is absent from the predictions, that is expected; do not silence it globally.

- [ ] **Step 5: Commit**

```bash
git add src/termo/descriptive tests/test_criteria.py
git commit -m "feat: descriptive criteria D1-D6 with the not-evaluable rule

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Stages register, run and report

**Files:**
- Create: `src/termo/descriptive/stages.py`
- Create: `tests/test_desc_stages.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_desc_stages.py`:

```python
"""Spec 3 end to end on the synthetic curve: register, run, report (the holdout is Task 8)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import fake_fred
from termo.config import CoreConfig
from termo.core import SETUP_TRIAL, registered_columns, take_snapshot
from termo.data.loader import load_curve
from termo.data.snapshot import snapshot_hash
from termo.dataset import ExperimentData, prepare
from termo.descriptive.criteria import APTO, NO_APTO
from termo.descriptive.stages import (
    FILES,
    PRE_HOLDOUT_DIR,
    analyse,
    desc_trial_id,
    load_analysis,
    register_desc,
    report_desc,
    run_desc,
)
from termo.validation.trials import TrialLog, TrialLogError

COMMIT = "test-commit-3"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("termo-desc")


@pytest.fixture(scope="module")
def snapshot_dir(workspace: Path, curve: pd.DataFrame, desc_config: CoreConfig) -> Path:
    target = workspace / "data" / "snapshots" / "2026-10-02"
    take_snapshot(desc_config, target, fake_fred(curve), "2026-10-02T00:00:00+00:00")
    return target


@pytest.fixture(scope="module")
def log(workspace: Path) -> TrialLog:
    return TrialLog(workspace / "trials" / "desc" / "trials.jsonl")


@pytest.fixture(scope="module")
def trials_dir(workspace: Path) -> Path:
    return workspace / "trials" / "desc"


@pytest.fixture(scope="module")
def data(snapshot_dir: Path, log: TrialLog, desc_config: CoreConfig) -> ExperimentData:
    curve = load_curve(snapshot_dir, desc_config.series, desc_config.start, desc_config.holdout_start)
    register_desc(desc_config, curve, log, snapshot_hash(snapshot_dir), COMMIT)
    return prepare(curve, desc_config, registered_columns(log))


@pytest.fixture(scope="module")
def finished(
    workspace: Path, data: ExperimentData, log: TrialLog, trials_dir: Path, snapshot_dir: Path
) -> Path:
    data_hash = snapshot_hash(snapshot_dir)
    messages: list[str] = []
    run_desc(data, log, trials_dir, data_hash, COMMIT, echo=messages.append)
    assert len(messages) == 1 and messages[0].isascii() and messages[0].startswith("[ok]")
    reports = workspace / "reports" / "desc"
    report_desc(data, log, trials_dir, reports, data_hash, COMMIT)
    return reports


def test_one_trial_is_registered_with_everything_it_is_bound_to(
    data: ExperimentData, log: TrialLog, desc_config: CoreConfig
) -> None:
    registrations = log.registrations()
    trial = desc_trial_id(desc_config)
    assert trial == "desc_k3" and set(registrations) == {SETUP_TRIAL, trial}
    assert registrations[trial]["config"] == {
        "model": "descriptive", "n_states": 3, "jump_penalty": 0.5
    }
    setup = registrations[SETUP_TRIAL]["config"]
    assert len(setup["columns"]) == 139 and "xgboost" in setup["environment"]
    assert setup["config"]["descriptive"]["fidelity_min"] == 0.8
    with pytest.raises(TrialLogError):
        register_desc(desc_config, data.curve, log, "x", COMMIT)


def test_analysis_covers_every_out_of_sample_day_without_look_ahead(
    data: ExperimentData, desc_config: CoreConfig
) -> None:
    analysis = analyse(data)
    days = analysis.labels.index
    assert days[0] > data.refits[0].cutoff and days[-1] == data.curve.index[-1]
    assert analysis.proba.index.equals(days) and analysis.shap_blocks.index.equals(days)
    assert analysis.shap_top.index.equals(days)
    assert list(analysis.proba.columns) == [0, 1, 2]
    assert list(analysis.shap_blocks.columns) == [
        "nivel corto", "nivel medio", "nivel largo", "pendientes", "curvatura", "volatilidad", "base"
    ]
    assert analysis.frozen_labels.index[0] > pd.Timestamp(desc_config.frozen_train_end)
    assert analysis.frozen_labels.index[-1] == days[-1]
    # no look-ahead: cutting the last 60 days leaves every earlier output unchanged
    shorter = analyse(prepare(data.curve.iloc[:-60], desc_config, data.columns))
    common = shorter.labels.index
    pd.testing.assert_series_equal(shorter.labels, analysis.labels.loc[common])
    pd.testing.assert_frame_equal(shorter.proba, analysis.proba.loc[common])
    pd.testing.assert_frame_equal(shorter.shap_blocks, analysis.shap_blocks.loc[common])


def test_run_writes_hashed_files_and_records_the_checks(
    finished: Path, log: TrialLog, trials_dir: Path, desc_config: CoreConfig
) -> None:
    result = log.results()[desc_trial_id(desc_config)]
    metrics = result["metrics"]
    assert set(metrics["files"]) == set(FILES)
    for name, expected in metrics["files"].items():
        path = trials_dir / PRE_HOLDOUT_DIR / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    assert result["labels_sha256"] == metrics["files"]["labels.csv"]
    assert [c["name"] for c in metrics["checks"]][0] == "D1_persistence" and len(metrics["checks"]) == 7
    assert metrics["verdict"] in {APTO, NO_APTO}
    assert set(metrics["calibration"]) == {"brier", "reliability"}
    json.dumps(metrics)


def test_run_never_repeats_and_files_round_trip(
    finished: Path, data: ExperimentData, log: TrialLog, trials_dir: Path, snapshot_dir: Path
) -> None:
    before = log.path.read_text(encoding="utf-8")
    messages: list[str] = []
    run_desc(data, log, trials_dir, snapshot_hash(snapshot_dir), COMMIT, echo=messages.append)
    assert messages == [] and log.path.read_text(encoding="utf-8") == before
    loaded = load_analysis(trials_dir / PRE_HOLDOUT_DIR, log.results()["desc_k3"]["metrics"]["files"])
    fresh = analyse(data)
    pd.testing.assert_series_equal(loaded.labels, fresh.labels, check_names=False, check_freq=False)
    assert np.allclose(loaded.proba.to_numpy(), fresh.proba.to_numpy())
    assert np.allclose(
        loaded.shap_blocks.to_numpy(), fresh.shap_blocks.to_numpy(), equal_nan=True, atol=1e-9
    )


def test_report_recomputes_from_the_files_and_is_logged(
    finished: Path, log: TrialLog, desc_config: CoreConfig
) -> None:
    payload = json.loads((finished / "diagnostic.json").read_text(encoding="utf-8"))
    logged = log.last_report()
    assert logged is not None and logged["stage"] == "diagnostic"
    assert logged["verdict"] == payload["verdict"] in {APTO, NO_APTO}
    assert payload["checks"] == log.results()["desc_k3"]["metrics"]["checks"]
    assert payload["final_trial_id"] == "desc_k3"
    assert payload["disclosure"]["n_trials_this_log"] == 1
    markdown = (finished / "diagnostic.md").read_text(encoding="utf-8")
    assert markdown.isascii() and "does not anticipate" in markdown
    assert "D6_fidelity" in markdown and "pre-holdout" in markdown


def test_report_refuses_tampered_files_and_a_changed_configuration(
    finished: Path, data: ExperimentData, log: TrialLog, trials_dir: Path, snapshot_dir: Path,
    tmp_path: Path,
) -> None:
    data_hash = snapshot_hash(snapshot_dir)
    other = replace(data, config=replace(data.config, refit_weeks=13))
    with pytest.raises(TrialLogError, match="configuration"):
        report_desc(other, log, trials_dir, tmp_path, data_hash, COMMIT)
    path = trials_dir / PRE_HOLDOUT_DIR / "proba.csv"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        with pytest.raises(TrialLogError, match="not the ones recorded"):
            report_desc(data, log, trials_dir, tmp_path, data_hash, COMMIT)
    finally:
        path.write_bytes(original)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_desc_stages.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'termo.descriptive.stages'`.

- [ ] **Step 3: Implement**

`src/termo/descriptive/stages.py`:

```python
"""The stages of spec 3: register one model, run it, report the pre-holdout diagnostic.

`run` writes every daily output to hashed CSV files. `report`, the holdout stage and
the weekly reading only read those files: they never refit anything.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from termo.config import CoreConfig
from termo.core import (
    SETUP_TRIAL,
    config_fingerprint,
    environment_fingerprint,
    prior_log_summary,
    verify_binding,
)
from termo.dataset import ExperimentData, first_window_columns
from termo.descriptive.criteria import Check, calibration, evaluate, verdict
from termo.experiment import effective_penalty
from termo.regime.model import jump_fitter
from termo.surrogate.explain import block_sums, explain, top_variables
from termo.surrogate.model import SurrogateParams, surrogate_walk
from termo.validation.trials import TrialLog, TrialLogError, TrialStatus
from termo.validation.walkforward import run_walkforward

MODEL_DESCRIPTIVE = "descriptive"
PRE_HOLDOUT_DIR, HOLDOUT_DIR = "pre_holdout", "holdout"
FILES = ("labels.csv", "frozen_labels.csv", "proba.csv", "shap_blocks.csv", "shap_top.csv")
TOP_VARIABLES = 5
CALIBRATION_BINS = 5
BASE = "base"
DISCLAIMER = (
    "TERMO describes the current phase of the curve. It does not anticipate the 10Y: "
    "in 62 registered trials the phases did not beat inertia."
)
LIMITS = (
    "K, the penalty and the phase names were chosen looking at pre-holdout data.",
    "D2-D4 are partly true by construction: the phases are built from ranks of these changes.",
    "The fidelity threshold is a judgement, not a calibrated value.",
    "SHAP explains the surrogate, not the market and not the jump model.",
    "Nothing here measures the ability to anticipate; that question is closed with NO-GO.",
)


def desc_trial_id(config: CoreConfig) -> str:
    return f"desc_k{config.k_values[0]}"


def _desc(config: CoreConfig) -> Any:
    if config.descriptive is None:
        raise TrialLogError("this configuration has no descriptive section")
    return config.descriptive


def surrogate_params(config: CoreConfig) -> SurrogateParams:
    s = _desc(config).surrogate
    return SurrogateParams(
        n_estimators=s.n_estimators,
        max_depth=s.max_depth,
        learning_rate=s.learning_rate,
        subsample=s.subsample,
        colsample_bytree=s.colsample_bytree,
        seed=s.seed,
    )


@dataclass(frozen=True, eq=False)
class Analysis:
    labels: pd.Series  # online phase of the jump model, every out-of-sample day
    frozen_labels: pd.Series  # the same from a model fitted once at frozen_train_end
    proba: pd.DataFrame  # surrogate probability per phase, same days as labels
    shap_blocks: pd.DataFrame  # block sums for the day's phase, plus the base value
    shap_top: pd.DataFrame  # one column "top": JSON list of [variable, contribution]


def analyse(data: ExperimentData) -> Analysis:
    """Jump model, frozen jump model, surrogate and SHAP over every out-of-sample day."""
    config = data.config
    desc = _desc(config)
    n_states = config.k_values[0]
    penalty = effective_penalty(config, config.jump_penalties[0], len(data.columns))
    fitter = jump_fitter(n_states, penalty)
    target = data.daily_change_10y
    walk = run_walkforward(data.refits, fitter, n_states, target)
    frozen = run_walkforward(data.frozen_refits, fitter, n_states, target).oos_labels
    surrogates = surrogate_walk(data.refits, walk.fits, n_states, surrogate_params(config))

    blocks: list[pd.DataFrame] = []
    tops: list[pd.Series] = []
    for refit, surrogate in zip(data.refits, surrogates.surrogates, strict=True):
        later = refit.features.loc[refit.features.index > refit.cutoff]
        explanation = explain(surrogate, later, walk.oos_labels.loc[later.index])
        sums = block_sums(explanation.values, desc.blocks)
        sums[BASE] = explanation.base
        blocks.append(sums)
        tops.append(
            explanation.values.apply(
                lambda row: json.dumps(top_variables(row, TOP_VARIABLES)), axis=1
            )
        )
    return Analysis(
        labels=walk.oos_labels,
        frozen_labels=frozen,
        proba=surrogates.proba,
        shap_blocks=pd.concat(blocks),
        shap_top=pd.concat(tops).to_frame("top"),
    )


def restrict(analysis: Analysis, start: pd.Timestamp) -> Analysis:
    """The same outputs from `start` on (used for the holdout days)."""
    return Analysis(
        labels=analysis.labels.loc[start:],
        frozen_labels=analysis.frozen_labels.loc[start:],
        proba=analysis.proba.loc[start:],
        shap_blocks=analysis.shap_blocks.loc[start:],
        shap_top=analysis.shap_top.loc[start:],
    )


def _write(frame: pd.DataFrame, path: Path) -> str:
    text: str = frame.to_csv(index_label="date", lineterminator="\n")
    data = text.encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def save_analysis(analysis: Analysis, directory: Path) -> dict[str, str]:
    """Write the five files and return the SHA-256 of each, to be stored in the log."""
    directory.mkdir(parents=True, exist_ok=True)
    proba = analysis.proba.rename(columns=lambda phase: f"p{phase}")
    frames = {
        "labels.csv": analysis.labels.rename("label").to_frame(),
        "frozen_labels.csv": analysis.frozen_labels.rename("label").to_frame(),
        "proba.csv": proba,
        "shap_blocks.csv": analysis.shap_blocks,
        "shap_top.csv": analysis.shap_top,
    }
    return {name: _write(frames[name], directory / name) for name in FILES}


def load_analysis(directory: Path, expected: Mapping[str, str]) -> Analysis:
    """Read the files back, refusing any whose bytes are not the ones the log recorded."""
    frames: dict[str, pd.DataFrame] = {}
    for name in FILES:
        path = directory / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
            raise TrialLogError(f"{name}: the file is not the ones recorded in the log")
        frames[name] = pd.read_csv(path, parse_dates=["date"], index_col="date")
    proba = frames["proba.csv"].rename(columns=lambda name: int(str(name)[1:]))
    return Analysis(
        labels=frames["labels.csv"]["label"],
        frozen_labels=frames["frozen_labels.csv"]["label"],
        proba=proba,
        shap_blocks=frames["shap_blocks.csv"],
        shap_top=frames["shap_top.csv"],
    )


def checks_of(analysis: Analysis, curve: pd.DataFrame, config: CoreConfig) -> tuple[Check, ...]:
    return evaluate(analysis.labels, curve, analysis.proba, analysis.frozen_labels, config)


def register_desc(
    config: CoreConfig, curve: pd.DataFrame, log: TrialLog, snapshot_hash: str, code_commit: str
) -> tuple[str, ...]:
    """Write the setup record and the single trial before anything is run."""
    desc = _desc(config)
    if code_commit.endswith("-dirty"):
        raise TrialLogError("commit the code and configuration before registering")
    columns = first_window_columns(curve, config)
    log.register(
        SETUP_TRIAL,
        "One model, its surrogate and the descriptive criteria, all fixed in advance.",
        {
            "columns": list(columns),
            "config": config_fingerprint(config),
            "environment": environment_fingerprint(),
            "hypotheses": [
                "The 3-phase map chosen on pre-holdout data stays persistent, keeps the meaning "
                "of its names and is imitated by the surrogate on the holdout (D1-D6)."
            ],
            "prior_trial_logs": prior_log_summary(config),
        },
        snapshot_hash,
        code_commit,
    )
    log.register(
        desc_trial_id(config),
        f"Descriptive tool: jump model with {config.k_values[0]} phases "
        f"({', '.join(desc.phase_names)}), XGBoost surrogate, criteria D1-D6.",
        {
            "model": MODEL_DESCRIPTIVE,
            "n_states": config.k_values[0],
            "jump_penalty": config.jump_penalties[0],
        },
        snapshot_hash,
        code_commit,
    )
    return columns


def run_desc(
    data: ExperimentData,
    log: TrialLog,
    trials_dir: Path,
    snapshot_hash: str,
    code_commit: str,
    echo: Callable[[str], None] = print,
) -> None:
    """Run the registered trial on pre-holdout data once. An exception leaves it pending."""
    config = data.config
    verify_binding(log, config, snapshot_hash, code_commit)
    trial = desc_trial_id(config)
    if not log.is_registered(trial):
        raise TrialLogError(f"trial {trial} is not registered")
    if log.has_result(trial):
        return
    analysis = analyse(data)
    checks = checks_of(analysis, data.curve, config)
    directory = trials_dir / PRE_HOLDOUT_DIR
    files = save_analysis(analysis, directory)
    log.record_result(
        trial,
        {
            "checks": [asdict(c) for c in checks],
            "verdict": verdict(checks),
            "calibration": calibration(analysis.labels, analysis.proba, CALIBRATION_BINS),
            "days": int(len(analysis.labels)),
            "files": files,
        },
        TrialStatus.KEPT,
        "",
        (directory / "labels.csv").as_posix(),
        files["labels.csv"],
    )
    echo(f"[ok] {trial} diagnostic={verdict(checks)} days={len(analysis.labels)}")


def disclosure(log: TrialLog) -> dict[str, Any]:
    registrations = log.registrations()
    prior = list(registrations[SETUP_TRIAL]["config"].get("prior_trial_logs", []))
    here = len([t for t in registrations if t != SETUP_TRIAL])
    return {
        "n_trials_this_log": here,
        "prior_logs": prior,
        "n_trials_total": here + sum(int(p["n_trials"]) for p in prior),
    }


def render(title: str, period: str, payload: Mapping[str, Any]) -> str:
    """ASCII Markdown of a diagnostic or holdout report."""
    lines = [
        f"# TERMO - {title}",
        "",
        f"**Verdict: {str(payload['verdict']).upper()}** ({period})",
        "",
        DISCLAIMER,
        "",
        "| Check | Value | Requirement | Result |",
        "|---|---|---|---|",
    ]
    for check in payload["checks"]:
        value = "-" if check["value"] is None else f"{check['value']:.4f}"
        result = {True: "pass", False: "FAIL", None: "not evaluable"}[check["passed"]]
        lines.append(f"| {check['name']} | {value} | {check['requirement']} | {result} |")
    lines += ["", f"Days evaluated: {payload['days']}", "", "## What these checks do not prove", ""]
    lines += [f"- {limit}" for limit in LIMITS]
    found = payload["disclosure"]
    lines += ["", "## Disclosure", "", f"- Trials in this log: {found['n_trials_this_log']}"]
    lines += [
        f"- {prior['path']}: {prior['n_trials']} trials, verdict {prior['verdict']}"
        for prior in found["prior_logs"]
    ]
    lines += [f"- Total registered trials: {found['n_trials_total']}", ""]
    return "\n".join(lines)


def write_report(name: str, title: str, period: str, payload: Mapping[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.md").write_text(render(title, period, payload), encoding="utf-8", newline="\n")
    (out / f"{name}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8", newline="\n"
    )


def report_desc(
    data: ExperimentData,
    log: TrialLog,
    trials_dir: Path,
    reports_dir: Path,
    snapshot_hash: str,
    code_commit: str,
) -> str:
    """Recompute the checks from the logged files, log the verdict, write the report."""
    config = data.config
    verify_binding(log, config, snapshot_hash, code_commit)
    trial = desc_trial_id(config)
    results = log.results()
    if trial not in results:
        raise TrialLogError(f"trial {trial} has no result: run the run stage first")
    recorded = results[trial]["metrics"]
    analysis = load_analysis(trials_dir / PRE_HOLDOUT_DIR, recorded["files"])
    checks = [asdict(c) for c in checks_of(analysis, data.curve, config)]
    payload = {
        "stage": "diagnostic",
        "verdict": verdict(checks_of(analysis, data.curve, config)),
        "final_trial_id": trial,
        "checks": checks,
        "days": int(len(analysis.labels)),
        "calibration": recorded["calibration"],
        "limits": list(LIMITS),
        "disclosure": disclosure(log),
    }
    log.record_report({**payload, "snapshot_hash": snapshot_hash, "code_commit": code_commit})
    write_report("diagnostic", "descriptive diagnostic", "pre-holdout, seen data", payload, reports_dir)
    return str(payload["verdict"])
```

Notes for the implementer:
- `registered_columns(log)` from `termo.core` reads `setup["config"]["columns"]`, so the setup record keeps that key.
- `verify_binding` compares `setup["config"]["config"]`, `["environment"]`, the snapshot hash and the code commit: the setup record must store exactly these keys.
- Values recomputed in `report_desc` from the CSV files can differ from the recorded ones in the last decimals (CSV round trip). The test compares `payload["checks"]` with the recorded checks for equality: if that fails only by float noise, round every check value to 10 decimals in `criteria._check` (`round(float(value), 10)`) so both paths agree, and say so in your report.

- [ ] **Step 4: Run the suite, lint, types** — all pass. The module-scoped fixtures fit ~8 jump models with 139 features and 8 small surrogates; expect well under two minutes for this test module.

- [ ] **Step 5: Commit**

```bash
git add src/termo/descriptive/stages.py tests/test_desc_stages.py
git commit -m "feat: descriptive stages register, run, report with hashed daily outputs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: The holdout stage (one shot)

**Files:**
- Modify: `src/termo/descriptive/stages.py` (append `holdout_desc`)
- Modify: `tests/test_desc_stages.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_desc_stages.py` (add `holdout_desc`, `HOLDOUT_DIR` to the stages import):

```python
def _approved_copy(log: TrialLog, tmp_path: Path, verdict: str = APTO) -> TrialLog:
    """A private copy of the finished log whose last report says `verdict`."""
    copy = TrialLog(tmp_path / "trials.jsonl")
    copy.path.write_text(log.path.read_text(encoding="utf-8"), encoding="utf-8")
    last = log.last_report()
    assert last is not None
    payload = {k: v for k, v in last.items() if k not in {"kind", "trial_id", "at"}}
    copy.record_report({**payload, "verdict": verdict})
    return copy


def test_holdout_needs_an_apt_diagnostic(
    finished: Path, tmp_path: Path, snapshot_dir: Path, log: TrialLog, trials_dir: Path,
    desc_config: CoreConfig,
) -> None:
    empty = TrialLog(tmp_path / "empty.jsonl")
    with pytest.raises(TrialLogError):
        holdout_desc(desc_config, snapshot_dir, empty, trials_dir, tmp_path, COMMIT)
    refused = _approved_copy(log, tmp_path, NO_APTO)
    with pytest.raises(TrialLogError, match="diagnostic"):
        holdout_desc(desc_config, snapshot_dir, refused, trials_dir, tmp_path, COMMIT)
    assert not refused.holdout_opened()


def test_holdout_checks_everything_before_opening(
    finished: Path, tmp_path: Path, snapshot_dir: Path, log: TrialLog, trials_dir: Path,
    desc_config: CoreConfig,
) -> None:
    copy = _approved_copy(log, tmp_path)
    with pytest.raises(TrialLogError, match="code commit"):
        holdout_desc(desc_config, snapshot_dir, copy, trials_dir, tmp_path, "another-commit")
    changed = replace(desc_config, refit_weeks=13)
    with pytest.raises(TrialLogError, match="configuration"):
        holdout_desc(changed, snapshot_dir, copy, trials_dir, tmp_path, COMMIT)
    assert not copy.holdout_opened()


def test_holdout_runs_once_on_holdout_days_only(
    finished: Path, tmp_path: Path, snapshot_dir: Path, log: TrialLog, desc_config: CoreConfig,
    curve: pd.DataFrame,
) -> None:
    copy = _approved_copy(log, tmp_path)
    private_trials, reports = tmp_path / "trials", tmp_path / "reports"
    result = holdout_desc(desc_config, snapshot_dir, copy, private_trials, reports, COMMIT)

    start = pd.Timestamp(desc_config.holdout_start)
    kinds = [r["kind"] for r in copy.records()]
    assert kinds[-2:] == ["holdout_opened", "holdout_result"]
    logged = copy.records()[-1]
    assert logged["verdict"] == result in {APTO, NO_APTO} and len(logged["checks"]) == 7
    assert logged["days"] == int((curve.index >= start).sum())
    loaded = load_analysis(private_trials / HOLDOUT_DIR, logged["files"])
    assert loaded.labels.index[0] >= start and loaded.proba.index.equals(loaded.labels.index)
    # the frozen map of the holdout was fitted on everything before it, never on holdout days
    assert loaded.frozen_labels.index.equals(loaded.labels.index)
    payload = json.loads((reports / "holdout.json").read_text(encoding="utf-8"))
    assert payload["verdict"] == result and payload["stage"] == "holdout"
    assert (reports / "holdout.md").read_text(encoding="utf-8").isascii()
    with pytest.raises(TrialLogError, match="already been opened"):
        holdout_desc(desc_config, snapshot_dir, copy, private_trials, reports, COMMIT)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_desc_stages.py -q`
Expected: FAIL with `ImportError: cannot import name 'holdout_desc'`.

- [ ] **Step 3: Implement**

Append to `src/termo/descriptive/stages.py` (add the imports `from dataclasses import replace`, `from datetime import timedelta`, `from termo.core import MIN_HOLDOUT_DAYS, registered_columns`, `from termo.data.loader import load_curve, snapshot_days`, `from termo.data.snapshot import snapshot_hash as hash_of_snapshot`, `from termo.dataset import prepare, recipe_for`, `from termo.descriptive.criteria import APTO`):

```python
def holdout_desc(
    config: CoreConfig,
    snapshot_dir: Path,
    log: TrialLog,
    trials_dir: Path,
    reports_dir: Path,
    code_commit: str,
) -> str:
    """The one evaluation on the holdout. The log refuses a second one.

    Everything that can fail by something knowable in advance is checked before the
    holdout is recorded as opened. Opened without a result means lost: that is the
    policy the user chose.
    """
    _desc(config)
    report = log.last_report()
    if report is None or report.get("stage") != "diagnostic" or report.get("verdict") != APTO:
        raise TrialLogError("the holdout opens only after an APTO pre-holdout diagnostic report")
    trial = desc_trial_id(config)
    if report.get("final_trial_id") != trial or not log.has_result(trial):
        raise TrialLogError(f"the diagnostic report is not about {trial}")
    verify_binding(log, config, hash_of_snapshot(snapshot_dir), code_commit)
    columns = registered_columns(log)
    recipe_for(config)
    days = snapshot_days(snapshot_dir, config.series, config.start)
    start = pd.Timestamp(config.holdout_start)
    holdout_days = int((days >= start).sum())
    if holdout_days < MIN_HOLDOUT_DAYS:
        raise TrialLogError(
            f"the snapshot has {holdout_days} holdout days; at least {MIN_HOLDOUT_DAYS} are needed"
        )
    # D5 on the holdout: a model fitted once on everything before it
    frozen_through = replace(config, frozen_train_end=config.holdout_start - timedelta(days=1))

    log.open_holdout(trial)  # recorded before any holdout row is used
    curve = load_curve(
        snapshot_dir, config.series, config.start, config.holdout_start, final_evaluation=True
    )
    analysis = restrict(analyse(prepare(curve, frozen_through, columns)), start)
    checks = checks_of(analysis, curve, config)
    directory = trials_dir / HOLDOUT_DIR
    files = save_analysis(analysis, directory)
    payload = {
        "stage": "holdout",
        "verdict": verdict(checks),
        "final_trial_id": trial,
        "checks": [asdict(c) for c in checks],
        "days": int(len(analysis.labels)),
        "calibration": calibration(analysis.labels, analysis.proba, CALIBRATION_BINS),
        "files": files,
        "limits": list(LIMITS),
        "disclosure": disclosure(log),
    }
    log.record_holdout_result(payload)
    write_report("holdout", "descriptive holdout", "holdout, clean data", payload, reports_dir)
    return str(payload["verdict"])
```

Note: `prepare(curve, frozen_through, columns)` is called with a configuration whose only difference from the registered one is `frozen_train_end`; `analyse` reads thresholds from `data.config`, which is fine because `checks_of(analysis, curve, config)` uses the registered `config`.

- [ ] **Step 4: Run the suite, lint, types** — all pass.

- [ ] **Step 5: Commit**

```bash
git add src/termo/descriptive/stages.py tests/test_desc_stages.py
git commit -m "feat: one-shot descriptive holdout, checked before opening

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: The weekly reading and the command line

**Files:**
- Create: `src/termo/reading.py`, `src/termo/desc_cli.py`
- Create: `tests/test_reading.py`, `tests/test_desc_cli.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_reading.py`:

```python
"""A reading is a lookup in hashed outputs: phase, confidence, drivers, disclaimer."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.descriptive.stages import DISCLAIMER, Analysis
from termo.reading import build_reading, render_reading, write_reading

BLOCKS = ["nivel corto", "nivel medio", "nivel largo", "pendientes", "curvatura", "volatilidad"]


def _analysis() -> Analysis:
    days = pd.bdate_range("1997-01-01", periods=30)
    labels = pd.Series([2] * 10 + [0] * 20, index=days)
    proba = pd.DataFrame({0: 0.7, 1: 0.2, 2: 0.1}, index=days)
    proba.iloc[:10] = [0.1, 0.2, 0.7]
    proba.iloc[25] = [0.45, 0.40, 0.15]  # phase 0 still first, but below the confidence line
    proba.iloc[26] = [0.30, 0.60, 0.10]  # the surrogate prefers another phase
    blocks = pd.DataFrame(0.0, index=days, columns=[*BLOCKS, "base"])
    blocks["nivel corto"], blocks["pendientes"], blocks["volatilidad"] = 1.5, -0.8, 0.1
    blocks["curvatura"] = 0.3
    top = pd.DataFrame(
        {"top": json.dumps([["d2_63_r252", 0.9], ["s3s10_21_r126", -0.4]])}, index=days
    )
    return Analysis(labels=labels, frozen_labels=labels, proba=proba, shap_blocks=blocks, shap_top=top)


def test_reading_reports_phase_run_confidence_and_drivers(desc_config: CoreConfig) -> None:
    analysis = _analysis()
    day = analysis.labels.index[20]
    reading = build_reading(analysis, day.date(), desc_config, {"diagnostic": "apto", "holdout": None})
    assert reading["date"] == day.date().isoformat()
    assert reading["phase"] == 0 and reading["phase_name"] == "rally de la parte corta"
    assert reading["days_in_phase"] == 11
    assert reading["episode_start"] == analysis.labels.index[10].date().isoformat()
    assert reading["probabilities"] == {
        "rally de la parte corta": 0.7, "rally de la parte larga": 0.2, "venta": 0.1
    }
    assert reading["confidence"] == 0.7 and reading["low_confidence"] is False
    assert [d["block"] for d in reading["drivers"]] == ["nivel corto", "pendientes", "curvatura"]
    assert reading["drivers"][1]["contribution"] == -0.8
    assert reading["top_variables"][0] == {"variable": "d2_63_r252", "contribution": 0.9}
    assert reading["validation"] == {"diagnostic": "apto", "holdout": None}
    assert reading["disclaimer"] == DISCLAIMER


def test_low_confidence_is_flagged_for_a_weak_or_disagreeing_surrogate(
    desc_config: CoreConfig,
) -> None:
    analysis = _analysis()
    status = {"diagnostic": "apto", "holdout": None}
    weak = build_reading(analysis, analysis.labels.index[25].date(), desc_config, status)
    assert weak["confidence"] == 0.45 and weak["low_confidence"] is True
    other = build_reading(analysis, analysis.labels.index[26].date(), desc_config, status)
    assert other["low_confidence"] is True and other["surrogate_agrees"] is False


def test_a_date_without_a_reading_is_refused(desc_config: CoreConfig) -> None:
    analysis = _analysis()
    with pytest.raises(KeyError, match="no reading"):
        build_reading(analysis, date(2030, 1, 1), desc_config, {"diagnostic": "apto", "holdout": None})


def test_markdown_is_ascii_and_never_speaks_about_what_comes_next(
    desc_config: CoreConfig, tmp_path: Path
) -> None:
    analysis = _analysis()
    reading = build_reading(
        analysis, analysis.labels.index[20].date(), desc_config, {"diagnostic": "apto", "holdout": None}
    )
    text = render_reading(reading)
    assert text.isascii() and DISCLAIMER in text
    assert "rally de la parte corta" in text and "nivel corto" in text and "+1.50" in text
    for word in ("forecast", "expected move", "next month", "will "):
        assert word not in text.lower()
    write_reading(reading, tmp_path)
    stem = tmp_path / reading["date"]
    assert json.loads(stem.with_suffix(".json").read_text(encoding="utf-8")) == reading
    assert stem.with_suffix(".md").read_text(encoding="utf-8") == text
    assert not np.isnan(reading["confidence"])
```

`tests/test_desc_cli.py`:

```python
"""The spec 3 command line on the synthetic curve."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import fake_fred
from termo.core import take_snapshot
from termo.desc_cli import main
from termo.validation.trials import TrialLogError
from test_cli import EXP2_SMALL_CONFIG

DESC_SMALL_CONFIG = (
    EXP2_SMALL_CONFIG.replace("grid: {k: [2], jump_penalty: [0.5, 3]}", "grid: {k: [3], jump_penalty: [0.5]}")
    .replace("frozen_train_end: 1995-12-31", "frozen_train_end: 1996-06-28")
    + """\
descriptive:
  phase_names: [rally de la parte corta, rally de la parte larga, venta]
  sell_phase: 2
  short_led_phase: 0
  long_led_phase: 1
  level_series: DGS10
  slope_long: DGS10
  slope_short: DGS2
  change_days: 63
  coherence_share_min: 0.6
  min_days_evaluable: 40
  map_ari_min: 0.6
  fidelity_min: 0.8
  low_confidence_below: 0.6
  surrogate: {n_estimators: 20, max_depth: 3, learning_rate: 0.1, subsample: 0.8, colsample_bytree: 0.8, seed: 0}
  blocks:
    nivel corto: [d1, d2, d3]
    nivel medio: [d5, d7]
    nivel largo: [d10, d30]
    pendientes: [s12m5s, s3s10, s10s30]
    curvatura: [c5]
    volatilidad: [vol1, vol2, vol3, vol5, vol7, vol10, vol30]
"""
)


def test_stages_and_reading_through_the_command_line(
    tmp_path: Path,
    curve: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert "k: [3]" in DESC_SMALL_CONFIG and "1996-06-28" in DESC_SMALL_CONFIG
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "desc.yaml"
    config_path.write_text(DESC_SMALL_CONFIG, encoding="utf-8")
    monkeypatch.setattr("termo.desc_cli.code_identity", lambda: "code-3")
    from termo.config import load_config

    snapshot = tmp_path / "data" / "snapshots" / "2026-10-02"
    take_snapshot(load_config(config_path), snapshot, fake_fred(curve), "2026-10-02T00:00:00+00:00")
    stage = ["--config", str(config_path), "--snapshot", str(snapshot)]

    assert main(["register", *stage]) == 0
    assert main(["run", *stage]) == 0
    assert main(["report", *stage]) == 0
    assert (tmp_path / "trials" / "desc" / "pre_holdout" / "proba.csv").exists()
    assert (tmp_path / "reports" / "desc" / "diagnostic.md").exists()

    assert main(["read", *stage, "--date", "1997-06-02"]) == 0
    reading = tmp_path / "reports" / "desc" / "readings" / "1997-06-02.md"
    assert reading.exists() and "does not anticipate" in reading.read_text(encoding="utf-8")
    assert capsys.readouterr().out.isascii()

    # a holdout date has no reading until a holdout result exists
    with pytest.raises(TrialLogError, match="no validated reading"):
        main(["read", *stage, "--date", "1998-06-01"])
    # the holdout does not open without the explicit flag
    with pytest.raises(SystemExit):
        main(["holdout", *stage])
    assert not any("holdout_opened" in line for line in (tmp_path / "trials" / "desc" / "trials.jsonl").read_text(encoding="utf-8").splitlines())
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reading.py tests/test_desc_cli.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'termo.reading'`.

- [ ] **Step 3: Implement the reading**

`src/termo/reading.py`:

```python
"""The weekly reading: what phase the curve is in on a date, how sure, and why.

A reading is a lookup in the hashed outputs of the run: it refits nothing and it
never contains a number about what happened after its date.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from termo.config import CoreConfig
from termo.descriptive.stages import BASE, DISCLAIMER, Analysis

DRIVERS = 3


def build_reading(
    analysis: Analysis, day: date, config: CoreConfig, validation: dict[str, str | None]
) -> dict[str, Any]:
    desc = config.descriptive
    if desc is None:
        raise ValueError("the configuration has no descriptive section")
    stamp = pd.Timestamp(day)
    if stamp not in analysis.labels.index:
        raise KeyError(f"no reading for {day.isoformat()}: not a day with data in this period")
    past = analysis.labels.loc[:stamp]
    phase = int(past.iloc[-1])
    changed = past.ne(phase)
    start = past.index[0] if not changed.any() else past.index[changed.to_numpy().nonzero()[0][-1] + 1]
    proba = analysis.proba.loc[stamp]
    confidence = float(proba[phase])
    agrees = int(proba.idxmax()) == phase
    blocks = analysis.shap_blocks.loc[stamp].drop(BASE)
    order = blocks.abs().sort_values(ascending=False, kind="stable").index[:DRIVERS]
    top = json.loads(str(analysis.shap_top.loc[stamp, "top"]))
    return {
        "date": day.isoformat(),
        "phase": phase,
        "phase_name": desc.phase_names[phase],
        "days_in_phase": int(len(past.loc[start:])),
        "episode_start": start.date().isoformat(),
        "probabilities": {desc.phase_names[int(p)]: float(proba[p]) for p in proba.index},
        "confidence": confidence,
        "surrogate_agrees": agrees,
        "low_confidence": bool(confidence < desc.low_confidence_below or not agrees),
        "drivers": [{"block": str(b), "contribution": float(blocks[b])} for b in order],
        "top_variables": [{"variable": str(n), "contribution": float(v)} for n, v in top],
        "validation": dict(validation),
        "disclaimer": DISCLAIMER,
    }


def render_reading(reading: dict[str, Any]) -> str:
    lines = [
        f"# TERMO - reading of {reading['date']}",
        "",
        f"**Phase: {reading['phase_name']}**",
        "",
        f"- In this phase since {reading['episode_start']} ({reading['days_in_phase']} trading days).",
        f"- Confidence (surrogate probability of this phase): {reading['confidence']:.2f}"
        + ("  [LOW CONFIDENCE]" if reading["low_confidence"] else ""),
        "",
        "| Phase | Probability |",
        "|---|---|",
    ]
    lines += [f"| {name} | {value:.2f} |" for name, value in reading["probabilities"].items()]
    lines += ["", "## What pushes toward this phase (log-odds of the surrogate)", ""]
    lines += [f"- {d['block']}: {d['contribution']:+.2f}" for d in reading["drivers"]]
    lines += ["", "Variables with the largest contribution:", ""]
    lines += [f"- {v['variable']}: {v['contribution']:+.2f}" for v in reading["top_variables"]]
    status = reading["validation"]
    lines += [
        "",
        "## Validation",
        "",
        f"- Pre-holdout diagnostic: {status['diagnostic']}",
        f"- Holdout: {status['holdout'] or 'not opened'}",
        "",
        reading["disclaimer"],
        "",
    ]
    return "\n".join(lines)


def write_reading(reading: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / reading["date"]
    stem.with_suffix(".md").write_text(render_reading(reading), encoding="utf-8", newline="\n")
    stem.with_suffix(".json").write_text(
        json.dumps(reading, indent=2), encoding="utf-8", newline="\n"
    )
```

- [ ] **Step 4: Implement the command line**

`src/termo/desc_cli.py`:

```python
"""Command line for spec 3 (descriptive tool). Console output is ASCII only."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from termo.cli import code_identity
from termo.config import load_config
from termo.core import registered_columns, verify_binding
from termo.data.loader import load_curve
from termo.data.snapshot import snapshot_hash
from termo.dataset import prepare
from termo.descriptive.stages import (
    HOLDOUT_DIR,
    PRE_HOLDOUT_DIR,
    desc_trial_id,
    holdout_desc,
    load_analysis,
    register_desc,
    report_desc,
    run_desc,
)
from termo.reading import build_reading, write_reading
from termo.validation.trials import HOLDOUT_TRIAL, RecordKind, TrialLog, TrialLogError

TRIALS_DIR = Path("trials") / "desc"
REPORTS_DIR = Path("reports") / "desc"
TRIALS_FILE = "trials.jsonl"
APPROVAL_FLAG = "--i-approve-opening-the-holdout"


def _holdout_result(log: TrialLog) -> dict[str, object] | None:
    found = [r for r in log.records() if r["kind"] == RecordKind.HOLDOUT_RESULT.value]
    return found[-1] if found else None


def _read(args: argparse.Namespace, log: TrialLog, data_hash: str, commit: str) -> Path:
    config = load_config(args.config)
    verify_binding(log, config, data_hash, commit)
    day = date.fromisoformat(args.date)
    report = log.last_report()
    trial = desc_trial_id(config)
    if report is None or trial not in log.results():
        raise TrialLogError("no validated reading: run the run and report stages first")
    holdout = _holdout_result(log)
    status = {
        "diagnostic": str(report["verdict"]),
        "holdout": None if holdout is None else str(holdout["verdict"]),
    }
    if pd.Timestamp(day) < pd.Timestamp(config.holdout_start):
        files = log.results()[trial]["metrics"]["files"]
        analysis = load_analysis(TRIALS_DIR / PRE_HOLDOUT_DIR, files)
    elif holdout is not None:
        analysis = load_analysis(TRIALS_DIR / HOLDOUT_DIR, holdout["files"])  # type: ignore[arg-type]
    else:
        raise TrialLogError("no validated reading for that date: the holdout has no result")
    try:
        reading = build_reading(analysis, day, config, status)
    except KeyError as error:
        raise TrialLogError(f"no validated reading for that date: {error.args[0]}") from error
    out = REPORTS_DIR / "readings"
    write_reading(reading, out)
    return out / f"{reading['date']}.md"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="termo-desc", description="TERMO: descriptive tool")
    parser.add_argument("stage", choices=["register", "run", "report", "holdout", "read"])
    parser.add_argument("--config", type=Path, default=Path("configs/desc.yaml"))
    parser.add_argument("--snapshot", type=Path, required=True, help="snapshot directory")
    parser.add_argument("--date", help="reading date, YYYY-MM-DD (stage read)")
    parser.add_argument(
        APPROVAL_FLAG,
        dest="approved",
        action="store_true",
        help="required by the holdout stage: it runs once and cannot be repeated",
    )
    args = parser.parse_args(argv)
    if args.stage == "read" and args.date is None:
        parser.error("--date is required for the read stage")
    if args.stage == "holdout" and not args.approved:
        parser.error(f"the holdout runs once and cannot be repeated: pass {APPROVAL_FLAG}")

    config = load_config(args.config)
    log = TrialLog(TRIALS_DIR / TRIALS_FILE)
    data_hash, commit = snapshot_hash(args.snapshot), code_identity()

    if args.stage == "read":
        path = _read(args, log, data_hash, commit)
        print(f"[ok] reading -> {path.as_posix()}")
        return 0
    if args.stage == "holdout":
        result = holdout_desc(config, args.snapshot, log, TRIALS_DIR, REPORTS_DIR, commit)
        print(f"[ok] holdout verdict: {result} -> {REPORTS_DIR.as_posix()}/holdout.md")
        return 0

    curve = load_curve(args.snapshot, config.series, config.start, config.holdout_start)
    if args.stage == "register":
        columns = register_desc(config, curve, log, data_hash, commit)
        print(f"[ok] registered {desc_trial_id(config)} with {len(columns)} features")
        return 0
    verify_binding(log, config, data_hash, commit)  # before any computation
    data = prepare(curve, config, registered_columns(log))
    if args.stage == "run":
        run_desc(data, log, TRIALS_DIR, data_hash, commit)
        print("[ok] the registered trial has a result")
    else:
        result = report_desc(data, log, TRIALS_DIR, REPORTS_DIR, data_hash, commit)
        print(f"[ok] diagnostic verdict: {result} -> {REPORTS_DIR.as_posix()}/diagnostic.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Notes for the implementer:
- `HOLDOUT_TRIAL` may be unused in this module; remove the import if ruff flags it.
- The `test_desc_cli` configuration string is built from `EXP2_SMALL_CONFIG` in `tests/test_cli.py`: check that the two `.replace` calls actually change the text (the test asserts it); if the grid line in that string is spelled differently, adjust the replaced substring, not the meaning.
- Importing `test_cli` from another test module works because pytest puts `tests/` on `sys.path` (rootdir conftest); if it does not, copy the small YAML into this test instead.

- [ ] **Step 5: Run the suite, lint, types** — all pass.

- [ ] **Step 6: Commit**

```bash
git add src/termo/reading.py src/termo/desc_cli.py tests/test_reading.py tests/test_desc_cli.py
git commit -m "feat: weekly reading and the descriptive command line

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Register spec 3 (log before you look)

- [ ] **Step 1:** `git status --porcelain -- src configs pyproject.toml` is empty; the full suite is green.

- [ ] **Step 2: Register**

```bash
.venv/Scripts/python.exe -m termo.desc_cli register --config configs/desc.yaml --snapshot data/snapshots/2026-10-02
```

Expected: `[ok] registered desc_k3 with 139 features`. Check: `trials/desc/trials.jsonl` has 2 registrations; its setup record lists both prior logs (29 trials no-go, 33 trials no-go); `trials/trials.jsonl` and `trials/exp2/trials.jsonl` are unchanged (`git status`).

- [ ] **Step 3: Commit**

```bash
git add trials/desc/trials.jsonl
git commit -m "trials: register the descriptive tool before any run

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Run and report the pre-holdout diagnostic

- [ ] **Step 1: Run** (about 10-20 minutes: 74 jump model fits, 74 surrogates, SHAP for ~9,200 days)

```bash
.venv/Scripts/python.exe -m termo.desc_cli run --config configs/desc.yaml --snapshot data/snapshots/2026-10-02
```

Expected: `[ok] desc_k3 diagnostic=apto|no-apto days=9194`. Sanity check that does not depend on the verdict: `trials/desc/pre_holdout/labels.csv` must be identical to `trials/exp2/labels/jm_k3_lam3.csv` (same model, same data, same code path).

- [ ] **Step 2: Commit the outputs**, then **Step 3: Report**

```bash
git add trials/desc && git commit -m "trials: record the descriptive diagnostic outputs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
.venv/Scripts/python.exe -m termo.desc_cli report --config configs/desc.yaml --snapshot data/snapshots/2026-10-02
```

- [ ] **Step 4: Commit the report and the context files** (`reports/desc`, `trials/desc`, `docs/context`).

- [ ] **Step 5: STOP.** Present the diagnostic to the user. The holdout stage is run only with the user's explicit approval, and only if the diagnostic is APTO:

```bash
.venv/Scripts/python.exe -m termo.desc_cli holdout --config configs/desc.yaml --snapshot data/snapshots/2026-10-02 --i-approve-opening-the-holdout
```

---

## Self-review

- **Spec coverage.** §2 fixed model: `configs/desc.yaml` (Task 5), sanity check against `jm_k3_lam3` labels (Task 11). §3.1 surrogate: Task 3. §3.2 confidence, low-confidence flag, calibration: Tasks 6, 9. §3.3 SHAP by blocks, top variables: Tasks 4, 7. §4 D1-D6, not-evaluable rule, verdict, holdout only after an APTO diagnostic: Tasks 6, 7, 8. §5 separate log, 1 trial, disclosure of 63, bindings with xgboost/shap, stages: Tasks 1, 7, 8, 10. §6 reading, refusal of holdout dates, disclaimer, nothing about later moves: Task 9. §7 tests: each task.
- **Placeholders.** None.
- **Type consistency.** `WalkForwardResult.fits: tuple[RefitLabels, ...]`; `surrogate_walk(refits, fits, n_states, params) -> SurrogateWalk(proba, surrogates)`; `explain(surrogate, features, phases) -> Explanation(values, base)`; `block_sums(values, blocks)` with `blocks` = `DescriptiveConfig.blocks`; `evaluate(labels, curve, proba, frozen_labels, config) -> tuple[Check, ...]`; `Analysis(labels, frozen_labels, proba, shap_blocks, shap_top)` used by stages, reading and CLI; `FILES` names match `save_analysis` / `load_analysis`; `BASE` column name shared by stages and reading.
