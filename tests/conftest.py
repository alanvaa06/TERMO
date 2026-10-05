"""Synthetic yield curves with planted regimes. No real market data in the tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from termo.config import (
    BootstrapConfig,
    CoreConfig,
    DescriptiveConfig,
    FticConfig,
    SurrogateConfig,
    Thresholds,
    TycclesConfig,
)

if TYPE_CHECKING:
    from termo.dataset import ExperimentData

FRED_URL_TEMPLATE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def fake_fred(curve: pd.DataFrame) -> Callable[[str], str]:
    """An HTTP getter that serves `curve` in FRED's CSV format, one series per URL."""

    def get(url: str) -> str:
        for series_id in curve.columns:
            if url == FRED_URL_TEMPLATE.format(series_id=series_id):
                lines = [f"observation_date,{series_id}"]
                lines += [f"{d:%Y-%m-%d},{v:.4f}" for d, v in curve[series_id].items()]
                return "\n".join(lines) + "\n"
        raise AssertionError(f"unexpected url {url}")

    return get


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


@pytest.fixture(scope="session")
def tyccles_config() -> CoreConfig:
    return make_tyccles_config()


@pytest.fixture(scope="session")
def tyccles_data(pre_holdout: pd.DataFrame, tyccles_config: CoreConfig) -> ExperimentData:
    from termo.dataset import prepare, recipe_for

    return prepare(pre_holdout, tyccles_config, recipe_for(tyccles_config).names)


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


def make_desc2_config() -> CoreConfig:
    """The post-holdout re-registration: names by direction, no curve-shape claim (no D4)."""
    config = make_desc_config()
    assert config.descriptive is not None
    return replace(
        config,
        descriptive=replace(
            config.descriptive,
            phase_names=("rally fuerte", "rally moderado", "venta"),
            short_led_phase=None,
            long_led_phase=None,
            slope_long=None,
            slope_short=None,
            holdout_already_seen=True,
            holdout_seen_by="desc_k3 (test)",
        ),
    )


@pytest.fixture(scope="session")
def desc2_config() -> CoreConfig:
    return make_desc2_config()
