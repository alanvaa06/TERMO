"""Synthetic yield curves with planted regimes. No real market data in the tests."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
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
    from termo.validation.trials import TrialLog

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


def make_macro(curve: pd.DataFrame, seed: int = 5) -> pd.DataFrame:
    """Two context series on the curve's calendar plus weekends for DFF, as FRED serves them."""
    rng = np.random.default_rng(seed)
    premium = pd.Series(np.cumsum(0.01 * rng.normal(size=len(curve))) + 1.0, index=curve.index)
    daily = pd.date_range(curve.index[0], curve.index[-1], freq="D")
    fed_funds = pd.Series(
        np.round(3.0 + np.cumsum(0.002 * rng.normal(size=len(daily))), 2), index=daily
    )
    return pd.DataFrame({"THREEFYTP10": premium, "DFF": fed_funds})


def fake_fred_frames(frames: Sequence[pd.DataFrame]) -> Callable[[str], str]:
    """An HTTP getter serving several frames, one FRED CSV per column; NaN rows are blank."""

    def get(url: str) -> str:
        for frame in frames:
            for series_id in frame.columns:
                if url == FRED_URL_TEMPLATE.format(series_id=series_id):
                    column = frame[series_id].dropna()
                    lines = [f"observation_date,{series_id}"]
                    lines += [f"{d:%Y-%m-%d},{v:.4f}" for d, v in column.items()]
                    return "\n".join(lines) + "\n"
        raise AssertionError(f"unexpected url {url}")

    return get


DESC2_COMMIT = "test-commit-desc2"
SHADOW_DAYS = 60  # business days the full curve has after the registered history


@dataclass(frozen=True)
class Desc2Registry:
    """A finished desc2 registry: registered, run, reported and holdout done, on `curve`."""

    config: CoreConfig
    log: TrialLog
    snapshot_dir: Path  # the registered snapshot: the curve the registry saw
    trials_dir: Path
    reports_dir: Path
    curve: pd.DataFrame
    commit: str

    def copy_to(self, root: Path) -> Desc2Registry:
        """A private copy of the trials and reports (the snapshot is shared, read only)."""
        from termo.validation.trials import TrialLog

        trials, reports = root / "trials" / "desc2", root / "reports" / "desc2"
        shutil.copytree(self.trials_dir, trials)
        shutil.copytree(self.reports_dir, reports)
        return replace(
            self, log=TrialLog(trials / self.log.path.name), trials_dir=trials, reports_dir=reports
        )


def build_desc2_registry(
    root: Path, curve: pd.DataFrame, config: CoreConfig, commit: str = DESC2_COMMIT
) -> Desc2Registry:
    """The four stages of spec 3 on `curve`, with the stage functions, in `root`.

    The synthetic curve fails the real criteria, so the diagnostic is approved by a forged
    APTO report before the holdout opens (as tests/test_desc_stages.py does).
    """
    from termo.core import registered_columns, take_snapshot
    from termo.data.loader import load_curve
    from termo.data.snapshot import snapshot_hash
    from termo.dataset import prepare
    from termo.descriptive.criteria import APTO
    from termo.descriptive.stages import holdout_desc, register_desc, report_desc, run_desc
    from termo.validation.trials import TrialLog

    snapshot_dir = root / "data" / "snapshots" / "registered"
    take_snapshot(config, snapshot_dir, fake_fred(curve), "2026-10-02T00:00:00+00:00")
    trials, reports = root / "trials" / "desc2", root / "reports" / "desc2"
    log = TrialLog(trials / "trials.jsonl")
    data_hash = snapshot_hash(snapshot_dir)
    pre = load_curve(snapshot_dir, config.series, config.start, config.holdout_start)
    register_desc(config, pre, log, data_hash, commit)
    data = prepare(pre, config, registered_columns(log))
    run_desc(data, log, trials, data_hash, commit, echo=lambda _: None)
    report_desc(data, log, trials, reports, data_hash, commit)
    last = log.last_report()
    assert last is not None
    if last["verdict"] != APTO:
        payload = {k: v for k, v in last.items() if k not in {"kind", "trial_id", "at"}}
        log.record_report({**payload, "verdict": APTO})
    holdout_desc(config, snapshot_dir, log, trials, reports, commit)
    return Desc2Registry(config, log, snapshot_dir, trials, reports, curve, commit)


@pytest.fixture(scope="session")
def desc2_registry(
    tmp_path_factory: pytest.TempPathFactory, curve: pd.DataFrame, desc2_config: CoreConfig
) -> Desc2Registry:
    """The registry built on the curve without its last SHADOW_DAYS: what the shadow extends.

    Tests that run the shadow stage work on `copy_to` copies: this one is never written.
    """
    root = tmp_path_factory.mktemp("termo-desc2-registry")
    return build_desc2_registry(root, curve.iloc[:-SHADOW_DAYS], desc2_config)
