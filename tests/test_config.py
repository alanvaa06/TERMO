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
