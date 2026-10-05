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


def test_experiment_2_differs_from_spec_1_only_where_the_spec_says() -> None:
    base, config = load_config(REPO_CONFIG), load_config(EXP2_CONFIG)
    aligned = replace(
        config,
        burn_in_days=base.burn_in_days,
        jump_penalties=base.jump_penalties,
        feature_set=base.feature_set,
        tyccles=base.tyccles,
        apply_collinearity_rule=base.apply_collinearity_rule,
        jump_penalty_per_feature=base.jump_penalty_per_feature,
        frozen_train_end=base.frozen_train_end,
        prior_trial_logs=base.prior_trial_logs,
    )
    assert aligned == base


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
        "nivel corto",
        "nivel medio",
        "nivel largo",
        "pendientes",
        "curvatura",
        "volatilidad",
    ]
    assert dict(desc.blocks)["nivel corto"] == ("d1", "d2", "d3")
    assert dict(desc.blocks)["volatilidad"] == (
        "vol1",
        "vol2",
        "vol3",
        "vol5",
        "vol7",
        "vol10",
        "vol30",
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
