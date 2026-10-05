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


def test_descriptive_series_must_exist_and_slope_legs_differ() -> None:
    config = load_config(DESC_CONFIG)
    assert config.descriptive is not None
    for change in ({"level_series": "DGS1O"}, {"slope_short": "DGS20"}, {"slope_short": "DGS10"}):
        with pytest.raises(ValueError, match="series of the curve"):
            replace(config, descriptive=replace(config.descriptive, **change))
    with pytest.raises(ValueError, match="different from each other"):
        replace(config, descriptive=replace(config.descriptive, phase_names=("a", "a", "b")))


def test_unknown_keys_in_the_descriptive_section_are_refused(tmp_path: Path) -> None:
    text = DESC_CONFIG.read_text(encoding="utf-8")
    for old, new in (
        ("  change_days: 63", "  chnge_days: 63" + chr(10) + "  change_days: 63"),
        ("    seed: 0", "    seed: 0" + chr(10) + "    tree_method: exact"),
    ):
        assert old in text
        path = tmp_path / "typo.yaml"
        path.write_text(text.replace(old, new), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown keys"):
            load_config(path)


DESC2_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "desc2.yaml"


def test_loads_the_post_holdout_re_registration() -> None:
    config = load_config(DESC2_CONFIG)
    desc = config.descriptive
    assert desc is not None
    assert desc.phase_names == ("rally fuerte", "rally moderado", "venta")
    assert desc.sell_phase == 2 and desc.level_series == "DGS10"
    assert desc.has_curve_shape_claim is False
    assert (desc.short_led_phase, desc.long_led_phase) == (None, None)
    assert (desc.slope_long, desc.slope_short) == (None, None)
    assert desc.holdout_already_seen is True
    assert desc.holdout_seen_by == (
        "desc_k3 (trials/desc), opened 2026-10-05, verdict no-apto by D4 only"
    )
    assert config.prior_trial_logs == (
        "trials/trials.jsonl",
        "trials/exp2/trials.jsonl",
        "trials/desc/trials.jsonl",
    )
    text = DESC2_CONFIG.read_text(encoding="utf-8").lower()
    assert "section 9" in text and "after seeing the holdout" in text and text.isascii()


def test_desc2_differs_from_desc_only_in_names_claim_and_holdout_bookkeeping() -> None:
    base, config = load_config(DESC_CONFIG), load_config(DESC2_CONFIG)
    assert base.descriptive is not None and config.descriptive is not None
    assert base.descriptive.has_curve_shape_claim and base.descriptive.holdout_already_seen is False
    aligned = replace(
        config,
        prior_trial_logs=base.prior_trial_logs,
        descriptive=replace(
            config.descriptive,
            phase_names=base.descriptive.phase_names,
            short_led_phase=base.descriptive.short_led_phase,
            long_led_phase=base.descriptive.long_led_phase,
            slope_long=base.descriptive.slope_long,
            slope_short=base.descriptive.slope_short,
            holdout_already_seen=base.descriptive.holdout_already_seen,
            holdout_seen_by=base.descriptive.holdout_seen_by,
        ),
    )
    assert aligned == base


def test_the_curve_shape_claim_is_all_four_fields_or_none() -> None:
    config = load_config(DESC_CONFIG)
    assert config.descriptive is not None
    whole: dict[str, None] = {
        "short_led_phase": None,
        "long_led_phase": None,
        "slope_long": None,
        "slope_short": None,
    }
    without = replace(config.descriptive, **whole)
    assert replace(config, descriptive=without).descriptive is not None
    for partial in (
        {"slope_short": None},
        {"short_led_phase": None},
        {"short_led_phase": None, "long_led_phase": None, "slope_long": None},
    ):
        with pytest.raises(ValueError, match="all four or none"):
            replace(config, descriptive=replace(config.descriptive, **partial))
    # without the claim the sell phase and the level series are still checked
    with pytest.raises(ValueError, match="sell phase"):
        replace(config, descriptive=replace(without, sell_phase=3))
    with pytest.raises(ValueError, match="series of the curve"):
        replace(config, descriptive=replace(without, level_series="DGS1O"))


def test_unknown_keys_in_desc2_are_refused(tmp_path: Path) -> None:
    text = DESC2_CONFIG.read_text(encoding="utf-8")
    old = "  holdout_already_seen: true"
    assert old in text
    path = tmp_path / "typo.yaml"
    path.write_text(text.replace(old, old + chr(10) + "  holdout_seen: true"), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)


def test_every_registered_variable_falls_in_exactly_one_block() -> None:
    from collections import Counter

    from termo.dataset import recipe_for
    from termo.surrogate.explain import block_map

    config = load_config(DESC_CONFIG)
    assert config.descriptive is not None
    mapping = block_map(recipe_for(config).names, config.descriptive.blocks)
    assert Counter(mapping.values()) == {
        "nivel corto": 36,
        "nivel medio": 24,
        "nivel largo": 24,
        "pendientes": 36,
        "curvatura": 12,
        "volatilidad": 7,
    }
    assert config.descriptive.surrogate.subsample == 0.8
    assert config.descriptive.surrogate.colsample_bytree == 0.8
