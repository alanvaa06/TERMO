from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.dataset import ExperimentData, prepare, recipe_for
from termo.experiment import (
    ConfigEvaluation,
    effective_penalty,
    evaluate_config,
    gate_tests,
    holdout_evaluation,
    pbo_of,
    saturated_wcss,
    separation,
    separation_frame,
)
from termo.features.pipeline import FEATURE_NAMES
from termo.regime.model import jump_fitter, kmeans_fitter
from termo.validation.metrics import (
    eta_squared,
    excess_eta_squared,
    run_lengths,
    shift_excess_eta_squared,
    weekly_last,
)
from termo.validation.walkforward import inertia_labels, run_walkforward


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
    assert jump.excess_short_shift < jump.eta_short
    assert jump.excess_short_shift != jump.excess_short  # two different chance levels
    metrics = jump.metrics()
    assert metrics["excess_short"] == jump.excess_short
    assert metrics["excess_short_shift"] == jump.excess_short_shift
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


def test_separation_decides_with_the_sign_flip_level(
    jump: ConfigEvaluation, data: ExperimentData
) -> None:
    """Pins which chance level feeds the score and which one is only reported."""
    config, boot = data.config, data.config.bootstrap
    frame = separation_frame(jump.oos_labels, data.yields_10y, config.horizon_short_days)
    values, groups = frame["change"].to_numpy(), frame["label"].to_numpy().astype(int)
    found = separation(jump.oos_labels, data.yields_10y, config.horizon_short_days, config)
    assert found.raw == eta_squared(values, groups)
    assert found.excess == excess_eta_squared(
        values, groups, boot.block_weeks, boot.n_resamples, boot.seed
    )
    assert found.excess_shift == shift_excess_eta_squared(values, groups)
    assert (jump.eta_short, jump.excess_short, jump.excess_short_shift) == (
        found.raw,
        found.excess,
        found.excess_shift,
    )


def test_holdout_is_judged_with_the_sign_flip_level(
    curve: pd.DataFrame, config: CoreConfig
) -> None:
    full = prepare(curve, config, FEATURE_NAMES)
    result = holdout_evaluation(full, 2, 50.0)

    walk = run_walkforward(full.refits, jump_fitter(2, 50.0), 2, full.daily_change_10y)
    labels = walk.oos_labels.loc[pd.Timestamp(config.holdout_start) :]
    frame = separation_frame(labels, full.yields_10y, config.horizon_short_days)
    values = frame["change"].to_numpy()
    model = frame["label"].to_numpy().astype(int)
    inertia = weekly_last(inertia_labels(full.refits)).reindex(frame.index).to_numpy().astype(int)
    boot = config.bootstrap
    draws = (boot.block_weeks, boot.n_resamples, boot.seed)
    assert result.excess_model == excess_eta_squared(values, model, *draws)
    assert result.excess_inertia == excess_eta_squared(values, inertia, *draws)
    assert result.excess_model != shift_excess_eta_squared(values, model)


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
    # behaviour, not only the reported field: the labels are those of lambda = c * p
    target = tyccles_data.daily_change_10y
    scaled = run_walkforward(tyccles_data.refits, jump_fitter(2, 69.5), 2, target).oos_labels
    unscaled = run_walkforward(tyccles_data.refits, jump_fitter(2, 0.5), 2, target).oos_labels
    pd.testing.assert_series_equal(evaluation.oos_labels, scaled)
    assert not evaluation.oos_labels.equals(unscaled)


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
    result = holdout_evaluation(full, 3, None, frozen=True)

    start = pd.Timestamp(tyccles_config.holdout_start)
    walk = run_walkforward(full.frozen_refits, kmeans_fitter(3), 3, full.daily_change_10y)
    labels = walk.oos_labels.loc[start:]
    frame = separation_frame(labels, full.yields_10y, tyccles_config.horizon_short_days)
    baseline = weekly_last(inertia_labels(full.frozen_refits)).reindex(frame.index)
    values = frame["change"].to_numpy()
    boot = tyccles_config.bootstrap
    args = (boot.block_weeks, boot.n_resamples, boot.seed)
    assert result.excess_model == pytest.approx(
        excess_eta_squared(values, frame["label"].to_numpy().astype(int), *args)
    )
    assert result.excess_inertia == pytest.approx(
        excess_eta_squared(values, baseline.to_numpy().astype(int), *args)
    )
    assert result.n_weeks == len(frame) > 0
    assert result.n_episodes == len(run_lengths(labels.to_numpy()))
    # the frozen model is not the refitted K-means: on this curve they read the holdout differently
    assert result.n_episodes != holdout_evaluation(full, 3, None).n_episodes
