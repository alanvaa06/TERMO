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
