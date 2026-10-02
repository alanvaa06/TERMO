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
