"""Probability of backtest overfitting (CSCV) and the effective number of trials."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from termo.validation.metrics import VARIANCE_FLOOR, adjusted_rand


def block_stats(values: np.ndarray, groups: np.ndarray, n_states: int, n_blocks: int) -> np.ndarray:
    """Per contiguous block and state: count, sum and sum of squares of `values`.

    Shape (n_blocks, n_states, 3). These add up across blocks, so eta-squared of any
    set of blocks can be computed without touching the rows again.
    """
    edges = np.linspace(0, len(values), n_blocks + 1).astype(int)
    stats = np.zeros((n_blocks, n_states, 3))
    for block in range(n_blocks):
        block_values = values[edges[block] : edges[block + 1]]
        block_groups = groups[edges[block] : edges[block + 1]]
        for state in range(n_states):
            member = block_values[block_groups == state]
            stats[block, state] = (len(member), member.sum(), (member**2).sum())
    return stats


def eta_from_stats(stats: np.ndarray) -> np.ndarray:
    """Eta-squared from stats of shape (..., n_states, 3), summed over the blocks in use."""
    count, total, squares = stats[..., 0], stats[..., 1], stats[..., 2]
    grand = total.sum(axis=-1) ** 2 / count.sum(axis=-1)
    ss_total = squares.sum(axis=-1) - grand
    with np.errstate(divide="ignore", invalid="ignore"):
        per_state = np.where(count > 0, total**2 / count, 0.0)
        eta = (per_state.sum(axis=-1) - grand) / ss_total
    constant = ss_total <= VARIANCE_FLOOR * np.maximum(1.0, squares.sum(axis=-1))
    result: np.ndarray = np.where(constant, 0.0, eta)
    return result


def pbo_cscv(stats: np.ndarray) -> float:
    """Share of block splits where the in-sample winner lands in the bottom half out of sample.

    `stats` has shape (n_configs, n_blocks, n_states, 3) from `block_stats`.
    """
    n_configs, n_blocks = stats.shape[0], stats.shape[1]
    if n_blocks % 2 != 0:
        raise ValueError("the number of blocks must be even")
    splits = list(combinations(range(n_blocks), n_blocks // 2))
    inside = np.zeros((len(splits), n_blocks))
    for row, split in enumerate(splits):
        inside[row, list(split)] = 1.0
    eta_in = eta_from_stats(np.einsum("cb,nbsk->cnsk", inside, stats))
    eta_out = eta_from_stats(np.einsum("cb,nbsk->cnsk", 1.0 - inside, stats))
    winner = eta_in.argmax(axis=1)
    winner_out = eta_out[np.arange(len(splits)), winner]
    # Mid-rank: configurations that tie with the winner count half. 1 = worst, n_configs = best.
    below = (eta_out < winner_out[:, None]).sum(axis=1)
    tied = (eta_out == winner_out[:, None]).sum(axis=1)  # includes the winner itself
    omega = (below + (tied + 1) / 2.0) / (n_configs + 1.0)
    logit = np.log(omega / (1.0 - omega))
    # Exactly at the median counts half, so identical configurations give 0.5, not 0 or 1.
    return float((logit < 0.0).mean() + 0.5 * np.isclose(logit, 0.0).mean())


def effective_n(labelings: Sequence[np.ndarray], cut: float) -> int:
    """Number of clusters of label series under distance 1 - ARI (average linkage)."""
    if len(labelings) < 2:
        return len(labelings)
    distances = [
        max(0.0, 1.0 - adjusted_rand(labelings[i], labelings[j]))
        for i in range(len(labelings))
        for j in range(i + 1, len(labelings))
    ]
    clusters = fcluster(linkage(np.array(distances), method="average"), t=cut, criterion="distance")
    return int(len(set(clusters.tolist())))
