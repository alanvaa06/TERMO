"""Keep regime names stable across refits. A permutation maps old label -> new label."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def order_by_target(labels: np.ndarray, target: np.ndarray, n_states: int) -> np.ndarray:
    """Name states by the mean of `target` inside each one, lowest first; empty states last."""
    means = np.full(n_states, np.inf)
    for state in range(n_states):
        mask = labels == state
        if mask.any():
            means[state] = float(target[mask].mean())
    permutation = np.empty(n_states, dtype=int)
    permutation[np.argsort(means, kind="stable")] = np.arange(n_states)
    return permutation


def match_to_reference(reference: np.ndarray, labels: np.ndarray, n_states: int) -> np.ndarray:
    """Rename `labels` so that they agree with `reference` on as many rows as possible."""
    if reference.shape != labels.shape:
        raise ValueError("reference and labels must cover the same rows")
    overlap = np.zeros((n_states, n_states), dtype=int)
    np.add.at(overlap, (labels, reference), 1)
    old, new = linear_sum_assignment(-overlap)
    permutation = np.empty(n_states, dtype=int)
    permutation[old] = new
    return permutation


def apply_permutation(labels: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    renamed: np.ndarray = permutation[labels]
    return renamed
