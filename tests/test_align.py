from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from termo.regime.align import apply_permutation, match_to_reference, order_by_target


def test_order_by_target_names_the_lowest_mean_zero() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    target = np.array([5.0, 5.0, -3.0, -3.0, 1.0, 1.0])
    permutation = order_by_target(labels, target, 3)
    assert permutation.tolist() == [2, 0, 1]
    assert apply_permutation(labels, permutation).tolist() == [2, 2, 0, 0, 1, 1]


def test_order_by_target_puts_empty_states_last() -> None:
    labels = np.array([2, 2, 0, 0])
    target = np.array([1.0, 1.0, 4.0, 4.0])
    assert order_by_target(labels, target, 3).tolist() == [1, 2, 0]


def test_match_rejects_different_lengths() -> None:
    with pytest.raises(ValueError):
        match_to_reference(np.array([0, 1]), np.array([0, 1, 1]), 2)


@given(
    st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=200),
    st.permutations([0, 1, 2, 3]),
)
def test_match_undoes_any_renaming(reference: list[int], renaming: list[int]) -> None:
    truth = np.array(reference)
    scrambled = np.array(renaming)[truth]
    permutation = match_to_reference(truth, scrambled, 4)
    assert sorted(permutation.tolist()) == [0, 1, 2, 3]
    assert (apply_permutation(scrambled, permutation) == truth).all()


def test_match_tolerates_some_disagreement() -> None:
    reference = np.array([0] * 50 + [1] * 50)
    labels = np.array([1] * 45 + [0] * 55)  # swapped names, boundary moved by 5 rows
    renamed = apply_permutation(labels, match_to_reference(reference, labels, 2))
    assert (renamed == reference).mean() == pytest.approx(0.95)
