"""Why the surrogate gives today's phase: SHAP values in log-odds, summed by block.

SHAP explains the surrogate, not the market and not the jump model. It is only
worth reading when the surrogate imitates the jump model well (criterion D6).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap  # type: ignore[import-untyped]

from termo.surrogate.model import Surrogate

Blocks = Sequence[tuple[str, Sequence[str]]]  # (block name, first tokens of its variables)


def first_token(name: str) -> str:
    """d10_63_r252 -> d10; vol2_r252 -> vol2."""
    return name.split("_", 1)[0]


def block_map(columns: Sequence[str], blocks: Blocks) -> dict[str, str]:
    """Variable -> block. Every variable must fall in exactly one block."""
    owners: dict[str, list[str]] = {}
    for block, tokens in blocks:
        for token in tokens:
            owners.setdefault(token, []).append(block)
    mapping: dict[str, str] = {}
    for column in columns:
        found = owners.get(first_token(column), [])
        if not found:
            raise ValueError(f"variable {column} belongs to no block")
        if len(found) > 1:
            raise ValueError(f"variable {column} belongs to more than one block: {found}")
        mapping[column] = found[0]
    return mapping


@dataclass(frozen=True, eq=False)
class Explanation:
    values: pd.DataFrame  # rows x variables: contribution to the log-odds of the row's phase
    base: pd.Series  # the surrogate's base value for that phase


def explain(surrogate: Surrogate, features: pd.DataFrame, phases: pd.Series) -> Explanation:
    """SHAP values of each row for ITS phase. NaN where the surrogate never saw that phase."""
    if not features.index.equals(phases.index):
        raise ValueError("features and phases must cover the same rows")
    rows = features[list(surrogate.columns)]
    explainer = shap.TreeExplainer(surrogate.model)
    raw = np.asarray(explainer.shap_values(rows), dtype=float)
    base = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
    if raw.ndim == 2:  # two classes: a single margin, that of the second class
        raw = np.stack([-raw, raw], axis=2)
        base = np.array([-base[0], base[0]])
    position = {phase: i for i, phase in enumerate(surrogate.classes)}
    values = np.full(rows.shape, np.nan)
    bases = np.full(len(rows), np.nan)
    for row, phase in enumerate(phases.to_numpy()):
        where = position.get(int(phase))
        if where is not None:
            values[row] = raw[row, :, where]
            bases[row] = base[where]
    return Explanation(
        values=pd.DataFrame(values, index=rows.index, columns=list(surrogate.columns)),
        base=pd.Series(bases, index=rows.index),
    )


def block_sums(values: pd.DataFrame, blocks: Blocks) -> pd.DataFrame:
    """Sum the contributions of each block. Exact: SHAP values are additive."""
    mapping = block_map(list(values.columns), blocks)
    out = pd.DataFrame(index=values.index)
    for block, _ in blocks:
        members = [column for column in values.columns if mapping[column] == block]
        out[block] = values[members].sum(axis=1, min_count=1) if members else 0.0
    return out


def top_variables(row: pd.Series, count: int) -> list[tuple[str, float]]:
    """The `count` variables with the largest absolute contribution, largest first."""
    ordered = row.dropna().abs().sort_values(ascending=False, kind="stable").index[:count]
    return [(str(name), float(row[name])) for name in ordered]
