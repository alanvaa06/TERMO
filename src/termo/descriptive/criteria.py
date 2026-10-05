"""The descriptive criteria of spec 3 (D1-D6). All contemporaneous.

Every change is taken over the PAST `change_days` days: nothing here looks at what
happened after a reading. D2-D4 are partly true by construction (the phases are built
from ranks of these same changes): they check that the names still hold on new data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from termo.config import CoreConfig
from termo.validation.metrics import BP_PER_PERCENT, adjusted_rand, median_durations

APTO, NO_APTO = "apto", "no-apto"
PHASE_CHECKS = (
    "D1_persistence",
    "D2_sell_coherence",
    "D3_rally_coherence",
    "D4_short_led_steepens",
    "D4_long_led_flattens",
)


@dataclass(frozen=True)
class Check:
    name: str
    value: float | None  # None: not evaluable on this period
    requirement: str
    passed: bool | None  # None: not evaluable; it is reported and does not fail


def _check(name: str, value: float | None, requirement: str, passed: bool | None) -> Check:
    return Check(name, None if value is None else float(value), requirement, passed)


def evaluate(
    labels: pd.Series,
    curve: pd.DataFrame,
    proba: pd.DataFrame,
    frozen_labels: pd.Series,
    config: CoreConfig,
) -> tuple[Check, ...]:
    """The seven checks on the days of `labels`.

    `curve` must reach back at least `change_days` rows before the first labelled day.
    `frozen_labels` are the labels of a model fitted once and never refitted; they may
    cover only some of the days.
    """
    desc = config.descriptive
    if desc is None:
        raise ValueError("the configuration has no descriptive section")
    if not proba.index.equals(labels.index):
        raise ValueError("labels and probabilities must cover the same days")
    n_states = config.k_values[0]
    change = (curve[desc.level_series].diff(desc.change_days) * BP_PER_PERCENT).reindex(
        labels.index
    )
    slope = curve[desc.slope_long] - curve[desc.slope_short]
    slope_change = (slope.diff(desc.change_days) * BP_PER_PERCENT).reindex(labels.index)
    if change.isna().any() or slope_change.isna().any():
        raise ValueError("the curve does not reach back far enough for the changes")

    days = labels.value_counts()
    evaluable = {p for p in range(n_states) if int(days.get(p, 0)) >= desc.min_days_evaluable}
    minimum = config.thresholds.min_median_duration_days
    share = desc.coherence_share_min

    durations = median_durations(labels.to_numpy(), n_states)
    kept = [durations[p] for p in sorted(evaluable)]
    d1 = (
        _check("D1_persistence", min(kept), f">= {minimum} days", min(kept) >= minimum)
        if kept
        else _check("D1_persistence", None, f">= {minimum} days", None)
    )

    def share_check(name: str, phases: Sequence[int], rising: bool) -> Check:
        requirement = f">= {share}"
        shares = []
        for phase in phases:
            if phase in evaluable:
                moves = change[labels == phase]
                shares.append(float((moves > 0).mean() if rising else (moves < 0).mean()))
        if not shares:
            return _check(name, None, requirement, None)
        return _check(name, min(shares), requirement, min(shares) >= share)

    def lead_check(name: str, phase: int, steepens: bool) -> Check:
        requirement = "> 0 bp" if steepens else "< 0 bp"
        if phase not in evaluable:
            return _check(name, None, requirement, None)
        mean = float(slope_change[labels == phase].mean())
        return _check(name, mean, requirement, mean > 0 if steepens else mean < 0)

    rallies = [p for p in range(n_states) if p != desc.sell_phase]
    shared = frozen_labels.index.intersection(labels.index)
    if len(shared) >= desc.min_days_evaluable:
        ari = adjusted_rand(frozen_labels.loc[shared].to_numpy(), labels.loc[shared].to_numpy())
        d5 = _check("D5_map_stable", ari, f">= {desc.map_ari_min}", ari >= desc.map_ari_min)
    else:
        d5 = _check("D5_map_stable", None, f">= {desc.map_ari_min}", None)

    guess = proba.idxmax(axis=1).to_numpy().astype(int)
    fidelity = float(balanced_accuracy_score(labels.to_numpy().astype(int), guess))
    d6 = _check("D6_fidelity", fidelity, f">= {desc.fidelity_min}", fidelity >= desc.fidelity_min)
    return (
        d1,
        share_check("D2_sell_coherence", [desc.sell_phase], rising=True),
        share_check("D3_rally_coherence", rallies, rising=False),
        lead_check("D4_short_led_steepens", desc.short_led_phase, steepens=True),
        lead_check("D4_long_led_flattens", desc.long_led_phase, steepens=False),
        d5,
        d6,
    )


def verdict(checks: Sequence[Check]) -> str:
    """APTO when every evaluable check passes and at least one phase could be judged."""
    by_name = {c.name: c for c in checks}
    if all(by_name[name].passed is None for name in PHASE_CHECKS):
        return NO_APTO  # no phase had enough days: no evidence is not a pass
    return NO_APTO if any(c.passed is False for c in checks) else APTO


def calibration(labels: pd.Series, proba: pd.DataFrame, bins: int) -> dict[str, Any]:
    """Brier score per phase and a reliability table of the top probability. Reported only."""
    truth = labels.to_numpy().astype(int)
    brier = {
        str(phase): float(np.mean((proba[phase].to_numpy() - (truth == phase)) ** 2))
        for phase in proba.columns
    }
    top = proba.max(axis=1).to_numpy()
    correct = proba.idxmax(axis=1).to_numpy().astype(int) == truth
    edges = np.linspace(0.0, 1.0, bins + 1)
    where = np.clip(np.digitize(top, edges[1:-1]), 0, bins - 1)
    reliability = [
        {
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "mean_probability": float(top[where == b].mean()) if (where == b).any() else None,
            "share_correct": float(correct[where == b].mean()) if (where == b).any() else None,
            "days": int((where == b).sum()),
        }
        for b in range(bins)
    ]
    return {"brier": brier, "reliability": reliability}
