"""The descriptive criteria of spec 3 (D1-D6). All contemporaneous.

Every change is taken over the PAST `change_days` days: nothing here looks at what
happened after a reading. D2-D4 are partly true by construction (the phases are built
from ranks of these same changes): they check that the names still hold on new data.
D4 exists only when the configuration makes the curve-shape claim (spec section 9).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from termo.config import CoreConfig, DescriptiveConfig
from termo.validation.metrics import (
    BP_PER_PERCENT,
    adjusted_rand,
    median_durations,
    run_lengths,
)

APTO, NO_APTO = "apto", "no-apto"
PHASE_CHECKS = (
    "D1_persistence",
    "D2_sell_coherence",
    "D3_rally_coherence",
    "D4_short_led_steepens",
    "D4_long_led_flattens",
)


def _descriptive(config: CoreConfig) -> DescriptiveConfig:
    if config.descriptive is None:
        raise ValueError("the configuration has no descriptive section")
    return config.descriptive


def _past_change(series: pd.Series, labels: pd.Series, change_days: int) -> pd.Series:
    """The change in bp over the PAST `change_days` days, on the labelled days."""
    change = (series.diff(change_days) * BP_PER_PERCENT).reindex(labels.index)
    if change.isna().any():
        raise ValueError("the curve does not reach back far enough for the changes")
    return change


def _evaluable(labels: pd.Series, n_states: int, min_days: int) -> set[int]:
    days = labels.value_counts()
    return {p for p in range(n_states) if int(days.get(p, 0)) >= min_days}


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
    """The checks on the days of `labels`: D1-D6, or D1-D3 and D5-D6 without the claim.

    `curve` must reach back at least `change_days` rows before the first labelled day.
    `frozen_labels` are the labels of a model fitted once and never refitted; they may
    cover only some of the days.
    """
    desc = _descriptive(config)
    if not proba.index.equals(labels.index):
        raise ValueError("labels and probabilities must cover the same days")
    n_states = config.k_values[0]
    change = _past_change(curve[desc.level_series], labels, desc.change_days)
    evaluable = _evaluable(labels, n_states, desc.min_days_evaluable)
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

    def lead_checks() -> tuple[Check, ...]:
        """D4, only when the configuration claims which end leads each rally."""
        if not desc.has_curve_shape_claim:
            return ()
        assert desc.slope_long is not None and desc.slope_short is not None
        slope_change = _past_change(
            curve[desc.slope_long] - curve[desc.slope_short], labels, desc.change_days
        )

        def lead(name: str, phase: int | None, steepens: bool) -> Check:
            requirement = "> 0 bp" if steepens else "< 0 bp"
            if phase not in evaluable:
                return _check(name, None, requirement, None)
            mean = float(slope_change[labels == phase].mean())
            return _check(name, mean, requirement, mean > 0 if steepens else mean < 0)

        return (
            lead("D4_short_led_steepens", desc.short_led_phase, steepens=True),
            lead("D4_long_led_flattens", desc.long_led_phase, steepens=False),
        )

    rallies = [p for p in range(n_states) if p != desc.sell_phase]
    shared = frozen_labels.index.intersection(labels.index)
    if len(shared) >= desc.min_days_evaluable:
        ari = adjusted_rand(frozen_labels.loc[shared].to_numpy(), labels.loc[shared].to_numpy())
        d5 = _check("D5_map_stable", ari, f">= {desc.map_ari_min}", ari >= desc.map_ari_min)
    else:
        d5 = _check("D5_map_stable", None, f">= {desc.map_ari_min}", None)

    # Like D1-D4, only phases with enough days count: three missed days of a phase that
    # barely appears must not decide. Recall of every phase is reported in `calibration`.
    judged = labels.isin(sorted(evaluable)).to_numpy()
    if judged.any():
        truth = labels.to_numpy().astype(int)[judged]
        guess = proba.idxmax(axis=1).to_numpy().astype(int)[judged]
        fidelity = float(balanced_accuracy_score(truth, guess))
        d6 = _check(
            "D6_fidelity", fidelity, f">= {desc.fidelity_min}", fidelity >= desc.fidelity_min
        )
    else:
        d6 = _check("D6_fidelity", None, f">= {desc.fidelity_min}", None)
    return (
        d1,
        share_check("D2_sell_coherence", [desc.sell_phase], rising=True),
        share_check("D3_rally_coherence", rallies, rising=False),
        *lead_checks(),
        d5,
        d6,
    )


def verdict(checks: Sequence[Check]) -> str:
    """APTO when every evaluable check passes and at least one phase could be judged."""
    if all(c.passed is None for c in checks if c.name in PHASE_CHECKS):
        return NO_APTO  # no phase had enough days: no evidence is not a pass
    return NO_APTO if any(c.passed is False for c in checks) else APTO


def phase_table(
    labels: pd.Series, curve: pd.DataFrame, proba: pd.DataFrame, config: CoreConfig
) -> list[dict[str, Any]]:
    """One row per phase, decided by nothing: days, evaluable, median duration, episodes,
    share of days with the level series moving the phase's way (up in the sell phase,
    down in the rallies) over the past `change_days`, and the surrogate's recall.
    """
    desc = _descriptive(config)
    n_states = config.k_values[0]
    change = _past_change(curve[desc.level_series], labels, desc.change_days)
    evaluable = _evaluable(labels, n_states, desc.min_days_evaluable)
    durations = median_durations(labels.to_numpy(), n_states)
    episodes = {p: 0 for p in range(n_states)}
    for state, _ in run_lengths(labels.to_numpy()):
        episodes[state] += 1
    guess = proba.idxmax(axis=1).to_numpy().astype(int)
    rows: list[dict[str, Any]] = []
    for phase in range(n_states):
        mask = (labels == phase).to_numpy()
        days = int(mask.sum())
        moves = change.to_numpy()[mask]
        rising = phase == desc.sell_phase
        rows.append(
            {
                "phase": phase,
                "name": desc.phase_names[phase],
                "days": days,
                "evaluable": phase in evaluable,
                "median_duration_days": None if days == 0 else float(durations[phase]),
                "direction_share": None
                if days == 0
                else float((moves > 0).mean() if rising else (moves < 0).mean()),
                "recall": None if days == 0 else float((guess[mask] == phase).mean()),
                "episodes": episodes[phase],
            }
        )
    return rows


def calibration(labels: pd.Series, proba: pd.DataFrame, bins: int) -> dict[str, Any]:
    """Brier per phase, reliability of the top probability, recall per phase. Reported only."""
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
    recall = {
        str(phase): {
            "days": int((truth == phase).sum()),
            "recall": float(correct[truth == phase].mean()) if (truth == phase).any() else None,
        }
        for phase in proba.columns
    }
    return {"brier": brier, "reliability": reliability, "recall_by_phase": recall}
