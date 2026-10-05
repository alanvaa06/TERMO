"""D1-D6 by hand: contemporaneous checks, never a look at what came after."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.descriptive.criteria import APTO, NO_APTO, calibration, evaluate, verdict

DAYS = 600


def _case() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
    """200 days of each phase, in order 2 (sell), 1 (long-led rally), 0 (short-led rally).

    Yields move so that every phase means what its name says, with a 63-day lead-in.
    """
    index = pd.bdate_range("2001-01-01", periods=DAYS + 63)
    step10 = np.r_[np.full(63, 0.02), np.full(200, 0.02), np.full(200, -0.02), np.full(200, -0.02)]
    step2 = np.r_[np.full(63, 0.02), np.full(200, 0.02), np.full(200, -0.01), np.full(200, -0.04)]
    curve = pd.DataFrame(
        {"DGS10": 20.0 + np.cumsum(step10), "DGS2": 20.0 + np.cumsum(step2)}, index=index
    )
    days = index[63:]
    labels = pd.Series(np.repeat([2, 1, 0], 200), index=days)
    proba = pd.DataFrame(0.05, index=days, columns=[0, 1, 2])
    for phase in (0, 1, 2):
        proba.loc[labels == phase, phase] = 0.9
    return labels, curve, proba, labels.copy()


def _by_name(checks: tuple, name: str):  # type: ignore[no-untyped-def]
    (found,) = [c for c in checks if c.name == name]
    return found


def test_a_coherent_map_passes_every_check(desc_config: CoreConfig) -> None:
    labels, curve, proba, frozen = _case()
    checks = evaluate(labels, curve, proba, frozen, desc_config)
    assert [c.name for c in checks] == [
        "D1_persistence",
        "D2_sell_coherence",
        "D3_rally_coherence",
        "D4_short_led_steepens",
        "D4_long_led_flattens",
        "D5_map_stable",
        "D6_fidelity",
    ]
    assert all(c.passed is True for c in checks)
    assert _by_name(checks, "D1_persistence").value == 200.0
    assert _by_name(checks, "D5_map_stable").value == pytest.approx(1.0)
    assert _by_name(checks, "D6_fidelity").value == pytest.approx(1.0)
    assert verdict(checks) == APTO


def test_each_check_can_fail_on_its_own(desc_config: CoreConfig) -> None:
    labels, curve, proba, frozen = _case()

    flicker = labels.copy()
    flicker.iloc[::7] = (flicker.iloc[::7] + 1) % 3  # runs of at most 6 days
    assert (
        _by_name(evaluate(flicker, curve, proba, frozen, desc_config), "D1_persistence").passed
        is False
    )

    falling = curve.copy()
    falling["DGS10"] = 20.0 - 0.02 * np.arange(len(falling))
    checks = evaluate(labels, falling, proba, frozen, desc_config)
    assert _by_name(checks, "D2_sell_coherence").passed is False

    rising = curve.copy()
    rising["DGS10"] = 20.0 + 0.02 * np.arange(len(rising))
    assert (
        _by_name(evaluate(labels, rising, proba, frozen, desc_config), "D3_rally_coherence").passed
        is False
    )

    swapped = labels.replace({0: 1, 1: 0})  # the names of the two rallies exchanged
    checks = evaluate(
        swapped, curve, proba.rename(columns={0: 1, 1: 0})[[0, 1, 2]], swapped, desc_config
    )
    assert _by_name(checks, "D4_short_led_steepens").passed is False
    assert _by_name(checks, "D4_long_led_flattens").passed is False

    rng = np.random.default_rng(0)
    other_map = pd.Series(rng.integers(0, 3, size=len(labels)), index=labels.index)
    assert (
        _by_name(evaluate(labels, curve, proba, other_map, desc_config), "D5_map_stable").passed
        is False
    )

    wrong = proba[[1, 2, 0]].set_axis([0, 1, 2], axis=1)  # the surrogate points at another phase
    assert (
        _by_name(evaluate(labels, curve, wrong, frozen, desc_config), "D6_fidelity").passed is False
    )
    assert verdict(evaluate(labels, curve, wrong, frozen, desc_config)) == NO_APTO


def test_a_phase_with_few_days_is_not_evaluable_and_does_not_fail(desc_config: CoreConfig) -> None:
    labels, curve, proba, frozen = _case()
    keep = labels.index[:430]  # 200 sell, 200 long-led rally, 30 short-led rally (< 40)
    checks = evaluate(labels.loc[keep], curve, proba.loc[keep], frozen.loc[keep], desc_config)
    assert _by_name(checks, "D4_short_led_steepens").passed is None
    assert _by_name(checks, "D4_short_led_steepens").value is None
    assert _by_name(checks, "D3_rally_coherence").passed is True  # judged on the long-led rally
    assert verdict(checks) == APTO


def test_no_evaluable_phase_means_not_fit(desc_config: CoreConfig) -> None:
    labels, curve, proba, frozen = _case()
    short = pd.Index(np.r_[labels.index[:30], labels.index[200:230], labels.index[400:430]])
    checks = evaluate(labels.loc[short], curve, proba.loc[short], frozen.loc[short], desc_config)
    assert all(_by_name(checks, n).passed is None for n in ("D1_persistence", "D2_sell_coherence"))
    assert verdict(checks) == NO_APTO


def test_the_frozen_map_may_cover_fewer_days(desc_config: CoreConfig) -> None:
    labels, curve, proba, frozen = _case()
    checks = evaluate(labels, curve, proba, frozen.iloc[300:], desc_config)
    assert _by_name(checks, "D5_map_stable").value == pytest.approx(1.0)
    few = evaluate(labels, curve, proba, frozen.iloc[:10], desc_config)
    assert _by_name(few, "D5_map_stable").passed is None


def test_checks_use_the_configured_thresholds(desc_config: CoreConfig) -> None:
    labels, curve, proba, frozen = _case()
    assert desc_config.descriptive is not None
    strict = replace(desc_config, descriptive=replace(desc_config.descriptive, fidelity_min=1.01))
    assert _by_name(evaluate(labels, curve, proba, frozen, strict), "D6_fidelity").passed is False


def test_calibration_reports_brier_and_reliability() -> None:
    labels, _, proba, _ = _case()
    report = calibration(labels, proba, bins=5)
    assert set(report) == {"brier", "reliability"}
    assert set(report["brier"]) == {"0", "1", "2"}
    assert all(0.0 <= value <= 1.0 for value in report["brier"].values())
    assert all(
        {"bin", "mean_probability", "share_correct", "days"} == set(row)
        for row in report["reliability"]
    )
