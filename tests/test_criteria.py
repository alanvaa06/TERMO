"""D1-D6 by hand: contemporaneous checks, never a look at what came after."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.descriptive.criteria import (
    APTO,
    NO_APTO,
    PHASE_CHECKS,
    calibration,
    evaluate,
    phase_table,
    verdict,
)

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
    assert set(report) == {"brier", "reliability", "recall_by_phase"}
    assert report["recall_by_phase"]["2"] == {"days": 200, "recall": 1.0}
    assert set(report["brier"]) == {"0", "1", "2"}
    assert all(0.0 <= value <= 1.0 for value in report["brier"].values())
    assert all(
        {"bin", "mean_probability", "share_correct", "days"} == set(row)
        for row in report["reliability"]
    )


def test_fidelity_ignores_a_phase_with_too_few_days(desc_config: CoreConfig) -> None:
    """Three missed days of a phase that barely appears must not decide the verdict."""
    labels, curve, proba, frozen = _case()
    keep = labels.index[:403]  # 200 sell, 200 long-led rally, 3 short-led rally
    few = proba.loc[keep].copy()
    few.iloc[-3:] = [0.05, 0.9, 0.05]  # the surrogate misses the three days of phase 0
    checks = evaluate(labels.loc[keep], curve, few, frozen.loc[keep], desc_config)
    assert _by_name(checks, "D6_fidelity").value == pytest.approx(1.0)
    assert _by_name(checks, "D6_fidelity").passed is True
    report = calibration(labels.loc[keep], few, bins=5)
    assert report["recall_by_phase"]["0"] == {"days": 3, "recall": 0.0}  # still visible
    # with enough days the same misses do count
    many = proba.copy()
    many.loc[labels == 0] = [0.05, 0.9, 0.05]
    failing = evaluate(labels, curve, many, frozen, desc_config)
    assert _by_name(failing, "D6_fidelity").value == pytest.approx(2 / 3)
    assert _by_name(failing, "D6_fidelity").passed is False


def test_without_the_curve_shape_claim_there_is_no_d4(
    desc_config: CoreConfig, desc2_config: CoreConfig
) -> None:
    labels, curve, proba, frozen = _case()
    assert desc2_config.descriptive is not None
    assert not desc2_config.descriptive.has_curve_shape_claim
    checks = evaluate(labels, curve, proba, frozen, desc2_config)
    assert [c.name for c in checks] == [
        "D1_persistence",
        "D2_sell_coherence",
        "D3_rally_coherence",
        "D5_map_stable",
        "D6_fidelity",
    ]
    assert all(c.passed is True for c in checks) and verdict(checks) == APTO
    # the same map with the two rallies exchanged: nothing about the slope is tested
    swapped = labels.replace({0: 1, 1: 0})
    exchanged = evaluate(
        swapped, curve, proba.rename(columns={0: 1, 1: 0})[[0, 1, 2]], swapped, desc2_config
    )
    assert verdict(exchanged) == APTO
    with_claim = evaluate(
        swapped, curve, proba.rename(columns={0: 1, 1: 0})[[0, 1, 2]], swapped, desc_config
    )
    assert verdict(with_claim) == NO_APTO  # the configuration with the claim is unchanged
    assert len(with_claim) == 7 and {"D4_short_led_steepens", "D4_long_led_flattens"} <= set(
        PHASE_CHECKS
    )
    # a slope leg missing from the curve is no longer needed
    only_level = curve[["DGS10"]]
    assert verdict(evaluate(labels, only_level, proba, frozen, desc2_config)) == APTO
    # no evaluable phase is still not a pass
    short = pd.Index(np.r_[labels.index[:30], labels.index[200:230], labels.index[400:430]])
    few = evaluate(labels.loc[short], curve, proba.loc[short], frozen.loc[short], desc2_config)
    assert verdict(few) == NO_APTO


@pytest.mark.parametrize("fixture", ["desc_config", "desc2_config"])
def test_phase_table_describes_every_phase(fixture: str, request: pytest.FixtureRequest) -> None:
    config: CoreConfig = request.getfixturevalue(fixture)
    assert config.descriptive is not None
    labels, curve, proba, _ = _case()
    keep = labels.index[:430]  # 200 sell, 200 of phase 1, 30 of phase 0 (< 40: not evaluable)
    few = proba.loc[keep].copy()
    few.iloc[-3:] = [0.05, 0.9, 0.05]  # the surrogate misses three days of phase 0
    rows = phase_table(labels.loc[keep], curve, few, config)
    assert [r["phase"] for r in rows] == [0, 1, 2]
    assert [r["name"] for r in rows] == list(config.descriptive.phase_names)
    assert set(rows[0]) == {
        "phase",
        "name",
        "days",
        "evaluable",
        "median_duration_days",
        "direction_share",
        "recall",
        "episodes",
    }
    assert [r["days"] for r in rows] == [30, 200, 200]
    assert [r["evaluable"] for r in rows] == [False, True, True]
    assert [r["median_duration_days"] for r in rows] == [30.0, 200.0, 200.0]
    assert [r["episodes"] for r in rows] == [1, 1, 1]
    # direction over the past 63 days: the sell phase rises every day (its lead-in rises
    # too); the first 31 days of phase 1 still carry the sell phase's rise in the window
    assert [r["direction_share"] for r in rows] == [1.0, 169 / 200, 1.0]
    assert rows[1]["direction_share"] == pytest.approx(
        float((curve["DGS10"].diff(63).reindex(keep)[labels.loc[keep] == 1] < 0).mean())
    )
    assert rows[0]["recall"] == pytest.approx(27 / 30) and rows[1]["recall"] == 1.0
    assert rows[2]["recall"] == 1.0
    assert all(isinstance(r["days"], int) and isinstance(r["episodes"], int) for r in rows)
    # a phase that never appears is still a row, with nothing to say about it
    absent = phase_table(labels.loc[keep[:400]], curve, few.loc[keep[:400]], config)
    assert absent[0] == {
        "phase": 0,
        "name": config.descriptive.phase_names[0],
        "days": 0,
        "evaluable": False,
        "median_duration_days": None,
        "direction_share": None,
        "recall": None,
        "episodes": 0,
    }
    # a rising 10Y during a rally is counted against the rally's direction
    rising = curve.copy()
    rising["DGS10"] = 20.0 + 0.02 * np.arange(len(rising))
    against = phase_table(labels.loc[keep], rising, few, config)
    assert [r["direction_share"] for r in against] == [0.0, 0.0, 1.0]
    # two episodes of the sell phase
    two = labels.loc[keep].copy()
    two.iloc[300:310] = 2
    assert phase_table(two, curve, few, config)[2]["episodes"] == 2
    assert phase_table(two, curve, few, config)[2]["median_duration_days"] == 105.0


def test_fidelity_is_not_evaluable_without_an_evaluable_phase(desc_config: CoreConfig) -> None:
    labels, curve, proba, frozen = _case()
    short = pd.Index(np.r_[labels.index[:30], labels.index[200:230], labels.index[400:430]])
    checks = evaluate(labels.loc[short], curve, proba.loc[short], frozen.loc[short], desc_config)
    assert _by_name(checks, "D6_fidelity").passed is None
    assert verdict(checks) == NO_APTO
