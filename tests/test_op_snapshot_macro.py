"""One snapshot holds the curve and the two context series; the panel reads them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import fake_fred_frames, make_macro
from termo.config import CoreConfig
from termo.data.fred import DataValidationError
from termo.data.snapshot import read_snapshot, snapshot_hash
from termo.operation.config import load_operation_config
from termo.operation.macro import macro_frame, macro_panel
from termo.operation.snapshot import take_operation_snapshot

REPO_OP = Path(__file__).resolve().parents[1] / "configs" / "operacion.yaml"


@pytest.fixture(scope="module")
def op_snapshot(
    tmp_path_factory: pytest.TempPathFactory, curve: pd.DataFrame, desc2_config: CoreConfig
) -> Path:
    target = tmp_path_factory.mktemp("op") / "snap"
    op = load_operation_config(REPO_OP)
    take_operation_snapshot(
        desc2_config,
        op,
        target,
        fake_fred_frames([curve, make_macro(curve)]),
        "2026-10-06T00:00:00+00:00",
    )
    return target


def test_snapshot_holds_curve_and_macro_series(op_snapshot: Path, desc2_config: CoreConfig) -> None:
    texts = read_snapshot(op_snapshot)
    assert set(texts) == set(desc2_config.series) | {"THREEFYTP10", "DFF"}
    assert len(snapshot_hash(op_snapshot)) == 64


def test_a_hole_in_the_curve_refuses_the_snapshot_but_a_hole_in_macro_does_not(
    tmp_path: Path, curve: pd.DataFrame, desc2_config: CoreConfig
) -> None:
    op = load_operation_config(REPO_OP)
    holed = curve.copy()
    holed.iloc[500:560, 3] = np.nan
    with pytest.raises(DataValidationError):
        take_operation_snapshot(
            desc2_config, op, tmp_path / "a", fake_fred_frames([holed, make_macro(curve)]), "now"
        )
    assert not (tmp_path / "a").exists()
    macro = make_macro(curve)
    macro.iloc[500:560, 0] = np.nan  # the premium is missing for three months: allowed, noted later
    take_operation_snapshot(
        desc2_config, op, tmp_path / "b", fake_fred_frames([curve, macro]), "now"
    )
    assert (tmp_path / "b" / "THREEFYTP10.csv").exists()


def test_a_negative_term_premium_is_accepted_by_the_snapshot_and_read_back(
    tmp_path: Path, curve: pd.DataFrame, desc2_config: CoreConfig
) -> None:
    """The Kim-Wright premium has many negative observations; a yield never has."""
    op = load_operation_config(REPO_OP)
    macro = make_macro(curve)
    macro["THREEFYTP10"] = macro["THREEFYTP10"] - macro["THREEFYTP10"].median()  # both signs
    assert (macro["THREEFYTP10"] < 0).any() and (macro["THREEFYTP10"] > 0).any()
    take_operation_snapshot(
        desc2_config, op, tmp_path / "neg", fake_fred_frames([curve, macro]), "now"
    )
    frame = macro_frame(tmp_path / "neg", desc2_config, op)
    day = curve.index[int(macro["THREEFYTP10"].to_numpy().argmin())]
    assert frame.loc[day, "THREEFYTP10"] < 0
    assert frame.loc[day, "THREEFYTP10"] == pytest.approx(macro.loc[day, "THREEFYTP10"], abs=1e-4)
    # a negative yield in the curve is still refused
    negative_curve = curve.copy()
    negative_curve.iloc[700, 0] = -0.01
    with pytest.raises(DataValidationError, match="outside"):
        take_operation_snapshot(
            desc2_config, op, tmp_path / "bad", fake_fred_frames([negative_curve, macro]), "now"
        )
    assert not (tmp_path / "bad").exists()


def test_macro_frame_is_on_the_curve_calendar_with_the_spread(
    op_snapshot: Path, curve: pd.DataFrame, desc2_config: CoreConfig
) -> None:
    op = load_operation_config(REPO_OP)
    frame = macro_frame(op_snapshot, desc2_config, op)
    assert list(frame.columns) == ["THREEFYTP10", "DGS2_menos_DFF"]
    assert frame.index.equals(curve.index)  # business days of the curve, weekends dropped
    macro = make_macro(curve)
    day = curve.index[1000]
    # the fake FRED serves four decimals, so the snapshot differs from the frame by < 1e-4
    assert frame.loc[day, "THREEFYTP10"] == pytest.approx(macro.loc[day, "THREEFYTP10"], abs=1e-4)
    assert frame.loc[day, "DGS2_menos_DFF"] == pytest.approx(
        curve.loc[day, "DGS2"] - macro.loc[day, "DFF"], abs=2e-4
    )


def test_macro_panel_gives_value_percentile_and_change_by_hand(
    op_snapshot: Path, curve: pd.DataFrame, desc2_config: CoreConfig
) -> None:
    op = load_operation_config(REPO_OP)
    frame = macro_frame(op_snapshot, desc2_config, op)
    day = curve.index[2000]
    panel = macro_panel(frame, op, day.date())
    assert [row["serie"] for row in panel] == [
        "prima por plazo 10 años (Kim-Wright)",
        "DGS2 - fed funds efectiva",
    ]
    window = frame["THREEFYTP10"].loc[:day].iloc[-op.percentile_window_days :]
    assert panel[0]["valor"] == pytest.approx(float(frame.loc[day, "THREEFYTP10"]))
    assert panel[0]["percentil_10a"] == pytest.approx(float((window <= window.iloc[-1]).mean()))
    history = frame["THREEFYTP10"].loc[:day]
    assert panel[0]["cambio_21d"] == pytest.approx(
        float(history.iloc[-1] - history.iloc[-1 - op.change_days])
    )
    assert panel[0]["dias_de_ventana"] == min(op.percentile_window_days, len(frame.loc[:day]))
    assert panel[0]["nota"] == ""


def test_a_missing_macro_value_uses_the_last_available_and_says_so(
    op_snapshot: Path, curve: pd.DataFrame, desc2_config: CoreConfig
) -> None:
    op = load_operation_config(REPO_OP)
    frame = macro_frame(op_snapshot, desc2_config, op)
    frame = frame.copy()
    day = curve.index[2100]
    frame.loc[curve.index[2095] : day, "THREEFYTP10"] = np.nan
    panel = macro_panel(frame, op, day.date())
    assert panel[0]["valor"] == pytest.approx(
        float(frame["THREEFYTP10"].loc[: curve.index[2094]].iloc[-1])
    )
    assert "último dato" in panel[0]["nota"]
    assert curve.index[2094].date().isoformat() in panel[0]["nota"]
