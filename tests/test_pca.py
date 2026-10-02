from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from termo.features.pca import CurvePCA


def test_three_factors_explain_the_synthetic_curve(curve: pd.DataFrame) -> None:
    pca = CurvePCA.fit(curve)
    assert pca.loadings.shape == (3, 7)
    assert pca.explained[0] > pca.explained[1] > pca.explained[2]
    assert pca.explained.sum() > 0.95


def test_sign_convention(curve: pd.DataFrame) -> None:
    pca = CurvePCA.fit(curve)
    tenors = list(curve.columns)
    short, belly = tenors.index("DGS1"), tenors.index("DGS5")
    ten, long_ = tenors.index("DGS10"), tenors.index("DGS30")
    level, slope, curvature = pca.loadings
    assert level[ten] > 0
    assert slope[long_] - slope[short] > 0
    assert curvature[belly] - (curvature[short] + curvature[long_]) / 2 > 0


def test_sign_convention_survives_a_mirrored_sample(curve: pd.DataFrame) -> None:
    # Mirroring the data flips nothing in the covariance; the loadings must come out the same.
    mirrored = 2 * curve.iloc[0] - curve
    original, flipped = CurvePCA.fit(curve), CurvePCA.fit(mirrored)
    assert np.allclose(original.loadings, flipped.loadings, atol=1e-8)


def test_parallel_shift_moves_the_level_only_upwards(curve: pd.DataFrame) -> None:
    pca = CurvePCA.fit(curve)
    shifted = curve.iloc[-5:] + 0.10  # +10 bp on every tenor
    delta = pca.scores(shifted) - pca.scores(curve.iloc[-5:])
    assert (delta["L"] > 0).all()


def test_scores_are_centered_on_the_training_mean(curve: pd.DataFrame) -> None:
    pca = CurvePCA.fit(curve)
    assert np.allclose(pca.scores(curve).mean().to_numpy(), 0.0, atol=1e-8)


def test_rejects_other_tenors(curve: pd.DataFrame) -> None:
    pca = CurvePCA.fit(curve)
    with pytest.raises(ValueError, match="do not match"):
        pca.scores(curve[["DGS1", "DGS2"]])
    with pytest.raises(ValueError, match="sign convention"):
        CurvePCA.fit(curve.drop(columns=["DGS30"]))
