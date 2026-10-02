from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from termo.config import CoreConfig
from termo.features.pipeline import FEATURE_NAMES
from termo.regime.model import jump_fitter
from termo.validation.stability import halves_ari, stability_score


def test_halves_agree_on_planted_regimes(pre_holdout: pd.DataFrame, config: CoreConfig) -> None:
    value = halves_ari(pre_holdout, FEATURE_NAMES, config.burn_in_days, jump_fitter(2, 50.0))
    assert 0.5 < value <= 1.0


def test_halves_disagree_on_noise(config: CoreConfig) -> None:
    rng = np.random.default_rng(0)
    index = pd.bdate_range("1990-01-01", periods=1400)
    tenors = ["DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS30"]
    walk = 6.0 + np.cumsum(0.03 * rng.normal(size=(1400, 7)), axis=0)
    noise = pd.DataFrame(walk, index=index, columns=tenors)
    value = halves_ari(noise, FEATURE_NAMES, config.burn_in_days, jump_fitter(2, 50.0))
    assert value < 0.5


def test_stability_score_averages_s1_and_s2() -> None:
    assert stability_score([0.9, 0.7], 0.4) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        stability_score([], 0.4)
