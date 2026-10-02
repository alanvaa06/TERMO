from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from termo.config import CoreConfig, load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "core.yaml"


def test_loads_the_registered_configuration() -> None:
    config = load_config(REPO_CONFIG)
    assert config.series == ("DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS30")
    assert config.start == date(1977, 2, 15)
    assert config.holdout_start == date(2024, 10, 1)
    assert len(config.k_values) * len(config.jump_penalties) == 24
    assert config.thresholds.min_median_duration_days == 20
    assert config.ftic.saturated_k == 6


def test_rejects_training_window_inside_the_holdout(config: CoreConfig) -> None:
    with pytest.raises(ValueError, match="start < first_train_end < holdout_start"):
        replace(config, first_train_end=config.holdout_start)


def test_rejects_odd_number_of_pbo_blocks(config: CoreConfig) -> None:
    with pytest.raises(ValueError, match="pbo_blocks"):
        replace(config, pbo_blocks=5)


def test_rejects_single_state_models(config: CoreConfig) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        replace(config, k_values=(1, 2))
