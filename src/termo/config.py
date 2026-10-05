"""Typed, validated view of configs/core.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Thresholds:
    stability_min: float
    independence_p_max: float
    min_median_duration_days: int
    pbo_max: float
    collinearity_max: float


@dataclass(frozen=True)
class BootstrapConfig:
    block_weeks: int
    n_resamples: int
    seed: int


@dataclass(frozen=True)
class FticConfig:
    k0: int
    mean_phase_days: int
    saturated_k: int
    max_jump_fraction: float


FEATURE_SETS = ("pca", "tyccles")


@dataclass(frozen=True)
class TycclesConfig:
    change_horizons_days: tuple[int, ...]
    rank_windows_days: tuple[int, ...]
    vol_window_days: int
    vol_rank_window_days: int


@dataclass(frozen=True)
class CoreConfig:
    series: tuple[str, ...]
    start: date
    holdout_start: date
    first_train_end: date
    refit_weeks: int
    burn_in_days: int
    k_values: tuple[int, ...]
    jump_penalties: tuple[float, ...]
    horizon_short_days: int
    horizon_long_days: int
    pbo_blocks: int
    effective_n_cut: float
    thresholds: Thresholds
    bootstrap: BootstrapConfig
    ftic: FticConfig
    feature_set: str = "pca"
    tyccles: TycclesConfig | None = None
    apply_collinearity_rule: bool = True
    jump_penalty_per_feature: bool = False  # lambda = value * number of features
    frozen_train_end: date | None = None  # K-means fitted once through this date, never refitted
    prior_trial_logs: tuple[str, ...] = ()  # earlier registries the report must count

    def __post_init__(self) -> None:
        if not self.start < self.first_train_end < self.holdout_start:
            raise ValueError("dates must satisfy start < first_train_end < holdout_start")
        if not self.k_values or min(self.k_values) < 2:
            raise ValueError("every K in the grid must be at least 2")
        if not self.jump_penalties or min(self.jump_penalties) <= 0:
            raise ValueError("every jump penalty in the grid must be positive")
        if self.pbo_blocks < 2 or self.pbo_blocks % 2 != 0:
            raise ValueError("pbo_blocks must be an even number >= 2")
        if self.refit_weeks < 1 or self.burn_in_days < 0:
            raise ValueError("refit_weeks must be >= 1 and burn_in_days >= 0")
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
        if (self.feature_set == "tyccles") != (self.tyccles is not None):
            raise ValueError("the tyccles feature set needs its parameters, and only that set")
        if self.frozen_train_end is not None and not (
            self.start < self.frozen_train_end < self.holdout_start
        ):
            raise ValueError(
                "frozen_train_end must satisfy start < frozen_train_end < holdout_start"
            )


def load_config(path: Path) -> CoreConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    tyccles = raw.get("tyccles")
    return CoreConfig(
        series=tuple(raw["series"]),
        start=raw["start"],
        holdout_start=raw["holdout_start"],
        first_train_end=raw["first_train_end"],
        refit_weeks=int(raw["refit_weeks"]),
        burn_in_days=int(raw["burn_in_days"]),
        k_values=tuple(int(k) for k in raw["grid"]["k"]),
        jump_penalties=tuple(float(x) for x in raw["grid"]["jump_penalty"]),
        horizon_short_days=int(raw["horizons_days"]["short"]),
        horizon_long_days=int(raw["horizons_days"]["long"]),
        pbo_blocks=int(raw["pbo_blocks"]),
        effective_n_cut=float(raw["effective_n_cut"]),
        thresholds=Thresholds(**raw["thresholds"]),
        bootstrap=BootstrapConfig(**raw["bootstrap"]),
        ftic=FticConfig(**raw["ftic"]),
        feature_set=str(raw.get("feature_set", "pca")),
        tyccles=None
        if tyccles is None
        else TycclesConfig(
            change_horizons_days=tuple(int(h) for h in tyccles["change_horizons_days"]),
            rank_windows_days=tuple(int(w) for w in tyccles["rank_windows_days"]),
            vol_window_days=int(tyccles["vol_window_days"]),
            vol_rank_window_days=int(tyccles["vol_rank_window_days"]),
        ),
        apply_collinearity_rule=bool(raw.get("apply_collinearity_rule", True)),
        jump_penalty_per_feature=bool(raw.get("jump_penalty_per_feature", False)),
        frozen_train_end=raw.get("frozen_train_end"),
        prior_trial_logs=tuple(str(p) for p in raw.get("prior_trial_logs", ())),
    )
