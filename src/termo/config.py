"""Typed, validated view of configs/core.yaml."""

from __future__ import annotations

from dataclasses import dataclass, fields
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
class SurrogateConfig:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    seed: int


@dataclass(frozen=True)
class DescriptiveConfig:
    """Spec 3: one fixed model read as a description, never as a forecast."""

    phase_names: tuple[str, ...]
    sell_phase: int
    short_led_phase: int  # the rally led by the short end: the curve steepens
    long_led_phase: int  # the rally led by the long end: the curve flattens
    level_series: str
    slope_long: str
    slope_short: str
    change_days: int  # contemporaneous changes are taken over the PAST this many days
    coherence_share_min: float
    min_days_evaluable: int
    map_ari_min: float
    fidelity_min: float
    low_confidence_below: float
    surrogate: SurrogateConfig
    blocks: tuple[tuple[str, tuple[str, ...]], ...]  # (block, first tokens of its variables)


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
    descriptive: DescriptiveConfig | None = None

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
        if self.descriptive is not None:
            desc = self.descriptive
            if len(self.k_values) != 1 or len(self.jump_penalties) != 1:
                raise ValueError("the descriptive tool is one model: one K and one jump penalty")
            if len(desc.phase_names) != self.k_values[0]:
                raise ValueError("the descriptive tool needs one name per phase")
            phases = {desc.sell_phase, desc.short_led_phase, desc.long_led_phase}
            if len(phases) != 3 or not phases <= set(range(self.k_values[0])):
                raise ValueError("sell, short-led and long-led phases must be three valid phases")
            if self.frozen_train_end is None:
                raise ValueError("the descriptive tool needs frozen_train_end for criterion D5")
            legs = {desc.level_series, desc.slope_long, desc.slope_short}
            if not legs <= set(self.series) or desc.slope_long == desc.slope_short:
                raise ValueError(
                    "level and slope series must be series of the curve, slope legs distinct"
                )
            if len(set(desc.phase_names)) != len(desc.phase_names):
                raise ValueError("phase names must be different from each other")


def _only_known_keys(section: dict[str, object], kind: type, where: str) -> None:
    """A misspelled or invented key would be dropped in silence and still be registered."""
    unknown = sorted(set(section) - {f.name for f in fields(kind)})
    if unknown:
        raise ValueError(f"unknown keys in {where}: {unknown}")


def load_config(path: Path) -> CoreConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    tyccles = raw.get("tyccles")
    desc = raw.get("descriptive")
    if desc is not None:
        _only_known_keys(desc, DescriptiveConfig, "descriptive")
        _only_known_keys(desc["surrogate"], SurrogateConfig, "descriptive.surrogate")
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
        descriptive=None
        if desc is None
        else DescriptiveConfig(
            phase_names=tuple(str(name) for name in desc["phase_names"]),
            sell_phase=int(desc["sell_phase"]),
            short_led_phase=int(desc["short_led_phase"]),
            long_led_phase=int(desc["long_led_phase"]),
            level_series=str(desc["level_series"]),
            slope_long=str(desc["slope_long"]),
            slope_short=str(desc["slope_short"]),
            change_days=int(desc["change_days"]),
            coherence_share_min=float(desc["coherence_share_min"]),
            min_days_evaluable=int(desc["min_days_evaluable"]),
            map_ari_min=float(desc["map_ari_min"]),
            fidelity_min=float(desc["fidelity_min"]),
            low_confidence_below=float(desc["low_confidence_below"]),
            surrogate=SurrogateConfig(
                n_estimators=int(desc["surrogate"]["n_estimators"]),
                max_depth=int(desc["surrogate"]["max_depth"]),
                learning_rate=float(desc["surrogate"]["learning_rate"]),
                subsample=float(desc["surrogate"]["subsample"]),
                colsample_bytree=float(desc["surrogate"]["colsample_bytree"]),
                seed=int(desc["surrogate"]["seed"]),
            ),
            blocks=tuple(
                (str(name), tuple(str(token) for token in tokens))
                for name, tokens in desc["blocks"].items()
            ),
        ),
    )
