"""Typed, validated view of configs/operacion.yaml (spec 4: operation in shadow mode)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from termo.config import _only_known_keys

REQUIRED_TEXTS = ("descargo", "nota_confianza", "rotulo_historia", "no_dice")
MIN_PERCENTILE_WINDOW_DAYS = 252
TOP_LEVEL_KEYS = frozenset(
    {
        "registry",
        "model_config",
        "macro",
        "alert_min_confidence",
        "percentile_window_days",
        "change_days",
        "shadow_eval_min_weeks",
        "texts",
    }
)


@dataclass(frozen=True)
class MacroSeries:
    series_id: str  # FRED id
    name: str  # Spanish label used in sheets and CSV
    spread_against: str | None  # when set, the panel shows (this curve series - series_id)


@dataclass(frozen=True)
class OperationConfig:
    registry: str  # the registry whose reading is operated, e.g. "desc2"
    model_config: Path  # the registered model configuration
    macro: tuple[MacroSeries, ...]
    alert_min_confidence: float  # (0, 1]
    percentile_window_days: int  # >= 252
    change_days: int  # >= 1
    shadow_eval_min_weeks: int  # >= 1
    texts: dict[str, str]  # keys: descargo, nota_confianza, rotulo_historia, no_dice

    def __post_init__(self) -> None:
        if not 0 < self.alert_min_confidence <= 1:
            raise ValueError("alert_min_confidence must be in (0, 1]")
        if self.percentile_window_days < MIN_PERCENTILE_WINDOW_DAYS:
            raise ValueError(f"percentile_window_days must be >= {MIN_PERCENTILE_WINDOW_DAYS}")
        if self.change_days < 1:
            raise ValueError("change_days must be >= 1")
        if self.shadow_eval_min_weeks < 1:
            raise ValueError("shadow_eval_min_weeks must be >= 1")
        ids = [m.series_id for m in self.macro]
        if len(set(ids)) != len(ids):
            raise ValueError("macro series ids must be distinct")
        missing = [key for key in REQUIRED_TEXTS if not self.texts.get(key)]
        if missing:
            raise ValueError(f"texts is missing the keys {missing}")


def load_operation_config(path: Path) -> OperationConfig:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    unknown = sorted(set(raw) - TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"unknown keys in {path.name}: {unknown}")
    for entry in raw["macro"]:
        _only_known_keys(entry, MacroSeries, "macro")
    return OperationConfig(
        registry=str(raw["registry"]),
        model_config=Path(raw["model_config"]),
        macro=tuple(
            MacroSeries(
                series_id=str(m["series_id"]),
                name=str(m["name"]),
                spread_against=None
                if m.get("spread_against") is None
                else str(m["spread_against"]),
            )
            for m in raw["macro"]
        ),
        alert_min_confidence=float(raw["alert_min_confidence"]),
        percentile_window_days=int(raw["percentile_window_days"]),
        change_days=int(raw["change_days"]),
        shadow_eval_min_weeks=int(raw["shadow_eval_min_weeks"]),
        texts={str(k): str(v) for k, v in raw["texts"].items()},
    )
