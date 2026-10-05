"""The weekly reading: what phase the curve is in on a date, how sure, and why.

A reading is a lookup in the hashed outputs of the run: it refits nothing and it
never contains a number about what happened after its date. It is produced every
week whatever the verdict: when the governing report is not APTO the sheet says so
in a banner before anything else.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from termo.config import CoreConfig
from termo.descriptive.criteria import APTO
from termo.descriptive.stages import BASE, DISCLAIMER, Analysis
from termo.validation.trials import TrialLogError

DRIVERS = 3
FIDELITY_CHECK = "D6_fidelity"
LOST_HOLDOUT = "opened, no result (lost)"
DRIVERS_TITLE = "## What pushes toward this phase (log-odds of the surrogate)"
FIDELITY_WARNING = "[SURROGATE FIDELITY FAILED: explanations are not validated]"


def span_periods(before: Analysis, after: Analysis) -> Analysis:
    """The outputs of `after` with the phase history of `before` in front of its labels.

    An episode that began before the holdout keeps its start and its length. The two
    periods must join: `after` starts on the business day right after `before` ends.
    """
    last, first = before.labels.index[-1], after.labels.index[0]
    if first != last + pd.offsets.BDay(1):
        raise TrialLogError(
            f"the two periods do not join: one ends {last.date()}, the other starts {first.date()}"
        )
    return replace(after, labels=pd.concat([before.labels, after.labels]))


def governing(validation: Mapping[str, Any]) -> tuple[str, str]:
    """The period whose report governs the reading, and its verdict.

    A holdout with a result governs; before one, or when the holdout was lost, the
    pre-holdout diagnostic does.
    """
    holdout = validation["holdout"]
    if holdout is None or holdout == LOST_HOLDOUT:
        return "pre-holdout diagnostic", str(validation["diagnostic"])
    return "holdout", str(holdout)


def build_reading(
    analysis: Analysis,
    day: date,
    config: CoreConfig,
    validation: Mapping[str, Any],
    generated_with: Mapping[str, Any],
) -> dict[str, Any]:
    desc = config.descriptive
    if desc is None:
        raise ValueError("the configuration has no descriptive section")
    stamp = pd.Timestamp(day)
    if stamp not in analysis.labels.index:
        raise KeyError(f"no reading for {day.isoformat()}: not a day with data in this period")
    past = analysis.labels.loc[:stamp]
    phase = int(past.iloc[-1])
    changed = past.ne(phase)
    start = (
        past.index[0] if not changed.any() else past.index[changed.to_numpy().nonzero()[0][-1] + 1]
    )
    proba = analysis.proba.loc[stamp]
    confidence = float(proba[phase])
    agrees = int(proba.idxmax()) == phase
    blocks = analysis.shap_blocks.loc[stamp].drop(BASE)
    order = blocks.abs().sort_values(ascending=False, kind="stable").index[:DRIVERS]
    top = json.loads(str(analysis.shap_top.loc[stamp, "top"]))
    return {
        "date": day.isoformat(),
        "phase": phase,
        "phase_name": desc.phase_names[phase],
        "days_in_phase": int(len(past.loc[start:])),
        "episode_start": start.date().isoformat(),
        "probabilities": {desc.phase_names[int(p)]: float(proba[p]) for p in proba.index},
        "confidence": confidence,
        "surrogate_agrees": agrees,
        "low_confidence": bool(confidence < desc.low_confidence_below or not agrees),
        "drivers": [{"block": str(b), "contribution": float(blocks[b])} for b in order],
        "top_variables": [{"variable": str(n), "contribution": float(v)} for n, v in top],
        "drivers_validated": not bool(validation["fidelity_failed"]),
        "validation": dict(validation),
        "generated_with": dict(generated_with),
        "disclaimer": DISCLAIMER,
    }


def _banner(validation: Mapping[str, Any]) -> list[str]:
    period, verdict = governing(validation)
    if verdict == APTO:
        return []
    failed = ", ".join(validation["failed_checks"]) or "none evaluable"
    return [f"**[NOT VALIDATED: {period} {verdict.upper()}; failed: {failed}]**", ""]


def _holdout_text(holdout: str | None) -> str:
    if holdout is None:
        return "not opened"
    return holdout if holdout == LOST_HOLDOUT else holdout.upper()


def render_reading(reading: dict[str, Any]) -> str:
    status = reading["validation"]
    lines = [f"# TERMO - reading of {reading['date']}", "", *_banner(status)]
    lines += [
        f"**Phase: {reading['phase_name']}**",
        "",
        f"- In this phase since {reading['episode_start']} "
        f"({reading['days_in_phase']} trading days).",
        f"- Confidence (surrogate probability of this phase): {reading['confidence']:.2f}"
        + ("  [LOW CONFIDENCE]" if reading["low_confidence"] else ""),
        "",
        "| Phase | Probability |",
        "|---|---|",
    ]
    lines += [f"| {name} | {value:.2f} |" for name, value in reading["probabilities"].items()]
    title = DRIVERS_TITLE if reading["drivers_validated"] else f"{DRIVERS_TITLE} {FIDELITY_WARNING}"
    lines += ["", title, ""]
    lines += [f"- {d['block']}: {d['contribution']:+.2f}" for d in reading["drivers"]]
    lines += ["", "Variables with the largest contribution:", ""]
    lines += [f"- {v['variable']}: {v['contribution']:+.2f}" for v in reading["top_variables"]]
    period, _ = governing(status)
    lines += [
        "",
        "## Validation",
        "",
        f"- Pre-holdout diagnostic: {str(status['diagnostic']).upper()}",
        f"- Holdout: {_holdout_text(status['holdout'])}",
        f"- Failed checks ({period}): {', '.join(status['failed_checks']) or 'none'}",
        "",
        reading["disclaimer"],
        "",
    ]
    made = reading["generated_with"]
    provenance = f"Generated with code {made['code_commit']}"
    if made["code_commit"] != made["registered_code_commit"]:
        provenance += f" (registered: {made['registered_code_commit']})"
    lines += [provenance, ""]
    return "\n".join(lines)


def write_reading(reading: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / reading["date"]
    stem.with_suffix(".md").write_text(render_reading(reading), encoding="utf-8", newline="\n")
    stem.with_suffix(".json").write_text(
        json.dumps(reading, indent=2), encoding="utf-8", newline="\n"
    )
