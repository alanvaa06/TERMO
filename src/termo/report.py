"""Turn the measurements into a go/no-go verdict and write it down."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from termo.config import Thresholds
from termo.validation.bootstrap import EtaDifference

REPORT_MD = "go_no_go.md"
REPORT_JSON = "go_no_go.json"


class Verdict(Enum):
    GO = "go"
    NO_GO = "no-go"


@dataclass(frozen=True)
class Criterion:
    name: str
    value: float
    requirement: str
    passed: bool
    blocking: bool  # a failed blocking criterion means no-go


@dataclass(frozen=True)
class CoreReport:
    verdict: Verdict
    final_trial_id: str | None
    criteria: tuple[Criterion, ...]
    notes: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


def blocking_criteria(
    stability: float,
    eta_difference: EtaDifference,
    independence_p: float,
    thresholds: Thresholds,
) -> tuple[Criterion, ...]:
    return (
        Criterion(
            name="stability",
            value=stability,
            requirement=f">= {thresholds.stability_min}",
            passed=stability >= thresholds.stability_min,
            blocking=True,
        ),
        Criterion(
            name="separation_vs_inertia_low95",
            value=eta_difference.low,
            requirement="> 0",
            passed=eta_difference.low > 0.0,
            blocking=True,
        ),
        Criterion(
            name="independence_p",
            value=independence_p,
            requirement=f"< {thresholds.independence_p_max}",
            passed=independence_p < thresholds.independence_p_max,
            blocking=True,
        ),
    )


def decide(criteria: Sequence[Criterion]) -> Verdict:
    if any(c.blocking and not c.passed for c in criteria):
        return Verdict.NO_GO
    return Verdict.GO


def render_markdown(report: CoreReport) -> str:
    lines = [
        "# TERMO - go/no-go",
        "",
        f"**Verdict: {report.verdict.value.upper()}**",
        "",
        f"Final model: `{report.final_trial_id}`",
        "",
        "| Criterion | Value | Requirement | Result | Blocking |",
        "|---|---|---|---|---|",
    ]
    for c in report.criteria:
        result = "pass" if c.passed else "FAIL"
        blocking = "yes" if c.blocking else "no"
        lines.append(f"| {c.name} | {c.value:.4f} | {c.requirement} | {result} | {blocking} |")
    if report.notes:
        lines += ["", "## Notes", ""] + [f"- {note}" for note in report.notes]
    families = report.details.get("families")
    if families:
        chosen = report.details.get("chosen_family")
        lines += [
            "",
            "## Families",
            "",
            "| Family | Verdict | Final model | Separation vs inertia (low 95%) | |",
            "|---|---|---|---|---|",
        ]
        for name, outcome in families.items():
            # A family without an eligible configuration has no final model and no gate.
            final = outcome.get("final_trial_id")
            gate = outcome.get("gate")
            model = "-" if final is None else f"`{final}`"
            low = "-" if gate is None else f"{gate['eta_difference_low']:.4f}"
            mark = "chosen" if name == chosen else ""
            lines.append(
                f"| {name} | {str(outcome['verdict']).upper()} | {model} | {low} | {mark} |"
            )
    disclosure = report.details.get("disclosure")
    if disclosure:
        lines += ["", "## Disclosure", ""]
        lines.append(f"- Trials in this log: {disclosure['n_trials_this_log']}")
        for prior in disclosure["prior_logs"]:
            lines.append(
                f"- {prior['path']}: {prior['n_trials']} trials, verdict {prior['verdict']}"
            )
        lines.append(f"- Total registered trials: {disclosure['n_trials_total']}")
        lines.append(f"- {disclosure['note']}")
    lines += ["", "## Details", "", "```json", json.dumps(report.details, indent=2), "```", ""]
    return "\n".join(lines)


def report_payload(report: CoreReport) -> dict[str, Any]:
    """The report as plain JSON data; the same content goes to the file and to the trial log."""
    return {
        "verdict": report.verdict.value,
        "final_trial_id": report.final_trial_id,
        "criteria": [asdict(c) for c in report.criteria],
        "notes": list(report.notes),
        "details": report.details,
    }


def write_report(report: CoreReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REPORT_MD).write_text(render_markdown(report), encoding="utf-8", newline="\n")
    (out_dir / REPORT_JSON).write_text(
        json.dumps(report_payload(report), indent=2), encoding="utf-8", newline="\n"
    )
