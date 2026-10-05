"""The stages of spec 3: register one model, run it, report the pre-holdout diagnostic.

`run` writes every daily output to hashed CSV files. `report`, the holdout stage and
the weekly reading only read those files: they never refit anything.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from termo.config import CoreConfig
from termo.core import (
    SETUP_TRIAL,
    config_fingerprint,
    environment_fingerprint,
    prior_log_summary,
    verify_binding,
)
from termo.dataset import ExperimentData, first_window_columns
from termo.descriptive.criteria import Check, calibration, evaluate, verdict
from termo.experiment import effective_penalty
from termo.regime.model import jump_fitter
from termo.surrogate.explain import block_map, block_sums, explain, top_variables
from termo.surrogate.model import SurrogateParams, surrogate_walk
from termo.validation.trials import TrialLog, TrialLogError, TrialStatus
from termo.validation.walkforward import run_walkforward

MODEL_DESCRIPTIVE = "descriptive"
PRE_HOLDOUT_DIR, HOLDOUT_DIR = "pre_holdout", "holdout"
FILES = ("labels.csv", "frozen_labels.csv", "proba.csv", "shap_blocks.csv", "shap_top.csv")
TOP_VARIABLES = 5
CALIBRATION_BINS = 5
BASE = "base"
DISCLAIMER = (
    "TERMO describes the current phase of the curve. It does not anticipate the 10Y: "
    "in 62 registered trials the phases did not beat inertia."
)
LIMITS = (
    "K, the penalty and the phase names were chosen looking at pre-holdout data.",
    "D2-D4 are partly true by construction: the phases are built from ranks of these changes.",
    "The fidelity threshold is a judgement, not a calibrated value.",
    "SHAP explains the surrogate, not the market and not the jump model.",
    "Nothing here measures the ability to anticipate; that question is closed with NO-GO.",
)


def desc_trial_id(config: CoreConfig) -> str:
    return f"desc_k{config.k_values[0]}"


def _desc(config: CoreConfig) -> Any:
    if config.descriptive is None:
        raise TrialLogError("this configuration has no descriptive section")
    return config.descriptive


def surrogate_params(config: CoreConfig) -> SurrogateParams:
    s = _desc(config).surrogate
    return SurrogateParams(
        n_estimators=s.n_estimators,
        max_depth=s.max_depth,
        learning_rate=s.learning_rate,
        subsample=s.subsample,
        colsample_bytree=s.colsample_bytree,
        seed=s.seed,
    )


@dataclass(frozen=True, eq=False)
class Analysis:
    labels: pd.Series  # online phase of the jump model, every out-of-sample day
    frozen_labels: pd.Series  # the same from a model fitted once at frozen_train_end
    proba: pd.DataFrame  # surrogate probability per phase, same days as labels
    shap_blocks: pd.DataFrame  # block sums for the day's phase, plus the base value
    shap_top: pd.DataFrame  # one column "top": JSON list of [variable, contribution]


def analyse(data: ExperimentData) -> Analysis:
    """Jump model, frozen jump model, surrogate and SHAP over every out-of-sample day."""
    config = data.config
    desc = _desc(config)
    n_states = config.k_values[0]
    penalty = effective_penalty(config, config.jump_penalties[0], len(data.columns))
    fitter = jump_fitter(n_states, penalty)
    target = data.daily_change_10y
    walk = run_walkforward(data.refits, fitter, n_states, target)
    frozen = run_walkforward(data.frozen_refits, fitter, n_states, target).oos_labels
    surrogates = surrogate_walk(data.refits, walk.fits, n_states, surrogate_params(config))

    blocks: list[pd.DataFrame] = []
    tops: list[pd.Series] = []
    for refit, surrogate in zip(data.refits, surrogates.surrogates, strict=True):
        later = refit.features.loc[refit.features.index > refit.cutoff]
        explanation = explain(surrogate, later, walk.oos_labels.loc[later.index])
        sums = block_sums(explanation.values, desc.blocks)
        sums[BASE] = explanation.base
        blocks.append(sums)
        tops.append(
            explanation.values.apply(
                lambda row: json.dumps(top_variables(row, TOP_VARIABLES)), axis=1
            )
        )
    return Analysis(
        labels=walk.oos_labels,
        frozen_labels=frozen,
        proba=surrogates.proba,
        shap_blocks=pd.concat(blocks),
        shap_top=pd.concat(tops).to_frame("top"),
    )


def restrict(analysis: Analysis, start: pd.Timestamp) -> Analysis:
    """The same outputs from `start` on (used for the holdout days)."""
    return Analysis(
        labels=analysis.labels.loc[start:],
        frozen_labels=analysis.frozen_labels.loc[start:],
        proba=analysis.proba.loc[start:],
        shap_blocks=analysis.shap_blocks.loc[start:],
        shap_top=analysis.shap_top.loc[start:],
    )


def _write(frame: pd.DataFrame, path: Path) -> str:
    text: str = frame.to_csv(index_label="date", lineterminator="\n")
    data = text.encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def save_analysis(analysis: Analysis, directory: Path) -> dict[str, str]:
    """Write the five files and return the SHA-256 of each, to be stored in the log."""
    directory.mkdir(parents=True, exist_ok=True)
    proba = analysis.proba.rename(columns=lambda phase: f"p{phase}")
    frames = {
        "labels.csv": analysis.labels.rename("label").to_frame(),
        "frozen_labels.csv": analysis.frozen_labels.rename("label").to_frame(),
        "proba.csv": proba,
        "shap_blocks.csv": analysis.shap_blocks,
        "shap_top.csv": analysis.shap_top,
    }
    return {name: _write(frames[name], directory / name) for name in FILES}


def load_analysis(directory: Path, expected: Mapping[str, str]) -> Analysis:
    """Read the files back, refusing any whose bytes are not the ones the log recorded."""
    frames: dict[str, pd.DataFrame] = {}
    for name in FILES:
        path = directory / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
            raise TrialLogError(f"{name}: the file is not the ones recorded in the log")
        frames[name] = pd.read_csv(path, parse_dates=["date"], index_col="date")
    proba = frames["proba.csv"].rename(columns=lambda name: int(str(name)[1:]))
    return Analysis(
        labels=frames["labels.csv"]["label"],
        frozen_labels=frames["frozen_labels.csv"]["label"],
        proba=proba,
        shap_blocks=frames["shap_blocks.csv"],
        shap_top=frames["shap_top.csv"],
    )


def checks_of(analysis: Analysis, curve: pd.DataFrame, config: CoreConfig) -> tuple[Check, ...]:
    return evaluate(analysis.labels, curve, analysis.proba, analysis.frozen_labels, config)


def register_desc(
    config: CoreConfig, curve: pd.DataFrame, log: TrialLog, snapshot_hash: str, code_commit: str
) -> tuple[str, ...]:
    """Write the setup record and the single trial before anything is run."""
    desc = _desc(config)
    if code_commit.endswith("-dirty"):
        raise TrialLogError("commit the code and configuration before registering")
    columns = first_window_columns(curve, config)
    # blocks that do not cover every variable exactly once fail here, before the log is written
    block_map(columns, desc.blocks)
    log.register(
        SETUP_TRIAL,
        "One model, its surrogate and the descriptive criteria, all fixed in advance.",
        {
            "columns": list(columns),
            "config": config_fingerprint(config),
            "environment": environment_fingerprint(),
            "hypotheses": [
                "The 3-phase map chosen on pre-holdout data stays persistent, keeps the meaning "
                "of its names and is imitated by the surrogate on the holdout (D1-D6)."
            ],
            "prior_trial_logs": prior_log_summary(config),
        },
        snapshot_hash,
        code_commit,
    )
    log.register(
        desc_trial_id(config),
        f"Descriptive tool: jump model with {config.k_values[0]} phases "
        f"({', '.join(desc.phase_names)}), XGBoost surrogate, criteria D1-D6.",
        {
            "model": MODEL_DESCRIPTIVE,
            "n_states": config.k_values[0],
            "jump_penalty": config.jump_penalties[0],
        },
        snapshot_hash,
        code_commit,
    )
    return columns


def run_desc(
    data: ExperimentData,
    log: TrialLog,
    trials_dir: Path,
    snapshot_hash: str,
    code_commit: str,
    echo: Callable[[str], None] = print,
) -> None:
    """Run the registered trial on pre-holdout data once. An exception leaves it pending."""
    config = data.config
    verify_binding(log, config, snapshot_hash, code_commit)
    trial = desc_trial_id(config)
    if not log.is_registered(trial):
        raise TrialLogError(f"trial {trial} is not registered")
    if log.has_result(trial):
        return
    analysis = analyse(data)
    checks = checks_of(analysis, data.curve, config)
    directory = trials_dir / PRE_HOLDOUT_DIR
    files = save_analysis(analysis, directory)
    log.record_result(
        trial,
        {
            "checks": [asdict(c) for c in checks],
            "verdict": verdict(checks),
            "calibration": calibration(analysis.labels, analysis.proba, CALIBRATION_BINS),
            "days": int(len(analysis.labels)),
            "files": files,
        },
        TrialStatus.KEPT,
        "",
        (directory / "labels.csv").as_posix(),
        files["labels.csv"],
    )
    echo(f"[ok] {trial} diagnostic={verdict(checks)} days={len(analysis.labels)}")


def disclosure(log: TrialLog) -> dict[str, Any]:
    registrations = log.registrations()
    prior = list(registrations[SETUP_TRIAL]["config"].get("prior_trial_logs", []))
    here = len([t for t in registrations if t != SETUP_TRIAL])
    return {
        "n_trials_this_log": here,
        "prior_logs": prior,
        "n_trials_total": here + sum(int(p["n_trials"]) for p in prior),
    }


def render(title: str, period: str, payload: Mapping[str, Any]) -> str:
    """ASCII Markdown of a diagnostic or holdout report."""
    lines = [
        f"# TERMO - {title}",
        "",
        f"**Verdict: {str(payload['verdict']).upper()}** ({period})",
        "",
        DISCLAIMER,
        "",
        "| Check | Value | Requirement | Result |",
        "|---|---|---|---|",
    ]
    for check in payload["checks"]:
        value = "-" if check["value"] is None else f"{check['value']:.4f}"
        result = {True: "pass", False: "FAIL", None: "not evaluable"}[check["passed"]]
        lines.append(f"| {check['name']} | {value} | {check['requirement']} | {result} |")
    lines += ["", f"Days evaluated: {payload['days']}", "", "## What these checks do not prove", ""]
    lines += [f"- {limit}" for limit in LIMITS]
    found = payload["disclosure"]
    lines += ["", "## Disclosure", "", f"- Trials in this log: {found['n_trials_this_log']}"]
    lines += [
        f"- {prior['path']}: {prior['n_trials']} trials, verdict {prior['verdict']}"
        for prior in found["prior_logs"]
    ]
    lines += [f"- Total registered trials: {found['n_trials_total']}", ""]
    return "\n".join(lines)


def write_report(name: str, title: str, period: str, payload: Mapping[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.md").write_text(render(title, period, payload), encoding="utf-8", newline="\n")
    (out / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")


def report_desc(
    data: ExperimentData,
    log: TrialLog,
    trials_dir: Path,
    reports_dir: Path,
    snapshot_hash: str,
    code_commit: str,
) -> str:
    """Recompute the checks from the logged files, log the verdict, write the report."""
    config = data.config
    verify_binding(log, config, snapshot_hash, code_commit)
    trial = desc_trial_id(config)
    results = log.results()
    if trial not in results:
        raise TrialLogError(f"trial {trial} has no result: run the run stage first")
    recorded = results[trial]["metrics"]
    analysis = load_analysis(trials_dir / PRE_HOLDOUT_DIR, recorded["files"])
    checks = checks_of(analysis, data.curve, config)
    payload = {
        "stage": "diagnostic",
        "verdict": verdict(checks),
        "final_trial_id": trial,
        "checks": [asdict(c) for c in checks],
        "days": int(len(analysis.labels)),
        "calibration": recorded["calibration"],
        "limits": list(LIMITS),
        "disclosure": disclosure(log),
    }
    log.record_report({**payload, "snapshot_hash": snapshot_hash, "code_commit": code_commit})
    write_report(
        "diagnostic", "descriptive diagnostic", "pre-holdout, seen data", payload, reports_dir
    )
    return str(payload["verdict"])
