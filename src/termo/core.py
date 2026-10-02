"""The stages of spec 1: snapshot, register, run, report, final holdout."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from termo.config import CoreConfig
from termo.data.fred import fetch_series_csv, parse_series_csv
from termo.data.loader import check_coverage, complete_days, load_curve
from termo.data.snapshot import snapshot_hash, write_snapshot
from termo.dataset import ExperimentData, first_window_columns, prepare
from termo.experiment import (
    HoldoutResult,
    evaluate_config,
    gate_tests,
    holdout_evaluation,
    pbo_of,
    saturated_wcss,
    separation,
)
from termo.report import (
    CoreReport,
    Criterion,
    Verdict,
    blocking_criteria,
    decide,
    report_payload,
    write_report,
)
from termo.selection import (
    Candidate,
    candidate_from_record,
    ftic_states,
    pick_winner,
    simpler_alternative,
)
from termo.validation.pbo import effective_n
from termo.validation.trials import TrialLog, TrialLogError, TrialStatus
from termo.validation.walkforward import inertia_labels

SETUP_TRIAL = "setup"
INERTIA_TRIAL = "inertia"
LABELS_DIR = "labels"
HOLDOUT_JSON = "holdout.json"
DIRTY_SUFFIX = "-dirty"
MODEL_JUMP, MODEL_KMEANS, MODEL_INERTIA = "jump", "kmeans", "inertia"
KNOWN_LIMITATIONS = (
    "DGS30 between 2002-02-19 and 2006-02-08 is built differently from the rest of the series.",
    "S1 is high almost by construction: consecutive windows share most of their data.",
    "PBO ranks configurations by raw eta-squared, so it favours grids that mix different K.",
    "Velocities are in basis points, so the 1980s can dominate the extreme regimes.",
    "A forward change of exactly zero counts as 'down' in the independence test.",
    "separation_shift uses the older chance level (groups slid in time). It is a declared "
    "sensitivity check: it is reported and never decides.",
    "Evidence for jump models comes from equities; these tests are the criterion for rates.",
)


def jump_trial_id(n_states: int, jump_penalty: float) -> str:
    return f"jm_k{n_states}_lam{jump_penalty:g}"


def kmeans_trial_id(n_states: int) -> str:
    return f"kmeans_k{n_states}"


def config_fingerprint(config: CoreConfig) -> dict[str, Any]:
    """The whole configuration in JSON form: what a registration is bound to."""

    def plain(value: Any) -> Any:
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        return value

    fingerprint: dict[str, Any] = plain(asdict(config))
    return fingerprint


def take_snapshot(
    config: CoreConfig, snapshot_dir: Path, get: Callable[[str], str], downloaded_at: str
) -> None:
    """Download, validate and store. Nothing is written unless every check passes.

    Coverage is checked over the whole download, holdout included: it only looks at
    which days have values, and a hole found later would be found after the holdout
    had been opened.
    """
    texts: dict[str, str] = {}
    parsed: list[pd.Series] = []
    for series_id in config.series:
        text = fetch_series_csv(series_id, get)
        parsed.append(parse_series_csv(text, series_id))
        texts[series_id] = text
    check_coverage(complete_days(parsed, config.start), config.start)
    write_snapshot(snapshot_dir, texts, downloaded_at)


def register_trials(
    config: CoreConfig, curve: pd.DataFrame, log: TrialLog, snapshot_hash: str, code_commit: str
) -> tuple[str, ...]:
    """Write every trial to the log before anything is run. Returns the fixed feature set.

    The setup record binds the experiment to this configuration, snapshot and commit.
    """
    if code_commit.endswith(DIRTY_SUFFIX):
        raise TrialLogError("commit the code and configuration before registering")
    columns = first_window_columns(curve, config)
    log.register(
        SETUP_TRIAL,
        "Feature set fixed on the first training window; every parameter fixed in advance.",
        {"columns": list(columns), "config": config_fingerprint(config)},
        snapshot_hash,
        code_commit,
    )
    for n_states in config.k_values:
        for jump_penalty in config.jump_penalties:
            log.register(
                jump_trial_id(n_states, jump_penalty),
                f"A jump model with {n_states} states and penalty {jump_penalty:g} "
                "gives stable regimes that separate the forward 10Y move.",
                {"model": MODEL_JUMP, "n_states": n_states, "jump_penalty": jump_penalty},
                snapshot_hash,
                code_commit,
            )
        log.register(
            kmeans_trial_id(n_states),
            "Baseline: the same model without a jump penalty.",
            {"model": MODEL_KMEANS, "n_states": n_states},
            snapshot_hash,
            code_commit,
        )
    log.register(
        INERTIA_TRIAL,
        "Baseline: sign of the smoothed 63-day change of the level.",
        {"model": MODEL_INERTIA},
        snapshot_hash,
        code_commit,
    )
    return columns


def _setup(log: TrialLog) -> dict[str, Any]:
    registrations = log.registrations()
    if SETUP_TRIAL not in registrations:
        raise TrialLogError("no setup registration: run the register stage first")
    return registrations[SETUP_TRIAL]


def registered_columns(log: TrialLog) -> tuple[str, ...]:
    return tuple(_setup(log)["config"]["columns"])


def verify_binding(log: TrialLog, config: CoreConfig, snapshot_hash: str, code_commit: str) -> None:
    """Refuse to work with anything other than what was registered."""
    setup = _setup(log)
    changed = [
        name
        for name, registered, current in (
            ("configuration", setup["config"]["config"], config_fingerprint(config)),
            ("data snapshot", setup["snapshot_hash"], snapshot_hash),
            ("code commit", setup["code_commit"], code_commit),
        )
        if registered != current
    ]
    if changed:
        raise TrialLogError("this does not match the registration; changed: " + ", ".join(changed))


def save_labels(labels: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.rename("label").to_csv(path, index_label="date", lineterminator="\n")


def load_labels(path: Path) -> pd.Series:
    return pd.read_csv(path, parse_dates=["date"], index_col="date")["label"]


def run_trials(
    data: ExperimentData,
    log: TrialLog,
    trials_dir: Path,
    snapshot_hash: str,
    code_commit: str,
    echo: Callable[[str], None] = print,
) -> None:
    """Run every registered trial that has no result yet. Safe to re-run after an interruption."""
    config = data.config
    verify_binding(log, config, snapshot_hash, code_commit)
    for trial_id, registration in log.registrations().items():
        if trial_id == SETUP_TRIAL or log.has_result(trial_id):
            continue
        spec = registration["config"]
        labels_path = trials_dir / LABELS_DIR / f"{trial_id}.csv"
        if spec["model"] == MODEL_INERTIA:
            labels = inertia_labels(data.refits)
            found = separation(labels, data.yields_10y, config.horizon_short_days, config)
            save_labels(labels, labels_path)
            log.record_result(
                trial_id,
                {
                    "eta_short": found.raw,
                    "excess_short": found.excess,
                    "excess_short_shift": found.excess_shift,
                },
                TrialStatus.KEPT,
                "baseline",
                str(labels_path),
            )
            echo(f"[ok] {trial_id} excess_short={found.excess:.4f}")
            continue
        try:
            evaluation = evaluate_config(data, int(spec["n_states"]), spec.get("jump_penalty"))
        except Exception as error:
            log.record_result(trial_id, {}, TrialStatus.FAILED, repr(error), None)
            raise
        save_labels(evaluation.oos_labels, labels_path)
        if spec["model"] == MODEL_KMEANS:
            status, reason = TrialStatus.KEPT, "baseline"
        elif evaluation.passes_duration:
            status, reason = TrialStatus.KEPT, ""
        else:
            status, reason = TrialStatus.DISCARDED, "fails the minimum median duration"
        log.record_result(trial_id, evaluation.metrics(), status, reason, str(labels_path))
        echo(f"[ok] {trial_id} score={evaluation.score:.4f} status={status.value}")


def _days_by_decade(labels: pd.Series) -> dict[str, dict[str, int]]:
    """How many days of each decade fall in each state: shows whether one era owns a regime."""
    counts = labels.groupby([labels, labels.index.year // 10 * 10]).size()
    table: dict[str, dict[str, int]] = {}
    for (state, decade), days in counts.items():
        table.setdefault(str(state), {})[str(decade)] = int(days)
    return table


def build_report(
    data: ExperimentData, log: TrialLog, trials_dir: Path, snapshot_hash: str, code_commit: str
) -> CoreReport:
    config = data.config
    verify_binding(log, config, snapshot_hash, code_commit)
    registrations, results = log.registrations(), log.results()
    trial_ids = [t for t in registrations if t != SETUP_TRIAL]
    pending = [t for t in trial_ids if t not in results]
    if pending:
        raise TrialLogError(f"trials without a result: {pending}")
    failed = [t for t in trial_ids if results[t]["status"] == TrialStatus.FAILED.value]
    if failed:
        raise TrialLogError(f"failed trials, the grid is incomplete: {failed}")

    def is_model(trial_id: str, model: str) -> bool:
        return bool(registrations[trial_id]["config"].get("model") == model)

    candidates = [
        candidate_from_record(t, registrations[t]["config"], results[t]["metrics"])
        for t in trial_ids
        if is_model(t, MODEL_JUMP)
    ]
    details: dict[str, object] = {
        "columns": list(data.columns),
        "configurations": [
            {
                "trial_id": c.trial_id,
                "stability": c.stability,
                "separation": c.separation,
                "separation_shift": results[c.trial_id]["metrics"]["excess_short_shift"],
                "score": c.score,
                "passes_duration": c.passes_duration,
            }
            for c in candidates
        ],
        "baselines_separation": {
            t: results[t]["metrics"]["excess_short"]
            for t in trial_ids
            if is_model(t, MODEL_KMEANS) or is_model(t, MODEL_INERTIA)
        },
        "baselines_separation_shift": {
            t: results[t]["metrics"]["excess_short_shift"]
            for t in trial_ids
            if is_model(t, MODEL_KMEANS) or is_model(t, MODEL_INERTIA)
        },
        "limitations": list(KNOWN_LIMITATIONS),
    }

    winner = pick_winner(candidates)
    if winner is None:
        return CoreReport(
            verdict=Verdict.NO_GO,
            final_trial_id=None,
            criteria=(),
            notes=(
                "No configuration passes the minimum median duration "
                "with positive stability and positive separation.",
            ),
            details=details,
        )

    labels_dir = trials_dir / LABELS_DIR
    inertia = load_labels(labels_dir / f"{INERTIA_TRIAL}.csv")

    def gates(candidate: Candidate) -> tuple[tuple[Criterion, ...], dict[str, float]]:
        labels = load_labels(labels_dir / f"{candidate.trial_id}.csv")
        gate = gate_tests(labels, inertia, data.yields_10y, config)
        criteria = blocking_criteria(
            candidate.stability, gate.eta_difference, gate.independence.p_value, config.thresholds
        )
        summary = {
            "eta_difference_point": gate.eta_difference.point,
            "eta_difference_low": gate.eta_difference.low,
            "eta_difference_high": gate.eta_difference.high,
            "independence_statistic": gate.independence.statistic,
            "n_weeks": float(gate.n_weeks),
        }
        return criteria, summary

    final = winner
    criteria, gate_summary = gates(winner)
    notes: list[str] = []
    n_obs, n_features = data.full_features.shape
    ftic_k = ftic_states(
        candidates,
        winner.jump_penalty,
        wcss_saturated=saturated_wcss(data),
        n_obs=n_obs,
        n_features=n_features,
        config=config.ftic,
    )
    if ftic_k is not None and ftic_k != winner.n_states:
        notes.append(f"FTIC prefers K={ftic_k}; the score prefers K={winner.n_states}.")
    alternative = simpler_alternative(candidates, winner, ftic_k)
    if alternative is not None and alternative.passes_duration:
        alt_criteria, alt_summary = gates(alternative)
        if decide(alt_criteria) is Verdict.GO:
            final, criteria, gate_summary = alternative, alt_criteria, alt_summary
            notes.append(
                "The simpler model passes every blocking criterion: it is the final model."
            )

    labelings = [load_labels(labels_dir / f"{c.trial_id}.csv") for c in candidates]
    pbo = pbo_of(labelings, data.yields_10y, config)
    final_metrics = results[final.trial_id]["metrics"]
    s2 = float(final_metrics["s2"])
    thresholds = config.thresholds
    extra = (
        Criterion("pbo", pbo, f"<= {thresholds.pbo_max}", pbo <= thresholds.pbo_max, False),
        Criterion(
            "s2_halves",
            s2,
            f">= {thresholds.stability_min}",
            s2 >= thresholds.stability_min,
            False,
        ),
    )
    if pbo > thresholds.pbo_max:
        notes.append("PBO above the limit: prune the grid and repeat as new trials.")
    if s2 < thresholds.stability_min:
        notes.append("S2 (halves) is below the threshold: review even though the average passes.")
    details.update(
        {
            "score_winner": winner.trial_id,
            "ftic_states": ftic_k,
            "gate": gate_summary,
            "n_trials": len(candidates),
            "n_effective": effective_n(
                [labels.to_numpy() for labels in labelings], config.effective_n_cut
            ),
            "final_model": {
                "s1_mean": final_metrics["s1_mean"],
                "s1_min": final_metrics["s1_min"],
                "s2": s2,
                "separation_long": final_metrics["excess_long"],
                "median_duration_days": final_metrics["durations"],
                "days_by_decade": _days_by_decade(
                    load_labels(labels_dir / f"{final.trial_id}.csv")
                ),
            },
        }
    )
    all_criteria = criteria + extra
    return CoreReport(
        verdict=decide(all_criteria),
        final_trial_id=final.trial_id,
        criteria=all_criteria,
        notes=tuple(notes),
        details=details,
    )


def publish_report(
    report: CoreReport, log: TrialLog, reports_dir: Path, snapshot_hash: str, code_commit: str
) -> None:
    """Write the report files and log the verdict. The log, not the files, is the record."""
    log.record_report(
        {**report_payload(report), "snapshot_hash": snapshot_hash, "code_commit": code_commit}
    )
    write_report(report, reports_dir)


def run_final_holdout(
    config: CoreConfig, snapshot_dir: Path, log: TrialLog, reports_dir: Path, code_commit: str
) -> HoldoutResult:
    """One-shot evaluation on the holdout. The log refuses a second run.

    Everything that can fail is checked before the holdout is recorded as opened.
    """
    report = log.last_report()
    if report is None:
        raise TrialLogError("no logged report: run the report stage first")
    final_trial_id = report["final_trial_id"]
    if report["verdict"] != Verdict.GO.value or final_trial_id is None:
        raise TrialLogError("the holdout is only opened for a model with a GO verdict")
    verify_binding(log, config, snapshot_hash(snapshot_dir), code_commit)
    spec = log.registrations()[final_trial_id]["config"]
    columns = registered_columns(log)
    # Reads and validates the whole snapshot, and returns no holdout row.
    load_curve(snapshot_dir, config.series, config.start, config.holdout_start)

    log.open_holdout(final_trial_id)  # recorded before any holdout row is used
    curve = load_curve(
        snapshot_dir, config.series, config.start, config.holdout_start, final_evaluation=True
    )
    result = holdout_evaluation(
        prepare(curve, config, columns), int(spec["n_states"]), float(spec["jump_penalty"])
    )
    payload = {**asdict(result), "passed": result.passed, "final_trial_id": final_trial_id}
    log.record_holdout_result(payload)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / HOLDOUT_JSON).write_text(
        json.dumps(payload, indent=2), encoding="utf-8", newline="\n"
    )
    return result
