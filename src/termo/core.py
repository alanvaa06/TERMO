"""The stages of spec 1: snapshot, register, run, report, final holdout."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Callable, Iterable, Mapping
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from termo.config import CoreConfig
from termo.data.fred import fetch_series_csv, parse_series_csv
from termo.data.loader import check_coverage, complete_days, load_curve, snapshot_days
from termo.data.snapshot import snapshot_hash, write_snapshot
from termo.dataset import ExperimentData, first_window_columns, prepare, recipe_for
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
    is_eligible,
    pick_winner,
    simpler_alternative,
)
from termo.validation.pbo import effective_n
from termo.validation.trials import RecordKind, TrialLog, TrialLogError, TrialStatus
from termo.validation.walkforward import inertia_labels

SETUP_TRIAL = "setup"
INERTIA_TRIAL = "inertia"
LABELS_DIR = "labels"
HOLDOUT_JSON = "holdout.json"
DIRTY_SUFFIX = "-dirty"
MIN_HOLDOUT_DAYS = 120  # below this the holdout comparison means nothing
BOUND_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "jumpmodels", "xgboost", "shap")
MODEL_JUMP, MODEL_KMEANS, MODEL_INERTIA = "jump", "kmeans", "inertia"
MODEL_KMEANS_FROZEN = "kmeans_frozen"
FAMILIES = (MODEL_JUMP, MODEL_KMEANS_FROZEN)  # each has its own winner and criteria
RUNNABLE_MODELS = frozenset({MODEL_JUMP, MODEL_KMEANS, MODEL_KMEANS_FROZEN, MODEL_INERTIA})
HYPOTHESES = (
    "H1 (diagnostic, not a criterion): with rank features the refit K-means baselines pass "
    "the minimum median duration.",
    "H2: at least one jump model configuration passes every blocking criterion.",
    "H3: the frozen K-means passes every blocking criterion.",
)
TYCCLES_LIMITATIONS = (
    "Ranks of multi-month changes are smooth by construction: stability and duration can pass "
    "without any information about the future; separation and independence decide.",
    "139 correlated inputs without pruning: the distance is dominated by level changes.",
    "The 102-input TYCCLES recipe is reconstructed from the text, not reproduced.",
    "The frozen K-means reads 1998-2024 with centroids from 1979-1997; HSBC fitted on 50 years.",
    "PBO over four frozen configurations is nearly blind.",
    "Second look at the same pre-holdout data: every test family has had two chances.",
)
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
NO_ELIGIBLE_REASON = (
    "no eligible configuration: none passes the minimum median duration "
    "with positive stability and positive separation"
)


def jump_trial_id(n_states: int, jump_penalty: float) -> str:
    return f"jm_k{n_states}_lam{jump_penalty:g}"


def kmeans_trial_id(n_states: int) -> str:
    return f"kmeans_k{n_states}"


def frozen_trial_id(n_states: int) -> str:
    return f"kmf_k{n_states}"


def _plain(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def config_fingerprint(config: CoreConfig) -> dict[str, Any]:
    """The whole configuration in JSON form: what a registration is bound to."""
    fingerprint: dict[str, Any] = _plain(asdict(config))
    return fingerprint


# Keys that later code added and an earlier registration may lack. Every key here must be
# one whose default reproduces the behaviour the registration had: never a key that
# changes results. Anything else that is missing, or any key the current code does not
# know, is a difference.
LATER_OPTIONAL_KEYS = frozenset(
    {"descriptive", "descriptive.holdout_already_seen", "descriptive.holdout_seen_by"}
)


def _with_later_defaults(registered: Any, current: Any, prefix: str = "") -> Any:
    """A registered fingerprint read with the keys later code added, by explicit list.

    A listed key the registration predates is read as its default; a required key it
    lacks stays missing, and a key it has that the code no longer knows is kept, so
    both show up as differences.
    """
    if not (is_dataclass(current) and isinstance(registered, dict)):
        return registered
    filled: dict[str, Any] = {}
    known = {field.name for field in fields(current)}
    for field in fields(current):
        dotted = f"{prefix}{field.name}"
        if field.name in registered:
            filled[field.name] = _with_later_defaults(
                registered[field.name], getattr(current, field.name), f"{dotted}."
            )
        elif dotted in LATER_OPTIONAL_KEYS and field.default is not MISSING:
            filled[field.name] = _plain(field.default)
    for key in registered.keys() - known:
        filled[key] = registered[key]
    return filled


def environment_fingerprint() -> dict[str, str]:
    """Versions of what computes the results. A different library can give different regimes."""
    versions = {name: metadata.version(name) for name in BOUND_PACKAGES}
    versions["python"] = platform.python_version()
    return versions


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


def prior_log_summary(config: CoreConfig) -> list[dict[str, Any]]:
    """What earlier registries hold, counted now so that the report cannot forget them."""
    summary: list[dict[str, Any]] = []
    for path in config.prior_trial_logs:
        log = TrialLog(Path(path))
        if not log.path.exists():
            raise TrialLogError(f"prior trial log not found: {path}")
        trials = [t for t in log.registrations() if t != SETUP_TRIAL]
        # the verdict that counts is the last one: a holdout result over an earlier report
        holdouts = [r for r in log.records() if r["kind"] == RecordKind.HOLDOUT_RESULT.value]
        last: dict[str, Any] | None = holdouts[-1] if holdouts else log.last_report()
        summary.append(
            {
                "path": path,
                "n_trials": len(trials),
                "verdict": None if last is None else last["verdict"],
            }
        )
    return summary


def register_trials(
    config: CoreConfig, curve: pd.DataFrame, log: TrialLog, snapshot_hash: str, code_commit: str
) -> tuple[str, ...]:
    """Write every trial to the log before anything is run. Returns the fixed feature set.

    The setup record binds the experiment to this configuration, snapshot, code and
    library versions.
    """
    if code_commit.endswith(DIRTY_SUFFIX):
        raise TrialLogError("commit the code and configuration before registering")
    columns = first_window_columns(curve, config)
    log.register(
        SETUP_TRIAL,
        "Feature set fixed on the first training window; every parameter fixed in advance.",
        {
            "columns": list(columns),
            "config": config_fingerprint(config),
            "environment": environment_fingerprint(),
            "hypotheses": list(HYPOTHESES),
            "prior_trial_logs": prior_log_summary(config),
        },
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
        if config.frozen_train_end is not None:
            log.register(
                frozen_trial_id(n_states),
                f"K-means fitted once through {config.frozen_train_end.isoformat()} with "
                f"{n_states} states gives stable regimes that separate the forward 10Y move.",
                {"model": MODEL_KMEANS_FROZEN, "n_states": n_states},
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


def _refuse_changes(bound: Iterable[tuple[str, Any, Any]]) -> None:
    changed = [name for name, registered, current in bound if registered != current]
    if changed:
        raise TrialLogError("this does not match the registration; changed: " + ", ".join(changed))


def _configuration_binding(setup: Mapping[str, Any], config: CoreConfig) -> tuple[str, Any, Any]:
    registered = _with_later_defaults(setup["config"]["config"], config)
    return ("configuration", registered, config_fingerprint(config))


def _data_binding(
    setup: Mapping[str, Any], config: CoreConfig, snapshot_hash: str
) -> list[tuple[str, Any, Any]]:
    return [
        _configuration_binding(setup, config),
        ("data snapshot", setup["snapshot_hash"], snapshot_hash),
    ]


def verify_binding(log: TrialLog, config: CoreConfig, snapshot_hash: str, code_commit: str) -> None:
    """Refuse to work with anything other than what was registered."""
    setup = _setup(log)
    _refuse_changes(
        [
            *_data_binding(setup, config, snapshot_hash),
            ("environment", setup["config"]["environment"], environment_fingerprint()),
            ("code commit", setup["code_commit"], code_commit),
        ]
    )


def verify_data_binding(log: TrialLog, config: CoreConfig, snapshot_hash: str) -> None:
    """Refuse any configuration or snapshot other than the registered ones.

    The code and the library versions are not checked: a reading only formats hashed
    outputs, so later code or a library upgrade must still be able to produce it.
    """
    _refuse_changes(_data_binding(_setup(log), config, snapshot_hash))


def verify_configuration(log: TrialLog, config: CoreConfig) -> None:
    """Refuse any configuration other than the registered one; nothing else is compared.

    For a stage that recomputes on a new snapshot every time (the shadow operation): the
    snapshot is new by design, and the code and the libraries prove themselves there by
    reproducing the registered history byte for byte.
    """
    _refuse_changes([_configuration_binding(_setup(log), config)])


def save_labels(labels: pd.Series, path: Path) -> str:
    """Write the label series and return the SHA-256 of the file, to be stored in the log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text: str = labels.rename("label").to_csv(index_label="date", lineterminator="\n")
    data = text.encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def load_labels(path: Path) -> pd.Series:
    return pd.read_csv(path, parse_dates=["date"], index_col="date")["label"]


def logged_labels(log: TrialLog, trials_dir: Path, trial_id: str) -> pd.Series:
    """The labels of a finished trial, refused if the file is not the one the log recorded.

    The report recomputes its criteria from these files, so they are decision inputs.
    """
    path = trials_dir / LABELS_DIR / f"{trial_id}.csv"
    expected = log.results()[trial_id]["labels_sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise TrialLogError(f"the labels of {trial_id} are not the ones recorded in the log")
    return load_labels(path)


def run_trials(
    data: ExperimentData,
    log: TrialLog,
    trials_dir: Path,
    snapshot_hash: str,
    code_commit: str,
    echo: Callable[[str], None] = print,
) -> None:
    """Run every registered trial that has no result yet. Safe to re-run after an interruption.

    A log that registers a model this stage cannot run belongs to another stage: nothing
    is evaluated or recorded, so that registry is not spoiled by a partial run.
    """
    config = data.config
    verify_binding(log, config, snapshot_hash, code_commit)
    pending = {
        trial_id: registration["config"]
        for trial_id, registration in log.registrations().items()
        if trial_id != SETUP_TRIAL and not log.has_result(trial_id)
    }
    foreign = [t for t, spec in pending.items() if spec.get("model") not in RUNNABLE_MODELS]
    if foreign:
        raise TrialLogError(f"registered models this stage cannot run: {foreign}")
    for trial_id, spec in pending.items():
        labels_path = trials_dir / LABELS_DIR / f"{trial_id}.csv"
        if spec["model"] == MODEL_INERTIA:
            labels = inertia_labels(data.refits)
            found = separation(labels, data.yields_10y, config.horizon_short_days, config)
            log.record_result(
                trial_id,
                {
                    "eta_short": found.raw,
                    "excess_short": found.excess,
                    "excess_short_shift": found.excess_shift,
                },
                TrialStatus.KEPT,
                "baseline",
                labels_path.as_posix(),
                save_labels(labels, labels_path),
            )
            echo(f"[ok] {trial_id} excess_short={found.excess:.4f}")
            continue
        # An exception leaves no result: the trial stays pending and the same command retries it.
        evaluation = evaluate_config(
            data,
            int(spec["n_states"]),
            spec.get("jump_penalty"),
            frozen=spec["model"] == MODEL_KMEANS_FROZEN,
        )
        if spec["model"] == MODEL_KMEANS:
            status, reason = TrialStatus.KEPT, "baseline"
        elif evaluation.passes_duration:
            status, reason = TrialStatus.KEPT, ""
        else:
            status, reason = TrialStatus.DISCARDED, "fails the minimum median duration"
        log.record_result(
            trial_id,
            evaluation.metrics(),
            status,
            reason,
            labels_path.as_posix(),
            save_labels(evaluation.oos_labels, labels_path),
        )
        echo(f"[ok] {trial_id} score={evaluation.score:.4f} status={status.value}")


def _days_by_decade(labels: pd.Series) -> dict[str, dict[str, int]]:
    """How many days of each decade fall in each state: shows whether one era owns a regime."""
    counts = labels.groupby([labels, labels.index.year // 10 * 10]).size()
    table: dict[str, dict[str, int]] = {}
    for (state, decade), days in counts.items():
        table.setdefault(str(state), {})[str(decade)] = int(days)
    return table


@dataclass(frozen=True)
class FamilyOutcome:
    family: str
    score_winner: Candidate
    final: Candidate
    criteria: tuple[Criterion, ...]
    notes: tuple[str, ...]
    gate: dict[str, float]
    ftic_states: int | None
    pbo: float
    n_candidates: int
    n_effective: int
    final_model: dict[str, Any]

    @property
    def verdict(self) -> Verdict:
        return decide(self.criteria)

    @property
    def separation_low(self) -> float:
        return self.gate["eta_difference_low"]

    def payload(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "score_winner": self.score_winner.trial_id,
            "final_trial_id": self.final.trial_id,
            "criteria": [asdict(c) for c in self.criteria],
            "notes": list(self.notes),
            "gate": self.gate,
            "ftic_states": self.ftic_states,
            "pbo": self.pbo,
            "n_trials": self.n_candidates,
            "n_effective": self.n_effective,
            "final_model": self.final_model,
        }


def _family_outcome(
    family: str,
    candidates: list[Candidate],
    data: ExperimentData,
    log: TrialLog,
    trials_dir: Path,
    inertia: pd.Series,
) -> FamilyOutcome | None:
    """Winner, gates, FTIC (jump family only), PBO and S2 of one family.

    None if nothing is eligible.
    """
    config = data.config
    results = log.results()
    winner = pick_winner(candidates)
    if winner is None:
        return None

    def gates(candidate: Candidate) -> tuple[tuple[Criterion, ...], dict[str, float]]:
        labels = logged_labels(log, trials_dir, candidate.trial_id)
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
    ftic_k: int | None = None
    if family == MODEL_JUMP:
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
        if alternative is not None and is_eligible(alternative):
            alt_criteria, alt_summary = gates(alternative)
            if decide(alt_criteria) is Verdict.GO:
                final, criteria, gate_summary = alternative, alt_criteria, alt_summary
                notes.append(
                    "The simpler model passes every blocking criterion: it is the final model."
                )

    labelings = [logged_labels(log, trials_dir, c.trial_id) for c in candidates]
    pbo = pbo_of(labelings, data.yields_10y, config)
    final_metrics = results[final.trial_id]["metrics"]
    s2 = float(final_metrics["s2"])
    thresholds = config.thresholds
    extra = (
        Criterion("pbo", pbo, f"<= {thresholds.pbo_max}", pbo <= thresholds.pbo_max, False),
        Criterion(
            "s2_halves", s2, f">= {thresholds.stability_min}", s2 >= thresholds.stability_min, False
        ),
    )
    if pbo > thresholds.pbo_max:
        notes.append("PBO above the limit: prune the grid and repeat as new trials.")
    if s2 < thresholds.stability_min <= final.stability:
        notes.append("S2 (halves) is below the threshold even though the average passes.")
    final_model = {
        "s1_mean": final_metrics["s1_mean"],
        "s1_min": final_metrics["s1_min"],
        "s2": s2,
        "separation_long": final_metrics["excess_long"],
        "median_duration_days": final_metrics["durations"],
        "days_by_decade": _days_by_decade(logged_labels(log, trials_dir, final.trial_id)),
    }
    return FamilyOutcome(
        family=family,
        score_winner=winner,
        final=final,
        criteria=criteria + extra,
        notes=tuple(notes),
        gate=gate_summary,
        ftic_states=ftic_k,
        pbo=pbo,
        n_candidates=len(candidates),
        n_effective=effective_n(
            [labels.to_numpy() for labels in labelings], config.effective_n_cut
        ),
        final_model=final_model,
    )


def disclosure(log: TrialLog) -> dict[str, Any]:
    """How many trials this project has registered, here and in every earlier registry."""
    prior = list(_setup(log)["config"].get("prior_trial_logs", []))
    here = len([t for t in log.registrations() if t != SETUP_TRIAL])
    return {
        "n_trials_this_log": here,
        "prior_logs": prior,
        "n_trials_total": here + sum(int(p["n_trials"]) for p in prior),
        "note": "The pre-holdout data were already examined by every prior log listed here. "
        "The holdout stays closed until a GO verdict and the user's explicit approval.",
    }


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

    def is_model(trial_id: str, model: str) -> bool:
        return bool(registrations[trial_id]["config"].get("model") == model)

    def candidates_of(family: str) -> list[Candidate]:
        return [
            candidate_from_record(t, registrations[t]["config"], results[t]["metrics"])
            for t in trial_ids
            if is_model(t, family)
        ]

    families = {family: candidates_of(family) for family in FAMILIES}
    limitations = list(KNOWN_LIMITATIONS)
    if config.feature_set == "tyccles":
        limitations += list(TYCCLES_LIMITATIONS)
    details: dict[str, object] = {
        "columns": list(data.columns),
        "configurations": [
            {
                "trial_id": c.trial_id,
                "family": family,
                "stability": c.stability,
                "separation": c.separation,
                "separation_shift": results[c.trial_id]["metrics"]["excess_short_shift"],
                "score": c.score,
                "passes_duration": c.passes_duration,
            }
            for family, candidates in families.items()
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
        "baselines_passes_duration": {
            t: bool(results[t]["metrics"]["passes_duration"])
            for t in trial_ids
            if is_model(t, MODEL_KMEANS)
        },
        "hypotheses": list(_setup(log)["config"].get("hypotheses", [])),
        "limitations": limitations,
        "disclosure": disclosure(log),
    }

    if not any(families.values()):
        raise TrialLogError("no candidate trials in the log")
    inertia = logged_labels(log, trials_dir, INERTIA_TRIAL)
    outcomes: dict[str, FamilyOutcome] = {}
    reported: dict[str, dict[str, Any]] = {}  # every family with candidates, eligible or not
    without_eligible: list[str] = []
    for family, candidates in families.items():
        if not candidates:
            continue
        outcome = _family_outcome(family, candidates, data, log, trials_dir, inertia)
        if outcome is None:
            reported[family] = {
                "verdict": Verdict.NO_GO.value,
                "final_trial_id": None,
                "n_trials": len(candidates),
                "reason": NO_ELIGIBLE_REASON,
            }
            without_eligible.append(f"Family {family}: no eligible configuration.")
        else:
            outcomes[family] = outcome
            reported[family] = outcome.payload()
    details["families"] = reported
    if not outcomes:
        return CoreReport(
            verdict=Verdict.NO_GO,
            final_trial_id=None,
            criteria=(),
            notes=(
                "No configuration passes the minimum median duration "
                "with positive stability and positive separation.",
                *without_eligible,
            ),
            details=details,
        )

    passing = [o for o in outcomes.values() if o.verdict is Verdict.GO]
    chosen = max(passing or list(outcomes.values()), key=lambda o: o.separation_low)
    notes = list(chosen.notes) + without_eligible
    if len(reported) > 1:
        notes.append(
            f"Family {chosen.family} decides: "
            + ("it passes every blocking criterion" if passing else "no family passes")
            + "; the others are reported above."
        )
    details.update(
        {
            "chosen_family": chosen.family,
            "score_winner": chosen.score_winner.trial_id,
            "ftic_states": chosen.ftic_states,
            "gate": chosen.gate,
            "n_trials": chosen.n_candidates,
            "n_effective": chosen.n_effective,
            "final_model": chosen.final_model,
        }
    )
    return CoreReport(
        verdict=chosen.verdict,
        final_trial_id=chosen.final.trial_id,
        criteria=chosen.criteria,
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
    if spec.get("model") not in FAMILIES:
        raise TrialLogError(f"{final_trial_id} is not a candidate model: nothing to evaluate")
    n_states = int(spec["n_states"])
    frozen = spec["model"] == MODEL_KMEANS_FROZEN
    jump_penalty = None if frozen else float(spec["jump_penalty"])
    columns = registered_columns(log)
    recipe_for(config)  # a configuration the features cannot be built from fails here
    # Every file against its hash, and coverage over the whole snapshot, by dates alone.
    days = snapshot_days(snapshot_dir, config.series, config.start)
    holdout_days = int((days >= pd.Timestamp(config.holdout_start)).sum())
    if holdout_days < MIN_HOLDOUT_DAYS:
        raise TrialLogError(
            f"the snapshot has {holdout_days} holdout days; at least {MIN_HOLDOUT_DAYS} are needed"
        )

    log.open_holdout(final_trial_id)  # recorded before any holdout row is used
    curve = load_curve(
        snapshot_dir, config.series, config.start, config.holdout_start, final_evaluation=True
    )
    result = holdout_evaluation(
        prepare(curve, config, columns), n_states, jump_penalty, frozen=frozen
    )
    payload = {**asdict(result), "passed": result.passed, "final_trial_id": final_trial_id}
    log.record_holdout_result(payload)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / HOLDOUT_JSON).write_text(
        json.dumps(payload, indent=2), encoding="utf-8", newline="\n"
    )
    return result
