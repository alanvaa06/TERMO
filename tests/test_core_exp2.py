"""Experiment 2 end to end on the synthetic curve: two families, a prior registry, disclosure."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from conftest import fake_fred
from termo.config import CoreConfig
from termo.core import (
    HYPOTHESES,
    INERTIA_TRIAL,
    MODEL_KMEANS_FROZEN,
    SETUP_TRIAL,
    build_report,
    frozen_trial_id,
    jump_trial_id,
    kmeans_trial_id,
    prior_log_summary,
    publish_report,
    register_trials,
    registered_columns,
    run_final_holdout,
    run_trials,
    take_snapshot,
)
from termo.data.loader import load_curve
from termo.data.snapshot import snapshot_hash
from termo.dataset import ExperimentData, prepare
from termo.experiment import GateResult
from termo.report import CoreReport, Verdict, render_markdown, report_payload
from termo.selection import Candidate, pick_winner
from termo.validation.bootstrap import EtaDifference, IndependenceTest
from termo.validation.trials import TrialLog, TrialLogError

COMMIT = "test-commit-2"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("termo-exp2")


@pytest.fixture(scope="module")
def prior_log(workspace: Path) -> Path:
    """A closed earlier registry: two trials and a no-go report."""
    log = TrialLog(workspace / "prior" / "trials.jsonl")
    log.register(SETUP_TRIAL, "setup", {}, "old-snapshot", "old-code")
    log.register("jm_k2_lam5", "h", {"model": "jump", "n_states": 2, "jump_penalty": 5}, "s", "c")
    log.register("inertia", "h", {"model": "inertia"}, "s", "c")
    log.record_report(report_payload(CoreReport(Verdict.NO_GO, "jm_k2_lam5", criteria=())))
    return log.path


@pytest.fixture(scope="module")
def exp2_config(tyccles_config: CoreConfig, prior_log: Path) -> CoreConfig:
    return replace(tyccles_config, prior_trial_logs=(prior_log.as_posix(),))


@pytest.fixture(scope="module")
def snapshot_dir(workspace: Path, curve: pd.DataFrame, exp2_config: CoreConfig) -> Path:
    target = workspace / "data" / "snapshots" / "2026-10-02"
    take_snapshot(exp2_config, target, fake_fred(curve), "2026-10-02T00:00:00+00:00")
    return target


@pytest.fixture(scope="module")
def log(workspace: Path) -> TrialLog:
    return TrialLog(workspace / "trials" / "exp2" / "trials.jsonl")


@pytest.fixture(scope="module")
def data_hash(snapshot_dir: Path) -> str:
    return snapshot_hash(snapshot_dir)


@pytest.fixture(scope="module")
def trials_dir(workspace: Path) -> Path:
    return workspace / "trials" / "exp2"


@pytest.fixture(scope="module")
def data(
    snapshot_dir: Path, data_hash: str, log: TrialLog, exp2_config: CoreConfig, prior_log: Path
) -> ExperimentData:
    """Register once for the module and prepare the data on the registered feature set."""
    before = hashlib.sha256(prior_log.read_bytes()).hexdigest()
    curve = load_curve(
        snapshot_dir, exp2_config.series, exp2_config.start, exp2_config.holdout_start
    )
    register_trials(exp2_config, curve, log, data_hash, COMMIT)
    assert hashlib.sha256(prior_log.read_bytes()).hexdigest() == before  # never written
    return prepare(curve, exp2_config, registered_columns(log))


@pytest.fixture(scope="module")
def finished(
    workspace: Path,
    data: ExperimentData,
    data_hash: str,
    trials_dir: Path,
    log: TrialLog,
    prior_log: Path,
) -> Path:
    """Run and report once for the module. Returns the reports directory."""
    before = hashlib.sha256(prior_log.read_bytes()).hexdigest()
    messages: list[str] = []
    run_trials(data, log, trials_dir, data_hash, COMMIT, echo=messages.append)
    assert len(messages) == 9 and all(m.isascii() and m.startswith("[ok]") for m in messages)
    report = build_report(data, log, trials_dir, data_hash, COMMIT)
    publish_report(report, log, workspace / "reports" / "exp2", data_hash, COMMIT)
    assert hashlib.sha256(prior_log.read_bytes()).hexdigest() == before  # never written
    return workspace / "reports" / "exp2"


def test_both_families_are_registered_with_the_hypotheses(
    finished: Path, log: TrialLog, exp2_config: CoreConfig
) -> None:
    registrations = log.registrations()
    expected = {SETUP_TRIAL, INERTIA_TRIAL}
    for k in exp2_config.k_values:
        expected |= {jump_trial_id(k, lam) for lam in exp2_config.jump_penalties}
        expected |= {kmeans_trial_id(k), frozen_trial_id(k)}
    assert set(registrations) == expected and len(registrations) == 10
    assert registrations["kmf_k2"]["config"] == {"model": MODEL_KMEANS_FROZEN, "n_states": 2}
    setup = registrations[SETUP_TRIAL]["config"]
    assert setup["hypotheses"] == list(HYPOTHESES) and len(HYPOTHESES) == 3
    assert setup["prior_trial_logs"] == [
        {"path": exp2_config.prior_trial_logs[0], "n_trials": 2, "verdict": "no-go"}
    ]
    assert len(setup["columns"]) == 139


def test_prior_logs_are_counted_at_registration(tmp_path: Path, tyccles_config: CoreConfig) -> None:
    assert prior_log_summary(tyccles_config) == []
    missing = replace(tyccles_config, prior_trial_logs=((tmp_path / "nope.jsonl").as_posix(),))
    with pytest.raises(TrialLogError, match="prior trial log not found"):
        prior_log_summary(missing)


def test_frozen_results_carry_the_family_markers(finished: Path, log: TrialLog) -> None:
    results = log.results()
    frozen = results["kmf_k2"]["metrics"]
    assert frozen["frozen"] is True and frozen["s1_mean"] is None
    assert frozen["stability"] == frozen["s2"]
    assert results["kmf_k2"]["status"] == ("kept" if frozen["passes_duration"] else "discarded")
    assert results["kmf_k2"]["reason"] != "baseline"
    jump = results["jm_k2_lam0.5"]["metrics"]
    assert jump["frozen"] is False and jump["jump_penalty_effective"] == pytest.approx(69.5)
    assert results[kmeans_trial_id(2)]["reason"] == "baseline"


def test_the_report_judges_each_family_and_discloses_every_trial(
    finished: Path, log: TrialLog, exp2_config: CoreConfig
) -> None:
    payload = json.loads((finished / "go_no_go.json").read_text(encoding="utf-8"))
    details = payload["details"]
    families = details["families"]
    assert set(families) == {"jump", "kmeans_frozen"}
    for family, outcome in families.items():
        assert outcome["verdict"] in {"go", "no-go"}
        assert [c["name"] for c in outcome["criteria"]] == [
            "stability",
            "separation_vs_inertia_low95",
            "independence_p",
            "pbo",
            "s2_halves",
        ]
        assert outcome["final_trial_id"].startswith("jm_" if family == "jump" else "kmf_")
    chosen = details["chosen_family"]
    assert chosen in families
    assert payload["final_trial_id"] == families[chosen]["final_trial_id"]
    assert payload["verdict"] == families[chosen]["verdict"]
    assert (payload["verdict"] == "go") == any(f["verdict"] == "go" for f in families.values())
    assert families["kmeans_frozen"]["final_model"]["s1_mean"] is None
    assert families["kmeans_frozen"]["ftic_states"] is None
    disclosure = details["disclosure"]
    assert disclosure["n_trials_this_log"] == 9
    assert disclosure["prior_logs"] == [
        {"path": exp2_config.prior_trial_logs[0], "n_trials": 2, "verdict": "no-go"}
    ]
    assert disclosure["n_trials_total"] == 11
    assert set(details["baselines_separation"]) == {"inertia", "kmeans_k2", "kmeans_k3"}
    assert any("reconstructed" in line for line in details["limitations"])
    markdown = (finished / "go_no_go.md").read_text(encoding="utf-8")
    assert markdown.isascii()
    assert "## Families" in markdown and "## Disclosure" in markdown
    assert "Total registered trials: 11" in markdown
    assert log.last_report() is not None and log.last_report()["verdict"] == payload["verdict"]


def test_final_holdout_runs_the_frozen_family(
    finished: Path, tmp_path: Path, snapshot_dir: Path, log: TrialLog, exp2_config: CoreConfig
) -> None:
    copy = TrialLog(tmp_path / "trials.jsonl")
    copy.path.write_text(log.path.read_text(encoding="utf-8"), encoding="utf-8")
    copy.record_report(report_payload(CoreReport(Verdict.GO, "kmf_k2", criteria=())))
    result = run_final_holdout(exp2_config, snapshot_dir, copy, tmp_path / "reports", COMMIT)
    assert result.n_weeks > 0 and copy.records()[-1]["final_trial_id"] == "kmf_k2"
    with pytest.raises(TrialLogError, match="already been opened"):
        run_final_holdout(exp2_config, snapshot_dir, copy, tmp_path / "reports", COMMIT)


NO_ELIGIBLE = (
    "no eligible configuration: none passes the minimum median duration "
    "with positive stability and positive separation"
)


@pytest.fixture
def copy(finished: Path, log: TrialLog, tmp_path: Path) -> TrialLog:
    """A private copy of the finished log: the decision tests never touch the shared one."""
    private = TrialLog(tmp_path / "trials.jsonl")
    private.path.write_text(log.path.read_text(encoding="utf-8"), encoding="utf-8")
    return private


def fix_gates(
    monkeypatch: pytest.MonkeyPatch,
    config: CoreConfig,
    jump: tuple[float, float],
    frozen: tuple[float, float],
) -> None:
    """Replace the gate by a fixed (lower bound, p-value) pair per family.

    The frozen labels start after the frozen cutoff; the jump labels start before it.
    """
    assert config.frozen_train_end is not None
    cutoff = pd.Timestamp(config.frozen_train_end)

    def fake(
        labels: pd.Series, inertia: pd.Series, yields: pd.Series, config: CoreConfig
    ) -> GateResult:
        low, p_value = frozen if labels.index[0] > cutoff else jump
        return GateResult(
            eta_difference=EtaDifference(point=low + 0.01, low=low, high=low + 0.02),
            independence=IndependenceTest(statistic=10.0, p_value=p_value),
            n_weeks=100,
        )

    monkeypatch.setattr("termo.core.gate_tests", fake)


def test_the_two_families_are_told_apart_by_the_first_label_date(
    finished: Path, trials_dir: Path, exp2_config: CoreConfig
) -> None:
    """What fix_gates relies on."""
    assert exp2_config.frozen_train_end is not None
    cutoff = pd.Timestamp(exp2_config.frozen_train_end)
    for path in (trials_dir / "labels").glob("*.csv"):
        first = pd.read_csv(path, parse_dates=["date"])["date"].iloc[0]
        if path.stem.startswith("kmf_"):
            assert first > cutoff, path.stem
        elif path.stem.startswith("jm_"):
            assert first <= cutoff, path.stem


def test_a_failing_family_with_a_higher_lower_bound_is_never_chosen(
    data: ExperimentData,
    copy: TrialLog,
    trials_dir: Path,
    data_hash: str,
    exp2_config: CoreConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_gates(monkeypatch, exp2_config, jump=(0.01, 0.001), frozen=(0.03, 0.5))
    report = build_report(data, copy, trials_dir, data_hash, COMMIT)
    families = report.details["families"]
    assert report.verdict is Verdict.GO and report.details["chosen_family"] == "jump"
    assert report.final_trial_id is not None and report.final_trial_id.startswith("jm_")
    assert families["jump"]["verdict"] == "go"
    assert families["kmeans_frozen"]["verdict"] == "no-go"
    assert any("Family jump decides: it passes" in note for note in report.notes)


@pytest.mark.parametrize(
    ("jump_low", "frozen_low", "chosen", "prefix"),
    [(0.01, 0.03, "kmeans_frozen", "kmf_"), (0.03, 0.01, "jump", "jm_")],
)
def test_between_two_passing_families_the_higher_lower_bound_decides(
    jump_low: float,
    frozen_low: float,
    chosen: str,
    prefix: str,
    data: ExperimentData,
    copy: TrialLog,
    trials_dir: Path,
    data_hash: str,
    exp2_config: CoreConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_gates(monkeypatch, exp2_config, jump=(jump_low, 0.001), frozen=(frozen_low, 0.001))
    report = build_report(data, copy, trials_dir, data_hash, COMMIT)
    families = report.details["families"]
    assert {f["verdict"] for f in families.values()} == {"go"}
    assert report.verdict is Verdict.GO and report.details["chosen_family"] == chosen
    assert report.final_trial_id == families[chosen]["final_trial_id"]
    assert report.final_trial_id is not None and report.final_trial_id.startswith(prefix)
    assert report.details["gate"]["eta_difference_low"] == max(jump_low, frozen_low)


def test_no_family_passing_is_a_no_go_that_says_so(
    data: ExperimentData,
    copy: TrialLog,
    trials_dir: Path,
    data_hash: str,
    exp2_config: CoreConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_gates(monkeypatch, exp2_config, jump=(-0.01, 0.001), frozen=(0.03, 0.5))
    report = build_report(data, copy, trials_dir, data_hash, COMMIT)
    assert report.verdict is Verdict.NO_GO
    assert {f["verdict"] for f in report.details["families"].values()} == {"no-go"}
    assert report.details["chosen_family"] == "kmeans_frozen"  # the highest lower bound
    assert any("no family passes" in note for note in report.notes)


def test_a_family_without_an_eligible_model_stays_in_the_report(
    data: ExperimentData,
    copy: TrialLog,
    trials_dir: Path,
    data_hash: str,
    exp2_config: CoreConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_frozen_winner(candidates: Sequence[Candidate]) -> Candidate | None:
        if candidates[0].trial_id.startswith("kmf_"):
            return None
        return pick_winner(candidates)

    monkeypatch.setattr("termo.core.pick_winner", no_frozen_winner)
    report = build_report(data, copy, trials_dir, data_hash, COMMIT)
    families = report.details["families"]
    assert list(families) == ["jump", "kmeans_frozen"]
    assert families["kmeans_frozen"] == {
        "verdict": "no-go",
        "final_trial_id": None,
        "n_trials": len(exp2_config.k_values),
        "reason": NO_ELIGIBLE,
    }
    assert "Family kmeans_frozen: no eligible configuration." in report.notes
    # The other family still decides, and the report says which one did.
    assert report.details["chosen_family"] == "jump"
    assert report.final_trial_id == families["jump"]["final_trial_id"]
    assert report.final_trial_id is not None and report.final_trial_id.startswith("jm_")
    assert report.verdict.value == families["jump"]["verdict"]
    assert any(note.startswith("Family jump decides") for note in report.notes)
    markdown = render_markdown(report)
    assert markdown.isascii()
    assert "| kmeans_frozen | NO-GO | - | - |  |" in markdown
    assert "- Family kmeans_frozen: no eligible configuration." in markdown
    json.dumps(report_payload(report))  # still plain JSON for the log


def test_no_eligible_model_anywhere_is_a_no_go_listing_every_family(
    data: ExperimentData,
    copy: TrialLog,
    trials_dir: Path,
    data_hash: str,
    exp2_config: CoreConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("termo.core.pick_winner", lambda candidates: None)
    report = build_report(data, copy, trials_dir, data_hash, COMMIT)
    assert report.verdict is Verdict.NO_GO and report.final_trial_id is None
    assert report.criteria == () and "chosen_family" not in report.details
    n_jump = len(exp2_config.k_values) * len(exp2_config.jump_penalties)
    families = report.details["families"]
    assert families == {
        "jump": {
            "verdict": "no-go",
            "final_trial_id": None,
            "n_trials": n_jump,
            "reason": NO_ELIGIBLE,
        },
        "kmeans_frozen": {
            "verdict": "no-go",
            "final_trial_id": None,
            "n_trials": len(exp2_config.k_values),
            "reason": NO_ELIGIBLE,
        },
    }
    for family in families:
        assert f"Family {family}: no eligible configuration." in report.notes
    markdown = render_markdown(report)
    assert markdown.isascii()
    assert "| jump | NO-GO | - | - |  |" in markdown
    assert "| kmeans_frozen | NO-GO | - | - |  |" in markdown


def test_the_report_answers_h1(finished: Path) -> None:
    details = json.loads((finished / "go_no_go.json").read_text(encoding="utf-8"))["details"]
    assert details["hypotheses"] == list(HYPOTHESES) and len(details["hypotheses"]) == 3
    passes = details["baselines_passes_duration"]
    assert set(passes) == {"kmeans_k2", "kmeans_k3"}
    assert all(isinstance(value, bool) for value in passes.values())


def test_run_refuses_a_log_with_a_model_it_cannot_run_before_running_anything(
    data: ExperimentData,
    log: TrialLog,
    tmp_path: Path,
    data_hash: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The spec 1 runner must not spoil a descriptive registry by running what it can."""
    setup = log.registrations()[SETUP_TRIAL]
    foreign = TrialLog(tmp_path / "trials.jsonl")
    foreign.register(SETUP_TRIAL, setup["hypothesis"], setup["config"], data_hash, COMMIT)
    jump = {"model": "jump", "n_states": 2, "jump_penalty": 0.5}
    foreign.register(jump_trial_id(2, 0.5), "h", jump, data_hash, COMMIT)
    desc = {"model": "descriptive", "n_states": 3, "jump_penalty": 0.5}
    foreign.register("desc_k3", "h", desc, data_hash, COMMIT)

    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("nothing may be evaluated")

    monkeypatch.setattr("termo.core.evaluate_config", never)
    monkeypatch.setattr("termo.core.inertia_labels", never)
    with pytest.raises(TrialLogError, match="desc_k3"):
        run_trials(data, foreign, tmp_path / "out", data_hash, COMMIT, echo=never)
    assert foreign.results() == {} and not (tmp_path / "out").exists()


def test_a_final_model_that_cannot_be_evaluated_does_not_burn_the_holdout(
    finished: Path, tmp_path: Path, snapshot_dir: Path, log: TrialLog, exp2_config: CoreConfig
) -> None:
    private = TrialLog(tmp_path / "trials.jsonl")
    private.path.write_text(log.path.read_text(encoding="utf-8"), encoding="utf-8")
    private.record_report(report_payload(CoreReport(Verdict.GO, INERTIA_TRIAL, criteria=())))
    with pytest.raises(TrialLogError, match="not a candidate model"):
        run_final_holdout(exp2_config, snapshot_dir, private, tmp_path / "reports", COMMIT)
    assert not private.holdout_opened()
