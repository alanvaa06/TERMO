from __future__ import annotations

import json
from pathlib import Path

from termo.config import Thresholds
from termo.report import (
    REPORT_JSON,
    REPORT_MD,
    CoreReport,
    Criterion,
    Verdict,
    blocking_criteria,
    decide,
    render_markdown,
    write_report,
)
from termo.validation.bootstrap import EtaDifference

THRESHOLDS = Thresholds(
    stability_min=0.6,
    independence_p_max=0.01,
    min_median_duration_days=20,
    pbo_max=0.05,
    collinearity_max=0.8,
)
GOOD_DIFFERENCE = EtaDifference(point=0.08, low=0.02, high=0.15)


def test_all_blocking_criteria_pass() -> None:
    criteria = blocking_criteria(0.7, GOOD_DIFFERENCE, 0.001, THRESHOLDS)
    assert [c.name for c in criteria] == [
        "stability",
        "separation_vs_inertia_low95",
        "independence_p",
    ]
    assert all(c.passed and c.blocking for c in criteria)
    assert decide(criteria) is Verdict.GO


def test_each_blocking_criterion_can_veto() -> None:
    overlapping = EtaDifference(point=0.08, low=-0.01, high=0.15)
    cases = [
        blocking_criteria(0.59, GOOD_DIFFERENCE, 0.001, THRESHOLDS),
        blocking_criteria(0.7, overlapping, 0.001, THRESHOLDS),
        blocking_criteria(0.7, GOOD_DIFFERENCE, 0.01, THRESHOLDS),
    ]
    for criteria in cases:
        assert sum(not c.passed for c in criteria) == 1
        assert decide(criteria) is Verdict.NO_GO


def test_non_blocking_failure_does_not_veto() -> None:
    criteria = (
        *blocking_criteria(0.7, GOOD_DIFFERENCE, 0.001, THRESHOLDS),
        Criterion("pbo", 0.4, "<= 0.05", passed=False, blocking=False),
    )
    assert decide(criteria) is Verdict.GO


def test_report_files(tmp_path: Path) -> None:
    criteria = blocking_criteria(0.7, GOOD_DIFFERENCE, 0.001, THRESHOLDS)
    report = CoreReport(
        verdict=decide(criteria),
        final_trial_id="jm_k3_lam80",
        criteria=criteria,
        notes=("FTIC prefers K=2; the score prefers K=3.",),
        details={"n_trials": 24},
    )
    write_report(report, tmp_path / "reports")
    markdown = (tmp_path / "reports" / REPORT_MD).read_text(encoding="utf-8")
    assert markdown == render_markdown(report)
    assert "**Verdict: GO**" in markdown
    assert "| stability | 0.7000 | >= 0.6 | pass | yes |" in markdown
    assert "FTIC prefers K=2" in markdown
    assert markdown.isascii()
    payload = json.loads((tmp_path / "reports" / REPORT_JSON).read_text(encoding="utf-8"))
    assert payload["verdict"] == "go"
    assert payload["final_trial_id"] == "jm_k3_lam80"
    assert payload["criteria"][0] == {
        "name": "stability",
        "value": 0.7,
        "requirement": ">= 0.6",
        "passed": True,
        "blocking": True,
    }
    assert payload["details"] == {"n_trials": 24}


def test_markdown_lists_families_and_disclosure_when_present() -> None:
    report = CoreReport(
        Verdict.NO_GO,
        "kmf_k3",
        criteria=(),
        details={
            "chosen_family": "kmeans_frozen",
            "families": {
                "jump": {
                    "verdict": "no-go",
                    "final_trial_id": "jm_k2_lam0.5",
                    "gate": {"eta_difference_low": -0.01},
                },
                "kmeans_frozen": {
                    "verdict": "no-go",
                    "final_trial_id": "kmf_k3",
                    "gate": {"eta_difference_low": -0.002},
                },
            },
            "disclosure": {
                "n_trials_this_log": 33,
                "prior_logs": [{"path": "trials/trials.jsonl", "n_trials": 30, "verdict": "no-go"}],
                "n_trials_total": 63,
                "note": "The pre-holdout data were already examined.",
            },
        },
    )
    text = render_markdown(report)
    assert (
        "## Families" in text and "| kmeans_frozen | NO-GO | `kmf_k3` | -0.0020 | chosen |" in text
    )
    assert "## Disclosure" in text
    assert "- Trials in this log: 33" in text
    assert "- trials/trials.jsonl: 30 trials, verdict no-go" in text
    assert "- Total registered trials: 63" in text
    assert text.isascii()
