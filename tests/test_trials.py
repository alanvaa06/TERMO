from __future__ import annotations

import json
from pathlib import Path

import pytest

from termo.validation.trials import TrialLog, TrialLogError, TrialStatus


@pytest.fixture
def log(tmp_path: Path) -> TrialLog:
    return TrialLog(tmp_path / "trials" / "trials.jsonl", clock=lambda: "2026-10-02T12:00:00+00:00")


def register(log: TrialLog, trial_id: str = "jm_k2_lam50") -> None:
    log.register(trial_id, "hypothesis", {"n_states": 2, "jump_penalty": 50.0}, "hash", "commit")


def test_result_without_registration_is_refused(log: TrialLog) -> None:
    with pytest.raises(TrialLogError, match="no prior registration"):
        log.record_result("jm_k2_lam50", {"score": 0.1}, TrialStatus.KEPT, "", None)
    assert log.records() == []


def test_second_result_is_refused(log: TrialLog) -> None:
    register(log)
    log.record_result("jm_k2_lam50", {"score": 0.1}, TrialStatus.KEPT, "", "labels.csv")
    with pytest.raises(TrialLogError, match="already has a result"):
        log.record_result("jm_k2_lam50", {"score": 0.9}, TrialStatus.KEPT, "", "labels.csv")


def test_second_registration_is_refused(log: TrialLog) -> None:
    register(log)
    with pytest.raises(TrialLogError, match="already registered"):
        register(log)


def test_log_is_append_only_json_lines(log: TrialLog) -> None:
    register(log)
    before = log.path.read_text(encoding="utf-8")
    log.record_result("jm_k2_lam50", {"score": 0.1}, TrialStatus.DISCARDED, "short", None)
    after = log.path.read_text(encoding="utf-8")
    assert after.startswith(before)
    first, second = (json.loads(line) for line in after.splitlines())
    assert first["kind"] == "registered" and first["at"] == "2026-10-02T12:00:00+00:00"
    assert first["config"] == {"n_states": 2, "jump_penalty": 50.0}
    assert first["snapshot_hash"] == "hash" and first["code_commit"] == "commit"
    assert second["kind"] == "result" and second["status"] == "discarded"
    assert log.is_registered("jm_k2_lam50") and log.has_result("jm_k2_lam50")
    assert set(log.registrations()) == set(log.results()) == {"jm_k2_lam50"}


def test_holdout_opens_once_and_only_for_a_finished_trial(log: TrialLog) -> None:
    register(log)
    with pytest.raises(TrialLogError, match="no recorded result"):
        log.open_holdout("jm_k2_lam50")
    log.record_result("jm_k2_lam50", {"score": 0.1}, TrialStatus.KEPT, "", None)
    assert not log.holdout_opened()
    log.open_holdout("jm_k2_lam50")
    assert log.holdout_opened()
    with pytest.raises(TrialLogError, match="already been opened"):
        log.open_holdout("jm_k2_lam50")
