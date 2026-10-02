from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from conftest import fake_fred
from termo.cli import _pre_holdout_curve, code_commit, main
from termo.config import CoreConfig
from termo.core import take_snapshot
from termo.data.loader import load_curve

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "core.yaml"


def test_stages_other_than_snapshot_need_a_snapshot_directory(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(REPO_CONFIG)])
    assert error.value.code == 2
    assert "--snapshot is required" in capsys.readouterr().err


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(SystemExit) as error:
        main(["tune"])
    assert error.value.code == 2


def test_code_commit_is_a_git_hash() -> None:
    commit = code_commit().removesuffix("-dirty")
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)


def git(repo: Path, *args: str) -> None:
    identity = ["-c", "user.email=test@example.com", "-c", "user.name=test"]
    subprocess.run(["git", *identity, *args], cwd=repo, check=True, capture_output=True)


def test_code_commit_follows_the_code_not_the_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Committing snapshots, trials or reports after registering must not break the binding."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "src" / "model.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "configs" / "core.yaml").write_text("a: 1\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "code")
    monkeypatch.chdir(repo)
    registered = code_commit()
    assert not registered.endswith("-dirty")

    (repo / "trials").mkdir()
    (repo / "trials" / "trials.jsonl").write_text("{}\n", encoding="utf-8")
    assert code_commit() == registered  # an uncommitted trial log is not a code change
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "trials")
    assert code_commit() == registered  # HEAD moved, the code did not

    (repo / "src" / "model.py").write_text("x = 2\n", encoding="utf-8")
    assert code_commit() == f"{registered}-dirty"
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "change the code")
    assert code_commit() not in {registered, f"{registered}-dirty"}


def test_ordinary_stages_never_load_the_holdout(
    tmp_path: Path, curve: pd.DataFrame, config: CoreConfig
) -> None:
    """The snapshot holds holdout rows; the curve the stages work with must not."""
    snapshot = tmp_path / "snap"
    take_snapshot(config, snapshot, fake_fred(curve), "2026-10-02T00:00:00+00:00")
    holdout_start = pd.Timestamp(config.holdout_start)
    everything = load_curve(
        snapshot, config.series, config.start, config.holdout_start, final_evaluation=True
    )
    assert everything.index[-1] >= holdout_start

    working = _pre_holdout_curve(config, snapshot)

    assert working.index[-1] < holdout_start
    assert working.index[-1] == everything.index[everything.index < holdout_start][-1]
