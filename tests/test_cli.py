from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from conftest import fake_fred
from termo.cli import _pre_holdout_curve, code_identity, main
from termo.config import CoreConfig
from termo.core import take_snapshot
from termo.data.loader import load_curve
from termo.validation.trials import TrialLogError

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "core.yaml"
SMALL_CONFIG = """\
series: [DGS1, DGS2, DGS3, DGS5, DGS7, DGS10, DGS30]
start: 1990-01-01
holdout_start: 1998-04-01
first_train_end: 1994-06-30
refit_weeks: 26
burn_in_days: 252
grid: {k: [2], jump_penalty: [10, 50]}
horizons_days: {short: 20, long: 65}
thresholds:
  stability_min: 0.6
  independence_p_max: 0.01
  min_median_duration_days: 20
  pbo_max: 0.05
  collinearity_max: 0.8
bootstrap: {block_weeks: 8, n_resamples: 200, seed: 0}
ftic: {k0: 3, mean_phase_days: 40, saturated_k: 6, max_jump_fraction: 0.4}
pbo_blocks: 4
effective_n_cut: 0.2
"""


def git(repo: Path, *args: str) -> None:
    identity = ["-c", "user.email=test@example.com", "-c", "user.name=test"]
    subprocess.run(["git", *identity, *args], cwd=repo, check=True, capture_output=True)


def commit_all(repo: Path, message: str) -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def code_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "src" / "model.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "configs" / "core.yaml").write_text("a: 1\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    git(repo, "init", "-q")
    commit_all(repo, "code")
    return repo


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


def test_code_identity_follows_the_code_not_the_data_or_the_history(code_repo: Path) -> None:
    """Committing snapshots and trials, or rewording a commit, must not break the binding."""
    registered = code_identity(code_repo)
    assert registered.count(".") == 2 and not registered.endswith("-dirty")

    (code_repo / "trials").mkdir()
    (code_repo / "trials" / "trials.jsonl").write_text("{}\n", encoding="utf-8")
    assert code_identity(code_repo) == registered  # an uncommitted trial log is not code
    commit_all(code_repo, "trials")
    assert code_identity(code_repo) == registered  # HEAD moved, the code did not
    git(code_repo, "commit", "-q", "--amend", "-m", "trials, reworded")
    assert code_identity(code_repo) == registered  # history rewritten, same code


def test_code_identity_changes_exactly_when_the_code_changes(code_repo: Path) -> None:
    registered = code_identity(code_repo)

    (code_repo / "src" / "model.py").write_text("x = 2\n", encoding="utf-8")
    assert code_identity(code_repo) == f"{registered}-dirty"
    commit_all(code_repo, "change the code")
    assert code_identity(code_repo) not in {registered, f"{registered}-dirty"}

    (code_repo / "src" / "model.py").write_text("x = 1\n", encoding="utf-8")
    commit_all(code_repo, "change it back")
    assert code_identity(code_repo) == registered  # same code, same identity

    (code_repo / "src" / "extra.py").write_text("y = 0\n", encoding="utf-8")
    assert code_identity(code_repo) == f"{registered}-dirty"  # untracked code counts

    (code_repo / "src" / "extra.py").unlink()
    (code_repo / "configs" / "core.yaml").write_text("a: 2\n", encoding="utf-8")
    assert code_identity(code_repo) == f"{registered}-dirty"  # configuration counts


def test_code_identity_does_not_depend_on_the_current_directory(
    code_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It describes the repository the package runs from, not wherever the command is typed."""
    own = code_identity()
    other = code_identity(code_repo)
    monkeypatch.chdir(code_repo)
    assert code_identity() == own != other
    monkeypatch.chdir(tmp_path)  # not a repository at all
    assert code_identity() == own


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


def test_the_command_line_runs_the_stages_and_enforces_the_binding(
    tmp_path: Path,
    curve: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """snapshot, register, run and report through `main`, as a user would call them."""
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "core.yaml"
    config_path.write_text(SMALL_CONFIG, encoding="utf-8")
    code = {"identity": "code-1-dirty"}
    monkeypatch.setattr("termo.cli.http_get", fake_fred(curve))
    monkeypatch.setattr("termo.cli.code_identity", lambda: code["identity"])
    base = ["--config", str(config_path)]

    assert main(["snapshot", *base]) == 0
    (snapshot,) = (tmp_path / "data" / "snapshots").iterdir()
    stage = [*base, "--snapshot", str(snapshot)]
    log_path = tmp_path / "trials" / "trials.jsonl"

    with pytest.raises(TrialLogError, match="commit the code"):
        main(["register", *stage])
    assert not log_path.exists()

    code["identity"] = "code-1"
    assert main(["register", *stage]) == 0
    assert main(["run", *stage]) == 0
    assert main(["report", *stage]) == 0

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    kinds = [record["kind"] for record in records]
    assert kinds == ["registered"] * 5 + ["result"] * 4 + ["report"]
    assert {record["code_commit"] for record in records if record["kind"] == "registered"} == {
        "code-1"
    }
    assert records[-1]["code_commit"] == "code-1"
    on_disk = json.loads((tmp_path / "reports" / "go_no_go.json").read_text(encoding="utf-8"))
    assert on_disk["verdict"] == records[-1]["verdict"]
    output = capsys.readouterr().out
    assert output.isascii()
    assert all(line.startswith("[ok]") for line in output.splitlines())

    code["identity"] = "code-2"  # the code changed after registering
    for name in ("run", "report"):
        with pytest.raises(TrialLogError, match="code commit"):
            main([name, *stage])
    with pytest.raises(TrialLogError):  # no GO verdict, or the changed code: refused either way
        main(["final-holdout", *stage])
    assert log_path.read_text(encoding="utf-8").count("\n") == len(records)


EXP2_SMALL_CONFIG = (
    (
        SMALL_CONFIG
        + """feature_set: tyccles
tyccles:
  change_horizons_days: [21, 42, 63, 84, 126, 189]
  rank_windows_days: [126, 252]
  vol_window_days: 21
  vol_rank_window_days: 252
apply_collinearity_rule: false
jump_penalty_per_feature: true
frozen_train_end: 1995-12-31
"""
    )
    .replace("burn_in_days: 252", "burn_in_days: 504")
    .replace("jump_penalty: [10, 50]", "jump_penalty: [0.5, 3]")
)


def test_trials_and_reports_directories_are_arguments(
    tmp_path: Path,
    curve: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "exp2.yaml"
    config_path.write_text(EXP2_SMALL_CONFIG, encoding="utf-8")
    monkeypatch.setattr("termo.cli.http_get", fake_fred(curve))
    monkeypatch.setattr("termo.cli.code_identity", lambda: "code-2")
    base = ["--config", str(config_path)]
    assert main(["snapshot", *base]) == 0
    (snapshot,) = (tmp_path / "data" / "snapshots").iterdir()
    stage = [
        *base,
        "--snapshot",
        str(snapshot),
        "--trials-dir",
        "trials/exp2",
        "--reports-dir",
        "reports/exp2",
    ]
    assert main(["register", *stage]) == 0
    assert (tmp_path / "trials" / "exp2" / "trials.jsonl").exists()
    assert not (tmp_path / "trials" / "trials.jsonl").exists()
    assert main(["run", *stage]) == 0
    assert (tmp_path / "trials" / "exp2" / "labels" / "kmf_k2.csv").exists()
    assert main(["report", *stage]) == 0
    out = capsys.readouterr().out
    assert out.isascii() and "reports/exp2/go_no_go.md" in out
    assert (tmp_path / "reports" / "exp2" / "go_no_go.json").exists()
