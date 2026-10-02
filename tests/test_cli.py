from __future__ import annotations

from pathlib import Path

import pytest

from termo.cli import code_commit, main

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
