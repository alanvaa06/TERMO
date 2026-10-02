from __future__ import annotations

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
