"""The spec 3 command line on the synthetic curve."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from conftest import fake_fred
from termo.config import load_config
from termo.core import environment_fingerprint, take_snapshot
from termo.desc_cli import LOST_HOLDOUT, REPORTS_DIR, TRIALS_DIR, TRIALS_FILE, main
from termo.descriptive.criteria import APTO, NO_APTO
from termo.descriptive.stages import (
    HOLDOUT_DIR,
    PRE_HOLDOUT_DIR,
    failed_checks,
    holdout_desc,
    holdout_result,
)
from termo.validation.trials import TrialLog, TrialLogError
from test_cli import EXP2_SMALL_CONFIG

DESC_SMALL_CONFIG = (
    EXP2_SMALL_CONFIG.replace(
        "grid: {k: [2], jump_penalty: [0.5, 3]}", "grid: {k: [3], jump_penalty: [0.5]}"
    ).replace("frozen_train_end: 1995-12-31", "frozen_train_end: 1996-06-28")
    + """\
descriptive:
  phase_names: [rally de la parte corta, rally de la parte larga, venta]
  sell_phase: 2
  short_led_phase: 0
  long_led_phase: 1
  level_series: DGS10
  slope_long: DGS10
  slope_short: DGS2
  change_days: 63
  coherence_share_min: 0.6
  min_days_evaluable: 40
  map_ari_min: 0.6
  fidelity_min: 0.8
  low_confidence_below: 0.6
  surrogate: {n_estimators: 20, max_depth: 3, learning_rate: 0.1, subsample: 0.8,
    colsample_bytree: 0.8, seed: 0}
  blocks:
    nivel corto: [d1, d2, d3]
    nivel medio: [d5, d7]
    nivel largo: [d10, d30]
    pendientes: [s12m5s, s3s10, s10s30]
    curvatura: [c5]
    volatilidad: [vol1, vol2, vol3, vol5, vol7, vol10, vol30]
"""
)
COMMIT = "code-3"
PRE_HOLDOUT_DATE, HOLDOUT_DATE = "1997-06-02", "1998-06-01"


def _stage(root: Path) -> list[str]:
    return ["--config", str(root / "desc.yaml"), "--snapshot", str(root / "snapshot")]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory, curve: pd.DataFrame) -> Path:
    """register, run and report done once through the command line, in their own directory."""
    assert "k: [3]" in DESC_SMALL_CONFIG and "1996-06-28" in DESC_SMALL_CONFIG
    root = tmp_path_factory.mktemp("termo-desc-cli")
    (root / "desc.yaml").write_text(DESC_SMALL_CONFIG, encoding="utf-8")
    config = load_config(root / "desc.yaml")
    take_snapshot(config, root / "snapshot", fake_fred(curve), "2026-10-02T00:00:00+00:00")
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        patch.setattr("termo.desc_cli.code_identity", lambda: COMMIT)
        assert main(["register", *_stage(root)]) == 0
        assert main(["run", *_stage(root)]) == 0
        assert main(["report", *_stage(root)]) == 0
    return root


@pytest.fixture
def here(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The shared workspace as the working directory; tests that use it leave the log alone."""
    monkeypatch.chdir(workspace)
    monkeypatch.setattr("termo.desc_cli.code_identity", lambda: COMMIT)
    return workspace


@pytest.fixture
def private(workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private copy of the finished trials and reports, for tests that change them."""
    shutil.copytree(workspace / TRIALS_DIR, tmp_path / TRIALS_DIR)
    shutil.copytree(workspace / REPORTS_DIR, tmp_path / REPORTS_DIR)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("termo.desc_cli.code_identity", lambda: COMMIT)
    return workspace


def _log() -> TrialLog:
    return TrialLog(TRIALS_DIR / TRIALS_FILE)


def _reading(day: str) -> dict[str, object]:
    path = REPORTS_DIR / "readings" / f"{day}.json"
    found: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return found


def _markdown(day: str) -> str:
    return (REPORTS_DIR / "readings" / f"{day}.md").read_text(encoding="utf-8")


def _banner(period: str, report: dict[str, object]) -> str:
    failed = ", ".join(failed_checks(report)) or "none evaluable"  # type: ignore[arg-type]
    return f"**[NOT VALIDATED: {period} {str(report['verdict']).upper()}; failed: {failed}]**"


def _approve(log: TrialLog) -> None:
    """Append a diagnostic report that says APTO, so that the holdout may open."""
    last = log.last_report()
    assert last is not None
    payload = {k: v for k, v in last.items() if k not in {"kind", "trial_id", "at"}}
    log.record_report({**payload, "verdict": APTO})


def test_stages_and_reading_through_the_command_line(
    here: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (TRIALS_DIR / PRE_HOLDOUT_DIR / "proba.csv").exists()
    assert (REPORTS_DIR / "diagnostic.md").exists()
    assert main(["read", *_stage(here), "--date", PRE_HOLDOUT_DATE]) == 0
    markdown = _markdown(PRE_HOLDOUT_DATE)
    assert "does not anticipate" in markdown and markdown.isascii()
    assert capsys.readouterr().out.isascii()
    report = _log().last_report()
    assert report is not None and report["verdict"] == NO_APTO  # the synthetic curve fails
    failed = failed_checks(report)
    assert failed  # a NO-APTO with named failures: the banner must name them
    reading = _reading(PRE_HOLDOUT_DATE)
    assert reading["validation"] == {
        "diagnostic": NO_APTO,
        "holdout": None,
        "failed_checks": failed,
        "fidelity_failed": "D6_fidelity" in failed,
    }
    assert reading["drivers_validated"] is ("D6_fidelity" not in failed)
    # the sheet is produced anyway, and says before anything else that it is not validated
    banner = _banner("pre-holdout diagnostic", report)
    assert markdown.splitlines()[2] == banner and "NO-APTO" in banner
    assert reading["drivers"] and "## What pushes toward this phase" in markdown
    assert markdown.rstrip().endswith(f"Generated with code {COMMIT}")

    # a holdout date has no reading until a holdout result exists
    with pytest.raises(TrialLogError, match="no reading"):
        main(["read", *_stage(here), "--date", HOLDOUT_DATE])
    # the holdout does not open without the explicit flag, nor with an abbreviation of it
    with pytest.raises(SystemExit):
        main(["holdout", *_stage(here)])
    with pytest.raises(SystemExit):
        main(["holdout", *_stage(here), "--i"])
    assert not _log().holdout_opened()
    # a date that is not YYYY-MM-DD is refused with a message, not a traceback
    with pytest.raises(SystemExit):
        main(["read", *_stage(here), "--date", "02/06/1997"])
    error = capsys.readouterr().err
    assert "02/06/1997" in error and "YYYY-MM-DD" in error and error.isascii()


def test_a_lost_holdout_is_said_in_the_reading_and_closes_holdout_dates(private: Path) -> None:
    _log().open_holdout("desc_k3")  # opened, and no result ever recorded
    assert main(["read", *_stage(private), "--date", PRE_HOLDOUT_DATE]) == 0
    validation = _reading(PRE_HOLDOUT_DATE)["validation"]
    assert isinstance(validation, dict) and validation["holdout"] == LOST_HOLDOUT
    assert LOST_HOLDOUT == "opened, no result (lost)"
    markdown = _markdown(PRE_HOLDOUT_DATE)
    assert f"- Holdout: {LOST_HOLDOUT}" in markdown
    report = _log().last_report()
    assert report is not None  # the diagnostic still governs: its banner, its failures
    assert markdown.splitlines()[2] == _banner("pre-holdout diagnostic", report)
    assert validation["failed_checks"] == failed_checks(report)
    with pytest.raises(TrialLogError, match="lost"):
        main(["read", *_stage(private), "--date", HOLDOUT_DATE])


def test_a_reading_binds_the_data_and_records_the_code_without_requiring_it(
    private: Path, curve: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Later code or libraries still read this log; another configuration or snapshot do not."""
    monkeypatch.setattr("termo.desc_cli.code_identity", lambda: "code-4")
    registered = environment_fingerprint()
    monkeypatch.setattr(
        "termo.core.environment_fingerprint", lambda: {**registered, "numpy": "0.0.0"}
    )
    # the stages that compute are still bound to the registered code and environment
    for stage in ("run", "report"):
        with pytest.raises(TrialLogError, match="code commit"):
            main([stage, *_stage(private)])
    with pytest.raises(TrialLogError, match="environment"):
        monkeypatch.setattr("termo.desc_cli.code_identity", lambda: COMMIT)
        main(["run", *_stage(private)])
    monkeypatch.setattr("termo.desc_cli.code_identity", lambda: "code-4")

    assert main(["read", *_stage(private), "--date", PRE_HOLDOUT_DATE]) == 0
    reading = _reading(PRE_HOLDOUT_DATE)
    assert reading["generated_with"] == {
        "code_commit": "code-4",
        "registered_code_commit": COMMIT,
        "environment": registered,
    }
    assert (
        _markdown(PRE_HOLDOUT_DATE)
        .rstrip()
        .endswith(f"Generated with code code-4 (registered: {COMMIT})")
    )

    changed = Path("changed.yaml")
    changed.write_text(
        DESC_SMALL_CONFIG.replace("refit_weeks: 26", "refit_weeks: 13"), encoding="utf-8"
    )
    assert "refit_weeks: 13" in changed.read_text(encoding="utf-8")
    with pytest.raises(TrialLogError, match="configuration"):
        main(
            [
                "read",
                "--config",
                str(changed),
                "--snapshot",
                str(private / "snapshot"),
                "--date",
                PRE_HOLDOUT_DATE,
            ]
        )

    other = Path("other-snapshot")
    take_snapshot(
        load_config(private / "desc.yaml"),
        other,
        fake_fred(curve + 0.05),
        "2026-10-03T00:00:00+00:00",
    )
    with pytest.raises(TrialLogError, match="data snapshot"):
        main(
            [
                "read",
                "--config",
                str(private / "desc.yaml"),
                "--snapshot",
                str(other),
                "--date",
                PRE_HOLDOUT_DATE,
            ]
        )


def test_a_holdout_reading_counts_the_episode_from_before_the_holdout(private: Path) -> None:
    config = load_config(private / "desc.yaml")
    _approve(_log())
    verdict = holdout_desc(config, private / "snapshot", _log(), TRIALS_DIR, REPORTS_DIR, COMMIT)
    assert verdict in {APTO, NO_APTO}

    def labels(directory: str) -> pd.Series:
        path = TRIALS_DIR / directory / "labels.csv"
        return pd.read_csv(path, parse_dates=["date"], index_col="date")["label"]

    before, after = labels(PRE_HOLDOUT_DIR), labels(HOLDOUT_DIR)
    start = pd.Timestamp(config.holdout_start)
    assert before.index[-1] < start <= after.index[0]
    # the synthetic curve does not change phase at the boundary: the episode began earlier
    assert before.iloc[-1] == after.iloc[0]
    first = after.index[0]
    assert main(["read", *_stage(private), "--date", first.date().isoformat()]) == 0
    reading = _reading(first.date().isoformat())
    assert reading["phase"] == int(after.iloc[0])
    holdout = holdout_result(_log())
    assert holdout is not None and holdout["verdict"] == verdict
    assert reading["validation"] == {
        "diagnostic": APTO,
        "holdout": verdict,
        "failed_checks": failed_checks(holdout),
        "fidelity_failed": "D6_fidelity" in failed_checks(holdout),
    }
    markdown = _markdown(first.date().isoformat())
    assert "- Pre-holdout diagnostic: APTO" in markdown
    assert f"- Holdout: {verdict.upper()}" in markdown
    if verdict == APTO:
        assert "NOT VALIDATED" not in markdown
    else:  # the holdout governs the banner once it has a result
        assert markdown.splitlines()[2] == _banner("holdout", holdout)

    whole = pd.concat([before, after]).loc[:first]
    changed = whole.ne(whole.iloc[-1]).to_numpy().nonzero()[0]
    episode = whole.index[changed[-1] + 1] if len(changed) else whole.index[0]
    assert episode < start
    assert reading["episode_start"] == episode.date().isoformat()
    assert reading["days_in_phase"] == len(whole.loc[episode:])
    assert reading["days_in_phase"] > 1  # the holdout alone would say 1
