# Operación en modo sombra (spec 4) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Operate the registered descriptive tool `desc2` week by week: a new FRED snapshot, a recomputation that must reproduce the registered history byte for byte, a reading of the last Friday with an alert on a phase change, a Spanish weekly sheet, a monthly report, a shadow evaluation every 26 weeks, and five CSV exports. Nothing predictive anywhere.

**Architecture:** A new package `src/termo/operation/` on top of the existing stages. `shadow.py` recomputes the full analysis from a fresh snapshot (`prepare` + `analyse` from `descriptive/stages.py`), checks the registered period against the logged hashes, builds the reading with `reading.build_reading`, decides the alert, and appends to an append-only `ShadowLog` (same JSONL pattern as `TrialLog`). `sheet_es.py` and `monthly.py` render Spanish Markdown from JSON payloads. `macro.py` reads the two context series from the snapshot. `exports.py` writes the CSVs. `shadow_eval.py` runs the registered criteria on shadow days. `op_cli.py` exposes `sombra`, `ficha`, `evaluar-sombra`, `exportar`. The registered configuration `configs/desc2.yaml`, `desc_cli.py` and the trial logs are not modified.

**Tech Stack:** Python 3.14, pandas 3, numpy 2.5, xgboost 3.4.1, shap 0.52.0, pytest, ruff, mypy. Project venv: `.venv\Scripts\python.exe`.

**Spec:** [`docs/superpowers/specs/2026-10-06-operacion-sombra-design.md`](../specs/2026-10-06-operacion-sombra-design.md). Read §3 (shadow stage), §4 (sheet), §5 (monthly), §8 (CSV) before the matching task.

**How this plan is written.** The two previous plans carried every line of code; their implementers still had to adapt lines to ruff/mypy and to the real signatures. This plan gives, per task: the exact public interface (signatures, dataclass fields, dict keys, file names), the algorithm where it is not obvious, and complete test code. Implementers write the module bodies to those interfaces, following the style of `src/termo/descriptive/stages.py` and `src/termo/reading.py`.

**Rules that bind every task:**
- Nothing about days AFTER a reading date is ever computed or printed. Historical durations and transitions are over the PAST and carry the label the spec fixes.
- Console output ASCII only. Spanish files (sheet, monthly report, CSV headers) are UTF-8 with accents.
- Never modify: `configs/desc2.yaml`, `configs/desc.yaml`, `configs/exp2.yaml`, `configs/core.yaml`, anything under `trials/`, `reports/`, `data/`, `src/termo/desc_cli.py` (except the one extraction in Task 1), `src/termo/descriptive/criteria.py`.
- Do not run the real CLI against the repo's data during Tasks 1–8; Task 9 does it once, by the orchestrator.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- After each task: `.venv\Scripts\python.exe -m pytest -q` green, `ruff check src tests`, `ruff format --check src tests`, `mypy src` clean.

**Existing interfaces you will call** (read the files; do not re-implement):
- `termo.core.take_snapshot(config, snapshot_dir, get, downloaded_at)`; `termo.data.snapshot.write_snapshot(dir, {series_id: csv_text}, downloaded_at)`, `read_snapshot(dir) -> {series_id: text}`, `snapshot_hash(dir)`; `termo.data.fred.fetch_series_csv(series_id, get)`, `parse_series_csv(text, series_id) -> pd.Series`, `http_get`; `termo.data.loader.load_curve(dir, series, start, holdout_start, *, end=None, final_evaluation=False)`, `complete_days`, `check_coverage`.
- `termo.dataset.prepare(curve, config, columns)`; `termo.core.registered_columns(log)`, `verify_data_binding(log, config, snapshot_hash)` (configuration + snapshot; the shadow stage uses only its configuration half — see Task 3), `config_fingerprint`, `environment_fingerprint`; `termo.cli.code_identity()`.
- `termo.descriptive.stages`: `Analysis(labels, frozen_labels, proba, shap_blocks, shap_top)`, `analyse(data)`, `restrict(analysis, start)`, `analysis_bytes(analysis) -> {name: bytes}`, `hashes_of(contents)`, `write_files(contents, dir)`, `load_analysis(dir, hashes)`, `FILES`, `BASE`, `PRE_HOLDOUT_DIR`, `HOLDOUT_DIR`, `holdout_result(log)`, `diagnostic_report(log)`, `failed_checks(report)`, `desc_trial_id(config)`, `DISCLAIMER`.
- `termo.reading.build_reading(analysis, day, config, validation, generated_with) -> dict` (keys: date, phase, phase_name, days_in_phase, episode_start, probabilities, confidence, surrogate_agrees, low_confidence, drivers, top_variables, drivers_validated, validation, generated_with, disclaimer), `span_periods(before, after)`.
- `termo.descriptive.criteria.evaluate(labels, curve, proba, frozen_labels, config)`, `verdict(checks)`, `phase_table(labels, curve, proba, config)`, `calibration(labels, proba, bins)`, `APTO`, `NO_APTO`.
- `termo.validation.trials.TrialLog` as the model for an append-only JSONL log (`_append`, `records`).
- `termo.validation.metrics.run_lengths(labels) -> list[(state, length)]`, `BP_PER_PERCENT`.

---

## File structure

| File | Responsibility |
|---|---|
| `src/termo/descriptive/stages.py` (modify) | `validation_from_log(log, config) -> dict` extracted from `desc_cli._read` (Task 1) |
| `src/termo/desc_cli.py` (modify, Task 1 only) | `_read` calls `validation_from_log` |
| `src/termo/operation/__init__.py` (new, empty) | |
| `src/termo/operation/config.py` | `OperationConfig`, `MacroSeries`, `load_operation_config` |
| `configs/operacion.yaml` | the operation configuration |
| `src/termo/operation/snapshot.py` | `take_operation_snapshot`: curve series + macro series in one snapshot |
| `src/termo/operation/macro.py` | `macro_panel(snapshot_dir, op_config, day) -> list[dict]`, `macro_frame(...)` |
| `src/termo/operation/shadow.py` | `ShadowLog`, `reading_date`, `history_check`, `alert_for`, `run_shadow` |
| `src/termo/operation/sheet_es.py` | `render_sheet(payload) -> str` (Spanish Markdown) |
| `src/termo/operation/monthly.py` | `episodes`, `durations_by_phase`, `transitions`, `build_monthly`, `render_monthly` |
| `src/termo/operation/exports.py` | `export_all(...)`: the five CSV |
| `src/termo/operation/shadow_eval.py` | `evaluate_shadow(...)` |
| `src/termo/op_cli.py` | stages `sombra`, `ficha`, `evaluar-sombra`, `exportar` |
| tests: `tests/test_op_config.py`, `test_op_snapshot_macro.py`, `test_shadow.py`, `test_sheet_es.py`, `test_monthly.py`, `test_exports.py`, `test_shadow_eval.py`, `test_op_cli.py`; `tests/conftest.py` (modify) | |

---

### Task 1: Extract `validation_from_log` and write the operation configuration

**Files:** modify `src/termo/descriptive/stages.py`, `src/termo/desc_cli.py`; create `src/termo/operation/__init__.py`, `src/termo/operation/config.py`, `configs/operacion.yaml`, `tests/test_op_config.py`; modify `tests/test_desc_cli.py` only if an import changes.

- [ ] **Step 1: Tests.** `tests/test_op_config.py`:

```python
"""The operation configuration: macro series, alert rule, cadences, fixed texts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from termo.operation.config import OperationConfig, load_operation_config

REPO_OP = Path(__file__).resolve().parents[1] / "configs" / "operacion.yaml"


def test_loads_the_repository_operation_configuration() -> None:
    op = load_operation_config(REPO_OP)
    assert [m.series_id for m in op.macro] == ["THREEFYTP10", "DFF"]
    assert op.macro[0].name == "prima por plazo 10 anos (Kim-Wright)"
    assert op.macro[1].spread_against == "DGS2"  # the proxy is DGS2 - DFF
    assert op.macro[0].spread_against is None
    assert op.alert_min_confidence == 0.6
    assert op.percentile_window_days == 2520 and op.change_days == 21
    assert op.shadow_eval_min_weeks == 26
    assert op.registry == "desc2" and op.model_config == Path("configs/desc2.yaml")
    assert "no anticipa" in op.texts["descargo"]
    assert "sobreconfiado" in op.texts["nota_confianza"]
    assert "no pronóstico" in op.texts["rotulo_historia"]


def test_unknown_keys_and_bad_values_are_refused(tmp_path: Path) -> None:
    text = REPO_OP.read_text(encoding="utf-8")
    bad = tmp_path / "op.yaml"
    bad.write_text(text.replace("alert_min_confidence: 0.6", "alert_min_confidenc: 0.6"), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_operation_config(bad)
    op = load_operation_config(REPO_OP)
    with pytest.raises(ValueError):
        replace(op, alert_min_confidence=1.5)
    with pytest.raises(ValueError):
        replace(op, shadow_eval_min_weeks=0)
    with pytest.raises(ValueError, match="distinct"):
        replace(op, macro=(op.macro[0], op.macro[0]))
```

Also add to `tests/test_desc_cli.py` nothing; the extraction must keep every CLI test green.

- [ ] **Step 2: Interface.**

`src/termo/operation/config.py`:

```python
@dataclass(frozen=True)
class MacroSeries:
    series_id: str          # FRED id
    name: str               # Spanish label used in sheets and CSV
    spread_against: str | None  # when set, the panel shows (this curve series - series_id)

@dataclass(frozen=True)
class OperationConfig:
    registry: str                     # "desc2"
    model_config: Path                # configs/desc2.yaml
    macro: tuple[MacroSeries, ...]
    alert_min_confidence: float       # (0, 1]
    percentile_window_days: int       # >= 252
    change_days: int                  # >= 1
    shadow_eval_min_weeks: int        # >= 1
    texts: dict[str, str]             # keys: descargo, nota_confianza, rotulo_historia, no_dice
    # __post_init__: ranges above; macro series ids distinct ("distinct"); texts has the four keys

def load_operation_config(path: Path) -> OperationConfig  # refuses unknown keys ("unknown keys in ...") at top level and in each macro entry
```

`configs/operacion.yaml` (UTF-8):

```yaml
# TERMO spec 4: operacion en modo sombra. No forma parte de la huella registrada de desc2.
registry: desc2
model_config: configs/desc2.yaml
macro:
  - series_id: THREEFYTP10
    name: prima por plazo 10 anos (Kim-Wright)
    spread_against: null
  - series_id: DFF
    name: fed funds efectiva
    spread_against: DGS2
alert_min_confidence: 0.6
percentile_window_days: 2520
change_days: 21
shadow_eval_min_weeks: 26
texts:
  descargo: "TERMO describe la fase actual de la curva. No anticipa la tasa a 10 años: en 62 pruebas registradas las fases no superaron a la inercia."
  nota_confianza: "El imitador es sobreconfiado: cuando dice 0.98 acierta cerca del 89% (medido en holdout). La confianza es su probabilidad, no una probabilidad calibrada."
  rotulo_historia: "Frecuencias del pasado, no pronóstico; no probadas en holdout."
  no_dice: "Lo que el modelo NO dice: hacia dónde irá la tasa ni qué posición tomar."
```

`stages.validation_from_log(log: TrialLog, config: CoreConfig) -> dict[str, Any]`: exactly the dict `desc_cli._read` builds today (keys diagnostic, holdout, failed_checks, fidelity_failed, by_phase, registered_verdicts, optional holdout_seen_by), raising `TrialLogError("no reading: run the run and report stages first")` when there is no diagnostic report or no result. `_read` calls it. `LOST_HOLDOUT` stays where it is.

- [ ] **Step 3: Run, lint, types, commit** `feat: operation configuration; validation status extracted from the reading CLI`.

---

### Task 2: Operation snapshot and macro panel

**Files:** create `src/termo/operation/snapshot.py`, `src/termo/operation/macro.py`, `tests/test_op_snapshot_macro.py`; modify `tests/conftest.py` (add `fake_fred_with_macro`).

- [ ] **Step 1: Tests.** In `tests/conftest.py` add:

```python
def make_macro(curve: pd.DataFrame, seed: int = 5) -> pd.DataFrame:
    """Two context series on the curve's calendar plus weekends for DFF, as FRED serves them."""
    rng = np.random.default_rng(seed)
    premium = pd.Series(np.cumsum(0.01 * rng.normal(size=len(curve))) + 1.0, index=curve.index)
    daily = pd.date_range(curve.index[0], curve.index[-1], freq="D")
    fed_funds = pd.Series(np.round(3.0 + np.cumsum(0.002 * rng.normal(size=len(daily))), 2), index=daily)
    return pd.DataFrame({"THREEFYTP10": premium, "DFF": fed_funds})


def fake_fred_frames(frames: Sequence[pd.DataFrame]) -> Callable[[str], str]:
    """An HTTP getter serving several frames, one FRED CSV per column; NaN rows are blank."""

    def get(url: str) -> str:
        for frame in frames:
            for series_id in frame.columns:
                if url == FRED_URL_TEMPLATE.format(series_id=series_id):
                    column = frame[series_id].dropna()
                    lines = [f"observation_date,{series_id}"]
                    lines += [f"{d:%Y-%m-%d},{v:.4f}" for d, v in column.items()]
                    return "\n".join(lines) + "\n"
        raise AssertionError(f"unexpected url {url}")

    return get
```

`tests/test_op_snapshot_macro.py`:

```python
"""One snapshot holds the curve and the two context series; the panel reads them."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import fake_fred_frames, make_macro
from termo.config import CoreConfig
from termo.data.snapshot import read_snapshot, snapshot_hash
from termo.operation.config import load_operation_config
from termo.operation.macro import macro_frame, macro_panel
from termo.operation.snapshot import take_operation_snapshot

REPO_OP = Path(__file__).resolve().parents[1] / "configs" / "operacion.yaml"


@pytest.fixture(scope="module")
def op_snapshot(tmp_path_factory: pytest.TempPathFactory, curve: pd.DataFrame, desc2_config: CoreConfig) -> Path:
    target = tmp_path_factory.mktemp("op") / "snap"
    op = load_operation_config(REPO_OP)
    take_operation_snapshot(desc2_config, op, target, fake_fred_frames([curve, make_macro(curve)]), "2026-10-06T00:00:00+00:00")
    return target


def test_snapshot_holds_curve_and_macro_series(op_snapshot: Path, desc2_config: CoreConfig) -> None:
    texts = read_snapshot(op_snapshot)
    assert set(texts) == set(desc2_config.series) | {"THREEFYTP10", "DFF"}
    assert len(snapshot_hash(op_snapshot)) == 64


def test_a_hole_in_the_curve_refuses_the_snapshot_but_a_hole_in_macro_does_not(
    tmp_path: Path, curve: pd.DataFrame, desc2_config: CoreConfig
) -> None:
    from termo.data.fred import DataValidationError

    op = load_operation_config(REPO_OP)
    holed = curve.copy()
    holed.iloc[500:560, 3] = np.nan
    with pytest.raises(DataValidationError):
        take_operation_snapshot(desc2_config, op, tmp_path / "a", fake_fred_frames([holed, make_macro(curve)]), "now")
    assert not (tmp_path / "a").exists()
    macro = make_macro(curve)
    macro.iloc[500:560, 0] = np.nan  # the premium is missing for three months: allowed, noted later
    take_operation_snapshot(desc2_config, op, tmp_path / "b", fake_fred_frames([curve, macro]), "now")
    assert (tmp_path / "b" / "THREEFYTP10.csv").exists()


def test_macro_frame_is_on_the_curve_calendar_with_the_spread(op_snapshot: Path, curve: pd.DataFrame, desc2_config: CoreConfig) -> None:
    op = load_operation_config(REPO_OP)
    frame = macro_frame(op_snapshot, desc2_config, op)
    assert list(frame.columns) == ["THREEFYTP10", "DGS2_menos_DFF"]
    assert frame.index.equals(curve.index)  # business days of the curve, weekends dropped
    macro = make_macro(curve)
    day = curve.index[1000]
    assert frame.loc[day, "THREEFYTP10"] == pytest.approx(macro.loc[day, "THREEFYTP10"])
    assert frame.loc[day, "DGS2_menos_DFF"] == pytest.approx(curve.loc[day, "DGS2"] - macro.loc[day, "DFF"])


def test_macro_panel_gives_value_percentile_and_change_by_hand(op_snapshot: Path, curve: pd.DataFrame, desc2_config: CoreConfig) -> None:
    op = load_operation_config(REPO_OP)
    frame = macro_frame(op_snapshot, desc2_config, op)
    day = curve.index[2000]
    panel = macro_panel(frame, op, day.date())
    assert [row["serie"] for row in panel] == ["prima por plazo 10 anos (Kim-Wright)", "DGS2 - fed funds efectiva"]
    window = frame["THREEFYTP10"].loc[:day].iloc[-op.percentile_window_days:]
    assert panel[0]["valor"] == pytest.approx(float(frame.loc[day, "THREEFYTP10"]))
    assert panel[0]["percentil_10a"] == pytest.approx(float((window <= window.iloc[-1]).mean()))
    assert panel[0]["cambio_21d"] == pytest.approx(float(frame["THREEFYTP10"].loc[:day].iloc[-1] - frame["THREEFYTP10"].loc[:day].iloc[-1 - op.change_days]))
    assert panel[0]["dias_de_ventana"] == min(op.percentile_window_days, len(frame.loc[:day]))
    assert panel[0]["nota"] == ""


def test_a_missing_macro_value_uses_the_last_available_and_says_so(op_snapshot: Path, curve: pd.DataFrame, desc2_config: CoreConfig) -> None:
    op = load_operation_config(REPO_OP)
    frame = macro_frame(op_snapshot, desc2_config, op)
    frame = frame.copy()
    day = curve.index[2100]
    frame.loc[curve.index[2095]:day, "THREEFYTP10"] = np.nan
    panel = macro_panel(frame, op, day.date())
    assert panel[0]["valor"] == pytest.approx(float(frame["THREEFYTP10"].loc[:curve.index[2094]].iloc[-1]))
    assert "último dato" in panel[0]["nota"] and curve.index[2094].date().isoformat() in panel[0]["nota"]
```

- [ ] **Step 2: Interface.**

`take_operation_snapshot(config: CoreConfig, op: OperationConfig, snapshot_dir: Path, get, downloaded_at: str) -> None`: fetch curve series (coverage checked exactly as `core.take_snapshot`) and macro series (parsed to prove they are valid CSV; no coverage check), then one `write_snapshot` with all texts. Nothing written if the curve check fails.

`macro_frame(snapshot_dir, config, op) -> pd.DataFrame`: index = business days of the curve from `load_curve(..., final_evaluation=True)`; one column per macro series: the raw series reindexed to that calendar (`reindex`, no fill) when `spread_against` is None, else `curve[spread_against] - series` named `f"{spread_against}_menos_{series_id}"`.

`macro_panel(frame, op, day: date) -> list[dict]`: per column, using rows `<= day` only: `valor` = last non-NaN value at or before day; `fecha_valor`; `percentil_10a` = share of the last `percentile_window_days` non-NaN values `<= valor`; `cambio_21d` = valor minus the value `change_days` rows earlier (non-NaN rows); `dias_de_ventana`; `nota` = "" or "último dato disponible: AAAA-MM-DD" when the value at `day` is NaN; `serie` = the Spanish name (for spreads: `f"{spread_against} - {name}"`).

- [ ] **Step 3: Run, lint, types, commit** `feat: operation snapshot with macro series; context panel`.

---

### Task 3: The shadow stage

**Files:** create `src/termo/operation/shadow.py`, `tests/test_shadow.py`; modify `tests/conftest.py` (a module-level helper that builds a finished desc2 registry on the synthetic curve is already in `tests/test_desc_stages.py` as fixtures — move the reusable parts into a `tests/desc2_fixture.py` helper module if that keeps the test files small; otherwise duplicate the minimal setup).

- [ ] **Step 1: Tests.** `tests/test_shadow.py` must cover, on the synthetic curve with the `desc2_config` fixture (holdout from 1998-04-01, curve ending ~1999-03):

```python
"""The weekly shadow run: reproduce the registered history, read the last Friday, alert, log."""
# Fixtures (module scope): a workspace with configs (desc2 small yaml with holdout_already_seen),
# a registered+run+reported+holdout desc2 registry built with the stage functions on a snapshot
# that ends at the registered end (curve.iloc[:-60]); then an "operation snapshot" built from the
# FULL curve plus macro (fake_fred_frames([curve, make_macro(curve)])) so the shadow run has 60 new days.

def test_reading_date_is_the_last_friday_with_data():
    # reading_date(index) where index ends on a Wednesday -> previous Friday;
    # ends on Friday -> that Friday; the Friday is missing (holiday) -> Thursday of that week.

def test_history_check_passes_when_the_past_is_reproduced_and_lists_days_when_not():
    # history_check(analysis, registered_hashes_pre, registered_hashes_hold, pre_end, hold_end) -> HistoryCheck(reproduced: bool, differing: dict[file, list[date]])
    # with the untouched curve -> reproduced True, differing == {}
    # with one yield altered on a past day (curve.loc[d, "DGS5"] += 0.05) -> reproduced False and d (and later days, since ranks look back) appear in differing["labels.csv"] or differing["proba.csv"]

def test_alert_rule_by_hand():
    # alert_for(previous_phase=2, reading={"phase": 1, "confidence": 0.7, "phase_name": "rally moderado", "date": "..."}, names, min_conf=0.6) -> Alert(de="venta", a="rally moderado", ...)
    # same phase -> None; different phase with confidence 0.5 -> None

def test_shadow_run_logs_reading_hashes_history_check_and_alert(shadow_workspace):
    # run_shadow(op, config, model_log, op_snapshot_dir, trials_dir, reports_dir, commit) -> ShadowRun
    # - the ShadowLog has one record with: run_at, snapshot_hash, reading_date, reading (full dict), files (5 hashes), history (reproduced, differing), alert, macro (panel list), code_commit
    # - trials/desc2/sombra/<reading_date>/ holds the five files matching the hashes
    # - the reading's phase equals the recomputed labels at that date; days_in_phase counts back into the holdout and pre-holdout (spanned analysis)
    # - the first run compares against the last holdout day's phase from the registered files (alert or not accordingly)
    # - a second run with the same snapshot appends a second record (no overwrite), identical reading

def test_shadow_run_refuses_a_changed_model_configuration_or_columns():
    # a config with refit_weeks changed -> TrialLogError("configuration"); nothing logged, nothing written

def test_shadow_run_marks_a_revised_history_but_still_produces_the_reading():
    # snapshot where a past yield was altered -> record.history.reproduced False, reading present, sheet has "HISTORIA REVISADA"
```

Write these as real tests (not comments) with the fixtures described; use `tests/test_desc_stages.py` as the model for building the registry.

- [ ] **Step 2: Interface.**

```python
SHADOW_DIR = "sombra"            # trials/<registry>/sombra/sombra.jsonl and trials/<registry>/sombra/<reading_date>/
class ShadowLog:                 # append-only JSONL, same mechanics as TrialLog._append/records
    def __init__(self, path: Path, clock=...)
    def records(self) -> list[dict]
    def append(self, record: Mapping[str, Any]) -> None
    def last_reading(self) -> dict | None   # last record with kind == "reading"

def reading_date(index: pd.DatetimeIndex) -> pd.Timestamp
    # last Friday <= index[-1] that is in index; if that Friday is not in index, the last index day
    # in the same ISO week (Mon-Fri) that is <= it; if the week has none, step back one week.

@dataclass(frozen=True)
class HistoryCheck: reproduced: bool; differing: dict[str, list[str]]  # file -> ISO dates
def history_check(analysis: Analysis, log: TrialLog, trials_dir: Path, pre_end: pd.Timestamp, hold_end: pd.Timestamp) -> HistoryCheck
    # bytes of restrict_between(analysis, start=None, end=pre_end) vs the logged pre_holdout hashes;
    # bytes of restrict_between(analysis, hold_start, hold_end) vs the logged holdout hashes.
    # On mismatch, load the registered files and list the dates whose rows differ, per file.

@dataclass(frozen=True)
class Alert: fecha: str; de: str; a: str; confianza: float
def alert_for(previous_phase: int | None, reading: Mapping[str, Any], names: Sequence[str], min_confidence: float) -> Alert | None

@dataclass(frozen=True, eq=False)
class ShadowRun: record: dict[str, Any]; analysis: Analysis; sheet_path: Path
def run_shadow(op: OperationConfig, config: CoreConfig, model_log: TrialLog, snapshot_dir: Path, trials_dir: Path, reports_dir: Path, code_commit: str, echo=print) -> ShadowRun
```

`run_shadow` algorithm:
1. `verify_configuration(model_log, config)`: the configuration half of `verify_data_binding` (write a small `core.verify_configuration(log, config)` that reuses `_data_binding` logic for the configuration pair only; the snapshot is new every week so it is not compared). Registered columns from the log; `recipe_for(config).names == columns` else refuse.
2. `curve = load_curve(snapshot_dir, config.series, config.start, config.holdout_start, final_evaluation=True)`; `data = prepare(curve, config, columns)`; `analysis = analyse(data)`.
3. `history = history_check(...)` with `pre_end` = last date of the registered pre-holdout labels (from the logged file), `hold_end` = last date of the registered holdout labels.
4. `day = reading_date(analysis.labels.index)`; `validation = validation_from_log(model_log, config)`; `validation["sombra"] = {"semanas": n_prior_readings + 1, "historia_reproducida": history.reproduced, "proxima_evaluacion_semanas": max(0, op.shadow_eval_min_weeks - n)}`; `reading = build_reading(analysis, day.date(), config, validation, generated_with)`.
5. previous phase = last shadow reading's phase, else the registered holdout labels' last value; `alert = alert_for(...)`.
6. `panel = macro_panel(macro_frame(snapshot_dir, config, op), op, day.date())`.
7. Save the five files of `restrict(analysis, hold_end + 1 day)` (shadow days only) under `trials_dir/SHADOW_DIR/<day>/`, hashes in the record; append the record (kind "reading"); render the sheet (Task 4) to `reports_dir/SHADOW_DIR/<day>.md` + `.json`; return.
   Order: compute everything, append the record, then write files and sheet (as the holdout stage does).

- [ ] **Step 3: Run, lint, types, commit** `feat: shadow stage with history reproduction, reading date, alert and log`.

---

### Task 4: Spanish weekly sheet

**Files:** create `src/termo/operation/sheet_es.py`, `tests/test_sheet_es.py`; `shadow.py` calls `render_sheet`.

- [ ] **Step 1: Tests.** `render_sheet(record: Mapping[str, Any], op: OperationConfig) -> str` on a hand-built record (reading dict as `build_reading` returns + `macro`, `history`, `alert`, `snapshot_hash`, `run_at`). Assert, in order of appearance: title "TERMO — hoja semanal del <fecha>"; snapshot date and hash prefix; alert line "CAMBIO DE FASE: venta → rally moderado" when present; banner "NO VALIDADO" when the governing verdict is not apto; "HISTORIA REVISADA" with the count of differing days when not reproduced; "Fase actual: **venta**" and "desde <fecha> (<n> días hábiles)"; confidence with `texts["nota_confianza"]`; probability table with the three names; "Qué la empuja" with three blocks and signs and five variables; "Contexto macro" table with serie, valor, percentil, cambio, nota; "Validación" with registered verdicts, the per-phase table, "semanas de sombra: n", "próxima evaluación de sombra en k semanas", and the holdout_seen_by line; `texts["descargo"]`; "Generado con código <commit>". Assert the text contains "á" or "ó" (UTF-8 Spanish) and contains none of: "pronóstico" outside the fixed texts, "siguiente mes", "próximo movimiento", "va a subir", "va a bajar". A second test: an APTO record without alert has neither banner nor alert line.

- [ ] **Step 2: Interface.** `render_sheet(record, op) -> str` and `write_sheet(record, op, out_dir) -> Path` (writes `<fecha>.md` UTF-8 and `<fecha>.json`). Numbers: probabilities and confidence with 2 decimals, contributions `+0.00`, macro values 2 decimals, percentiles as percentages without decimals.

- [ ] **Step 3: Run, lint, types, commit** `feat: Spanish weekly sheet`.

---

### Task 5: Monthly report

**Files:** create `src/termo/operation/monthly.py`, `tests/test_monthly.py`.

- [ ] **Step 1: Tests.**

```python
def test_episodes_durations_and_transitions_by_hand():
    # labels: 2 x10 days, 1 x5, 0 x7, 2 x3, 1 x4  (index of business days)
    # episodes(labels, curve, names) -> DataFrame rows: inicio, fin, fase, nombre, dias, cambio_10y_pb, cambio_2y_pb
    #   5 rows; dias == [10,5,7,3,4]; cambio_10y_pb == 100*(DGS10[fin]-DGS10[inicio]) per row
    # durations_by_phase(episodes_df, names) -> per phase: episodios, mediana_dias, p25, p75 (hand values)
    # transitions(episodes_df, names) -> matrix 3x3 of shares by row: from 2: {1: 0.5, 0: 0.5}? compute by hand from the sequence 2->1->0->2->1: from 2: to 1 twice (1.0); from 1: to 0 once (1.0, the last episode has no successor); from 0: to 2 once (1.0). Rows sum to 1 or are all NaN when the phase never has a successor.

def test_monthly_payload_covers_the_month_only_from_shadow_readings_and_registered_history(...):
    # build_monthly(month="1999-02", shadow_log, model_log, trials_dir, analysis_latest, curve, config, op, macro_frame) -> dict
    # keys: mes, fase_cierre, desde, lecturas (list of {fecha, fase, nombre, confianza}), alertas, bloques_promedio (6 blocks averaged over the month's readings), duraciones (durations_by_phase over the whole history up to month end), transiciones, rotulo (op.texts["rotulo_historia"]), macro (panel at month end with cambio over the month), validacion, no_dice, descargo, generado_con
    # lecturas are exactly the shadow readings whose reading_date falls in the month
    # duraciones use labels up to the last day of the month only (nothing after)

def test_monthly_render_is_spanish_and_has_no_forecast_section():
    # render_monthly(payload, op) -> str; contains "Ficha mensual", "Duraciones históricas", "Transiciones", the rotulo, "Lo que el modelo NO dice", descargo; does not contain "Movimiento típico" nor "Qué suele venir"
```

- [ ] **Step 2: Interface.** As in the tests; `write_monthly(payload, op, out_dir) -> Path` writes `reports/<registry>/fichas/AAAA-MM.md` + `.json`. Transitions table in Markdown: rows "de", columns "a", cells as percentages; a header line with the rotulo text directly above the durations and transitions sections.

- [ ] **Step 3: Run, lint, types, commit** `feat: monthly report with historical durations and transitions, labelled`.

---

### Task 6: CSV exports

**Files:** create `src/termo/operation/exports.py`, `tests/test_exports.py`.

- [ ] **Step 1: Tests.** `export_all(shadow_log, model_log, trials_dir, analysis_latest, curve, config, op, macro_frame, out_dir, snapshot_hash) -> dict[str, Path]` writes `lecturas.csv`, `historia_diaria.csv`, `episodios.csv`, `macro.csv`, `alertas.csv` into `out_dir`. Each file starts with two comment lines `# snapshot_hash=<hash>` and `# generado=<ISO timestamp>` then the header. Tests: read each with `pd.read_csv(path, comment="#")`; exact column lists per spec §8 (write them out in the test); `historia_diaria` has one row per labelled day and `periodo` in {pre_holdout, holdout, sombra} with the boundaries at `holdout_start` and at the registered holdout end; `episodios` matches `monthly.episodes`; `lecturas` has one row per shadow reading with `bloque_1..3` and `aporte_1..3`; `alertas` has one row per alert (possibly zero rows with header); UTF-8 with Spanish header names where the spec uses them (`dias_en_fase`, `inicio_episodio`, ...). Numeric columns must be floats, not strings.

- [ ] **Step 2: Run, lint, types, commit** `feat: CSV exports`.

---

### Task 7: Shadow evaluation

**Files:** create `src/termo/operation/shadow_eval.py`, `tests/test_shadow_eval.py`.

- [ ] **Step 1: Tests.** `evaluate_shadow(op, config, model_log, shadow_log, trials_dir, reports_dir, curve_latest, analysis_latest, code_commit, min_weeks=None) -> dict`: refuses (`TrialLogError`) when the shadow log has fewer than `op.shadow_eval_min_weeks` readings (test with `min_weeks` overridden to 2 to make the synthetic case runnable); evaluates D1, D2, D3, D5, D6 (`criteria.evaluate` with the desc2 config, which has no claim) on the days AFTER the registered holdout end, with `frozen_labels` from a jump model fitted once on data through the holdout end (`prepare(curve, replace(config, frozen_train_end=hold_end.date()), columns)` → `analyse` → its `frozen_labels`); payload keys: stage "sombra", verdict, checks, by_phase, days, weeks, period (first and last day), calibration, limits, disclosure (from the model log); appended to the shadow log with kind "evaluation"; `reports/<registry>/sombra_eval_<fecha>.md/.json` written in Spanish (reuse `phase_rows` style, Spanish headers); the model trial log `trials/<registry>/trials.jsonl` is byte-unchanged.

- [ ] **Step 2: Run, lint, types, commit** `feat: shadow evaluation of the registered criteria on shadow days`.

---

### Task 8: Operation command line

**Files:** create `src/termo/op_cli.py`, `tests/test_op_cli.py`.

- [ ] **Step 1: Tests.** `main(argv)` with stages: `sombra --snapshot <dir>` (or `--download` to take a new operation snapshot into `data/snapshots/<today>/` first; in tests always `--snapshot`), `ficha --mes AAAA-MM --snapshot <dir>`, `evaluar-sombra --snapshot <dir> [--min-weeks N]`, `exportar --snapshot <dir>`; `--operacion configs/operacion.yaml` default; the model config path comes from the operation config; registry directories from `op.registry`; `allow_abbrev=False`; ASCII console output (`[ok] hoja -> reports/desc2/sombra/1999-02-26.md`, etc.). Drive the whole flow in a temp dir as `tests/test_desc_cli.py` does: build the desc2 registry with `termo.desc_cli.main` (register/run/report/holdout with the flag) on a snapshot ending at the registered end, take an operation snapshot of the full curve, then `sombra` twice (second with a snapshot 5 more days? keep simple: same snapshot, two records), `exportar`, `ficha` for the month of the reading, `evaluar-sombra --min-weeks 1`. Assert files exist, console lines are ASCII, the sheet is UTF-8 Spanish, and the model trial log is unchanged.

- [ ] **Step 2: Run, lint, types, commit** `feat: operation command line (sombra, ficha, evaluar-sombra, exportar)`.

---

### Task 9: First real shadow run (orchestrator)

- [ ] Take the first operation snapshot: `.venv\Scripts\python.exe -m termo.op_cli sombra --download` (downloads FRED into `data/snapshots/<today>/`, then runs). Expected console: `[ok] historia reproducida` (or `[x] HISTORIA REVISADA: n dias`), `[ok] lectura <viernes> fase=<nombre> conf=<x>`, `[ok] hoja -> ...`, `[ok] csv -> reports/desc2/csv`.
- [ ] Commit `data/snapshots/<today>`, `trials/desc2/sombra`, `reports/desc2/sombra`, `reports/desc2/csv`.
- [ ] `ficha --mes 2026-09` and `--mes 2026-10`; commit `reports/desc2/fichas`.
- [ ] Update `docs/context/*`; present the first Spanish sheet to the user.

---

## Self-review

- **Spec coverage:** §3.1 snapshot (T2), §3.2 history reproduction (T3), §3.3 reading date and alert (T3), §3.4 shadow log (T3), §4 sheet (T4), §5 monthly (T5), §6 shadow evaluation (T7), §7 macro (T2), §8 CSV (T6), §9 config (T1), §10 modules and CLI (T1–T8), first real run (T9).
- **Placeholders:** none; where code bodies are not given, the interface and algorithm are fixed and the tests pin the behaviour.
- **Type consistency:** `OperationConfig`/`MacroSeries` fields used identically in T2–T8; `ShadowLog.records/append/last_reading`; `HistoryCheck(reproduced, differing)`; `Alert(fecha, de, a, confianza)`; `run_shadow(...) -> ShadowRun(record, analysis, sheet_path)`; `macro_frame(snapshot_dir, config, op)` and `macro_panel(frame, op, day)`; `validation_from_log(log, config)`; `episodes/durations_by_phase/transitions/build_monthly/render_monthly/write_monthly`; `export_all(...) -> dict[str, Path]`; `evaluate_shadow(...) -> dict`.
