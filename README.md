# TERMO — Treasury Environment & Regime MOnitor

Weekly context tool that tells an investment committee which **phase** the US Treasury market is in, **how confident** it is, **which inputs drive** the reading, and **what usually comes next** historically.

TERMO is a thermometer, not an autopilot: it does not forecast yields and does not issue trade orders.

## Status

Design phase. No code yet.

- [`docs/design/TERMO-diseno-comite.md`](docs/design/TERMO-diseno-comite.md) — committee design (what and why, plain language)
- [`docs/design/TERMO-diseno-tecnico.md`](docs/design/TERMO-diseno-tecnico.md) — technical design (features, jump model, traceability, validation protocol, operations)

## At a glance

| Item | Choice |
|---|---|
| Data | Public US Treasury curve (FRED); macro panel from public Fed / CME sources |
| Regime engine | Statistical jump model (persistent regimes, penalty-controlled) |
| Confidence | Continuous jump model probabilities |
| Traceability | XGBoost translator + SHAP, cross-checked against regime centroids |
| Cadence | Weekly read, monthly committee report, alert on regime change, semiannual retrain |

## Weekly run

```bash
python run_termo.py --download
python run_termo.py --snapshot data/snapshots/<fecha> --mes AAAA-MM
python run_termo.py --solo-reporte output/<fecha> --snapshot data/snapshots/<fecha>
```

The first takes today's snapshot, logs the shadow reading and builds the report; the
second runs on an existing snapshot and also writes the monthly ficha; the third only
rebuilds the report from the files of a past week (the model does not run).

The deliverable lands in `output/<reading date>/`: `reporte.html` (visual report with
Plotly embedded; works offline; not versioned, rebuild it with `--solo-reporte`),
`hoja.md/json`, the five CSV and, with `--mes`, the ficha. An optional
`output/<fecha>/comentario.md` written by the analyst appears in the report as
"Comentario del analista".
