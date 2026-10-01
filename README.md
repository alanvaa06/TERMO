# TERMO — Treasury Environment & Regime MOnitor

Weekly context tool that tells an investment committee which **phase** the US Treasury market is in, **how confident** it is, **which inputs drive** the reading, and **what usually comes next** historically.

TERMO is a thermometer, not an autopilot: it does not forecast yields and does not issue trade orders.

## Status

Design phase. No code yet. See [`docs/design/TERMO-diseno-comite.md`](docs/design/TERMO-diseno-comite.md).

## At a glance

| Item | Choice |
|---|---|
| Data | Public US Treasury curve (FRED); macro panel from public Fed / CME sources |
| Regime engine | Statistical jump model (persistent regimes, penalty-controlled) |
| Confidence | Continuous jump model probabilities |
| Traceability | XGBoost translator + SHAP, cross-checked against regime centroids |
| Cadence | Weekly read, monthly committee report, alert on regime change, semiannual retrain |
