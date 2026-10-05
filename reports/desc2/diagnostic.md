# TERMO - descriptive diagnostic

**Verdict: APTO** (pre-holdout, seen data)

TERMO describes the current phase of the curve. It does not anticipate the 10Y: in 62 registered trials the phases did not beat inertia.

| Check | Value | Requirement | Result |
|---|---|---|---|
| D1_persistence | 54.0000 | >= 20 days | pass |
| D2_sell_coherence | 0.8001 | >= 0.6 | pass |
| D3_rally_coherence | 0.8009 | >= 0.6 | pass |
| D5_map_stable | 0.9712 | >= 0.6 | pass |
| D6_fidelity | 0.8274 | >= 0.8 | pass |

Days evaluated: 9194

## By phase

| Phase | Days | Evaluable | Median duration (days) | Direction share | Recall | Episodes |
|---|---|---|---|---|---|---|
| rally fuerte | 1852 | yes | 74.0 | 0.82 | 0.88 | 21 |
| rally moderado | 3019 | yes | 54.0 | 0.80 | 0.72 | 48 |
| venta | 4323 | yes | 82.0 | 0.80 | 0.88 | 53 |

## What these checks do not prove

- K, the penalty and the phase names were chosen looking at pre-holdout data.
- D2-D3 are partly true by construction: the phases are built from ranks of these changes.
- The fidelity threshold is a judgement, not a calibrated value.
- SHAP explains the surrogate, not the market and not the jump model.
- Nothing here measures the ability to anticipate; that question is closed with NO-GO.
- Phase names describe direction and intensity only; no claim about the shape of the curve is made or tested.

## Disclosure

- Trials in this log: 1
- trials/trials.jsonl: 29 trials, verdict no-go
- trials/exp2/trials.jsonl: 33 trials, verdict no-go
- trials/desc/trials.jsonl: 1 trials, verdict no-apto
- Total registered trials: 64
