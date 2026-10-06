# TERMO - descriptive holdout

**Verdict: APTO** (holdout, data ALREADY SEEN by desc_k3 (trials/desc), opened 2026-10-05, verdict no-apto by D4 only)

TERMO describes the current phase of the curve. It does not anticipate the 10Y: in 62 registered trials the phases did not beat inertia.

| Check | Value | Requirement | Result |
|---|---|---|---|
| D1_persistence | 141.0000 | >= 20 days | pass |
| D2_sell_coherence | 0.9149 | >= 0.6 | pass |
| D3_rally_coherence | 0.7071 | >= 0.6 | pass |
| D5_map_stable | 0.9923 | >= 0.6 | pass |
| D6_fidelity | 0.8434 | >= 0.8 | pass |

Days evaluated: 499

## By phase

| Phase | Days | Evaluable | Median duration (days) | Direction share | Recall | Episodes |
|---|---|---|---|---|---|---|
| rally fuerte | 19 | no | 19.0 | 0.89 | 0.58 | 1 |
| rally moderado | 198 | yes | 198.0 | 0.71 | 0.72 | 1 |
| venta | 282 | yes | 141.0 | 0.91 | 0.96 | 2 |

## What these checks do not prove

- K, the penalty and the phase names were chosen looking at pre-holdout data.
- D2-D3 are partly true by construction: the phases are built from ranks of these changes.
- The fidelity threshold is a judgement, not a calibrated value.
- SHAP explains the surrogate, not the market and not the jump model.
- Nothing here measures the ability to anticipate; that question is closed with NO-GO.
- Phase names describe direction and intensity only; no claim about the shape of the curve is made or tested.
- The holdout period was already examined by desc_k3 (trials/desc), opened 2026-10-05, verdict no-apto by D4 only: these numbers are not clean evidence; only shadow readings are.

## Disclosure

- Trials in this log: 1
- trials/trials.jsonl: 29 trials, verdict no-go
- trials/exp2/trials.jsonl: 33 trials, verdict no-go
- trials/desc/trials.jsonl: 1 trials, verdict no-apto
- Total registered trials: 64
