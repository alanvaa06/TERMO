# TERMO - descriptive diagnostic

**Verdict: APTO** (pre-holdout, seen data)

TERMO describes the current phase of the curve. It does not anticipate the 10Y: in 62 registered trials the phases did not beat inertia.

| Check | Value | Requirement | Result |
|---|---|---|---|
| D1_persistence | 54.0000 | >= 20 days | pass |
| D2_sell_coherence | 0.8001 | >= 0.6 | pass |
| D3_rally_coherence | 0.8009 | >= 0.6 | pass |
| D4_short_led_steepens | 18.4449 | > 0 bp | pass |
| D4_long_led_flattens | -13.7224 | < 0 bp | pass |
| D5_map_stable | 0.9712 | >= 0.6 | pass |
| D6_fidelity | 0.8274 | >= 0.8 | pass |

Days evaluated: 9194

## What these checks do not prove

- K, the penalty and the phase names were chosen looking at pre-holdout data.
- D2-D4 are partly true by construction: the phases are built from ranks of these changes.
- The fidelity threshold is a judgement, not a calibrated value.
- SHAP explains the surrogate, not the market and not the jump model.
- Nothing here measures the ability to anticipate; that question is closed with NO-GO.

## Disclosure

- Trials in this log: 1
- trials/trials.jsonl: 29 trials, verdict no-go
- trials/exp2/trials.jsonl: 33 trials, verdict no-go
- Total registered trials: 63
