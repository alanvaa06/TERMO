# TERMO - go/no-go

**Verdict: NO-GO**

Final model: `jm_k4_lam3`

| Criterion | Value | Requirement | Result | Blocking |
|---|---|---|---|---|
| stability | 0.5665 | >= 0.6 | FAIL | yes |
| separation_vs_inertia_low95 | -0.0049 | > 0 | FAIL | yes |
| independence_p | 0.0387 | < 0.01 | FAIL | yes |
| pbo | 0.2566 | <= 0.05 | FAIL | no |
| s2_halves | 0.3546 | >= 0.6 | FAIL | no |

## Notes

- FTIC prefers K=5; the score prefers K=4.
- PBO above the limit: prune the grid and repeat as new trials.
- Family kmeans_frozen: no eligible configuration.
- Family jump decides: no family passes; the others are reported above.

## Families

| Family | Verdict | Final model | Separation vs inertia (low 95%) | |
|---|---|---|---|---|
| jump | NO-GO | `jm_k4_lam3` | -0.0049 | chosen |
| kmeans_frozen | NO-GO | - | - |  |

## Disclosure

- Trials in this log: 33
- trials/trials.jsonl: 29 trials, verdict no-go
- Total registered trials: 62
- The pre-holdout data were already examined by every prior log listed here. The holdout stays closed until a GO verdict and the user's explicit approval.

## Details

```json
{
  "columns": [
    "d1_21_r126",
    "d1_21_r252",
    "d1_42_r126",
    "d1_42_r252",
    "d1_63_r126",
    "d1_63_r252",
    "d1_84_r126",
    "d1_84_r252",
    "d1_126_r126",
    "d1_126_r252",
    "d1_189_r126",
    "d1_189_r252",
    "d2_21_r126",
    "d2_21_r252",
    "d2_42_r126",
    "d2_42_r252",
    "d2_63_r126",
    "d2_63_r252",
    "d2_84_r126",
    "d2_84_r252",
    "d2_126_r126",
    "d2_126_r252",
    "d2_189_r126",
    "d2_189_r252",
    "d3_21_r126",
    "d3_21_r252",
    "d3_42_r126",
    "d3_42_r252",
    "d3_63_r126",
    "d3_63_r252",
    "d3_84_r126",
    "d3_84_r252",
    "d3_126_r126",
    "d3_126_r252",
    "d3_189_r126",
    "d3_189_r252",
    "d5_21_r126",
    "d5_21_r252",
    "d5_42_r126",
    "d5_42_r252",
    "d5_63_r126",
    "d5_63_r252",
    "d5_84_r126",
    "d5_84_r252",
    "d5_126_r126",
    "d5_126_r252",
    "d5_189_r126",
    "d5_189_r252",
    "d7_21_r126",
    "d7_21_r252",
    "d7_42_r126",
    "d7_42_r252",
    "d7_63_r126",
    "d7_63_r252",
    "d7_84_r126",
    "d7_84_r252",
    "d7_126_r126",
    "d7_126_r252",
    "d7_189_r126",
    "d7_189_r252",
    "d10_21_r126",
    "d10_21_r252",
    "d10_42_r126",
    "d10_42_r252",
    "d10_63_r126",
    "d10_63_r252",
    "d10_84_r126",
    "d10_84_r252",
    "d10_126_r126",
    "d10_126_r252",
    "d10_189_r126",
    "d10_189_r252",
    "d30_21_r126",
    "d30_21_r252",
    "d30_42_r126",
    "d30_42_r252",
    "d30_63_r126",
    "d30_63_r252",
    "d30_84_r126",
    "d30_84_r252",
    "d30_126_r126",
    "d30_126_r252",
    "d30_189_r126",
    "d30_189_r252",
    "s12m5s_21_r126",
    "s12m5s_21_r252",
    "s12m5s_42_r126",
    "s12m5s_42_r252",
    "s12m5s_63_r126",
    "s12m5s_63_r252",
    "s12m5s_84_r126",
    "s12m5s_84_r252",
    "s12m5s_126_r126",
    "s12m5s_126_r252",
    "s12m5s_189_r126",
    "s12m5s_189_r252",
    "s3s10_21_r126",
    "s3s10_21_r252",
    "s3s10_42_r126",
    "s3s10_42_r252",
    "s3s10_63_r126",
    "s3s10_63_r252",
    "s3s10_84_r126",
    "s3s10_84_r252",
    "s3s10_126_r126",
    "s3s10_126_r252",
    "s3s10_189_r126",
    "s3s10_189_r252",
    "s10s30_21_r126",
    "s10s30_21_r252",
    "s10s30_42_r126",
    "s10s30_42_r252",
    "s10s30_63_r126",
    "s10s30_63_r252",
    "s10s30_84_r126",
    "s10s30_84_r252",
    "s10s30_126_r126",
    "s10s30_126_r252",
    "s10s30_189_r126",
    "s10s30_189_r252",
    "c5_21_r126",
    "c5_21_r252",
    "c5_42_r126",
    "c5_42_r252",
    "c5_63_r126",
    "c5_63_r252",
    "c5_84_r126",
    "c5_84_r252",
    "c5_126_r126",
    "c5_126_r252",
    "c5_189_r126",
    "c5_189_r252",
    "vol1_r252",
    "vol2_r252",
    "vol3_r252",
    "vol5_r252",
    "vol7_r252",
    "vol10_r252",
    "vol30_r252"
  ],
  "configurations": [
    {
      "trial_id": "jm_k2_lam0.5",
      "family": "jump",
      "stability": 0.9102355285823582,
      "separation": -0.0005677423405124624,
      "separation_shift": -0.0021978841210521976,
      "score": -0.0005167792494149465,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k2_lam1.2",
      "family": "jump",
      "stability": 0.9223880491050257,
      "separation": -0.0014600228793410434,
      "separation_shift": -0.0029835906827016144,
      "score": -0.0013467076553240873,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k2_lam3",
      "family": "jump",
      "stability": 0.9080085877194057,
      "separation": -0.0019640366714399455,
      "separation_shift": -0.0033685740653669138,
      "score": -0.0017833621642633073,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k2_lam8",
      "family": "jump",
      "stability": 0.9191956243571179,
      "separation": -0.0019857445742211236,
      "separation_shift": -0.003124283048604636,
      "score": -0.0018252877237149448,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k2_lam20",
      "family": "jump",
      "stability": 0.8379847596458883,
      "separation": -0.0011847352171524025,
      "separation_shift": -0.0022074870584942357,
      "score": -0.0009927900561894753,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k2_lam50",
      "family": "jump",
      "stability": 0.3807259215627954,
      "separation": -0.0008253852968449678,
      "separation_shift": -0.0004955509898360098,
      "score": -0.00031424557778568176,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k3_lam0.5",
      "family": "jump",
      "stability": 0.8307217161032651,
      "separation": 0.0012321300083422675,
      "separation_shift": -0.0005847191064819873,
      "score": 0.0010235571549924189,
      "passes_duration": false
    },
    {
      "trial_id": "jm_k3_lam1.2",
      "family": "jump",
      "stability": 0.7826093540365712,
      "separation": -0.0019236240612638836,
      "separation_shift": -0.0037313682808949268,
      "score": -0.0015054461839949336,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k3_lam3",
      "family": "jump",
      "stability": 0.8362182009489076,
      "separation": -0.0023361441444855672,
      "separation_shift": -0.003965319620503446,
      "score": -0.001953526253659046,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k3_lam8",
      "family": "jump",
      "stability": 0.6774140302593311,
      "separation": -0.0014348111910399363,
      "separation_shift": -0.0024203791714584645,
      "score": -0.0009719612315835543,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k3_lam20",
      "family": "jump",
      "stability": 0.7626231899975091,
      "separation": -0.0041639031137904944,
      "separation_shift": -0.005324848907955541,
      "score": -0.0031754890754794683,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k3_lam50",
      "family": "jump",
      "stability": 0.3538140152399934,
      "separation": 0.0026347639996930636,
      "separation_shift": 0.0033673061682971458,
      "score": 0.0009322164299411876,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k4_lam0.5",
      "family": "jump",
      "stability": 0.7693791330727762,
      "separation": 0.0050277894299029825,
      "separation_shift": 0.0028083584162557637,
      "score": 0.003868276272851224,
      "passes_duration": false
    },
    {
      "trial_id": "jm_k4_lam1.2",
      "family": "jump",
      "stability": 0.6140898488040396,
      "separation": 0.004839252426911702,
      "separation_shift": 0.002752710259280501,
      "score": 0.002971735791166789,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k4_lam3",
      "family": "jump",
      "stability": 0.5664748879422363,
      "separation": 0.014005419630160157,
      "separation_shift": 0.01162736103345409,
      "score": 0.007933718515578971,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k4_lam8",
      "family": "jump",
      "stability": 0.5250994267800115,
      "separation": -0.005024853692363208,
      "separation_shift": -0.006221389980434453,
      "score": -0.0026385477935133446,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k4_lam20",
      "family": "jump",
      "stability": 0.5527228362212884,
      "separation": 0.00803800630021376,
      "separation_shift": 0.006833719236194321,
      "score": 0.004442789639818734,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k4_lam50",
      "family": "jump",
      "stability": 0.35196990982617965,
      "separation": 0.0013737266348316894,
      "separation_shift": -0.0003966143472512807,
      "score": 0.0004835104397875309,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k5_lam0.5",
      "family": "jump",
      "stability": 0.722849936490624,
      "separation": 0.002098564859046131,
      "separation_shift": -0.0005561296629944101,
      "score": 0.001516947475082951,
      "passes_duration": false
    },
    {
      "trial_id": "jm_k5_lam1.2",
      "family": "jump",
      "stability": 0.59392550004735,
      "separation": -0.0002680473767798317,
      "separation_shift": -0.0038560460603292218,
      "score": -0.00015920017229034198,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k5_lam3",
      "family": "jump",
      "stability": 0.6352307497353842,
      "separation": 0.00790986402092484,
      "separation_shift": 0.00515800982140324,
      "score": 0.005024588852317027,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k5_lam8",
      "family": "jump",
      "stability": 0.5400702941147126,
      "separation": -0.001929298462348605,
      "separation_shift": -0.0035305559459707196,
      "score": -0.001041956787995674,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k5_lam20",
      "family": "jump",
      "stability": 0.4770441078406313,
      "separation": -0.0017603584890679286,
      "separation_shift": -0.0017446897110190047,
      "score": -0.0008397686448970916,
      "passes_duration": true
    },
    {
      "trial_id": "jm_k5_lam50",
      "family": "jump",
      "stability": 0.3520155520972947,
      "separation": 0.006849783126877408,
      "separation_shift": 0.007383816518474368,
      "score": 0.002411230189154484,
      "passes_duration": false
    },
    {
      "trial_id": "kmf_k2",
      "family": "kmeans_frozen",
      "stability": 0.8466964905044113,
      "separation": 0.0005179451541572383,
      "separation_shift": -0.0011223315567533525,
      "score": 0.0004385423442987,
      "passes_duration": false
    },
    {
      "trial_id": "kmf_k3",
      "family": "kmeans_frozen",
      "stability": 0.6435050760946689,
      "separation": -0.0008967423388857644,
      "separation_shift": -0.00344279171877925,
      "score": -0.0005770582470219952,
      "passes_duration": false
    },
    {
      "trial_id": "kmf_k4",
      "family": "kmeans_frozen",
      "stability": 0.6435107348466523,
      "separation": 0.0038032842963044954,
      "separation_shift": 1.0687861947680441e-05,
      "score": 0.0024474542723456386,
      "passes_duration": false
    },
    {
      "trial_id": "kmf_k5",
      "family": "kmeans_frozen",
      "stability": 0.44171453167285707,
      "separation": -0.0038700868774219858,
      "separation_shift": -0.007221457490614328,
      "score": -0.0017094736125937223,
      "passes_duration": false
    }
  ],
  "baselines_separation": {
    "kmeans_k2": 0.0015962739752384259,
    "kmeans_k3": 0.0022787222632868753,
    "kmeans_k4": 0.005241599170240375,
    "kmeans_k5": 0.0038146467318667682,
    "inertia": 0.0032400170328432763
  },
  "baselines_separation_shift": {
    "kmeans_k2": -2.7158917154651054e-05,
    "kmeans_k3": 0.0002490479160562574,
    "kmeans_k4": 0.0026836042444208386,
    "kmeans_k5": 0.0010918467377391671,
    "inertia": 0.001997736616050268
  },
  "baselines_passes_duration": {
    "kmeans_k2": false,
    "kmeans_k3": false,
    "kmeans_k4": false,
    "kmeans_k5": false
  },
  "hypotheses": [
    "H1 (diagnostic, not a criterion): with rank features the refit K-means baselines pass the minimum median duration.",
    "H2: at least one jump model configuration passes every blocking criterion.",
    "H3: the frozen K-means passes every blocking criterion."
  ],
  "limitations": [
    "DGS30 between 2002-02-19 and 2006-02-08 is built differently from the rest of the series.",
    "S1 is high almost by construction: consecutive windows share most of their data.",
    "PBO ranks configurations by raw eta-squared, so it favours grids that mix different K.",
    "Velocities are in basis points, so the 1980s can dominate the extreme regimes.",
    "A forward change of exactly zero counts as 'down' in the independence test.",
    "separation_shift uses the older chance level (groups slid in time). It is a declared sensitivity check: it is reported and never decides.",
    "Evidence for jump models comes from equities; these tests are the criterion for rates.",
    "Ranks of multi-month changes are smooth by construction: stability and duration can pass without any information about the future; separation and independence decide.",
    "139 correlated inputs without pruning: the distance is dominated by level changes.",
    "The 102-input TYCCLES recipe is reconstructed from the text, not reproduced.",
    "The frozen K-means reads 1998-2024 with centroids from 1979-1997; HSBC fitted on 50 years.",
    "PBO over four frozen configurations is nearly blind.",
    "Second look at the same pre-holdout data: every test family has had two chances."
  ],
  "disclosure": {
    "n_trials_this_log": 33,
    "prior_logs": [
      {
        "n_trials": 29,
        "path": "trials/trials.jsonl",
        "verdict": "no-go"
      }
    ],
    "n_trials_total": 62,
    "note": "The pre-holdout data were already examined by every prior log listed here. The holdout stays closed until a GO verdict and the user's explicit approval."
  },
  "families": {
    "jump": {
      "verdict": "no-go",
      "score_winner": "jm_k4_lam3",
      "final_trial_id": "jm_k4_lam3",
      "criteria": [
        {
          "name": "stability",
          "value": 0.5664748879422363,
          "requirement": ">= 0.6",
          "passed": false,
          "blocking": true
        },
        {
          "name": "separation_vs_inertia_low95",
          "value": -0.004942166928100889,
          "requirement": "> 0",
          "passed": false,
          "blocking": true
        },
        {
          "name": "independence_p",
          "value": 0.03868269733403032,
          "requirement": "< 0.01",
          "passed": false,
          "blocking": true
        },
        {
          "name": "pbo",
          "value": 0.25664335664335663,
          "requirement": "<= 0.05",
          "passed": false,
          "blocking": false
        },
        {
          "name": "s2_halves",
          "value": 0.3545935894910687,
          "requirement": ">= 0.6",
          "passed": false,
          "blocking": false
        }
      ],
      "notes": [
        "FTIC prefers K=5; the score prefers K=4.",
        "PBO above the limit: prune the grid and repeat as new trials."
      ],
      "gate": {
        "eta_difference_point": 0.01076540259731688,
        "eta_difference_low": -0.004942166928100889,
        "eta_difference_high": 0.0393963313960134,
        "independence_statistic": 32.79320007376975,
        "n_weeks": 1913.0
      },
      "ftic_states": 5,
      "pbo": 0.25664335664335663,
      "n_trials": 24,
      "n_effective": 21,
      "final_model": {
        "s1_mean": 0.778356186393404,
        "s1_min": 0.40888212735988483,
        "s2": 0.3545935894910687,
        "separation_long": 0.001710025277286328,
        "median_duration_days": {
          "0": 45.0,
          "1": 46.0,
          "2": 34.0,
          "3": 44.5
        },
        "days_by_decade": {
          "0": {
            "1980": 156,
            "1990": 360,
            "2000": 660,
            "2010": 226,
            "2020": 272
          },
          "1": {
            "1980": 74,
            "1990": 761,
            "2000": 577,
            "2010": 1001,
            "2020": 202
          },
          "2": {
            "1980": 97,
            "1990": 480,
            "2000": 669,
            "2010": 666,
            "2020": 267
          },
          "3": {
            "1980": 173,
            "1990": 902,
            "2000": 595,
            "2010": 608,
            "2020": 448
          }
        }
      }
    },
    "kmeans_frozen": {
      "verdict": "no-go",
      "final_trial_id": null,
      "n_trials": 4,
      "reason": "no eligible configuration: none passes the minimum median duration with positive stability and positive separation"
    }
  },
  "chosen_family": "jump",
  "score_winner": "jm_k4_lam3",
  "ftic_states": 5,
  "gate": {
    "eta_difference_point": 0.01076540259731688,
    "eta_difference_low": -0.004942166928100889,
    "eta_difference_high": 0.0393963313960134,
    "independence_statistic": 32.79320007376975,
    "n_weeks": 1913.0
  },
  "n_trials": 24,
  "n_effective": 21,
  "final_model": {
    "s1_mean": 0.778356186393404,
    "s1_min": 0.40888212735988483,
    "s2": 0.3545935894910687,
    "separation_long": 0.001710025277286328,
    "median_duration_days": {
      "0": 45.0,
      "1": 46.0,
      "2": 34.0,
      "3": 44.5
    },
    "days_by_decade": {
      "0": {
        "1980": 156,
        "1990": 360,
        "2000": 660,
        "2010": 226,
        "2020": 272
      },
      "1": {
        "1980": 74,
        "1990": 761,
        "2000": 577,
        "2010": 1001,
        "2020": 202
      },
      "2": {
        "1980": 97,
        "1990": 480,
        "2000": 669,
        "2010": 666,
        "2020": 267
      },
      "3": {
        "1980": 173,
        "1990": 902,
        "2000": 595,
        "2010": 608,
        "2020": 448
      }
    }
  }
}
```
