# RQ1 (Arm B) — Disaggregated Fairness Audit of the Acne-Severity Classifier

_Tone labels: validated ITA **ita_B** (Method B, NB1b). Audit restricted to Light vs Medium; Deep excluded as artifact-laden (NB1b montage evidence)._

- Predictions joined: **1457 / 1457**
- Tone-band counts (validated):
    - Medium: 724
    - Light: 538
    - Deep: 195

- **Overall** (all groups): accuracy **0.766**, macro-F1 **0.746**

## 1. Per-group performance (bootstrapped 95% CIs)

| group | n | accuracy | macro-F1 | ECE |
|---|---|---|---|---|
| Light | 538 | 0.762 [0.723, 0.797] | 0.739 [0.696, 0.780] | 0.120 |
| Medium | 724 | 0.782 [0.751, 0.812] | 0.735 [0.684, 0.779] | 0.092 |

## 2. Fairness gaps (Light − Medium)

- **Accuracy gap:** -0.020 [95% CI -0.070, +0.026]
- **Macro-F1 gap:** +0.003 [95% CI -0.059, +0.070]
- The accuracy-gap CI does NOT exclude zero → disparity is not statistically resolved at this sample size.

## 3. Equal-opportunity view — per-class recall by group

| severity class | recall Light | recall Medium | gap |
|---|---|---|---|
| 0 | 0.881 | 0.810 | +0.072 |
| 1 | 0.660 | 0.808 | -0.148 |
| 2 | 0.615 | 0.615 | +0.000 |
| 3 | 0.841 | 0.655 | +0.186 |

## 4. Figures

- `reports/figures/armB_fair_accuracy.png`
- `reports/figures/armB_fair_ece.png`

## 5. Reading this audit

- **Scope is Light vs Medium only.** ACNE04 has no auditable dark-skin population (NB1b): the apparent 'Deep' images were lighting/acne artifacts. This is itself a reported finding on the limits of facial acne corpora.
- **Interpret via the CIs, not point estimates.** The Light group is small, so its interval is wide; a point 'gap' that the CI straddles zero is not a demonstrated disparity.
- **Calibration matters as much as accuracy** — a group can match on accuracy yet be worse-calibrated (confidently wrong), which the per-group ECE exposes and which justifies the abstention threshold in the demonstrator.
- **This is the applied arm (Arm B).** The powered, ground-truth-tone audit is Arm A (Fitzpatrick); RQ6 compares the two.

