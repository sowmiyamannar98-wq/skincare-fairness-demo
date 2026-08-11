# Architecture Comparison (proposal §7.1)

_Results from `kaggle/nb6_architecture_comparison.ipynb`: 4 architectures ×
3 seeds = 12 training runs, identical split protocol, augmentation, optimiser,
schedule, and epoch budget throughout._

## 1. Results

| architecture | params (M) | accuracy | macro-F1 | severe F1 | ECE | \|tone gap\| | min/run |
|---|---|---|---|---|---|---|---|
| **resnet50** | 23.5 | **0.753 ± 0.037** | **0.731 ± 0.026** | **0.582** | 0.108 | 0.052 | 17 |
| efficientnet_b0 | 4.0 | 0.715 ± 0.040 | 0.690 ± 0.039 | 0.549 | 0.190 | **0.022** | 17 |
| efficientnet_b3 | 10.7 | 0.709 ± 0.017 | 0.679 ± 0.030 | 0.509 | 0.183 | 0.073 | 21 |
| vit_base | 85.8 | 0.636 ± 0.047 | 0.639 ± 0.038 | 0.492 | **0.085** | 0.068 | 21 |

## 2. This comparison resolves — unlike the mitigation experiment

Pooled seed standard deviation is **0.033**; the spread across architectures is
**0.092**, nearly three times larger. Every non-baseline architecture is worse
than ResNet-50 by more than one seed standard deviation:

- efficientnet_b0 — worse by 0.041
- efficientnet_b3 — worse by 0.052
- vit_base — worse by 0.092

This is the direct contrast with RQ3, where between-strategy differences
(≤ 0.008) sat well inside the same noise floor. The identical test applied to
both questions returns *unresolved* for mitigation and *resolved* for
architecture, which is evidence that the test is discriminating rather than
merely conservative.

**A reproducibility check falls out of this for free.** ResNet-50 under the
shared protocol scored macro-F1 **0.731 ± 0.026** here, against **0.735 ± 0.031**
for the `erm` condition in NB5 — a separate notebook, separate runs, same
protocol. The agreement to within 0.004 indicates the pipeline is stable and
the seed variance is being characterised correctly rather than accumulating
from implementation drift.

## 3. Capacity does not help on this corpus

The largest model is the worst by a wide margin: ViT-Base carries **85.8M
parameters**, 3.6× ResNet-50, and loses 0.092 macro-F1. EfficientNet-B3 (10.7M)
also loses to EfficientNet-B0 (4.0M), so this is not a simple monotonic
"smaller is better" — but the top of the capacity range is clearly penalised.

The standard explanation applies and is worth stating: **Vision Transformers
lack the convolutional inductive biases** — locality and translation
equivariance — that CNNs encode architecturally, and compensate for their
absence with data. With **1,093 training images**, that compensation is
unavailable. ACNE04 sits far below the scale at which ViTs become competitive,
and the result is consistent with the wider literature on transformer
fine-tuning in low-data regimes.

The practical reading: **on a corpus this size, architecture capacity is not
the binding constraint — data is.** Adding parameters does not buy performance
that the dataset cannot support.

## 4. An honest confound

All four models were trained at **LR = 3e-4**, chosen for the ResNet-50
baseline in NB3 and held constant so that differences would be attributable to
architecture rather than to tuning. That control cuts both ways.

**ViT-Base is typically fine-tuned at a lower learning rate** (commonly 1e-4 to
5e-5 with AdamW); 3e-4 is likely too aggressive for it. Part of ViT's deficit
is therefore plausibly attributable to the shared protocol rather than to the
architecture itself, and the honest claim is narrower than "ViT is worse":

> Under a protocol tuned to the CNN baseline, ViT-Base underperforms
> substantially on ACNE04. A per-architecture learning-rate sweep would be
> required to separate architectural limitation from hyperparameter mismatch.

This does not disturb the deployment decision — the model that is actually
shipped was selected under the budget actually available — but it does bound
the generality of the comparison, and the same caveat applies in weaker form to
both EfficientNets.

## 5. Calibration runs opposite to accuracy

An unexpected pattern: **ViT-Base is the best-calibrated model (ECE 0.085)
despite being the least accurate**, while the EfficientNets are the worst
calibrated (0.190 and 0.183) despite outperforming it. ResNet-50 sits between
(0.108).

Accuracy and calibration are therefore not ordered together here. A model can
be more often wrong yet more honest about its uncertainty. This matters
directly for the demonstrator, where the abstention threshold (§11) depends on
confidence being meaningful: had EfficientNet been selected on accuracy alone,
its poorer calibration would have degraded abstention behaviour without
appearing in any accuracy metric.

## 6. Conclusion

**ResNet-50 is retained as the deployed model**, and that choice is now
evidenced rather than assumed. It is best on accuracy, macro-F1, and
severe-class F1; second-best on calibration; and among the cheapest to train
(17 min/run). The proposal's commitment to comparing against EfficientNet and
ViT is discharged, and the outcome supports the original baseline.

## 7. Artefacts

- `arch_comparison_raw.csv` — all 12 runs
- `arch_comparison_summary.csv` — aggregated with standard deviations
- `arch_comparison.png` — macro-F1 by architecture, and capacity vs performance
- `arch_result.json` — machine-readable summary
