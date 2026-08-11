# Abstention Threshold (derived, not chosen)

_The demonstrator declines to advise when the classifier is not confident enough. The threshold below is derived from the same out-of-fold predictions the fairness audit used, and is checked per tone group so that abstention does not silently fall harder on one group._

- Target selective accuracy: **85%**
- Minimum acceptable coverage: **50%**
- Base accuracy at full coverage: **0.766**

## 1. Chosen threshold

**tau = 0.85** -- lowest threshold meeting 85% selective accuracy while retaining >=50% coverage.

| metric | value |
|---|---|
| coverage (users advised) | 0.677 |
| selective accuracy | 0.851 |
| coverage (Light) | 0.697 |
| selective accuracy (Light) | 0.851 |
| coverage (Medium) | 0.689 |
| selective accuracy (Medium) | 0.864 |

## 2. Does abstention fall unevenly across tone groups?

- Coverage gap (|Light - Medium|): **0.008**
- Selective-accuracy gap: **0.013**

This check exists because RQ1 found the Light band less well calibrated (ECE 0.120 vs 0.092). If abstention were much more frequent for one group, the system would be quietly denying service to that group -- an allocative harm of its own. The gap here is small, so abstention is broadly even-handed.

## 3. Behaviour below the threshold

The interface states that it cannot assess the image with confidence and offers to proceed on declared attributes alone, rather than substituting a plausible-sounding guess. This is surfaced explicitly to the user, never hidden.

Figure: `reports/figures/abstention_curve.png`

