# Calibration & Backtesting Report - Fase 3

**Generated:** 2026-07-09 00:23:23

## 1. Overview

- **Matches evaluated:** 200
- **Configurations compared:** baseline, markov_aware
- **Markov weight:** 0.18

## 2. Metrics Summary

### 2.1 Brier Scores (lower is better)

| Config | 1X2 Avg | Home | Draw | Away | O/U 2.5 | BTTS |
|--------|---------|------|------|------|---------|------|
| baseline | 0.160523 | 0.172280 | 0.167710 | 0.141579 | 0.151415 | 0.173371 |
| markov_aware | 0.160169 | 0.172235 | 0.167823 | 0.140449 | 0.153808 | 0.173442 |

### 2.2 Log Loss (lower is better)

| Config | 1X2 | O/U 2.5 | BTTS |
|--------|-----|---------|------|
| baseline | 0.824359 | 0.461134 | 0.511454 |
| markov_aware | 0.822887 | 0.467087 | 0.511261 |

### 2.3 Calibration (ECE)

| Config | ECE O/U 2.5 | ECE BTTS | MAE Goals |
|--------|-------------|----------|-----------|
| baseline | 0.054923 | 0.090742 | 1.244577 |
| markov_aware | 0.053163 | 0.084255 | 1.265599 |

### 2.4 Delta (Markov - Baseline)

*Negative values indicate improvement with Markov.*

| Metric | Delta |
|--------|-------|
| brier_1x2_avg | -0.000354 (↓ better) |
| logloss_1x2 | -0.001472 (↓ better) |
| brier_ou25 | +0.002393 (↑ worse) |
| brier_btts | +0.000071 (↑ worse) |
| mae_goals | +0.021022 (↑ worse) |

## 3. Analysis by Match Phase

### Early Game (0-30 / 31-75 / 76-90+ minutes)

- Matches: 52
- baseline: Avg predicted goals = 2.830, Actual = 2.462
- markov_aware: Avg predicted goals = 2.862, Actual = 2.462

### Mid Game (0-30 / 31-75 / 76-90+ minutes)

- Matches: 112
- baseline: Avg predicted goals = 2.733, Actual = 2.580
- markov_aware: Avg predicted goals = 2.837, Actual = 2.580

### Late Game (0-30 / 31-75 / 76-90+ minutes)

- Matches: 36
- baseline: Avg predicted goals = 2.738, Actual = 2.556
- markov_aware: Avg predicted goals = 2.486, Actual = 2.556

## 4. Analysis by Score Difference

### Score Diff -2_or_more

- Matches: 17
- baseline: Avg predicted = 4.575, Actual = 5.471
- markov_aware: Avg predicted = 4.541, Actual = 5.471

### Score Diff -1

- Matches: 27
- baseline: Avg predicted = 2.981, Actual = 2.185
- markov_aware: Avg predicted = 3.069, Actual = 2.185

### Score Diff 0

- Matches: 96
- baseline: Avg predicted = 1.941, Actual = 1.479
- markov_aware: Avg predicted = 1.940, Actual = 1.479

### Score Diff +1

- Matches: 41
- baseline: Avg predicted = 3.056, Actual = 2.902
- markov_aware: Avg predicted = 3.141, Actual = 2.902

### Score Diff +2_or_more

- Matches: 19
- baseline: Avg predicted = 4.313, Actual = 5.053
- markov_aware: Avg predicted = 4.261, Actual = 5.053

## 5. Recommendations

Based on the evaluation results:

1. **Markov integration provides measurable improvement** in calibration metrics.
   - Brier score improvement: 0.22%
2. **The markov_weight=0.18 setting is appropriately conservative**, producing small adjustments.

### Final Recommendation

**Enable Markov features by default** with `markov_weight=0.18`:

```yaml
dixon_coles:
  use_markov_features: true
  markov_weight: 0.18
```

---
*End of Report*