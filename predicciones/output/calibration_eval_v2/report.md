# Calibration & Backtesting Report - Version 2 (Corrected Pipeline)

**Generated:** 2026-07-09 03:08:17

**Pipeline changes in this version:**
- Ratings normalized around 1.0 (attack/defense)
- Ranking factor double-counting eliminated
- Lambda sanity checks added

## 1. Overview

- **Matches evaluated:** 200
- **Configurations compared:** baseline, markov_aware
- **Markov weight:** 0.18

## 2. Metrics Summary

### 2.1 Brier Scores (lower is better)

| Config | 1X2 Avg | Home | Draw | Away | O/U 2.5 | BTTS |
|--------|---------|------|------|------|---------|------|
| baseline | 0.219962 | 0.253738 | 0.189196 | 0.216952 | 0.258962 | 0.264764 |
| markov_aware | 0.216093 | 0.247414 | 0.188691 | 0.212175 | 0.258790 | 0.264283 |

### 2.2 Log Loss (lower is better)

| Config | 1X2 | O/U 2.5 | BTTS |
|--------|-----|---------|------|
| baseline | 1.088888 | 0.742281 | 0.749444 |
| markov_aware | 1.073533 | 0.744271 | 0.749447 |

### 2.3 Calibration (ECE)

| Config | ECE O/U 2.5 | ECE BTTS | MAE Goals |
|--------|-------------|----------|-----------|
| baseline | 0.154704 | 0.171843 | 1.821372 |
| markov_aware | 0.142648 | 0.166773 | 1.844335 |

### 2.4 Delta (Markov - Baseline)

*Negative values indicate improvement with Markov.*

| Metric | Delta |
|--------|-------|
| brier_home_win | -0.006324 (↓ better) |
| brier_draw | -0.000505 (↓ better) |
| brier_away_win | -0.004777 (↓ better) |
| brier_1x2_avg | -0.003869 (↓ better) |
| logloss_1x2 | -0.015355 (↓ better) |
| brier_ou25 | -0.000172 (↓ better) |
| logloss_ou25 | +0.001990 (↑ worse) |
| brier_btts | -0.000481 (↓ better) |
| logloss_btts | +0.000003 (↑ worse) |
| mae_goals | +0.022963 (↑ worse) |
| ece_ou25 | -0.012056 (↓ better) |
| ece_btts | -0.005070 (↓ better) |

## 3. Analysis by Match Phase

### Early Game (0-30 / 31-75 / 76-90+ minutes)

- Matches: 51
- baseline: Avg predicted goals = 4.245, Actual = 3.373
- markov_aware: Avg predicted goals = 4.282, Actual = 3.373

### Mid Game (0-30 / 31-75 / 76-90+ minutes)

- Matches: 106
- baseline: Avg predicted goals = 4.225, Actual = 3.396
- markov_aware: Avg predicted goals = 4.387, Actual = 3.396

### Late Game (0-30 / 31-75 / 76-90+ minutes)

- Matches: 43
- baseline: Avg predicted goals = 3.990, Actual = 3.302
- markov_aware: Avg predicted goals = 3.564, Actual = 3.302

## 4. Analysis by Score Difference

### Score Diff -2_or_more

- Matches: 9
- baseline: Avg predicted = 4.858, Actual = 4.222
- markov_aware: Avg predicted = 4.727, Actual = 4.222

### Score Diff -1

- Matches: 32
- baseline: Avg predicted = 3.985, Actual = 3.531
- markov_aware: Avg predicted = 4.083, Actual = 3.531

### Score Diff 0

- Matches: 85
- baseline: Avg predicted = 4.259, Actual = 3.200
- markov_aware: Avg predicted = 4.225, Actual = 3.200

### Score Diff +1

- Matches: 48
- baseline: Avg predicted = 4.157, Actual = 3.146
- markov_aware: Avg predicted = 4.261, Actual = 3.146

### Score Diff +2_or_more

- Matches: 26
- baseline: Avg predicted = 3.967, Actual = 3.846
- markov_aware: Avg predicted = 3.840, Actual = 3.846

## 5. Lambda Distribution (Corrected Pipeline)

### baseline

- lambda_home: mean=2.0936, median=2.0115, std=0.4857
- lambda_away: mean=2.0860, median=2.0115, std=0.4735
- lambda_total: mean=4.1796, median=4.0129, std=0.9435

### markov_aware

- lambda_home: mean=2.1076, median=1.9922, std=0.5171
- lambda_away: mean=2.0758, median=1.9710, std=0.5102
- lambda_total: mean=4.1834, median=3.9842, std=1.0081

## 6. Recommendations

Based on the evaluation results:

1. **Markov integration provides measurable improvement** in calibration metrics.
   - Brier score improvement: 1.76%
2. **The markov_weight=0.18 setting is appropriately conservative**, producing small adjustments.

### Final Recommendation

**Enable Markov features by default** with `markov_weight=0.18`:

```yaml
dixon_coles:
  use_markov_features: true
  markov_weight: 0.18
```

## 7. Operational Readiness Assessment

### Lambda Distribution Health

- Average lambda_total: 4.18 (expected: 2.2-3.2 for international football)
- % matches with lambda_total > 5.0: 18.5%
- % lambda_home in valid range [0.05, 4.0]: 100.0%

### ⚠ ADDITIONAL CALIBRATION RECOMMENDED

Areas to review before deployment:
- Lambda values may be too aggressive
- Too many extreme predictions

---
*End of Report*