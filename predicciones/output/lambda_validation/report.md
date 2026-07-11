# Lambda Distribution Validation Report

**Generated:** 2026-07-09 03:04:17

## 1. Overview

- **Matches analyzed:** 50
- **Thresholds:** lambda_home > 3.0, lambda_total > 5.0

## 2. Lambda Distribution Summary

### 2.1 lambda_home

- Mean: 1.5913
- Median: 1.3408
- Std Dev: 1.1274
- Range: [0.2092, 4.0000]

| P10 | P25 | P50 | P75 | P90 | P95 |
|-----|-----|-----|-----|-----|-----|
| 0.4791 | 0.6429 | 1.3408 | 2.3440 | 3.3747 | 3.9956 |

### 2.2 lambda_away

- Mean: 1.2975
- Median: 1.0031
- Std Dev: 0.9454
- Range: [0.1359, 3.7274]

| P10 | P25 | P50 | P75 | P90 | P95 |
|-----|-----|-----|-----|-----|-----|
| 0.3962 | 0.4970 | 1.0031 | 1.9570 | 2.7685 | 3.0599 |

### 2.3 lambda_total

- Mean: 2.8888
- Median: 2.7756
- Std Dev: 1.2582
- Range: [0.9676, 6.1516]

| P10 | P25 | P50 | P75 | P90 | P95 |
|-----|-----|-----|-----|-----|-----|
| 1.5413 | 1.8451 | 2.7756 | 3.7131 | 4.4362 | 5.3097 |

## 3. Threshold Exceedances

Matches exceeding configured warning thresholds:

- **lambda_home > 3.0:** 8 (16.0%)
- **lambda_away > 3.0:** 3 (6.0%)
- **lambda_total > 5.0:** 4 (8.0%)

## 4. Sanity Checks

- **lambda_home in [0.05, 4.0]:** 100.0%
- **lambda_away in [0.05, 4.0]:** 100.0%
- **lambda_total <= 8.0:** 100.0%

## 5. High Lambda Matches

Found 11 matches exceeding thresholds:

| Match | Home | Away | λ_home | λ_away | λ_total |
|-------|------|------|--------|--------|---------|
| 6 | Francia | Portugal | 3.187 | 2.965 | 6.152 |
| 18 | Francia | England | 3.128 | 2.979 | 6.107 |
| 19 | Argentina | Estados Unidos | 3.990 | 1.500 | 5.491 |
| 41 | Argentina | Senegal | 4.000 | 1.088 | 5.088 |
| 1 | Japón | England | 1.262 | 3.334 | 4.596 |
| 4 | España | Canadá | 4.000 | 0.418 | 4.418 |
| 10 | Brasil | Ukraine | 3.308 | 0.918 | 4.226 |
| 38 | Brasil | Nueva Zelanda | 4.000 | 0.182 | 4.182 |
| 5 | Francia | Bolivia | 3.976 | 0.136 | 4.112 |
| 13 | Republica Democratica del Congo | Inglaterra | 0.332 | 3.727 | 4.059 |

## 6. Assessment

✓ lambda_total mean is within expected range for international football
✓ lambda_home mean is within expected range
✓ lambda_away mean is within expected range
✓ Low percentage (8.0%) of extreme lambda_total values
✓ Nearly all lambda values within clipping bounds

## 7. Conclusion

**The lambda distribution appears reasonable for operational use.**

Key observations:
- Average total expected goals: 2.89 (typical for international football)
- Only 8.0% of matches exceed the warning threshold
- Ratings normalization appears effective

---
*End of Report*