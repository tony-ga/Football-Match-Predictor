# Match State Pattern Analysis Report

## Detected Patterns

### just_conceded_corners_response

**Condition**: Team just conceded -> corners in next 15'

- Sample Size: 274
- Mean: 0.825
- Median: 0.000
- Std Dev: 1.167
- Baseline Mean: 0.830
- Lift vs Baseline: -0.6%
- 95% CI: [0.687, 0.963]

**Quality Assessment**:
- ✅ Good sample size
- ℹ️ Minimal lift vs baseline

---

### just_scored_opponent_response

**Condition**: Team just scored -> opponent shots in next 15'

- Sample Size: 249
- Mean: 1.000
- Median: 1.000
- Std Dev: 1.225
- Baseline Mean: 1.274
- Lift vs Baseline: -21.5%
- 95% CI: [0.848, 1.152]

**Quality Assessment**:
- ✅ Good sample size
- 📈 Moderate lift vs baseline

---

### leading_by_1_after_60_shots_against

**Condition**: Team winning by 1+ goal after 60' -> opponent shots in next 15'

- Sample Size: 236
- Mean: 0.919
- Median: 0.000
- Std Dev: 1.389
- Baseline Mean: 1.274
- Lift vs Baseline: -27.8%
- 95% CI: [0.742, 1.097]

**Quality Assessment**:
- ✅ Good sample size
- 📈 Moderate lift vs baseline

---

### tied_after_60_total_shots

**Condition**: Tied game after 60' -> total shots in next 15'

- Sample Size: 208
- Mean: 2.332
- Median: 2.000
- Std Dev: 2.101
- Baseline Mean: 2.517
- Lift vs Baseline: -7.4%
- 95% CI: [2.046, 2.617]

**Quality Assessment**:
- ✅ Good sample size
- ℹ️ Minimal lift vs baseline

---

### trailing_by_1_after_60_corners

**Condition**: Team losing by 1+ goal after 60' -> corners in next 15'

- Sample Size: 207
- Mean: 0.676
- Median: 0.000
- Std Dev: 1.064
- Baseline Mean: 0.830
- Lift vs Baseline: -18.5%
- 95% CI: [0.531, 0.821]

**Quality Assessment**:
- ✅ Good sample size
- 📈 Moderate lift vs baseline

---

### trailing_by_1_after_60_shots

**Condition**: Team losing by 1+ goal after 60' -> shots in next 15'

- Sample Size: 207
- Mean: 0.986
- Median: 0.000
- Std Dev: 1.363
- Baseline Mean: 1.243
- Lift vs Baseline: -20.7%
- 95% CI: [0.800, 1.171]

**Quality Assessment**:
- ✅ Good sample size
- 📈 Moderate lift vs baseline

---

### trailing_fouls_desperation

**Condition**: Team losing by 1+ after 60' -> fouls in next 15'

- Sample Size: 207
- Mean: 2.295
- Median: 2.000
- Std Dev: 2.492
- Baseline Mean: 2.964
- Lift vs Baseline: -22.6%
- 95% CI: [1.955, 2.634]

**Quality Assessment**:
- ✅ Good sample size
- 📈 Moderate lift vs baseline

---

### red_card_down_concession

**Condition**: Team with red card -> goals conceded in next 15'

- Sample Size: 74
- Mean: 0.257
- Median: 0.000
- Std Dev: 0.498
- Baseline Mean: 0.194
- Lift vs Baseline: +32.0%
- 95% CI: [0.143, 0.370]

**Quality Assessment**:
- ✅ Good sample size
- 🚀 Strong lift vs baseline

---

## Summary

**Robust patterns detected**: 5

Most promising patterns for modeling:
- red_card_down_concession (lift: +32.0%, n=74)
- leading_by_1_after_60_shots_against (lift: -27.8%, n=236)
- trailing_fouls_desperation (lift: -22.6%, n=207)
- just_scored_opponent_response (lift: -21.5%, n=249)
- trailing_by_1_after_60_shots (lift: -20.7%, n=207)
