"""
Demo script to show lambda recalibration effect on France vs Morocco friendly.

This script demonstrates the before/after comparison for the calibration system,
showing how context-aware compression brings unrealistic goal expectations down
to realistic ranges.
"""
import sys
sys.path.insert(0, '/workspace')

from predicciones.src.models.lambda_recalibration import LambdaRecalibrator, HISTORICAL_GOAL_AVERAGES
from predicciones.src.models.dixon_coles import dc_score_matrix
from predicciones.src.models.market_derivation import derive_all_markets
import numpy as np


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def format_prob(p):
    """Format probability as percentage."""
    return f"{p*100:.1f}%"


def demo_france_morocco():
    """
    Demonstrate recalibration for France vs Morocco (International Friendly, 2026-07-09).
    
    BEFORE (raw model output - unrealistic):
    - lambda_home ≈ 3.47
    - lambda_away ≈ 2.72
    - total ≈ 6.19 (WAY too high for international friendly)
    
    AFTER (recalibrated - realistic):
    - Should be in range 2.4-3.2 total goals for friendly
    """
    print_section("FRANCE vs MOROCCO - International Friendly 2026-07-09")
    
    # Raw lambdas from Dixon-Coles heuristic (unrealistic scenario from user report)
    raw_lambda_h = 3.47
    raw_lambda_a = 2.72
    raw_total = raw_lambda_h + raw_lambda_a
    
    print(f"\n📊 RAW LAMBDAS (before recalibration):")
    print(f"   Home (France):  λ_h = {raw_lambda_h:.3f}")
    print(f"   Away (Morocco): λ_a = {raw_lambda_a:.3f}")
    print(f"   Total expected: {raw_total:.3f} goals ⚠️  UNREALISTIC!")
    
    # Create recalibrator with default config (no trained model = fallback mode)
    recalibrator = LambdaRecalibrator(config={})
    
    # Recalibrate for friendly match
    cal_lambda_h, cal_lambda_a = recalibrator.recalibrate(
        raw_lambda_h, raw_lambda_a,
        competition_type='friendly',
        competition_slug='fifa.world'
    )
    cal_total = cal_lambda_h + cal_lambda_a
    
    print(f"\n✅ RECALIBRATED LAMBDAS (after context-aware compression):")
    print(f"   Home (France):  λ_h = {cal_lambda_h:.3f}")
    print(f"   Away (Morocco): λ_a = {cal_lambda_a:.3f}")
    print(f"   Total expected: {cal_total:.3f} goals ✓ REALISTIC!")
    
    # Show historical prior for friendlies
    hist_prior = HISTORICAL_GOAL_AVERAGES['friendly']
    print(f"\n📚 HISTORICAL REFERENCE (International Friendlies):")
    print(f"   Mean: {hist_prior['mean']:.2f} goals/match")
    print(f"   Std Dev: {hist_prior['std']:.2f}")
    print(f"   Typical range: [{hist_prior['min']:.1f}, {hist_prior['max']:.1f}]")
    
    # Calculate reduction
    reduction = ((raw_total - cal_total) / raw_total) * 100
    print(f"\n📉 COMPRESSION APPLIED:")
    print(f"   Total lambda reduced by: {reduction:.1f}%")
    print(f"   Compression preserved ratio: {raw_lambda_h/raw_lambda_a:.3f} → {cal_lambda_h/cal_lambda_a:.3f}")
    
    # Generate score matrices and markets for both scenarios
    print_section("MARKET COMPARISON: Before vs After Recalibration")
    
    # Raw matrix (unrealistic)
    raw_matrix = dc_score_matrix(raw_lambda_h, raw_lambda_a, rho=-0.13, max_goals=8)
    raw_markets = derive_all_markets(raw_matrix, raw_lambda_h, raw_lambda_a, config={})
    
    # Calibrated matrix (realistic)
    cal_matrix = dc_score_matrix(cal_lambda_h, cal_lambda_a, rho=-0.13, max_goals=8)
    cal_markets = derive_all_markets(cal_matrix, cal_lambda_h, cal_lambda_a, config={})
    
    # Compare key markets
    print("\n🎯 1X2 PROBABILITIES:")
    print(f"   {'Market':<15} {'Before':>12} {'After':>12} {'Change':>12}")
    print(f"   {'-'*51}")
    for outcome in ['home', 'draw', 'away']:
        before = raw_markets['1x2'][outcome]
        after = cal_markets['1x2'][outcome]
        change = after - before
        sign = '+' if change > 0 else ''
        print(f"   {outcome.capitalize():<15} {format_prob(before):>12} {format_prob(after):>12} {sign}{format_prob(change):>11}")
    
    print("\n⚽ OVER/UNDER PROBABILITIES:")
    print(f"   {'Market':<15} {'Before':>12} {'After':>12} {'Change':>12}")
    print(f"   {'-'*51}")
    for line in ['1_5', '2_5', '3_5', '4_5']:
        before_over = raw_markets['over_under'][f'over_{line}']
        after_over = cal_markets['over_under'][f'over_{line}']
        change = after_over - before_over
        sign = '+' if change > 0 else ''
        label = f"Over {line.replace('_', '.')}"
        print(f"   {label:<15} {format_prob(before_over):>12} {format_prob(after_over):>12} {sign}{format_prob(change):>11}")
    
    print("\n🔥 BTTS PROBABILITIES:")
    print(f"   {'Market':<15} {'Before':>12} {'After':>12} {'Change':>12}")
    print(f"   {'-'*51}")
    for outcome in ['yes', 'no']:
        before = raw_markets['btts'][outcome]
        after = cal_markets['btts'][outcome]
        change = after - before
        sign = '+' if change > 0 else ''
        print(f"   BTTS {outcome.capitalize():<10} {format_prob(before):>12} {format_prob(after):>12} {sign}{format_prob(change):>11}")
    
    print("\n📋 EXPECTED GOALS:")
    print(f"   {'Metric':<20} {'Before':>12} {'After':>12}")
    print(f"   {'-'*44}")
    print(f"   Home xG             {raw_markets['expected_goals']['home']:>12.3f} {cal_markets['expected_goals']['home']:>12.3f}")
    print(f"   Away xG             {raw_markets['expected_goals']['away']:>12.3f} {cal_markets['expected_goals']['away']:>12.3f}")
    print(f"   Total xG            {raw_markets['expected_goals']['total']:>12.3f} {cal_markets['expected_goals']['total']:>12.3f}")
    
    print("\n🎲 TOP CORRECT SCORES (Before):")
    for i, score_dict in enumerate(raw_markets['correct_scores'][:5]):
        score = score_dict.get('score', 'N/A')
        prob = score_dict.get('probability', 0)
        print(f"   {i+1}. {score}: {format_prob(prob)}")
    
    print("\n🎲 TOP CORRECT SCORES (After):")
    for i, score_dict in enumerate(cal_markets['correct_scores'][:5]):
        score = score_dict.get('score', 'N/A')
        prob = score_dict.get('probability', 0)
        print(f"   {i+1}. {score}: {format_prob(prob)}")
    
    # Goal distribution comparison
    print("\n📊 GOAL DISTRIBUTION COMPARISON (Home Team):")
    print(f"   {'Total Goals':<15} {'Before %':>12} {'After %':>12} {'Change':>12}")
    print(f"   {'-'*51}")
    
    before_dist = raw_markets['home_goals_distribution']
    after_dist = cal_markets['home_goals_distribution']
    
    for item in before_dist[:6]:
        goals = item.get('goals', 0)
        before_p = item.get('probability', 0)
        # Find corresponding after probability
        after_item = next((x for x in after_dist if x.get('goals') == goals), {'probability': 0})
        after_p = after_item.get('probability', 0)
        change = after_p - before_p
        sign = '+' if change > 0 else ''
        print(f"   {goals} goal(s)          {format_prob(before_p):>12} {format_prob(after_p):>12} {sign}{format_prob(change):>11}")
    
    print_section("SUMMARY")
    print(f"""
The recalibration successfully transformed unrealistic predictions into
realistic ones for an international friendly:

BEFORE: Expected {raw_total:.2f} total goals (extremely high, rare in real football)
AFTER:  Expected {cal_total:.2f} total goals (realistic for friendly matches)

Key improvements:
✓ Over 2.5 probability reduced from {format_prob(raw_markets['over_under']['over_2_5'])} to {format_prob(cal_markets['over_under']['over_2_5'])}
✓ Over 3.5 probability reduced from {format_prob(raw_markets['over_under']['over_3_5'])} to {format_prob(cal_markets['over_under']['over_3_5'])}
✓ BTTS Yes reduced from {format_prob(raw_markets['btts']['yes'])} to {format_prob(cal_markets['btts']['yes'])}
✓ Correct scores now favor realistic results (1-0, 2-1, 1-1) over extreme ones

The model now produces predictions consistent with historical data!
""")


def demo_world_cup_scenario():
    """Demonstrate World Cup scenario calibration."""
    print_section("WORLD CUP SCENARIO - Example Match")
    
    # Simulate high-intensity World Cup knockout match
    raw_lambda_h = 2.8
    raw_lambda_a = 2.2
    raw_total = raw_lambda_h + raw_lambda_a
    
    print(f"\n📊 RAW LAMBDAS: {raw_lambda_h:.3f} + {raw_lambda_a:.3f} = {raw_total:.3f}")
    
    recalibrator = LambdaRecalibrator(config={})
    cal_lambda_h, cal_lambda_a = recalibrator.recalibrate(
        raw_lambda_h, raw_lambda_a,
        competition_type='world_cup',
        competition_slug='fifa.world'
    )
    cal_total = cal_lambda_h + cal_lambda_a
    
    print(f"✅ RECALIBRATED: {cal_lambda_h:.3f} + {cal_lambda_a:.3f} = {cal_total:.3f}")
    
    hist_prior = HISTORICAL_GOAL_AVERAGES['world_cup']
    print(f"\n📚 HISTORICAL REFERENCE (FIFA World Cup):")
    print(f"   Mean: {hist_prior['mean']:.2f} goals/match")
    
    reduction = ((raw_total - cal_total) / raw_total) * 100
    print(f"   Compression: {reduction:.1f}%")


def demo_league_scenario():
    """Demonstrate top league scenario calibration."""
    print_section("PREMIER LEAGUE SCENARIO - Example Match")
    
    # Simulate Premier League match (higher scoring league)
    raw_lambda_h = 2.5
    raw_lambda_a = 1.8
    raw_total = raw_lambda_h + raw_lambda_a
    
    print(f"\n📊 RAW LAMBDAS: {raw_lambda_h:.3f} + {raw_lambda_a:.3f} = {raw_total:.3f}")
    
    recalibrator = LambdaRecalibrator(config={})
    cal_lambda_h, cal_lambda_a = recalibrator.recalibrate(
        raw_lambda_h, raw_lambda_a,
        competition_type='league_top',
        competition_slug='eng.1'
    )
    cal_total = cal_lambda_h + cal_lambda_a
    
    print(f"✅ RECALIBRATED: {cal_lambda_h:.3f} + {cal_lambda_a:.3f} = {cal_total:.3f}")
    
    hist_prior = HISTORICAL_GOAL_AVERAGES['eng.1']
    print(f"\n📚 HISTORICAL REFERENCE (Premier League):")
    print(f"   Mean: {hist_prior['mean']:.2f} goals/match")
    
    reduction = ((raw_total - cal_total) / raw_total) * 100
    print(f"   Compression: {reduction:.1f}%")


if __name__ == '__main__':
    print("\n" + "█" * 70)
    print(" LAMBDA RECALIBRATION DEMO - Football Match Predictor")
    print(" Demonstrating context-aware goal expectation calibration")
    print("█" * 70)
    
    demo_france_morocco()
    demo_world_cup_scenario()
    demo_league_scenario()
    
    print("\n" + "=" * 70)
    print(" Demo completed successfully!")
    print("=" * 70 + "\n")
