#!/usr/bin/env python3
"""
Validate Markov Model with Walk-Forward Testing

Fase 2: Valida el modelo Markov usando validación temporal walk-forward.

Uso:
python scripts/validate_markov_model.py \
  --input output/analysis/markov_ready_transitions.csv \
  --output-dir output/markov_validation \
  --method walk-forward \
  --min-state-sample 20
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


def parse_state(state_str):
    """Parse state JSON string to dict."""
    if isinstance(state_str, dict):
        return state_str
    if pd.isna(state_str) or state_str == '':
        return None
    try:
        cleaned = state_str.replace('""', '"')
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None


def state_to_key(state_dict):
    """Convert state dict to hashable key."""
    if state_dict is None:
        return "unknown"
    parts = []
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        parts.append(f"{k}={v}")
    return "|".join(parts)


def load_match_dates(transitions_df, coverage_df=None):
    """
    Load match dates for temporal ordering.
    Try to get dates from coverage report or infer from match_id.
    """
    match_dates = {}
    
    if coverage_df is not None and 'date' in coverage_df.columns:
        for _, row in coverage_df.iterrows():
            match_id = str(row['match_id'])
            date = row.get('date')
            if pd.notna(date) and date != '':
                match_dates[match_id] = date
    
    # If no dates found, use match_id as proxy (assuming higher IDs are later)
    if not match_dates:
        match_ids = transitions_df['match_id'].unique()
        for i, mid in enumerate(sorted(match_ids)):
            match_dates[str(mid)] = f"2025-01-{i+1:02d}"  # Proxy date
    
    return match_dates


def build_markov_model(train_df, min_sample=20):
    """
    Build Markov model from training data.
    Returns dictionaries for transition and event probabilities.
    """
    # State transition probabilities
    state_transitions = defaultdict(lambda: defaultdict(int))
    state_totals = defaultdict(int)
    
    # State event counts
    state_events = defaultdict(lambda: {
        'goals': [], 'conceded': [], 'corners': [], 'shots': []
    })
    
    for _, row in train_df.iterrows():
        state_t = state_to_key(parse_state(row.get('state_t')))
        next_state = state_to_key(parse_state(row.get('next_state_t1')))
        
        if state_t and next_state:
            state_transitions[state_t][next_state] += 1
            state_totals[state_t] += 1
            
            state_events[state_t]['goals'].append(int(row.get('goals_next_window', 0) or 0))
            state_events[state_t]['conceded'].append(int(row.get('concede_next_window', 0) or 0))
            state_events[state_t]['corners'].append(int(row.get('corners_next_window', 0) or 0))
            state_events[state_t]['shots'].append(int(row.get('shots_next_window', 0) or 0))
    
    # Build probability dictionaries with Laplace smoothing
    trans_probs = {}
    event_probs = {}
    
    all_next_states = set()
    for state_t in state_transitions:
        all_next_states.update(state_transitions[state_t].keys())
    n_next_states = len(all_next_states) if all_next_states else 1
    
    for state_t in state_transitions:
        # Transition probabilities
        transitions = state_transitions[state_t]
        total = state_totals[state_t]
        
        if total >= min_sample:
            smoothed = {}
            for next_state in transitions:
                smoothed[next_state] = (transitions[next_state] + 1) / (total + n_next_states)
            trans_probs[state_t] = smoothed
        
        # Event probabilities
        events = state_events[state_t]
        n = len(events['goals'])
        
        if n >= min_sample:
            event_probs[state_t] = {
                'p_goal': (sum(1 for g in events['goals'] if g > 0) + 1) / (n + 2),
                'p_concede': (sum(1 for c in events['conceded'] if c > 0) + 1) / (n + 2),
                'p_corner': (sum(1 for c in events['corners'] if c > 0) + 1) / (n + 2),
                'p_shot': (sum(1 for s in events['shots'] if s > 0) + 1) / (n + 2)
            }
    
    return trans_probs, event_probs


def compute_baseline_probs(df):
    """Compute global baseline probabilities."""
    n = len(df)
    goals = df['goals_next_window'].fillna(0).astype(int)
    conceded = df['concede_next_window'].fillna(0).astype(int)
    corners = df['corners_next_window'].fillna(0).astype(int)
    shots = df['shots_next_window'].fillna(0).astype(int)
    
    return {
        'p_goal': (sum(goals > 0) + 1) / (n + 2),
        'p_concede': (sum(conceded > 0) + 1) / (n + 2),
        'p_corner': (sum(corners > 0) + 1) / (n + 2),
        'p_shot': (sum(shots > 0) + 1) / (n + 2)
    }


def evaluate_predictions(y_true, y_pred_proba, model_name):
    """Evaluate probabilistic predictions."""
    results = {}
    
    # Brier score
    brier = brier_score_loss(y_true, y_pred_proba)
    results[f'{model_name}_brier'] = brier
    
    # Log loss
    try:
        ll = log_loss(y_true, y_pred_proba, labels=[0, 1])
        results[f'{model_name}_logloss'] = ll
    except:
        results[f'{model_name}_logloss'] = float('inf')
    
    return results


def walk_forward_validation(df, match_dates, min_sample=20, n_folds=5):
    """
    Perform walk-forward validation.
    
    Split data chronologically into training and test sets.
    Train on earlier matches, validate on later matches.
    """
    # Add date column
    df = df.copy()
    df['match_date'] = df['match_id'].astype(str).map(match_dates)
    
    # Sort by date
    df = df.sort_values('match_date').reset_index(drop=True)
    
    # Get unique matches in chronological order
    unique_matches = df['match_id'].unique()
    n_matches = len(unique_matches)
    
    fold_size = max(1, n_matches // n_folds)
    
    fold_metrics = []
    all_metrics = {
        'markov_brier_goal': [], 'markov_logloss_goal': [],
        'markov_brier_concede': [], 'markov_logloss_concede': [],
        'markov_brier_corner': [], 'markov_logloss_corner': [],
        'markov_brier_shot': [], 'markov_logloss_shot': [],
        'baseline_brier_goal': [], 'baseline_logloss_goal': [],
        'baseline_brier_concede': [], 'baseline_logloss_concede': [],
        'baseline_brier_corner': [], 'baseline_logloss_corner': [],
        'baseline_brier_shot': [], 'baseline_logloss_shot': []
    }
    
    print(f"\nWalk-Forward Validation ({n_folds} folds)")
    print(f"Total matches: {n_matches}")
    print(f"Fold size: ~{fold_size} matches")
    print("-" * 60)
    
    for fold in range(n_folds):
        # Test set: last fold_size matches of this iteration
        test_end = (fold + 1) * fold_size
        test_start = fold * fold_size
        
        test_matches = unique_matches[test_start:test_end]
        train_matches = unique_matches[:test_start] if test_start > 0 else []
        
        if len(train_matches) < 10:  # Need enough training data
            print(f"Fold {fold+1}: Skipping (insufficient training data)")
            continue
        
        train_df = df[df['match_id'].isin(train_matches)]
        test_df = df[df['match_id'].isin(test_matches)]
        
        print(f"Fold {fold+1}: Train={len(train_matches)} matches, Test={len(test_matches)} matches")
        
        # Build model on training data
        trans_probs, event_probs = build_markov_model(train_df, min_sample)
        baseline_probs = compute_baseline_probs(train_df)
        
        # Evaluate on test data
        fold_results = {
            'fold': fold + 1,
            'train_matches': len(train_matches),
            'test_matches': len(test_matches),
            'train_transitions': len(train_df),
            'test_transitions': len(test_df)
        }
        
        # Markov model predictions
        markov_goals_true = []
        markov_goals_pred = []
        markov_concede_true = []
        markov_concede_pred = []
        markov_corner_true = []
        markov_corner_pred = []
        markov_shot_true = []
        markov_shot_pred = []
        
        # Baseline predictions
        baseline_goals_pred = [baseline_probs['p_goal']] * len(test_df)
        baseline_concede_pred = [baseline_probs['p_concede']] * len(test_df)
        baseline_corner_pred = [baseline_probs['p_corner']] * len(test_df)
        baseline_shot_pred = [baseline_probs['p_shot']] * len(test_df)
        
        for _, row in test_df.iterrows():
            state_t = state_to_key(parse_state(row.get('state_t')))
            
            # Markov predictions (use baseline if state not in model)
            if state_t in event_probs:
                markov_goals_pred.append(event_probs[state_t]['p_goal'])
                markov_concede_pred.append(event_probs[state_t]['p_concede'])
                markov_corner_pred.append(event_probs[state_t]['p_corner'])
                markov_shot_pred.append(event_probs[state_t]['p_shot'])
            else:
                markov_goals_pred.append(baseline_probs['p_goal'])
                markov_concede_pred.append(baseline_probs['p_concede'])
                markov_corner_pred.append(baseline_probs['p_corner'])
                markov_shot_pred.append(baseline_probs['p_shot'])
            
            # True values
            markov_goals_true.append(1 if row.get('goals_next_window', 0) > 0 else 0)
            markov_concede_true.append(1 if row.get('concede_next_window', 0) > 0 else 0)
            markov_corner_true.append(1 if row.get('corners_next_window', 0) > 0 else 0)
            markov_shot_true.append(1 if row.get('shots_next_window', 0) > 0 else 0)
        
        # Evaluate Markov
        if len(markov_goals_true) > 0:
            metrics = evaluate_predictions(markov_goals_true, markov_goals_pred, 'markov_goal')
            fold_results.update(metrics)
            
            metrics = evaluate_predictions(markov_concede_true, markov_concede_pred, 'markov_concede')
            fold_results.update(metrics)
            
            metrics = evaluate_predictions(markov_corner_true, markov_corner_pred, 'markov_corner')
            fold_results.update(metrics)
            
            metrics = evaluate_predictions(markov_shot_true, markov_shot_pred, 'markov_shot')
            fold_results.update(metrics)
        
        # Evaluate Baseline
        baseline_goals_true = test_df['goals_next_window'].fillna(0).astype(int) > 0
        baseline_concede_true = test_df['concede_next_window'].fillna(0).astype(int) > 0
        baseline_corner_true = test_df['corners_next_window'].fillna(0).astype(int) > 0
        baseline_shot_true = test_df['shots_next_window'].fillna(0).astype(int) > 0
        
        metrics = evaluate_predictions(baseline_goals_true.astype(int), baseline_goals_pred, 'baseline_goal')
        fold_results.update(metrics)
        
        metrics = evaluate_predictions(baseline_concede_true.astype(int), baseline_concede_pred, 'baseline_concede')
        fold_results.update(metrics)
        
        metrics = evaluate_predictions(baseline_corner_true.astype(int), baseline_corner_pred, 'baseline_corner')
        fold_results.update(metrics)
        
        metrics = evaluate_predictions(baseline_shot_true.astype(int), baseline_shot_pred, 'baseline_shot')
        fold_results.update(metrics)
        
        fold_metrics.append(fold_results)
        
        # Accumulate for overall metrics
        for key in all_metrics:
            if key in fold_results:
                all_metrics[key].append(fold_results[key])
        
        # Print fold summary
        if 'markov_brier_goal' in fold_results:
            print(f"  Markov Goal Brier: {fold_results['markov_brier_goal']:.4f} vs Baseline: {fold_results['baseline_brier_goal']:.4f}")
    
    # Overall summary
    overall_summary = {}
    for key, values in all_metrics.items():
        if values:
            overall_summary[f'{key}_mean'] = np.mean(values)
            overall_summary[f'{key}_std'] = np.std(values)
    
    return pd.DataFrame(fold_metrics), overall_summary


def generate_validation_report(fold_metrics_df, overall_summary, output_path):
    """Generate markdown validation report."""
    import math
    
    report = []
    report.append("# Markov Model Validation Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## Walk-Forward Validation Summary\n")
    
    report.append("### Fold-by-Fold Metrics\n")
    report.append("| Fold | Train Matches | Test Matches | Markov Goal Brier | Baseline Goal Brier | Markov Concede Brier | Baseline Concede Brier |")
    report.append("|------|---------------|--------------|-------------------|---------------------|----------------------|------------------------|")
    
    for _, row in fold_metrics_df.iterrows():
        def fmt(val, digits=4):
            if val is None:
                return "N/A"
            try:
                float_val = float(val)
                if math.isnan(float_val) or math.isinf(float_val):
                    return "N/A"
                return f"{float_val:.{digits}f}"
            except (TypeError, ValueError):
                return "N/A"
        
        markov_brier_goal = row.get('markov_goal_brier')
        baseline_brier_goal = row.get('baseline_goal_brier')
        markov_brier_concede = row.get('markov_concede_brier')
        baseline_brier_concede = row.get('baseline_concede_brier')
        
        report.append(
            f"| {int(row['fold'])} | {int(row['train_matches'])} | {int(row['test_matches'])} | "
            f"{fmt(markov_brier_goal)} | "
            f"{fmt(baseline_brier_goal)} | "
            f"{fmt(markov_brier_concede)} | "
            f"{fmt(baseline_brier_concede)} |"
        )
    
    report.append("\n### Overall Performance\n")
    report.append("| Metric | Mean | Std Dev |")
    report.append("|--------|------|---------|")
    
    for key, value in sorted(overall_summary.items()):
        if '_mean' in key:
            metric_name = key.replace('_mean', '').replace('_', ' ').title()
            std_key = key.replace('_mean', '_std')
            std_value = overall_summary.get(std_key, 0)
            report.append(f"| {metric_name} | {value:.4f} | {std_value:.4f} |")
    
    report.append("\n### Interpretation\n")
    report.append("- **Brier Score**: Lower is better (range 0-1 for binary outcomes)")
    report.append("- **Log Loss**: Lower is better (penalizes confident wrong predictions)")
    report.append("- **Markov vs Baseline**: Markov should outperform baseline if state information is valuable")
    
    report.append("\n### Recommendations\n")
    
    # Check if Markov outperforms baseline
    if 'markov_brier_goal_mean' in overall_summary and 'baseline_brier_goal_mean' in overall_summary:
        markov_brier = overall_summary['markov_brier_goal_mean']
        baseline_brier = overall_summary['baseline_brier_goal_mean']
        
        if markov_brier < baseline_brier:
            improvement = (baseline_brier - markov_brier) / baseline_brier * 100
            report.append(f"- ✅ Markov model improves goal prediction Brier score by {improvement:.1f}% vs baseline")
        else:
            report.append(f"- ⚠️ Markov model does not improve over baseline for goal prediction")
    
    report.append("\n---")
    report.append("*Report generated by validate_markov_model.py*")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))


def main():
    parser = argparse.ArgumentParser(
        description='Validate Markov model with walk-forward testing'
    )
    parser.add_argument('--input', required=True, help='Input CSV with markov_ready_transitions')
    parser.add_argument('--output-dir', required=True, help='Output directory for validation results')
    parser.add_argument('--method', default='walk-forward', choices=['walk-forward'],
                        help='Validation method (default: walk-forward)')
    parser.add_argument('--min-state-sample', type=int, default=20,
                        help='Minimum sample size per state (default: 20)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of walk-forward folds (default: 5)')
    parser.add_argument('--coverage', help='Optional: coverage report CSV for match dates')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading transitions...")
    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df)} transitions")
    
    # Load coverage report for dates if available
    coverage_df = None
    if args.coverage and os.path.exists(args.coverage):
        print(f"  Loading coverage report for dates...")
        coverage_df = pd.read_csv(args.coverage)
    
    # Get match dates
    match_dates = load_match_dates(df, coverage_df)
    print(f"  Match dates mapped for {len(match_dates)} matches")
    
    # Run walk-forward validation
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running {args.method} validation...")
    fold_metrics_df, overall_summary = walk_forward_validation(
        df, match_dates, args.min_state_sample, args.n_folds
    )
    
    # Save fold metrics
    fold_metrics_path = os.path.join(args.output_dir, 'fold_metrics.csv')
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    print(f"\n  Saved fold metrics: {fold_metrics_path}")
    
    # Generate markdown report
    report_path = os.path.join(args.output_dir, 'markov_validation_report.md')
    generate_validation_report(fold_metrics_df, overall_summary, report_path)
    print(f"  Saved validation report: {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDACIÓN WALK-FORWARD - RESUMEN")
    print("="*60)
    
    if fold_metrics_df is not None and len(fold_metrics_df) > 0:
        print(f"Folds ejecutados: {len(fold_metrics_df)}")
        
        if 'markov_brier_goal_mean' in overall_summary:
            print(f"\nGoal Prediction:")
            print(f"  Markov Brier:  {overall_summary['markov_brier_goal_mean']:.4f} (+/- {overall_summary.get('markov_brier_goal_std', 0):.4f})")
            print(f"  Baseline Brier: {overall_summary['baseline_brier_goal_mean']:.4f} (+/- {overall_summary.get('baseline_brier_goal_std', 0):.4f})")
            
            if overall_summary['markov_brier_goal_mean'] < overall_summary['baseline_brier_goal_mean']:
                improvement = (overall_summary['baseline_brier_goal_mean'] - overall_summary['markov_brier_goal_mean']) / overall_summary['baseline_brier_goal_mean'] * 100
                print(f"  → Markov mejora {improvement:.1f}% sobre baseline")
        
        if 'markov_brier_concede_mean' in overall_summary:
            print(f"\nConcede Prediction:")
            print(f"  Markov Brier:  {overall_summary['markov_brier_concede_mean']:.4f} (+/- {overall_summary.get('markov_brier_concede_std', 0):.4f})")
            print(f"  Baseline Brier: {overall_summary['baseline_brier_concede_mean']:.4f} (+/- {overall_summary.get('baseline_brier_concede_std', 0):.4f})")
    
    print("\nArchivos generados:")
    print(f"  - {fold_metrics_path}")
    print(f"  - {report_path}")
    print("="*60)


if __name__ == '__main__':
    main()
