#!/usr/bin/env python3
"""
Build Markov Transition Matrix and Conditional Event Probabilities

Fase 2: Construye matrices de transición y probabilidades condicionales de eventos
dado el estado del partido.

Uso:
python scripts/build_markov_transition_matrix.py \
  --input output/analysis/markov_ready_transitions.csv \
  --output-dir output/markov \
  --min-state-sample 20
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np


def parse_state(state_str):
    """Parse state JSON string to dict."""
    if isinstance(state_str, dict):
        return state_str
    if pd.isna(state_str) or state_str == '':
        return None
    try:
        # Handle escaped quotes
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


def laplace_smooth(counts, n_categories, alpha=1.0):
    """Apply Laplace smoothing to counts."""
    total = sum(counts.values())
    smoothed = {}
    for cat, count in counts.items():
        smoothed[cat] = (count + alpha) / (total + alpha * n_categories)
    return smoothed


def build_transition_matrix(df, min_sample=20):
    """
    Build state transition matrix from transitions dataframe.
    
    Returns:
        - transition_counts: DataFrame with state_t, next_state_t1, transition_count
        - transition_probs: DataFrame with state_t, next_state_t1, transition_probability
        - state_samples: dict with sample size per state
    """
    # Group by state_t and next_state_t1
    state_transitions = defaultdict(lambda: defaultdict(int))
    state_totals = defaultdict(int)
    
    for _, row in df.iterrows():
        state_t = state_to_key(parse_state(row.get('state_t')))
        next_state = state_to_key(parse_state(row.get('next_state_t1')))
        
        if state_t and next_state:
            state_transitions[state_t][next_state] += 1
            state_totals[state_t] += 1
    
    # Build DataFrames
    transition_counts_rows = []
    transition_probs_rows = []
    
    all_next_states = set()
    for state_t in state_transitions:
        all_next_states.update(state_transitions[state_t].keys())
    
    n_next_states = len(all_next_states) if all_next_states else 1
    
    for state_t in sorted(state_transitions.keys()):
        transitions = state_transitions[state_t]
        total = state_totals[state_t]
        
        # Apply Laplace smoothing for probability estimation
        smoothed_probs = laplace_smooth(transitions, n_next_states, alpha=1.0)
        
        for next_state in sorted(transitions.keys()):
            count = transitions[next_state]
            prob = smoothed_probs[next_state]
            
            transition_counts_rows.append({
                'state_t': state_t,
                'next_state_t1': next_state,
                'transition_count': count,
                'sample_size': total,
                'warning': total < min_sample
            })
            
            transition_probs_rows.append({
                'state_t': state_t,
                'next_state_t1': next_state,
                'transition_probability': round(prob, 6),
                'sample_size': total,
                'warning': total < min_sample
            })
    
    transition_counts_df = pd.DataFrame(transition_counts_rows)
    transition_probs_df = pd.DataFrame(transition_probs_rows)
    
    return transition_counts_df, transition_probs_df, dict(state_totals)


def build_event_probabilities(df, min_sample=20):
    """
    Build conditional event probabilities given state.
    
    Events:
    - P(goal_next_window = 1 | state_t)
    - P(concede_next_window = 1 | state_t)
    - P(corner_next_window >= 1 | state_t)
    - P(shots_next_window >= 1 | state_t)
    - E[corners_next_window | state_t]
    - E[shots_next_window | state_t]
    - E[shots_on_target_next_window | state_t]
    """
    state_events = defaultdict(lambda: {
        'goals': [], 'conceded': [], 'corners': [], 'shots': [], 
        'shots_on_target': [], 'total': 0
    })
    
    for _, row in df.iterrows():
        state_t = state_to_key(parse_state(row.get('state_t')))
        if not state_t:
            continue
        
        state_events[state_t]['total'] += 1
        state_events[state_t]['goals'].append(int(row.get('goals_next_window', 0) or 0))
        state_events[state_t]['conceded'].append(int(row.get('concede_next_window', 0) or 0))
        state_events[state_t]['corners'].append(int(row.get('corners_next_window', 0) or 0))
        state_events[state_t]['shots'].append(int(row.get('shots_next_window', 0) or 0))
        state_events[state_t]['shots_on_target'].append(int(row.get('shots_on_target_next_window', 0) or 0))
    
    rows = []
    for state_t in sorted(state_events.keys()):
        data = state_events[state_t]
        n = data['total']
        
        goals = data['goals']
        conceded = data['conceded']
        corners = data['corners']
        shots = data['shots']
        shots_on_target = data['shots_on_target']
        
        # Probabilities (with Laplace smoothing for binary events)
        p_goal = (sum(1 for g in goals if g > 0) + 1) / (n + 2)
        p_concede = (sum(1 for c in conceded if c > 0) + 1) / (n + 2)
        p_corner = (sum(1 for c in corners if c > 0) + 1) / (n + 2)
        p_shot = (sum(1 for s in shots if s > 0) + 1) / (n + 2)
        
        # Expectations
        e_corners = np.mean(corners) if corners else 0
        e_shots = np.mean(shots) if shots else 0
        e_sot = np.mean(shots_on_target) if shots_on_target else 0
        
        rows.append({
            'state_t': state_t,
            'sample_size': n,
            'p_goal_next_window': round(p_goal, 6),
            'p_concede_next_window': round(p_concede, 6),
            'p_corner_next_window_ge1': round(p_corner, 6),
            'p_shot_next_window_ge1': round(p_shot, 6),
            'e_corners_next_window': round(e_corners, 4),
            'e_shots_next_window': round(e_shots, 4),
            'e_shots_on_target_next_window': round(e_sot, 4),
            'warning': n < min_sample
        })
    
    return pd.DataFrame(rows)


def build_baselines(df):
    """
    Build baseline probabilities without state conditioning.
    
    Baselines:
    - Global baseline (all data)
    - By minute_bucket only
    - By score_diff_bucket only
    """
    baselines = {}
    
    # Global baseline
    n = len(df)
    goals = df['goals_next_window'].fillna(0).astype(int)
    conceded = df['concede_next_window'].fillna(0).astype(int)
    corners = df['corners_next_window'].fillna(0).astype(int)
    shots = df['shots_next_window'].fillna(0).astype(int)
    
    baselines['global'] = {
        'p_goal': (sum(goals > 0) + 1) / (n + 2),
        'p_concede': (sum(conceded > 0) + 1) / (n + 2),
        'p_corner': (sum(corners > 0) + 1) / (n + 2),
        'p_shot': (sum(shots > 0) + 1) / (n + 2),
        'e_corners': corners.mean(),
        'e_shots': shots.mean(),
        'sample_size': n
    }
    
    # By minute_bucket
    minute_baselines = defaultdict(lambda: {'goals': [], 'conceded': [], 'corners': [], 'shots': []})
    for _, row in df.iterrows():
        state = parse_state(row.get('state_t'))
        if state:
            bucket = state.get('minute_bucket', 'unknown')
            minute_baselines[bucket]['goals'].append(int(row.get('goals_next_window', 0) or 0))
            minute_baselines[bucket]['conceded'].append(int(row.get('concede_next_window', 0) or 0))
            minute_baselines[bucket]['corners'].append(int(row.get('corners_next_window', 0) or 0))
            minute_baselines[bucket]['shots'].append(int(row.get('shots_next_window', 0) or 0))
    
    baselines['by_minute'] = {}
    for bucket, data in minute_baselines.items():
        n = len(data['goals'])
        if n > 0:
            baselines['by_minute'][bucket] = {
                'p_goal': (sum(1 for g in data['goals'] if g > 0) + 1) / (n + 2),
                'p_concede': (sum(1 for c in data['conceded'] if c > 0) + 1) / (n + 2),
                'p_corner': (sum(1 for c in data['corners'] if c > 0) + 1) / (n + 2),
                'p_shot': (sum(1 for s in data['shots'] if s > 0) + 1) / (n + 2),
                'sample_size': n
            }
    
    # By score_diff_bucket
    score_baselines = defaultdict(lambda: {'goals': [], 'conceded': [], 'corners': [], 'shots': []})
    for _, row in df.iterrows():
        state = parse_state(row.get('state_t'))
        if state:
            bucket = state.get('score_diff_bucket', 'unknown')
            score_baselines[bucket]['goals'].append(int(row.get('goals_next_window', 0) or 0))
            score_baselines[bucket]['conceded'].append(int(row.get('concede_next_window', 0) or 0))
            score_baselines[bucket]['corners'].append(int(row.get('corners_next_window', 0) or 0))
            score_baselines[bucket]['shots'].append(int(row.get('shots_next_window', 0) or 0))
    
    baselines['by_score'] = {}
    for bucket, data in score_baselines.items():
        n = len(data['goals'])
        if n > 0:
            baselines['by_score'][bucket] = {
                'p_goal': (sum(1 for g in data['goals'] if g > 0) + 1) / (n + 2),
                'p_concede': (sum(1 for c in data['conceded'] if c > 0) + 1) / (n + 2),
                'p_corner': (sum(1 for c in data['corners'] if c > 0) + 1) / (n + 2),
                'p_shot': (sum(1 for s in data['shots'] if s > 0) + 1) / (n + 2),
                'sample_size': n
            }
    
    return baselines


def generate_state_quality_report(df, state_samples, min_sample=20):
    """Generate state quality report."""
    states = list(state_samples.keys())
    n_states = len(states)
    
    # State frequency
    state_freq = sorted([(s, state_samples[s]) for s in states], key=lambda x: -x[1])
    
    # Rare states
    rare_states = [s for s, n in state_freq if n < min_sample]
    
    # Coverage analysis
    minute_buckets = set()
    score_buckets = set()
    strength_gap_unknown = 0
    venue_neutral = 0
    
    for state_str in states:
        state = parse_state(state_str)
        if state:
            minute_buckets.add(state.get('minute_bucket', 'unknown'))
            score_buckets.add(state.get('score_diff_bucket', 'unknown'))
            if state.get('strength_gap_bucket') == 'unknown':
                strength_gap_unknown += 1
            if state.get('venue_context') == 'neutral':
                venue_neutral += 1
    
    report_rows = []
    for state, count in state_freq:
        state_data = parse_state(state)
        report_rows.append({
            'state': state,
            'sample_size': count,
            'minute_bucket': state_data.get('minute_bucket') if state_data else 'unknown',
            'score_diff_bucket': state_data.get('score_diff_bucket') if state_data else 'unknown',
            'phase': state_data.get('phase') if state_data else 'unknown',
            'warning': count < min_sample
        })
    
    return pd.DataFrame(report_rows), {
        'n_unique_states': n_states,
        'n_rare_states': len(rare_states),
        'n_minute_buckets': len(minute_buckets),
        'n_score_buckets': len(score_buckets),
        'strength_gap_unknown_count': strength_gap_unknown,
        'venue_neutral_count': venue_neutral,
        'top_states': state_freq[:10],
        'rare_states': rare_states[:20]
    }


def main():
    parser = argparse.ArgumentParser(
        description='Build Markov transition matrix and conditional event probabilities'
    )
    parser.add_argument('--input', required=True, help='Input CSV with markov_ready_transitions')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--min-state-sample', type=int, default=20, 
                        help='Minimum sample size per state (default: 20)')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading transitions...")
    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df)} transitions")
    
    # Build transition matrix
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Building transition matrix...")
    trans_counts, trans_probs, state_samples = build_transition_matrix(df, args.min_state_sample)
    
    trans_counts_path = os.path.join(args.output_dir, 'state_transition_counts.csv')
    trans_probs_path = os.path.join(args.output_dir, 'state_transition_matrix.csv')
    
    trans_counts.to_csv(trans_counts_path, index=False)
    trans_probs.to_csv(trans_probs_path, index=False)
    
    print(f"  Saved transition counts: {trans_counts_path}")
    print(f"  Saved transition probabilities: {trans_probs_path}")
    print(f"  Unique states: {len(state_samples)}")
    
    # Build event probabilities
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Building event probabilities...")
    event_probs = build_event_probabilities(df, args.min_state_sample)
    
    event_probs_path = os.path.join(args.output_dir, 'state_event_probabilities.csv')
    event_probs.to_csv(event_probs_path, index=False)
    print(f"  Saved event probabilities: {event_probs_path}")
    
    # Build baselines
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Computing baselines...")
    baselines = build_baselines(df)
    
    # Save baselines summary
    baseline_rows = []
    baseline_rows.append({
        'baseline_type': 'global',
        'category': 'all',
        **baselines['global']
    })
    
    for bucket, data in baselines.get('by_minute', {}).items():
        baseline_rows.append({
            'baseline_type': 'by_minute',
            'category': bucket,
            **data
        })
    
    for bucket, data in baselines.get('by_score', {}).items():
        baseline_rows.append({
            'baseline_type': 'by_score',
            'category': bucket,
            **data
        })
    
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_path = os.path.join(args.output_dir, 'baseline_probabilities.csv')
    baseline_df.to_csv(baseline_path, index=False)
    print(f"  Saved baselines: {baseline_path}")
    
    # Generate state quality report
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating state quality report...")
    state_quality_df, quality_summary = generate_state_quality_report(df, state_samples, args.min_state_sample)
    
    state_quality_path = os.path.join(args.output_dir, 'state_quality_report.csv')
    state_quality_df.to_csv(state_quality_path, index=False)
    print(f"  Saved state quality report: {state_quality_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("FASE 2 MARKOV - RESUMEN")
    print("="*60)
    print(f"Transiciones procesadas: {len(df)}")
    print(f"Estados únicos: {quality_summary['n_unique_states']}")
    print(f"Estados raros (sample < {args.min_state_sample}): {quality_summary['n_rare_states']}")
    print(f"Buckets de minuto: {quality_summary['n_minute_buckets']}")
    print(f"Buckets de diferencia de gol: {quality_summary['n_score_buckets']}")
    print(f"Estados con strength_gap unknown: {quality_summary['strength_gap_unknown_count']}")
    print(f"Estados con venue neutral: {quality_summary['venue_neutral_count']}")
    
    warnings = trans_probs[trans_probs['warning']]['state_t'].nunique()
    print(f"\nEstados con warning (sample < {args.min_state_sample}): {warnings}")
    
    print("\nArchivos generados:")
    print(f"  - {trans_counts_path}")
    print(f"  - {trans_probs_path}")
    print(f"  - {event_probs_path}")
    print(f"  - {baseline_path}")
    print(f"  - {state_quality_path}")
    print("="*60)


if __name__ == '__main__':
    main()
