#!/usr/bin/env python3
"""
Demo: Markov Features for Match Prediction

This script demonstrates how to use the Markov features module to compute
features from match states and integrate them into a goal prediction model.

Usage:
    python scripts/demo_markov_features.py
    
The script will:
1. Load Markov probability tables from output/markov/
2. Show features for several example match states
3. Demonstrate fallback behavior for unknown states
4. Show how warnings are handled for low-sample states
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from predicciones.src.features.markov_features import (
    load_markov_tables,
    get_markov_features,
    get_markov_features_for_both_teams,
    build_state_from_match_context,
    build_state_for_away_team,
    state_to_key,
)


def print_separator(title: str = ""):
    """Print a visual separator."""
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def print_features(features: dict, prefix: str = ""):
    """Pretty-print Markov features."""
    for key, value in sorted(features.items()):
        if key.startswith('_'):
            continue  # Skip metadata in main output
        if isinstance(value, float):
            print(f"  {prefix}{key}: {value:.6f}")
        else:
            print(f"  {prefix}{key}: {value}")


def main():
    print_separator("DEMO: MARKOV FEATURES FOR MATCH PREDICTION")
    
    # Define paths relative to project root
    markov_dir = project_root / "output" / "markov"
    event_probs_path = markov_dir / "state_event_probabilities.csv"
    baselines_path = markov_dir / "baseline_probabilities.csv"
    
    # Check if files exist
    if not event_probs_path.exists():
        print(f"ERROR: Event probabilities file not found: {event_probs_path}")
        print("Run: python scripts/build_markov_transition_matrix.py first")
        return
    
    if not baselines_path.exists():
        print(f"ERROR: Baselines file not found: {baselines_path}")
        print("Run: python scripts/build_markov_transition_matrix.py first")
        return
    
    print(f"\nLoading Markov tables from:")
    print(f"  - {event_probs_path}")
    print(f"  - {baselines_path}")
    
    # Load tables
    event_probs_df, baselines = load_markov_tables(event_probs_path, baselines_path)
    print(f"\nLoaded {len(event_probs_df)} states from event probabilities table")
    print(f"Baseline types: {list(baselines.keys())}")
    print(f"  - Global baseline: p_goal={baselines['global']['p_goal']:.4f}")
    print(f"  - Minute buckets: {list(baselines['by_minute'].keys())}")
    print(f"  - Score buckets: {list(baselines['by_score'].keys())}")
    
    # -------------------------------------------------------------------------
    # Example 1: Early game, tied match
    # -------------------------------------------------------------------------
    print_separator("EXAMPLE 1: Early Game (0-15 min), Tied (0-0)")
    
    state_early_tied = build_state_from_match_context(
        minute=5,
        score_diff=0,
        home_red_cards=0,
        away_red_cards=0,
        phase="regular_time",
        venue_context="neutral"
    )
    
    print(f"\nState: {state_to_key(state_early_tied)}")
    features = get_markov_features(state_early_tied, event_probs_df, baselines)
    print("\nMarkov Features:")
    print_features(features)
    
    metadata = features.get('_markov_metadata', {})
    print(f"\nMetadata:")
    print(f"  - Sample size: {metadata.get('sample_size', 'N/A')}")
    print(f"  - Warning: {metadata.get('warning', False)}")
    print(f"  - Fallback used: {metadata.get('fallback_used', False)}")
    
    # -------------------------------------------------------------------------
    # Example 2: Second half, home team trailing by 1
    # -------------------------------------------------------------------------
    print_separator("EXAMPLE 2: Second Half (46-60 min), Home Trailing (-1)")
    
    state_trailing = build_state_from_match_context(
        minute=50,
        score_diff=-1,
        home_red_cards=0,
        away_red_cards=0,
        phase="regular_time",
        venue_context="neutral"
    )
    
    print(f"\nState: {state_to_key(state_trailing)}")
    features = get_markov_features(state_trailing, event_probs_df, baselines)
    print("\nMarkov Features:")
    print_features(features)
    
    metadata = features.get('_markov_metadata', {})
    print(f"\nMetadata:")
    print(f"  - Sample size: {metadata.get('sample_size', 'N/A')}")
    print(f"  - Warning: {metadata.get('warning', False)}")
    
    # -------------------------------------------------------------------------
    # Example 3: Late game, home team leading by 1
    # -------------------------------------------------------------------------
    print_separator("EXAMPLE 3: Late Game (76-90+ min), Home Leading (+1)")
    
    state_leading_late = build_state_from_match_context(
        minute=80,
        score_diff=1,
        home_red_cards=0,
        away_red_cards=0,
        phase="regular_time",
        venue_context="neutral"
    )
    
    print(f"\nState: {state_to_key(state_leading_late)}")
    features = get_markov_features(state_leading_late, event_probs_df, baselines)
    print("\nMarkov Features:")
    print_features(features)
    
    metadata = features.get('_markov_metadata', {})
    print(f"\nMetadata:")
    print(f"  - Sample size: {metadata.get('sample_size', 'N/A')}")
    print(f"  - Warning: {metadata.get('warning', False)}")
    
    # -------------------------------------------------------------------------
    # Example 4: Both teams' perspective (home vs away)
    # -------------------------------------------------------------------------
    print_separator("EXAMPLE 4: Both Teams' Perspective")
    
    home_state = build_state_from_match_context(
        minute=55,
        score_diff=0,
        home_red_cards=0,
        away_red_cards=0,
        phase="regular_time",
        venue_context="neutral"
    )
    
    away_state = build_state_for_away_team(home_state)
    
    print(f"\nHome team state: {state_to_key(home_state)}")
    print(f"Away team state: {state_to_key(away_state)}")
    
    combined = get_markov_features_for_both_teams(
        home_state, away_state, event_probs_df, baselines
    )
    
    print("\nCombined Features (home_*/away_* prefixes):")
    for key, value in sorted(combined.items()):
        if key.startswith('_'):
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # -------------------------------------------------------------------------
    # Example 5: Unknown state (fallback demonstration)
    # -------------------------------------------------------------------------
    print_separator("EXAMPLE 5: Unknown State (Fallback Behavior)")
    
    # Create a state that likely doesn't exist in the data
    unknown_state = {
        "minute_bucket": "0-15",
        "score_diff_bucket": "+2_or_more",  # Rare in early minutes
        "home_red_cards": "1",  # Red card in first 15 min is rare
        "away_red_cards": "1",
        "phase": "regular_time",
        "strength_gap_bucket": "unknown",
        "venue_context": "neutral"
    }
    
    print(f"\nState: {state_to_key(unknown_state)}")
    print("(This state likely has very few or no samples)")
    
    features = get_markov_features(unknown_state, event_probs_df, baselines)
    print("\nMarkov Features (with fallback):")
    print_features(features)
    
    metadata = features.get('_markov_metadata', {})
    print(f"\nMetadata:")
    print(f"  - Sample size: {metadata.get('sample_size', 'N/A')}")
    print(f"  - Warning: {metadata.get('warning', False)}")
    print(f"  - Fallback used: {metadata.get('fallback_used', False)}")
    
    # -------------------------------------------------------------------------
    # Example 6: Comparison with baselines
    # -------------------------------------------------------------------------
    print_separator("EXAMPLE 6: Delta vs Baseline Analysis")
    
    test_states = [
        ("Early tied", 5, 0),
        ("Mid tied", 35, 0),
        ("Second half tied", 55, 0),
        ("Late tied", 80, 0),
    ]
    
    print("\nComparing expected shots vs minute baseline:")
    print(f"{'State':<20} {'E[Shots]':>10} {'Baseline':>10} {'Delta':>10}")
    print("-" * 55)
    
    for name, minute, score_diff in test_states:
        state = build_state_from_match_context(minute=minute, score_diff=score_diff)
        features = get_markov_features(state, event_probs_df, baselines)
        
        e_shots = features['markov_expected_shots_next_window']
        delta = features['delta_shot_vs_minute_baseline']
        baseline = e_shots - delta
        
        print(f"{name:<20} {e_shots:>10.4f} {baseline:>10.4f} {delta:>10.4f}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_separator("SUMMARY")
    
    print("""
Key features available for integration:

1. INTENSITY FEATURES (shots/corners):
   - markov_expected_shots_next_window: E[shots] in next window
   - markov_p_shot_next_window_ge1: P(at least 1 shot)
   - markov_expected_corners_next_window: E[corners] in next window
   - markov_p_corner_next_window_ge1: P(at least 1 corner)

2. GOAL RISK FEATURES (softer signal):
   - markov_p_goal_next_window: P(score at least 1 goal)
   - markov_p_concede_next_window: P(concede at least 1 goal)

3. DELTA FEATURES (vs baseline):
   - delta_shot_vs_minute_baseline: deviation from minute-based average
   - delta_corner_vs_minute_baseline: deviation from minute-based average
   - delta_goal_vs_minute_baseline: deviation from minute-based average

USAGE IN MODEL:
   
   from predicciones.src.features.markov_features import (
       load_markov_tables,
       get_markov_features,
       build_state_from_match_context
   )
   
   # At startup
   event_probs, baselines = load_markov_tables(...)
   
   # During match prediction
   state = build_state_from_match_context(minute=55, score_diff=0, ...)
   features = get_markov_features(state, event_probs, baselines)
   
   # Add to your model's feature vector
   model_input.update(features)
""")
    
    print_separator("DEMO COMPLETE")


if __name__ == "__main__":
    main()
