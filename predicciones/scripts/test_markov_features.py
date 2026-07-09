#!/usr/bin/env python3
"""
Basic tests for Markov Features module.

Run: python scripts/test_markov_features.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from predicciones.src.features.markov_features import (
    load_markov_tables,
    get_markov_features,
    build_state_from_match_context,
    build_state_for_away_team,
    state_to_key,
    clear_cache,
)


def test_load_tables():
    """Test that tables load correctly."""
    print("Test 1: Loading Markov tables...")
    
    markov_dir = project_root / "output" / "markov"
    event_probs_path = markov_dir / "state_event_probabilities.csv"
    baselines_path = markov_dir / "baseline_probabilities.csv"
    
    event_probs_df, baselines = load_markov_tables(event_probs_path, baselines_path)
    
    assert event_probs_df is not None, "Event probs DataFrame should not be None"
    assert len(event_probs_df) > 0, "Event probs should have rows"
    assert 'state_t' in event_probs_df.columns, "Missing state_t column"
    assert 'p_goal_next_window' in event_probs_df.columns, "Missing p_goal_next_window column"
    
    assert 'global' in baselines, "Missing global baseline"
    assert 'by_minute' in baselines, "Missing by_minute baseline"
    assert 'by_score' in baselines, "Missing by_score baseline"
    
    print(f"  ✓ Loaded {len(event_probs_df)} states")
    print(f"  ✓ Baselines: {list(baselines.keys())}")
    return True


def test_get_features_known_state():
    """Test feature lookup for a known state."""
    print("\nTest 2: Feature lookup for known state...")
    
    state = build_state_from_match_context(
        minute=5,
        score_diff=0,
        home_red_cards=0,
        away_red_cards=0,
    )
    
    features = get_markov_features(state)
    
    # Check required keys exist
    required_keys = [
        'markov_expected_shots_next_window',
        'markov_p_shot_next_window_ge1',
        'markov_expected_corners_next_window',
        'markov_p_corner_next_window_ge1',
        'markov_p_goal_next_window',
        'markov_p_concede_next_window',
        'delta_shot_vs_minute_baseline',
        'delta_corner_vs_minute_baseline',
        'delta_goal_vs_minute_baseline',
    ]
    
    for key in required_keys:
        assert key in features, f"Missing required key: {key}"
    
    # Check metadata exists
    assert '_markov_metadata' in features, "Missing _markov_metadata"
    metadata = features['_markov_metadata']
    assert 'state_key' in metadata, "Missing state_key in metadata"
    assert 'sample_size' in metadata, "Missing sample_size in metadata"
    
    # Check values are reasonable
    assert features['markov_p_goal_next_window'] >= 0, "Probability should be >= 0"
    assert features['markov_p_goal_next_window'] <= 1, "Probability should be <= 1"
    assert features['markov_expected_shots_next_window'] >= 0, "Expected shots should be >= 0"
    
    print(f"  ✓ All required keys present")
    print(f"  ✓ Sample size: {metadata.get('sample_size')}")
    print(f"  ✓ Warning: {metadata.get('warning')}")
    return True


def test_get_features_unknown_state():
    """Test fallback behavior for unknown state."""
    print("\nTest 3: Fallback for unknown state...")
    
    # Create a state that likely doesn't exist
    unknown_state = {
        "minute_bucket": "0-15",
        "score_diff_bucket": "+2_or_more",
        "home_red_cards": "2",  # Very rare
        "away_red_cards": "2",
        "phase": "regular_time",
        "strength_gap_bucket": "unknown",
        "venue_context": "neutral"
    }
    
    features = get_markov_features(unknown_state)
    metadata = features['_markov_metadata']
    
    # Should use fallback
    assert metadata.get('fallback_used', False) == True, "Should use fallback for unknown state"
    assert metadata.get('warning', False) == True, "Should flag warning for unknown state"
    
    # Values should still be valid (from baseline)
    assert features['markov_p_goal_next_window'] > 0, "Fallback should provide valid probability"
    
    print(f"  ✓ Fallback used: {metadata.get('fallback_used')}")
    print(f"  ✓ Warning flagged: {metadata.get('warning')}")
    return True


def test_low_sample_smoothing():
    """Test smoothing for low-sample states."""
    print("\nTest 4: Smoothing for low-sample states...")
    
    # Find a state with low sample size
    clear_cache()
    markov_dir = project_root / "output" / "markov"
    event_probs_df, baselines = load_markov_tables(
        markov_dir / "state_event_probabilities.csv",
        markov_dir / "baseline_probabilities.csv"
    )
    
    # Look for a state with warning=True
    low_sample_states = event_probs_df[event_probs_df['warning'] == True]
    
    if len(low_sample_states) > 0:
        # Get the first low-sample state
        row = low_sample_states.iloc[0]
        state_key = row['state_t']
        
        # Parse state key back to dict
        parts = state_key.split('|')
        state = {}
        for part in parts:
            k, v = part.split('=')
            state[k] = v
        
        features = get_markov_features(state, event_probs_df, baselines)
        metadata = features['_markov_metadata']
        
        # Should have warning
        assert metadata.get('warning', False) == True, "Should flag warning for low-sample state"
        
        print(f"  ✓ Found low-sample state with n={metadata.get('sample_size')}")
        print(f"  ✓ Warning flagged: {metadata.get('warning')}")
    else:
        print("  ⊘ No low-sample states found (all have n >= 20)")
    
    return True


def test_build_state_helpers():
    """Test state building helper functions."""
    print("\nTest 5: State building helpers...")
    
    # Test build_state_from_match_context
    state = build_state_from_match_context(
        minute=50,
        score_diff=-1,
        home_red_cards=0,
        away_red_cards=1,
    )
    
    assert state['minute_bucket'] == "46-60", f"Wrong minute bucket: {state['minute_bucket']}"
    assert state['score_diff_bucket'] == "-1", f"Wrong score_diff: {state['score_diff_bucket']}"
    assert state['away_red_cards'] == "1", f"Wrong away_red_cards: {state['away_red_cards']}"
    
    # Test build_state_for_away_team
    home_state = {
        "minute_bucket": "46-60",
        "score_diff_bucket": "+1",
        "home_red_cards": "0",
        "away_red_cards": "1",
        "phase": "regular_time",
        "strength_gap_bucket": "unknown",
        "venue_context": "neutral"
    }
    
    away_state = build_state_for_away_team(home_state)
    
    assert away_state['score_diff_bucket'] == "-1", "Away score_diff should be inverted"
    assert away_state['home_red_cards'] == "1", "Away perspective: home_reds should be swapped"
    assert away_state['away_red_cards'] == "0", "Away perspective: away_reds should be swapped"
    
    print(f"  ✓ Minute buckets correct")
    print(f"  ✓ Score diff inversion works")
    print(f"  ✓ Red card swap works")
    return True


def test_both_teams_features():
    """Test getting features for both teams."""
    print("\nTest 6: Both teams features...")
    
    from predicciones.src.features.markov_features import get_markov_features_for_both_teams
    
    home_state = build_state_from_match_context(minute=55, score_diff=0)
    away_state = build_state_for_away_team(home_state)
    
    combined = get_markov_features_for_both_teams(home_state, away_state)
    
    # Check home and away prefixes
    assert 'home_markov_expected_shots_next_window' in combined, "Missing home_ prefix"
    assert 'away_markov_expected_shots_next_window' in combined, "Missing away_ prefix"
    
    # Metadata should be separate
    assert '_home_markov_metadata' in combined, "Missing home metadata"
    assert '_away_markov_metadata' in combined, "Missing away metadata"
    
    print(f"  ✓ Home/away prefixes correct")
    print(f"  ✓ Metadata separated correctly")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("MARKOV FEATURES MODULE - BASIC TESTS")
    print("=" * 60)
    
    tests = [
        ("Load tables", test_load_tables),
        ("Known state features", test_get_features_known_state),
        ("Unknown state fallback", test_get_features_unknown_state),
        ("Low-sample smoothing", test_low_sample_smoothing),
        ("State helpers", test_build_state_helpers),
        ("Both teams features", test_both_teams_features),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"  PASS: {name}")
            else:
                failed += 1
                print(f"  FAIL: {name}")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {name} - {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
