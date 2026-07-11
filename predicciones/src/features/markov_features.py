"""
Markov Features Module

Provides functions to lookup and compute Markov-based features from pre-computed
state_event_probabilities.csv and baseline_probabilities.csv tables.

This module is read-only: it consumes the output of the Markov analysis pipeline
without modifying any underlying data.

Usage:
    from football_predictor.markov_features import get_markov_features, load_markov_tables
    
    # Load tables (do this once at startup)
    event_probs, baselines = load_markov_tables(
        event_probs_path="output/markov/state_event_probabilities.csv",
        baselines_path="output/markov/baseline_probabilities.csv"
    )
    
    # Define current match state
    state_t = {
        "minute_bucket": "46-60",
        "score_diff_bucket": "-1",
        "home_red_cards": "0",
        "away_red_cards": "0",
        "phase": "regular_time",
        "strength_gap_bucket": "unknown",
        "venue_context": "neutral"
    }
    
    # Get features for this state
    features = get_markov_features(state_t, event_probs, baselines)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Global cache for loaded tables (optional, for efficiency)
# -----------------------------------------------------------------------------
_cached_event_probs: Optional[pd.DataFrame] = None
_cached_baselines: Optional[Dict[str, Any]] = None


def _parse_state_json(state_str: str) -> Optional[Dict[str, Any]]:
    """
    Parse state JSON string to dict.
    
    Handles escaped quotes ("") format used in CSV exports.
    """
    if isinstance(state_str, dict):
        return state_str
    if pd.isna(state_str) or state_str == '':
        return None
    try:
        cleaned = state_str.replace('""', '"')
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None


def state_to_key(state_dict: Optional[Dict[str, Any]]) -> str:
    """
    Convert state dict to hashable key (same format as in CSV files).
    
    Keys are sorted alphabetically and joined with '|'.
    Example: "away_red_cards=0|home_red_cards=0|minute_bucket=46-60|..."
    """
    if state_dict is None:
        return "unknown"
    parts = []
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        parts.append(f"{k}={v}")
    return "|".join(parts)


def load_markov_tables(
    event_probs_path: str | Path,
    baselines_path: str | Path,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load Markov probability tables from CSV files.
    
    Args:
        event_probs_path: Path to state_event_probabilities.csv
        baselines_path: Path to baseline_probabilities.csv
        use_cache: If True, cache loaded tables globally (default).
    
    Returns:
        Tuple of (event_probs_df, baselines_dict)
        - event_probs_df: DataFrame with state_t index and probability columns
        - baselines_dict: Dict with 'global', 'by_minute', 'by_score' keys
    """
    global _cached_event_probs, _cached_baselines
    
    if use_cache and _cached_event_probs is not None and _cached_baselines is not None:
        return _cached_event_probs, _cached_baselines
    
    event_probs_path = Path(event_probs_path)
    baselines_path = Path(baselines_path)
    
    logger.info(f"Loading Markov event probabilities from {event_probs_path}")
    event_probs_df = pd.read_csv(event_probs_path)
    
    logger.info(f"Loading Markov baselines from {baselines_path}")
    baselines_df = pd.read_csv(baselines_path)
    
    # Build baselines dict
    baselines = {
        'global': {},
        'by_minute': {},
        'by_score': {}
    }
    
    for _, row in baselines_df.iterrows():
        baseline_type = row['baseline_type']
        category = row['category']
        
        baseline_data = {
            'p_goal': float(row.get('p_goal', 0)),
            'p_concede': float(row.get('p_concede', 0)),
            'p_corner': float(row.get('p_corner', 0)),
            'p_shot': float(row.get('p_shot', 0)),
            'sample_size': int(row.get('sample_size', 0))
        }
        
        # Also include e_corners/e_shots if available (for future extension)
        if 'e_corners' in row and pd.notna(row['e_corners']):
            baseline_data['e_corners'] = float(row['e_corners'])
        if 'e_shots' in row and pd.notna(row['e_shots']):
            baseline_data['e_shots'] = float(row['e_shots'])
        
        if baseline_type == 'global':
            baselines['global'] = baseline_data
        elif baseline_type == 'by_minute':
            baselines['by_minute'][category] = baseline_data
        elif baseline_type == 'by_score':
            baselines['by_score'][category] = baseline_data
    
    if use_cache:
        _cached_event_probs = event_probs_df
        _cached_baselines = baselines
    
    return event_probs_df, baselines


def clear_cache() -> None:
    """Clear cached tables (useful for testing or reloading)."""
    global _cached_event_probs, _cached_baselines
    _cached_event_probs = None
    _cached_baselines = None


def _lookup_state_in_event_probs(
    state_key: str,
    event_probs_df: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """
    Lookup a state in the event probabilities DataFrame.
    
    Returns the row as a dict if found, None otherwise.
    """
    mask = event_probs_df['state_t'] == state_key
    if mask.any():
        row = event_probs_df[mask].iloc[0]
        return row.to_dict()
    return None


def _get_baseline_by_minute(
    minute_bucket: str,
    baselines: Dict[str, Any],
) -> Dict[str, float]:
    """Get baseline probabilities for a specific minute bucket."""
    by_minute = baselines.get('by_minute', {})
    if minute_bucket in by_minute:
        return by_minute[minute_bucket]
    # Fallback to global
    return baselines.get('global', {})


def _get_baseline_by_score(
    score_diff_bucket: str,
    baselines: Dict[str, Any],
) -> Dict[str, float]:
    """Get baseline probabilities for a specific score_diff bucket."""
    by_score = baselines.get('by_score', {})
    if score_diff_bucket in by_score:
        return by_score[score_diff_bucket]
    # Fallback to global
    return baselines.get('global', {})


def get_markov_features(
    state_t: Dict[str, Any],
    event_probs_df: Optional[pd.DataFrame] = None,
    baselines: Optional[Dict[str, Any]] = None,
    min_sample_warning_threshold: int = 20,
) -> Dict[str, Any]:
    """
    Compute Markov features for a given match state.
    
    This function looks up the state in pre-computed tables and returns
    a feature dict suitable for inclusion in a goal prediction model.
    
    Args:
        state_t: Current match state dict with keys:
            - minute_bucket (e.g., "46-60")
            - score_diff_bucket (e.g., "0", "+1", "-1", "+2_or_more", "-2_or_more")
            - home_red_cards (e.g., "0")
            - away_red_cards (e.g., "0")
            - phase (e.g., "regular_time")
            - strength_gap_bucket (e.g., "unknown")
            - venue_context (e.g., "neutral")
        event_probs_df: Pre-loaded event probabilities DataFrame.
            If None, will attempt to use cached table.
        baselines: Pre-loaded baselines dict.
            If None, will attempt to use cached tables.
        min_sample_warning_threshold: Sample size below which to flag warnings.
    
    Returns:
        Dict with Markov features:
        - markov_expected_shots_next_window
        - markov_p_shot_next_window_ge1
        - markov_expected_corners_next_window
        - markov_p_corner_next_window_ge1
        - markov_p_goal_next_window
        - markov_p_concede_next_window
        - delta_shot_vs_minute_baseline
        - delta_corner_vs_minute_baseline
        - delta_goal_vs_minute_baseline (optional)
        - _markov_metadata: dict with sample_size, warning, state_key, fallback_used
    
    Raises:
        ValueError: If required tables are not available and not cached.
    """
    # Load tables if not provided
    if event_probs_df is None or baselines is None:
        global _cached_event_probs, _cached_baselines
        if _cached_event_probs is None or _cached_baselines is None:
            raise ValueError(
                "Markov tables not loaded. Call load_markov_tables() first "
                "or pass event_probs_df and baselines explicitly."
            )
        event_probs_df = _cached_event_probs
        baselines = _cached_baselines
    
    # Build state key
    state_key = state_to_key(state_t)
    minute_bucket = state_t.get('minute_bucket', 'unknown')
    score_diff_bucket = state_t.get('score_diff_bucket', '0')
    
    # Lookup state in event probabilities
    state_data = _lookup_state_in_event_probs(state_key, event_probs_df)
    
    # Track metadata
    metadata = {
        'state_key': state_key,
        'sample_size': None,
        'warning': False,
        'fallback_used': False
    }
    
    # Initialize features with baseline fallback values
    fallback_baseline = _get_baseline_by_minute(minute_bucket, baselines)
    
    if state_data is None:
        # State not found: use minute-based baseline as fallback
        logger.debug(f"State not found: {state_key}. Using minute baseline fallback.")
        metadata['fallback_used'] = True
        metadata['warning'] = True
        
        # Use baseline values
        p_goal = fallback_baseline.get('p_goal', baselines['global']['p_goal'])
        p_concede = fallback_baseline.get('p_concede', baselines['global']['p_concede'])
        p_corner = fallback_baseline.get('p_corner', baselines['global']['p_corner'])
        p_shot = fallback_baseline.get('p_shot', baselines['global']['p_shot'])
        e_corners = fallback_baseline.get('e_corners', 0.0)
        e_shots = fallback_baseline.get('e_shots', 0.0)
        
    else:
        # State found: extract values
        metadata['sample_size'] = int(state_data.get('sample_size', 0))
        metadata['warning'] = state_data.get('warning', False) or (
            metadata['sample_size'] < min_sample_warning_threshold
        )
        
        p_goal = float(state_data.get('p_goal_next_window', 0))
        p_concede = float(state_data.get('p_concede_next_window', 0))
        p_corner = float(state_data.get('p_corner_next_window_ge1', 0))
        p_shot = float(state_data.get('p_shot_next_window_ge1', 0))
        e_corners = float(state_data.get('e_corners_next_window', 0))
        e_shots = float(state_data.get('e_shots_next_window', 0))
        
        # Apply smoothing for low-sample states
        if metadata['warning']:
            # Blend with baseline (50/50 for very low samples)
            blend_factor = min(1.0, metadata['sample_size'] / min_sample_warning_threshold)
            p_goal = blend_factor * p_goal + (1 - blend_factor) * fallback_baseline.get('p_goal', p_goal)
            p_concede = blend_factor * p_concede + (1 - blend_factor) * fallback_baseline.get('p_concede', p_concede)
            p_corner = blend_factor * p_corner + (1 - blend_factor) * fallback_baseline.get('p_corner', p_corner)
            p_shot = blend_factor * p_shot + (1 - blend_factor) * fallback_baseline.get('p_shot', p_shot)
            e_corners = blend_factor * e_corners + (1 - blend_factor) * fallback_baseline.get('e_corners', e_corners)
            e_shots = blend_factor * e_shots + (1 - blend_factor) * fallback_baseline.get('e_shots', e_shots)
    
    # Compute deltas vs minute baseline
    minute_baseline = _get_baseline_by_minute(minute_bucket, baselines)
    
    baseline_shots = minute_baseline.get('e_shots', 0.0)
    baseline_corners = minute_baseline.get('e_corners', 0.0)
    baseline_p_goal = minute_baseline.get('p_goal', 0.0)
    
    delta_shot = e_shots - baseline_shots if baseline_shots > 0 else 0.0
    delta_corner = e_corners - baseline_corners if baseline_corners > 0 else 0.0
    delta_goal = p_goal - baseline_p_goal
    
    # Build feature dict
    features = {
        'markov_expected_shots_next_window': round(e_shots, 4),
        'markov_p_shot_next_window_ge1': round(p_shot, 6),
        'markov_expected_corners_next_window': round(e_corners, 4),
        'markov_p_corner_next_window_ge1': round(p_corner, 6),
        'markov_p_goal_next_window': round(p_goal, 6),
        'markov_p_concede_next_window': round(p_concede, 6),
        'delta_shot_vs_minute_baseline': round(delta_shot, 4),
        'delta_corner_vs_minute_baseline': round(delta_corner, 4),
        'delta_goal_vs_minute_baseline': round(delta_goal, 6),
        '_markov_metadata': metadata
    }
    
    return features


def get_markov_features_for_both_teams(
    home_state_t: Dict[str, Any],
    away_state_t: Dict[str, Any],
    event_probs_df: Optional[pd.DataFrame] = None,
    baselines: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute Markov features for both teams in a match.
    
    Note: For most match states, both teams share the same state_t
    (same minute, same score_diff but inverted, etc.).
    
    Args:
        home_state_t: State dict for home team perspective.
        away_state_t: State dict for away team perspective.
        event_probs_df: Pre-loaded event probabilities DataFrame.
        baselines: Pre-loaded baselines dict.
    
    Returns:
        Dict with features for both teams:
        - home_markov_... features
        - away_markov_... features
    """
    home_features = get_markov_features(home_state_t, event_probs_df, baselines)
    away_features = get_markov_features(away_state_t, event_probs_df, baselines)
    
    combined = {}
    for key, value in home_features.items():
        if key != '_markov_metadata':
            combined[f'home_{key}'] = value
    for key, value in away_features.items():
        if key != '_markov_metadata':
            combined[f'away_{key}'] = value
    
    # Include metadata separately
    combined['_home_markov_metadata'] = home_features.get('_markov_metadata', {})
    combined['_away_markov_metadata'] = away_features.get('_markov_metadata', {})
    
    return combined


def build_state_from_match_context(
    minute: int,
    score_diff: int,  # home_score - away_score
    home_red_cards: int = 0,
    away_red_cards: int = 0,
    phase: str = "regular_time",
    strength_gap_bucket: str = "unknown",
    venue_context: str = "neutral",
) -> Dict[str, str]:
    """
    Build a state_t dict from raw match context values.
    
    This helper converts raw match state (minute, score, cards) into
    the bucketized format expected by get_markov_features().
    
    Args:
        minute: Current match minute (0-90+).
        score_diff: Home score minus away score.
        home_red_cards: Number of home team red cards.
        away_red_cards: Number of away team red cards.
        phase: Match phase ("regular_time", "first_half", "second_half", etc.).
        strength_gap_bucket: Team strength gap bucket (usually "unknown").
        venue_context: Venue context ("neutral", "home", "away").
    
    Returns:
        Dict suitable for passing to get_markov_features().
    """
    # Map minute to bucket
    if minute < 16:
        minute_bucket = "0-15"
    elif minute < 31:
        minute_bucket = "16-30"
    elif minute < 46:
        minute_bucket = "31-45+"
    elif minute < 61:
        minute_bucket = "46-60"
    elif minute < 76:
        minute_bucket = "61-75"
    else:
        minute_bucket = "76-90+"
    
    # Map score_diff to bucket
    if score_diff >= 2:
        score_diff_bucket = "+2_or_more"
    elif score_diff == 1:
        score_diff_bucket = "+1"
    elif score_diff == 0:
        score_diff_bucket = "0"
    elif score_diff == -1:
        score_diff_bucket = "-1"
    else:
        score_diff_bucket = "-2_or_more"
    
    return {
        "minute_bucket": minute_bucket,
        "score_diff_bucket": score_diff_bucket,
        "home_red_cards": str(home_red_cards),
        "away_red_cards": str(away_red_cards),
        "phase": phase,
        "strength_gap_bucket": strength_gap_bucket,
        "venue_context": venue_context
    }


def build_state_for_away_team(
    home_state_t: Dict[str, str],
) -> Dict[str, str]:
    """
    Convert a home team state to an away team state.
    
    Mainly inverts the score_diff_bucket sign.
    
    Args:
        home_state_t: State dict from home team perspective.
    
    Returns:
        State dict from away team perspective.
    """
    away_state = home_state_t.copy()
    
    # Invert score_diff_bucket
    score_diff_map = {
        "+2_or_more": "-2_or_more",
        "+1": "-1",
        "0": "0",
        "-1": "+1",
        "-2_or_more": "+2_or_more"
    }
    
    current_score_diff = home_state_t.get('score_diff_bucket', '0')
    away_state['score_diff_bucket'] = score_diff_map.get(current_score_diff, '0')
    
    # Swap red cards perspective
    away_state['home_red_cards'] = home_state_t.get('away_red_cards', '0')
    away_state['away_red_cards'] = home_state_t.get('home_red_cards', '0')
    
    return away_state
