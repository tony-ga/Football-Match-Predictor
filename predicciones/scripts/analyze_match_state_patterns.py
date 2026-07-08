#!/usr/bin/env python
"""
ESPN Match State Pattern Analysis Module - Phase 1 of Markov Modeling.

This module provides tools for detecting and quantifying conditional patterns
in soccer matches that can later be used to build Markov chain models.

Key Features:
- Extract chronological events from ESPN summary/commentary
- Build discrete match states (score diff, time buckets, cards, etc.)
- Create temporal windows anchored to events or fixed intervals
- Quantify pattern transitions (e.g., "trailing by 1 after 60' -> more corners")
- Generate Markov-ready transition datasets
- Load ALL cached matches + optionally fetch missing from API
- Data quality validation and coverage reporting
- Temporal ordering for walk-forward validation

Data Sources:
- commentary: Minute-by-minute event timeline
- keyEvents: Highlight events (goals, cards, subs)
- boxscore: Team statistics (final match stats)
- header/competitors: Score progression by period

Limitations:
- No granular per-minute stats (possession, shots) - only cumulative final stats
- Commentary provides event timestamps but not continuous metrics
- winProbability/predictor nodes rarely available for soccer
"""
from __future__ import annotations

import json
import logging
import os
import re
import csv
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, UTC
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path
import math

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND ENUMS
# =============================================================================

class MinuteBucket(Enum):
    """Time buckets for match state discretization."""
    M0_15 = "0-15"
    M16_30 = "16-30"
    M31_45 = "31-45+"
    M46_60 = "46-60"
    M61_75 = "61-75"
    M76_90 = "76-90+"
    ET1 = "ET1"  # Extra time first half
    ET2 = "ET2"  # Extra time second half
    
    @classmethod
    def from_minute(cls, minute: int, period: int = 1) -> "MinuteBucket":
        """Convert minute + period to bucket."""
        if period == 1:
            if minute <= 15:
                return cls.M0_15
            elif minute <= 30:
                return cls.M16_30
            else:
                return cls.M31_45
        elif period == 2:
            effective_minute = minute  # Already absolute in our processing
            if effective_minute <= 60:
                return cls.M46_60
            elif effective_minute <= 75:
                return cls.M61_75
            else:
                return cls.M76_90
        elif period == 3:  # ET first half
            return cls.ET1
        elif period == 4:  # ET second half
            return cls.ET2
        else:
            return cls.M0_15


class ScoreDiffBucket(Enum):
    """Score difference buckets from team perspective."""
    LOSING_2PLUS = "-2_or_more"  # Losing by 2+ goals
    LOSING_1 = "-1"  # Losing by 1 goal
    TIED = "0"  # Tied
    WINNING_1 = "+1"  # Winning by 1 goal
    WINNING_2PLUS = "+2_or_more"  # Winning by 2+ goals
    
    @classmethod
    def from_diff(cls, diff: int) -> "ScoreDiffBucket":
        """Convert score difference to bucket."""
        if diff <= -2:
            return cls.LOSING_2PLUS
        elif diff == -1:
            return cls.LOSING_1
        elif diff == 0:
            return cls.TIED
        elif diff == 1:
            return cls.WINNING_1
        else:
            return cls.WINNING_2PLUS


class CardBucket(Enum):
    """Red card buckets."""
    NONE = "0"
    ONE_PLUS = "1_plus"
    
    @classmethod
    def from_count(cls, count: int) -> "CardBucket":
        """Convert red card count to bucket."""
        if count >= 1:
            return cls.ONE_PLUS
        return cls.NONE


class Phase(Enum):
    """Match phase."""
    REGULAR = "regular_time"
    STOPPAGE = "stoppage_time"
    EXTRA = "extra_time"
    
    @classmethod
    def from_period(cls, period: int, is_stoppage: bool = False) -> "Phase":
        """Determine phase from period number."""
        if period in (1, 2):
            if is_stoppage:
                return cls.STOPPAGE
            return cls.REGULAR
        elif period in (3, 4):
            return cls.EXTRA
        return cls.REGULAR


@dataclass
class MatchState:
    """
    Discrete state representation of a match at a given moment.
    
    Attributes:
        minute_bucket: Time bucket (e.g., "61-75")
        score_diff_bucket: Score diff from perspective (-2 to +2+)
        home_red_cards: Red cards for home team (0 or 1+)
        away_red_cards: Red cards for away team (0 or 1+)
        phase: Match phase (regular, stoppage, extra)
        strength_gap: Pre-match strength gap proxy (if available)
        venue_context: Home/away/neutral
    """
    minute_bucket: str
    score_diff_bucket: str
    home_red_cards: str
    away_red_cards: str
    phase: str
    strength_gap_bucket: str = "unknown"
    venue_context: str = "neutral"
    
    def to_key(self) -> str:
        """Generate unique state key for transition tracking."""
        return f"{self.minute_bucket}|{self.score_diff_bucket}|{self.home_red_cards}|{self.away_red_cards}|{self.phase}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class StateWindow:
    """
    A window of observation anchored to a state.
    
    Attributes:
        match_id: Event ID
        team_id: Team being analyzed
        team_name: Team name
        state_t: State at start of window
        window_start_minute: Start minute
        window_end_minute: End minute
        events_in_window: List of events in this window
        corners_for: Corners won by team in window
        corners_against: Corners won by opponent in window
        shots_for: Shots by team in window
        shots_against: Shots by opponent in window
        fouls_for: Fouls committed by team in window
        fouls_against: Fouls committed against team in window
        goals_for: Goals scored by team in window
        goals_against: Goals conceded by team in window
        yellow_cards: Yellow cards received in window
        red_cards: Red cards received in window
        next_state_t1: State at end of window
    """
    match_id: str
    team_id: str
    team_name: str
    state_t: Dict[str, Any]
    window_start_minute: int
    window_end_minute: int
    events_in_window: List[Dict[str, Any]] = field(default_factory=list)
    corners_for: int = 0
    corners_against: int = 0
    shots_for: int = 0
    shots_against: int = 0
    shots_on_target_for: int = 0
    shots_on_target_against: int = 0
    fouls_for: int = 0
    fouls_against: int = 0
    goals_for: int = 0
    goals_against: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    dangerous_attacks_for: Optional[int] = None
    possession_shift: Optional[float] = None
    next_state_t1: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "match_id": self.match_id,
            "team_id": self.team_id,
            "team_name": self.team_name,
            "state_t": self.state_t,
            "window_start_minute": self.window_start_minute,
            "window_end_minute": self.window_end_minute,
            "corners_for": self.corners_for,
            "corners_against": self.corners_against,
            "shots_for": self.shots_for,
            "shots_against": self.shots_against,
            "shots_on_target_for": self.shots_on_target_for,
            "shots_on_target_against": self.shots_on_target_against,
            "fouls_for": self.fouls_for,
            "fouls_against": self.fouls_against,
            "goals_for": self.goals_for,
            "goals_against": self.goals_against,
            "yellow_cards": self.yellow_cards,
            "red_cards": self.red_cards,
        }
        if self.dangerous_attacks_for is not None:
            result["dangerous_attacks_for"] = self.dangerous_attacks_for
        if self.possession_shift is not None:
            result["possession_shift"] = self.possession_shift
        if self.next_state_t1:
            result["next_state_t1"] = self.next_state_t1
        return result


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_match_summary(event_id: str, league: str = "fifa.world", 
                       cache_dir: str = "data/cache/espn") -> Optional[Dict[str, Any]]:
    """
    Load match summary from cache or fetch from API.
    
    Args:
        event_id: ESPN event ID
        league: League slug
        cache_dir: Directory for cached summaries
    
    Returns:
        Summary dict or None if not found
    """
    import os
    from urllib.parse import quote
    
    # Try cache first
    cache_filename = f'https:__site.api.espn.com_apis_site_v2_sports_soccer_{league}_summary_{{"event": "{event_id}"}}.json'
    cache_path = os.path.join(cache_dir, cache_filename)
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('response', data)
    
    # Try alternative path structure
    alt_cache_dir = os.path.join("predicciones", cache_dir)
    alt_cache_path = os.path.join(alt_cache_dir, cache_filename)
    
    if os.path.exists(alt_cache_path):
        with open(alt_cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('response', data)
    
    logger.warning(f"No cached summary found for event {event_id}")
    return None


def load_all_summaries_from_cache(cache_dir: str = "data/cache/espn",
                                   league_filter: Optional[str] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Load all match summaries from cache directory.
    
    Args:
        cache_dir: Directory for cached summaries
        league_filter: Optional league to filter by
    
    Returns:
        List of (event_id, league, summary) tuples
    """
    import os
    import re
    
    results = []
    
    # Check both possible cache directories
    dirs_to_check = [cache_dir, os.path.join("predicciones", cache_dir)]
    
    for check_dir in dirs_to_check:
        if not os.path.exists(check_dir):
            continue
            
        for filename in os.listdir(check_dir):
            if not filename.endswith('.json'):
                continue
            
            # Parse filename for event ID and league
            # Format: https:__site.api.espn.com_apis_site_v2_sports_soccer_{league}_summary_{"event": "{event_id}"}.json
            match = re.search(r'soccer_([^_]+)_summary_\{"event": "(\d+)"\}', filename)
            if match:
                league = match.group(1)
                event_id = match.group(2)
                
                if league_filter and league != league_filter:
                    continue
                
                filepath = os.path.join(check_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        summary = data.get('response', data)
                        results.append((event_id, league, summary))
                except Exception as e:
                    logger.debug(f"Error loading {filename}: {e}")
    
    return results


# =============================================================================
# EVENT EXTRACTION AND NORMALIZATION
# =============================================================================

def extract_commentary_events(summary: Dict[str, Any], event_id: str) -> List[Dict[str, Any]]:
    """
    Extract and normalize commentary events from summary.
    
    Args:
        summary: Match summary dict
        event_id: Event ID
    
    Returns:
        List of normalized event dicts
    """
    commentary = summary.get('commentary', [])
    events = []
    
    for idx, item in enumerate(commentary):
        play = item.get('play', item)  # Some items have nested play
        
        # Extract timing
        time_data = item.get('time', {})
        clock_data = play.get('clock', {})
        
        display_value = time_data.get('displayValue', '') or clock_data.get('displayValue', '')
        time_value = time_data.get('value', 0) or clock_data.get('value', 0)
        
        # Calculate minute
        minute = int(time_value // 60) if time_value else 0
        
        # Get period
        period_data = play.get('period', {})
        period = period_data.get('number', 1) if isinstance(period_data, dict) else 1
        
        # Get event type
        type_data = play.get('type', {})
        event_type = type_data.get('type', type_data.get('text', 'unknown')).lower().replace('-', '_')
        
        # Get team
        team_data = play.get('team', {})
        team_name = team_data.get('displayName', '')
        team_id = team_data.get('id', '')
        
        # Get players
        participants = play.get('participants', [])
        player_names = []
        for p in participants:
            athlete = p.get('athlete', p)
            name = athlete.get('displayName', athlete.get('fullName', ''))
            if name:
                player_names.append(name)
        
        # Normalize event type based on text
        text = item.get('text', '').lower()
        if 'goal!' in text or type_data.get('type') == 'goal':
            event_type = 'goal'
        elif 'own goal' in text:
            event_type = 'own_goal'
        elif 'yellow card' in text or type_data.get('type') == 'yellow-card':
            event_type = 'yellow_card'
        elif 'red card' in text or type_data.get('type') == 'red-card':
            event_type = 'red_card'
        elif 'corner' in text and 'win' not in text:
            event_type = 'corner'
        elif 'substitution' in text or type_data.get('type') == 'substitution':
            event_type = 'substitution'
        elif 'offside' in text:
            event_type = 'offside'
        elif 'shot on target' in text or type_data.get('type') == 'shot-on-target':
            event_type = 'shot_on_target'
        elif 'shot off target' in text or type_data.get('type') == 'shot-off-target':
            event_type = 'shot_off_target'
        elif 'attempt saved' in text:
            event_type = 'shot_on_target'
        elif 'attempt missed' in text:
            event_type = 'shot_off_target'
        elif 'attempt blocked' in text:
            event_type = 'shot_blocked'
        elif 'foul' in text and 'wins' not in text:
            event_type = 'foul'
        
        events.append({
            'sequence': idx,
            'minute': minute,
            'clock_display': display_value,
            'period': period,
            'event_type': event_type,
            'team_name': team_name,
            'team_id': team_id,
            'players': player_names,
            'description': item.get('text', ''),
            'raw': item
        })
    
    return events


def extract_key_events(summary: Dict[str, Any], event_id: str) -> List[Dict[str, Any]]:
    """
    Extract and normalize keyEvents from summary.
    
    Args:
        summary: Match summary dict
        event_id: Event ID
    
    Returns:
        List of normalized event dicts
    """
    key_events = summary.get('keyEvents', [])
    events = []
    
    for idx, item in enumerate(key_events):
        # Get timing
        clock_data = item.get('clock', {})
        display_value = clock_data.get('displayValue', '')
        time_value = clock_data.get('value', 0)
        
        minute = int(time_value // 60) if time_value else 0
        
        # Get period
        period_data = item.get('period', {})
        period = period_data.get('number', 1) if isinstance(period_data, dict) else 1
        
        # Get event type
        type_data = item.get('type', {})
        event_type = type_data.get('type', type_data.get('text', 'unknown')).lower().replace('-', '_')
        
        # Get team
        team_data = item.get('team', {})
        team_name = team_data.get('displayName', '')
        team_id = team_data.get('id', '')
        
        # Map event types
        type_mapping = {
            'goal': 'goal',
            'own-goal': 'own_goal',
            'yellow-card': 'yellow_card',
            'red-card': 'red_card',
            'second-yellow-card': 'red_card',
            'substitution': 'substitution',
            'kickoff': 'kickoff',
            'half-time': 'halftime',
            'full-time': 'fulltime',
            'start-of-second-half': 'second_half_start',
        }
        event_type = type_mapping.get(event_type, event_type)
        
        events.append({
            'sequence': idx,
            'minute': minute,
            'clock_display': display_value,
            'period': period,
            'event_type': event_type,
            'team_name': team_name,
            'team_id': team_id,
            'description': item.get('text', ''),
            'is_scoring_play': item.get('scoringPlay', False),
            'raw': item
        })
    
    return events


def extract_boxscore_stats(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract team statistics from boxscore.
    
    Args:
        summary: Match summary dict
    
    Returns:
        Dict mapping team name/hometype to stats
    """
    boxscore = summary.get('boxscore', {})
    teams = boxscore.get('teams', [])
    
    stats_by_team = {}
    
    for team_data in teams:
        team_info = team_data.get('team', {})
        team_name = team_info.get('displayName', team_info.get('name', 'Unknown'))
        home_away = team_data.get('homeAway', 'unknown')
        
        statistics = team_data.get('statistics', [])
        stats_dict = {}
        
        for stat in statistics:
            stat_name = stat.get('name', '')
            stat_value = stat.get('displayValue', stat.get('value', 0))
            
            # Convert to numeric if possible
            try:
                if isinstance(stat_value, str):
                    stat_value = float(stat_value.replace('%', ''))
            except:
                pass
            
            stats_dict[stat_name] = stat_value
        
        stats_by_team[team_name] = {
            'home_away': home_away,
            'stats': stats_dict
        }
    
    return stats_by_team


def extract_score_progression(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract score progression by period from header.
    
    Args:
        summary: Match summary dict
    
    Returns:
        List of score states by period
    """
    header = summary.get('header', {})
    competitions = header.get('competitions', [])
    
    if not competitions:
        return []
    
    comp = competitions[0]
    competitors = comp.get('competitors', [])
    
    home_team = None
    away_team = None
    home_linescores = []
    away_linescores = []
    
    for comp_data in competitors:
        team_name = comp_data.get('team', {}).get('displayName', '')
        home_away = comp_data.get('homeAway', '')
        linescores = comp_data.get('linescores', [])
        
        if home_away == 'home':
            home_team = team_name
            home_linescores = [ls.get('displayValue', '0') for ls in linescores]
        else:
            away_team = team_name
            away_linescores = [ls.get('displayValue', '0') for ls in linescores]
    
    # Build cumulative score by period
    progression = []
    home_cumulative = 0
    away_cumulative = 0
    
    period_names = ['1st Half', '2nd Half', 'ET1', 'ET2', 'Penalties']
    
    for i in range(max(len(home_linescores), len(away_linescores))):
        try:
            home_period = int(home_linescores[i]) if i < len(home_linescores) else 0
        except:
            home_period = 0
        
        try:
            away_period = int(away_linescores[i]) if i < len(away_linescores) else 0
        except:
            away_period = 0
        
        home_cumulative += home_period
        away_cumulative += away_period
        
        progression.append({
            'period': i + 1,
            'period_name': period_names[i] if i < len(period_names) else f'Period {i+1}',
            'home_period_score': home_period,
            'away_period_score': away_period,
            'home_cumulative': home_cumulative,
            'away_cumulative': away_cumulative,
            'score_diff': home_cumulative - away_cumulative
        })
    
    return progression


# =============================================================================
# STATE CONSTRUCTION
# =============================================================================

def get_match_context(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract basic match context.
    
    Returns:
        Dict with event_id, teams, date, status, etc.
    """
    header = summary.get('header', {})
    competitions = header.get('competitions', [])
    
    if not competitions:
        return {}
    
    comp = competitions[0]
    competitors = comp.get('competitors', [])
    
    home_team = None
    away_team = None
    home_id = None
    away_id = None
    
    for comp_data in competitors:
        team_info = comp_data.get('team', {})
        if comp_data.get('homeAway') == 'home':
            home_team = team_info.get('displayName', '')
            home_id = team_info.get('id', '')
        else:
            away_team = team_info.get('displayName', '')
            away_id = team_info.get('id', '')
    
    status = comp.get('status', {})
    status_text = status.get('type', {}).get('description', 'Unknown')
    
    return {
        'event_id': header.get('id', ''),
        'date': header.get('date', ''),
        'home_team': home_team,
        'away_team': away_team,
        'home_team_id': home_id,
        'away_team_id': away_id,
        'status': status_text,
        'neutral_site': comp.get('neutralSite', False)
    }


def build_state_at_minute(
    minute: int,
    period: int,
    home_score: int,
    away_score: int,
    home_red_cards: int,
    away_red_cards: int,
    home_team: str,
    away_team: str,
    is_stoppage: bool = False,
    strength_gap: Optional[float] = None
) -> MatchState:
    """
    Build a MatchState object for a specific minute.
    
    Args:
        minute: Current minute (absolute, accounting for period)
        period: Period number (1, 2, 3, 4)
        home_score: Home team score
        away_score: Away team score
        home_red_cards: Home team red cards
        away_red_cards: Away team red cards
        home_team: Home team name (for perspective)
        away_team: Away team name
        is_stoppage: Whether in stoppage time
        strength_gap: Pre-match strength gap (optional)
    
    Returns:
        MatchState object
    """
    # Determine minute bucket
    minute_bucket = MinuteBucket.from_minute(minute, period)
    
    # Score diff from home team perspective
    score_diff = home_score - away_score
    score_diff_bucket = ScoreDiffBucket.from_diff(score_diff)
    
    # Card buckets
    home_card_bucket = CardBucket.from_count(home_red_cards)
    away_card_bucket = CardBucket.from_count(away_red_cards)
    
    # Phase
    phase = Phase.from_period(period, is_stoppage)
    
    # Strength gap bucket
    if strength_gap is not None:
        if strength_gap > 10:
            strength_gap_bucket = "home_heavy_favorite"
        elif strength_gap > 5:
            strength_gap_bucket = "home_favorite"
        elif strength_gap < -10:
            strength_gap_bucket = "away_heavy_favorite"
        elif strength_gap < -5:
            strength_gap_bucket = "away_favorite"
        else:
            strength_gap_bucket = "even"
    else:
        strength_gap_bucket = "unknown"
    
    # Venue context
    venue_context = "neutral"  # Default for international tournaments
    
    return MatchState(
        minute_bucket=minute_bucket.value,
        score_diff_bucket=score_diff_bucket.value,
        home_red_cards=home_card_bucket.value,
        away_red_cards=away_card_bucket.value,
        phase=phase.value,
        strength_gap_bucket=strength_gap_bucket,
        venue_context=venue_context
    )


def compute_cumulative_stats_up_to_minute(
    events: List[Dict[str, Any]],
    minute: int,
    team_name: str,
    opponent_name: str
) -> Dict[str, int]:
    """
    Compute cumulative statistics up to a given minute.
    
    Args:
        events: List of normalized events
        minute: Cut-off minute
        team_name: Team to compute stats for
        opponent_name: Opponent team name
    
    Returns:
        Dict with cumulative stats
    """
    stats = {
        'corners_for': 0,
        'corners_against': 0,
        'shots_for': 0,
        'shots_against': 0,
        'shots_on_target_for': 0,
        'shots_on_target_against': 0,
        'fouls_for': 0,
        'fouls_against': 0,
        'goals_for': 0,
        'goals_against': 0,
        'yellow_cards': 0,
        'red_cards': 0
    }
    
    for event in events:
        if event['minute'] > minute:
            continue
        
        event_team = event.get('team_name', '')
        event_type = event.get('event_type', '')
        
        # Count events for team
        if event_team == team_name:
            if event_type == 'corner':
                stats['corners_for'] += 1
            elif event_type in ('shot_on_target', 'shot_off_target', 'shot_blocked'):
                stats['shots_for'] += 1
                if event_type == 'shot_on_target':
                    stats['shots_on_target_for'] += 1
            elif event_type == 'foul':
                stats['fouls_for'] += 1
            elif event_type == 'goal':
                stats['goals_for'] += 1
            elif event_type == 'yellow_card':
                stats['yellow_cards'] += 1
            elif event_type == 'red_card':
                stats['red_cards'] += 1
        
        # Count events against team (opponent actions)
        if event_team == opponent_name:
            if event_type == 'corner':
                stats['corners_against'] += 1
            elif event_type in ('shot_on_target', 'shot_off_target', 'shot_blocked'):
                stats['shots_against'] += 1
                if event_type == 'shot_on_target':
                    stats['shots_on_target_against'] += 1
            elif event_type == 'foul':
                stats['fouls_against'] += 1
            elif event_type == 'goal':
                stats['goals_against'] += 1
    
    return stats


def compute_score_at_minute(
    events: List[Dict[str, Any]],
    minute: int,
    home_team: str,
    away_team: str
) -> Tuple[int, int]:
    """
    Compute score at a given minute based on goals in events.
    
    Args:
        events: List of normalized events
        minute: Cut-off minute
        home_team: Home team name
        away_team: Away team name
    
    Returns:
        Tuple of (home_score, away_score)
    """
    home_score = 0
    away_score = 0
    
    for event in events:
        if event['minute'] > minute:
            continue
        
        if event.get('event_type') not in ('goal', 'own_goal'):
            continue
        
        event_team = event.get('team_name', '')
        
        # Check for own goal
        description = event.get('description', '').lower()
        is_own_goal = 'own goal' in description or event.get('event_type') == 'own_goal'
        
        if is_own_goal:
            # Own goal counts for the opponent
            if event_team == home_team:
                away_score += 1
            else:
                home_score += 1
        else:
            if event_team == home_team:
                home_score += 1
            elif event_team == away_team:
                away_score += 1
    
    return home_score, away_score


def compute_red_cards_at_minute(
    events: List[Dict[str, Any]],
    minute: int,
    team_name: str
) -> int:
    """
    Compute red cards for a team up to a given minute.
    
    Args:
        events: List of normalized events
        minute: Cut-off minute
        team_name: Team name
    
    Returns:
        Red card count
    """
    count = 0
    
    for event in events:
        if event['minute'] > minute:
            continue
        
        if event.get('event_type') == 'red_card' and event.get('team_name') == team_name:
            count += 1
    
    return count


# =============================================================================
# WINDOW GENERATION
# =============================================================================

def generate_fixed_windows(
    events: List[Dict[str, Any]],
    match_context: Dict[str, Any],
    window_size: int = 15
) -> List[StateWindow]:
    """
    Generate fixed-size temporal windows for analysis.
    
    Args:
        events: List of normalized events
        match_context: Match context dict
        window_size: Window size in minutes
    
    Returns:
        List of StateWindow objects
    """
    home_team = match_context.get('home_team', '')
    away_team = match_context.get('away_team', '')
    home_id = match_context.get('home_team_id', '')
    away_id = match_context.get('away_team_id', '')
    event_id = match_context.get('event_id', '')
    
    windows = []
    
    # Generate windows for each team
    for team_name, team_id, opponent_name in [
        (home_team, home_id, away_team),
        (away_team, away_id, home_team)
    ]:
        if not team_name:
            continue
        
        # Create windows at regular intervals
        start_minute = 0
        while start_minute < 90:
            end_minute = min(start_minute + window_size, 90)
            
            # Get state at start of window
            home_score, away_score = compute_score_at_minute(events, start_minute, home_team, away_team)
            home_reds = compute_red_cards_at_minute(events, start_minute, home_team)
            away_reds = compute_red_cards_at_minute(events, start_minute, away_team)
            
            # Determine period based on minute
            period = 1 if start_minute < 45 else 2
            
            state = build_state_at_minute(
                minute=start_minute,
                period=period,
                home_score=home_score,
                away_score=away_score,
                home_red_cards=home_reds,
                away_red_cards=away_reds,
                home_team=home_team,
                away_team=away_team
            )
            
            # Get events in window
            window_events = [e for e in events if start_minute < e['minute'] <= end_minute]
            
            # Compute stats in window
            stats = {'corners_for': 0, 'corners_against': 0, 'shots_for': 0, 
                     'shots_against': 0, 'shots_on_target_for': 0, 'shots_on_target_against': 0,
                     'fouls_for': 0, 'fouls_against': 0, 'goals_for': 0, 'goals_against': 0,
                     'yellow_cards': 0, 'red_cards': 0}
            
            for event in window_events:
                event_team = event.get('team_name', '')
                event_type = event.get('event_type', '')
                
                if event_team == team_name:
                    if event_type == 'corner':
                        stats['corners_for'] += 1
                    elif event_type in ('shot_on_target', 'shot_off_target', 'shot_blocked'):
                        stats['shots_for'] += 1
                        if event_type == 'shot_on_target':
                            stats['shots_on_target_for'] += 1
                    elif event_type == 'foul':
                        stats['fouls_for'] += 1
                    elif event_type == 'goal':
                        stats['goals_for'] += 1
                    elif event_type == 'yellow_card':
                        stats['yellow_cards'] += 1
                    elif event_type == 'red_card':
                        stats['red_cards'] += 1
                
                if event_team == opponent_name:
                    if event_type == 'corner':
                        stats['corners_against'] += 1
                    elif event_type in ('shot_on_target', 'shot_off_target', 'shot_blocked'):
                        stats['shots_against'] += 1
                        if event_type == 'shot_on_target':
                            stats['shots_on_target_against'] += 1
                    elif event_type == 'foul':
                        stats['fouls_against'] += 1
                    elif event_type == 'goal':
                        stats['goals_against'] += 1
            
            # Get state at end of window
            end_home_score, end_away_score = compute_score_at_minute(events, end_minute, home_team, away_team)
            end_home_reds = compute_red_cards_at_minute(events, end_minute, home_team)
            end_away_reds = compute_red_cards_at_minute(events, end_minute, away_team)
            end_period = 1 if end_minute < 45 else 2
            
            end_state = build_state_at_minute(
                minute=end_minute,
                period=end_period,
                home_score=end_home_score,
                away_score=end_away_score,
                home_red_cards=end_home_reds,
                away_red_cards=end_away_reds,
                home_team=home_team,
                away_team=away_team
            )
            
            window = StateWindow(
                match_id=event_id,
                team_id=team_id,
                team_name=team_name,
                state_t=state.to_dict(),
                window_start_minute=start_minute,
                window_end_minute=end_minute,
                events_in_window=window_events,
                corners_for=stats['corners_for'],
                corners_against=stats['corners_against'],
                shots_for=stats['shots_for'],
                shots_against=stats['shots_against'],
                shots_on_target_for=stats['shots_on_target_for'],
                shots_on_target_against=stats['shots_on_target_against'],
                fouls_for=stats['fouls_for'],
                fouls_against=stats['fouls_against'],
                goals_for=stats['goals_for'],
                goals_against=stats['goals_against'],
                yellow_cards=stats['yellow_cards'],
                red_cards=stats['red_cards'],
                next_state_t1=end_state.to_dict()
            )
            
            windows.append(window)
            start_minute += window_size
    
    return windows


def generate_event_anchored_windows(
    events: List[Dict[str, Any]],
    match_context: Dict[str, Any],
    lookforward_minutes: int = 15
) -> List[StateWindow]:
    """
    Generate windows anchored to key events (goals, cards, etc.).
    
    Args:
        events: List of normalized events
        match_context: Match context dict
        lookforward_minutes: Minutes to look forward from anchor
    
    Returns:
        List of StateWindow objects
    """
    home_team = match_context.get('home_team', '')
    away_team = match_context.get('away_team', '')
    home_id = match_context.get('home_team_id', '')
    away_id = match_context.get('away_team_id', '')
    event_id = match_context.get('event_id', '')
    
    windows = []
    
    # Anchor event types to track
    anchor_types = ['goal', 'red_card', 'yellow_card']
    
    # Find anchor events
    anchors = [e for e in events if e.get('event_type') in anchor_types]
    
    for anchor in anchors:
        anchor_minute = anchor['minute']
        anchor_team = anchor.get('team_name', '')
        anchor_type = anchor.get('event_type', '')
        
        if not anchor_team:
            continue
        
        # Determine opponent
        if anchor_team == home_team:
            opponent = away_team
            team_id = home_id
        else:
            opponent = home_team
            team_id = away_id
        
        # Get state at anchor minute
        home_score, away_score = compute_score_at_minute(events, anchor_minute, home_team, away_team)
        home_reds = compute_red_cards_at_minute(events, anchor_minute, home_team)
        away_reds = compute_red_cards_at_minute(events, anchor_minute, away_team)
        
        period = 1 if anchor_minute < 45 else 2
        
        state = build_state_at_minute(
            minute=anchor_minute,
            period=period,
            home_score=home_score,
            away_score=away_score,
            home_red_cards=home_reds,
            away_red_cards=away_reds,
            home_team=home_team,
            away_team=away_team
        )
        
        # Define window end
        end_minute = min(anchor_minute + lookforward_minutes, 90)
        
        # Get events in window
        window_events = [e for e in events if anchor_minute < e['minute'] <= end_minute]
        
        # Compute stats in window (same logic as fixed windows)
        stats = {'corners_for': 0, 'corners_against': 0, 'shots_for': 0, 
                 'shots_against': 0, 'shots_on_target_for': 0, 'shots_on_target_against': 0,
                 'fouls_for': 0, 'fouls_against': 0, 'goals_for': 0, 'goals_against': 0,
                 'yellow_cards': 0, 'red_cards': 0}
        
        for event in window_events:
            event_team = event.get('team_name', '')
            event_type = event.get('event_type', '')
            
            if event_team == anchor_team:
                if event_type == 'corner':
                    stats['corners_for'] += 1
                elif event_type in ('shot_on_target', 'shot_off_target', 'shot_blocked'):
                    stats['shots_for'] += 1
                    if event_type == 'shot_on_target':
                        stats['shots_on_target_for'] += 1
                elif event_type == 'foul':
                    stats['fouls_for'] += 1
                elif event_type == 'goal':
                    stats['goals_for'] += 1
                elif event_type == 'yellow_card':
                    stats['yellow_cards'] += 1
                elif event_type == 'red_card':
                    stats['red_cards'] += 1
            
            if event_team == opponent:
                if event_type == 'corner':
                    stats['corners_against'] += 1
                elif event_type in ('shot_on_target', 'shot_off_target', 'shot_blocked'):
                    stats['shots_against'] += 1
                    if event_type == 'shot_on_target':
                        stats['shots_on_target_against'] += 1
                elif event_type == 'foul':
                    stats['fouls_against'] += 1
                elif event_type == 'goal':
                    stats['goals_against'] += 1
        
        # Get state at end
        end_home_score, end_away_score = compute_score_at_minute(events, end_minute, home_team, away_team)
        end_home_reds = compute_red_cards_at_minute(events, end_minute, home_team)
        end_away_reds = compute_red_cards_at_minute(events, end_minute, away_team)
        end_period = 1 if end_minute < 45 else 2
        
        end_state = build_state_at_minute(
            minute=end_minute,
            period=end_period,
            home_score=end_home_score,
            away_score=end_away_score,
            home_red_cards=end_home_reds,
            away_red_cards=end_away_reds,
            home_team=home_team,
            away_team=away_team
        )
        
        window = StateWindow(
            match_id=event_id,
            team_id=team_id,
            team_name=anchor_team,
            state_t=state.to_dict(),
            window_start_minute=anchor_minute,
            window_end_minute=end_minute,
            events_in_window=window_events,
            corners_for=stats['corners_for'],
            corners_against=stats['corners_against'],
            shots_for=stats['shots_for'],
            shots_against=stats['shots_against'],
            shots_on_target_for=stats['shots_on_target_for'],
            shots_on_target_against=stats['shots_on_target_against'],
            fouls_for=stats['fouls_for'],
            fouls_against=stats['fouls_against'],
            goals_for=stats['goals_for'],
            goals_against=stats['goals_against'],
            yellow_cards=stats['yellow_cards'],
            red_cards=stats['red_cards'],
            next_state_t1=end_state.to_dict()
        )
        
        # Add metadata about anchor
        window.events_in_window.insert(0, {**anchor, '_is_anchor': True})
        
        windows.append(window)
    
    return windows


# =============================================================================
# PATTERN DETECTION
# =============================================================================

@dataclass
class PatternResult:
    """Result of pattern analysis."""
    pattern_name: str
    condition_description: str
    sample_size: int
    mean: float
    median: float
    std_dev: float
    baseline_mean: float
    lift: float  # (mean - baseline) / baseline
    confidence_interval_95: Optional[Tuple[float, float]] = None
    home_split: Optional[Dict[str, float]] = None
    away_split: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'pattern_name': self.pattern_name,
            'condition_description': self.condition_description,
            'sample_size': self.sample_size,
            'mean': round(self.mean, 3),
            'median': round(self.median, 3),
            'std_dev': round(self.std_dev, 3),
            'baseline_mean': round(self.baseline_mean, 3),
            'lift': round(self.lift, 3),
        }
        if self.confidence_interval_95:
            result['confidence_interval_95'] = [round(x, 3) for x in self.confidence_interval_95]
        if self.home_split:
            result['home_split'] = {k: round(v, 3) for k, v in self.home_split.items()}
        if self.away_split:
            result['away_split'] = {k: round(v, 3) for k, v in self.away_split.items()}
        return result


def compute_pattern_statistics(
    windows: List[StateWindow],
    condition_fn,
    metric_fn,
    pattern_name: str,
    condition_description: str
) -> PatternResult:
    """
    Compute statistics for a pattern.
    
    Args:
        windows: List of StateWindow objects
        condition_fn: Function that returns True if window matches condition
        metric_fn: Function that extracts metric value from window
        pattern_name: Name for the pattern
        condition_description: Human-readable description
    
    Returns:
        PatternResult object
    """
    import statistics
    
    # Filter windows matching condition
    matching = [w for w in windows if condition_fn(w)]
    
    if len(matching) < 2:
        return PatternResult(
            pattern_name=pattern_name,
            condition_description=condition_description,
            sample_size=len(matching),
            mean=0.0,
            median=0.0,
            std_dev=0.0,
            baseline_mean=0.0,
            lift=0.0
        )
    
    # Extract metric values
    values = [metric_fn(w) for w in matching]
    values = [v for v in values if v is not None]
    
    if not values:
        return PatternResult(
            pattern_name=pattern_name,
            condition_description=condition_description,
            sample_size=len(matching),
            mean=0.0,
            median=0.0,
            std_dev=0.0,
            baseline_mean=0.0,
            lift=0.0
        )
    
    mean_val = statistics.mean(values)
    median_val = statistics.median(values)
    std_val = statistics.stdev(values) if len(values) > 1 else 0.0
    
    # Compute baseline (all windows)
    all_values = [metric_fn(w) for w in windows]
    all_values = [v for v in all_values if v is not None]
    baseline_mean = statistics.mean(all_values) if all_values else 0.0
    
    # Compute lift
    lift = (mean_val - baseline_mean) / baseline_mean if baseline_mean > 0 else 0.0
    
    # Compute 95% CI (simple approximation)
    n = len(values)
    if n >= 30:
        margin = 1.96 * std_val / math.sqrt(n)
        ci = (mean_val - margin, mean_val + margin)
    else:
        ci = None
    
    # Home/away splits
    home_vals = [metric_fn(w) for w in matching if w.state_t.get('venue_context') == 'home']
    home_vals = [v for v in home_vals if v is not None]
    away_vals = [metric_fn(w) for w in matching if w.state_t.get('venue_context') == 'away']
    away_vals = [v for v in away_vals if v is not None]
    
    home_split = {'mean': statistics.mean(home_vals)} if home_vals else None
    away_split = {'mean': statistics.mean(away_vals)} if away_vals else None
    
    return PatternResult(
        pattern_name=pattern_name,
        condition_description=condition_description,
        sample_size=len(matching),
        mean=mean_val,
        median=median_val,
        std_dev=std_val,
        baseline_mean=baseline_mean,
        lift=lift,
        confidence_interval_95=ci,
        home_split=home_split,
        away_split=away_split
    )


def detect_common_patterns(windows: List[StateWindow]) -> List[PatternResult]:
    """
    Detect common conditional patterns in match data.
    
    Patterns include:
    - Trailing by 1 after 60' -> corners/shots in next 15'
    - Leading by 1 after 60' -> defensive metrics
    - Red card down -> concession risk
    - Goal just scored -> opponent pressure
    - Tied at halftime -> second half intensity
    
    Args:
        windows: List of StateWindow objects
    
    Returns:
        List of PatternResult objects
    """
    patterns = []
    
    # Helper conditions
    def trailing_by_1_after_60(w: StateWindow) -> bool:
        return (w.window_start_minute >= 60 and 
                w.state_t.get('score_diff_bucket') in ('-1', '-2_or_more'))
    
    def leading_by_1_after_60(w: StateWindow) -> bool:
        return (w.window_start_minute >= 60 and 
                w.state_t.get('score_diff_bucket') in ('+1', '+2_or_more'))
    
    def tied_after_60(w: StateWindow) -> bool:
        return w.window_start_minute >= 60 and w.state_t.get('score_diff_bucket') == '0'
    
    def team_down_red_card(w: StateWindow) -> bool:
        # From team perspective, they have a red card
        return w.state_t.get('home_red_cards') == '1_plus' or w.state_t.get('away_red_cards') == '1_plus'
    
    def just_conceded_goal(w: StateWindow) -> bool:
        # Window starts right after conceding
        return w.goals_against > 0 and w.window_start_minute > 0
    
    def just_scored_goal(w: StateWindow) -> bool:
        return w.goals_for > 0 and w.window_start_minute > 0
    
    # Pattern 1: Trailing by 1 after 60' -> corners in next 15'
    patterns.append(compute_pattern_statistics(
        windows,
        trailing_by_1_after_60,
        lambda w: w.corners_for,
        "trailing_by_1_after_60_corners",
        "Team losing by 1+ goal after 60' -> corners in next 15'"
    ))
    
    # Pattern 2: Trailing by 1 after 60' -> shots in next 15'
    patterns.append(compute_pattern_statistics(
        windows,
        trailing_by_1_after_60,
        lambda w: w.shots_for,
        "trailing_by_1_after_60_shots",
        "Team losing by 1+ goal after 60' -> shots in next 15'"
    ))
    
    # Pattern 3: Leading by 1 after 60' -> shots against in next 15'
    patterns.append(compute_pattern_statistics(
        windows,
        leading_by_1_after_60,
        lambda w: w.shots_against,
        "leading_by_1_after_60_shots_against",
        "Team winning by 1+ goal after 60' -> opponent shots in next 15'"
    ))
    
    # Pattern 4: Tied after 60' -> total shots in next 15'
    patterns.append(compute_pattern_statistics(
        windows,
        tied_after_60,
        lambda w: w.shots_for + w.shots_against,
        "tied_after_60_total_shots",
        "Tied game after 60' -> total shots in next 15'"
    ))
    
    # Pattern 5: Red card down -> goals against in next 15'
    patterns.append(compute_pattern_statistics(
        windows,
        team_down_red_card,
        lambda w: w.goals_against,
        "red_card_down_concession",
        "Team with red card -> goals conceded in next 15'"
    ))
    
    # Pattern 6: Just conceded -> corners for in next 15'
    patterns.append(compute_pattern_statistics(
        windows,
        just_conceded_goal,
        lambda w: w.corners_for,
        "just_conceded_corners_response",
        "Team just conceded -> corners in next 15'"
    ))
    
    # Pattern 7: Just scored -> opponent shots in next 15'
    patterns.append(compute_pattern_statistics(
        windows,
        just_scored_goal,
        lambda w: w.shots_against,
        "just_scored_opponent_response",
        "Team just scored -> opponent shots in next 15'"
    ))
    
    # Pattern 8: Trailing -> fouls committed (desperation)
    patterns.append(compute_pattern_statistics(
        windows,
        trailing_by_1_after_60,
        lambda w: w.fouls_for,
        "trailing_fouls_desperation",
        "Team losing by 1+ after 60' -> fouls in next 15'"
    ))
    
    return patterns


# =============================================================================
# MARKOV-READY DATASET GENERATION
# =============================================================================

def generate_markov_ready_dataset(
    windows: List[StateWindow]
) -> List[Dict[str, Any]]:
    """
    Generate Markov-ready transition dataset.
    
    Each row represents a state transition with associated outcomes.
    
    Args:
        windows: List of StateWindow objects
    
    Returns:
        List of dicts ready for CSV export
    """
    rows = []
    
    for w in windows:
        row = {
            'match_id': w.match_id,
            'team_id': w.team_id,
            'team_name': w.team_name,
            'state_t': json.dumps(w.state_t),
            'state_t_minute_bucket': w.state_t.get('minute_bucket', ''),
            'state_t_score_diff': w.state_t.get('score_diff_bucket', ''),
            'state_t_home_reds': w.state_t.get('home_red_cards', ''),
            'state_t_away_reds': w.state_t.get('away_red_cards', ''),
            'state_t_phase': w.state_t.get('phase', ''),
            'window_start': w.window_start_minute,
            'window_end': w.window_end_minute,
            'corners_next_window': w.corners_for,
            'shots_next_window': w.shots_for,
            'shots_on_target_next_window': w.shots_on_target_for,
            'fouls_next_window': w.fouls_for,
            'goals_next_window': w.goals_for,
            'concede_next_window': w.goals_against,
            'yellow_cards_next_window': w.yellow_cards,
            'red_cards_next_window': w.red_cards,
        }
        
        if w.next_state_t1:
            row['next_state_t1'] = json.dumps(w.next_state_t1)
            row['next_state_score_diff'] = w.next_state_t1.get('score_diff_bucket', '')
            row['next_state_minute_bucket'] = w.next_state_t1.get('minute_bucket', '')
        
        rows.append(row)
    
    return rows


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_windows_to_csv(
    windows: List[StateWindow],
    output_path: str
) -> None:
    """
    Export windows to CSV file.
    
    Args:
        windows: List of StateWindow objects
        output_path: Output file path
    """
    import csv
    
    if not windows:
        logger.warning("No windows to export")
        return
    
    # Flatten to dicts
    rows = [w.to_dict() for w in windows]
    
    # Get all keys
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    
    fieldnames = sorted(list(all_keys))
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Serialize nested dicts
            clean_row = {}
            for k, v in row.items():
                if isinstance(v, dict):
                    clean_row[k] = json.dumps(v)
                elif isinstance(v, list) and k == 'events_in_window':
                    clean_row[k] = len(v)  # Just count
                else:
                    clean_row[k] = v
            writer.writerow(clean_row)
    
    logger.info(f"Exported {len(windows)} windows to {output_path}")


def export_markov_transitions_to_csv(
    transitions: List[Dict[str, Any]],
    output_path: str
) -> None:
    """
    Export Markov transitions to CSV.
    
    Args:
        transitions: List of transition dicts
        output_path: Output file path
    """
    import csv
    
    if not transitions:
        logger.warning("No transitions to export")
        return
    
    # Get all keys
    all_keys = set()
    for row in transitions:
        all_keys.update(row.keys())
    
    fieldnames = sorted(list(all_keys))
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in transitions:
            writer.writerow(row)
    
    logger.info(f"Exported {len(transitions)} transitions to {output_path}")


def export_patterns_to_csv(
    patterns: List[PatternResult],
    output_path: str
) -> None:
    """
    Export pattern analysis to CSV.
    
    Args:
        patterns: List of PatternResult objects
        output_path: Output file path
    """
    import csv
    
    if not patterns:
        logger.warning("No patterns to export")
        return
    
    fieldnames = ['pattern_name', 'condition_description', 'sample_size', 
                  'mean', 'median', 'std_dev', 'baseline_mean', 'lift',
                  'ci_lower', 'ci_upper', 'home_mean', 'away_mean']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for p in patterns:
            row = {
                'pattern_name': p.pattern_name,
                'condition_description': p.condition_description,
                'sample_size': p.sample_size,
                'mean': round(p.mean, 3),
                'median': round(p.median, 3),
                'std_dev': round(p.std_dev, 3),
                'baseline_mean': round(p.baseline_mean, 3),
                'lift': round(p.lift, 3),
                'ci_lower': round(p.confidence_interval_95[0], 3) if p.confidence_interval_95 else None,
                'ci_upper': round(p.confidence_interval_95[1], 3) if p.confidence_interval_95 else None,
                'home_mean': round(p.home_split['mean'], 3) if p.home_split else None,
                'away_mean': round(p.away_split['mean'], 3) if p.away_split else None,
            }
            writer.writerow(row)
    
    logger.info(f"Exported {len(patterns)} patterns to {output_path}")


def export_patterns_report(
    patterns: List[PatternResult],
    output_path: str
) -> None:
    """
    Export human-readable patterns report.
    
    Args:
        patterns: List of PatternResult objects
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Match State Pattern Analysis Report\n\n")
        f.write("## Detected Patterns\n\n")
        
        # Sort by sample size
        sorted_patterns = sorted(patterns, key=lambda p: p.sample_size, reverse=True)
        
        for p in sorted_patterns:
            f.write(f"### {p.pattern_name}\n\n")
            f.write(f"**Condition**: {p.condition_description}\n\n")
            f.write(f"- Sample Size: {p.sample_size}\n")
            f.write(f"- Mean: {p.mean:.3f}\n")
            f.write(f"- Median: {p.median:.3f}\n")
            f.write(f"- Std Dev: {p.std_dev:.3f}\n")
            f.write(f"- Baseline Mean: {p.baseline_mean:.3f}\n")
            f.write(f"- Lift vs Baseline: {p.lift:+.1%}\n")
            
            if p.confidence_interval_95:
                f.write(f"- 95% CI: [{p.confidence_interval_95[0]:.3f}, {p.confidence_interval_95[1]:.3f}]\n")
            
            if p.home_split or p.away_split:
                f.write("\n**Splits**:\n")
                if p.home_split:
                    f.write(f"- Home Mean: {p.home_split['mean']:.3f}\n")
                if p.away_split:
                    f.write(f"- Away Mean: {p.away_split['mean']:.3f}\n")
            
            # Quality assessment
            f.write("\n**Quality Assessment**:\n")
            if p.sample_size < 10:
                f.write("- ⚠️ Small sample size - interpret with caution\n")
            elif p.sample_size < 30:
                f.write("- ⚡ Moderate sample size\n")
            else:
                f.write("- ✅ Good sample size\n")
            
            if abs(p.lift) < 0.1:
                f.write("- ℹ️ Minimal lift vs baseline\n")
            elif abs(p.lift) < 0.3:
                f.write("- 📈 Moderate lift vs baseline\n")
            else:
                f.write("- 🚀 Strong lift vs baseline\n")
            
            f.write("\n---\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        robust_patterns = [p for p in patterns if p.sample_size >= 30 and abs(p.lift) >= 0.2]
        f.write(f"**Robust patterns detected**: {len(robust_patterns)}\n\n")
        
        if robust_patterns:
            f.write("Most promising patterns for modeling:\n")
            for p in sorted(robust_patterns, key=lambda x: abs(x.lift), reverse=True)[:5]:
                f.write(f"- {p.pattern_name} (lift: {p.lift:+.1%}, n={p.sample_size})\n")


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def run_match_state_analysis(
    league_filter: Optional[str] = None,
    days_back: int = 365,
    output_dir: str = "data/derived",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete match state pattern analysis pipeline.
    
    Args:
        league_filter: Optional league to filter by
        days_back: Days back to consider (not enforced without dates)
        output_dir: Output directory for derived datasets
        verbose: Print progress
    
    Returns:
        Summary dict with counts and paths
    """
    import os
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all summaries
    if verbose:
        print("Loading match summaries from cache...")
    
    summaries = load_all_summaries_from_cache(league_filter=league_filter)
    
    if not summaries:
        print("No summaries found in cache.")
        return {'error': 'No summaries found'}
    
    if verbose:
        print(f"Loaded {len(summaries)} match summaries")
    
    # Process each match
    all_windows = []
    all_transitions = []
    
    for event_id, league, summary in summaries:
        if verbose:
            print(f"Processing match {event_id} ({league})...")
        
        # Get match context
        context = get_match_context(summary)
        if not context:
            continue
        
        # Extract events
        events = extract_commentary_events(summary, event_id)
        
        if len(events) < 5:
            if verbose:
                print(f"  Skipping - too few events ({len(events)})")
            continue
        
        # Generate fixed windows
        fixed_windows = generate_fixed_windows(events, context, window_size=15)
        all_windows.extend(fixed_windows)
        
        # Generate event-anchored windows
        anchored_windows = generate_event_anchored_windows(events, context, lookforward_minutes=15)
        all_windows.extend(anchored_windows)
    
    if verbose:
        print(f"\nGenerated {len(all_windows)} total windows")
    
    if not all_windows:
        return {'error': 'No windows generated'}
    
    # Detect patterns
    if verbose:
        print("Detecting patterns...")
    
    patterns = detect_common_patterns(all_windows)
    
    # Generate Markov-ready dataset
    transitions = generate_markov_ready_dataset(all_windows)
    
    # Export datasets
    windows_path = os.path.join(output_dir, 'match_state_windows.csv')
    patterns_path = os.path.join(output_dir, 'state_pattern_summary.csv')
    transitions_path = os.path.join(output_dir, 'markov_ready_transitions.csv')
    report_path = os.path.join(output_dir, 'pattern_analysis_report.md')
    
    export_windows_to_csv(all_windows, windows_path)
    export_patterns_to_csv(patterns, patterns_path)
    export_markov_transitions_to_csv(transitions, transitions_path)
    export_patterns_report(patterns, report_path)
    
    if verbose:
        print(f"\nExported:")
        print(f"  - Windows: {windows_path}")
        print(f"  - Patterns: {patterns_path}")
        print(f"  - Transitions: {transitions_path}")
        print(f"  - Report: {report_path}")
    
    return {
        'matches_processed': len(summaries),
        'windows_generated': len(all_windows),
        'patterns_detected': len(patterns),
        'transitions_count': len(transitions),
        'output_files': {
            'windows': windows_path,
            'patterns': patterns_path,
            'transitions': transitions_path,
            'report': report_path
        }
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze match state patterns for Markov modeling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze_match_state_patterns.py --league fifa.world --days-back 365
  python scripts/analyze_match_state_patterns.py --verbose
  python scripts/analyze_match_state_patterns.py --output-dir output/analysis
        """
    )
    
    parser.add_argument('--league', type=str, default=None,
                        help='Filter by league (e.g., fifa.world, eng.1)')
    parser.add_argument('--days-back', type=int, default=365,
                        help='Days back to consider (default: 365)')
    parser.add_argument('--output-dir', type=str, default='data/derived',
                        help='Output directory for derived datasets')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress information')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    result = run_match_state_analysis(
        league_filter=args.league,
        days_back=args.days_back,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    print("\n=== Analysis Complete ===")
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Matches processed: {result['matches_processed']}")
        print(f"Windows generated: {result['windows_generated']}")
        print(f"Patterns detected: {result['patterns_detected']}")
        print(f"Transitions: {result['transitions_count']}")
