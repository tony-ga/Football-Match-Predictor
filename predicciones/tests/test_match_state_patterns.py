#!/usr/bin/env python
"""
Tests for match_state_analyzer module.

Tests cover:
- Minute bucket construction
- Score diff bucket calculation
- State detection (trailing/leading/tied)
- Window generation
- State transition generation
- Robustness with missing events/minutes
- CSV export correctness
"""
import pytest
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.match_state_analyzer import (
    MinuteBucket,
    ScoreDiffBucket,
    CardBucket,
    Phase,
    MatchState,
    StateWindow,
    build_state_at_minute,
    compute_score_at_minute,
    compute_red_cards_at_minute,
    generate_fixed_windows,
    generate_event_anchored_windows,
    get_match_context,
    extract_commentary_events,
    detect_common_patterns,
    generate_markov_ready_dataset,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_match_context():
    """Sample match context for testing."""
    return {
        'event_id': '760500',
        'date': '2026-07-03T22:00Z',
        'home_team': 'Argentina',
        'away_team': 'Cabo Verde',
        'home_team_id': '202',
        'away_team_id': '2597',
        'status': 'Final',
        'neutral_site': True
    }


@pytest.fixture
def sample_events():
    """Sample events for testing window generation."""
    return [
        {'sequence': 0, 'minute': 0, 'period': 1, 'event_type': 'kickoff', 
         'team_name': '', 'team_id': '', 'players': [], 'description': 'Kickoff'},
        {'sequence': 10, 'minute': 12, 'period': 1, 'event_type': 'goal',
         'team_name': 'Argentina', 'team_id': '202', 'players': ['L. Messi'],
         'description': 'Goal! Argentina 1, Cabo Verde 0'},
        {'sequence': 20, 'minute': 35, 'period': 1, 'event_type': 'yellow_card',
         'team_name': 'Cabo Verde', 'team_id': '2597', 'players': ['Player A'],
         'description': 'Yellow card'},
        {'sequence': 30, 'minute': 45, 'period': 1, 'event_type': 'halftime',
         'team_name': '', 'team_id': '', 'players': [], 'description': 'Halftime'},
        {'sequence': 31, 'minute': 46, 'period': 2, 'event_type': 'second_half_start',
         'team_name': '', 'team_id': '', 'players': [], 'description': 'Second half starts'},
        {'sequence': 40, 'minute': 59, 'period': 2, 'event_type': 'goal',
         'team_name': 'Cabo Verde', 'team_id': '2597', 'players': ['Player B'],
         'description': 'Goal! Argentina 1, Cabo Verde 1'},
        {'sequence': 50, 'minute': 68, 'period': 2, 'event_type': 'goal',
         'team_name': 'Argentina', 'team_id': '202', 'players': ['L. Martinez'],
         'description': 'Goal! Argentina 2, Cabo Verde 1'},
        {'sequence': 60, 'minute': 75, 'period': 2, 'event_type': 'corner',
         'team_name': 'Cabo Verde', 'team_id': '2597', 'players': [],
         'description': 'Corner'},
        {'sequence': 70, 'minute': 82, 'period': 2, 'event_type': 'shot_on_target',
         'team_name': 'Cabo Verde', 'team_id': '2597', 'players': ['Player C'],
         'description': 'Shot on target'},
        {'sequence': 80, 'minute': 90, 'period': 2, 'event_type': 'fulltime',
         'team_name': '', 'team_id': '', 'players': [], 'description': 'Full time'},
    ]


@pytest.fixture
def sample_summary():
    """Sample summary dict mimicking ESPN response structure."""
    return {
        'header': {
            'id': '760500',
            'date': '2026-07-03T22:00Z',
            'competitions': [{
                'id': '401882926',
                'date': '2026-07-03T22:00Z',
                'neutralSite': True,
                'status': {'type': {'description': 'Final'}},
                'competitors': [
                    {
                        'team': {'id': '202', 'displayName': 'Argentina'},
                        'homeAway': 'home',
                        'score': '3',
                        'linescores': [
                            {'displayValue': '1'},
                            {'displayValue': '0'},
                            {'displayValue': '1'},
                            {'displayValue': '1'}
                        ]
                    },
                    {
                        'team': {'id': '2597', 'displayName': 'Cabo Verde'},
                        'homeAway': 'away',
                        'score': '2',
                        'linescores': [
                            {'displayValue': '0'},
                            {'displayValue': '1'},
                            {'displayValue': '1'},
                            {'displayValue': '0'}
                        ]
                    }
                ]
            }]
        },
        'commentary': [
            {
                'sequence': 0,
                'time': {'value': 0, 'displayValue': ''},
                'text': 'Lineups announced',
                'play': {
                    'type': {'text': 'Lineups', 'type': 'lineups'},
                    'period': {'number': 1},
                    'clock': {'value': 0, 'displayValue': ''}
                }
            },
            {
                'sequence': 1,
                'time': {'value': 720, 'displayValue': "12'"},
                'text': 'Goal! Argentina 1, Cabo Verde 0. L. Messi scores!',
                'play': {
                    'type': {'text': 'Goal', 'type': 'goal'},
                    'period': {'number': 1},
                    'clock': {'value': 720, 'displayValue': "12'"},
                    'team': {'displayName': 'Argentina', 'id': '202'},
                    'participants': [{'athlete': {'displayName': 'L. Messi'}}]
                }
            }
        ],
        'keyEvents': [
            {
                'id': '1',
                'type': {'text': 'Goal', 'type': 'goal'},
                'period': {'number': 1},
                'clock': {'value': 720, 'displayValue': "12'"},
                'scoringPlay': True,
                'team': {'displayName': 'Argentina', 'id': '202'}
            }
        ],
        'boxscore': {
            'teams': [
                {
                    'team': {'displayName': 'Argentina'},
                    'homeAway': 'home',
                    'statistics': [
                        {'name': 'totalShots', 'displayValue': '15'},
                        {'name': 'possessionPct', 'displayValue': '62'}
                    ]
                }
            ]
        }
    }


# =============================================================================
# TEST: MINUTE BUCKET CONSTRUCTION
# =============================================================================

class TestMinuteBucket:
    """Tests for MinuteBucket enum and from_minute method."""
    
    def test_first_half_early(self):
        """Minutes 0-15 in first half -> M0_15."""
        assert MinuteBucket.from_minute(0, 1) == MinuteBucket.M0_15
        assert MinuteBucket.from_minute(10, 1) == MinuteBucket.M0_15
        assert MinuteBucket.from_minute(15, 1) == MinuteBucket.M0_15
    
    def test_first_half_middle(self):
        """Minutes 16-30 in first half -> M16_30."""
        assert MinuteBucket.from_minute(16, 1) == MinuteBucket.M16_30
        assert MinuteBucket.from_minute(25, 1) == MinuteBucket.M16_30
        assert MinuteBucket.from_minute(30, 1) == MinuteBucket.M16_30
    
    def test_first_half_late(self):
        """Minutes 31-45+ in first half -> M31_45."""
        assert MinuteBucket.from_minute(31, 1) == MinuteBucket.M31_45
        assert MinuteBucket.from_minute(45, 1) == MinuteBucket.M31_45
    
    def test_second_half_early(self):
        """Minutes 46-60 -> M46_60."""
        assert MinuteBucket.from_minute(46, 2) == MinuteBucket.M46_60
        assert MinuteBucket.from_minute(55, 2) == MinuteBucket.M46_60
        assert MinuteBucket.from_minute(60, 2) == MinuteBucket.M46_60
    
    def test_second_half_middle(self):
        """Minutes 61-75 -> M61_75."""
        assert MinuteBucket.from_minute(61, 2) == MinuteBucket.M61_75
        assert MinuteBucket.from_minute(70, 2) == MinuteBucket.M61_75
        assert MinuteBucket.from_minute(75, 2) == MinuteBucket.M61_75
    
    def test_second_half_late(self):
        """Minutes 76-90+ -> M76_90."""
        assert MinuteBucket.from_minute(76, 2) == MinuteBucket.M76_90
        assert MinuteBucket.from_minute(90, 2) == MinuteBucket.M76_90


# =============================================================================
# TEST: SCORE DIFF BUCKET
# =============================================================================

class TestScoreDiffBucket:
    """Tests for ScoreDiffBucket enum and from_diff method."""
    
    def test_losing_heavily(self):
        """Diff <= -2 -> LOSING_2PLUS."""
        assert ScoreDiffBucket.from_diff(-2) == ScoreDiffBucket.LOSING_2PLUS
        assert ScoreDiffBucket.from_diff(-3) == ScoreDiffBucket.LOSING_2PLUS
    
    def test_losing_by_one(self):
        """Diff = -1 -> LOSING_1."""
        assert ScoreDiffBucket.from_diff(-1) == ScoreDiffBucket.LOSING_1
    
    def test_tied(self):
        """Diff = 0 -> TIED."""
        assert ScoreDiffBucket.from_diff(0) == ScoreDiffBucket.TIED
    
    def test_winning_by_one(self):
        """Diff = +1 -> WINNING_1."""
        assert ScoreDiffBucket.from_diff(1) == ScoreDiffBucket.WINNING_1
    
    def test_winning_heavily(self):
        """Diff >= +2 -> WINNING_2PLUS."""
        assert ScoreDiffBucket.from_diff(2) == ScoreDiffBucket.WINNING_2PLUS
        assert ScoreDiffBucket.from_diff(3) == ScoreDiffBucket.WINNING_2PLUS


# =============================================================================
# TEST: CARD BUCKET
# =============================================================================

class TestCardBucket:
    """Tests for CardBucket enum and from_count method."""
    
    def test_no_cards(self):
        """0 red cards -> NONE."""
        assert CardBucket.from_count(0) == CardBucket.NONE
    
    def test_one_plus_cards(self):
        """1+ red cards -> ONE_PLUS."""
        assert CardBucket.from_count(1) == CardBucket.ONE_PLUS
        assert CardBucket.from_count(2) == CardBucket.ONE_PLUS


# =============================================================================
# TEST: STATE CONSTRUCTION
# =============================================================================

class TestBuildStateAtMinute:
    """Tests for build_state_at_minute function."""
    
    def test_early_tied_state(self):
        """State at minute 0, tied 0-0."""
        state = build_state_at_minute(
            minute=0, period=1,
            home_score=0, away_score=0,
            home_red_cards=0, away_red_cards=0,
            home_team='Home', away_team='Away'
        )
        
        assert state.minute_bucket == '0-15'
        assert state.score_diff_bucket == '0'
        assert state.home_red_cards == '0'
        assert state.away_red_cards == '0'
        assert state.phase == 'regular_time'
    
    def test_trailing_state_after_60(self):
        """State when home team trailing after 60 minutes."""
        state = build_state_at_minute(
            minute=65, period=2,
            home_score=1, away_score=2,
            home_red_cards=0, away_red_cards=1,
            home_team='Home', away_team='Away'
        )
        
        assert state.minute_bucket == '61-75'
        assert state.score_diff_bucket == '-1'
        assert state.home_red_cards == '0'
        assert state.away_red_cards == '1_plus'
    
    def test_state_to_key(self):
        """State to_key generates unique identifier."""
        state = build_state_at_minute(
            minute=70, period=2,
            home_score=2, away_score=1,
            home_red_cards=0, away_red_cards=0,
            home_team='Home', away_team='Away'
        )
        
        key = state.to_key()
        assert '61-75' in key
        assert '+1' in key


# =============================================================================
# TEST: SCORE COMPUTATION AT MINUTE
# =============================================================================

class TestComputeScoreAtMinute:
    """Tests for compute_score_at_minute function."""
    
    def test_score_at_kickoff(self, sample_events):
        """Score should be 0-0 at minute 0."""
        home, away = compute_score_at_minute(
            sample_events, 0, 'Argentina', 'Cabo Verde'
        )
        assert home == 0
        assert away == 0
    
    def test_score_after_first_goal(self, sample_events):
        """Score after first goal at minute 12."""
        home, away = compute_score_at_minute(
            sample_events, 15, 'Argentina', 'Cabo Verde'
        )
        assert home == 1
        assert away == 0
    
    def test_score_at_halftime(self, sample_events):
        """Score at halftime (after minute 45)."""
        home, away = compute_score_at_minute(
            sample_events, 45, 'Argentina', 'Cabo Verde'
        )
        assert home == 1
        assert away == 0
    
    def test_score_after_equalizer(self, sample_events):
        """Score after equalizer at minute 59."""
        home, away = compute_score_at_minute(
            sample_events, 60, 'Argentina', 'Cabo Verde'
        )
        assert home == 1
        assert away == 1
    
    def test_final_score(self, sample_events):
        """Final score after all goals."""
        home, away = compute_score_at_minute(
            sample_events, 90, 'Argentina', 'Cabo Verde'
        )
        assert home == 2
        assert away == 1


# =============================================================================
# TEST: RED CARD COMPUTATION
# =============================================================================

class TestComputeRedCardsAtMinute:
    """Tests for compute_red_cards_at_minute function."""
    
    def test_no_red_cards(self, sample_events):
        """No red cards in sample events."""
        count = compute_red_cards_at_minute(sample_events, 90, 'Argentina')
        assert count == 0


# =============================================================================
# TEST: FIXED WINDOW GENERATION
# =============================================================================

class TestGenerateFixedWindows:
    """Tests for generate_fixed_windows function."""
    
    def test_generates_windows_for_both_teams(self, sample_events, sample_match_context):
        """Should generate windows for both home and away teams."""
        windows = generate_fixed_windows(sample_events, sample_match_context, window_size=15)
        
        # Should have windows for both teams (6 windows per team for 90 min / 15)
        assert len(windows) > 0
        
        # Check that we have both teams represented
        team_names = set(w.team_name for w in windows)
        assert 'Argentina' in team_names
        assert 'Cabo Verde' in team_names
    
    def test_window_has_correct_structure(self, sample_events, sample_match_context):
        """Each window should have required attributes."""
        windows = generate_fixed_windows(sample_events, sample_match_context, window_size=15)
        
        if windows:
            w = windows[0]
            assert w.match_id == '760500'
            assert w.team_name in ('Argentina', 'Cabo Verde')
            assert isinstance(w.state_t, dict)
            assert 'minute_bucket' in w.state_t
            assert 'score_diff_bucket' in w.state_t
            assert w.window_start_minute >= 0
            assert w.window_end_minute <= 90
    
    def test_window_state_tracks_score_progression(self, sample_events, sample_match_context):
        """State should reflect score at window start."""
        windows = generate_fixed_windows(sample_events, sample_match_context, window_size=15)
        
        # Find a window starting after first goal (minute 12)
        for w in windows:
            if w.window_start_minute >= 15 and w.team_name == 'Argentina':
                # Argentina should be winning at this point
                assert w.state_t['score_diff_bucket'] in ('+1', '+2_or_more')
                break


# =============================================================================
# TEST: EVENT-ANCHORED WINDOW GENERATION
# =============================================================================

class TestGenerateEventAnchoredWindows:
    """Tests for generate_event_anchored_windows function."""
    
    def test_anchors_to_goals(self, sample_events, sample_match_context):
        """Should create windows anchored to goal events."""
        windows = generate_event_anchored_windows(sample_events, sample_match_context)
        
        # Should have at least one window per goal
        goal_windows = [w for w in windows if any(
            e.get('_is_anchor') and e.get('event_type') == 'goal' 
            for e in w.events_in_window
        )]
        
        assert len(goal_windows) >= 2  # At least 2 goals in sample
    
    def test_window_starts_at_anchor_minute(self, sample_events, sample_match_context):
        """Window should start at the anchor event minute."""
        windows = generate_event_anchored_windows(sample_events, sample_match_context)
        
        for w in windows:
            anchor_events = [e for e in w.events_in_window if e.get('_is_anchor')]
            if anchor_events:
                anchor_minute = anchor_events[0]['minute']
                assert w.window_start_minute == anchor_minute


# =============================================================================
# TEST: MATCH CONTEXT EXTRACTION
# =============================================================================

class TestGetMatchContext:
    """Tests for get_match_context function."""
    
    def test_extracts_basic_info(self, sample_summary):
        """Should extract event_id, teams, status."""
        context = get_match_context(sample_summary)
        
        assert context['event_id'] == '760500'
        assert context['home_team'] == 'Argentina'
        assert context['away_team'] == 'Cabo Verde'
        assert context['status'] == 'Final'
    
    def test_handles_missing_competitions(self):
        """Should return empty dict if no competitions."""
        context = get_match_context({})
        assert context == {}


# =============================================================================
# TEST: COMMENTARY EVENT EXTRACTION
# =============================================================================

class TestExtractCommentaryEvents:
    """Tests for extract_commentary_events function."""
    
    def test_extracts_events_with_timing(self, sample_summary):
        """Should extract events with minute and period info."""
        events = extract_commentary_events(sample_summary, '760500')
        
        assert len(events) > 0
        
        # Check structure of first event
        event = events[0]
        assert 'minute' in event
        assert 'period' in event
        assert 'event_type' in event
    
    def test_normalizes_goal_events(self, sample_summary):
        """Goal events should be normalized correctly."""
        events = extract_commentary_events(sample_summary, '760500')
        
        goals = [e for e in events if e['event_type'] == 'goal']
        assert len(goals) >= 1
        
        # Check goal has team and player info
        goal = goals[0]
        assert goal['team_name'] == 'Argentina'
        assert 'Messi' in str(goal.get('players', []))


# =============================================================================
# TEST: PATTERN DETECTION
# =============================================================================

class TestDetectCommonPatterns:
    """Tests for detect_common_patterns function."""
    
    def test_returns_pattern_results(self, sample_events, sample_match_context):
        """Should return list of PatternResult objects."""
        windows = generate_fixed_windows(sample_events, sample_match_context)
        patterns = detect_common_patterns(windows)
        
        assert len(patterns) > 0
        
        # Check pattern structure
        pattern = patterns[0]
        assert hasattr(pattern, 'pattern_name')
        assert hasattr(pattern, 'sample_size')
        assert hasattr(pattern, 'mean')
        assert hasattr(pattern, 'lift')
    
    def test_patterns_have_reasonable_stats(self, sample_events, sample_match_context):
        """Pattern statistics should be reasonable."""
        windows = generate_fixed_windows(sample_events, sample_match_context)
        patterns = detect_common_patterns(windows)
        
        for p in patterns:
            # Sample size should be non-negative
            assert p.sample_size >= 0
            # Mean should be non-negative for count metrics
            assert p.mean >= 0


# =============================================================================
# TEST: MARKOV-READY DATASET
# =============================================================================

class TestGenerateMarkovReadyDataset:
    """Tests for generate_markov_ready_dataset function."""
    
    def test_generates_transitions(self, sample_events, sample_match_context):
        """Should generate list of transition dicts."""
        windows = generate_fixed_windows(sample_events, sample_match_context)
        transitions = generate_markov_ready_dataset(windows)
        
        assert len(transitions) == len(windows)
    
    def test_transition_has_required_fields(self, sample_events, sample_match_context):
        """Each transition should have required fields."""
        windows = generate_fixed_windows(sample_events, sample_match_context)
        transitions = generate_markov_ready_dataset(windows)
        
        if transitions:
            t = transitions[0]
            required_fields = [
                'match_id', 'team_id', 'team_name', 'state_t',
                'window_start', 'window_end',
                'corners_next_window', 'shots_next_window',
                'goals_next_window', 'concede_next_window'
            ]
            
            for field in required_fields:
                assert field in t, f"Missing field: {field}"
    
    def test_state_serialized_as_json(self, sample_events, sample_match_context):
        """State should be JSON-serialized string."""
        windows = generate_fixed_windows(sample_events, sample_match_context)
        transitions = generate_markov_ready_dataset(windows)
        
        if transitions:
            t = transitions[0]
            state_dict = json.loads(t['state_t'])
            assert 'minute_bucket' in state_dict
            assert 'score_diff_bucket' in state_dict


# =============================================================================
# TEST: ROBUSTNESS WITH MISSING DATA
# =============================================================================

class TestRobustnessWithMissingData:
    """Tests for handling missing events/minutes gracefully."""
    
    def test_handles_empty_events(self, sample_match_context):
        """Should handle empty event list without crashing."""
        windows = generate_fixed_windows([], sample_match_context)
        
        # Should still generate windows based on time intervals
        assert len(windows) > 0
    
    def test_handles_sparse_events(self, sample_match_context):
        """Should handle very few events."""
        sparse_events = [
            {'sequence': 0, 'minute': 0, 'period': 1, 'event_type': 'kickoff',
             'team_name': '', 'team_id': '', 'players': [], 'description': ''}
        ]
        
        windows = generate_fixed_windows(sparse_events, sample_match_context)
        assert len(windows) > 0
    
    def test_handles_missing_team_info(self):
        """Should handle missing team information."""
        context = {
            'event_id': '123',
            'home_team': '',  # Empty
            'away_team': 'Away',
            'home_team_id': '',
            'away_team_id': '2'
        }
        
        events = []
        windows = generate_fixed_windows(events, context)
        
        # Should not crash, may have fewer windows
        assert isinstance(windows, list)


# =============================================================================
# TEST: CSV EXPORT READINESS
# =============================================================================

class TestCSVExportReadiness:
    """Tests for ensuring data is ready for CSV export."""
    
    def test_state_window_to_dict_is_serializable(self, sample_events, sample_match_context):
        """StateWindow.to_dict() should produce JSON-serializable output."""
        windows = generate_fixed_windows(sample_events, sample_match_context)
        
        if windows:
            w = windows[0]
            d = w.to_dict()
            
            # Should be able to serialize to JSON
            json_str = json.dumps(d)
            assert len(json_str) > 0
    
    def test_markov_transition_is_csv_ready(self, sample_events, sample_match_context):
        """Markov transitions should be flat enough for CSV."""
        windows = generate_fixed_windows(sample_events, sample_match_context)
        transitions = generate_markov_ready_dataset(windows)
        
        if transitions:
            t = transitions[0]
            # Nested dicts should be JSON strings
            assert isinstance(t['state_t'], str)
            
            # Verify it's valid JSON
            state = json.loads(t['state_t'])
            assert isinstance(state, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
