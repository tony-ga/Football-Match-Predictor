"""
Tests for ESPN player parsers and market availability.

Tests cover:
1. Parse rosters
2. Parse leaders  
3. Parse keyEvents with player
4. Combine player signals
5. Partial player props availability
6. Full player props unavailable when SOT/assists missing
7. No exceptions if optional fields missing
"""
import pytest
from typing import Dict, Any, List

from src.data.espn_player_parsers import (
    extract_roster_players,
    extract_player_signals,
    extract_player_events,
    build_player_match_rows,
    check_player_data_availability,
)
from src.models.market_availability import MarketAvailabilityEvaluator


class TestExtractRosterPlayers:
    """Test roster extraction from ESPN summary."""
    
    def test_extract_from_boxscore_teams_roster(self):
        """Test extracting players from boxscore.teams[].roster format."""
        summary = {
            "boxscore": {
                "teams": [
                    {
                        "homeAway": "home",
                        "team": {"id": "123", "displayName": "Team A"},
                        "roster": [
                            {
                                "id": "1001",
                                "displayName": "Player One",
                                "jersey": "10",
                                "position": {"abbreviation": "FWD"},
                                "starter": True,
                            },
                            {
                                "id": "1002",
                                "displayName": "Player Two",
                                "jersey": "11",
                                "position": {"abbreviation": "MID"},
                                "starter": False,
                            }
                        ]
                    }
                ]
            }
        }
        
        players = extract_roster_players(summary)
        
        assert len(players) == 2
        assert players[0]["athlete_id"] == "1001"
        assert players[0]["player_name"] == "Player One"
        assert players[0]["team_id"] == "123"
        assert players[0]["is_starter"] is True
        assert players[0]["position"] == "FWD"
        assert players[0]["jersey"] == "10"
    
    def test_extract_with_missing_fields(self):
        """Test that missing optional fields don't cause errors."""
        summary = {
            "boxscore": {
                "teams": [
                    {
                        "homeAway": "away",
                        "team": {"id": "456"},
                        "roster": [
                            {
                                "id": "2001",
                                # Missing displayName, position, jersey, starter
                            }
                        ]
                    }
                ]
            }
        }
        
        players = extract_roster_players(summary)
        
        assert len(players) == 1
        assert players[0]["athlete_id"] == "2001"
        assert players[0]["player_name"] == ""
        assert players[0]["position"] == ""
        assert players[0]["jersey"] == ""
        assert players[0]["is_starter"] is False
    
    def test_empty_summary(self):
        """Test empty summary returns empty list."""
        assert extract_roster_players({}) == []
        assert extract_roster_players(None) == []


class TestExtractPlayerSignals:
    """Test signal extraction from leaders and boxscore."""
    
    def test_extract_from_leaders(self):
        """Test extracting signals from leaders block."""
        summary = {
            "leaders": [
                {
                    "name": "goals",
                    "displayName": "Goals",
                    "topPerformers": [
                        {
                            "athlete": {
                                "id": "3001",
                                "displayName": "Goal Scorer",
                                "team": {"id": "789", "displayName": "Team B"}
                            },
                            "value": 5,
                        }
                    ]
                }
            ]
        }
        
        signals = extract_player_signals(summary)
        
        assert len(signals) == 1
        assert signals[0]["athlete_id"] == "3001"
        assert signals[0]["signal_type"] == "goal_scorer"
        assert signals[0]["is_leader"] is True
    
    def test_infer_signal_types(self):
        """Test different signal type inference."""
        summary = {
            "leaders": [
                {"name": "assists", "topPerformers": [{"athlete": {"id": "1"}}]},
                {"name": "shots on target", "topPerformers": [{"athlete": {"id": "2"}}]},
                {"name": "total shots", "topPerformers": [{"athlete": {"id": "3"}}]},
                {"name": "saves", "topPerformers": [{"athlete": {"id": "4"}}]},
            ]
        }
        
        signals = extract_player_signals(summary)
        
        signal_types = {s["stat_name"]: s["signal_type"] for s in signals}
        assert signal_types["assists"] == "playmaker"
        assert signal_types["shots on target"] == "shot_accuracy"
        assert signal_types["total shots"] == "volume_shooter"
        assert signal_types["saves"] == "goalkeeper"
    
    def test_empty_leaders(self):
        """Test empty leaders returns empty list."""
        assert extract_player_signals({"leaders": []}) == []


class TestExtractPlayerEvents:
    """Test event extraction from keyEvents."""
    
    def test_extract_goals(self):
        """Test extracting goal events."""
        summary = {
            "keyEvents": [
                {
                    "text": "Goal scored",
                    "type": {"id": "goal"},
                    "period": {"number": 1},
                    "clock": {"displayValue": "23:45"},
                    "participants": [
                        {
                            "id": "4001",
                            "displayName": "Scorer",
                            "team": {"id": "111"}
                        }
                    ]
                }
            ]
        }
        
        events = extract_player_events(summary)
        
        assert len(events) == 1
        assert events[0]["athlete_id"] == "4001"
        assert events[0]["event_type"] == "goal"
        assert events[0]["period"] == 1
    
    def test_extract_cards(self):
        """Test extracting card events."""
        summary = {
            "keyEvents": [
                {"text": "Yellow card shown", "participants": [{"id": "5001", "displayName": "Player"}]},
                {"text": "Red card shown", "participants": [{"id": "5002", "displayName": "Player"}]},
            ]
        }
        
        events = extract_player_events(summary)
        
        assert len(events) == 2
        event_types = {e["athlete_id"]: e["event_type"] for e in events}
        assert event_types["5001"] == "yellow_card"
        assert event_types["5002"] == "red_card"
    
    def test_extract_substitutions(self):
        """Test extracting substitution events."""
        summary = {
            "keyEvents": [
                {"text": "Substitution made", "participants": [{"id": "6001"}]}
            ]
        }
        
        events = extract_player_events(summary)
        
        assert len(events) == 1
        assert events[0]["event_type"] == "substitution"
    
    def test_empty_keyevents(self):
        """Test empty keyEvents returns empty list."""
        assert extract_player_events({"keyEvents": []}) == []


class TestBuildPlayerMatchRows:
    """Test combining roster + signals + events."""
    
    def test_combine_all_sources(self):
        """Test building complete player rows from all sources."""
        summary = {
            "boxscore": {
                "teams": [{
                    "homeAway": "home",
                    "team": {"id": "1", "displayName": "Team"},
                    "roster": [
                        {"id": "p1", "displayName": "Player 1", "starter": True}
                    ]
                }]
            },
            "leaders": [
                {
                    "name": "goals",
                    "topPerformers": [
                        {"athlete": {"id": "p1", "displayName": "Player 1", "team": {"id": "1"}}}
                    ]
                }
            ],
            "keyEvents": [
                {
                    "text": "Goal",
                    "participants": [{"id": "p1", "displayName": "Player 1", "team": {"id": "1"}}]
                }
            ]
        }
        
        rows = build_player_match_rows(summary)
        
        assert len(rows) >= 1
        row = rows[0]
        assert row["athlete_id"] == "p1"
        assert row["is_goal_leader"] is True
        assert row["goals"] == 1


class TestCheckPlayerDataAvailability:
    """Test availability checking at multiple levels."""
    
    def test_full_availability(self):
        """Test detection of full player data availability."""
        summary = {
            "boxscore": {
                "teams": [{
                    "roster": [{"id": "1", "displayName": "P1", "starter": True}]
                }]
            },
            "leaders": [{"name": "goals", "topPerformers": [{"athlete": {"id": "1"}}]}],
            "keyEvents": [{"text": "Goal", "participants": [{"id": "1"}]}]
        }
        
        avail = check_player_data_availability(summary)
        
        assert avail["player_roster_available"] is True
        assert avail["player_signal_available"] is True
        assert avail["player_event_available"] is True
        assert avail["has_offensive_signals"] is True
        assert avail["has_goal_events"] is True
        assert avail["player_props_partial"] is True


class TestMarketAvailabilityEvaluator:
    """Test multi-level market availability evaluation."""
    
    def test_evaluate_player_props_multi_level(self):
        """Test player props availability at multiple levels."""
        evaluator = MarketAvailabilityEvaluator()
        
        # Matches with only rosters
        matches_roster_only = [
            {"players": [{"athlete_id": "1"}]},
            {"players": [{"athlete_id": "2"}]},
            {"players": [{"athlete_id": "3"}]},
        ]
        
        result = evaluator.evaluate_player_props_availability(
            matches_roster_only, [],
            home_lineup_available=False,
            away_lineup_available=False,
        )
        
        assert result["available"] is True
        assert result["coverage_levels"]["matches_with_rosters"] == 3
        assert result["prop_availability"]["anytime_scorer"]["available"] is False
        
    def test_evaluate_with_signals_enables_scorer_props(self):
        """Test that signals enable scorer props."""
        evaluator = MarketAvailabilityEvaluator()
        
        matches_with_signals = [
            {
                "players": [{"athlete_id": "1"}],
                "player_signals": [{"athlete_id": "1", "signal_type": "goal_scorer"}]
            },
            {
                "players": [{"athlete_id": "2"}],
                "player_signals": [{"athlete_id": "2", "signal_type": "goal_scorer"}]
            },
            {
                "players": [{"athlete_id": "3"}],
                "player_signals": [{"athlete_id": "3", "signal_type": "goal_scorer"}]
            },
        ]
        
        result = evaluator.evaluate_player_props_availability(
            matches_with_signals, [],
            home_lineup_available=False,
            away_lineup_available=False,
        )
        
        assert result["coverage_levels"]["matches_with_player_signals"] == 3
        assert result["prop_availability"]["anytime_scorer"]["available"] is True
        
    def test_stats_props_require_full_stats(self):
        """Test that SOT/assists props require full player stats."""
        evaluator = MarketAvailabilityEvaluator()
        
        # Matches without full stats
        matches_partial = [
            {"players": [{"athlete_id": "1"}]},
            {"players": [{"athlete_id": "2"}]},
        ]
        
        result = evaluator.evaluate_player_props_availability(
            matches_partial, [],
        )
        
        assert result["prop_availability"]["shots_on_target"]["available"] is False
        assert result["prop_availability"]["assists"]["available"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
