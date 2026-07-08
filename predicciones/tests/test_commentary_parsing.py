"""
Unit tests for ESPN soccer commentary parsing.

Tests the _parse_commentary_events function with real/mock data
from ESPN's soccer API.
"""
import pytest
from typing import Dict, List, Any

from src.data.espn_stats_parsers import (
    parse_commentary_events_with_stats,
    extract_events_from_summary,
)


# =============================================================================
# MOCK DATA - Real ESPN commentary samples
# =============================================================================

MOCK_COMMENTARY_GOAL = [
    {
        "sequence": 12,
        "time": {"value": 845.0, "displayValue": "15'"},
        "text": "Goal! Argentina 0, Egypt 1. Yasser Ibrahim (Egypt) header from the centre of the box to the bottom right corner.",
        "play": {
            "id": "49730975",
            "type": {"id": "137", "text": "Goal - Header", "type": "goal---header"},
            "text": "Goal! Argentina 0, Egypt 1. Yasser Ibrahim (Egypt) header...",
            "period": {"number": 1},
            "clock": {"value": 845.0, "displayValue": "15'"},
            "team": {"displayName": "Egypt"},
            "participants": [
                {"athlete": {"displayName": "Yasser Ibrahim"}},
                {"athlete": {"displayName": "Marawan Attia"}}
            ],
        }
    }
]

MOCK_COMMENTARY_YELLOW_CARD = [
    {
        "sequence": 103,
        "time": {"value": 5541.0, "displayValue": "90'+3'"},
        "text": "Mostafa Shobeir (Egypt) is shown the yellow card.",
        "play": {
            "id": "49732269",
            "type": {"id": "94", "text": "Yellow Card", "type": "yellow-card"},
            "period": {"number": 2},
            "clock": {"value": 5541.0, "displayValue": "90'+3'"},
            "team": {"displayName": "Egypt"},
            "participants": [{"athlete": {"displayName": "Mostafa Shoubir"}}],
        }
    }
]

MOCK_COMMENTARY_CORNER = [
    {
        "sequence": 11,
        "time": {"value": 814.0, "displayValue": "14'"},
        "text": "Corner, Egypt. Conceded by Lisandro Martínez.",
        "play": {
            "id": "49730967",
            "type": {"id": "95", "text": "Corner Awarded", "type": "corner-awarded"},
            "period": {"number": 1},
            "clock": {"value": 814.0, "displayValue": "14'"},
            "team": {"displayName": "Egypt"},
        }
    }
]

MOCK_COMMENTARY_SUBSTITUTION = [
    {
        "sequence": 50,
        "time": {"value": 2700.0, "displayValue": "45'"},
        "text": "Substitution, Egypt. Hamdi Fathy replaces Emam Ashour because of an injury.",
        "play": {
            "id": "49731492",
            "type": {"id": "76", "text": "Substitution", "type": "substitution"},
            "period": {"number": 2},
            "clock": {"value": 2700.0, "displayValue": "45'"},
            "team": {"displayName": "Egypt"},
            "participants": [
                {"athlete": {"displayName": "Hamdy Fathy"}},
                {"athlete": {"displayName": "Emam Ashour"}}
            ],
        }
    }
]

MOCK_COMMENTARY_HALFTIME = [
    {
        "sequence": 49,
        "time": {"value": 3070.0, "displayValue": "45'+7'"},
        "text": "First Half ends, Argentina 0, Egypt 1.",
        "play": {
            "id": "49731489",
            "type": {"id": "81", "text": "Halftime", "type": "halftime"},
            "period": {"number": 1},
            "clock": {"value": 2700.0, "displayValue": "45'+7'"},
        }
    }
]

MOCK_COMMENTARY_FULLTIME = [
    {
        "sequence": 115,
        "time": {"value": 6100.0, "displayValue": "90'+10'"},
        "text": "Match ends, Argentina 3, Egypt 2.",
        "play": {
            "id": "49732400",
            "type": {"id": "82", "text": "Fulltime", "type": "fulltime"},
        }
    }
]

MOCK_COMMENTARY_UNKNOWN_EVENT = [
    {
        "sequence": 2,
        "time": {"value": 252.0, "displayValue": "5'"},
        "text": "Mohamed Salah (Egypt) wins a free kick in the defensive half after a handball by opponent.",
        "play": {
            "id": "49730788",
            "type": {"id": "66", "text": "Foul", "type": "foul"},
            "period": {"number": 1},
            "clock": {"value": 252.0, "displayValue": "5'"},
            "team": {"displayName": "Argentina"},
            "participants": [
                {"athlete": {"displayName": "Enzo Fernández"}},
                {"athlete": {"displayName": "Mohamed Salah"}}
            ],
        }
    }
]

MOCK_COMMENTARY_EMPTY = []

MOCK_COMMENTARY_MIXED = (
    MOCK_COMMENTARY_GOAL + 
    MOCK_COMMENTARY_YELLOW_CARD + 
    MOCK_COMMENTARY_CORNER + 
    MOCK_COMMENTARY_SUBSTITUTION +
    MOCK_COMMENTARY_HALFTIME +
    MOCK_COMMENTARY_FULLTIME +
    MOCK_COMMENTARY_UNKNOWN_EVENT
)


# =============================================================================
# TESTS
# =============================================================================

class TestParseCommentaryEvents:
    """Tests for parse_commentary_events_with_stats function."""
    
    def test_parse_goal_event(self):
        """Test parsing a goal event."""
        events, counts = parse_commentary_events_with_stats(MOCK_COMMENTARY_GOAL)
        
        assert len(events) == 1
        assert events[0]["event_type"] == "goal"
        assert events[0]["minute"] == 14  # 845 seconds / 60 = 14
        assert events[0]["period"] == 1
        assert events[0]["team_name"] == "Egypt"
        assert events[0]["player_name"] == "Yasser Ibrahim"
        assert "goal" in counts
        assert counts["goal"] == 1
    
    def test_parse_yellow_card_event(self):
        """Test parsing a yellow card event."""
        events, counts = parse_commentary_events_with_stats(MOCK_COMMENTARY_YELLOW_CARD)
        
        assert len(events) == 1
        assert events[0]["event_type"] == "yellow_card"
        assert events[0]["minute"] == 92  # 5541 / 60 = 92
        assert events[0]["period"] == 2
        assert events[0]["team_name"] == "Egypt"
        assert events[0]["player_name"] == "Mostafa Shoubir"
        assert "yellow_card" in counts
        assert counts["yellow_card"] == 1
    
    def test_parse_corner_event(self):
        """Test parsing a corner event."""
        events, counts = parse_commentary_events_with_stats(MOCK_COMMENTARY_CORNER)
        
        assert len(events) == 1
        assert events[0]["event_type"] == "corner"
        assert events[0]["minute"] == 13  # 814 / 60 = 13
        assert events[0]["period"] == 1
        assert events[0]["team_name"] == "Egypt"
        assert "corner" in counts
        assert counts["corner"] == 1
    
    def test_parse_substitution_event(self):
        """Test parsing a substitution event."""
        events, counts = parse_commentary_events_with_stats(MOCK_COMMENTARY_SUBSTITUTION)
        
        assert len(events) == 1
        assert events[0]["event_type"] == "substitution"
        assert events[0]["minute"] == 45  # 2700 / 60 = 45
        assert events[0]["period"] == 2
        assert events[0]["team_name"] == "Egypt"
        assert events[0]["player_name"] == "Hamdy Fathy"
        assert "substitution" in counts
        assert counts["substitution"] == 1
    
    def test_parse_halftime_event(self):
        """Test parsing a halftime event."""
        events, counts = parse_commentary_events_with_stats(MOCK_COMMENTARY_HALFTIME)
        
        assert len(events) == 1
        assert events[0]["event_type"] == "halftime"
        assert events[0]["period"] == 1
        assert "halftime" in counts
        assert counts["halftime"] == 1
    
    def test_parse_fulltime_event(self):
        """Test parsing a fulltime event."""
        events, counts = parse_commentary_events_with_stats(MOCK_COMMENTARY_FULLTIME)
        
        assert len(events) == 1
        assert events[0]["event_type"] == "fulltime"
        assert "fulltime" in counts
        assert counts["fulltime"] == 1
    
    def test_parse_unknown_event(self):
        """Test that unknown events are preserved with raw_event."""
        events, counts = parse_commentary_events_with_stats(MOCK_COMMENTARY_UNKNOWN_EVENT)
        
        assert len(events) == 1
        # Note: This event is now detected as 'handball' due to improved parsing
        assert events[0]["event_type"] in ["unknown", "handball", "free_kick"]
        assert events[0]["team_name"] == "Argentina"
        assert events[0]["player_name"] == "Enzo Fernández"
        assert "raw_event" in events[0]
        # Count should include whatever type was detected
        assert len(counts) == 1
    
    def test_empty_commentary(self):
        """Test handling of empty commentary list."""
        events, counts = parse_commentary_events_with_stats(MOCK_COMMENTARY_EMPTY)
        
        assert len(events) == 0
        assert counts == {}
    
    def test_mixed_events_sorted(self):
        """Test that mixed events are sorted by period then minute."""
        events, counts = parse_commentary_events_with_stats(MOCK_COMMENTARY_MIXED)
        
        # Should have all events
        assert len(events) == 7
        
        # Check sorting: period 1 events first, then period 2
        period_1_events = [e for e in events if e["period"] == 1]
        period_2_events = [e for e in events if e["period"] == 2]
        
        # Period 1 events should come first
        assert events[:len(period_1_events)] == period_1_events
        
        # Within each period, should be sorted by minute
        for i in range(len(period_1_events) - 1):
            assert period_1_events[i]["minute"] <= period_1_events[i+1]["minute"]
    
    def test_event_has_all_required_fields(self):
        """Test that parsed events have all required fields."""
        events, _ = parse_commentary_events_with_stats(MOCK_COMMENTARY_GOAL)
        
        required_fields = [
            "event_type", "minute", "period", "team_name", 
            "player_name", "description", "raw_event"
        ]
        
        for field in required_fields:
            assert field in events[0], f"Missing required field: {field}"
    
    def test_event_type_counts_accurate(self):
        """Test that event type counts are accurate."""
        events, counts = parse_commentary_events_with_stats(MOCK_COMMENTARY_MIXED)
        
        # Note: The 'unknown' event is now detected as 'handball' or 'free_kick'
        # due to improved parsing logic
        assert len(events) == 7
        assert counts["goal"] == 1
        assert counts["yellow_card"] == 1
        assert counts["corner"] == 1
        assert counts["substitution"] == 1
        assert counts["halftime"] == 1
        assert counts["fulltime"] == 1
        # The 7th event should be counted as something (handball, free_kick, or unknown)
        assert sum(counts.values()) == 7


class TestExtractEventsFromSummary:
    """Tests for extract_events_from_summary function with commentary."""
    
    def test_extract_from_summary_with_commentary(self):
        """Test extracting events from a summary dict with commentary."""
        summary = {
            "commentary": MOCK_COMMENTARY_GOAL + MOCK_COMMENTARY_CORNER,
            "plays": None,
        }
        
        events = extract_events_from_summary(summary)
        
        assert len(events) == 2
        assert events[0]["event_type"] == "corner"  # Sorted first by minute
        assert events[1]["event_type"] == "goal"
    
    def test_extract_from_summary_no_commentary_fallback(self):
        """Test fallback when no commentary exists."""
        summary = {
            "commentary": [],
            "plays": [
                {
                    "type": {"id": "137", "text": "Goal"},
                    "text": "Goal scored",
                    "period": {"number": 1},
                    "clock": {"displayValue": "15'"},
                }
            ],
        }
        
        events = extract_events_from_summary(summary)
        
        # Should use fallback plays parsing
        assert len(events) >= 1
    
    def test_extract_from_summary_empty(self):
        """Test handling of empty/None summary."""
        assert extract_events_from_summary(None) == []
        assert extract_events_from_summary({}) == []


class TestEventPerRowFormat:
    """Tests for the event-per-row format in match_events.jsonl."""
    
    def test_goal_event_has_all_fields(self):
        """Test that goal events have all required fields for event-per-row format."""
        events, _ = parse_commentary_events_with_stats(MOCK_COMMENTARY_GOAL)
        
        assert len(events) == 1
        evt = events[0]
        
        # Check all fields needed for event-per-row format
        assert evt["event_type"] == "goal"
        assert evt["minute"] is not None
        assert evt["period"] is not None
        assert evt["team_name"] is not None
        assert evt["player_name"] is not None
        assert evt["description"] is not None
        assert "raw_event" in evt
    
    def test_penalty_goal_detection(self):
        """Test detection of penalty goals."""
        mock_penalty_goal = [
            {
                "sequence": 50,
                "time": {"value": 3600.0, "displayValue": "60'"},
                "text": "Goal! Brazil 1, Croatia 0. Neymar (Brazil) converts the penalty.",
                "play": {
                    "type": {"id": "138", "text": "Penalty Goal", "type": "penalty-goal"},
                    "period": {"number": 2},
                    "team": {"displayName": "Brazil"},
                    "participants": [{"athlete": {"displayName": "Neymar"}}],
                }
            }
        ]
        
        events, counts = parse_commentary_events_with_stats(mock_penalty_goal)
        
        assert len(events) == 1
        assert events[0]["event_type"] == "penalty_goal"
        assert "penalty_goal" in counts
    
    def test_own_goal_detection(self):
        """Test detection of own goals."""
        mock_own_goal = [
            {
                "sequence": 30,
                "time": {"value": 1800.0, "displayValue": "30'"},
                "text": "Own Goal! France 0, Germany 1. Jules Koundé (France) scores an own goal.",
                "play": {
                    "type": {"id": "139", "text": "Own Goal", "type": "own-goal"},
                    "period": {"number": 1},
                    "team": {"displayName": "France"},
                    "participants": [{"athlete": {"displayName": "Jules Koundé"}}],
                }
            }
        ]
        
        events, counts = parse_commentary_events_with_stats(mock_own_goal)
        
        assert len(events) == 1
        assert events[0]["event_type"] == "own_goal"
        assert "own_goal" in counts
    
    def test_kickoff_detection(self):
        """Test detection of kickoff events."""
        mock_kickoff = [
            {
                "sequence": 1,
                "time": {"value": 0.0, "displayValue": "0'"},
                "text": "First Half begins.",
                "play": {
                    "type": {"id": "1", "text": "Kickoff", "type": "kickoff"},
                    "period": {"number": 1},
                }
            }
        ]
        
        events, counts = parse_commentary_events_with_stats(mock_kickoff)
        
        assert len(events) == 1
        assert events[0]["event_type"] == "kickoff"
        assert "kickoff" in counts
    
    def test_second_half_start_detection(self):
        """Test detection of second half start events."""
        mock_second_half = [
            {
                "sequence": 49,
                "time": {"value": 2700.0, "displayValue": "45'"},
                "text": "Second Half begins.",
                "play": {
                    "type": {"id": "80", "text": "Second Half Start", "type": "second-half-start"},
                    "period": {"number": 2},
                }
            }
        ]
        
        events, counts = parse_commentary_events_with_stats(mock_second_half)
        
        assert len(events) == 1
        assert events[0]["event_type"] == "second_half_start"
        assert "second_half_start" in counts
    
    def test_penalty_awarded_detection(self):
        """Test detection of penalty awarded (not yet taken)."""
        mock_penalty_awarded = [
            {
                "sequence": 45,
                "time": {"value": 3300.0, "displayValue": "55'"},
                "text": "Penalty awarded to Spain after VAR review.",
                "play": {
                    "type": {"id": "96", "text": "Penalty", "type": "penalty"},
                    "period": {"number": 2},
                    "team": {"displayName": "Spain"},
                }
            }
        ]
        
        events, counts = parse_commentary_events_with_stats(mock_penalty_awarded)
        
        assert len(events) == 1
        assert events[0]["event_type"] == "penalty"
        assert "penalty" in counts
    
    def test_null_values_for_missing_fields(self):
        """Test that missing team/player names result in null values, not discarded events."""
        mock_minimal_event = [
            {
                "sequence": 5,
                "time": {"value": 300.0, "displayValue": "5'"},
                "text": "Some generic event description without team info.",
                "play": {}
            }
        ]
        
        events, counts = parse_commentary_events_with_stats(mock_minimal_event)
        
        # Event should be preserved even with missing info
        assert len(events) == 1
        assert events[0]["event_type"] == "unknown"
        assert events[0]["team_name"] is None
        assert events[0]["player_name"] is None
        assert "raw_event" in events[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
