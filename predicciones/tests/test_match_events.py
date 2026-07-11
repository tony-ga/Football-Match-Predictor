#!/usr/bin/env python
"""
Tests for ESPN Match Events module.

Covers:
- Parsing valid summary mock with commentary
- Extracting home/away scores and stats
- Event normalization from commentary and keyEvents
- Filtering by event type
- Handling incomplete summaries
- Missing venue / leaders / commentary handling
- Temporal ordering of events
"""
import json
import pytest
from typing import Any, Dict, List

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.espn_match_events import (
    fetch_match_summary,
    fetch_match_core_plays,
    extract_commentary_events,
    extract_key_events,
    extract_core_plays_events,
    normalize_match_events,
    get_match_event_timeline,
    filter_events_by_type,
    _parse_clock_display,
    _normalize_commentary_event,
    _normalize_key_event,
    _merge_events,
)


# =============================================================================
# Mock Data Fixtures
# =============================================================================

@pytest.fixture
def mock_summary_with_commentary() -> Dict[str, Any]:
    """Complete mock summary with commentary events."""
    return {
        "header": {
            "id": "760500",
            "date": "2026-07-05T18:00Z",
            "competitions": [
                {
                    "id": "401234567",
                    "status": {
                        "type": {
                            "name": "STATUS_FULL_TIME",
                            "state": "post"
                        }
                    },
                    "competitors": [
                        {
                            "homeAway": "home",
                            "team": {"displayName": "Argentina", "abbreviation": "ARG"},
                            "score": "2"
                        },
                        {
                            "homeAway": "away",
                            "team": {"displayName": "Cape Verde", "abbreviation": "CPV"},
                            "score": "0"
                        }
                    ]
                }
            ]
        },
        "commentary": [
            {
                "type": {"name": "Kickoff"},
                "text": "First Half begins.",
                "clock": {"displayTime": "0'"},
                "period": {"number": 1},
                "team": None,
                "player": None,
            },
            {
                "type": {"name": "Goal"},
                "text": "Goal! Argentina 1, Cape Verde 0. Lionel Messi (Argentina) right footed shot.",
                "clock": {"displayTime": "12'"},
                "period": {"number": 1},
                "team": {"displayName": "Argentina", "abbreviation": "ARG"},
                "player": {"fullName": "Lionel Messi"},
            },
            {
                "type": {"name": "Yellow Card"},
                "text": "Ryan Mendes (Cape Verde) is shown the yellow card.",
                "clock": {"displayTime": "24'"},
                "period": {"number": 1},
                "team": {"displayName": "Cape Verde", "abbreviation": "CPV"},
                "player": {"fullName": "Ryan Mendes"},
            },
            {
                "type": {"name": "First Half Ends"},
                "text": "First Half ends, Argentina 1, Cape Verde 0.",
                "clock": {"displayTime": "45+2'"},
                "period": {"number": 1},
                "team": None,
                "player": None,
            },
            {
                "type": {"name": "Second Half Begins"},
                "text": "Second Half begins Argentina 1, Cape Verde 0.",
                "clock": {"displayTime": "46'"},
                "period": {"number": 2},
                "team": None,
                "player": None,
            },
            {
                "type": {"name": "Substitution"},
                "text": "Substitution, Cape Verde. Djamal Neves replaces Ryan Mendes.",
                "clock": {"displayTime": "60'"},
                "period": {"number": 2},
                "team": {"displayName": "Cape Verde", "abbreviation": "CPV"},
                "player": {"fullName": "Djamal Neves"},
            },
            {
                "type": {"name": "Goal"},
                "text": "Goal! Argentina 2, Cape Verde 0. Julian Alvarez (Argentina) left footed shot.",
                "clock": {"displayTime": "78'"},
                "period": {"number": 2},
                "team": {"displayName": "Argentina", "abbreviation": "ARG"},
                "player": {"fullName": "Julian Alvarez"},
            },
            {
                "type": {"name": "Match Ends"},
                "text": "Match ends, Argentina 2, Cape Verde 0.",
                "clock": {"displayTime": "90+4'"},
                "period": {"number": 2},
                "team": None,
                "player": None,
            },
        ],
        "keyEvents": [
            {
                "typeId": "goal",
                "text": "Lionel Messi scores!",
                "minute": 12,
                "period": {"number": 1},
                "team": {"displayName": "Argentina", "abbreviation": "ARG"},
                "participants": [{"athlete": {"fullName": "Lionel Messi"}}]
            },
            {
                "typeId": "yellow-card",
                "text": "Ryan Mendes booked",
                "minute": 24,
                "period": {"number": 1},
                "team": {"displayName": "Cape Verde", "abbreviation": "CPV"},
                "participants": [{"athlete": {"fullName": "Ryan Mendes"}}]
            },
            {
                "typeId": "goal",
                "text": "Julian Alvarez scores!",
                "minute": 78,
                "period": {"number": 2},
                "team": {"displayName": "Argentina", "abbreviation": "ARG"},
                "participants": [{"athlete": {"fullName": "Julian Alvarez"}}]
            },
        ],
    }


@pytest.fixture
def mock_summary_empty_commentary() -> Dict[str, Any]:
    """Summary with empty commentary."""
    return {
        "header": {
            "id": "760501",
            "date": "2026-07-06T18:00Z",
            "competitions": [
                {
                    "status": {"type": {"name": "STATUS_FULL_TIME"}},
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Team A"}, "score": "1"},
                        {"homeAway": "away", "team": {"displayName": "Team B"}, "score": "1"},
                    ]
                }
            ]
        },
        "commentary": [],
        "keyEvents": [],
    }


@pytest.fixture
def mock_summary_missing_commentary() -> Dict[str, Any]:
    """Summary without commentary key at all."""
    return {
        "header": {
            "id": "760502",
            "competitions": [
                {
                    "status": {"type": {"name": "STATUS_SCHEDULED"}},
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Team X"}, "score": None},
                        {"homeAway": "away", "team": {"displayName": "Team Y"}, "score": None},
                    ]
                }
            ]
        },
    }


@pytest.fixture
def mock_summary_only_keyevents() -> Dict[str, Any]:
    """Summary with only keyEvents, no commentary."""
    return {
        "header": {
            "id": "760503",
            "competitions": [
                {
                    "status": {"type": {"name": "STATUS_FULL_TIME"}},
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Home FC"}, "score": "3"},
                        {"homeAway": "away", "team": {"displayName": "Away Utd"}, "score": "1"},
                    ]
                }
            ]
        },
        "commentary": [],
        "keyEvents": [
            {
                "typeId": "goal",
                "text": "Goal scored",
                "minute": 15,
                "period": {"number": 1},
                "team": {"displayName": "Home FC"},
            },
            {
                "typeId": "goal",
                "text": "Goal scored",
                "minute": 55,
                "period": {"number": 2},
                "team": {"displayName": "Away Utd"},
            },
        ],
    }


@pytest.fixture
def mock_core_plays() -> Dict[str, Any]:
    """Mock Core API plays response."""
    return {
        "plays": [
            {
                "type": {"text": "Kickoff"},
                "text": "Match starts",
                "period": {"number": 1},
                "clock": {"displayTime": "0'"},
                "team": None,
                "athletes": [],
            },
            {
                "type": {"text": "Goal"},
                "text": "Goal scored by Player X",
                "period": {"number": 1},
                "clock": {"displayTime": "23'"},
                "team": {"displayName": "Team Alpha"},
                "athletes": [{"fullName": "Player X"}],
            },
        ]
    }


# =============================================================================
# Tests: Clock Parsing
# =============================================================================

class TestClockParsing:
    """Tests for _parse_clock_display function."""
    
    def test_parse_zero_minutes(self):
        assert _parse_clock_display("0'") == 0
    
    def test_parse_regular_minute(self):
        assert _parse_clock_display("45'") == 45
        assert _parse_clock_display("90'") == 90
    
    def test_parse_added_time(self):
        assert _parse_clock_display("45+2'") == 47
        assert _parse_clock_display("90+4'") == 94
        assert _parse_clock_display("90+10'") == 100
    
    def test_parse_halftime_fulltime(self):
        assert _parse_clock_display("HT") == 45
        assert _parse_clock_display("FT") == 90
    
    def test_parse_no_apostrophe(self):
        assert _parse_clock_display("45") == 45
        assert _parse_clock_display("0") == 0
    
    def test_parse_empty_or_none(self):
        assert _parse_clock_display("") == 0
        assert _parse_clock_display(None) == 0
    
    def test_parse_invalid(self):
        assert _parse_clock_display("invalid") == 0


# =============================================================================
# Tests: Commentary Event Normalization
# =============================================================================

class TestCommentaryNormalization:
    """Tests for _normalize_commentary_event function."""
    
    def test_normalize_kickoff(self):
        raw = {
            "type": {"name": "Kickoff"},
            "text": "First Half begins.",
            "clock": {"displayTime": "0'"},
            "period": {"number": 1},
            "team": None,
            "player": None,
        }
        result = _normalize_commentary_event(raw, 0, "760500")
        
        assert result["event_id"] == "760500"
        assert result["sequence_index"] == 0
        assert result["minute"] == 0
        assert result["event_type"] == "kickoff"
        assert result["source"] == "commentary"
        assert result["description"] == "First Half begins."
        assert result["team_name"] is None
        assert result["player_name"] is None
    
    def test_normalize_goal(self):
        raw = {
            "type": {"name": "Goal"},
            "text": "Goal! Argentina 1-0. Lionel Messi scores.",
            "clock": {"displayTime": "12'"},
            "period": {"number": 1},
            "team": {"displayName": "Argentina", "abbreviation": "ARG"},
            "player": {"fullName": "Lionel Messi"},
        }
        result = _normalize_commentary_event(raw, 5, "760500")
        
        assert result["minute"] == 12
        assert result["event_type"] == "goal"
        assert result["team_name"] == "Argentina"
        assert result["team_abbr"] == "ARG"
        assert result["player_name"] == "Lionel Messi"
    
    def test_normalize_yellow_card(self):
        raw = {
            "type": {"name": "Yellow Card"},
            "text": "Yellow card shown",
            "clock": {"displayTime": "30'"},
            "period": {"number": 1},
            "team": {"displayName": "Egypt", "abbreviation": "EGY"},
            "player": {"fullName": "Player Name"},
        }
        result = _normalize_commentary_event(raw, 10, "760500")
        
        assert result["event_type"] == "yellow_card"
        assert result["minute"] == 30
    
    def test_normalize_substitution(self):
        raw = {
            "type": {"name": "Substitution"},
            "text": "Substitution made",
            "clock": {"displayTime": "60'"},
            "period": {"number": 2},
            "team": {"displayName": "Team", "abbreviation": "TM"},
            "player": {"fullName": "Sub Player"},
        }
        result = _normalize_commentary_event(raw, 20, "760500")
        
        assert result["event_type"] == "substitution"
        assert result["period"] == 2


# =============================================================================
# Tests: Key Events Normalization
# =============================================================================

class TestKeyEventsNormalization:
    """Tests for _normalize_key_event function."""
    
    def test_normalize_goal_keyevent(self):
        raw = {
            "typeId": "goal",
            "text": "Goal scored",
            "minute": 25,
            "period": {"number": 1},
            "team": {"displayName": "Home Team", "abbreviation": "HOM"},
            "participants": [{"athlete": {"fullName": "Scorer"}}]
        }
        result = _normalize_key_event(raw, 0, "760500")
        
        assert result["event_type"] == "goal"
        assert result["minute"] == 25
        assert result["player_name"] == "Scorer"
        assert result["source"] == "keyEvents"
    
    def test_normalize_yellow_card_keyevent(self):
        raw = {
            "typeId": "yellow-card",
            "text": "Player booked",
            "minute": 40,
            "period": {"number": 1},
            "team": {"displayName": "Away", "abbreviation": "AWY"},
        }
        result = _normalize_key_event(raw, 1, "760500")
        
        assert result["event_type"] == "yellow_card"
        assert result["minute"] == 40
    
    def test_normalize_numeric_typeid(self):
        raw = {
            "typeId": 123,
            "text": "Some event",
            "minute": 10,
            "period": {"number": 1},
        }
        result = _normalize_key_event(raw, 0, "760500")
        
        # Should not crash, should default to unknown or infer from description
        assert result["minute"] == 10


# =============================================================================
# Tests: Extract Functions
# =============================================================================

class TestExtractFunctions:
    """Tests for extract_commentary_events and extract_key_events."""
    
    def test_extract_commentary_events(self, mock_summary_with_commentary):
        events = extract_commentary_events(mock_summary_with_commentary)
        
        assert len(events) == 8
        assert events[0]["event_type"] == "kickoff"
        assert events[-1]["event_type"] == "fulltime"
    
    def test_extract_commentary_events_empty(self, mock_summary_empty_commentary):
        events = extract_commentary_events(mock_summary_empty_commentary)
        assert len(events) == 0
    
    def test_extract_commentary_events_missing(self, mock_summary_missing_commentary):
        events = extract_commentary_events(mock_summary_missing_commentary)
        assert len(events) == 0
    
    def test_extract_key_events(self, mock_summary_with_commentary):
        events = extract_key_events(mock_summary_with_commentary)
        
        assert len(events) == 3
        assert all(e["source"] == "keyEvents" for e in events)
    
    def test_extract_key_events_empty(self, mock_summary_empty_commentary):
        events = extract_key_events(mock_summary_empty_commentary)
        assert len(events) == 0


# =============================================================================
# Tests: Core Plays Extraction
# =============================================================================

class TestCorePlaysExtraction:
    """Tests for extract_core_plays_events."""
    
    def test_extract_core_plays(self, mock_core_plays):
        events = extract_core_plays_events(mock_core_plays, "760500")
        
        assert len(events) == 2
        assert events[0]["event_type"] == "unknown"  # Kickoff not mapped in core plays
        assert events[1]["event_type"] == "goal"
        assert events[1]["source"] == "core_plays"
    
    def test_extract_core_plays_empty(self):
        events = extract_core_plays_events({}, "760500")
        assert len(events) == 0
        
        events = extract_core_plays_events({"other": "data"}, "760500")
        assert len(events) == 0


# =============================================================================
# Tests: Merge Events
# =============================================================================

class TestMergeEvents:
    """Tests for _merge_events function."""
    
    def test_merge_with_no_keyevents(self):
        commentary = [{"minute": 10, "event_type": "goal", "team_abbr": "ARG", "sequence_index": 0}]
        result = _merge_events(commentary, [])
        assert len(result) == 1
    
    def test_merge_with_no_commentary(self):
        key_events = [{"minute": 20, "event_type": "goal", "team_abbr": "EGY", "sequence_index": 0}]
        result = _merge_events([], key_events)
        assert len(result) == 1
    
    def test_merge_deduplicates(self):
        commentary = [
            {"minute": 10, "event_type": "goal", "team_abbr": "ARG", "sequence_index": 0}
        ]
        key_events = [
            {"minute": 10, "event_type": "goal", "team_abbr": "ARG", "sequence_index": 0}
        ]
        result = _merge_events(commentary, key_events)
        
        # Should deduplicate based on signature
        assert len(result) == 1
    
    def test_merge_adds_unique_keyevents(self):
        commentary = [
            {"minute": 10, "event_type": "goal", "team_abbr": "ARG", "sequence_index": 0}
        ]
        key_events = [
            {"minute": 10, "event_type": "goal", "team_abbr": "ARG", "sequence_index": 0},
            {"minute": 45, "event_type": "halftime", "team_abbr": None, "sequence_index": 1}
        ]
        result = _merge_events(commentary, key_events)
        
        # Should have kickoff + halftime
        assert len(result) == 2


# =============================================================================
# Tests: Normalize Match Events
# =============================================================================

class TestNormalizeMatchEvents:
    """Tests for normalize_match_events function."""
    
    def test_normalize_prefers_commentary(self, mock_summary_with_commentary):
        events, meta = normalize_match_events(mock_summary_with_commentary, "commentary")
        
        assert meta["commentary_available"] is True
        assert meta["commentary_count"] == 8
        assert len(events) >= 8  # May include merged keyEvents
    
    def test_normalize_auto_fallback_to_keyevents(self, mock_summary_only_keyevents):
        events, meta = normalize_match_events(mock_summary_only_keyevents, "auto")
        
        assert meta["used_source"] == "keyEvents_fallback"
        assert len(events) == 2
    
    def test_normalize_keyevents_explicit(self, mock_summary_with_commentary):
        events, meta = normalize_match_events(mock_summary_with_commentary, "keyEvents")
        
        assert meta["used_source"] == "keyEvents"
        assert len(events) == 3
    
    def test_normalize_empty_summary(self, mock_summary_empty_commentary):
        events, meta = normalize_match_events(mock_summary_empty_commentary, "auto")
        
        assert len(events) == 0
        assert meta["total_events"] == 0


# =============================================================================
# Tests: Filter Events By Type
# =============================================================================

class TestFilterEventsByType:
    """Tests for filter_events_by_type function."""
    
    def test_filter_goals_only(self, mock_summary_with_commentary):
        events, _ = normalize_match_events(mock_summary_with_commentary, "commentary")
        filtered = filter_events_by_type(events, ["goal"])
        
        assert len(filtered) == 2
        assert all(e["event_type"] == "goal" for e in filtered)
    
    def test_filter_cards_only(self, mock_summary_with_commentary):
        events, _ = normalize_match_events(mock_summary_with_commentary, "commentary")
        filtered = filter_events_by_type(events, ["yellow_card", "red_card"])
        
        assert len(filtered) == 1
        assert filtered[0]["event_type"] == "yellow_card"
    
    def test_filter_multiple_types(self, mock_summary_with_commentary):
        events, _ = normalize_match_events(mock_summary_with_commentary, "commentary")
        filtered = filter_events_by_type(events, ["goal", "substitution"])
        
        # 2 goals + 1 substitution
        assert len(filtered) == 3
    
    def test_filter_empty_list(self):
        filtered = filter_events_by_type([], ["goal"])
        assert len(filtered) == 0


# =============================================================================
# Tests: Get Match Event Timeline
# =============================================================================

class TestGetMatchEventTimeline:
    """Tests for get_match_event_timeline function."""
    
    def test_timeline_structure(self, mock_summary_with_commentary, monkeypatch):
        """Test that timeline has correct structure."""
        # Mock fetch_match_summary to return our mock
        monkeypatch.setattr(
            "src.data.espn_match_events.fetch_match_summary",
            lambda event_id, league: mock_summary_with_commentary
        )
        
        timeline = get_match_event_timeline("760500", "fifa.world", "commentary")
        
        # Check top-level keys
        assert "event_id" in timeline
        assert "league" in timeline
        assert "match" in timeline
        assert "sources" in timeline
        assert "events" in timeline
        
        # Check match info
        match_info = timeline["match"]
        assert match_info["short_name"] == "Argentina vs Cape Verde"
        assert match_info["home_team"] == "Argentina"
        assert match_info["away_team"] == "Cape Verde"
        assert match_info["home_score"] == 2.0
        assert match_info["away_score"] == 0.0
        
        # Check sources metadata
        sources = timeline["sources"]
        assert sources["commentary_available"] is True
        assert sources["commentary_count"] == 8
        assert sources["total_events"] > 0
        
        # Check events are sorted
        events = timeline["events"]
        minutes = [e["minute"] for e in events]
        assert minutes == sorted(minutes)
    
    def test_timeline_with_missing_data(self, mock_summary_missing_commentary, monkeypatch):
        """Test timeline handles missing data gracefully."""
        monkeypatch.setattr(
            "src.data.espn_match_events.fetch_match_summary",
            lambda event_id, league: mock_summary_missing_commentary
        )
        
        timeline = get_match_event_timeline("760502", "fifa.world")
        
        # Should not crash, should have values (teams exist in mock)
        assert timeline["event_id"] == "760502"
        assert timeline["match"]["home_team"] == "Team X"
        assert timeline["match"]["away_team"] == "Team Y"
        # Scores should be None for scheduled matches
        assert timeline["match"]["home_score"] is None
        assert timeline["match"]["away_score"] is None


# =============================================================================
# Tests: Temporal Ordering
# =============================================================================

class TestTemporalOrdering:
    """Tests for correct temporal ordering of events."""
    
    def test_kickoff_before_all(self, mock_summary_with_commentary):
        events = extract_commentary_events(mock_summary_with_commentary)
        
        # First event should be kickoff
        assert events[0]["event_type"] == "kickoff"
        assert events[0]["minute"] == 0
    
    def test_halftime_after_first_half_events(self, mock_summary_with_commentary):
        events = extract_commentary_events(mock_summary_with_commentary)
        
        halftime_events = [e for e in events if e["event_type"] == "halftime"]
        first_half_events = [e for e in events if e["period"] == 1 and e["event_type"] not in ["kickoff", "halftime"]]
        
        if halftime_events and first_half_events:
            halftime_minute = halftime_events[0]["minute"]
            max_first_half_minute = max(e["minute"] for e in first_half_events)
            assert halftime_minute >= max_first_half_minute
    
    def test_fulltime_at_end(self, mock_summary_with_commentary):
        events = extract_commentary_events(mock_summary_with_commentary)
        
        # Last event should be fulltime or close to it
        last_event = events[-1]
        assert last_event["event_type"] == "fulltime"
        assert last_event["minute"] >= 90


# =============================================================================
# Tests: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_fetch_summary_empty_event_id(self):
        with pytest.raises(ValueError, match="Event ID cannot be empty"):
            fetch_match_summary("", "fifa.world")
        
        with pytest.raises(ValueError, match="Event ID cannot be empty"):
            fetch_match_summary("   ", "fifa.world")
    
    def test_normalize_event_with_missing_fields(self):
        """Test normalization handles missing fields gracefully."""
        raw = {
            "type": {},
            "clock": {},
            "period": {},
        }
        result = _normalize_commentary_event(raw, 0, "760500")
        
        assert result["event_type"] == "unknown"
        assert result["minute"] == 0
        assert result["period"] == 1
        assert result["team_name"] is None
        assert result["player_name"] is None
    
    def test_keyevent_inference_from_description(self):
        """Test that keyEvents can infer type from description."""
        raw = {
            "typeId": "unknown-type",
            "text": "Yellow card shown to player",
            "minute": 30,
            "period": {"number": 1},
        }
        result = _normalize_key_event(raw, 0, "760500")
        
        # Should infer yellow_card from description
        assert result["event_type"] == "yellow_card"
    
    def test_scores_as_floats_in_timeline(self, mock_summary_with_commentary, monkeypatch):
        """Test that scores are properly converted to floats."""
        monkeypatch.setattr(
            "src.data.espn_match_events.fetch_match_summary",
            lambda event_id, league: mock_summary_with_commentary
        )
        
        timeline = get_match_event_timeline("760500")
        
        # Scores should be floats
        assert isinstance(timeline["match"]["home_score"], float)
        assert isinstance(timeline["match"]["away_score"], float)


# =============================================================================
# Tests: JSON Serialization
# =============================================================================

class TestJsonSerialization:
    """Tests for JSON serializability of outputs."""
    
    def test_timeline_is_json_serializable(self, mock_summary_with_commentary, monkeypatch):
        """Test that timeline output is JSON serializable."""
        monkeypatch.setattr(
            "src.data.espn_match_events.fetch_match_summary",
            lambda event_id, league: mock_summary_with_commentary
        )
        
        timeline = get_match_event_timeline("760500")
        
        # Should not raise
        json_str = json.dumps(timeline)
        assert len(json_str) > 0
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed["event_id"] == "760500"
    
    def test_events_have_raw_event_preserved(self, mock_summary_with_commentary):
        """Test that raw_event is preserved in normalized events."""
        events = extract_commentary_events(mock_summary_with_commentary)
        
        for evt in events:
            assert "raw_event" in evt
            assert evt["raw_event"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
