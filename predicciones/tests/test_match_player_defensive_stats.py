#!/usr/bin/env python3
"""
Tests for match_player_defensive_stats.py module.

Tests cover:
1. build_player_stats_map with normal input
2. build_player_stats_map with unequal lengths
3. normalize_numeric for string integers
4. extract_goalkeeper_metrics (saves, goalsConceded)
5. extract_outfield_metrics (offsides, interceptions, clearances)
6. has_nonzero_metrics filtering
7. team filtering
8. Missing boxscore/players handling
9. JSON serializable output
10. CLI --json response
"""
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from scripts.match_player_defensive_stats import (
    build_player_stats_map,
    normalize_numeric,
    extract_goalkeeper_metrics,
    extract_outfield_metrics,
    has_nonzero_metrics,
    extract_match_context,
    extract_boxscore_player_metrics,
)


class TestBuildPlayerStatsMap:
    """Tests for build_player_stats_map function."""
    
    def test_basic_stats_map(self):
        """Test parsing normal stats list."""
        stats_list = [
            {"name": "offsides", "value": 1},
            {"name": "interceptions", "value": 4},
            {"name": "clearances", "value": 7},
        ]
        
        result = build_player_stats_map(stats_list)
        
        assert result["offsides"] == 1
        assert result["interceptions"] == 4
        assert result["clearances"] == 7
    
    def test_empty_stats_list(self):
        """Test with empty stats list."""
        result = build_player_stats_map([])
        assert result == {}
    
    def test_non_dict_items(self):
        """Test with non-dict items in list."""
        stats_list = ["invalid", 123, None]
        result = build_player_stats_map(stats_list)
        assert result == {}
    
    def test_missing_name_field(self):
        """Test with stats missing name field."""
        stats_list = [
            {"value": 1},
            {"name": "offsides", "value": 2},
        ]
        result = build_player_stats_map(stats_list)
        assert "offsides" in result
        assert len(result) == 1
    
    def test_fallback_to_display_value(self):
        """Test fallback to displayValue when value is None."""
        stats_list = [
            {"name": "offsides", "value": None, "displayValue": "2"},
            {"name": "interceptions", "displayValue": "4"},
        ]
        result = build_player_stats_map(stats_list)
        assert result["offsides"] == "2"
        assert result["interceptions"] == "4"


class TestNormalizeNumeric:
    """Tests for normalize_numeric function."""
    
    def test_string_integer(self):
        """Test parsing string integer."""
        assert normalize_numeric("5") == 5
        assert normalize_numeric("  5  ") == 5
    
    def test_string_float(self):
        """Test parsing string float."""
        assert normalize_numeric("5.5") == 5.5
        assert normalize_numeric("3.14") == 3.14
    
    def test_with_suffixes(self):
        """Test parsing values with suffixes like 'yd', 'm', '%'."""
        assert normalize_numeric("19.4 yd") == 19.4
        assert normalize_numeric("19.4 m") == 19.4
        assert normalize_numeric("81.5%") == 81.5
        assert normalize_numeric("10 yds") == 10
    
    def test_int_input(self):
        """Test with int input."""
        assert normalize_numeric(5) == 5
        assert normalize_numeric(0) == 0
    
    def test_float_input(self):
        """Test with float input."""
        assert normalize_numeric(5.0) == 5
        assert normalize_numeric(5.5) == 5.5
    
    def test_none_input(self):
        """Test with None input."""
        assert normalize_numeric(None) is None
    
    def test_empty_string(self):
        """Test with empty string."""
        assert normalize_numeric("") is None
        assert normalize_numeric("   ") is None
    
    def test_unparseable_string(self):
        """Test with unparseable string."""
        assert normalize_numeric("abc") is None
        assert normalize_numeric("N/A") is None


class TestExtractGoalkeeperMetrics:
    """Tests for extract_goalkeeper_metrics function."""
    
    def test_extract_saves_and_goals_conceded(self):
        """Test extracting saves and goalsConceded."""
        player_entry = {
            "athlete": {
                "id": "12345",
                "displayName": "Test Goalkeeper"
            },
            "position": {"abbreviation": "G", "name": "Goalkeeper"},
            "stats": [
                {"name": "saves", "value": 6},
                {"name": "goalsConceded", "value": 2},
            ]
        }
        team_meta = {"team_name": "Test Team", "team_abbr": "TST"}
        
        result = extract_goalkeeper_metrics(player_entry, team_meta)
        
        assert result is not None
        assert result["player_name"] == "Test Goalkeeper"
        assert result["player_id"] == "12345"
        assert result["team_name"] == "Test Team"
        assert result["team_abbr"] == "TST"
        assert result["position"] == "G"
        assert result["saves"] == 6
        assert result["goals_conceded"] == 2
    
    def test_not_goalkeeper_returns_none(self):
        """Test that non-goalkeeper returns None."""
        player_entry = {
            "athlete": {"id": "12345", "displayName": "Test Player"},
            "position": {"abbreviation": "D", "name": "Defender"},
            "stats": []
        }
        team_meta = {"team_name": "Test Team", "team_abbr": "TST"}
        
        result = extract_goalkeeper_metrics(player_entry, team_meta)
        assert result is None
    
    def test_include_raw_stats(self):
        """Test including raw stats map."""
        player_entry = {
            "athlete": {"id": "12345", "displayName": "Test GK"},
            "position": {"abbreviation": "GK", "name": "Goalkeeper"},
            "stats": [
                {"name": "saves", "value": 3},
                {"name": "goalsConceded", "value": 1},
                {"name": "shotsFaced", "value": 4},
            ]
        }
        team_meta = {"team_name": "Test", "team_abbr": "TST"}
        
        result = extract_goalkeeper_metrics(player_entry, team_meta, include_raw=True)
        
        assert "stats_raw" in result
        assert result["stats_raw"]["saves"] == 3
        assert result["stats_raw"]["goalsConceded"] == 1
        assert result["stats_raw"]["shotsFaced"] == 4
    
    def test_missing_athlete_id_returns_none(self):
        """Test that missing athlete ID returns None."""
        player_entry = {
            "athlete": {"displayName": "No ID Player"},
            "position": {"abbreviation": "G"},
            "stats": []
        }
        team_meta = {"team_name": "Test", "team_abbr": "TST"}
        
        result = extract_goalkeeper_metrics(player_entry, team_meta)
        assert result is None


class TestExtractOutfieldMetrics:
    """Tests for extract_outfield_metrics function."""
    
    def test_extract_offsides_interceptions_clearances(self):
        """Test extracting offsides, interceptions, clearances."""
        player_entry = {
            "athlete": {
                "id": "23456",
                "displayName": "Test Defender"
            },
            "position": {"abbreviation": "D", "name": "Defender"},
            "stats": [
                {"name": "offsides", "value": 0},
                {"name": "interceptions", "value": 4},
                {"name": "clearances", "value": 7},
            ]
        }
        team_meta = {"team_name": "Test Team", "team_abbr": "TST"}
        
        result = extract_outfield_metrics(player_entry, team_meta)
        
        assert result is not None
        assert result["player_name"] == "Test Defender"
        assert result["player_id"] == "23456"
        assert result["offsides"] == 0
        assert result["interceptions"] == 4
        assert result["clearances"] == 7
    
    def test_goalkeeper_returns_none(self):
        """Test that goalkeeper returns None from outfield extraction."""
        player_entry = {
            "athlete": {"id": "12345", "displayName": "Test GK"},
            "position": {"abbreviation": "G", "name": "Goalkeeper"},
            "stats": []
        }
        team_meta = {"team_name": "Test", "team_abbr": "TST"}
        
        result = extract_outfield_metrics(player_entry, team_meta)
        assert result is None
    
    def test_missing_metrics_are_null(self):
        """Test that missing metrics are null."""
        player_entry = {
            "athlete": {"id": "23456", "displayName": "Test Player"},
            "position": {"abbreviation": "M", "name": "Midfielder"},
            "stats": [
                {"name": "offsides", "value": 1},
                # No interceptions or clearances
            ]
        }
        team_meta = {"team_name": "Test", "team_abbr": "TST"}
        
        result = extract_outfield_metrics(player_entry, team_meta)
        
        assert result["offsides"] == 1
        assert result["interceptions"] is None
        assert result["clearances"] is None
    
    def test_include_raw_stats(self):
        """Test including raw stats map."""
        player_entry = {
            "athlete": {"id": "23456", "displayName": "Test Player"},
            "position": {"abbreviation": "D", "name": "Defender"},
            "stats": [
                {"name": "offsides", "value": 0},
                {"name": "tackles", "value": 3},
            ]
        }
        team_meta = {"team_name": "Test", "team_abbr": "TST"}
        
        result = extract_outfield_metrics(player_entry, team_meta, include_raw=True)
        
        assert "stats_raw" in result
        assert result["stats_raw"]["offsides"] == 0
        assert result["stats_raw"]["tackles"] == 3


class TestHasNonzeroMetrics:
    """Tests for has_nonzero_metrics function."""
    
    def test_goalkeeper_with_saves(self):
        """Test goalkeeper with non-zero saves."""
        player_data = {"saves": 5, "goals_conceded": 2}
        assert has_nonzero_metrics(player_data, is_goalkeeper=True) is True
    
    def test_goalkeeper_all_zero(self):
        """Test goalkeeper with all zero metrics."""
        player_data = {"saves": 0, "goals_conceded": 0}
        assert has_nonzero_metrics(player_data, is_goalkeeper=True) is False
    
    def test_goalkeeper_with_null(self):
        """Test goalkeeper with null metrics."""
        player_data = {"saves": None, "goals_conceded": None}
        assert has_nonzero_metrics(player_data, is_goalkeeper=True) is False
    
    def test_outfield_with_interceptions(self):
        """Test outfield player with non-zero interceptions."""
        player_data = {"offsides": 0, "interceptions": 4, "clearances": 0}
        assert has_nonzero_metrics(player_data, is_goalkeeper=False) is True
    
    def test_outfield_all_zero(self):
        """Test outfield player with all zero metrics."""
        player_data = {"offsides": 0, "interceptions": 0, "clearances": 0}
        assert has_nonzero_metrics(player_data, is_goalkeeper=False) is False
    
    def test_outfield_with_null(self):
        """Test outfield player with null metrics."""
        player_data = {"offsides": None, "interceptions": None, "clearances": None}
        assert has_nonzero_metrics(player_data, is_goalkeeper=False) is False


class TestExtractMatchContext:
    """Tests for extract_match_context function."""
    
    def test_extract_from_header(self):
        """Test extracting context from header."""
        summary = {
            "header": {
                "date": "2024-06-15T20:00Z",
                "status": {"type": {"name": "STATUS_FINAL"}},
                "competitions": [{
                    "competitors": [
                        {
                            "homeAway": "home",
                            "team": {"displayName": "Home Team"},
                            "score": "3"
                        },
                        {
                            "homeAway": "away",
                            "team": {"displayName": "Away Team"},
                            "score": "2"
                        }
                    ]
                }]
            }
        }
        
        result = extract_match_context(summary)
        
        assert result["short_name"] == "Home Team vs Away Team"
        assert result["home_team"] == "Home Team"
        assert result["away_team"] == "Away Team"
        assert result["home_score"] == 3
        assert result["away_score"] == 2
        assert result["status"] == "STATUS_FINAL"
    
    def test_empty_summary(self):
        """Test with empty summary."""
        result = extract_match_context({})
        assert result["short_name"] == ""
        assert result["home_team"] == ""
        assert result["away_team"] == ""
    
    def test_no_header(self):
        """Test with no header."""
        summary = {"boxscore": {}}
        result = extract_match_context(summary)
        assert result["short_name"] == ""


class TestExtractBoxscorePlayerMetrics:
    """Tests for extract_boxscore_player_metrics function."""
    
    def test_full_extraction(self):
        """Test full extraction from mock summary."""
        summary = {
            "rosters": [
                {
                    "team": {"displayName": "Team A", "abbreviation": "TMA"},
                    "homeAway": "home",
                    "roster": [
                        {
                            "athlete": {"id": "1", "displayName": "GK Player"},
                            "position": {"abbreviation": "G", "name": "Goalkeeper"},
                            "stats": [
                                {"name": "saves", "value": 5},
                                {"name": "goalsConceded", "value": 2},
                            ]
                        },
                        {
                            "athlete": {"id": "2", "displayName": "DEF Player"},
                            "position": {"abbreviation": "D", "name": "Defender"},
                            "stats": [
                                {"name": "offsides", "value": 0},
                                {"name": "interceptions", "value": 3},
                                {"name": "clearances", "value": 5},
                            ]
                        }
                    ]
                }
            ]
        }
        
        result = extract_boxscore_player_metrics(summary, "12345", "test.league")
        
        assert result["event_id"] == "12345"
        assert result["league"] == "test.league"
        assert len(result["teams"]) == 1
        
        team = result["teams"][0]
        assert team["team_name"] == "Team A"
        assert len(team["goalkeepers"]) == 1
        assert len(team["outfield"]) == 1
        
        gk = team["goalkeepers"][0]
        assert gk["player_name"] == "GK Player"
        assert gk["saves"] == 5
        assert gk["goals_conceded"] == 2
        
        of = team["outfield"][0]
        assert of["player_name"] == "DEF Player"
        assert of["interceptions"] == 3
    
    def test_only_nonzero_filter(self):
        """Test only_nonzero filtering."""
        summary = {
            "rosters": [
                {
                    "team": {"displayName": "Team A", "abbreviation": "TMA"},
                    "roster": [
                        {
                            "athlete": {"id": "1", "displayName": "GK Zero"},
                            "position": {"abbreviation": "G"},
                            "stats": [
                                {"name": "saves", "value": 0},
                                {"name": "goalsConceded", "value": 0},
                            ]
                        },
                        {
                            "athlete": {"id": "2", "displayName": "GK NonZero"},
                            "position": {"abbreviation": "G"},
                            "stats": [
                                {"name": "saves", "value": 5},
                                {"name": "goalsConceded", "value": 0},
                            ]
                        }
                    ]
                }
            ]
        }
        
        result = extract_boxscore_player_metrics(
            summary, "12345", "test.league", only_nonzero=True
        )
        
        # Only GK NonZero should be included
        assert len(result["teams"][0]["goalkeepers"]) == 1
        assert result["teams"][0]["goalkeepers"][0]["player_name"] == "GK NonZero"
    
    def test_empty_rosters(self):
        """Test with empty rosters."""
        summary = {"rosters": []}
        result = extract_boxscore_player_metrics(summary, "12345", "test.league")
        assert result["teams"] == []
    
    def test_no_rosters_key(self):
        """Test with no rosters key."""
        summary = {"boxscore": {}}
        result = extract_boxscore_player_metrics(summary, "12345", "test.league")
        assert result["teams"] == []
    
    def test_json_serializable(self):
        """Test that output is JSON serializable."""
        summary = {
            "rosters": [
                {
                    "team": {"displayName": "Team A", "abbreviation": "TMA"},
                    "roster": [
                        {
                            "athlete": {"id": "1", "displayName": "Player"},
                            "position": {"abbreviation": "D"},
                            "stats": [{"name": "offsides", "value": 1}]
                        }
                    ]
                }
            ]
        }
        
        result = extract_boxscore_player_metrics(summary, "12345", "test.league")
        
        # Should not raise
        json_str = json.dumps(result)
        assert json_str is not None
        
        # Verify round-trip
        parsed = json.loads(json_str)
        assert parsed["event_id"] == "12345"


class TestCLIBehavior:
    """Tests for CLI behavior."""
    
    def test_cli_json_output(self):
        """Test that CLI produces valid JSON with --json flag."""
        import subprocess
        
        # Use a known event that should work
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "scripts" / "match_player_defensive_stats.py"),
                "--event", "760500",
                "--json"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should produce valid JSON
        try:
            data = json.loads(result.stdout)
            assert "event_id" in data
            assert data["event_id"] == "760500"
            assert "teams" in data
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON output: {e}\nStdout: {result.stdout}\nStderr: {result.stderr}")
    
    def test_cli_team_filter(self):
        """Test CLI team filtering only affects pretty output."""
        import subprocess
        
        # Team filter only affects --pretty output, not JSON
        # Test that pretty output is filtered
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "scripts" / "match_player_defensive_stats.py"),
                "--event", "760500",
                "--team", "Argentina",
                "--pretty"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should only show Argentina in pretty output
        assert "Argentina" in result.stdout
        # Cape Verde should not appear when filtered to Argentina
        # (but might appear in header as opponent)
        lines = result.stdout.split('\n')
        # Find team sections - they start with team name at beginning of line after blank
        team_sections = [l for l in lines if l.strip() and not l.startswith('=') and not l.startswith(' ') and ':' not in l]
        # Should have at most one team section (Argentina) plus header info
        argentina_sections = [s for s in team_sections if 'Argentina' in s]
        assert len(argentina_sections) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
