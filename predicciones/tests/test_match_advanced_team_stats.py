"""
Tests for match_advanced_team_stats.py CLI script.

Tests cover:
1. Parsing fraction stats like "412/505"
2. Parsing crosses like "7/22"
3. Parsing percentages like "81.5%", "81.5", 81.5, 0.815
4. Parsing floats like "19.4", "19.4 yd", "19.4 m"
5. Extracting metrics from a mock valid summary
6. Handling summary without attacks
7. Handling summary without crossPercentage
8. Handling summary without averageShotDistance
9. JSON serializable output
10. CLI responds correctly with --json
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.match_advanced_team_stats import (
    fetch_match_summary,
    extract_match_context,
    build_stats_map,
    parse_fraction_stat,
    parse_percentage,
    parse_float,
    extract_advanced_team_metrics,
    extract_advanced_match_team_stats,
    print_advanced_team_stats,
    save_report,
)
from src.domain.exceptions import EspnApiError


# ==================================================
# Test parse_fraction_stat
# ==================================================
class TestParseFractionStat:
    def test_parse_standard_fraction(self):
        """Test parsing standard fraction like '412/505'."""
        completed, attempted = parse_fraction_stat("412/505")
        assert completed == 412
        assert attempted == 505

    def test_parse_crosses_format(self):
        """Test parsing crosses format like '7/22'."""
        completed, attempted = parse_fraction_stat("7/22")
        assert completed == 7
        assert attempted == 22

    def test_parse_with_spaces(self):
        """Test parsing with spaces around slash."""
        completed, attempted = parse_fraction_stat(" 412 / 505 ")
        assert completed == 412
        assert attempted == 505

    def test_parse_single_number(self):
        """Test parsing single number."""
        completed, attempted = parse_fraction_stat("100")
        assert completed == 100
        assert attempted is None

    def test_parse_integer_input(self):
        """Test parsing integer input."""
        completed, attempted = parse_fraction_stat(100)
        assert completed == 100
        assert attempted is None

    def test_parse_float_input(self):
        """Test parsing float input."""
        completed, attempted = parse_fraction_stat(100.5)
        assert completed == 100
        assert attempted is None

    def test_parse_none_input(self):
        """Test parsing None input."""
        completed, attempted = parse_fraction_stat(None)
        assert completed is None
        assert attempted is None

    def test_parse_invalid_string(self):
        """Test parsing invalid string."""
        completed, attempted = parse_fraction_stat("invalid")
        assert completed is None
        assert attempted is None

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        completed, attempted = parse_fraction_stat("")
        assert completed is None
        assert attempted is None


# ==================================================
# Test parse_percentage
# ==================================================
class TestParsePercentage:
    def test_parse_percentage_with_symbol(self):
        """Test parsing percentage with % symbol."""
        result = parse_percentage("81.5%")
        assert result == 81.5

    def test_parse_percentage_without_symbol(self):
        """Test parsing percentage without % symbol."""
        result = parse_percentage("81.5")
        assert result == 81.5

    def test_parse_percentage_as_float(self):
        """Test parsing percentage as float value."""
        result = parse_percentage(81.5)
        assert result == 81.5

    def test_parse_decimal_percentage(self):
        """Test parsing decimal percentage (0-1 scale)."""
        result = parse_percentage(0.815)
        assert result == 81.5

    def test_parse_decimal_string(self):
        """Test parsing decimal string (0-1 scale)."""
        result = parse_percentage("0.815")
        assert result == 81.5

    def test_parse_none_input(self):
        """Test parsing None input."""
        result = parse_percentage(None)
        assert result is None

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = parse_percentage("")
        assert result is None

    def test_parse_invalid_string(self):
        """Test parsing invalid string."""
        result = parse_percentage("invalid")
        assert result is None

    def test_parse_zero(self):
        """Test parsing zero."""
        result = parse_percentage("0")
        assert result == 0.0

    def test_parse_hundred(self):
        """Test parsing 100%."""
        result = parse_percentage("100%")
        assert result == 100.0


# ==================================================
# Test parse_float
# ==================================================
class TestParseFloat:
    def test_parse_simple_float(self):
        """Test parsing simple float string."""
        result = parse_float("19.4")
        assert result == 19.4

    def test_parse_float_with_yards(self):
        """Test parsing float with yards suffix."""
        result = parse_float("19.4 yd")
        assert result == 19.4

    def test_parse_float_with_meters(self):
        """Test parsing float with meters suffix."""
        result = parse_float("19.4 m")
        assert result == 19.4

    def test_parse_float_as_number(self):
        """Test parsing float as numeric value."""
        result = parse_float(19.4)
        assert result == 19.4

    def test_parse_integer_as_float(self):
        """Test parsing integer as float."""
        result = parse_float(19)
        assert result == 19.0

    def test_parse_none_input(self):
        """Test parsing None input."""
        result = parse_float(None)
        assert result is None

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = parse_float("")
        assert result is None

    def test_parse_negative_float(self):
        """Test parsing negative float."""
        result = parse_float("-19.4")
        assert result == -19.4

    def test_parse_invalid_string(self):
        """Test parsing invalid string."""
        result = parse_float("invalid")
        assert result is None


# ==================================================
# Test build_stats_map
# ==================================================
class TestBuildStatsMap:
    def test_build_stats_map_from_list(self):
        """Test building stats map from list of stat dicts."""
        stats_list = [
            {"name": "totalPasses", "displayValue": "505"},
            {"name": "accuratePasses", "displayValue": "412"},
            {"name": "passPct", "displayValue": "81.5"},
        ]
        stats_map = build_stats_map(stats_list)
        
        assert len(stats_map) == 3
        assert "totalpasses" in stats_map
        assert "accuratepasses" in stats_map
        assert "passpct" in stats_map
        assert stats_map["totalpasses"]["displayValue"] == "505"

    def test_build_stats_map_empty(self):
        """Test building stats map from empty list."""
        stats_map = build_stats_map([])
        assert len(stats_map) == 0

    def test_build_stats_map_none(self):
        """Test building stats map from None."""
        stats_map = build_stats_map(None)
        assert len(stats_map) == 0


# ==================================================
# Test extract_match_context
# ==================================================
class TestExtractMatchContext:
    @pytest.fixture
    def mock_summary_complete(self):
        """Complete mock summary."""
        return {
            "header": {
                "id": "760500",
                "date": "2026-07-03T22:00Z",
                "status": {
                    "type": {
                        "name": "STATUS_FINAL_AET",
                        "state": "post"
                    }
                },
                "competitions": [
                    {
                        "competitors": [
                            {
                                "homeAway": "home",
                                "team": {
                                    "displayName": "Argentina",
                                    "name": "Argentina"
                                },
                                "score": "3"
                            },
                            {
                                "homeAway": "away",
                                "team": {
                                    "displayName": "Cape Verde",
                                    "name": "Cape Verde"
                                },
                                "score": "2"
                            }
                        ]
                    }
                ]
            },
            "boxscore": {
                "teams": []
            }
        }

    def test_extract_context_complete(self, mock_summary_complete):
        """Test extracting context from complete summary."""
        context = extract_match_context(mock_summary_complete)
        
        assert context["short_name"] == "Argentina vs Cape Verde"
        assert context["date"] == "2026-07-03T22:00Z"
        assert context["status"] == "STATUS_FINAL_AET"
        assert context["home_team"] == "Argentina"
        assert context["away_team"] == "Cape Verde"
        assert context["home_score"] == 3
        assert context["away_score"] == 2

    def test_extract_context_empty(self):
        """Test extracting context from empty summary."""
        context = extract_match_context({})
        
        assert context["short_name"] is None
        assert context["date"] is None
        assert context["status"] is None
        assert context["home_team"] is None
        assert context["away_team"] is None


# ==================================================
# Test extract_advanced_team_metrics
# ==================================================
class TestExtractAdvancedTeamMetrics:
    @pytest.fixture
    def mock_team_block_complete(self):
        """Complete mock team block with all stats."""
        return {
            "team": {
                "displayName": "Argentina",
                "abbreviation": "ARG"
            },
            "statistics": [
                {"name": "accuratePasses", "displayValue": "412"},
                {"name": "totalPasses", "displayValue": "505"},
                {"name": "passPct", "displayValue": "81.5"},
                {"name": "attacks", "displayValue": "87"},
                {"name": "dangerousAttacks", "displayValue": "49"},
                {"name": "accurateCrosses", "displayValue": "7"},
                {"name": "totalCrosses", "displayValue": "22"},
                {"name": "crossPct", "displayValue": "31.8"},
                {"name": "shotsOnTarget", "displayValue": "7"},
                {"name": "shotsOffTarget", "displayValue": "6"},
                {"name": "blockedShots", "displayValue": "3"},
                {"name": "averageShotDistance", "displayValue": "19.4"},
                {"name": "hitWoodwork", "displayValue": "1"},
            ]
        }

    @pytest.fixture
    def mock_team_block_espn_format(self):
        """Mock team block with ESPN's actual format (separate accurate/total)."""
        return {
            "team": {
                "displayName": "Argentina",
                "abbreviation": "ARG"
            },
            "statistics": [
                {"name": "accuratePasses", "displayValue": "779"},
                {"name": "totalPasses", "displayValue": "850"},
                {"name": "passPct", "displayValue": "0.9"},
                {"name": "accurateCrosses", "displayValue": "6"},
                {"name": "totalCrosses", "displayValue": "15"},
                {"name": "crossPct", "displayValue": "0.4"},
                {"name": "shotsOnTarget", "displayValue": "10"},
                {"name": "blockedShots", "displayValue": "7"},
            ]
        }

    def test_extract_metrics_complete(self, mock_team_block_complete):
        """Test extracting metrics from complete team block."""
        metrics = extract_advanced_team_metrics(mock_team_block_complete)
        
        assert metrics["team_name"] == "Argentina"
        assert metrics["team_abbr"] == "ARG"
        
        passing = metrics["advanced_metrics"]["passing"]
        assert passing["passes_raw"] == "412/505"
        assert passing["passes_completed"] == 412
        assert passing["passes_attempted"] == 505
        assert passing["pass_percentage"] == 81.5
        
        attacks = metrics["advanced_metrics"]["attacks"]
        assert attacks["attacks"] == 87
        assert attacks["dangerous_attacks"] == 49
        assert attacks["crosses_raw"] == "7/22"
        assert attacks["crosses_completed"] == 7
        assert attacks["crosses_attempted"] == 22
        assert attacks["cross_percentage"] == 31.8
        
        shooting = metrics["advanced_metrics"]["shooting"]
        assert shooting["shots_on_target"] == 7
        assert shooting["shots_off_target"] == 6
        assert shooting["blocked_shots"] == 3
        assert shooting["average_shot_distance"] == 19.4
        assert shooting["hit_woodwork"] == 1

    def test_extract_metrics_espn_format(self, mock_team_block_espn_format):
        """Test extracting metrics from ESPN's actual format."""
        metrics = extract_advanced_team_metrics(mock_team_block_espn_format)
        
        passing = metrics["advanced_metrics"]["passing"]
        assert passing["passes_raw"] == "779/850"
        assert passing["passes_completed"] == 779
        assert passing["passes_attempted"] == 850
        assert passing["pass_percentage"] == 90.0
        
        attacks = metrics["advanced_metrics"]["attacks"]
        assert attacks["crosses_raw"] == "6/15"
        assert attacks["crosses_completed"] == 6
        assert attacks["crosses_attempted"] == 15
        assert attacks["cross_percentage"] == 40.0

    def test_extract_metrics_no_attacks(self):
        """Test extracting metrics when attacks are missing."""
        team_block = {
            "team": {"displayName": "Team A"},
            "statistics": [
                {"name": "accuratePasses", "displayValue": "100"},
                {"name": "totalPasses", "displayValue": "120"},
            ]
        }
        metrics = extract_advanced_team_metrics(team_block)
        
        attacks = metrics["advanced_metrics"]["attacks"]
        assert attacks["attacks"] is None
        assert attacks["dangerous_attacks"] is None

    def test_extract_metrics_no_cross_percentage(self):
        """Test extracting metrics when crossPercentage is missing."""
        team_block = {
            "team": {"displayName": "Team A"},
            "statistics": [
                {"name": "accurateCrosses", "displayValue": "5"},
                {"name": "totalCrosses", "displayValue": "15"},
            ]
        }
        metrics = extract_advanced_team_metrics(team_block)
        
        attacks = metrics["advanced_metrics"]["attacks"]
        assert attacks["crosses_completed"] == 5
        assert attacks["crosses_attempted"] == 15
        assert attacks["cross_percentage"] is None

    def test_extract_metrics_no_shot_distance(self):
        """Test extracting metrics when averageShotDistance is missing."""
        team_block = {
            "team": {"displayName": "Team A"},
            "statistics": [
                {"name": "shotsOnTarget", "displayValue": "5"},
            ]
        }
        metrics = extract_advanced_team_metrics(team_block)
        
        shooting = metrics["advanced_metrics"]["shooting"]
        assert shooting["shots_on_target"] == 5
        assert shooting["average_shot_distance"] is None

    def test_extract_metrics_has_stats_raw(self):
        """Test that stats_raw is populated."""
        team_block = {
            "team": {"displayName": "Team A"},
            "statistics": [
                {"name": "totalPasses", "displayValue": "100"},
                {"name": "possessionPct", "displayValue": "55.5"},
            ]
        }
        metrics = extract_advanced_team_metrics(team_block)
        
        assert "stats_raw" in metrics
        assert "totalpasses" in metrics["stats_raw"]
        assert metrics["stats_raw"]["totalpasses"] == "100"


# ==================================================
# Test extract_advanced_match_team_stats
# ==================================================
class TestExtractAdvancedMatchTeamStats:
    @pytest.fixture
    def mock_summary_for_extraction(self):
        """Mock summary for full extraction test."""
        return {
            "header": {
                "date": "2026-07-03T22:00Z",
                "status": {"type": {"name": "STATUS_FINAL"}}
            },
            "competitions": [{
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {"displayName": "Home Team"},
                        "score": "2"
                    },
                    {
                        "homeAway": "away",
                        "team": {"displayName": "Away Team"},
                        "score": "1"
                    }
                ]
            }],
            "boxscore": {
                "teams": [
                    {
                        "homeAway": "home",
                        "team": {"displayName": "Home Team", "abbreviation": "HOME"},
                        "statistics": [
                            {"name": "accuratePasses", "displayValue": "400"},
                            {"name": "totalPasses", "displayValue": "500"},
                            {"name": "passPct", "displayValue": "80"},
                        ]
                    },
                    {
                        "homeAway": "away",
                        "team": {"displayName": "Away Team", "abbreviation": "AWAY"},
                        "statistics": [
                            {"name": "accuratePasses", "displayValue": "300"},
                            {"name": "totalPasses", "displayValue": "400"},
                            {"name": "passPct", "displayValue": "75"},
                        ]
                    }
                ]
            }
        }

    def test_extract_full_report(self, mock_summary_for_extraction):
        """Test extracting full match report."""
        report = extract_advanced_match_team_stats(
            mock_summary_for_extraction, "123456", "test.league"
        )
        
        assert report["event_id"] == "123456"
        assert report["league"] == "test.league"
        assert report["match"]["home_team"] == "Home Team"
        assert report["match"]["away_team"] == "Away Team"
        assert report["match"]["home_score"] == 2
        assert report["match"]["away_score"] == 1
        assert len(report["teams"]) == 2

    def test_extract_full_report_with_raw(self, mock_summary_for_extraction):
        """Test extracting full report with raw stats."""
        report = extract_advanced_match_team_stats(
            mock_summary_for_extraction, "123456", "test.league", include_raw=True
        )
        
        assert "stats_raw" in report["teams"][0]
        assert "stats_raw" in report["teams"][1]

    def test_extract_full_report_without_raw(self, mock_summary_for_extraction):
        """Test extracting full report without raw stats."""
        report = extract_advanced_match_team_stats(
            mock_summary_for_extraction, "123456", "test.league", include_raw=False
        )
        
        assert "stats_raw" not in report["teams"][0]
        assert "stats_raw" not in report["teams"][1]

    def test_extract_empty_boxscore(self):
        """Test extracting from summary with empty boxscore."""
        summary = {
            "header": {},
            "boxscore": {"teams": []}
        }
        report = extract_advanced_match_team_stats(summary, "123", "test")
        
        assert len(report["teams"]) == 0


# ==================================================
# Test JSON serializability
# ==================================================
class TestJsonSerializability:
    def test_report_is_json_serializable(self):
        """Test that the report can be serialized to JSON."""
        report = {
            "event_id": "123456",
            "league": "test.league",
            "match": {
                "short_name": "Team A vs Team B",
                "date": "2026-07-03T22:00Z",
                "status": "STATUS_FINAL",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_score": 2,
                "away_score": 1
            },
            "teams": [
                {
                    "team_name": "Team A",
                    "team_abbr": "A",
                    "advanced_metrics": {
                        "passing": {
                            "passes_raw": "400/500",
                            "passes_completed": 400,
                            "passes_attempted": 500,
                            "pass_percentage_raw": "80",
                            "pass_percentage": 80.0
                        },
                        "attacks": {
                            "attacks": None,
                            "dangerous_attacks": None,
                            "crosses_raw": None,
                            "crosses_completed": None,
                            "crosses_attempted": None,
                            "cross_percentage_raw": None,
                            "cross_percentage": None
                        },
                        "shooting": {
                            "shots_on_target": 5,
                            "shots_off_target": None,
                            "blocked_shots": 2,
                            "average_shot_distance_raw": None,
                            "average_shot_distance": None,
                            "hit_woodwork": None
                        }
                    }
                }
            ]
        }
        
        # This should not raise an exception
        json_str = json.dumps(report)
        assert json_str is not None
        
        # And should be deserializable
        parsed = json.loads(json_str)
        assert parsed["event_id"] == "123456"


# ==================================================
# Test CLI integration
# ==================================================
class TestCliIntegration:
    def test_cli_json_output(self, capsys, mock_summary_for_extraction):
        """Test that CLI produces valid JSON with --json flag."""
        from scripts.match_advanced_team_stats import main
        
        with patch('scripts.match_advanced_team_stats.fetch_match_summary') as mock_fetch:
            mock_fetch.return_value = mock_summary_for_extraction
            
            # Simulate command line args
            with patch.object(sys, 'argv', [
                'match_advanced_team_stats.py',
                '--event', '123456',
                '--league', 'test.league',
                '--json'
            ]):
                ret = main()
            
            captured = capsys.readouterr()
            
            # Should return success
            assert ret == 0
            
            # Should produce valid JSON
            output = captured.out.strip()
            parsed = json.loads(output)
            assert parsed["event_id"] == "123456"

    @pytest.fixture
    def mock_summary_for_extraction(self):
        """Mock summary for CLI test."""
        return {
            "header": {
                "date": "2026-07-03T22:00Z",
                "status": {"type": {"name": "STATUS_FINAL"}}
            },
            "competitions": [{
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {"displayName": "Home Team"},
                        "score": "2"
                    },
                    {
                        "homeAway": "away",
                        "team": {"displayName": "Away Team"},
                        "score": "1"
                    }
                ]
            }],
            "boxscore": {
                "teams": [
                    {
                        "homeAway": "home",
                        "team": {"displayName": "Home Team"},
                        "statistics": [
                            {"name": "accuratePasses", "displayValue": "400"},
                            {"name": "totalPasses", "displayValue": "500"},
                        ]
                    }
                ]
            }
        }


# ==================================================
# Test save_report
# ==================================================
class TestSaveReport:
    def test_save_report_creates_file(self, tmp_path):
        """Test that save_report creates a JSON file."""
        report = {"event_id": "123", "match": {"short_name": "Test"}}
        output_path = tmp_path / "test_report.json"
        
        save_report(report, str(output_path))
        
        assert output_path.exists()
        
        with open(output_path, "r") as f:
            saved = json.load(f)
        
        assert saved["event_id"] == "123"
