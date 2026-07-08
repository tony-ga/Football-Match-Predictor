"""
Tests for match_recap.py CLI script.

Tests cover:
- Parsing a mock valid summary
- Extracting home/away score and stats
- --json output functionality
- Handling incomplete summary
- Handling missing venue / leaders / commentary
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.match_recap import (
    fetch_match_summary,
    extract_match_recap,
    print_match_recap,
    save_recap,
    _extract_team_stats,
    _parse_numeric,
    _extract_leaders,
)
from src.domain.exceptions import EspnApiError


# ==================================================
# Mock Summary Data
# ==================================================
@pytest.fixture
def mock_summary_complete():
    """Complete mock summary with all fields."""
    return {
        "header": {
            "id": "760509",
            "date": "2026-07-07T18:00:00Z",
            "status": {
                "type": {
                    "name": "STATUS_FINAL",
                    "state": "post"
                }
            }
        },
        "competitions": [
            {
                "venue": {
                    "fullName": "Estadio Monumental"
                },
                "attendance": [45000],
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
                            "displayName": "Egypt",
                            "name": "Egypt"
                        },
                        "score": "2"
                    }
                ]
            }
        ],
        "boxscore": {
            "teams": [
                {
                    "homeAway": "home",
                    "statistics": [
                        {"name": "possession", "displayValue": "61"},
                        {"name": "total shots", "displayValue": "15"},
                        {"name": "shots on target", "displayValue": "7"},
                        {"name": "expected goals (xG)", "displayValue": "2.43"},
                        {"name": "corners", "displayValue": "6"},
                        {"name": "fouls", "displayValue": "12"},
                    ]
                },
                {
                    "homeAway": "away",
                    "statistics": [
                        {"name": "possession", "displayValue": "39"},
                        {"name": "total shots", "displayValue": "9"},
                        {"name": "shots on target", "displayValue": "4"},
                        {"name": "expected goals (xG)", "displayValue": "1.35"},
                        {"name": "corners", "displayValue": "3"},
                        {"name": "fouls", "displayValue": "15"},
                    ]
                }
            ]
        },
        "leaders": [
            {
                "label": "Goals",
                "items": [
                    {
                        "athlete": {"displayName": "Lionel Messi", "shortName": "L. Messi"},
                        "team": {"abbreviation": "ARG", "displayName": "Argentina"},
                        "value": 1
                    }
                ]
            },
            {
                "label": "Assists",
                "items": [
                    {
                        "athlete": {"displayName": "Angel Di Maria", "shortName": "A. Di Maria"},
                        "team": {"abbreviation": "ARG", "displayName": "Argentina"},
                        "value": 2
                    }
                ]
            }
        ],
        "commentary": [{"id": 1}, {"id": 2}, {"id": 3}] * 38  # 114 events
    }


@pytest.fixture
def mock_summary_minimal():
    """Minimal mock summary with only essential fields."""
    return {
        "header": {
            "date": "2026-07-07T18:00:00Z"
        },
        "competitions": [
            {
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {"displayName": "Team A"},
                        "score": "1"
                    },
                    {
                        "homeAway": "away",
                        "team": {"displayName": "Team B"},
                        "score": "0"
                    }
                ]
            }
        ],
        "boxscore": {
            "teams": []
        }
    }


@pytest.fixture
def mock_summary_missing_venue():
    """Summary without venue information."""
    return {
        "header": {"date": "2026-07-07T18:00:00Z"},
        "competitions": [
            {
                "venue": {},
                "competitors": [
                    {"homeAway": "home", "team": {"displayName": "Team A"}, "score": "1"},
                    {"homeAway": "away", "team": {"displayName": "Team B"}, "score": "0"}
                ]
            }
        ],
        "boxscore": {"teams": []}
    }


@pytest.fixture
def mock_summary_no_leaders():
    """Summary without leaders data."""
    return {
        "header": {"date": "2026-07-07T18:00:00Z"},
        "competitions": [
            {
                "competitors": [
                    {"homeAway": "home", "team": {"displayName": "Team A"}, "score": "1"},
                    {"homeAway": "away", "team": {"displayName": "Team B"}, "score": "0"}
                ]
            }
        ],
        "boxscore": {"teams": []},
        "leaders": []
    }


# ==================================================
# Test _parse_numeric
# ==================================================
class TestParseNumeric:
    def test_parse_integer_string(self):
        assert _parse_numeric("15") == 15.0

    def test_parse_float_string(self):
        assert _parse_numeric("2.43") == 2.43

    def test_parse_percentage(self):
        assert _parse_numeric("61%") == 61.0

    def test_parse_none(self):
        assert _parse_numeric(None) is None

    def test_parse_invalid_string(self):
        assert _parse_numeric("invalid") is None

    def test_parse_already_numeric(self):
        assert _parse_numeric(15) == 15.0
        assert _parse_numeric(2.43) == 2.43


# ==================================================
# Test _extract_team_stats
# ==================================================
class TestExtractTeamStats:
    def test_extract_home_stats(self, mock_summary_complete):
        boxscore = mock_summary_complete["boxscore"]
        stats = _extract_team_stats(boxscore, "home")

        assert stats["possession"] == 61.0
        assert stats["shots"] == 15.0
        assert stats["shots_on_target"] == 7.0
        assert stats["xg"] == 2.43
        assert stats["corners"] == 6.0
        assert stats["fouls"] == 12.0

    def test_extract_away_stats(self, mock_summary_complete):
        boxscore = mock_summary_complete["boxscore"]
        stats = _extract_team_stats(boxscore, "away")

        assert stats["possession"] == 39.0
        assert stats["shots"] == 9.0
        assert stats["shots_on_target"] == 4.0
        assert stats["xg"] == 1.35

    def test_empty_boxscore(self):
        stats = _extract_team_stats({"teams": []}, "home")
        assert stats["possession"] is None
        assert stats["shots"] is None

    def test_missing_statistics_array(self):
        boxscore = {
            "teams": [
                {"homeAway": "home", "statistics": None}
            ]
        }
        stats = _extract_team_stats(boxscore, "home")
        assert stats["possession"] is None


# ==================================================
# Test _extract_leaders
# ==================================================
class TestExtractLeaders:
    def test_extract_goals_leader(self, mock_summary_complete):
        leaders_data = mock_summary_complete["leaders"]
        leaders = _extract_leaders(leaders_data)

        assert "Goals" in leaders
        assert len(leaders["Goals"]) > 0
        assert leaders["Goals"][0]["name"] == "Lionel Messi"
        assert leaders["Goals"][0]["team"] == "ARG"
        assert leaders["Goals"][0]["value"] == 1

    def test_extract_assists_leader(self, mock_summary_complete):
        leaders_data = mock_summary_complete["leaders"]
        leaders = _extract_leaders(leaders_data)

        assert "Assists" in leaders
        assert leaders["Assists"][0]["name"] == "Angel Di Maria"

    def test_empty_leaders(self):
        leaders = _extract_leaders([])
        assert leaders == {}

    def test_none_leaders(self):
        leaders = _extract_leaders(None)
        assert leaders == {}


# ==================================================
# Test extract_match_recap
# ==================================================
class TestExtractMatchRecap:
    def test_extract_complete_recap(self, mock_summary_complete):
        recap = extract_match_recap(
            summary=mock_summary_complete,
            league="fifa.world",
            event_id="760509",
            include_leaders=True,
            include_commentary_count=True
        )

        assert recap["event_id"] == "760509"
        assert recap["league"] == "fifa.world"
        assert recap["short_name"] == "Argentina vs Egypt"
        assert recap["status"] in ["STATUS_FINAL", "post"]
        assert recap["home_team"] == "Argentina"
        assert recap["away_team"] == "Egypt"
        assert recap["home_score"] == 3.0
        assert recap["away_score"] == 2.0
        assert recap["venue"] == "Estadio Monumental"
        assert recap["attendance"] == 45000
        assert "leaders" in recap
        assert recap["commentary_count"] > 100

    def test_extract_basic_recap(self, mock_summary_minimal):
        recap = extract_match_recap(
            summary=mock_summary_minimal,
            league="fifa.world",
            event_id="123456"
        )

        assert recap["event_id"] == "123456"
        assert recap["home_team"] == "Team A"
        assert recap["away_team"] == "Team B"
        assert recap["home_score"] == 1.0
        assert recap["away_score"] == 0.0

    def test_missing_venue_shows_na(self, mock_summary_missing_venue):
        recap = extract_match_recap(
            summary=mock_summary_missing_venue,
            league="fifa.world",
            event_id="123"
        )

        assert recap["venue"] == "N/A"

    def test_no_leaders_when_not_included(self, mock_summary_complete):
        recap = extract_match_recap(
            summary=mock_summary_complete,
            league="fifa.world",
            event_id="760509",
            include_leaders=False
        )

        assert "leaders" not in recap

    def test_leaders_when_included(self, mock_summary_complete):
        recap = extract_match_recap(
            summary=mock_summary_complete,
            league="fifa.world",
            event_id="760509",
            include_leaders=True
        )

        assert "leaders" in recap

    def test_no_commentary_count_when_not_included(self, mock_summary_complete):
        recap = extract_match_recap(
            summary=mock_summary_complete,
            league="fifa.world",
            event_id="760509",
            include_commentary_count=False
        )

        assert "commentary_count" not in recap

    def test_commentary_count_when_included(self, mock_summary_complete):
        recap = extract_match_recap(
            summary=mock_summary_complete,
            league="fifa.world",
            event_id="760509",
            include_commentary_count=True
        )

        assert "commentary_count" in recap
        assert recap["commentary_count"] == 114

    def test_team_stats_structure(self, mock_summary_complete):
        recap = extract_match_recap(
            summary=mock_summary_complete,
            league="fifa.world",
            event_id="760509"
        )

        assert "team_stats" in recap
        assert "home" in recap["team_stats"]
        assert "away" in recap["team_stats"]
        assert recap["team_stats"]["home"]["possession"] == 61.0
        assert recap["team_stats"]["away"]["possession"] == 39.0


# ==================================================
# Test print_match_recap
# ==================================================
class TestPrintMatchRecap:
    def test_print_format(self, mock_summary_complete, capsys):
        recap = extract_match_recap(
            summary=mock_summary_complete,
            league="fifa.world",
            event_id="760509"
        )

        print_match_recap(recap)
        captured = capsys.readouterr()

        assert "Argentina vs Egypt" in captured.out
        assert "Event ID: 760509" in captured.out
        assert "Argentina" in captured.out
        assert "Egypt" in captured.out
        assert "Possession:" in captured.out

    def test_print_with_leaders(self, mock_summary_complete, capsys):
        recap = extract_match_recap(
            summary=mock_summary_complete,
            league="fifa.world",
            event_id="760509",
            include_leaders=True
        )

        print_match_recap(recap)
        captured = capsys.readouterr()

        assert "Leaders" in captured.out
        assert "Messi" in captured.out or "Lionel" in captured.out

    def test_print_with_commentary_count(self, mock_summary_complete, capsys):
        recap = extract_match_recap(
            summary=mock_summary_complete,
            league="fifa.world",
            event_id="760509",
            include_commentary_count=True
        )

        print_match_recap(recap)
        captured = capsys.readouterr()

        assert "Commentary events:" in captured.out


# ==================================================
# Test save_recap
# ==================================================
class TestSaveRecap:
    def test_save_to_file(self, mock_summary_complete, tmp_path):
        recap = extract_match_recap(
            summary=mock_summary_complete,
            league="fifa.world",
            event_id="760509"
        )

        output_file = tmp_path / "test_recap.json"
        save_recap(recap, str(output_file))

        assert output_file.exists()

        with open(output_file, "r") as f:
            saved_data = json.load(f)

        assert saved_data["event_id"] == "760509"
        assert saved_data["home_team"] == "Argentina"

    def test_save_creates_parent_directories(self, mock_summary_complete, tmp_path):
        recap = extract_match_recap(
            summary=mock_summary_complete,
            league="fifa.world",
            event_id="760509"
        )

        nested_path = tmp_path / "subdir" / "output" / "recap.json"
        save_recap(recap, str(nested_path))

        assert nested_path.exists()


# ==================================================
# Test fetch_match_summary
# ==================================================
class TestFetchMatchSummary:
    @patch('scripts.match_recap.EspnClient')
    def test_fetch_success(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.get_summary.return_value = {"header": {}}
        mock_client_class.return_value = mock_client

        result = fetch_match_summary("123456", "fifa.world")

        assert result == {"header": {}}
        mock_client.get_summary.assert_called_once_with("123456")

    @patch('scripts.match_recap.EspnClient')
    def test_fetch_empty_response_raises_error(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.get_summary.return_value = {}
        mock_client_class.return_value = mock_client

        with pytest.raises(EspnApiError):
            fetch_match_summary("123456", "fifa.world")

    @patch('scripts.match_recap.EspnClient')
    def test_fetch_error_response_raises_error(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.get_summary.return_value = {"error": True, "message": "Not found"}
        mock_client_class.return_value = mock_client

        with pytest.raises(EspnApiError) as exc_info:
            fetch_match_summary("123456", "fifa.world")

        assert "ESPN returned error" in str(exc_info.value)

    def test_fetch_empty_event_id_raises_error(self):
        with pytest.raises(ValueError):
            fetch_match_summary("", "fifa.world")

    def test_fetch_whitespace_event_id_raises_error(self):
        with pytest.raises(ValueError):
            fetch_match_summary("   ", "fifa.world")


# ==================================================
# Test CLI main function
# ==================================================
class TestMainFunction:
    @patch('scripts.match_recap.fetch_match_summary')
    @patch('scripts.match_recap.print_match_recap')
    def test_main_basic(self, mock_print, mock_fetch, mock_summary_complete):
        mock_fetch.return_value = mock_summary_complete

        with patch.object(sys, 'argv', ['match_recap.py', '--event', '760509']):
            from scripts.match_recap import main
            result = main()

        assert result == 0
        mock_fetch.assert_called_once()
        mock_print.assert_called_once()

    @patch('scripts.match_recap.fetch_match_summary')
    def test_main_json_output(self, mock_fetch, mock_summary_complete, capsys):
        mock_fetch.return_value = mock_summary_complete

        with patch.object(sys, 'argv', ['match_recap.py', '--event', '760509', '--json']):
            from scripts.match_recap import main
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["event_id"] == "760509"

    @patch('scripts.match_recap.fetch_match_summary')
    @patch('scripts.match_recap.save_recap')
    def test_main_save_option(self, mock_save, mock_fetch, mock_summary_complete):
        mock_fetch.return_value = mock_summary_complete

        with patch.object(sys, 'argv', ['match_recap.py', '--event', '760509', '--save', 'output.json']):
            from scripts.match_recap import main
            result = main()

        assert result == 0
        mock_save.assert_called_once()

    @patch('scripts.match_recap.fetch_match_summary')
    def test_main_api_error(self, mock_fetch):
        mock_fetch.side_effect = EspnApiError("API Error")

        with patch.object(sys, 'argv', ['match_recap.py', '--event', '760509']):
            from scripts.match_recap import main
            result = main()

        assert result == 2

    @patch('scripts.match_recap.fetch_match_summary')
    def test_main_value_error(self, mock_fetch):
        mock_fetch.side_effect = ValueError("Invalid event ID")

        with patch.object(sys, 'argv', ['match_recap.py', '--event', '760509']):
            from scripts.match_recap import main
            result = main()

        assert result == 1


# ==================================================
# Test edge cases
# ==================================================
class TestEdgeCases:
    def test_summary_without_competitions(self):
        summary = {
            "header": {"date": "2026-07-07"},
            "boxscore": {"teams": []}
        }

        recap = extract_match_recap(summary, "fifa.world", "123")

        assert recap["home_team"] is None
        assert recap["away_team"] is None
        assert recap["venue"] is None

    def test_summary_with_single_competitor(self):
        summary = {
            "header": {"date": "2026-07-07"},
            "competitions": [
                {
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Only Team"}, "score": "1"}
                    ]
                }
            ],
            "boxscore": {"teams": []}
        }

        recap = extract_match_recap(summary, "fifa.world", "123")

        assert recap["home_team"] == "Only Team"
        assert recap["away_team"] is None

    def test_score_as_int_instead_of_string(self):
        summary = {
            "header": {"date": "2026-07-07"},
            "competitions": [
                {
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Team A"}, "score": 3},
                        {"homeAway": "away", "team": {"displayName": "Team B"}, "score": 1}
                    ]
                }
            ],
            "boxscore": {"teams": []}
        }

        recap = extract_match_recap(summary, "fifa.world", "123")

        assert recap["home_score"] == 3.0
        assert recap["away_score"] == 1.0

    def test_attendance_as_single_value(self):
        summary = {
            "header": {"date": "2026-07-07"},
            "competitions": [
                {
                    "attendance": 50000,
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Team A"}, "score": "1"},
                        {"homeAway": "away", "team": {"displayName": "Team B"}, "score": "0"}
                    ]
                }
            ],
            "boxscore": {"teams": []}
        }

        recap = extract_match_recap(summary, "fifa.world", "123")

        assert recap["attendance"] == 50000

    def test_leaders_with_missing_athlete(self):
        leaders_data = [
            {
                "label": "Goals",
                "items": [
                    {"team": {"abbreviation": "ARG"}, "value": 1}  # Missing athlete
                ]
            }
        ]

        leaders = _extract_leaders(leaders_data)

        # Should skip entries without athlete name
        assert "Goals" not in leaders or len(leaders.get("Goals", [])) == 0

    def test_commentary_not_list(self):
        summary = {
            "header": {"date": "2026-07-07"},
            "competitions": [
                {
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Team A"}, "score": "1"},
                        {"homeAway": "away", "team": {"displayName": "Team B"}, "score": "0"}
                    ]
                }
            ],
            "boxscore": {"teams": []},
            "commentary": "not a list"
        }

        recap = extract_match_recap(
            summary, "fifa.world", "123",
            include_commentary_count=True
        )

        assert recap["commentary_count"] == 0
