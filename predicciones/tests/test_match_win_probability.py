"""
Tests for espn_match_predictor.py module and match_win_probability.py CLI.

Tests cover:
1. normalize_probability with various input formats
2. extract_predictor with valid data
3. extract_predictor with missing data
4. extract_win_probability_flow with valid data
5. extract_win_probability_flow with missing data
6. build_match_probability_report JSON serializable
7. CLI --json output
8. CLI --include-flow --limit
9. Handling predictor absent
10. Handling winProbability absent
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.espn_match_predictor import (
    normalize_probability,
    extract_predictor,
    extract_win_probability_flow,
    build_match_probability_report,
    print_match_probability_report,
    save_report,
    fetch_match_summary,
    extract_match_context,
)
from src.domain.exceptions import EspnApiError


# ==================================================
# Test normalize_probability
# ==================================================
class TestNormalizeProbability:
    def test_normalize_string_numeric(self):
        """Test normalize_probability with string numeric value."""
        assert normalize_probability("54.2") == 54.2
        assert normalize_probability("23.8") == 23.8
    
    def test_normalize_string_with_percent(self):
        """Test normalize_probability with percentage string."""
        assert normalize_probability("54.2%") == 54.2
        assert normalize_probability("23.8% ") == 23.8
        assert normalize_probability(" 22.0%") == 22.0
    
    def test_normalize_numeric(self):
        """Test normalize_probability with numeric values."""
        assert normalize_probability(54.2) == 54.2
        assert normalize_probability(23) == 23.0
        assert normalize_probability(22.0) == 22.0
    
    def test_normalize_none_or_empty(self):
        """Test normalize_probability with None or empty values."""
        assert normalize_probability(None) is None
        assert normalize_probability("") is None
        assert normalize_probability("  ") is None
        assert normalize_probability("%") is None
    
    def test_normalize_invalid(self):
        """Test normalize_probability with invalid values."""
        assert normalize_probability("invalid") is None
        assert normalize_probability({}) is None
        assert normalize_probability([]) is None


# ==================================================
# Test extract_predictor
# ==================================================
class TestExtractPredictor:
    def test_extract_predictor_valid(self):
        """Test extraction of predictor with valid data."""
        summary = {
            "predictor": {
                "homeTeamWinPercentage": "54.2",
                "awayTeamWinPercentage": "23.8",
                "tiePercentage": "22.0"
            }
        }
        
        result = extract_predictor(summary)
        
        assert result is not None
        assert result["home_team_win_percentage"] == 54.2
        assert result["away_team_win_percentage"] == 23.8
        assert result["tie_percentage"] == 22.0
    
    def test_extract_predictor_absent(self):
        """Test extraction when predictor is absent."""
        summary = {"competitions": []}
        
        result = extract_predictor(summary)
        
        assert result is None
    
    def test_extract_predictor_partial(self):
        """Test extraction when only some percentages exist."""
        summary = {
            "predictor": {
                "homeTeamWinPercentage": 54.2,
                # awayTeamWinPercentage and tiePercentage missing
            }
        }
        
        result = extract_predictor(summary)
        
        assert result is not None
        assert result["home_team_win_percentage"] == 54.2
        assert result["away_team_win_percentage"] is None
        assert result["tie_percentage"] is None
    
    def test_extract_predictor_numeric_values(self):
        """Test extraction with numeric (not string) values."""
        summary = {
            "predictor": {
                "homeTeamWinPercentage": 54.2,
                "awayTeamWinPercentage": 23.8,
                "tiePercentage": 22.0
            }
        }
        
        result = extract_predictor(summary)
        
        assert result is not None
        assert result["home_team_win_percentage"] == 54.2
        assert result["away_team_win_percentage"] == 23.8
        assert result["tie_percentage"] == 22.0


# ==================================================
# Test extract_win_probability_flow
# ==================================================
class TestExtractWinProbabilityFlow:
    def test_extract_flow_valid(self):
        """Test extraction of win probability flow with valid data."""
        summary = {
            "winProbability": [
                {
                    "homeWinPercentage": "61.3",
                    "awayWinPercentage": "18.5",
                    "tiePercentage": "20.2",
                    "play": {
                        "clock": {
                            "displayValue": "12'",
                            "value": 720
                        },
                        "period": {"number": 1},
                        "text": "Goal! Argentina 1, Cabo Verde 0..."
                    }
                },
                {
                    "homeWinPercentage": "58.0",
                    "awayWinPercentage": "17.0",
                    "tiePercentage": "25.0",
                    "play": {
                        "clock": {"displayValue": "45+2'"},
                        "period": {"number": 1},
                        "text": "End of first half"
                    }
                }
            ]
        }
        
        result = extract_win_probability_flow(summary)
        
        assert len(result) == 2
        assert result[0]["sequence_index"] == 0
        assert result[0]["clock_display"] == "12'"
        assert result[0]["clock_value"] == 720
        assert result[0]["period"] == 1
        assert result[0]["home_win_percentage"] == 61.3
        assert result[0]["away_win_percentage"] == 18.5
        assert result[0]["tie_percentage"] == 20.2
        assert "Goal!" in result[0]["play_text"]
        assert "raw_event" in result[0]
    
    def test_extract_flow_absent(self):
        """Test extraction when winProbability is absent."""
        summary = {"competitions": []}
        
        result = extract_win_probability_flow(summary)
        
        assert result == []
    
    def test_extract_flow_empty_list(self):
        """Test extraction when winProbability is empty list."""
        summary = {"winProbability": []}
        
        result = extract_win_probability_flow(summary)
        
        assert result == []
    
    def test_extract_flow_partial_fields(self):
        """Test extraction with missing fields in flow entries."""
        summary = {
            "winProbability": [
                {
                    "homeWinPercentage": 61.3,
                    # Missing awayWinPercentage, tiePercentage, play
                }
            ]
        }
        
        result = extract_win_probability_flow(summary)
        
        assert len(result) == 1
        assert result[0]["home_win_percentage"] == 61.3
        assert result[0]["away_win_percentage"] is None
        assert result[0]["tie_percentage"] is None
        assert result[0]["clock_display"] is None
        assert result[0]["period"] is None


# ==================================================
# Test extract_match_context
# ==================================================
class TestExtractMatchContext:
    def test_extract_context_valid(self):
        """Test extraction of match context with valid data."""
        summary = {
            "date": "2026-07-07T18:00:00Z",
            "competitions": [
                {
                    "status": {
                        "type": {"name": "STATUS_FINAL", "state": "post"}
                    },
                    "competitors": [
                        {
                            "homeAway": "home",
                            "team": {"displayName": "Argentina"},
                            "score": "3"
                        },
                        {
                            "homeAway": "away",
                            "team": {"displayName": "Cabo Verde"},
                            "score": "2"
                        }
                    ]
                }
            ]
        }
        
        result = extract_match_context(summary)
        
        assert result["short_name"] == "Argentina vs Cabo Verde"
        assert result["date"] == "2026-07-07T18:00:00Z"
        assert result["status"] == "STATUS_FINAL"
        assert result["home_team"] == "Argentina"
        assert result["away_team"] == "Cabo Verde"
        assert result["home_score"] == 3
        assert result["away_score"] == 2
    
    def test_extract_context_empty(self):
        """Test extraction with empty summary."""
        result = extract_match_context({})
        
        assert result["short_name"] == ""
        assert result["home_team"] == ""
        assert result["away_team"] == ""
        assert result["home_score"] is None
        assert result["away_score"] is None


# ==================================================
# Test build_match_probability_report
# ==================================================
class TestBuildMatchProbabilityReport:
    def test_build_report_with_predictor(self):
        """Test building report with predictor available."""
        summary = {
            "date": "2026-07-07T18:00:00Z",
            "competitions": [
                {
                    "status": {"type": {"name": "STATUS_FINAL"}},
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Home Team"}, "score": "2"},
                        {"homeAway": "away", "team": {"displayName": "Away Team"}, "score": "1"}
                    ]
                }
            ],
            "predictor": {
                "homeTeamWinPercentage": "54.2",
                "awayTeamWinPercentage": "23.8",
                "tiePercentage": "22.0"
            }
        }
        
        report = build_match_probability_report(summary, "123456", "fifa.world", include_flow=False)
        
        assert report["event_id"] == "123456"
        assert report["league"] == "fifa.world"
        assert report["predictor"]["available"] is True
        assert report["predictor"]["home_team_win_percentage"] == 54.2
        assert report["predictor"]["away_team_win_percentage"] == 23.8
        assert report["predictor"]["tie_percentage"] == 22.0
        assert "win_probability_flow" not in report
    
    def test_build_report_without_predictor(self):
        """Test building report without predictor."""
        summary = {
            "competitions": [
                {
                    "status": {"type": {"name": "STATUS_FINAL"}},
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Home Team"}},
                        {"homeAway": "away", "team": {"displayName": "Away Team"}}
                    ]
                }
            ]
        }
        
        report = build_match_probability_report(summary, "123456", "fifa.world")
        
        assert report["predictor"]["available"] is False
        assert report["predictor"]["home_team_win_percentage"] is None
        assert report["predictor"]["away_team_win_percentage"] is None
        assert report["predictor"]["tie_percentage"] is None
    
    def test_build_report_with_flow(self):
        """Test building report with win probability flow."""
        summary = {
            "competitions": [
                {
                    "status": {"type": {"name": "STATUS_FINAL"}},
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Home Team"}},
                        {"homeAway": "away", "team": {"displayName": "Away Team"}}
                    ]
                }
            ],
            "winProbability": [
                {"homeWinPercentage": "60.0", "awayWinPercentage": "20.0", "tiePercentage": "20.0"}
            ]
        }
        
        report = build_match_probability_report(summary, "123456", "fifa.world", include_flow=True)
        
        assert "win_probability_flow" in report
        assert report["win_probability_flow"]["available"] is True
        assert report["win_probability_flow"]["count"] == 1
        assert len(report["win_probability_flow"]["items"]) == 1
    
    def test_build_report_json_serializable(self):
        """Test that report is JSON serializable."""
        summary = {
            "competitions": [
                {
                    "status": {"type": {"name": "STATUS_FINAL"}},
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Home Team"}},
                        {"homeAway": "away", "team": {"displayName": "Away Team"}}
                    ]
                }
            ],
            "predictor": {
                "homeTeamWinPercentage": "54.2",
                "awayTeamWinPercentage": "23.8",
                "tiePercentage": "22.0"
            },
            "winProbability": [
                {
                    "homeWinPercentage": "60.0",
                    "play": {"clock": {"displayValue": "12'", "value": 720}}
                }
            ]
        }
        
        report = build_match_probability_report(summary, "123456", "fifa.world", include_flow=True)
        
        # Should not raise exception
        json_str = json.dumps(report)
        assert json_str is not None
        
        # Verify round-trip
        loaded = json.loads(json_str)
        assert loaded["event_id"] == "123456"
        assert loaded["predictor"]["available"] is True


# ==================================================
# Test save_report
# ==================================================
class TestSaveReport:
    def test_save_report(self, tmp_path):
        """Test saving report to file."""
        report = {
            "event_id": "123456",
            "league": "fifa.world",
            "predictor": {"available": True, "home_team_win_percentage": 54.2}
        }
        
        output_file = tmp_path / "test_report.json"
        save_report(report, str(output_file))
        
        assert output_file.exists()
        
        with open(output_file, "r") as f:
            loaded = json.load(f)
        
        assert loaded["event_id"] == "123456"


# ==================================================
# Test CLI
# ==================================================
class TestCLI:
    @patch('scripts.match_win_probability.fetch_match_summary')
    def test_cli_json_output(self, mock_fetch):
        """Test CLI with --json flag."""
        mock_fetch.return_value = {
            "competitions": [
                {
                    "status": {"type": {"name": "STATUS_FINAL"}},
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Home Team"}},
                        {"homeAway": "away", "team": {"displayName": "Away Team"}}
                    ]
                }
            ],
            "predictor": {
                "homeTeamWinPercentage": "54.2",
                "awayTeamWinPercentage": "23.8",
                "tiePercentage": "22.0"
            }
        }
        
        from scripts.match_win_probability import main
        import io
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            with patch.object(sys, 'argv', ['match_win_probability.py', '--event', '760500', '--json']):
                main()
            
            output = sys.stdout.getvalue()
            result = json.loads(output)
            
            assert result["event_id"] == "760500"
            assert result["predictor"]["available"] is True
        finally:
            sys.stdout = old_stdout
    
    @patch('scripts.match_win_probability.fetch_match_summary')
    def test_cli_include_flow_limit(self, mock_fetch):
        """Test CLI with --include-flow --limit."""
        mock_fetch.return_value = {
            "competitions": [
                {
                    "status": {"type": {"name": "STATUS_FINAL"}},
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Home Team"}},
                        {"homeAway": "away", "team": {"displayName": "Away Team"}}
                    ]
                }
            ],
            "winProbability": [
                {"homeWinPercentage": f"{60 + i}", "play": {"clock": {"displayValue": f"{i}'"}}}
                for i in range(50)
            ]
        }
        
        from scripts.match_win_probability import main
        import io
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            with patch.object(sys, 'argv', [
                'match_win_probability.py', '--event', '760500', 
                '--include-flow', '--limit', '5'
            ]):
                main()
            
            output = sys.stdout.getvalue()
            
            # Should mention limit
            assert "5 more entries" in output or "... and" in output
        finally:
            sys.stdout = old_stdout
    
    @patch('scripts.match_win_probability.fetch_match_summary')
    def test_cli_no_predictor_message(self, mock_fetch):
        """Test CLI shows message when predictor not available."""
        mock_fetch.return_value = {
            "competitions": [
                {
                    "status": {"type": {"name": "STATUS_FINAL"}},
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Home Team"}},
                        {"homeAway": "away", "team": {"displayName": "Away Team"}}
                    ]
                }
            ]
            # No predictor node
        }
        
        from scripts.match_win_probability import main
        import io
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            with patch.object(sys, 'argv', ['match_win_probability.py', '--event', '760500']):
                main()
            
            output = sys.stdout.getvalue()
            
            # Should mention predictor not available
            assert "not available" in output.lower() or "predictor" in output.lower()
        finally:
            sys.stdout = old_stdout
    
    @patch('scripts.match_win_probability.fetch_match_summary')
    def test_cli_save_option(self, mock_fetch, tmp_path):
        """Test CLI with --save option."""
        mock_fetch.return_value = {
            "competitions": [
                {
                    "status": {"type": {"name": "STATUS_FINAL"}},
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Home Team"}},
                        {"homeAway": "away", "team": {"displayName": "Away Team"}}
                    ]
                }
            ],
            "predictor": {
                "homeTeamWinPercentage": "54.2"
            }
        }
        
        from scripts.match_win_probability import main
        import io
        
        output_file = tmp_path / "saved_report.json"
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            with patch.object(sys, 'argv', [
                'match_win_probability.py', '--event', '760500',
                '--save', str(output_file)
            ]):
                main()
            
            output = sys.stdout.getvalue()
            
            # File should be created
            assert output_file.exists()
            assert "saved" in output.lower()
        finally:
            sys.stdout = old_stdout


# ==================================================
# Integration test with real ESPN API structure
# ==================================================
class TestIntegrationWithMockedESPN:
    @patch('src.data.espn_client_v2.EspnClient.get_summary')
    def test_full_workflow_with_mocked_api(self, mock_get_summary):
        """Test full workflow with mocked ESPN API response."""
        # Mock a realistic ESPN summary response
        mock_get_summary.return_value = {
            "id": "760500",
            "date": "2026-07-07T18:00:00Z",
            "competitions": [
                {
                    "status": {
                        "type": {"name": "STATUS_FINAL", "state": "post"}
                    },
                    "competitors": [
                        {
                            "homeAway": "home",
                            "team": {"displayName": "Argentina"},
                            "score": "3"
                        },
                        {
                            "homeAway": "away",
                            "team": {"displayName": "Cabo Verde"},
                            "score": "2"
                        }
                    ]
                }
            ],
            "predictor": {
                "homeTeamWinPercentage": 54.2,
                "awayTeamWinPercentage": 23.8,
                "tiePercentage": 22.0
            },
            "winProbability": [
                {
                    "sequenceIndex": 0,
                    "homeWinPercentage": 61.3,
                    "awayWinPercentage": 18.5,
                    "tiePercentage": 20.2,
                    "play": {
                        "clock": {"displayValue": "12'", "value": 720},
                        "period": {"number": 1},
                        "text": "Goal! Argentina 1, Cabo Verde 0"
                    }
                }
            ]
        }
        
        # Test the full workflow
        summary = fetch_match_summary("760500", "fifa.world")
        
        assert summary is not None
        assert summary["id"] == "760500"
        
        report = build_match_probability_report(
            summary=summary,
            event_id="760500",
            league="fifa.world",
            include_flow=True
        )
        
        assert report["event_id"] == "760500"
        assert report["predictor"]["available"] is True
        assert report["predictor"]["home_team_win_percentage"] == 54.2
        assert report["win_probability_flow"]["available"] is True
        assert report["win_probability_flow"]["count"] == 1
