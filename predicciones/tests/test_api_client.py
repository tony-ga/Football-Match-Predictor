import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from src.data.api_client import FootballAPIClient
from src.data.feature_builder import MatchFeatureBuilder

@pytest.fixture
def mock_api_client():
    client = FootballAPIClient(api_key="test_key", source="api_football")
    return client

@patch("requests.Session.get")
def test_get_world_cup_fixtures_api_football(mock_get, mock_api_client):
    # Mock API-Football response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": [
            {
                "fixture": {"id": 100, "date": "2026-06-15T18:00:00Z", "neutral": True},
                "league": {"name": "World Cup"},
                "teams": {
                    "home": {"name": "Argentina"},
                    "away": {"name": "Qatar"}
                },
                "goals": {"home": 3, "away": 0}
            }
        ]
    }
    mock_get.return_value = mock_response

    fixtures = mock_api_client.get_world_cup_fixtures(season=2026)
    assert len(fixtures) == 1
    assert fixtures[0]["home_team"] == "Argentina"
    assert fixtures[0]["away_team"] == "Qatar"
    assert fixtures[0]["home_score"] == 3
    assert fixtures[0]["away_score"] == 0

def test_resolve_team_id(mock_api_client):
    # Test direct mapping
    mock_api_client.mappings = {
        "api_football": {
            "Argentina": 26,
            "Congo DR": 1482
        }
    }
    
    assert mock_api_client.resolve_team_id("Argentina", "api_football") == 26
    # Test fuzzy matching
    assert mock_api_client.resolve_team_id("Congo D.R.", "api_football") == 1482

@patch("src.data.api_client.FootballAPIClient.get_world_cup_fixtures")
@patch("src.data.api_client.FootballAPIClient.get_team_last_matches")
def test_feature_builder_profile(mock_last_matches, mock_wc, mock_api_client):
    # Set up mock fixtures
    ref_date = datetime(2026, 7, 1)
    
    mock_wc.return_value = [
        {
            "fixture_id": 100,
            "date": (ref_date - timedelta(days=5)).isoformat(),
            "home_team": "Portugal",
            "away_team": "Ghana",
            "home_score": 3,
            "away_score": 1,
            "competition": "World Cup",
            "stage": "group"
        }
    ]
    
    mock_last_matches.return_value = [
        {
            "fixture_id": 50,
            "date": (ref_date - timedelta(days=20)).isoformat(),
            "home_team": "Spain",
            "away_team": "Portugal",
            "home_score": 1,
            "away_score": 2,
            "competition": "UEFA Nations League",
            "stage": "regular"
        }
    ]
    
    builder = MatchFeatureBuilder(mock_api_client)
    profile = builder.build_team_profile("Portugal", "2026-07-01")
    
    assert profile.team_name == "Portugal"
    assert profile.effective_weight_matches == 2
    assert profile.wc_form["played"] == 1
    assert profile.wc_form["record"] == "W1 D0 L0"
    assert profile.recent_form["goals_scored_avg"] == 2.0
