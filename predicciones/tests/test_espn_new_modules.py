"""
Tests for ESPN integration, parsers, normalizers, and match input factory.
"""
import pytest
from unittest.mock import MagicMock, patch, Mock

from src.data.espn_parsers import (
    parse_scoreboard_event,
    parse_summary_to_context,
    _parse_stage,
    _parse_status,
)
from src.data.espn_normalizers import TeamNormalizer
from src.domain.models import UpcomingMatch, EspnTeamRef, TeamNormalizationResult
from src.domain.exceptions import (
    EspnApiError,
    MatchSelectionError,
    MatchInputBuildError,
)


# ==================================================
# 1. Test parse_scoreboard_event
# ==================================================
def test_parse_scoreboard_event_basic():
    """Test basic scoreboard event parsing."""
    mock_event = {
        "id": "401234567",
        "date": "2026-06-15T18:00:00Z",
        "league": {"name": "FIFA World Cup"},
        "season": {"slug": "2026"},
        "competitions": [
            {
                "status": {"type": {"name": "STATUS_SCHEDULED"}},
                "notes": "Group Stage",
                "venue": {"fullName": "Estadio Azteca"},
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {"displayName": "Mexico"},
                    },
                    {
                        "homeAway": "away",
                        "team": {"displayName": "Poland"},
                    }
                ],
            }
        ]
    }
    
    result = parse_scoreboard_event(mock_event)
    
    assert result is not None
    assert result.event_id == "401234567"
    assert result.home_team == "Mexico"
    assert result.away_team == "Poland"
    assert result.competition == "FIFA World Cup"
    assert result.status == "pre"


def test_parse_scoreboard_event_missing_competitions():
    """Test that events without competitions return None."""
    mock_event = {
        "id": "401234567",
        "date": "2026-06-15T18:00:00Z",
        "competitions": []
    }
    
    result = parse_scoreboard_event(mock_event)
    assert result is None


def test_parse_scoreboard_event_missing_teams():
    """Test that events without proper teams return None."""
    mock_event = {
        "id": "401234567",
        "competitions": [
            {
                "competitors": []
            }
        ]
    }
    
    result = parse_scoreboard_event(mock_event)
    assert result is None


# ==================================================
# 2. Test stage mapping
# ==================================================
def test_parse_stage_final():
    """Test final stage detection."""
    event = {"season": {"slug": "2026"}, "week": ""}
    comp = {"notes": "Final", "type": {"shortDetail": ""}}
    
    assert _parse_stage(event, comp) == "final"


def test_parse_stage_semi_final():
    """Test semi-final stage detection."""
    event = {"season": {"slug": "2026"}, "week": ""}
    comp = {"notes": "Semifinal", "type": {"shortDetail": ""}}
    
    assert _parse_stage(event, comp) == "semi_final"


def test_parse_stage_quarter_final():
    """Test quarter-final stage detection."""
    event = {"season": {"slug": "2026"}, "week": ""}
    comp = {"notes": "Quarterfinal", "type": {"shortDetail": ""}}
    
    assert _parse_stage(event, comp) == "quarter_final"


def test_parse_stage_round_of_16():
    """Test round of 16 detection."""
    event = {"season": {"slug": "2026"}, "week": ""}
    comp = {"notes": "Round of 16", "type": {"shortDetail": ""}}
    
    assert _parse_stage(event, comp) == "round_of_16"


def test_parse_stage_group():
    """Test group stage detection."""
    event = {"season": {"slug": "2026"}, "week": "1"}
    comp = {"notes": "Group Stage", "type": {"shortDetail": ""}}
    
    assert _parse_stage(event, comp) == "group"


def test_parse_stage_default():
    """Test default stage fallback."""
    event = {"season": {"slug": "2026"}, "week": ""}
    comp = {"notes": "", "type": {"shortDetail": ""}}
    
    assert _parse_stage(event, comp) == "regular"


# ==================================================
# 3. Test status parsing
# ==================================================
def test_parse_status_pre():
    """Test pre-match status."""
    comp = {"status": {"type": {"name": "STATUS_SCHEDULED"}}}
    assert _parse_status(comp) == "pre"


def test_parse_status_in():
    """Test in-progress status."""
    comp = {"status": {"type": {"name": "STATUS_IN_PROGRESS"}}}
    assert _parse_status(comp) == "in"


def test_parse_status_post():
    """Test post-match status."""
    comp = {"status": {"type": {"name": "STATUS_FINAL"}}}
    assert _parse_status(comp) == "post"


# ==================================================
# 4. Test team name normalization
# ==================================================
def test_team_normalizer_aliases():
    """Test team name alias resolution."""
    normalizer = TeamNormalizer()
    
    # Test explicit aliases
    assert normalizer.normalize("México") == "Mexico"
    assert normalizer.normalize("Inglaterra") == "England"
    assert normalizer.normalize("Estados Unidos") == "USA"
    assert normalizer.normalize("Holanda") == "Netherlands"


def test_team_normalizer_case_insensitive():
    """Test case-insensitive matching."""
    normalizer = TeamNormalizer()
    
    assert normalizer.normalize("mexico") == "Mexico"
    assert normalizer.normalize("MEXICO") == "Mexico"
    assert normalizer.normalize("england") == "England"


def test_team_normalizer_unknown_team():
    """Test unknown team returns original."""
    normalizer = TeamNormalizer()
    
    result = normalizer.normalize("UnknownTeamXYZ")
    assert result == "UnknownTeamXYZ"


def test_team_normalizer_find_team():
    """Test team finding with available teams list."""
    normalizer = TeamNormalizer()
    available = ["Mexico", "England", "Brazil", "Argentina"]
    
    # Exact match
    result = normalizer.find_team("Mexico", available)
    assert result.found is True
    assert result.normalized_name == "Mexico"
    assert result.confidence == 1.0
    
    # Alias match
    result = normalizer.find_team("México", available)
    assert result.found is True
    assert result.normalized_name == "Mexico"
    
    # Not found
    result = normalizer.find_team("UnknownTeam", available)
    assert result.found is False


# ==================================================
# 5. Test domain models
# ==================================================
def test_upcoming_match_to_display_row():
    """Test UpcomingMatch to_display_row method."""
    match = UpcomingMatch(
        event_id="401234567",
        date="2026-06-15T18:00:00Z",
        competition="FIFA World Cup",
        stage="group",
        status="pre",
        home_team="Mexico",
        away_team="Poland",
        venue="Estadio Azteca",
    )
    
    row = match.to_display_row()
    
    assert row["event_id"] == "401234567"
    assert row["home_team"] == "Mexico"
    assert row["away_team"] == "Poland"
    assert row["stage"] == "group"
    assert row["status"] == "PRE"


def test_espn_team_ref():
    """Test EspnTeamRef dataclass."""
    team = EspnTeamRef(
        team_id="123",
        name="Mexico",
        display_name="Mexico National Team",
        short_name="Mexico",
        abbreviation="MEX",
    )
    
    assert team.team_id == "123"
    assert team.name == "Mexico"
    assert team.abbreviation == "MEX"


# ==================================================
# 6. Test exceptions
# ==================================================
def test_espn_api_error():
    """Test EspnApiError exception."""
    error = EspnApiError(
        "Request failed",
        status_code=500,
        url="https://example.com"
    )
    
    assert error.message == "Request failed"
    assert error.status_code == 500
    assert "500" in str(error)


def test_match_selection_error():
    """Test MatchSelectionError exception."""
    error = MatchSelectionError(
        "No matches available",
        available_matches=["match1", "match2"]
    )
    
    assert error.message == "No matches available"
    assert len(error.available_matches) == 2


def test_match_input_build_error():
    """Test MatchInputBuildError exception."""
    error = MatchInputBuildError(
        "Failed to build input",
        source="espn",
        details={"event_id": "123"}
    )
    
    assert error.message == "Failed to build input"
    assert error.source == "espn"
    assert error.details["event_id"] == "123"


# ==================================================
# 7. Test MatchSelector (with mocks)
# ==================================================
@patch('src.application.match_selector.EspnClient')
def test_match_selector_get_upcoming(mock_client_class):
    """Test MatchSelector.get_upcoming_matches."""
    mock_client = MagicMock()
    mock_client.get_scoreboard.return_value = {
        "events": [
            {
                "id": "401234567",
                "date": "2026-06-15T18:00:00Z",
                "league": {"name": "FIFA World Cup"},
                "competitions": [{
                    "status": {"type": {"name": "STATUS_SCHEDULED"}},
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Mexico"}},
                        {"homeAway": "away", "team": {"displayName": "Poland"}},
                    ],
                }]
            }
        ]
    }
    mock_client_class.return_value = mock_client
    
    from src.application.match_selector import MatchSelector
    
    selector = MatchSelector()
    matches = selector.get_upcoming_matches(limit=10)
    
    assert len(matches) == 1
    assert matches[0].event_id == "401234567"
    assert matches[0].home_team == "Mexico"


# ==================================================
# 8. Test MatchInputFactory (with mocks)
# ==================================================
@patch('src.domain.match_input_factory.EspnClient')
@patch('src.domain.match_input_factory.TeamNormalizer')
def test_match_input_factory_build_from_event_id(mock_normalizer_class, mock_client_class):
    """Test building MatchInput from event ID."""
    mock_client = MagicMock()
    mock_client.get_summary.return_value = {
        "id": "401234567",
        "date": "2026-06-15T18:00:00Z",
        "league": {"name": "FIFA World Cup"},
        "season": {"slug": "2026"},
        "competitions": [{
            "status": {"type": {"name": "STATUS_SCHEDULED"}},
            "notes": "Group Stage",
            "venue": {"fullName": "Estadio Azteca"},
            "competitors": [
                {"homeAway": "home", "team": {"displayName": "Mexico"}},
                {"homeAway": "away", "team": {"displayName": "Poland"}},
            ],
        }]
    }
    mock_client_class.return_value = mock_client
    
    mock_normalizer = MagicMock()
    mock_normalizer.normalize.side_effect = lambda x: x  # Return as-is
    mock_normalizer_class.return_value = mock_normalizer
    
    from src.domain.match_input_factory import MatchInputFactory
    
    factory = MatchInputFactory()
    match_input = factory.build_from_event_id("401234567")
    
    assert match_input.metadata.home_team == "Mexico"
    assert match_input.metadata.away_team == "Poland"
    assert match_input.metadata.competition == "FIFA World Cup"
    assert match_input.metadata.stage.value == "group"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
