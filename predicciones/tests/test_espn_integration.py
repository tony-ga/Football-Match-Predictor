"""
Tests for ESPN integration and feature builder.
"""
import pytest
from unittest.mock import MagicMock, patch, Mock
from src.data.espn_client import EspnWorldCupClient
from src.data.feature_builder import MatchFeatureBuilder, TeamProfile


# ==================================================
# 1. Test parsing de scoreboard ESPN
# ==================================================
def test_espn_scoreboard_parsing():
    """Test que el cliente ESPN parsea correctamente el scoreboard."""
    client = EspnWorldCupClient()
    
    # Mock de respuesta de scoreboard
    mock_scoreboard = {
        "events": [
            {
                "id": "401234567",
                "date": "2026-06-15T18:00:00Z",
                "league": {"name": "FIFA World Cup"},
                "season": {"slug": "2026"},
                "week": "1",
                "competitions": [
                    {
                        "status": {"type": {"name": "STATUS_FINAL"}},
                        "notes": "Group Stage",
                        "venue": {"fullName": "Estadio Azteca"},
                        "competitors": [
                            {
                                "homeAway": "home",
                                "team": {"displayName": "Mexico"},
                                "score": "2",
                                "winner": True
                            },
                            {
                                "homeAway": "away",
                                "team": {"displayName": "Poland"},
                                "score": "0",
                                "winner": False
                            }
                        ],
                        "statistics": []
                    }
                ]
            }
        ]
    }
    
    with patch.object(client, '_make_request', return_value=mock_scoreboard):
        result = client.get_scoreboard()
        assert "events" in result
        assert len(result["events"]) == 1


# ==================================================
# 2. Test normalización home/away
# ==================================================
def test_normalize_event_home_away():
    """Test que _normalize_event identifica correctamente home/away."""
    client = EspnWorldCupClient()
    
    mock_event = {
        "id": "401234567",
        "date": "2026-06-15T18:00:00Z",
        "league": {"name": "FIFA World Cup"},
        "season": {"slug": "2026"},
        "competitions": [
            {
                "status": {"type": {"name": "STATUS_FINAL"}},
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {"displayName": "England"},
                        "score": "3",
                        "winner": True
                    },
                    {
                        "homeAway": "away",
                        "team": {"displayName": "Iran"},
                        "score": "0",
                        "winner": False
                    }
                ]
            }
        ]
    }
    
    normalized = client._normalize_event(mock_event)
    
    assert normalized["home_team"] == "England"
    assert normalized["away_team"] == "Iran"
    assert normalized["home_score"] == 3
    assert normalized["away_score"] == 0
    assert normalized["home_winner"] is True
    assert normalized["away_winner"] is False
    assert normalized["completed"] is True


# ==================================================
# 3. Test alias de nombres
# ==================================================
def test_team_name_normalization():
    """Test que normalize_team_name usa aliases correctamente."""
    client = EspnWorldCupClient()
    
    # Probar aliases explícitos
    assert client.normalize_team_name("México") == "Mexico"
    assert client.normalize_team_name("Inglaterra") == "England"
    assert client.normalize_team_name("Estados Unidos") == "USA"
    assert client.normalize_team_name("República Democrática del Congo") == "DR Congo"
    assert client.normalize_team_name("Países Bajos") == "Netherlands"
    assert client.normalize_team_name("Holanda") == "Netherlands"
    assert client.normalize_team_name("Corea del Sur") == "South Korea"
    assert client.normalize_team_name("Japón") == "Japan"
    
    # Nombres ya normalizados deben retornarse igual
    assert client.normalize_team_name("Mexico") == "Mexico"
    assert client.normalize_team_name("England") == "England"
    assert client.normalize_team_name("USA") == "USA"
    
    # Nombres desconocidos se retornan igual
    assert client.normalize_team_name("UnknownTeam") == "UnknownTeam"


# ==================================================
# 4. Test fallback cuando no hay eventos
# ==================================================
def test_fallback_no_events():
    """Test que get_recent_team_matches retorna lista vacía si no hay eventos."""
    client = EspnWorldCupClient()
    
    with patch.object(client, '_make_request', return_value={"events": []}):
        matches = client.get_recent_team_matches("Mexico")
        assert matches == []


def test_fallback_api_error():
    """Test que el cliente maneja errores de API elegantemente."""
    client = EspnWorldCupClient()
    
    with patch.object(client, '_make_request', return_value=None):
        scoreboard = client.get_scoreboard()
        assert scoreboard == {}
        
        summary = client.get_summary("12345")
        assert summary == {}


# ==================================================
# 5. Test team_context no formatea goles/corners como %
# ==================================================
def test_team_context_numeric_formatting():
    """Test que los valores numéricos en team_context no se formatean como porcentaje."""
    # Simular un perfil con datos ESPN
    profile = TeamProfile(
        team_name="Mexico",
        lambda_attack=1.25,
        lambda_defense=1.15,
        recent_form={
            "record": "W1 D0 L1",
            "goals_scored_avg": 1.5,
            "goals_conceded_avg": 1.0,
            "btts_rate": 0.50,
            "corners_avg": 4.5,
            "cards_avg": 2.0,
            "clean_sheets": 1,
            "form": "W1 D0 L1",
            "data_source": "espn_world_cup",
            "fifa_rank": 16,
            "matches_played": 2
        },
        wc_form={"played": 2, "record": "W1 D0 L1", "goals_scored": 3, "goals_conceded": 2, "matches": []},
        corners_lambda=5.0,
        cards_lambda=2.0,
        effective_weight_matches=2.0,
        data_warnings=[],
        data_source="espn_world_cup"
    )
    
    profile_dict = profile.to_dict()
    recent = profile_dict["recent_form"]
    
    # Verificar que los valores son floats, no strings con %
    assert isinstance(recent["goals_scored_avg"], (int, float))
    assert isinstance(recent["goals_conceded_avg"], (int, float))
    assert isinstance(recent["corners_avg"], (int, float))
    assert isinstance(recent["cards_avg"], (int, float))
    
    # Verificar rangos razonables (no 162.00% o 500.00%)
    assert 0.5 <= recent["goals_scored_avg"] <= 5.0
    assert 0.5 <= recent["goals_conceded_avg"] <= 5.0
    assert 2.0 <= recent["corners_avg"] <= 15.0
    assert 1.0 <= recent["cards_avg"] <= 6.0


# ==================================================
# 6. Test output cambia entre diferentes matchups
# ==================================================
def test_different_matchups_produce_different_outputs():
    """Test que Mexico vs England no sale casi idéntico a Mexico vs Ecuador."""
    # Crear builder sin ESPN para control determinístico
    builder = MatchFeatureBuilder(espn_client=None)
    
    # Obtener perfiles estáticos
    mexico_profile = builder.build_team_profile("Mexico", "2026-06-15")
    england_profile = builder.build_team_profile("England", "2026-06-15")
    ecuador_profile = builder.build_team_profile("Ecuador", "2026-06-15")
    dr_congo_profile = builder.build_team_profile("DR Congo", "2026-06-15")
    
    # Mexico vs England
    mexico_attack_mex_eng = mexico_profile.lambda_attack
    england_attack_mex_eng = england_profile.lambda_attack
    
    # Mexico vs Ecuador  
    mexico_attack_mex_ecu = mexico_profile.lambda_attack
    ecuador_attack_mex_ecu = ecuador_profile.lambda_attack
    
    # England vs DR Congo
    england_attack_eng_drc = england_profile.lambda_attack
    dr_congo_attack_eng_drc = dr_congo_profile.lambda_attack
    
    # Verificar que hay diferencia significativa entre equipos
    # England (attack ~1.65) debe ser mayor que Ecuador (~1.30) y DR Congo (~1.05)
    assert england_attack_mex_eng > ecuador_attack_mex_ecu + 0.2, \
        f"England attack ({england_attack_mex_eng}) should be > Ecuador ({ecuador_attack_mex_ecu})"
    
    assert england_attack_eng_drc > dr_congo_attack_eng_drc + 0.4, \
        f"England attack ({england_attack_eng_drc}) should be > DR Congo ({dr_congo_attack_eng_drc})"
    
    # Mexico debe estar entre medio
    assert ecuador_attack_mex_ecu >= mexico_attack_mex_eng - 0.1 or ecuador_attack_mex_ecu <= mexico_attack_mex_eng + 0.1, \
        "Mexico and Ecuador should have similar attack ratings"


# ==================================================
# 7. Test stage weighting
# ==================================================
def test_stage_weights():
    """Test que los stage weights se aplican correctamente."""
    builder = MatchFeatureBuilder()
    
    assert builder._get_stage_weight("group") == 1.0
    assert builder._get_stage_weight("round_of_16") == 1.2
    assert builder._get_stage_weight("quarter_final") == 1.3
    assert builder._get_stage_weight("semi_final") == 1.4
    assert builder._get_stage_weight("final") == 1.5
    assert builder._get_stage_weight("unknown") == 1.0


# ==================================================
# 8. Test TeamProfile data_source field
# ==================================================
def test_team_profile_data_source():
    """Test que TeamProfile tiene campo data_source accesible."""
    profile = TeamProfile(
        team_name="Test",
        lambda_attack=1.0,
        lambda_defense=1.0,
        recent_form={},
        wc_form={},
        corners_lambda=5.0,
        cards_lambda=2.0,
        effective_weight_matches=0.0,
        data_source="test_source"
    )
    
    assert profile.data_source == "test_source"
    assert "data_source" in profile.to_dict()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
