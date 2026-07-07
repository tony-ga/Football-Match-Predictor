"""
Tests for market prediction modules.

Tests cover:
1. Market availability evaluation
2. Corners model predictions
3. Cards model predictions
4. Shots model predictions
5. Player props output structure
6. Fallback when data is missing
7. Cache filename safety (Windows/Linux)
"""
import pytest
from unittest.mock import MagicMock, patch
import json
import hashlib
from pathlib import Path

from src.domain.market_types import (
    MarketType,
    DataSource,
    ConfidenceLevel,
    MarketAvailability,
    AllMarketsOutput,
    CornersMarketPrediction,
    CardsMarketPrediction,
    ShotsMarketPrediction,
    PlayerPropsPrediction,
)
from src.models.market_availability import MarketAvailabilityEvaluator
from src.models.market_models import (
    CornersModel,
    CardsModel,
    ShotsModel,
    PlayerPropsModel,
    poisson_sf,
)
from src.data.cache_manager import EspnCacheManager
from src.data.espn_stats_parsers import (
    extract_team_stats_from_summary,
    extract_player_stats_from_summary,
    extract_events_from_summary,
)


# ==================================================
# 1. Test Market Availability Evaluator
# ==================================================
class TestMarketAvailabilityEvaluator:
    
    def test_evaluate_corners_availability_sufficient_data(self):
        """Test corners availability with sufficient data."""
        evaluator = MarketAvailabilityEvaluator()
        
        # Create mock matches with corner stats
        home_matches = [
            {"stats": {"home_corners": 5, "away_corners": 3}},
            {"stats": {"home_corners": 4, "away_corners": 4}},
            {"stats": {"home_corners": 6, "away_corners": 2}},
            {"stats": {"home_corners": 3, "away_corners": 5}},
        ]
        away_matches = [
            {"stats": {"away_corners": 4, "home_corners": 3}},
            {"stats": {"away_corners": 5, "home_corners": 2}},
        ]
        
        total_avail, event_avail, other_avail = evaluator.evaluate_corners_availability(
            home_matches, away_matches, has_event_data=False
        )
        
        assert total_avail.available == True
        assert total_avail.sample_size == 6
        assert total_avail.confidence in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM]
    
    def test_evaluate_corners_availability_insufficient_data(self):
        """Test corners availability with insufficient data."""
        evaluator = MarketAvailabilityEvaluator()
        
        # Only 1 match with corners
        home_matches = [
            {"stats": {"home_corners": 5}},
            {"stats": {}},
            {"stats": {}},
        ]
        away_matches = []
        
        total_avail, event_avail, other_avail = evaluator.evaluate_corners_availability(
            home_matches, away_matches, has_event_data=False
        )
        
        assert total_avail.available == False
        assert "Insufficient" in total_avail.reason
    
    def test_evaluate_cards_availability(self):
        """Test cards availability evaluation."""
        evaluator = MarketAvailabilityEvaluator()
        
        home_matches = [
            {"stats": {"home_yellow_cards": 2, "home_red_cards": 0}},
            {"stats": {"home_total_cards": 3}},
            {"stats": {"home_yellow_cards": 1}},
        ]
        away_matches = [
            {"stats": {"away_yellow_cards": 2}},
        ]
        
        total_avail, event_avail = evaluator.evaluate_cards_availability(
            home_matches, away_matches, has_event_data=False
        )
        
        assert total_avail.available == True
        assert total_avail.sample_size >= 3
    
    def test_evaluate_shots_availability(self):
        """Test shots on target availability."""
        evaluator = MarketAvailabilityEvaluator()
        
        home_matches = [
            {"stats": {"home_shots_on_target": 4}},
            {"stats": {"home_shots_on_target": 5}},
            {"stats": {"home_shots_on_target": 3}},
        ]
        away_matches = [
            {"stats": {"away_shots_on_target": 4}},
        ]
        
        avail = evaluator.evaluate_shots_availability(home_matches, away_matches)
        
        assert avail.available == True
        assert avail.sample_size == 4
    
    def test_evaluate_player_props_no_lineup(self):
        """Test player props unavailable without lineup data."""
        evaluator = MarketAvailabilityEvaluator()
        
        home_matches = [
            {"players": [{"player_id": "1", "goals": 1}]},
            {"players": [{"player_id": "2", "goals": 0}]},
            {"players": [{"player_id": "1", "goals": 2}]},
        ]
        away_matches = []
        
        avail = evaluator.evaluate_player_props_availability(
            home_matches, away_matches,
            home_lineup_available=False,
            away_lineup_available=False,
        )
        
        assert avail.available == False
        assert "lineup" in avail.reason.lower()


# ==================================================
# 2. Test Corners Model
# ==================================================
class TestCornersModel:
    
    def test_predict_total_corners_basic(self):
        """Test basic total corners prediction."""
        model = CornersModel()
        
        result = model.predict_total_corners(
            home_avg_corners=5.0,
            away_avg_corners=4.0,
        )
        
        assert "expected_total" in result
        assert "lines" in result
        assert abs(result["expected_total"] - 9.0) < 0.1
        
        # Check lines exist
        assert "over_8" in result["lines"]
        assert "under_8" in result["lines"]
    
    def test_predict_total_corners_with_conceded(self):
        """Test corners prediction with conceded stats."""
        model = CornersModel()
        
        result = model.predict_total_corners(
            home_avg_corners=5.0,
            away_avg_corners=4.0,
            home_avg_corners_conceded=3.0,
            away_avg_corners_conceded=4.0,
        )
        
        assert "expected_total" in result
        # Should use both attack and defense stats
    
    def test_predict_team_corners(self):
        """Test team corner totals."""
        model = CornersModel()
        
        result = model.predict_team_corners(
            home_avg_corners=5.0,
            away_avg_corners=4.0,
        )
        
        assert "home_over_4" in result
        assert "away_over_4" in result
    
    def test_predict_more_corners(self):
        """Test more corners team prediction."""
        model = CornersModel()
        
        result = model.predict_more_corners(
            home_avg_corners=6.0,
            away_avg_corners=4.0,
        )
        
        assert "home" in result
        assert "away" in result
        assert "tie" in result
        
        # Home should have higher probability
        assert result["home"] > result["away"]
        
        # Probabilities should sum to ~1
        assert abs(sum(result.values()) - 1.0) < 0.01
    
    def test_poisson_survival_function(self):
        """Test Poisson SF utility function."""
        # P(X > 8) for lambda=9 should be > 0.5
        prob = poisson_sf(8, 9.0)
        assert prob > 0.5
        
        # P(X > 8) for lambda=5 should be < 0.5
        prob = poisson_sf(8, 5.0)
        assert prob < 0.5


# ==================================================
# 3. Test Cards Model
# ==================================================
class TestCardsModel:
    
    def test_predict_total_cards(self):
        """Test total cards prediction."""
        model = CardsModel()
        
        result = model.predict_total_cards(
            home_avg_cards=2.5,
            away_avg_cards=2.0,
        )
        
        assert "expected_total" in result
        assert "lines" in result
        
        # Check typical card lines
        assert "over_4" in result["lines"]
        assert "under_4" in result["lines"]
    
    def test_predict_team_cards(self):
        """Test team card totals."""
        model = CardsModel()
        
        result = model.predict_team_cards(
            home_avg_cards=2.5,
            away_avg_cards=2.0,
        )
        
        assert "home_over_1" in result
        assert "away_over_1" in result
    
    def test_predict_more_cards(self):
        """Test more cards team prediction."""
        model = CardsModel()
        
        result = model.predict_more_cards(
            home_avg_cards=3.0,
            away_avg_cards=2.0,
        )
        
        assert "home" in result
        assert "away" in result
        assert "tie" in result
        assert abs(sum(result.values()) - 1.0) < 0.01


# ==================================================
# 4. Test Shots Model
# ==================================================
class TestShotsModel:
    
    def test_predict_total_sot(self):
        """Test total SOT prediction."""
        model = ShotsModel()
        
        result = model.predict_total_sot(
            home_avg_sot=5.0,
            away_avg_sot=4.0,
        )
        
        assert "expected_total" in result
        assert "lines" in result
        assert abs(result["expected_total"] - 9.0) < 0.1
    
    def test_predict_team_sot(self):
        """Test team SOT totals."""
        model = ShotsModel()
        
        result = model.predict_team_sot(
            home_avg_sot=5.0,
            away_avg_sot=4.0,
        )
        
        assert "home_over_3" in result
        assert "away_over_3" in result


# ==================================================
# 5. Test Player Props Model
# ==================================================
class TestPlayerPropsModel:
    
    def test_predict_anytime_scorer_starter(self):
        """Test anytime scorer for starter."""
        model = PlayerPropsModel()
        
        player_data = {
            "goals": 5,
            "matches_played": 10,
            "shots": 20,
            "is_starter": True,
            "minutes": 90,
        }
        
        prob = model.predict_anytime_scorer(
            player_data=player_data,
            team_xg=15.0,
            team_total_shots=100,
        )
        
        assert 0.0 <= prob <= 1.0
        # Starter with good record should have reasonable probability
        assert prob > 0.1
    
    def test_predict_anytime_scorer_substitute(self):
        """Test anytime scorer for substitute."""
        model = PlayerPropsModel()
        
        player_data = {
            "goals": 2,
            "matches_played": 10,
            "shots": 8,
            "is_starter": False,
            "minutes": 30,
        }
        
        prob = model.predict_anytime_scorer(
            player_data=player_data,
            team_xg=15.0,
            team_total_shots=100,
        )
        
        assert 0.0 <= prob <= 1.0
        # Substitute should have lower probability than equivalent starter
    
    def test_predict_first_scorer(self):
        """Test first scorer prediction."""
        model = PlayerPropsModel()
        
        anytime_prob = 0.4
        
        starter_prob = model.predict_first_scorer(
            player_data={},
            anytime_prob=anytime_prob,
            is_starter=True,
        )
        
        sub_prob = model.predict_first_scorer(
            player_data={},
            anytime_prob=anytime_prob,
            is_starter=False,
        )
        
        # Starter should have higher first scorer probability
        assert starter_prob > sub_prob
        assert starter_prob < anytime_prob  # First scorer < anytime
    
    def test_predict_player_sot(self):
        """Test player SOT over/under."""
        model = PlayerPropsModel()
        
        player_data = {
            "shots": 20,
            "shots_on_target": 10,
            "matches_played": 10,
            "is_starter": True,
            "minutes": 90,
        }
        
        result = model.predict_player_sot(
            player_data=player_data,
            team_avg_sot=50.0,
        )
        
        assert "expected_sot" in result
        assert "lines" in result
        assert "over_0" in result["lines"]


# ==================================================
# 6. Test ESPN Stats Parsers
# ==================================================
class TestEspnStatsParsers:
    
    def test_extract_team_stats_empty(self):
        """Test parsing empty summary."""
        result = extract_team_stats_from_summary({})
        
        assert "home_corners" in result
        assert "away_corners" in result
        assert "home_yellow_cards" in result
        assert "home_shots_on_target" in result
    
    def test_extract_team_stats_with_boxscore(self):
        """Test parsing stats from boxscore."""
        summary = {
            "boxscore": {
                "teams": [
                    {
                        "homeAway": "home",
                        "score": 2,
                        "statistics": [
                            {"name": "Corners", "displayValue": 6},
                            {"name": "Yellow Cards", "displayValue": 2},
                            {"name": "Shots on Target", "displayValue": 5},
                        ]
                    },
                    {
                        "homeAway": "away",
                        "score": 1,
                        "statistics": [
                            {"name": "Corners", "displayValue": 4},
                            {"name": "Yellow Cards", "displayValue": 3},
                            {"name": "Shots on Target", "displayValue": 3},
                        ]
                    }
                ]
            }
        }
        
        result = extract_team_stats_from_summary(summary)
        
        assert result["home_corners"] == 6
        assert result["away_corners"] == 4
        assert result["home_yellow_cards"] == 2
        assert result["away_yellow_cards"] == 3
        assert result["home_shots_on_target"] == 5
    
    def test_extract_player_stats_empty(self):
        """Test parsing empty player stats."""
        result = extract_player_stats_from_summary({})
        assert result == []
    
    def test_extract_events_empty(self):
        """Test parsing empty events."""
        result = extract_events_from_summary({})
        assert result == []


# ==================================================
# 7. Test Cache Manager (Windows/Linux safe)
# ==================================================
class TestCacheManager:
    
    def test_cache_key_is_sha256(self):
        """Test cache keys are SHA256 hashes (safe for all platforms)."""
        manager = EspnCacheManager()
        
        key = manager._make_cache_key("scoreboard", {"dates": "20250101"})
        
        # SHA256 produces 64-character hex string
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)
        
        # Same input should produce same key
        key2 = manager._make_cache_key("scoreboard", {"dates": "20250101"})
        assert key == key2
        
        # Different input should produce different key
        key3 = manager._make_cache_key("scoreboard", {"dates": "20250102"})
        assert key != key3
    
    def test_cache_key_no_special_chars(self):
        """Test cache keys contain no special characters."""
        manager = EspnCacheManager()
        
        # Test with various params that might produce special chars
        params_list = [
            {"event": "401234567"},
            {"dates": "20250101-20250131", "limit": 100},
            {"league": "fifa.world", "season": "2026"},
        ]
        
        for params in params_list:
            key = manager._make_cache_key("summary", params)
            # Should only contain lowercase hex
            assert key.isalnum()
            assert key.islower()


# ==================================================
# 8. Test Market Output Structures
# ==================================================
class TestMarketOutputStructures:
    
    def test_market_availability_to_dict(self):
        """Test MarketAvailability serialization."""
        avail = MarketAvailability(
            available=False,
            reason="Insufficient data",
            sample_size=2,
            confidence=ConfidenceLevel.LOW,
            data_source=DataSource.UNAVAILABLE,
        )
        
        d = avail.to_dict()
        
        assert d["available"] == False
        assert d["reason"] == "Insufficient data"
        assert d["sample_size"] == 2
        assert d["confidence"] == "low"
        assert d["data_source"] == "unavailable"
    
    def test_all_markets_output_to_dict(self):
        """Test AllMarketsOutput serialization."""
        output = AllMarketsOutput(
            corners=CornersMarketPrediction(
                market_type=MarketType.CORNERS_TOTAL,
                availability=MarketAvailability(available=True, sample_size=5),
                predictions={"expected_total": 9.5},
            ),
            cards=None,  # Unavailable
            shots_on_target=None,
            player_props=None,
        )
        
        d = output.to_dict()
        
        assert "corners" in d
        assert "cards" not in d  # None values excluded
        assert d["corners"]["available"] == True
        assert d["corners"]["predictions"]["expected_total"] == 9.5
    
    def test_corners_prediction_structure(self):
        """Test corners prediction has expected fields."""
        avail = MarketAvailability(available=True, sample_size=6)
        pred = CornersMarketPrediction(
            market_type=MarketType.CORNERS_TOTAL,
            availability=avail,
            predictions={
                "total_over_under": {"over_8": 0.6, "under_8": 0.4},
                "team_totals": {"home_over_4": 0.55},
                "more_corners_team": {"home": 0.4, "away": 0.45, "tie": 0.15},
            }
        )
        
        d = pred.to_dict()
        
        assert d["available"] == True
        assert "total_over_under" in d["predictions"]
        assert "team_totals" in d["predictions"]
        assert "more_corners_team" in d["predictions"]


# ==================================================
# 9. Test Fallback Behavior
# ==================================================
class TestFallbackBehavior:
    
    def test_empty_matches_returns_fallback_stats(self):
        """Test that empty match list returns fallback stats."""
        from src.pipeline.predict_markets import MarketsPredictor
        
        predictor = MarketsPredictor()
        stats = predictor._empty_team_stats()
        
        assert stats["avg_corners_for"] == 5.0  # League average
        assert stats["avg_cards_for"] == 2.0
        assert stats["avg_sot_for"] == 4.0
    
    def test_unavailable_market_returns_proper_structure(self):
        """Test unavailable markets return proper structure."""
        from src.pipeline.predict_markets import MarketsPredictor
        
        predictor = MarketsPredictor()
        
        # No matches = unavailable
        result = predictor._predict_corners(
            home_stats=predictor._empty_team_stats(),
            away_stats=predictor._empty_team_stats(),
            has_event_data=False,
            home_matches=[],
            away_matches=[],
        )
        
        assert result.availability.available == False
        assert "Insufficient" in result.availability.reason
        assert result.predictions == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
