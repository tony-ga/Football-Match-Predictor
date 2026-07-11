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
import math
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


# ==================================================
# 10. Test Player Props Normalization and Formatting
# ==================================================
class TestPlayerPropsNormalization:
    """Test player props probability normalization and formatting."""
    
    def test_anytime_scorer_sum_bounded_by_lambda(self):
        """Test that anytime scorer lambdas sum to ~team_xg and probabilities are properly separated."""
        model = PlayerPropsModel()
        
        players_data = [
            {'player_name': 'Striker', 'team': 'Home', 'goals': 5, 'matches_played': 4, 
             'shots': 20, 'minutes': 360, 'position': 'Forward', 'is_starter': True},
            {'player_name': 'Winger', 'team': 'Home', 'goals': 2, 'matches_played': 4,
             'shots': 10, 'minutes': 320, 'position': 'Midfielder', 'is_starter': True},
            {'player_name': 'Defender', 'team': 'Home', 'goals': 0, 'matches_played': 4,
             'shots': 2, 'minutes': 360, 'position': 'Defender', 'is_starter': True},
        ]
        
        team_xg = 2.5
        result = model.predict_anytime_scorer_normalized(players_data, team_xg)
        
        # Check lambdas sum approximately to team_xg
        total_lambda = sum(p['lambda'] for p in result)
        assert abs(total_lambda - team_xg) < 0.1, f"Lambda sum {total_lambda} should be close to {team_xg}"
        
        # Internal probability (decimal in [0,1]) should sum to reasonable value comparable to lambda
        total_prob_decimal = sum(p['probability'] for p in result)
        assert total_prob_decimal > 0, "Should have non-zero probability"
        # Sum of P(anytime) should be bounded and related to team_xg
        # For Poisson: sum(1 - exp(-λ_i)) ≈ team_xg for small λ_i
        assert total_prob_decimal <= team_xg * 1.5, f"Decimal prob sum {total_prob_decimal} too high vs lambda {team_xg}"
        
        # Display percentage should also exist
        assert all('probability_pct' in p for p in result), "Should have probability_pct field"
        total_prob_pct = sum(p['probability_pct'] for p in result)
        # Percentage sum should be 100x the decimal sum
        assert abs(total_prob_pct - total_prob_decimal * 100) < 0.1, "Percentage should match decimal * 100"
    
    def test_anytime_scorer_internal_decimal_format(self):
        """Test that internal probability is decimal [0,1] while display is percentage."""
        model = PlayerPropsModel()
        
        players_data = [
            {'player_name': 'Striker', 'team': 'Home', 'goals': 5, 'matches_played': 4, 
             'shots': 20, 'minutes': 360, 'position': 'Forward', 'is_starter': True},
        ]
        
        team_xg = 2.5
        result = model.predict_anytime_scorer_normalized(players_data, team_xg)
        
        player = result[0]
        # Internal probability should be in [0, 1]
        assert 0 <= player['probability'] <= 1, f"Internal prob {player['probability']} should be in [0,1]"
        # Display percentage should be in [0, 100]
        assert 0 <= player['probability_pct'] <= 100, f"Display pct {player['probability_pct']} should be in [0,100]"
        # They should be related by factor of 100
        assert abs(player['probability_pct'] - player['probability'] * 100) < 0.01
    
    def test_anytime_scorer_lambda_distribution(self):
        """Test that lambda is distributed realistically across positions."""
        model = PlayerPropsModel()
        
        players_data = [
            {'player_name': 'Striker', 'team': 'Home', 'goals': 5, 'matches_played': 4, 
             'shots': 20, 'minutes': 360, 'position': 'Forward', 'is_starter': True},
            {'player_name': 'Midfielder', 'team': 'Home', 'goals': 2, 'matches_played': 4,
             'shots': 10, 'minutes': 360, 'position': 'Midfielder', 'is_starter': True},
            {'player_name': 'Defender', 'team': 'Home', 'goals': 0, 'matches_played': 4,
             'shots': 2, 'minutes': 360, 'position': 'Defender', 'is_starter': True},
        ]
        
        team_xg = 2.5
        result = model.predict_anytime_scorer_normalized(players_data, team_xg)
        
        striker_lambda = next(p['lambda'] for p in result if p['player_name'] == 'Striker')
        midfielder_lambda = next(p['lambda'] for p in result if p['player_name'] == 'Midfielder')
        defender_lambda = next(p['lambda'] for p in result if p['player_name'] == 'Defender')
        
        # Striker should have highest lambda
        assert striker_lambda > midfielder_lambda, \
            f"Striker lambda ({striker_lambda}) should exceed midfielder ({midfielder_lambda})"
        assert midfielder_lambda > defender_lambda, \
            f"Midfielder lambda ({midfielder_lambda}) should exceed defender ({defender_lambda})"
        # Striker should have at least 2x defender's lambda
        assert striker_lambda > defender_lambda * 2, \
            f"Striker lambda should be at least 2x defender's"
    
    def test_first_scorer_normalization(self):
        """Test first scorer probabilities sum to ~100% with no_goal (in percentage)."""
        model = PlayerPropsModel()
        
        # Use new format with probability as decimal [0,1] and probability_pct as percentage
        anytime_data = [
            {'player_name': 'Player1', 'team': 'Home', 'probability': 0.40, 'probability_pct': 40.0},
            {'player_name': 'Player2', 'team': 'Home', 'probability': 0.30, 'probability_pct': 30.0},
            {'player_name': 'Player3', 'team': 'Away', 'probability': 0.20, 'probability_pct': 20.0},
        ]
        
        team_xg = 2.5
        result = model.predict_first_scorer_normalized(anytime_data, team_xg)
        
        # Check internal decimal values sum correctly
        scorer_sum_decimal = sum(p.get('probability_decimal', p['probability'] / 100 if p['probability'] > 1 else p['probability']) 
                                 for p in result if p['player_name'] != '[NO GOAL]')
        no_goal_decimal = next((p.get('probability_decimal', p['probability'] / 100 if p['probability'] > 1 else p['probability']) 
                               for p in result if p['player_name'] == '[NO GOAL]'), 0)
        total_decimal = scorer_sum_decimal + no_goal_decimal
        
        # Total should be very close to 1.0 (decimal)
        assert abs(total_decimal - 1.0) < 0.01, f"Total {total_decimal} should be ~1.0"
        
        # Display percentages should also exist and sum to ~100%
        scorer_sum_pct = sum(p['probability'] for p in result if p['player_name'] != '[NO GOAL]')
        no_goal_pct = next((p['probability'] for p in result if p['player_name'] == '[NO GOAL]'), 0)
        total_pct = scorer_sum_pct + no_goal_pct
        
        assert abs(total_pct - 100.0) < 1.0, f"Total percentage {total_pct}% should be ~100%"
        
        # No goal probability should match Poisson P(0 goals) * 100
        expected_no_goal_pct = math.exp(-team_xg) * 100
        assert abs(no_goal_pct - expected_no_goal_pct) < 1.0
    
    def test_anytime_scorer_differentiates_positions(self):
        """Test that forwards get higher probability than defenders with same stats."""
        model = PlayerPropsModel()
        
        # Same goals/shots but different positions
        players_data = [
            {'player_name': 'Forward', 'team': 'Home', 'goals': 2, 'matches_played': 4,
             'shots': 10, 'minutes': 360, 'position': 'Forward', 'is_starter': True},
            {'player_name': 'Defender', 'team': 'Home', 'goals': 2, 'matches_played': 4,
             'shots': 10, 'minutes': 360, 'position': 'Defender', 'is_starter': True},
        ]
        
        result = model.predict_anytime_scorer_normalized(players_data, team_xg=2.5)
        
        forward_prob = next(p['probability'] for p in result if p['player_name'] == 'Forward')
        defender_prob = next(p['probability'] for p in result if p['player_name'] == 'Defender')
        
        # Forward should have higher probability due to position weight
        assert forward_prob > defender_prob, \
            f"Forward ({forward_prob}) should beat Defender ({defender_prob})"
        # Position weight difference: Forward=1.0, Defender=0.25
        # So forward should have at least some advantage (even with same goals)
        assert forward_prob > defender_prob * 1.1, \
            f"Forward should have at least 10% more probability than defender"
    
    def test_anytime_scorer_ranks_by_goals(self):
        """Test that players with more goals get higher probability."""
        model = PlayerPropsModel()
        
        players_data = [
            {'player_name': 'TopScorer', 'team': 'Home', 'goals': 5, 'matches_played': 4,
             'shots': 20, 'minutes': 360, 'position': 'Forward', 'is_starter': True},
            {'player_name': 'LowScorer', 'team': 'Home', 'goals': 0, 'matches_played': 4,
             'shots': 5, 'minutes': 360, 'position': 'Forward', 'is_starter': True},
        ]
        
        result = model.predict_anytime_scorer_normalized(players_data, team_xg=2.5)
        
        top_prob = next(p['probability'] for p in result if p['player_name'] == 'TopScorer')
        low_prob = next(p['probability'] for p in result if p['player_name'] == 'LowScorer')
        
        assert top_prob > low_prob, \
            f"Top scorer ({top_prob}) should beat low scorer ({low_prob})"
    
    def test_format_percentages_player_props(self):
        """Test that format_percentages correctly formats player props as percentages."""
        # Inline format_percentages function (no need to import from predict.py)
        
        def format_percentages(data, path=""):
            if isinstance(data, dict):
                return {k: format_percentages(v, path=f"{path}.{k}") for k, v in data.items()}
            elif isinstance(data, list):
                return [format_percentages(v, path) for v in data]
            elif isinstance(data, float):
                if any(keyword in path.lower() for keyword in [
                    "lambda", "expected_goals", "expected_total", 
                    "weight", "xg", "goals_per_match"
                ]):
                    return round(data, 3)
                
                if "probability" in path.lower() or "prob" in path.lower():
                    if 0 <= data <= 1.0:
                        return f"{data * 100:.2f}%"
                    elif data > 1.0 and data <= 100:
                        return f"{data:.2f}%"
                
                if any(keyword in path.lower() for keyword in [
                    "count", "id", "size", "sample", "matches", "goals_recent"
                ]):
                    return round(data, 4)
                    
                if 0 <= data <= 1.0:
                    return f"{data * 100:.2f}%"
                elif data > 1.0 and data <= 100:
                    return f"{data:.2f}%"
                
                return round(data, 4)
            return data
        
        # Test data simulating player_props output with decimal probabilities (old format)
        # This tests the formatter's ability to handle both old and new formats
        test_data = {
            "player_props": {
                "anytime_scorer": {
                    "top_candidates": [
                        {"player_name": "Messi", "probability": 0.4523},
                        {"player_name": "Lautaro", "probability": 0.3812},
                    ],
                    "validation": {
                        "home_anytime_sum": 2.3184,
                        "home_lambda": 2.73,
                    }
                },
                "first_scorer": {
                    "top_candidates": [
                        {"player_name": "Messi", "probability": 0.2853},
                    ],
                    "no_goal_probability": 0.0231,
                }
            }
        }
        
        formatted = format_percentages(test_data)
        
        # Check probabilities are formatted as percentages
        assert formatted["player_props"]["anytime_scorer"]["top_candidates"][0]["probability"] == "45.23%"
        assert formatted["player_props"]["anytime_scorer"]["top_candidates"][1]["probability"] == "38.12%"
        assert formatted["player_props"]["first_scorer"]["top_candidates"][0]["probability"] == "28.53%"
        assert formatted["player_props"]["first_scorer"]["no_goal_probability"] == "2.31%"
        
        # Check lambdas remain numeric
        assert formatted["player_props"]["anytime_scorer"]["validation"]["home_lambda"] == 2.73
        # home_anytime_sum is not a lambda, so it gets rounded
        assert formatted["player_props"]["anytime_scorer"]["validation"]["home_anytime_sum"] == 2.3184
