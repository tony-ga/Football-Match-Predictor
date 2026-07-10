import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.market_evaluation import (
    MarketFilterConfig,
    MarketSelectionStatus,
    ProbabilityReference,
    compute_edge,
    compute_ev,
    evaluate_market_candidate,
    odds_to_implied_probability,
    remove_vig_three_way,
    remove_vig_two_way,
)


def test_odds_to_implied_probability():
    assert math.isclose(odds_to_implied_probability(2.0), 0.5)
    assert odds_to_implied_probability(None) is None


def test_remove_vig_two_way_normalizes_probabilities():
    result = remove_vig_two_way(1.80, 2.10)
    assert math.isclose(result["a"] + result["b"], 1.0, rel_tol=1e-9)
    assert result["overround"] > 0


def test_remove_vig_three_way_normalizes_probabilities():
    result = remove_vig_three_way(2.40, 3.20, 3.10)
    assert math.isclose(result["a"] + result["b"] + result["c"], 1.0, rel_tol=1e-9)
    assert result["overround"] > 0


def test_compute_ev_and_edge():
    assert math.isclose(compute_edge(0.60, 0.52), 0.08, rel_tol=1e-9)
    assert math.isclose(compute_ev(0.60, 2.0), 0.20, rel_tol=1e-9)


def test_evaluate_market_candidate_uses_no_vig_when_available():
    evaluation = evaluate_market_candidate(
        match_name="Mexico vs England",
        market_key="1x2_home",
        market_name="Mexico win",
        model_probability=0.45,
        confidence_score=0.62,
        sportsbook_odds={
            "1x2": {"home": 2.40, "draw": 3.20, "away": 3.10},
            "double_chance": {},
            "over_under": {},
            "btts": {},
        },
        filter_config=MarketFilterConfig(
            min_model_prob=0.35,
            min_edge=0.01,
            min_ev=-0.05,
            min_confidence=0.50,
        ),
    )

    assert evaluation.reference_probability_type == ProbabilityReference.NO_VIG_FAIR
    assert evaluation.no_vig_probability is not None
    assert evaluation.ev is not None
    assert evaluation.status == MarketSelectionStatus.ACCEPTED


def test_evaluate_market_candidate_marks_model_only_without_odds():
    evaluation = evaluate_market_candidate(
        match_name="Francia vs Marruecos",
        market_key="over_2_5",
        market_name="Over 2.5 goals",
        model_probability=0.59,
        confidence_score=0.57,
        sportsbook_odds=None,
        filter_config=MarketFilterConfig(
            min_model_prob=0.50,
            min_confidence=0.50,
        ),
    )

    assert evaluation.reference_probability_type == ProbabilityReference.MODEL_ONLY
    assert evaluation.ev is None
    assert evaluation.implied_probability is None
    assert evaluation.status == MarketSelectionStatus.ACCEPTED
