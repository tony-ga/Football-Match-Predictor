import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.ticket_structure import (
    CORE_PARLAY_MARKETS,
    MarketRelationType,
    build_market_relation_matrix,
    classify_market_relation,
    evaluate_ticket_structure,
)


def test_build_market_relation_matrix_covers_core_markets():
    matrix = build_market_relation_matrix()
    assert set(matrix.keys()) == set(CORE_PARLAY_MARKETS)
    for market_a in CORE_PARLAY_MARKETS:
        for market_b in CORE_PARLAY_MARKETS:
            if market_a == market_b:
                continue
            assert market_b in matrix[market_a]
            assert matrix[market_a][market_b].relation in MarketRelationType


def test_detects_contradictory_home_win_vs_away_or_draw():
    relation = classify_market_relation("1x2_home", "double_chance_away_or_draw")
    assert relation.relation == MarketRelationType.CONTRADICTORY


def test_detects_contradictory_under_1_5_vs_btts_yes():
    relation = classify_market_relation("under_1_5", "btts_yes")
    assert relation.relation == MarketRelationType.CONTRADICTORY


def test_detects_contradictory_over_3_5_vs_under_2_5():
    relation = classify_market_relation("over_3_5", "under_2_5")
    assert relation.relation == MarketRelationType.CONTRADICTORY


def test_detects_nested_redundancy_over_chain():
    relation = classify_market_relation("over_1_5", "over_2_5")
    assert relation.relation == MarketRelationType.NESTED_REDUNDANT


def test_detects_nested_redundancy_under_chain():
    relation = classify_market_relation("under_4_5", "under_3_5")
    assert relation.relation == MarketRelationType.NESTED_REDUNDANT


def test_detects_nested_redundancy_double_chance_contains_1x2():
    relation = classify_market_relation("double_chance_home_or_draw", "1x2_home")
    assert relation.relation == MarketRelationType.NESTED_REDUNDANT


def test_detects_weak_low_information_pair():
    relation = classify_market_relation("under_4_5", "over_1_5")
    assert relation.relation == MarketRelationType.WEAK_LOW_INFORMATION


def test_detects_complementary_btts_yes_over_2_5():
    relation = classify_market_relation("btts_yes", "over_2_5")
    assert relation.relation == MarketRelationType.COMPLEMENTARY


def test_detects_complementary_home_win_over_1_5():
    relation = classify_market_relation("1x2_home", "over_1_5")
    assert relation.relation == MarketRelationType.COMPLEMENTARY


def test_detects_complementary_home_or_draw_under_3_5():
    relation = classify_market_relation("double_chance_home_or_draw", "under_3_5")
    assert relation.relation == MarketRelationType.COMPLEMENTARY


def test_rejects_redundant_ticket_over_1_5_over_2_5_over_3_5():
    evaluation = evaluate_ticket_structure(["over_1_5", "over_2_5", "over_3_5"])
    assert evaluation.is_valid is False
    assert evaluation.redundancy_penalty >= 0.85
    assert any("nested" in reason.lower() or "overlap" in reason.lower() or "over legs" in reason.lower()
               for reason in evaluation.rejection_reasons)


def test_rejects_double_chance_plus_contained_1x2():
    evaluation = evaluate_ticket_structure(["double_chance_home_or_draw", "1x2_home"])
    assert evaluation.is_valid is False
    assert evaluation.redundancy_penalty >= 0.85


def test_rejects_overly_broad_low_information_total_range():
    evaluation = evaluate_ticket_structure(["under_4_5", "over_1_5"])
    assert evaluation.is_valid is False
    assert evaluation.weak_information_penalty >= 0.70


def test_rejects_stacked_under_chain():
    evaluation = evaluate_ticket_structure(["under_4_5", "under_3_5", "under_2_5"])
    assert evaluation.is_valid is False
    assert any("under legs" in reason.lower() for reason in evaluation.rejection_reasons)


def test_accepts_complementary_btts_yes_over_2_5():
    evaluation = evaluate_ticket_structure(["btts_yes", "over_2_5"])
    assert evaluation.is_valid is True
    assert evaluation.compatibility_score > 0.70
    assert evaluation.family_diversity_score >= 0.65


def test_accepts_complementary_diverse_ticket_structure():
    evaluation = evaluate_ticket_structure(["double_chance_home_or_draw", "under_2_5", "btts_no"])
    assert evaluation.is_valid is True
    assert evaluation.compatibility_score > 0.70
    assert evaluation.family_diversity_score >= 1.0
    assert evaluation.information_gain_score > 0.60


def test_rejects_contradictory_ticket():
    evaluation = evaluate_ticket_structure(["1x2_home", "double_chance_away_or_draw"])
    assert evaluation.is_valid is False
    assert evaluation.contradiction_penalty >= 1.0
