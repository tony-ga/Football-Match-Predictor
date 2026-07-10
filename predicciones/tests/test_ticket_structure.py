import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.ticket_structure import (
    MarketRelationType,
    classify_market_relation,
    evaluate_ticket_structure,
)


def test_detects_contradictory_markets():
    relation = classify_market_relation("1x2_home", "double_chance_away_or_draw")
    assert relation.relation == MarketRelationType.CONTRADICTORY


def test_detects_nested_redundancy_in_totals_chain():
    relation = classify_market_relation("over_1_5", "over_2_5")
    assert relation.relation == MarketRelationType.NESTED_REDUNDANT


def test_detects_weak_low_information_pair():
    relation = classify_market_relation("under_4_5", "over_1_5")
    assert relation.relation == MarketRelationType.WEAK_LOW_INFORMATION


def test_detects_complementary_pair():
    relation = classify_market_relation("btts_yes", "over_2_5")
    assert relation.relation == MarketRelationType.COMPLEMENTARY


def test_rejects_redundant_ticket_structure():
    evaluation = evaluate_ticket_structure(["over_1_5", "over_2_5", "over_3_5"])
    assert evaluation.is_valid is False
    assert evaluation.redundancy_penalty > 0.90
    assert any("redundant" in reason.lower() or "overlap" in reason.lower() for reason in evaluation.rejection_reasons)


def test_rejects_double_chance_plus_contained_1x2():
    evaluation = evaluate_ticket_structure(["double_chance_home_or_draw", "1x2_home"])
    assert evaluation.is_valid is False
    assert evaluation.redundancy_penalty >= 0.85


def test_rejects_overly_broad_low_information_total_range():
    evaluation = evaluate_ticket_structure(["under_4_5", "over_1_5"])
    assert evaluation.is_valid is False
    assert evaluation.weak_information_penalty >= 0.70


def test_accepts_complementary_diverse_ticket_structure():
    evaluation = evaluate_ticket_structure(["double_chance_home_or_draw", "under_2_5", "btts_no"])
    assert evaluation.is_valid is True
    assert evaluation.compatibility_score > 0.70
    assert evaluation.family_diversity_score >= 1.0
    assert evaluation.information_gain_score > 0.60
