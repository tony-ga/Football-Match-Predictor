from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple


class MarketRelationType(Enum):
    CONTRADICTORY = "contradictory"
    NESTED_REDUNDANT = "nested_redundant"
    WEAK_LOW_INFORMATION = "weak_low_information"
    COMPLEMENTARY = "complementary"
    NEUTRAL = "neutral"


# Totals nesting hierarchy (strict ordering used for redundancy detection).
OVER_TOTALS_CHAIN: Tuple[float, ...] = (0.5, 1.5, 2.5, 3.5, 4.5)
UNDER_TOTALS_CHAIN: Tuple[float, ...] = (4.5, 3.5, 2.5, 1.5)

CORE_PARLAY_MARKETS: Tuple[str, ...] = (
    "1x2_home",
    "1x2_draw",
    "1x2_away",
    "double_chance_home_or_draw",
    "double_chance_away_or_draw",
    "double_chance_home_or_away",
    "over_1_5",
    "over_2_5",
    "over_3_5",
    "over_4_5",
    "under_1_5",
    "under_2_5",
    "under_3_5",
    "under_4_5",
    "btts_yes",
    "btts_no",
)


@dataclass
class PairRelation:
    leg_a: str
    leg_b: str
    relation: MarketRelationType
    strength: float
    reason: str


@dataclass
class TicketStructureEvaluation:
    pair_relations: List[PairRelation]
    contradiction_penalty: float
    redundancy_penalty: float
    weak_information_penalty: float
    information_gain_score: float
    family_diversity_score: float
    compatibility_score: float
    final_structure_score: float
    is_valid: bool
    rejection_reasons: List[str] = field(default_factory=list)


def _normalize_market_key(leg: Any) -> str:
    if isinstance(leg, str):
        return leg.replace(".", "_")
    if hasattr(leg, "market_key"):
        return str(getattr(leg, "market_key")).replace(".", "_")
    market_type = getattr(leg, "market_type", None)
    if market_type is not None and hasattr(market_type, "value"):
        return str(market_type.value).replace(".", "_")
    raise ValueError(f"Unsupported leg object for market relation classification: {leg!r}")


def _market_family(market_key: str) -> str:
    if market_key.startswith("1x2_") or market_key.startswith("double_chance_"):
        return "result"
    if market_key.startswith("over_") or market_key.startswith("under_"):
        return "totals"
    if market_key.startswith("btts_"):
        return "btts"
    return "other"


def _parse_total_market(market_key: str) -> Tuple[Optional[str], Optional[float]]:
    if market_key.startswith("over_"):
        return "over", float(market_key.split("_", 1)[1].replace("_", "."))
    if market_key.startswith("under_"):
        return "under", float(market_key.split("_", 1)[1].replace("_", "."))
    return None, None


def _totals_chain_rank(side: str, line: float) -> Optional[int]:
    chain = OVER_TOTALS_CHAIN if side == "over" else UNDER_TOTALS_CHAIN
    if line not in chain:
        return None
    return chain.index(line)


def _nested_totals_strength(side: str, line_a: float, line_b: float) -> float:
    rank_a = _totals_chain_rank(side, line_a)
    rank_b = _totals_chain_rank(side, line_b)
    if rank_a is None or rank_b is None:
        return 0.85
    distance = abs(rank_a - rank_b)
    if distance == 1:
        return 0.85
    if distance == 2:
        return 0.75
    return 0.70


def _double_chance_outcomes(market_key: str) -> Optional[set[str]]:
    mapping = {
        "double_chance_home_or_draw": {"home", "draw"},
        "double_chance_away_or_draw": {"away", "draw"},
        "double_chance_home_or_away": {"home", "away"},
    }
    return mapping.get(market_key)


def _single_outcome(market_key: str) -> Optional[str]:
    if market_key.startswith("1x2_"):
        return market_key.split("1x2_", 1)[1]
    return None


def _specificity_hint(market_key: str) -> float:
    hints = {
        "1x2_home": 0.75,
        "1x2_away": 0.75,
        "1x2_draw": 0.65,
        "double_chance_home_or_draw": 0.40,
        "double_chance_away_or_draw": 0.40,
        "double_chance_home_or_away": 0.30,
        "over_0_5": 0.10,
        "over_1_5": 0.20,
        "over_2_5": 0.50,
        "over_3_5": 0.70,
        "over_4_5": 0.85,
        "under_1_5": 0.80,
        "under_2_5": 0.55,
        "under_3_5": 0.35,
        "under_4_5": 0.15,
        "btts_yes": 0.55,
        "btts_no": 0.55,
    }
    return hints.get(market_key, 0.50)


def _count_same_chain_legs(market_keys: Iterable[str]) -> Dict[str, int]:
    counts = {"over": 0, "under": 0}
    for market_key in market_keys:
        side, line = _parse_total_market(market_key)
        if side is None or line is None:
            continue
        if _totals_chain_rank(side, line) is not None:
            counts[side] += 1
    return counts


def build_market_relation_matrix(
    markets: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, PairRelation]]:
    """
    Build the full pairwise relation matrix for core parlay markets.

    Returns a nested dict: matrix[leg_a][leg_b] -> PairRelation (leg_a <= leg_b lexicographically).
    """
    market_list = list(markets) if markets is not None else list(CORE_PARLAY_MARKETS)
    matrix: Dict[str, Dict[str, PairRelation]] = {}
    for leg_a in market_list:
        matrix[leg_a] = {}
        for leg_b in market_list:
            if leg_a == leg_b:
                continue
            relation = classify_market_relation(leg_a, leg_b)
            matrix[leg_a][leg_b] = relation
    return matrix


def classify_market_relation(leg_a: Any, leg_b: Any) -> PairRelation:
    market_a = _normalize_market_key(leg_a)
    market_b = _normalize_market_key(leg_b)
    ordered_a, ordered_b = sorted([market_a, market_b])

    if market_a == market_b:
        return PairRelation(ordered_a, ordered_b, MarketRelationType.NESTED_REDUNDANT, 1.0, "Duplicate market.")

    single_a = _single_outcome(market_a)
    single_b = _single_outcome(market_b)
    dc_a = _double_chance_outcomes(market_a)
    dc_b = _double_chance_outcomes(market_b)
    total_side_a, total_line_a = _parse_total_market(market_a)
    total_side_b, total_line_b = _parse_total_market(market_b)

    contradictory_pairs = {
        frozenset({"1x2_home", "1x2_away"}),
        frozenset({"1x2_home", "1x2_draw"}),
        frozenset({"1x2_away", "1x2_draw"}),
        frozenset({"btts_yes", "btts_no"}),
    }
    if frozenset({market_a, market_b}) in contradictory_pairs:
        return PairRelation(ordered_a, ordered_b, MarketRelationType.CONTRADICTORY, 1.0, "Mutually exclusive outcomes.")

    if single_a and dc_b and single_a not in dc_b:
        return PairRelation(ordered_a, ordered_b, MarketRelationType.CONTRADICTORY, 1.0, "1X2 outcome conflicts with double chance coverage.")
    if single_b and dc_a and single_b not in dc_a:
        return PairRelation(ordered_a, ordered_b, MarketRelationType.CONTRADICTORY, 1.0, "1X2 outcome conflicts with double chance coverage.")

    if dc_a and dc_b and dc_a.isdisjoint(dc_b):
        return PairRelation(ordered_a, ordered_b, MarketRelationType.CONTRADICTORY, 1.0, "Double chance markets do not overlap.")

    if total_side_a and total_side_b and total_side_a != total_side_b:
        if total_side_a == "over" and total_line_a >= total_line_b:
            return PairRelation(ordered_a, ordered_b, MarketRelationType.CONTRADICTORY, 1.0, "Over line exceeds or matches under line.")
        if total_side_b == "over" and total_line_b >= total_line_a:
            return PairRelation(ordered_a, ordered_b, MarketRelationType.CONTRADICTORY, 1.0, "Over line exceeds or matches under line.")

    if "btts_yes" in {market_a, market_b}:
        other = market_b if market_a == "btts_yes" else market_a
        other_side, other_line = _parse_total_market(other)
        if other_side == "under" and other_line <= 1.5:
            return PairRelation(ordered_a, ordered_b, MarketRelationType.CONTRADICTORY, 1.0, "BTTS Yes cannot coexist with Under 1.5.")

    if total_side_a and total_side_b and total_side_a == total_side_b:
        strength = _nested_totals_strength(total_side_a, total_line_a, total_line_b)
        return PairRelation(
            ordered_a,
            ordered_b,
            MarketRelationType.NESTED_REDUNDANT,
            strength,
            "Both legs belong to the same totals chain and overlap heavily.",
        )

    if single_a and dc_b and single_a in dc_b:
        return PairRelation(ordered_a, ordered_b, MarketRelationType.NESTED_REDUNDANT, 0.90, "Double chance partially contains the 1X2 outcome.")
    if single_b and dc_a and single_b in dc_a:
        return PairRelation(ordered_a, ordered_b, MarketRelationType.NESTED_REDUNDANT, 0.90, "Double chance partially contains the 1X2 outcome.")

    if total_side_a and total_side_b and total_side_a != total_side_b:
        lower = min(total_line_a, total_line_b)
        upper = max(total_line_a, total_line_b)
        if (upper - lower) >= 2.0:
            return PairRelation(
                ordered_a,
                ordered_b,
                MarketRelationType.WEAK_LOW_INFORMATION,
                0.70,
                "The total-goals corridor is too wide and adds little script specificity.",
            )

    weak_pairs = {
        frozenset({"over_1_5", "btts_yes"}): ("Over 1.5 adds little beyond BTTS Yes in many scripts.", 0.45),
        frozenset({"under_4_5", "over_1_5"}): ("This broad goals range is valid but weakly informative.", 0.80),
    }
    weak_match = weak_pairs.get(frozenset({market_a, market_b}))
    if weak_match:
        reason, strength = weak_match
        return PairRelation(ordered_a, ordered_b, MarketRelationType.WEAK_LOW_INFORMATION, strength, reason)

    complementary_pairs = {
        frozenset({"btts_yes", "over_2_5"}): ("BTTS Yes adds distribution while Over 2.5 adds scoring volume.", 0.90),
        frozenset({"btts_yes", "over_3_5"}): ("Open match script with both distribution and higher total volume.", 0.82),
        frozenset({"1x2_home", "over_1_5"}): ("Winner plus minimum scoring volume.", 0.68),
        frozenset({"1x2_away", "over_1_5"}): ("Winner plus minimum scoring volume.", 0.68),
        frozenset({"1x2_home", "btts_no"}): ("Home win with clean-sheet script.", 0.78),
        frozenset({"1x2_away", "btts_no"}): ("Away win with clean-sheet script.", 0.78),
        frozenset({"double_chance_home_or_draw", "under_3_5"}): ("Result protection plus controlled scoring script.", 0.82),
        frozenset({"double_chance_away_or_draw", "under_3_5"}): ("Result protection plus controlled scoring script.", 0.82),
        frozenset({"double_chance_home_or_draw", "under_2_5"}): ("More specific low-event script with protection.", 0.86),
        frozenset({"double_chance_away_or_draw", "under_2_5"}): ("More specific low-event script with protection.", 0.86),
        frozenset({"under_2_5", "btts_no"}): ("Low-scoring script reinforced by one-sided distribution.", 0.74),
    }
    comp_match = complementary_pairs.get(frozenset({market_a, market_b}))
    if comp_match:
        reason, strength = comp_match
        return PairRelation(ordered_a, ordered_b, MarketRelationType.COMPLEMENTARY, strength, reason)

    if _market_family(market_a) != _market_family(market_b):
        return PairRelation(ordered_a, ordered_b, MarketRelationType.NEUTRAL, 0.55, "Different market families without clear conflict.")
    return PairRelation(ordered_a, ordered_b, MarketRelationType.NEUTRAL, 0.35, "Same family without hard contradiction or nesting.")


def compute_contradiction_penalty(relations: Iterable[PairRelation]) -> float:
    penalty = sum(relation.strength for relation in relations if relation.relation == MarketRelationType.CONTRADICTORY)
    return min(1.0, penalty)


def compute_redundancy_penalty(relations: Iterable[PairRelation]) -> float:
    penalty = sum(relation.strength for relation in relations if relation.relation == MarketRelationType.NESTED_REDUNDANT)
    return min(1.0, penalty)


def compute_weak_information_penalty(relations: Iterable[PairRelation]) -> float:
    penalty = sum(relation.strength for relation in relations if relation.relation == MarketRelationType.WEAK_LOW_INFORMATION)
    return min(1.0, penalty)


def compute_information_gain(legs: Iterable[Any], relations: Iterable[PairRelation]) -> float:
    market_keys = [_normalize_market_key(leg) for leg in legs]
    if len(market_keys) <= 1:
        return 1.0

    base_specificity = sum(_specificity_hint(key) for key in market_keys) / len(market_keys)
    pair_scores = []
    for relation in relations:
        if relation.relation == MarketRelationType.CONTRADICTORY:
            pair_scores.append(0.0)
        elif relation.relation == MarketRelationType.NESTED_REDUNDANT:
            pair_scores.append(max(0.0, 0.18 - (relation.strength * 0.10)))
        elif relation.relation == MarketRelationType.WEAK_LOW_INFORMATION:
            pair_scores.append(max(0.0, 0.30 - (relation.strength * 0.15)))
        elif relation.relation == MarketRelationType.COMPLEMENTARY:
            pair_scores.append(min(1.0, 0.72 + (relation.strength * 0.20)))
        else:
            pair_scores.append(min(1.0, 0.45 + (relation.strength * 0.25)))

    pair_component = sum(pair_scores) / len(pair_scores) if pair_scores else 1.0
    return min(1.0, (base_specificity * 0.45) + (pair_component * 0.55))


def compute_family_diversity(legs: Iterable[Any]) -> float:
    market_keys = [_normalize_market_key(leg) for leg in legs]
    if not market_keys:
        return 0.0

    families = [_market_family(key) for key in market_keys]
    unique_families = len(set(families))
    target = min(len(market_keys), 3)
    diversity = unique_families / max(1, target)

    max_family_count = max(families.count(family) for family in set(families))
    if max_family_count >= 3:
        diversity *= 0.65
    return min(1.0, diversity)


def compute_compatibility_score(relations: Iterable[PairRelation]) -> float:
    relations = list(relations)
    if not relations:
        return 1.0

    pair_scores = []
    for relation in relations:
        if relation.relation == MarketRelationType.CONTRADICTORY:
            pair_scores.append(0.0)
        elif relation.relation == MarketRelationType.NESTED_REDUNDANT:
            pair_scores.append(max(0.0, 0.18 - relation.strength * 0.10))
        elif relation.relation == MarketRelationType.WEAK_LOW_INFORMATION:
            pair_scores.append(max(0.0, 0.35 - relation.strength * 0.15))
        elif relation.relation == MarketRelationType.COMPLEMENTARY:
            pair_scores.append(min(1.0, 0.75 + relation.strength * 0.20))
        else:
            pair_scores.append(min(1.0, 0.55 + relation.strength * 0.20))
    return sum(pair_scores) / len(pair_scores)


def evaluate_ticket_structure(legs: Iterable[Any]) -> TicketStructureEvaluation:
    normalized_legs = [_normalize_market_key(leg) for leg in legs]
    pair_relations: List[PairRelation] = []
    for index, leg_a in enumerate(normalized_legs):
        for leg_b in normalized_legs[index + 1:]:
            pair_relations.append(classify_market_relation(leg_a, leg_b))

    contradiction_penalty = compute_contradiction_penalty(pair_relations)
    redundancy_penalty = compute_redundancy_penalty(pair_relations)
    weak_information_penalty = compute_weak_information_penalty(pair_relations)
    information_gain_score = compute_information_gain(normalized_legs, pair_relations)
    family_diversity_score = compute_family_diversity(normalized_legs)
    compatibility_score = compute_compatibility_score(pair_relations)

    rejection_reasons: List[str] = []
    families = [_market_family(key) for key in normalized_legs]
    chain_counts = _count_same_chain_legs(normalized_legs)

    if contradiction_penalty >= 1.0:
        rejection_reasons.append("Ticket contains contradictory legs.")
    if redundancy_penalty >= 0.85:
        rejection_reasons.append("Ticket contains nested or redundant legs with excessive overlap.")
    if chain_counts["over"] >= 2:
        rejection_reasons.append("Ticket stacks nested Over legs on the same totals chain.")
    if chain_counts["under"] >= 2:
        rejection_reasons.append("Ticket stacks nested Under legs on the same totals chain.")
    if weak_information_penalty >= 0.70 and information_gain_score < 0.25:
        rejection_reasons.append("Ticket is too broad and adds little information.")
    if families and max(families.count(family) for family in set(families)) >= 3:
        rejection_reasons.append("Ticket over-concentrates legs in a single market family.")

    final_structure_score = (
        (compatibility_score * 0.35) +
        (information_gain_score * 0.30) +
        (family_diversity_score * 0.20) -
        (contradiction_penalty * 0.90) -
        (redundancy_penalty * 0.55) -
        (weak_information_penalty * 0.35)
    )
    final_structure_score = max(0.0, min(1.0, final_structure_score))

    return TicketStructureEvaluation(
        pair_relations=pair_relations,
        contradiction_penalty=contradiction_penalty,
        redundancy_penalty=redundancy_penalty,
        weak_information_penalty=weak_information_penalty,
        information_gain_score=information_gain_score,
        family_diversity_score=family_diversity_score,
        compatibility_score=compatibility_score,
        final_structure_score=final_structure_score,
        is_valid=len(rejection_reasons) == 0,
        rejection_reasons=rejection_reasons,
    )
