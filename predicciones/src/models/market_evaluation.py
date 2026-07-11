from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class MarketSelectionStatus(Enum):
    ACCEPTED = "accepted"
    MARGINAL = "marginal"
    DISCARDED = "discarded"


class ProbabilityReference(Enum):
    RAW_IMPLIED = "raw_implied"
    NO_VIG_FAIR = "no_vig_fair"
    DERIVED_NO_VIG = "derived_no_vig"
    MODEL_ONLY = "model_only"


@dataclass
class MarketFilterConfig:
    min_model_prob: float = 0.50
    min_edge: float = 0.02
    min_ev: float = 0.0
    max_odds: Optional[float] = None
    min_confidence: float = 0.50
    marginal_tolerance: float = 0.01


@dataclass
class EvaluatedMarket:
    match_name: str
    market_key: str
    market_name: str
    market_family: str
    model_probability: float
    confidence_score: Optional[float]
    sportsbook_odds: Optional[float]
    implied_probability: Optional[float]
    no_vig_probability: Optional[float]
    reference_probability: Optional[float]
    reference_probability_type: ProbabilityReference
    edge: Optional[float]
    ev: Optional[float]
    status: MarketSelectionStatus
    reasons: List[str] = field(default_factory=list)

    @property
    def is_model_only(self) -> bool:
        return self.reference_probability_type == ProbabilityReference.MODEL_ONLY


def odds_to_implied_probability(decimal_odds: Optional[float]) -> Optional[float]:
    if decimal_odds is None or decimal_odds <= 1.0:
        return None
    return 1.0 / decimal_odds


def remove_vig_two_way(odds_a: float, odds_b: float) -> Dict[str, float]:
    implied_a = odds_to_implied_probability(odds_a)
    implied_b = odds_to_implied_probability(odds_b)
    if implied_a is None or implied_b is None:
        raise ValueError("Two-way vig removal requires valid decimal odds > 1.0.")

    total = implied_a + implied_b
    return {
        "a": implied_a / total,
        "b": implied_b / total,
        "overround": total - 1.0,
    }


def remove_vig_three_way(odds_a: float, odds_b: float, odds_c: float) -> Dict[str, float]:
    implied_a = odds_to_implied_probability(odds_a)
    implied_b = odds_to_implied_probability(odds_b)
    implied_c = odds_to_implied_probability(odds_c)
    if implied_a is None or implied_b is None or implied_c is None:
        raise ValueError("Three-way vig removal requires valid decimal odds > 1.0.")

    total = implied_a + implied_b + implied_c
    return {
        "a": implied_a / total,
        "b": implied_b / total,
        "c": implied_c / total,
        "overround": total - 1.0,
    }


def compute_edge(model_prob: float, reference_prob: Optional[float]) -> Optional[float]:
    if reference_prob is None:
        return None
    return model_prob - reference_prob


def compute_ev(model_prob: float, decimal_odds: Optional[float]) -> Optional[float]:
    if decimal_odds is None or decimal_odds <= 1.0:
        return None
    return (model_prob * decimal_odds) - 1.0


def _classify_threshold(
    value: Optional[float],
    threshold: Optional[float],
    tolerance: float,
    higher_is_better: bool = True,
) -> Optional[str]:
    if value is None or threshold is None:
        return None

    gap = (value - threshold) if higher_is_better else (threshold - value)
    if gap >= 0:
        return "pass"
    if gap >= -tolerance:
        return "marginal"
    return "fail"


def _market_family_from_key(market_key: str) -> str:
    if market_key.startswith("1x2_") or market_key.startswith("double_chance_"):
        return "result"
    if market_key.startswith("over_") or market_key.startswith("under_"):
        return "totals"
    if market_key.startswith("btts_"):
        return "btts"
    return "other"


def _get_reference_from_sportsbook(
    market_key: str,
    sportsbook_odds: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    fallback = {
        "sportsbook_odds": None,
        "implied_probability": None,
        "no_vig_probability": None,
        "reference_probability": None,
        "reference_probability_type": ProbabilityReference.MODEL_ONLY,
    }
    if not sportsbook_odds:
        return fallback

    one_x_two = sportsbook_odds.get("1x2") or {}
    double_chance = sportsbook_odds.get("double_chance") or {}
    over_under = sportsbook_odds.get("over_under") or {}
    btts = sportsbook_odds.get("btts") or {}

    if market_key.startswith("1x2_"):
        selection = market_key.split("1x2_", 1)[1]
        decimal_odds = one_x_two.get(selection)
        implied = odds_to_implied_probability(decimal_odds)
        fair = None
        ref_type = ProbabilityReference.RAW_IMPLIED if implied is not None else ProbabilityReference.MODEL_ONLY
        if all(one_x_two.get(k) for k in ("home", "draw", "away")):
            fair_probs = remove_vig_three_way(one_x_two["home"], one_x_two["draw"], one_x_two["away"])
            fair = fair_probs[{"home": "a", "draw": "b", "away": "c"}[selection]]
            ref_type = ProbabilityReference.NO_VIG_FAIR
        return {
            "sportsbook_odds": decimal_odds,
            "implied_probability": implied,
            "no_vig_probability": fair,
            "reference_probability": fair if fair is not None else implied,
            "reference_probability_type": ref_type,
        }

    if market_key.startswith("double_chance_"):
        selection = market_key.split("double_chance_", 1)[1]
        decimal_odds = double_chance.get(selection)
        implied = odds_to_implied_probability(decimal_odds)
        fair = None
        ref_type = ProbabilityReference.MODEL_ONLY

        if selection in ("home_or_draw", "away_or_draw", "home_or_away"):
            if all(one_x_two.get(k) for k in ("home", "draw", "away")):
                fair_probs = remove_vig_three_way(one_x_two["home"], one_x_two["draw"], one_x_two["away"])
                if selection == "home_or_draw":
                    fair = fair_probs["a"] + fair_probs["b"]
                elif selection == "away_or_draw":
                    fair = fair_probs["b"] + fair_probs["c"]
                else:
                    fair = fair_probs["a"] + fair_probs["c"]
                ref_type = ProbabilityReference.DERIVED_NO_VIG

        if decimal_odds is not None:
            ref_type = ProbabilityReference.RAW_IMPLIED
            if selection in ("home_or_draw", "away_or_draw", "home_or_away"):
                opposite_map = {
                    "home_or_draw": "away",
                    "away_or_draw": "home",
                    "home_or_away": "draw",
                }
                opposite = opposite_map[selection]
                opposite_odds = one_x_two.get(opposite)
                if opposite_odds:
                    two_way = remove_vig_two_way(decimal_odds, opposite_odds)
                    fair = two_way["a"]
                    ref_type = ProbabilityReference.NO_VIG_FAIR

        return {
            "sportsbook_odds": decimal_odds,
            "implied_probability": implied,
            "no_vig_probability": fair,
            "reference_probability": fair if fair is not None else implied if implied is not None else fair,
            "reference_probability_type": ref_type,
        }

    if market_key.startswith(("over_", "under_")):
        side, line = market_key.split("_", 1)
        line_odds = over_under.get(line) or {}
        decimal_odds = line_odds.get(side)
        opposite_side = "under" if side == "over" else "over"
        opposite_odds = line_odds.get(opposite_side)
        implied = odds_to_implied_probability(decimal_odds)
        fair = None
        ref_type = ProbabilityReference.RAW_IMPLIED
        if decimal_odds and opposite_odds:
            two_way = remove_vig_two_way(decimal_odds, opposite_odds)
            fair = two_way["a"]
            ref_type = ProbabilityReference.NO_VIG_FAIR
        return {
            "sportsbook_odds": decimal_odds,
            "implied_probability": implied,
            "no_vig_probability": fair,
            "reference_probability": fair if fair is not None else implied,
            "reference_probability_type": ref_type if implied is not None else ProbabilityReference.MODEL_ONLY,
        }

    if market_key.startswith("btts_"):
        selection = market_key.split("btts_", 1)[1]
        decimal_odds = btts.get(selection)
        opposite = "no" if selection == "yes" else "yes"
        opposite_odds = btts.get(opposite)
        implied = odds_to_implied_probability(decimal_odds)
        fair = None
        ref_type = ProbabilityReference.RAW_IMPLIED
        if decimal_odds and opposite_odds:
            two_way = remove_vig_two_way(decimal_odds, opposite_odds)
            fair = two_way["a"]
            ref_type = ProbabilityReference.NO_VIG_FAIR
        return {
            "sportsbook_odds": decimal_odds,
            "implied_probability": implied,
            "no_vig_probability": fair,
            "reference_probability": fair if fair is not None else implied,
            "reference_probability_type": ref_type if implied is not None else ProbabilityReference.MODEL_ONLY,
        }

    return fallback


def evaluate_market_candidate(
    match_name: str,
    market_key: str,
    market_name: str,
    model_probability: float,
    sportsbook_odds: Optional[Dict[str, Any]] = None,
    confidence_score: Optional[float] = None,
    filter_config: Optional[MarketFilterConfig] = None,
) -> EvaluatedMarket:
    config = filter_config or MarketFilterConfig()
    reference = _get_reference_from_sportsbook(market_key, sportsbook_odds)

    edge = compute_edge(model_probability, reference["reference_probability"])
    ev = compute_ev(model_probability, reference["sportsbook_odds"])

    checks = {
        "model_probability": _classify_threshold(
            model_probability,
            config.min_model_prob,
            config.marginal_tolerance,
            higher_is_better=True,
        ),
        "confidence_score": _classify_threshold(
            confidence_score,
            config.min_confidence,
            config.marginal_tolerance,
            higher_is_better=True,
        ),
        "edge": _classify_threshold(
            edge,
            config.min_edge,
            config.marginal_tolerance,
            higher_is_better=True,
        ),
        "ev": _classify_threshold(
            ev,
            config.min_ev,
            config.marginal_tolerance,
            higher_is_better=True,
        ),
        "odds": _classify_threshold(
            reference["sportsbook_odds"],
            config.max_odds,
            config.marginal_tolerance,
            higher_is_better=False,
        ),
    }

    reasons: List[str] = []
    failed = []
    marginal = []
    for field_name, outcome in checks.items():
        if outcome == "fail":
            failed.append(field_name)
        elif outcome == "marginal":
            marginal.append(field_name)

    if reference["reference_probability_type"] == ProbabilityReference.MODEL_ONLY:
        reasons.append("Model-only evaluation: sportsbook odds unavailable for this market.")
    elif reference["reference_probability_type"] == ProbabilityReference.NO_VIG_FAIR:
        reasons.append("Using no-vig fair probability as benchmark.")
    elif reference["reference_probability_type"] == ProbabilityReference.DERIVED_NO_VIG:
        reasons.append("Using no-vig benchmark derived from complementary 1X2 prices.")
    else:
        reasons.append("Using raw implied probability as benchmark.")

    if failed:
        reasons.append(f"Failed thresholds: {', '.join(failed)}.")
        status = MarketSelectionStatus.DISCARDED
    elif marginal:
        reasons.append(f"Near thresholds: {', '.join(marginal)}.")
        status = MarketSelectionStatus.MARGINAL
    else:
        reasons.append("Passed configured quality filters.")
        status = MarketSelectionStatus.ACCEPTED

    return EvaluatedMarket(
        match_name=match_name,
        market_key=market_key,
        market_name=market_name,
        market_family=_market_family_from_key(market_key),
        model_probability=model_probability,
        confidence_score=confidence_score,
        sportsbook_odds=reference["sportsbook_odds"],
        implied_probability=reference["implied_probability"],
        no_vig_probability=reference["no_vig_probability"],
        reference_probability=reference["reference_probability"],
        reference_probability_type=reference["reference_probability_type"],
        edge=edge,
        ev=ev,
        status=status,
        reasons=reasons,
    )


def filter_ev_positive_markets(
    evaluations: Iterable[EvaluatedMarket],
    *,
    accepted_statuses: Optional[Iterable[MarketSelectionStatus]] = None,
    require_real_ev: bool = False,
) -> List[EvaluatedMarket]:
    allowed = set(accepted_statuses or [MarketSelectionStatus.ACCEPTED, MarketSelectionStatus.MARGINAL])
    filtered: List[EvaluatedMarket] = []
    for evaluation in evaluations:
        if evaluation.status not in allowed:
            continue
        if require_real_ev and evaluation.ev is None:
            continue
        if evaluation.ev is not None and evaluation.ev < 0:
            continue
        filtered.append(evaluation)
    return filtered


def evaluate_core_market_set(
    match_name: str,
    home_team: str,
    away_team: str,
    predictions: Dict[str, Any],
    sportsbook_odds: Optional[Dict[str, Any]] = None,
    confidence_scores: Optional[Dict[str, float]] = None,
    filter_config: Optional[MarketFilterConfig] = None,
) -> List[EvaluatedMarket]:
    confidence_scores = confidence_scores or {}
    market_specs = [
        ("1x2_home", f"Victoria {home_team}", predictions["1x2"]["home"]),
        ("1x2_draw", "Empate", predictions["1x2"]["draw"]),
        ("1x2_away", f"Victoria {away_team}", predictions["1x2"]["away"]),
        ("double_chance_home_or_draw", f"{home_team} o Empate", predictions["double_chance"]["home_or_draw"]),
        ("double_chance_away_or_draw", f"{away_team} o Empate", predictions["double_chance"]["away_or_draw"]),
        ("double_chance_home_or_away", f"{home_team} o {away_team}", predictions["double_chance"]["home_or_away"]),
        ("over_1_5", "Más de 1.5 goles", predictions["over_under"]["over_1_5"]),
        ("under_1_5", "Menos de 1.5 goles", predictions["over_under"]["under_1_5"]),
        ("over_2_5", "Más de 2.5 goles", predictions["over_under"]["over_2_5"]),
        ("under_2_5", "Menos de 2.5 goles", predictions["over_under"]["under_2_5"]),
        ("over_3_5", "Más de 3.5 goles", predictions["over_under"]["over_3_5"]),
        ("under_3_5", "Menos de 3.5 goles", predictions["over_under"]["under_3_5"]),
        ("over_4_5", "Más de 4.5 goles", predictions["over_under"]["over_4_5"]),
        ("under_4_5", "Menos de 4.5 goles", predictions["over_under"]["under_4_5"]),
        ("btts_yes", "BTTS: Sí", predictions["btts"]["yes"]),
        ("btts_no", "BTTS: No", predictions["btts"]["no"]),
    ]

    evaluations = [
        evaluate_market_candidate(
            match_name=match_name,
            market_key=market_key,
            market_name=market_name,
            model_probability=model_prob,
            sportsbook_odds=sportsbook_odds,
            confidence_score=confidence_scores.get(market_key),
            filter_config=filter_config,
        )
        for market_key, market_name, model_prob in market_specs
    ]
    evaluations.sort(
        key=lambda item: (
            item.status != MarketSelectionStatus.ACCEPTED,
            item.status == MarketSelectionStatus.DISCARDED,
            -(item.ev if item.ev is not None else -999.0) if item.ev is not None else 999.0,
            -item.model_probability,
        )
    )
    return evaluations


def render_market_evaluation_report(
    evaluations: List[EvaluatedMarket],
    console: Any,
    *,
    title: str = "Market Evaluation",
    limit: int = 12,
) -> None:
    from rich.table import Table

    if not evaluations:
        return

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Market", style="cyan")
    table.add_column("Model", justify="right")
    table.add_column("Odds", justify="right")
    table.add_column("Implied", justify="right")
    table.add_column("No-vig", justify="right")
    table.add_column("Edge", justify="right")
    table.add_column("EV", justify="right")
    table.add_column("Mode", style="white")
    table.add_column("Status", style="white")

    status_colors = {
        MarketSelectionStatus.ACCEPTED: "green",
        MarketSelectionStatus.MARGINAL: "yellow",
        MarketSelectionStatus.DISCARDED: "red",
    }

    for evaluation in evaluations[:limit]:
        color = status_colors[evaluation.status]
        table.add_row(
            evaluation.market_name,
            f"{evaluation.model_probability:.1%}",
            f"{evaluation.sportsbook_odds:.2f}" if evaluation.sportsbook_odds is not None else "-",
            f"{evaluation.implied_probability:.1%}" if evaluation.implied_probability is not None else "-",
            f"{evaluation.no_vig_probability:.1%}" if evaluation.no_vig_probability is not None else "-",
            f"{evaluation.edge:+.1%}" if evaluation.edge is not None else "-",
            f"{evaluation.ev:+.3f}" if evaluation.ev is not None else "-",
            evaluation.reference_probability_type.value,
            f"[{color}]{evaluation.status.value.upper()}[/{color}]",
        )

    console.print(table)
