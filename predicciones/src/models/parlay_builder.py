
"""
Same Game Parlay Builder
Generates intelligent, non-redundant, game-script-based parlays using full combination evaluation
"""
from __future__ import annotations

import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
import numpy as np

from .market_evaluation import (
    EvaluatedMarket,
    MarketFilterConfig,
    MarketSelectionStatus,
    evaluate_core_market_set,
    render_market_evaluation_report,
)
from .ticket_structure import (
    MarketRelationType,
    TicketStructureEvaluation,
    evaluate_ticket_structure,
)

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MarketType(Enum):
    ONE_X_TWO_HOME = "1x2_home"
    ONE_X_TWO_DRAW = "1x2_draw"
    ONE_X_TWO_AWAY = "1x2_away"
    DOUBLE_CHANCE_HOME_OR_DRAW = "double_chance_home_or_draw"
    DOUBLE_CHANCE_AWAY_OR_DRAW = "double_chance_away_or_draw"
    DOUBLE_CHANCE_HOME_OR_AWAY = "double_chance_home_or_away"
    OVER_1_5 = "over_1.5"
    OVER_2_5 = "over_2.5"
    OVER_3_5 = "over_3.5"
    OVER_4_5 = "over_4.5"
    UNDER_1_5 = "under_1.5"
    UNDER_2_5 = "under_2.5"
    UNDER_3_5 = "under_3.5"
    UNDER_4_5 = "under_4.5"
    BTTS_YES = "btts_yes"
    BTTS_NO = "btts_no"


@dataclass
class PickCandidate:
    """Single pick candidate for a same game parlay leg"""
    match_id: str
    home_team: str
    away_team: str
    market_type: MarketType
    market_name: str
    model_probability: float
    confidence_score: float
    final_score: float
    rationale: str
    is_calibrated: bool = False
    market_evaluation: Optional[EvaluatedMarket] = None


@dataclass
class SameGameParlayResult:
    """Result of a same game parlay build"""
    risk_level: RiskLevel
    picks: List[PickCandidate]
    combined_probability: float
    ticket_score: float
    game_script_rationale: str
    is_valid: bool = True
    structure_evaluation: Optional[TicketStructureEvaluation] = None


class CalibrationStatus:
    def __init__(self):
        self.missing_calibrators: List[str] = []
        self.is_calibrated: bool = True
        
    def add_missing(self, name: str):
        self.missing_calibrators.append(name)
        self.is_calibrated = False
        
    def get_warning(self) -> str:
        if not self.missing_calibrators:
            return ""
        return f"Calibration status: missing {len(self.missing_calibrators)} calibrators, using fallback probabilities"


def check_calibration() -> CalibrationStatus:
    from pathlib import Path
    status = CalibrationStatus()
    project_root = Path(__file__).parent.parent.parent.parent
    calib_dir = project_root / "output" / "calibrators"
    
    expected_calibrators = [
        "lambda_recalibrator.pkl",
        "cal_1x2.pkl",
        "cal_btts.pkl",
        "cal_ou15.pkl",
        "cal_ou25.pkl",
        "cal_ou35.pkl",
    ]
    
    if not calib_dir.exists():
        for name in expected_calibrators:
            status.add_missing(name)
        return status
        
    for name in expected_calibrators:
        if not (calib_dir / name).exists():
            status.add_missing(name)
            
    return status


def get_market_family(market_type: MarketType) -> str:
    """Return which family a market belongs to"""
    if market_type.value.startswith("1x2"):
        return "result"
    if market_type.value.startswith("double_chance"):
        return "result"
    if market_type.value.startswith("over") or market_type.value.startswith("under"):
        return "totals"
    if market_type.value.startswith("btts"):
        return "btts"
    return "other"


def get_market_specificity(market_type: MarketType) -> float:
    """Return a specificity score (higher means more specific)"""
    specificity_map = {
        MarketType.ONE_X_TWO_HOME: 0.7,
        MarketType.ONE_X_TWO_AWAY: 0.7,
        MarketType.ONE_X_TWO_DRAW: 0.6,
        MarketType.DOUBLE_CHANCE_HOME_OR_DRAW: 0.4,
        MarketType.DOUBLE_CHANCE_AWAY_OR_DRAW: 0.4,
        MarketType.DOUBLE_CHANCE_HOME_OR_AWAY: 0.3,
        MarketType.OVER_1_5: 0.2,
        MarketType.UNDER_4_5: 0.1,
        MarketType.OVER_2_5: 0.5,
        MarketType.UNDER_3_5: 0.3,
        MarketType.OVER_3_5: 0.7,
        MarketType.UNDER_2_5: 0.5,
        MarketType.OVER_4_5: 0.8,
        MarketType.UNDER_1_5: 0.7,
        MarketType.BTTS_YES: 0.5,
        MarketType.BTTS_NO: 0.5,
    }
    return specificity_map.get(market_type, 0.5)


def compute_real_confidence(
    pred_data: Dict[str, Any],
    market_type: MarketType,
    model_prob: float,
    calib_status: CalibrationStatus
) -> Tuple[float, str]:
    """Compute a real confidence score based on multiple factors"""
    confidence = model_prob
    rationale = f"Model probability: {model_prob:.1%}."
    
    # 1. Add confidence for double chance
    if "double_chance" in market_type.value:
        confidence += 0.05
        rationale += " Market: double chance (more stable)."
        
    # 2. 1X2 confidence from outcome gap
    if market_type in [MarketType.ONE_X_TWO_HOME, MarketType.ONE_X_TWO_DRAW, MarketType.ONE_X_TWO_AWAY]:
        probs = [
            pred_data['predictions']['1x2']['home'],
            pred_data['predictions']['1x2']['draw'],
            pred_data['predictions']['1x2']['away'],
        ]
        probs_sorted = sorted(probs, reverse=True)
        gap = probs_sorted[0] - probs_sorted[1] if len(probs) >= 2 else 0
        confidence += gap * 0.3
        rationale += f" Outcome gap: {gap:.1%}."
        
    # 3. Calibration penalty
    if not calib_status.is_calibrated:
        confidence *= 0.85
        rationale += " Penalty: uncalibrated model."
        
    # 4. Lambda stability
    total_lambda = (
        pred_data['team_context']['home']['lambda_attack'] +
        pred_data['team_context']['away']['lambda_attack']
    )
    if total_lambda < 2.0:
        confidence += 0.03
        rationale += f" Lambda stability: low total ({total_lambda:.1f})."
    elif total_lambda > 3.5:
        confidence -= 0.03
        rationale += f" Lambda volatility: high total ({total_lambda:.1f})."
        
    return np.clip(confidence, 0.0, 1.0), rationale


def _market_key_from_type(market_type: MarketType) -> str:
    return market_type.value.replace(".", "_")


def _base_market_filter_config() -> MarketFilterConfig:
    return MarketFilterConfig(
        min_model_prob=0.35,
        min_edge=0.02,
        min_ev=0.0,
        max_odds=5.50,
        min_confidence=0.35,
        marginal_tolerance=0.015,
    )


def _selectivity_adjustment(evaluation: Optional[EvaluatedMarket]) -> float:
    if evaluation is None:
        return 0.0

    adjustment = {
        MarketSelectionStatus.ACCEPTED: 0.08,
        MarketSelectionStatus.MARGINAL: 0.02,
        MarketSelectionStatus.DISCARDED: -0.18,
    }[evaluation.status]

    if evaluation.edge is not None:
        adjustment += float(np.clip(evaluation.edge, -0.10, 0.10) * 0.6)
    if evaluation.ev is not None:
        adjustment += float(np.clip(evaluation.ev, -0.25, 0.25) * 0.35)
    if evaluation.is_model_only:
        adjustment -= 0.01

    return adjustment


def generate_same_game_candidates(
    pred_data: Dict[str, Any],
    home_team: str,
    away_team: str,
    calib_status: CalibrationStatus
) -> List[PickCandidate]:
    """Generate candidate picks using model confidence plus phase-1 market selectivity."""
    candidates = []
    match_id = f"{home_team}_{away_team}"
    match_name = f"{home_team} vs {away_team}"

    markets = pred_data['predictions']

    market_configs = [
        (MarketType.ONE_X_TWO_HOME, f"{home_team} win", markets['1x2']['home']),
        (MarketType.ONE_X_TWO_DRAW, "Draw", markets['1x2']['draw']),
        (MarketType.ONE_X_TWO_AWAY, f"{away_team} win", markets['1x2']['away']),
        (MarketType.DOUBLE_CHANCE_HOME_OR_DRAW, f"{home_team} or Draw", markets['double_chance']['home_or_draw']),
        (MarketType.DOUBLE_CHANCE_AWAY_OR_DRAW, f"{away_team} or Draw", markets['double_chance']['away_or_draw']),
        (MarketType.OVER_2_5, "Over 2.5 goals", markets['over_under']['over_2_5']),
        (MarketType.OVER_3_5, "Over 3.5 goals", markets['over_under']['over_3_5']),
        (MarketType.UNDER_2_5, "Under 2.5 goals", markets['over_under']['under_2_5']),
        (MarketType.UNDER_3_5, "Under 3.5 goals", markets['over_under']['under_3_5']),
        (MarketType.BTTS_YES, "BTTS: Yes", markets['btts']['yes']),
        (MarketType.BTTS_NO, "BTTS: No", markets['btts']['no']),
    ]

    confidence_map: Dict[str, float] = {}
    confidence_rationales: Dict[str, str] = {}
    for market_type, _market_name, prob in market_configs:
        if prob < 0.25:
            continue
        conf, conf_rationale = compute_real_confidence(pred_data, market_type, prob, calib_status)
        confidence_map[_market_key_from_type(market_type)] = conf
        confidence_rationales[_market_key_from_type(market_type)] = conf_rationale

    evaluations = evaluate_core_market_set(
        match_name=match_name,
        home_team=home_team,
        away_team=away_team,
        predictions=markets,
        sportsbook_odds=pred_data.get("sportsbook_odds"),
        confidence_scores=confidence_map,
        filter_config=_base_market_filter_config(),
    )
    pred_data["market_evaluations"] = evaluations
    evaluation_by_key = {item.market_key: item for item in evaluations}

    for market_type, market_name, prob in market_configs:
        if prob < 0.3:
            continue

        market_key = _market_key_from_type(market_type)
        conf = confidence_map.get(market_key)
        if conf is None:
            conf, conf_rationale = compute_real_confidence(pred_data, market_type, prob, calib_status)
            confidence_rationales[market_key] = conf_rationale
        conf_rationale = confidence_rationales.get(market_key, "")

        evaluation = evaluation_by_key.get(market_key)
        if evaluation and evaluation.status == MarketSelectionStatus.DISCARDED:
            continue

        candidates.append(PickCandidate(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            market_type=market_type,
            market_name=market_name,
            model_probability=prob,
            confidence_score=conf,
            final_score=(prob * conf) + _selectivity_adjustment(evaluation),
            rationale=conf_rationale,
            is_calibrated=calib_status.is_calibrated,
            market_evaluation=evaluation,
        ))

    candidates.sort(key=lambda c: -c.final_score)
    return candidates


def evaluate_parlay(
    parlay: List[PickCandidate],
    risk_level: RiskLevel,
    pred_data: Dict[str, Any],
) -> Tuple[float, Dict[str, float], TicketStructureEvaluation]:
    """
    Evaluate a complete parlay and return a total score along with individual components
    """
    scores: Dict[str, float] = {}

    structure_evaluation = evaluate_ticket_structure(parlay)
    scores["contradiction_penalty"] = -structure_evaluation.contradiction_penalty
    scores["redundancy_penalty"] = -structure_evaluation.redundancy_penalty
    scores["weak_information_penalty"] = -structure_evaluation.weak_information_penalty
    scores["information_gain_score"] = structure_evaluation.information_gain_score
    scores["family_diversity_score"] = structure_evaluation.family_diversity_score
    scores["compatibility_score"] = structure_evaluation.compatibility_score
    scores["structure_score"] = structure_evaluation.final_structure_score

    if not structure_evaluation.is_valid:
        scores["final"] = 0.0
        return 0.0, scores, structure_evaluation

    # 1. Probability score (product of individual probabilities)
    combined_prob = np.prod([p.model_probability for p in parlay])
    if combined_prob < 1e-10:
        scores["final"] = 0.0
        return 0.0, scores, structure_evaluation
        
    # Normalize probability score by risk level
    risk_level_prob_targets = {
        RiskLevel.LOW: 0.4,
        RiskLevel.MEDIUM: 0.2,
        RiskLevel.HIGH: 0.1,
    }
    prob_score = min(1.0, combined_prob / risk_level_prob_targets[risk_level])
    scores["probability"] = prob_score
    
    # 2. Confidence score (average of individual confidences)
    avg_conf = np.mean([p.confidence_score for p in parlay])
    scores["confidence"] = avg_conf

    phase1_selectivity = []
    for pick in parlay:
        if pick.market_evaluation is None:
            phase1_selectivity.append(0.45)
            continue
        status_score = {
            MarketSelectionStatus.ACCEPTED: 1.0,
            MarketSelectionStatus.MARGINAL: 0.65,
            MarketSelectionStatus.DISCARDED: 0.0,
        }[pick.market_evaluation.status]
        ev_bonus = 0.0 if pick.market_evaluation.ev is None else float(np.clip(pick.market_evaluation.ev, -0.15, 0.25))
        edge_bonus = 0.0 if pick.market_evaluation.edge is None else float(np.clip(pick.market_evaluation.edge, -0.08, 0.12))
        phase1_selectivity.append(np.clip((status_score * 0.7) + (ev_bonus * 0.5) + (edge_bonus * 0.8), 0.0, 1.0))
    selectivity_score = float(np.mean(phase1_selectivity))
    scores["phase1_selectivity_score"] = selectivity_score

    complementary_pairs = sum(
        1 for relation in structure_evaluation.pair_relations
        if relation.relation == MarketRelationType.COMPLEMENTARY
    )
    complementarity_bonus = min(0.35, complementary_pairs * 0.12)
    scores["complementarity_bonus"] = complementarity_bonus

    # 3. Game script specificity
    specificity = np.mean([get_market_specificity(p.market_type) for p in parlay])
    scores["specificity"] = specificity

    # Calculate final ticket score
    final_score = (
        (prob_score * 0.18) +
        (avg_conf * 0.17) +
        (selectivity_score * 0.15) +
        (structure_evaluation.compatibility_score * 0.10) +
        (structure_evaluation.information_gain_score * 0.12) +
        (structure_evaluation.family_diversity_score * 0.10) +
        (structure_evaluation.final_structure_score * 0.12) +
        (specificity * 0.06) +
        (complementarity_bonus * 0.05) -
        (structure_evaluation.contradiction_penalty * 0.85) -
        (structure_evaluation.redundancy_penalty * 0.55) -
        (structure_evaluation.weak_information_penalty * 0.30)
    )
    final_score = np.clip(final_score, 0.0, 1.0)
    scores["final"] = final_score
    
    return final_score, scores, structure_evaluation


def generate_game_script_explanation(
    parlay: List[PickCandidate],
    risk_level: RiskLevel,
    pred_data: Dict[str, Any],
) -> str:
    """Generate a detailed explanation of the game script for the parlay"""
    if not parlay:
        return "No valid picks."
        
    home_team = parlay[0].home_team
    away_team = parlay[0].away_team
    
    # Build a narrative
    parts = []
    for pick in parlay:
        if "win" in pick.market_name or "Draw" in pick.market_name:
            if "double_chance" in pick.market_type.value:
                if "home" in pick.market_type.value:
                    parts.append(f"Result protection for {home_team}, avoiding a loss")
                else:
                    parts.append(f"Result protection for {away_team}, avoiding a loss")
            else:
                parts.append(f"Result pick: {pick.market_name.lower()}")
        elif "BTTS" in pick.market_name:
            if "Yes" in pick.market_name:
                parts.append("Distribution pick: both teams expected to score")
            else:
                parts.append("Distribution pick: at least one team expected to keep a clean sheet")
        else:
            parts.append(f"Total goals pick: {pick.market_name.lower()}")
            
    # Combine into a coherent explanation
    risk_prefix = {
        RiskLevel.LOW: "Conservative script: ",
        RiskLevel.MEDIUM: "Moderate conviction script: ",
        RiskLevel.HIGH: "Aggressive, specific script: ",
    }[risk_level]
    
    narrative = risk_prefix
    
    if len(parts) == 1:
        narrative += parts[0]
    elif len(parts) == 2:
        narrative += f"{parts[0]} while {parts[1].lower()}"
    else:
        narrative += ", ".join(p.lower() for p in parts[:-1]) + f", and {parts[-1].lower()}"
        
    # Add a closing sentence
    closing = (
        "This combination avoids redundant markets and builds a "
        f"{'conservative' if risk_level == RiskLevel.LOW else 'specific' if risk_level == RiskLevel.HIGH else 'balanced'} "
        "narrative about how the match might play out."
    )
    return narrative + ". " + closing


def build_all_same_game_parlays(
    pred_data: Dict[str, Any],
    home_team: str,
    away_team: str,
    calib_status: CalibrationStatus
) -> Tuple[List[SameGameParlayResult], CalibrationStatus]:
    """Build all three same game parlays for a single match using combination search"""
    # Generate candidate picks
    candidates = generate_same_game_candidates(pred_data, home_team, away_team, calib_status)
    if not candidates:
        return [], calib_status
        
    # Define leg counts per risk level
    risk_leg_counts = {
        RiskLevel.LOW: [2],
        RiskLevel.MEDIUM: [2, 3],
        RiskLevel.HIGH: [3, 4],
    }
    
    results = []
    
    for risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]:
        best_parlay = None
        best_score = 0.0
        best_structure = None
        
        # Try all possible leg counts for this risk level
        for num_legs in risk_leg_counts[risk_level]:
            if num_legs > len(candidates):
                continue
                
            # Generate all possible combinations (use top N candidates to limit search space)
            top_candidates = candidates[:12]  # Keep it manageable
            for parlay_tuple in combinations(top_candidates, num_legs):
                parlay = list(parlay_tuple)
                    
                # Evaluate the parlay
                score, _, structure = evaluate_parlay(parlay, risk_level, pred_data)
                if not structure.is_valid:
                    continue
                
                if score > best_score:
                    best_score = score
                    best_parlay = parlay
                    best_structure = structure
                    
        if best_parlay is None or best_score < 0.1:
            results.append(SameGameParlayResult(
                risk_level=risk_level,
                picks=[],
                combined_probability=0.0,
                ticket_score=0.0,
                game_script_rationale=f"No valid parlay found for {risk_level.value} risk",
                is_valid=False,
                structure_evaluation=None,
            ))
        else:
            combined_prob = np.prod([p.model_probability for p in best_parlay])
            results.append(SameGameParlayResult(
                risk_level=risk_level,
                picks=best_parlay,
                combined_probability=combined_prob,
                ticket_score=best_score,
                game_script_rationale=generate_game_script_explanation(
                    best_parlay, risk_level, pred_data
                ),
                is_valid=True,
                structure_evaluation=best_structure,
            ))
            
    return results, calib_status


def render_same_game_parlay_report(
    parlays: List[SameGameParlayResult],
    pred_data: Dict[str, Any],
    home_team: str,
    away_team: str,
    calib_status: CalibrationStatus,
    console: Any,
) -> None:
    """Render the full same game parlay report"""
    from rich.panel import Panel
    from rich.table import Table
    
    console.print("\n[bold blue]=== SAME GAME PARLAY BUILDER ===[/bold blue]\n")
    
    # Show match summary
    markets = pred_data['predictions']
    total_lambda = (
        pred_data['team_context']['home']['lambda_attack'] +
        pred_data['team_context']['away']['lambda_attack']
    )
    
    console.print(f"[bold]Selected Match: {home_team} vs {away_team}[/bold]\n")
    
    summary_table = Table(title="Model Summary", show_header=False, header_style="bold magenta")
    summary_table.add_column("", style="cyan")
    summary_table.add_column("", style="white")
    
    summary_table.add_row("Home win", f"{markets['1x2']['home']:.1%}")
    summary_table.add_row("Draw", f"{markets['1x2']['draw']:.1%}")
    summary_table.add_row("Away win", f"{markets['1x2']['away']:.1%}")
    summary_table.add_row("Over 2.5", f"{markets['over_under']['over_2_5']:.1%}")
    summary_table.add_row("Under 2.5", f"{markets['over_under']['under_2_5']:.1%}")
    summary_table.add_row("BTTS Yes", f"{markets['btts']['yes']:.1%}")
    summary_table.add_row("BTTS No", f"{markets['btts']['no']:.1%}")
    summary_table.add_row("Expected total goals", f"{total_lambda:.1f}")
    
    console.print(summary_table)
    console.print()

    sportsbook_odds = pred_data.get("sportsbook_odds") or {}
    if sportsbook_odds.get("notes"):
        for note in sportsbook_odds["notes"]:
            console.print(f"[dim]Odds note: {note}[/dim]")
        console.print()
    
    # Show calibration warning if needed
    if calib_status.get_warning():
        console.print(f"[yellow]{calib_status.get_warning()}[/yellow]\n")

    evaluations = pred_data.get("market_evaluations") or []
    if evaluations:
        render_market_evaluation_report(
            evaluations,
            console,
            title="Phase 1 Market Evaluation",
            limit=16,
        )
        console.print()
        
    # Render each parlay
    for parlay in parlays:
        _render_single_same_game_parlay(parlay, console)


def _render_single_same_game_parlay(
    parlay: SameGameParlayResult,
    console: Any,
) -> None:
    from rich.panel import Panel
    from rich.table import Table
    
    title = f"{parlay.risk_level.value.upper()} RISK SAME GAME PARLAY"
    border_color = {
        RiskLevel.LOW: "green",
        RiskLevel.MEDIUM: "blue",
        RiskLevel.HIGH: "red",
    }[parlay.risk_level]
    
    if not parlay.is_valid or not parlay.picks:
        console.print(Panel(
            f"[yellow]{title}[/yellow]\n\n[red]{parlay.game_script_rationale}[/red]",
            title=title,
            border_style="yellow",
        ))
        console.print()
        return
        
    # Build the picks table
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Pick", style="green")
    table.add_column("Probability", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Edge", justify="right")
    table.add_column("EV", justify="right")
    table.add_column("Status", justify="right")
    
    for i, pick in enumerate(parlay.picks, 1):
        evaluation = pick.market_evaluation
        table.add_row(
            str(i),
            pick.market_name,
            f"{pick.model_probability:.1%}",
            f"{pick.confidence_score:.1%}",
            f"{evaluation.edge:+.1%}" if evaluation and evaluation.edge is not None else "-",
            f"{evaluation.ev:+.3f}" if evaluation and evaluation.ev is not None else "-",
            evaluation.status.value.upper() if evaluation else "-",
        )
        
    # Render the panel and details
    console.print(Panel(
        table,
        title=title,
        border_style=border_color,
    ))
    console.print(f"[bold]Combined Probability: {parlay.combined_probability:.1%}[/bold]")
    if parlay.structure_evaluation is not None:
        structure = parlay.structure_evaluation
        console.print(
            "[dim]"
            f"Structure: compatibility={structure.compatibility_score:.2f} | "
            f"information_gain={structure.information_gain_score:.2f} | "
            f"family_diversity={structure.family_diversity_score:.2f} | "
            f"redundancy_penalty={structure.redundancy_penalty:.2f} | "
            f"contradiction_penalty={structure.contradiction_penalty:.2f}"
            "[/dim]"
        )
    console.print(f"[italic]{parlay.game_script_rationale}[/italic]\n")
