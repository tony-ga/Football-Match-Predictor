
"""
Automatic Parlay Builder module.
Generates low/medium/high risk parlays based on model predictions.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class PickCandidate:
    """Represents a single pick candidate for a parlay leg."""
    match_id: str
    home_team: str
    away_team: str
    market_type: str  # e.g., "1x2_home", "double_chance_home_or_draw", "over_2_5", "btts_yes"
    market_name: str  # Readable name for display
    model_probability: float
    confidence_score: float  # 0-1, higher means more confident
    value_score: float  # 0-1, higher means more value
    final_score: float  # Combined score
    rationale: str
    market_group: str  # "1x2", "double_chance", "over_under", "btts"


def generate_market_candidates(
    match_prediction: Dict[str, Any],
    home_team: str,
    away_team: str
) -> List[PickCandidate]:
    """
    Generate all possible pick candidates for a single match.
    
    Args:
        match_prediction: Prediction result from predict_match_pipeline()
        home_team: Name of home team
        away_team: Name of away team
        
    Returns:
        List of PickCandidate objects
    """
    candidates = []
    match_id = f"{home_team}_{away_team}"
    
    markets = match_prediction['predictions']
    one_x_two = markets['1x2']
    double_chance = markets['double_chance']
    btts = markets['btts']
    over_under = markets['over_under']
    lambda_h = match_prediction['team_context']['home']['lambda_attack']
    lambda_a = match_prediction['team_context']['away']['lambda_attack']
    total_lambda = lambda_h + lambda_a
    
    # --- 1X2 Candidates ---
    for outcome in ['home', 'draw', 'away']:
        prob = one_x_two[outcome]
        # Calculate confidence: higher probability and larger gap to next outcome
        probs_sorted = sorted(one_x_two.values(), reverse=True)
        gap = probs_sorted[0] - probs_sorted[1]
        confidence = min(1.0, prob + gap * 0.5)
        
        # Rationale
        if outcome == 'home':
            market_name = f"{home_team} win"
        elif outcome == 'draw':
            market_name = "Draw"
        else:
            market_name = f"{away_team} win"
            
        rationale = f"Model probability: {prob:.1%}. Confidence based on gap to next outcome: {gap:.1%}."
        
        candidates.append(PickCandidate(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            market_type=f"1x2_{outcome}",
            market_name=market_name,
            model_probability=prob,
            confidence_score=confidence,
            value_score=prob,
            final_score=prob * confidence,
            rationale=rationale,
            market_group="1x2"
        ))
        
    # --- Double Chance Candidates ---
    dc_outcomes = [
        ('home_or_draw', f"{home_team} or Draw", "home_or_draw"),
        ('away_or_draw', f"{away_team} or Draw", "away_or_draw"),
        ('home_or_away', f"{home_team} or {away_team}", "home_or_away"),
    ]
    for dc_key, dc_name, dc_type in dc_outcomes:
        prob = double_chance[dc_key]
        confidence = min(1.0, prob * 0.9)  # DC has higher inherent confidence
        
        rationale = f"Double chance covers two outcomes. Model probability: {prob:.1%}."
        
        candidates.append(PickCandidate(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            market_type=f"double_chance_{dc_type}",
            market_name=dc_name,
            model_probability=prob,
            confidence_score=confidence,
            value_score=prob,
            final_score=prob * confidence,
            rationale=rationale,
            market_group="double_chance"
        ))
        
    # --- Over/Under Candidates ---
    ou_thresholds = ['1_5', '2_5', '3_5']
    for threshold in ou_thresholds:
        # Over
        over_prob = over_under[f"over_{threshold}"]
        over_conf = min(1.0, over_prob + (total_lambda - 2.5) * 0.1 if total_lambda > 2.5 else over_prob)
        over_name = f"Over {threshold.replace('_', '.')} goals"
        over_rationale = f"Model probability: {over_prob:.1%}. Expected total goals: {total_lambda:.1f}."
        
        candidates.append(PickCandidate(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            market_type=f"over_{threshold}",
            market_name=over_name,
            model_probability=over_prob,
            confidence_score=over_conf,
            value_score=over_prob,
            final_score=over_prob * over_conf,
            rationale=over_rationale,
            market_group="over_under"
        ))
        
        # Under
        under_prob = over_under[f"under_{threshold}"]
        under_conf = min(1.0, under_prob + (2.5 - total_lambda) * 0.1 if total_lambda < 2.5 else under_prob)
        under_name = f"Under {threshold.replace('_', '.')} goals"
        under_rationale = f"Model probability: {under_prob:.1%}. Expected total goals: {total_lambda:.1f}."
        
        candidates.append(PickCandidate(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            market_type=f"under_{threshold}",
            market_name=under_name,
            model_probability=under_prob,
            confidence_score=under_conf,
            value_score=under_prob,
            final_score=under_prob * under_conf,
            rationale=under_rationale,
            market_group="over_under"
        ))
        
    # --- BTTS Candidates ---
    for btts_outcome in ['yes', 'no']:
        prob = btts[btts_outcome]
        lambda_product = lambda_h * lambda_a
        conf = min(1.0, prob + (lambda_product - 1.0) * 0.1 if lambda_product > 1.0 else prob)
        
        btts_name = "BTTS: Yes" if btts_outcome == 'yes' else "BTTS: No"
        btts_rationale = f"Model probability: {prob:.1%}. Expected goal product: {lambda_product:.1f}."
        
        candidates.append(PickCandidate(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            market_type=f"btts_{btts_outcome}",
            market_name=btts_name,
            model_probability=prob,
            confidence_score=conf,
            value_score=prob,
            final_score=prob * conf,
            rationale=btts_rationale,
            market_group="btts"
        ))
        
    return candidates


def compute_correlation_penalty(pick_a: PickCandidate, pick_b: PickCandidate) -> float:
    """
    Compute a penalty score (0-1) for picks that are too correlated.
    
    Args:
        pick_a: First pick
        pick_b: Second pick
        
    Returns:
        Correlation penalty (0 = no penalty, 1 = max penalty)
    """
    if pick_a.match_id != pick_b.match_id:
        # Different matches: low correlation
        return 0.0
        
    # Same match: check market groups
    group_a = pick_a.market_group
    group_b = pick_b.market_group
    
    # Highly correlated combinations
    high_corr_pairs = [
        ("1x2", "over_under"),
        ("1x2", "btts"),
        ("double_chance", "over_under"),
        ("double_chance", "btts")
    ]
    
    if (group_a, group_b) in high_corr_pairs or (group_b, group_a) in high_corr_pairs:
        return 0.8
        
    if group_a == group_b:
        # Same market group (e.g., two 1x2 picks): max penalty
        return 1.0
        
    return 0.2


def filter_candidates(candidates: List[PickCandidate]) -> List[PickCandidate]:
    """
    Filter candidates to only keep the best ones.
    
    Args:
        candidates: All possible pick candidates
        
    Returns:
        Filtered list of candidates
    """
    # Group candidates by match
    match_to_candidates: Dict[str, List[PickCandidate]] = {}
    for candidate in candidates:
        if candidate.match_id not in match_to_candidates:
            match_to_candidates[candidate.match_id] = []
        match_to_candidates[candidate.match_id].append(candidate)
        
    filtered = []
    
    for match_candidates in match_to_candidates.values():
        # Sort by final score descending
        match_candidates_sorted = sorted(match_candidates, key=lambda x: x.final_score, reverse=True)
        
        # Keep top 3 candidates per match
        filtered.extend(match_candidates_sorted[:3])
        
    # Now filter out candidates with too low probability
    min_prob = 0.4  # At least 40% probability
    filtered = [c for c in filtered if c.model_probability >= min_prob]
    
    # Sort overall by final score
    filtered.sort(key=lambda x: x.final_score, reverse=True)
    
    return filtered


def build_parlay(
    candidates: List[PickCandidate],
    risk_level: RiskLevel,
    max_correlation_penalty: float = 0.5
) -> List[PickCandidate]:
    """
    Build a parlay for a given risk level.
    
    Args:
        candidates: Filtered pick candidates
        risk_level: Desired risk level
        max_correlation_penalty: Maximum allowed correlation penalty per addition
        
    Returns:
        List of picks forming the parlay
    """
    # Risk level settings
    risk_settings = {
        RiskLevel.LOW: {
            "min_prob": 0.65,
            "max_legs": 3,
            "prefer_groups": ["double_chance", "over_under"],
            "min_confidence": 0.7
        },
        RiskLevel.MEDIUM: {
            "min_prob": 0.55,
            "max_legs": 4,
            "prefer_groups": ["1x2", "double_chance", "over_under"],
            "min_confidence": 0.5
        },
        RiskLevel.HIGH: {
            "min_prob": 0.45,
            "max_legs": 5,
            "prefer_groups": ["1x2", "over_under", "btts"],
            "min_confidence": 0.3
        }
    }
    
    settings = risk_settings[risk_level]
    parlay = []
    used_matches = set()
    
    # First filter candidates by risk level settings
    eligible = [
        c for c in candidates
        if c.model_probability >= settings['min_prob']
        and c.confidence_score >= settings['min_confidence']
        and c.market_group in settings['prefer_groups']
    ]
    
    # Sort eligible by final score
    eligible_sorted = sorted(eligible, key=lambda x: x.final_score, reverse=True)
    
    for candidate in eligible_sorted:
        if len(parlay) >= settings['max_legs']:
            break
            
        if candidate.match_id in used_matches:
            continue
            
        # Check correlation with existing parlay legs
        total_penalty = 0.0
        for leg in parlay:
            penalty = compute_correlation_penalty(candidate, leg)
            total_penalty += penalty
            
        if total_penalty <= max_correlation_penalty:
            parlay.append(candidate)
            used_matches.add(candidate.match_id)
            
    return parlay


def calculate_parlay_probability(parlay: List[PickCandidate]) -> float:
    """Calculate combined probability of a parlay (assuming independence)."""
    prob = 1.0
    for pick in parlay:
        prob *= pick.model_probability
    return prob


def render_parlay(
    parlay: List[PickCandidate],
    risk_level: RiskLevel,
    console: Any
) -> None:
    """Render a parlay in a readable format using Rich Console."""
    from rich.panel import Panel
    from rich.table import Table
    
    if not parlay:
        console.print(f"[yellow]No picks available for {risk_level.value} risk parlay.[/yellow]")
        return
        
    combined_prob = calculate_parlay_probability(parlay)
    
    title = f"{risk_level.value.upper()} RISK PARLAY"
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Match", style="white")
    table.add_column("Pick", style="green")
    table.add_column("Probability", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Rationale", style="dim")
    
    for i, pick in enumerate(parlay, 1):
        match_str = f"{pick.home_team} vs {pick.away_team}"
        prob_str = f"{pick.model_probability:.1%}"
        conf_str = f"{pick.confidence_score:.1%}"
        
        table.add_row(
            str(i),
            match_str,
            pick.market_name,
            prob_str,
            conf_str,
            pick.rationale
        )
        
    console.print(Panel(table, title=title, border_style="blue"))
    console.print(f"[bold]Combined Probability: {combined_prob:.1%}[/bold]")
    console.print(f"[bold]Number of Legs: {len(parlay)}[/bold]\n")


def build_all_parlays(
    match_predictions: List[Dict[str, Any]]
) -> Dict[RiskLevel, List[PickCandidate]]:
    """
    Build low/medium/high risk parlays from multiple match predictions.
    
    Args:
        match_predictions: List of prediction results from predict_match_pipeline()
        
    Returns:
        Dict mapping RiskLevel to list of picks
    """
    # Collect all candidates
    all_candidates = []
    for match_pred in match_predictions:
        home_team = match_pred['team_context']['home']['team']
        away_team = match_pred['team_context']['away']['team']
        
        candidates = generate_market_candidates(match_pred, home_team, away_team)
        all_candidates.extend(candidates)
        
    # Filter candidates
    filtered_candidates = filter_candidates(all_candidates)
    
    # Build parlays
    parlays = {}
    for risk in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]:
        parlays[risk] = build_parlay(filtered_candidates, risk)
        
    return parlays
