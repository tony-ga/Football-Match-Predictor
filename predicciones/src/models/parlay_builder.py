
"""
Automatic Parlay Builder module (v2 - improved risk management)
Generates low/medium/high risk parlays based on model predictions
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
    """Represents a single pick candidate for a parlay leg"""
    match_id: str
    home_team: str
    away_team: str
    market_type: str
    market_name: str
    model_probability: float
    confidence_score: float
    value_score: float
    final_score: float
    rationale: str
    market_group: str
    is_calibrated: bool = False


@dataclass
class ParlayResult:
    risk_level: RiskLevel
    picks: List[PickCandidate]
    combined_probability: float
    reason: str
    is_valid: bool = True


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
    """Check which calibrators are available and return status"""
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


def compute_real_confidence(
    pred_data: Dict[str, Any],
    market_type: str,
    model_prob: float,
    calib_status: CalibrationStatus
) -> Tuple[float, str]:
    """
    Compute a real confidence score (not just copy of probability)
    Takes into account:
        - distance between outcomes
        - market robustness
        - calibration status
        - lambda stability
    """
    confidence = model_prob
    rationale = ""
    
    # 1. Distance between outcomes
    if market_type.startswith("1x2"):
        probs = sorted(pred_data['predictions']['1x2'].values(), reverse=True)
        gap = probs[0] - probs[1] if len(probs) >=2 else 0
        confidence += gap * 0.3
        rationale += f" Outcome gap: {gap:.1%}."
        
    # 2. Double chance is inherently more stable
    if market_type.startswith("double_chance"):
        confidence += 0.05
        rationale += " Market: double chance (more stable)."
        
    # 3. Calibration penalty
    if not calib_status.is_calibrated:
        confidence *= 0.85
        rationale += " Penalty: uncalibrated model."
        
    # 4. Lambda stability (lower total lambda = more predictable)
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
        
    # 5. Clip to 0-1
    confidence = np.clip(confidence, 0.0, 1.0)
    return confidence, rationale


def generate_market_candidates(
    match_prediction: Dict[str, Any],
    home_team: str,
    away_team: str,
    calib_status: CalibrationStatus
) -> List[PickCandidate]:
    """Generate all possible pick candidates for a single match"""
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
        if outcome == 'home':
            market_name = f"{home_team} win"
        elif outcome == 'draw':
            market_name = "Draw"
        else:
            market_name = f"{away_team} win"
            
        conf, conf_rationale = compute_real_confidence(
            match_prediction, f"1x2_{outcome}", prob, calib_status
        )
        rationale = f"Model probability: {prob:.1%}.{conf_rationale}"
        
        candidates.append(PickCandidate(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            market_type=f"1x2_{outcome}",
            market_name=market_name,
            model_probability=prob,
            confidence_score=conf,
            value_score=prob,
            final_score=prob * conf,
            rationale=rationale,
            market_group="1x2",
            is_calibrated=calib_status.is_calibrated
        ))
        
    # --- Double Chance Candidates ---
    dc_outcomes = [
        ('home_or_draw', f"{home_team} or Draw", "home_or_draw"),
        ('away_or_draw', f"{away_team} or Draw", "away_or_draw"),
        ('home_or_away', f"{home_team} or {away_team}", "home_or_away"),
    ]
    for dc_key, dc_name, dc_type in dc_outcomes:
        prob = double_chance[dc_key]
        conf, conf_rationale = compute_real_confidence(
            match_prediction, f"double_chance_{dc_type}", prob, calib_status
        )
        rationale = f"Model probability: {prob:.1%}.{conf_rationale}"
        
        candidates.append(PickCandidate(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            market_type=f"double_chance_{dc_type}",
            market_name=dc_name,
            model_probability=prob,
            confidence_score=conf,
            value_score=prob,
            final_score=prob * conf,
            rationale=rationale,
            market_group="double_chance",
            is_calibrated=calib_status.is_calibrated
        ))
        
    # --- Over/Under Candidates ---
    ou_thresholds = ['1.5', '2.5', '3.5']
    for threshold in ou_thresholds:
        threshold_key = threshold.replace('.', '_')
        over_prob = over_under[f"over_{threshold_key}"]
        under_prob = over_under[f"under_{threshold_key}"]
        
        # Over
        conf, conf_rationale = compute_real_confidence(
            match_prediction, f"over_{threshold_key}", over_prob, calib_status
        )
        rationale = f"Model probability: {over_prob:.1%}. Expected total goals: {total_lambda:.1f}.{conf_rationale}"
        candidates.append(PickCandidate(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            market_type=f"over_{threshold_key}",
            market_name=f"Over {threshold} goals",
            model_probability=over_prob,
            confidence_score=conf,
            value_score=over_prob,
            final_score=over_prob * conf,
            rationale=rationale,
            market_group="over_under",
            is_calibrated=calib_status.is_calibrated
        ))
        
        # Under
        conf, conf_rationale = compute_real_confidence(
            match_prediction, f"under_{threshold_key}", under_prob, calib_status
        )
        rationale = f"Model probability: {under_prob:.1%}. Expected total goals: {total_lambda:.1f}.{conf_rationale}"
        candidates.append(PickCandidate(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            market_type=f"under_{threshold_key}",
            market_name=f"Under {threshold} goals",
            model_probability=under_prob,
            confidence_score=conf,
            value_score=under_prob,
            final_score=under_prob * conf,
            rationale=rationale,
            market_group="over_under",
            is_calibrated=calib_status.is_calibrated
        ))
        
    # --- BTTS Candidates ---
    for btts_outcome in ['yes', 'no']:
        prob = btts[btts_outcome]
        conf, conf_rationale = compute_real_confidence(
            match_prediction, f"btts_{btts_outcome}", prob, calib_status
        )
        btts_name = "BTTS: Yes" if btts_outcome == 'yes' else "BTTS: No"
        lambda_product = lambda_h * lambda_a
        rationale = f"Model probability: {prob:.1%}. Expected goal product: {lambda_product:.1f}.{conf_rationale}"
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
            rationale=rationale,
            market_group="btts",
            is_calibrated=calib_status.is_calibrated
        ))
        
    return candidates


def filter_candidates_by_risk(candidates: List[PickCandidate], risk_level: RiskLevel) -> List[PickCandidate]:
    """Filter candidates specifically for a given risk level"""
    filtered = []
    
    # Risk-specific thresholds
    if risk_level == RiskLevel.LOW:
        min_prob = 0.65
        min_conf = 0.60
        allowed_groups = ['double_chance', 'over_under']
        allowed_ou = ['1.5', '3.5']
    elif risk_level == RiskLevel.MEDIUM:
        min_prob = 0.55
        min_conf = 0.45
        allowed_groups = ['double_chance', 'over_under', '1x2']
        allowed_ou = ['1.5', '2.5', '3.5']
    else:  # HIGH
        min_prob = 0.40
        min_conf = 0.30
        allowed_groups = ['double_chance', 'over_under', '1x2', 'btts']
        allowed_ou = ['1.5', '2.5', '3.5']
        
    for cand in candidates:
        # Basic filters
        if cand.model_probability < min_prob:
            continue
        if cand.confidence_score < min_conf:
            continue
        if cand.market_group not in allowed_groups:
            continue
            
        # Over/Under specific filters
        if cand.market_group == 'over_under':
            is_allowed = False
            for threshold in allowed_ou:
                if threshold.replace('.', '_') in cand.market_type:
                    is_allowed = True
                    break
            if not is_allowed:
                continue
                
        # 1x2 specific: avoid very balanced matches for LOW risk
        if risk_level == RiskLevel.LOW and cand.market_group == '1x2':
            continue
            
        filtered.append(cand)
        
    # Sort by final score descending
    filtered.sort(key=lambda x: x.final_score, reverse=True)
    return filtered


def compute_correlation_penalty(pick_a: PickCandidate, pick_b: PickCandidate) -> float:
    """
    Compute real correlation penalty between two picks
    """
    if pick_a.match_id != pick_b.match_id:
        return 0.0
        
    # Same match - higher correlation
    group_a = pick_a.market_group
    group_b = pick_b.market_group
    
    # Maximum penalty for same group
    if group_a == group_b:
        return 1.0
        
    # High correlation pairs
    high_corr = [
        ('1x2', 'over_under'), ('over_under', '1x2'),
        ('1x2', 'btts'), ('btts', '1x2'),
        ('double_chance', 'over_under'), ('over_under', 'double_chance'),
        ('double_chance', 'btts'), ('btts', 'double_chance'),
    ]
    if (group_a, group_b) in high_corr:
        return 0.75
        
    # Default penalty for same match but different groups
    return 0.35


def build_low_risk_parlay(
    all_candidates: List[PickCandidate],
    match_count: int
) -> ParlayResult:
    """Build LOW RISK parlay"""
    filtered = filter_candidates_by_risk(all_candidates, RiskLevel.LOW)
    if not filtered:
        return ParlayResult(
            risk_level=RiskLevel.LOW,
            picks=[],
            combined_probability=0.0,
            reason="No valid low-risk picks available",
            is_valid=False
        )
        
    parlay = []
    used_matches = set()
    max_legs = min(2, match_count)
    
    # Sort by final score descending
    sorted_candidates = sorted(filtered, key=lambda x: x.final_score, reverse=True)
    
    for cand in sorted_candidates:
        if len(parlay) >= max_legs:
            break
        if cand.match_id in used_matches:
            continue
            
        # Check correlation with existing picks
        penalty_sum = 0.0
        for leg in parlay:
            penalty = compute_correlation_penalty(cand, leg)
            penalty_sum += penalty
            
        if penalty_sum > 0.3:  # Strict correlation limit for low risk
            continue
            
        parlay.append(cand)
        used_matches.add(cand.match_id)
        
    combined_prob = np.prod([p.model_probability for p in parlay]) if parlay else 0.0
    
    return ParlayResult(
        risk_level=RiskLevel.LOW,
        picks=parlay,
        combined_probability=combined_prob,
        reason="Conservative picks, low volatility, broad coverage (double chance / over/under)"
    )


def build_medium_risk_parlay(
    all_candidates: List[PickCandidate],
    match_count: int,
    low_risk_picks: List[PickCandidate]
) -> ParlayResult:
    """Build MEDIUM RISK parlay (different from low risk!)"""
    filtered = filter_candidates_by_risk(all_candidates, RiskLevel.MEDIUM)
    
    # Exclude picks used in low risk (to make it different)
    low_risk_ids = {(p.match_id, p.market_type) for p in low_risk_picks}
    filtered = [p for p in filtered if (p.match_id, p.market_type) not in low_risk_ids]
    
    if not filtered:
        return ParlayResult(
            risk_level=RiskLevel.MEDIUM,
            picks=[],
            combined_probability=0.0,
            reason="No valid medium-risk picks available (all used by low risk)",
            is_valid=False
        )
        
    parlay = []
    used_matches = set()
    max_legs = min(3, match_count)
    
    sorted_candidates = sorted(filtered, key=lambda x: x.final_score, reverse=True)
    
    for cand in sorted_candidates:
        if len(parlay) >= max_legs:
            break
        if cand.match_id in used_matches:
            continue
            
        penalty_sum = 0.0
        for leg in parlay:
            penalty = compute_correlation_penalty(cand, leg)
            penalty_sum += penalty
            
        if penalty_sum > 0.6:  # More tolerant for medium risk
            continue
            
        parlay.append(cand)
        used_matches.add(cand.match_id)
        
    combined_prob = np.prod([p.model_probability for p in parlay]) if parlay else 0.0
    
    return ParlayResult(
        risk_level=RiskLevel.MEDIUM,
        picks=parlay,
        combined_probability=combined_prob,
        reason="Balanced picks, better payout, moderate volatility (avoids picks used in low risk)"
    )


def build_high_risk_parlay(
    all_candidates: List[PickCandidate],
    match_count: int,
    low_risk_picks: List[PickCandidate],
    medium_risk_picks: List[PickCandidate]
) -> ParlayResult:
    """Build HIGH RISK parlay (different from both!)"""
    filtered = filter_candidates_by_risk(all_candidates, RiskLevel.HIGH)
    
    # Exclude picks from low and medium risk
    used_ids = set()
    for p in low_risk_picks:
        used_ids.add((p.match_id, p.market_type))
    for p in medium_risk_picks:
        used_ids.add((p.match_id, p.market_type))
        
    filtered = [p for p in filtered if (p.match_id, p.market_type) not in used_ids]
    
    if not filtered:
        return ParlayResult(
            risk_level=RiskLevel.HIGH,
            picks=[],
            combined_probability=0.0,
            reason="No valid high-risk picks available (all used by lower risk parlays)",
            is_valid=False
        )
        
    parlay = []
    used_matches = set()
    max_legs = min(3, match_count)  # Use at least 2 legs if possible
    
    sorted_candidates = sorted(filtered, key=lambda x: x.final_score, reverse=True)
    
    for cand in sorted_candidates:
        if len(parlay) >= max_legs:
            break
        if cand.match_id in used_matches:
            continue
            
        penalty_sum = 0.0
        for leg in parlay:
            penalty = compute_correlation_penalty(cand, leg)
            penalty_sum += penalty
            
        if penalty_sum > 0.9:  # Most tolerant for high risk
            continue
            
        parlay.append(cand)
        used_matches.add(cand.match_id)
        
    combined_prob = np.prod([p.model_probability for p in parlay]) if parlay else 0.0
    
    return ParlayResult(
        risk_level=RiskLevel.HIGH,
        picks=parlay,
        combined_probability=combined_prob,
        reason="Aggressive picks, higher payout, more volatility (avoids picks used in lower risk parlays)"
    )


def calculate_parlay_probability(parlay: List[PickCandidate]) -> float:
    """Calculate combined probability assuming independence"""
    if not parlay:
        return 0.0
    prob = 1.0
    for pick in parlay:
        prob *= pick.model_probability
    return prob


def render_parlay(parlay_result: ParlayResult, console: Any) -> None:
    """Render a parlay result nicely in the terminal using Rich"""
    from rich.panel import Panel
    from rich.table import Table
    
    title = f"{parlay_result.risk_level.value.upper()} RISK PARLAY"
    
    if not parlay_result.is_valid or not parlay_result.picks:
        console.print(Panel(
            f"[yellow]{title}[/]\n\n[red]{parlay_result.reason}[/]",
            title=title,
            border_style="yellow"
        ))
        return
        
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Match", style="white")
    table.add_column("Pick", style="green")
    table.add_column("Probability", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Rationale", style="dim")
    
    for i, pick in enumerate(parlay_result.picks, 1):
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
        
    console.print(Panel(
        table,
        title=title,
        border_style={
            RiskLevel.LOW: "green",
            RiskLevel.MEDIUM: "blue",
            RiskLevel.HIGH: "red"
        }[parlay_result.risk_level]
    ))
    
    console.print(f"[bold]Combined Probability: {parlay_result.combined_probability:.1%}[/]")
    console.print(f"[bold]Number of Legs: {len(parlay_result.picks)}[/]")
    console.print(f"[italic]{parlay_result.reason}[/]\n")


def build_all_parlays(
    match_predictions: List[Dict[str, Any]]
) -> Tuple[List[ParlayResult], CalibrationStatus]:
    """Build all three risk-level parlays"""
    calib_status = check_calibration()
    
    # Collect all candidates
    all_candidates = []
    for pred in match_predictions:
        home = pred['team_context']['home']['team']
        away = pred['team_context']['away']['team']
        cands = generate_market_candidates(pred, home, away, calib_status)
        all_candidates.extend(cands)
        
    match_count = len(match_predictions)
    
    # Build each parlay in order
    low_parlay = build_low_risk_parlay(all_candidates, match_count)
    medium_parlay = build_medium_risk_parlay(all_candidates, match_count, low_parlay.picks)
    high_parlay = build_high_risk_parlay(all_candidates, match_count, low_parlay.picks, medium_parlay.picks)
    
    parlays = [low_parlay, medium_parlay, high_parlay]
    
    return parlays, calib_status
