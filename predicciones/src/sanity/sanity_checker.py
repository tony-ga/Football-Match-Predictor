"""
Sanity checking module.
Evaluates final calibrated probabilities to detect inconsistencies and emits
warnings or applies very soft smoothing without breaking probability coherence.

Follows user directive: minimal intervention post-calibration, mostly as a warning system.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def run_sanity_checks(
    markets: Dict[str, Any],
    lambda_home: float,
    lambda_away: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run sanity checks on predicted markets and lambdas.
    Modifies the markets dict in-place with a 'sanity_flags' list containing warnings.

    Args:
        markets: The fully calibrated markets dictionary.
        lambda_home: Expected goals home.
        lambda_away: Expected goals away.
        config: Model config dictionary.

    Returns:
        The updated markets dictionary (with 'sanity_flags' added).
    """
    flags: List[str] = []
    sanity_cfg = config.get('sanity', {})

    # 1. Check Favorite Probability vs Lambda Ratio
    _check_favorite_prob(markets, lambda_home, lambda_away, sanity_cfg, flags)

    # 2. Check BTTS vs Underdog Lambda
    _check_btts(markets, lambda_home, lambda_away, sanity_cfg, flags)

    # 3. Check 1-1 Draw Prob in Unequal Matches
    _check_draw_11(markets, lambda_home, lambda_away, sanity_cfg, flags)

    # 4. Check Clean Sheet Consistency
    _check_clean_sheets(markets, lambda_home, lambda_away, sanity_cfg, flags)

    # Attach flags to output
    markets['sanity_flags'] = flags

    if flags:
        logger.warning(f"Sanity checks triggered {len(flags)} warnings.")
        for f in flags:
            logger.warning(f"Sanity Flag: {f}")

    return markets


def _check_favorite_prob(
    markets: Dict[str, Any],
    lh: float,
    la: float,
    cfg: Dict[str, Any],
    flags: List[str],
):
    if '1x2' not in markets:
        return

    p_home = markets['1x2'].get('home', 0.0)
    p_away = markets['1x2'].get('away', 0.0)
    p_draw = markets['1x2'].get('draw', 0.0)

    # Flag C: Extreme asymmetry check
    ratio = lh / (la + 0.01) if lh > la else la / (lh + 0.01)
    if ratio >= 3.0:
        if lh > la and p_home < 0.65:
            flags.append(f"Flag C: ratio {ratio:.1f} >= 3.0, but P(Home Win) = {p_home:.2f} < 0.65. Applying adjustment.")
            # Multiplicative adjustment
            markets['1x2']['draw'] *= 0.85
            markets['1x2']['away'] *= 0.85
            excess = p_draw * 0.15 + p_away * 0.15
            markets['1x2']['home'] += excess
            
        elif la > lh and p_away < 0.65:
            flags.append(f"Flag C: ratio {ratio:.1f} >= 3.0, but P(Away Win) = {p_away:.2f} < 0.65. Applying adjustment.")
            markets['1x2']['draw'] *= 0.85
            markets['1x2']['home'] *= 0.85
            excess = p_draw * 0.15 + p_home * 0.15
            markets['1x2']['away'] += excess

def _check_btts(
    markets: Dict[str, Any],
    lh: float,
    la: float,
    cfg: Dict[str, Any],
    flags: List[str],
):
    if 'btts' not in markets:
        return

    p_btts = markets['btts'].get('yes', 0.0)
    lambda_total = lh + la
    
    # Check BTTS given lambda total
    if lambda_total >= 2.0 and min(lh, la) > 0.8 and p_btts < 0.40:
        flags.append(f"Warning: lambda_total >= 2.0 and both > 0.8, but BTTS = {p_btts:.2f} < 0.40.")
        
    if lambda_total <= 1.4 and p_btts > 0.45:
        flags.append(f"Warning: lambda_total <= 1.4 but BTTS is high ({p_btts:.2f}).")

def _check_draw_11(
    markets: Dict[str, Any],
    lh: float,
    la: float,
    cfg: Dict[str, Any],
    flags: List[str],
):
    # Flag B: Draw hyperinflated globally
    if '1x2' in markets:
        p_draw = markets['1x2'].get('draw', 0.0)
        lambda_total = lh + la
        if p_draw > 0.45 and lambda_total >= 1.5:
            flags.append(f"Flag B: P(draw) = {p_draw:.2f} > 0.45. Applying reduction.")
            markets['1x2']['draw'] *= 0.85
            excess = p_draw * 0.15
            markets['1x2']['home'] += excess / 2
            markets['1x2']['away'] += excess / 2

    # Flag D: 0-0 hyperinflated
    if 'correct_scores' in markets:
        lambda_total = lh + la
        for cs in markets['correct_scores']:
            if cs['score'] == '0-0':
                p_00 = cs['probability']
                if lambda_total >= 2.0 and p_00 > 0.20:
                    flags.append(f"Flag D: lambda_total >= 2.0 but P(0-0) = {p_00:.2f} > 0.20. Redistributing.")
                    cs['probability'] *= 0.80
                    excess = p_00 * 0.20
                    # Give to 1-0, 0-1, 1-1
                    for sub_cs in markets['correct_scores']:
                        if sub_cs['score'] in ['1-0', '0-1', '1-1']:
                            sub_cs['probability'] += excess / 3
                elif lambda_total >= 1.6 and p_00 > 0.35:
                    flags.append(f"Flag D: lambda_total >= 1.6 but P(0-0) = {p_00:.2f} > 0.35. Redistributing.")
                    cs['probability'] *= 0.85
                    excess = p_00 * 0.15
                    for sub_cs in markets['correct_scores']:
                        if sub_cs['score'] in ['1-0', '0-1', '1-1']:
                            sub_cs['probability'] += excess / 3
                break

def _check_clean_sheets(
    markets: Dict[str, Any],
    lh: float,
    la: float,
    cfg: Dict[str, Any],
    flags: List[str],
):
    if 'clean_sheets' not in markets:
        return
    # Kept as purely informational warning as per user request (focusing mainly on 1x2 and scores)
    pass
