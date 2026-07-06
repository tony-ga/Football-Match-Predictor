import os
import logging
from datetime import datetime
from typing import Dict, Any, Tuple

from ..utils.config_loader import config
from ..data.feature_builder import MatchFeatureBuilder
from ..data.espn_client import EspnWorldCupClient
from ..models.dixon_coles import DixonColesModel
from ..models.lambda_recalibration import LambdaRecalibrator
from ..models.market_derivation import derive_all_markets
from ..models.calibration import CalibrationManager
from ..sanity.sanity_checker import run_sanity_checks

logger = logging.getLogger(__name__)


def _compute_form_factor(recent_form: Dict[str, Any]) -> float:
    """
    Deriva form_factor de los últimos 3-5 partidos ESPN.
    - buena forma: 1.03 a 1.10
    - mala forma: 0.90 a 0.99
    - neutral: 1.0
    """
    if not recent_form or recent_form.get("record") == "N/A":
        return 1.0
    
    matches_played = recent_form.get("matches_played", 0)
    if matches_played == 0:
        return 1.0
    
    # Extraer W/D/L del record
    record = recent_form.get("record", "")
    wins = 0
    draws = 0
    losses = 0
    
    if "W" in record:
        try:
            wins = int(record.split("W")[1].split()[0].replace("D", "").split("L")[0] if "D" in record else record.split("W")[1].split("L")[0])
        except (ValueError, IndexError):
            pass
    
    if "D" in record:
        try:
            draws = int(record.split("D")[1].split()[0].replace("L", ""))
        except (ValueError, IndexError):
            pass
    
    if "L" in record:
        try:
            losses = int(record.split("L")[1].split()[0])
        except (ValueError, IndexError):
            pass
    
    # Calcular puntos por partido (3 por victoria, 1 por empate)
    points_per_match = (wins * 3 + draws) / matches_played if matches_played > 0 else 0
    
    # Mapear a form_factor
    # 3.0 ppM -> 1.10, 2.0 ppM -> 1.05, 1.0 ppM -> 1.0, 0.5 ppM -> 0.95, 0 -> 0.90
    if points_per_match >= 2.5:
        form_factor = 1.08 + (points_per_match - 2.5) * 0.04  # up to 1.10
    elif points_per_match >= 2.0:
        form_factor = 1.03 + (points_per_match - 2.0) * 0.10  # up to 1.08
    elif points_per_match >= 1.5:
        form_factor = 1.00 + (points_per_match - 1.5) * 0.06  # up to 1.03
    elif points_per_match >= 1.0:
        form_factor = 0.97 + (points_per_match - 1.0) * 0.06  # up to 1.00
    elif points_per_match >= 0.5:
        form_factor = 0.93 + (points_per_match - 0.5) * 0.08  # up to 0.97
    else:
        form_factor = 0.90 + points_per_match * 0.06  # up to 0.93
    
    return round(max(0.90, min(1.10, form_factor)), 3)


def _compute_ranking_factor(fifa_rank: int, opponent_rank: int = None) -> float:
    """
    Deriva ranking_factor del FIFA rank con efecto pequeño.
    Top 10: 1.02-1.05, Top 20: 1.00-1.02, Resto: 0.95-1.00
    """
    if fifa_rank is None or fifa_rank == "N/A":
        return 1.0
    
    try:
        rank = int(fifa_rank)
    except (ValueError, TypeError):
        return 1.0
    
    if rank <= 5:
        factor = 1.05
    elif rank <= 10:
        factor = 1.03
    elif rank <= 20:
        factor = 1.01
    elif rank <= 30:
        factor = 1.00
    elif rank <= 50:
        factor = 0.98
    else:
        factor = 0.96
    
    return round(factor, 3)


def _compute_context_modifier(
    stage: str = "group",
    neutral_venue: bool = False,
    wc_form: Dict[str, Any] = None
) -> float:
    """
    Pequeño ajuste por knockout stage, neutralidad y stats recientes.
    Rango típico: -0.05 a +0.05
    """
    modifier = 0.0
    
    # Knockout stage bonus
    stage_bonus = {
        "round_of_32": 0.01,
        "round_of_16": 0.02,
        "quarter_final": 0.03,
        "semi_final": 0.04,
        "final": 0.05,
        "third_place": 0.01,
    }
    modifier += stage_bonus.get(stage, 0.0)
    
    # Neutral venue: pequeño penalty para ambos (quita home advantage implícito)
    if neutral_venue:
        modifier -= 0.01
    
    # Bonus por buena forma en WC
    if wc_form and wc_form.get("played", 0) > 0:
        goals_scored = wc_form.get("goals_scored", 0)
        goals_conceded = wc_form.get("goals_conceded", 0)
        goal_diff = goals_scored - goals_conceded
        
        if goal_diff > 0:
            modifier += min(0.02, goal_diff * 0.01)
        elif goal_diff < 0:
            modifier += max(-0.02, goal_diff * 0.01)
    
    return round(max(-0.05, min(0.05, modifier)), 3)


def predict_match_pipeline(
    home_team: str,
    away_team: str,
    match_date: str = None,
    neutral_venue: bool = False,
    **kwargs  # absorbe refresh_data, api_source, etc. para compatibilidad
) -> Dict[str, Any]:
    """
    Unified football prediction pipeline using ESPN World Cup data with static ratings fallback.
    """
    if match_date is None:
        match_date = datetime.today().strftime("%Y-%m-%d")

    # Step 1: Initialize ESPN client and Feature Builder
    espn_client = EspnWorldCupClient()
    builder = MatchFeatureBuilder(api_client=None, espn_client=espn_client)

    # Step 2: Build Team Profiles from ESPN + static ratings
    logger.info(f"Building profile for {home_team}...")
    home_profile = builder.build_team_profile(home_team, match_date, include_wc_matches=True)
    
    logger.info(f"Building profile for {away_team}...")
    away_profile = builder.build_team_profile(away_team, match_date, include_wc_matches=True)

    # Step 3: Compute features with ESPN-derived factors
    home_recent = home_profile.recent_form or {}
    away_recent = away_profile.recent_form or {}
    home_wc = home_profile.wc_form or {}
    away_wc = away_profile.wc_form or {}
    
    # Form factors from ESPN recent matches
    home_form_factor = _compute_form_factor(home_recent)
    away_form_factor = _compute_form_factor(away_recent)
    
    # Ranking factors from FIFA rank
    home_ranking_factor = _compute_ranking_factor(home_recent.get("fifa_rank"))
    away_ranking_factor = _compute_ranking_factor(away_recent.get("fifa_rank"))
    
    # Context modifiers
    home_context = _compute_context_modifier("group", neutral_venue, home_wc)
    away_context = _compute_context_modifier("group", neutral_venue, away_wc)
    
    # H2H y squad se mantienen en 1.0 si no hay datos
    h2h_factor = 1.0
    squad_multiplier = 1.0

    home_features = {
        'nombre': home_team,
        'attack_rating': home_profile.lambda_attack,
        'defense_rating': home_profile.lambda_defense,
        'form_factor': home_form_factor,
        'ranking_factor': home_ranking_factor,
        'h2h_factor': h2h_factor,
        'squad_multiplier': squad_multiplier,
        'home_advantage_log': 0.0 if neutral_venue else config.get('dixon_coles', {}).get('home_advantage', 0.25),
        'context_modifier': home_context
    }
    
    away_features = {
        'nombre': away_team,
        'attack_rating': away_profile.lambda_attack,
        'defense_rating': away_profile.lambda_defense,
        'form_factor': away_form_factor,
        'ranking_factor': away_ranking_factor,
        'h2h_factor': h2h_factor,
        'squad_multiplier': squad_multiplier,
        'home_advantage_log': 0.0,
        'context_modifier': away_context
    }

    # Step 4: Dixon-Coles model & Lambda Recalibration
    dc_model = DixonColesModel(config=config)
    dc_model.max_goals = 6
    
    raw_lambda_h, raw_lambda_a = dc_model.predict_lambdas(home_features, away_features)
    
    # Load and run Lambda Recalibrator
    recalibrator_path = "output/calibrators/lambda_recalibrator.pkl"
    recalibrator = LambdaRecalibrator.load(recalibrator_path, config=config)
    lambda_h, lambda_a = recalibrator.recalibrate(raw_lambda_h, raw_lambda_a)

    # Calculate score matrix with Cap of 6 goals
    matrix = dc_model.score_matrix(lambda_h, lambda_a)

    # Step 5: Derive markets
    markets = derive_all_markets(matrix, lambda_h, lambda_a, config)

    # Step 6: Calibrate markets
    calibrator_mgr = CalibrationManager(config=config)
    try:
        calibrator_mgr.load_from_config()
    except Exception as e:
        logger.warning(f"Could not load calibrators: {e}")
    
    calibrated_markets = calibrator_mgr.calibrate_markets(markets)

    # Step 7: Apply Sanity Checks after calibration
    final_markets = run_sanity_checks(calibrated_markets, lambda_h, lambda_a, config)

    # Determine overall data source
    all_warnings = home_profile.data_warnings + away_profile.data_warnings
    if "espn_world_cup" in home_profile.data_source or "espn_world_cup" in away_profile.data_source:
        if "blended" in home_profile.data_source or "blended" in away_profile.data_source:
            overall_source = "espn_world_cup_blended_static"
        else:
            overall_source = "espn_world_cup"
    elif "static" in home_profile.data_source and "static" in away_profile.data_source:
        overall_source = "static_ratings"
    else:
        overall_source = "confederation_prior"

    # Construct output JSON with enriched team context
    response = {
        "match": f"{home_team} vs {away_team}",
        "predictions": final_markets,
        "team_context": {
            "home": {
                "team": home_team,
                "data_source": home_profile.data_source,
                "last_10_external": home_profile.recent_form,
                "wc_2026_matches": home_profile.wc_form,
                "lambda_attack": round(home_profile.lambda_attack, 3),
                "lambda_defense": round(home_profile.lambda_defense, 3),
                "corners_lambda": round(home_profile.corners_lambda, 3),
                "cards_lambda": round(home_profile.cards_lambda, 3),
                "effective_weight_matches": home_profile.effective_weight_matches,
                "data_warnings": home_profile.data_warnings,
                "form_factor": home_form_factor,
                "ranking_factor": home_ranking_factor,
                "context_modifier": home_context
            },
            "away": {
                "team": away_team,
                "data_source": away_profile.data_source,
                "last_10_external": away_profile.recent_form,
                "wc_2026_matches": away_profile.wc_form,
                "lambda_attack": round(away_profile.lambda_attack, 3),
                "lambda_defense": round(away_profile.lambda_defense, 3),
                "corners_lambda": round(away_profile.corners_lambda, 3),
                "cards_lambda": round(away_profile.cards_lambda, 3),
                "effective_weight_matches": away_profile.effective_weight_matches,
                "data_warnings": away_profile.data_warnings,
                "form_factor": away_form_factor,
                "ranking_factor": away_ranking_factor,
                "context_modifier": away_context
            }
        },
        "data_freshness": {
            "fetched_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": overall_source,
            "home_data_quality": "high" if len(home_profile.data_warnings) == 0 else "medium" if len(home_profile.data_warnings) <= 2 else "low",
            "away_data_quality": "high" if len(away_profile.data_warnings) == 0 else "medium" if len(away_profile.data_warnings) <= 2 else "low",
            "warnings": all_warnings,
            "match_sample_home": len(home_profile.wc_form.get("matches", [])),
            "match_sample_away": len(away_profile.wc_form.get("matches", []))
        }
    }
    return response
