import os
import logging
from datetime import datetime
from typing import Dict, Any, Tuple

from ..utils.config_loader import config
from ..data.api_client import FootballAPIClient
from ..data.feature_builder import MatchFeatureBuilder
from ..models.dixon_coles import DixonColesModel
from ..models.lambda_recalibration import LambdaRecalibrator
from ..models.market_derivation import derive_all_markets
from ..models.calibration import CalibrationManager
from ..sanity.sanity_checker import run_sanity_checks

logger = logging.getLogger(__name__)

def predict_match_pipeline(
    home_team: str,
    away_team: str,
    match_date: str = None,
    neutral_venue: bool = False,
    refresh_data: bool = False,
    api_source: str = "auto"
) -> Dict[str, Any]:
    """
    Unified football prediction pipeline with real-time API fetching,
    decay-weighted feature building, Dixon-Coles prediction, market calibration,
    sanity checks, and enriched JSON output.
    """
    if match_date is None:
        match_date = datetime.today().strftime("%Y-%m-%d")

    # Step 1: Initialize API Client & Feature Builder
    # If refresh_data is True, we can clear cache or ignore it.
    api_key = os.getenv("API_FOOTBALL_KEY") or config.get("api_data_fetching", {}).get("api_football_key")
    api_client = FootballAPIClient(api_key=api_key, source=api_source)
    
    if refresh_data:
        # Clear in-memory cache
        api_client.cache.clear()
        logger.info("Cleared API client cache per request.")

    builder = MatchFeatureBuilder(api_client)

    # Step 2: Build Team Profiles (Data Fetch & Feature Building)
    logger.info(f"Building profile for {home_team}...")
    home_profile = builder.build_team_profile(home_team, match_date, include_wc_matches=True)
    
    logger.info(f"Building profile for {away_team}...")
    away_profile = builder.build_team_profile(away_team, match_date, include_wc_matches=True)

    # Extract source info for freshness metadata
    home_source = "local_dataset"
    if home_profile.effective_weight_matches > 0:
        # Infer source from profile matches
        # We checked api_football first
        home_source = "api_football" if api_client.api_football_key else "football_data"

    away_source = "local_dataset"
    if away_profile.effective_weight_matches > 0:
        away_source = "api_football" if api_client.api_football_key else "football_data"

    # Step 3: Compute raw lambdas using profile values
    # We construct mock feature dictionaries representing the profiles
    # to fit into the existing DixonColesModel heuristic estimation.
    
    home_features = {
        'nombre': home_team,
        'attack_rating': home_profile.lambda_attack,
        'defense_rating': home_profile.lambda_defense,
        'form_factor': 1.0, # Handled by weighted goal averages in profile
        'ranking_factor': 1.0,
        'h2h_factor': 1.0,
        'squad_multiplier': 1.0,
        'home_advantage_log': 0.0 if neutral_venue else config.get('dixon_coles', {}).get('home_advantage', 0.25),
        'context_modifier': 0.0
    }
    
    away_features = {
        'nombre': away_team,
        'attack_rating': away_profile.lambda_attack,
        'defense_rating': away_profile.lambda_defense,
        'form_factor': 1.0,
        'ranking_factor': 1.0,
        'h2h_factor': 1.0,
        'squad_multiplier': 1.0,
        'home_advantage_log': 0.0,
        'context_modifier': 0.0
    }

    # Step 4: Dixon-Coles model & Lambda Recalibration
    dc_model = DixonColesModel(config=config)
    # Override model's max goals to 6 per user request
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

    # Step 6: Calibrate markets (Before Sanity Checks, per user spec)
    calibrator_mgr = CalibrationManager(config=config)
    try:
        calibrator_mgr.load_from_config()
    except Exception as e:
        logger.warning(f"Could not load calibrators: {e}")
    
    calibrated_markets = calibrator_mgr.calibrate_markets(markets)

    # Step 7: Apply Sanity Checks after calibration
    final_markets = run_sanity_checks(calibrated_markets, lambda_h, lambda_a, config)

    # Construct output JSON with enriched team context
    response = {
        "match": f"{home_team} vs {away_team}",
        "predictions": final_markets,
        "team_context": {
            "home": {
                "team": home_team,
                "data_source": home_source,
                "last_10_external": home_profile.recent_form,
                "wc_2026_matches": home_profile.wc_form,
                "lambda_attack": round(home_profile.lambda_attack, 3),
                "lambda_defense": round(home_profile.lambda_defense, 3),
                "corners_lambda": round(home_profile.corners_lambda, 3),
                "cards_lambda": round(home_profile.cards_lambda, 3),
                "effective_weight_matches": home_profile.effective_weight_matches
            },
            "away": {
                "team": away_team,
                "data_source": away_source,
                "last_10_external": away_profile.recent_form,
                "wc_2026_matches": away_profile.wc_form,
                "lambda_attack": round(away_profile.lambda_attack, 3),
                "lambda_defense": round(away_profile.lambda_defense, 3),
                "corners_lambda": round(away_profile.corners_lambda, 3),
                "cards_lambda": round(away_profile.cards_lambda, 3),
                "effective_weight_matches": away_profile.effective_weight_matches
            }
        },
        "data_freshness": {
            "fetched_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wc_fixtures_last_update": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cache_hit": False # We set it based on client caches later if desired
        }
    }
    return response
