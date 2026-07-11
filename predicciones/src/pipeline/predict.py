"""
Unified football prediction pipeline using ESPN World Cup data with static ratings fallback.
Integrates derived market datasets (team_match_stats, player_match_stats) for corners,
cards, shots on target, and player props predictions.
"""
import os
import logging
from datetime import datetime, UTC
from typing import Dict, Any, Tuple, Optional

from ..utils.config_loader import config
from ..data.feature_builder import MatchFeatureBuilder
from ..data.espn_client import EspnWorldCupClient
from ..models.dixon_coles import DixonColesModel
from ..models.lambda_recalibration import LambdaRecalibrator
from ..models.market_derivation import derive_all_markets
from ..models.calibration import CalibrationManager
from ..sanity.sanity_checker import run_sanity_checks
from ..models.market_models import CornersModel, CardsModel, ShotsModel, PlayerPropsModel
from ..ingestion.jsonl_loader import TeamMatchStatsLoader, PlayerMatchStatsLoader, MatchEventsLoader

logger = logging.getLogger(__name__)


def _fetch_match_sportsbook_odds(
    espn_client: EspnWorldCupClient,
    home_team: str,
    away_team: str,
    match_date: str,
) -> Dict[str, Any]:
    """
    Try to fetch sportsbook odds for the target match from ESPN scoreboard data.

    Current real coverage:
    - 1X2 odds from ESPN when present

    Future-ready placeholders:
    - double_chance
    - over_under
    - btts
    """
    default_payload = {
        "source": "espn_scoreboard",
        "matched_event": False,
        "1x2": {"home": None, "draw": None, "away": None},
        "double_chance": {},
        "over_under": {},
        "btts": {},
        "notes": [],
    }

    try:
        scoreboard_date = datetime.strptime(match_date, "%Y-%m-%d").strftime("%Y%m%d")
    except ValueError:
        default_payload["notes"].append("Could not normalize match_date for sportsbook lookup.")
        return default_payload

    try:
        matches = espn_client.get_world_cup_matches(dates=scoreboard_date, limit=500)
    except Exception as exc:
        default_payload["notes"].append(f"Sportsbook lookup failed: {exc}")
        return default_payload

    normalized_home = espn_client.normalize_team_name(home_team)
    normalized_away = espn_client.normalize_team_name(away_team)

    for match in matches:
        if (
            match.get("home_team") == normalized_home
            and match.get("away_team") == normalized_away
        ):
            payload = dict(default_payload)
            payload["matched_event"] = True
            payload["1x2"] = match.get("odds") or default_payload["1x2"]
            if not any(payload["1x2"].values()):
                payload["notes"].append("Matched ESPN event, but no 1X2 odds were available.")
            return payload

    default_payload["notes"].append("No matching ESPN event found for sportsbook odds.")
    return default_payload


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
    Deriva ranking_factor del FIFA rank con efecto MÁS PRONUNCIADO.
    Este factor debe contrarrestar cuando los datos de torneos simulados
    dan ventajas injustificadas a equipos inferiormente rankeados.
    
    Top 5: 1.10-1.15, Top 10: 1.05-1.10, Top 20: 1.02-1.05, Resto: 0.90-1.00
    """
    if fifa_rank is None or fifa_rank == "N/A":
        return 1.0
    
    try:
        rank = int(fifa_rank)
    except (ValueError, TypeError):
        return 1.0
    
    # Ajuste más agresivo para reflejar diferencias de calidad real
    if rank <= 5:
        factor = 1.12  # Elite teams get significant boost
    elif rank <= 10:
        factor = 1.08
    elif rank <= 15:
        factor = 1.05
    elif rank <= 20:
        factor = 1.03
    elif rank <= 30:
        factor = 1.00
    elif rank <= 40:
        factor = 0.97
    elif rank <= 50:
        factor = 0.94
    else:
        factor = 0.90  # Teams outside top 50 get penalized more
    
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


def _build_alternative_markets(
    home_team: str,
    away_team: str,
    lambda_h: float,
    lambda_a: float,
    team_loader: TeamMatchStatsLoader,
    player_loader: PlayerMatchStatsLoader,
    events_loader: MatchEventsLoader,
) -> Dict[str, Any]:
    """
    Build alternative markets (corners, cards, SOT, player props) from derived datasets.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        lambda_h: Expected goals home (for player props)
        lambda_a: Expected goals away (for player props)
        team_loader: Loader for team match stats
        player_loader: Loader for player match stats
        events_loader: Loader for match events
        
    Returns:
        Dict with corners, cards, shots_on_target, player_props markets
    """
    # Initialize models
    corners_model = CornersModel()
    cards_model = CardsModel()
    shots_model = ShotsModel()
    player_model = PlayerPropsModel()
    
    # Get team aggregate stats
    home_stats = team_loader.get_aggregate_stats(home_team, max_matches=10)
    away_stats = team_loader.get_aggregate_stats(away_team, max_matches=10)
    
    # Check data availability
    has_corners_data = (
        home_stats.get("matches_with_corners", 0) >= 3 or 
        away_stats.get("matches_with_corners", 0) >= 3
    )
    has_cards_data = (
        home_stats.get("matches_with_cards", 0) >= 3 or
        away_stats.get("matches_with_cards", 0) >= 3
    )
    has_sot_data = (
        home_stats.get("matches_with_sot", 0) >= 3 or
        away_stats.get("matches_with_sot", 0) >= 3
    )
    has_events_data = events_loader.has_data()
    
    # Get sample sizes for regularization
    home_corners_ss = home_stats.get("matches_with_corners", 0)
    away_corners_ss = away_stats.get("matches_with_corners", 0)
    home_cards_ss = home_stats.get("matches_with_cards", 0)
    away_cards_ss = away_stats.get("matches_with_cards", 0)
    
    # Build corners market with regularization and xG coupling
    if has_corners_data:
        home_corners_avg = home_stats.get("corners_avg") or 5.0
        away_corners_avg = away_stats.get("corners_avg") or 5.0
        
        total_corners_pred = corners_model.predict_total_corners(
            home_corners_avg, away_corners_avg,
            home_sample_size=home_corners_ss,
            away_sample_size=away_corners_ss,
            home_xg=lambda_h,
            away_xg=lambda_a,
        )
        team_corners_pred = corners_model.predict_team_corners(
            home_corners_avg, away_corners_avg,
            home_sample_size=home_corners_ss,
            away_sample_size=away_corners_ss,
        )
        more_corners_pred = corners_model.predict_more_corners(
            home_corners_avg, away_corners_avg,
            home_sample_size=home_corners_ss,
            away_sample_size=away_corners_ss,
        )
        
        corners_market = {
            "available": True,
            "expected_total": total_corners_pred["expected_total"],
            "total_lines": total_corners_pred["lines"],
            "team_lines": team_corners_pred,
            "more_corners": more_corners_pred,
            "first_corner": {"available": has_events_data} if has_events_data else {"available": False, "reason": "No event sequence data"},
            "sample_size": home_corners_ss + away_corners_ss,
            "effective_sample_size": total_corners_pred.get("effective_sample_size", home_corners_ss + away_corners_ss),
        }
    else:
        corners_market = {
            "available": False,
            "reason": f"Insufficient corner data (home: {home_corners_ss}, away: {away_corners_ss})",
        }
    
    # Build cards market with regularization and xG coupling
    if has_cards_data:
        home_cards_avg = home_stats.get("cards_avg") or 2.0
        away_cards_avg = away_stats.get("cards_avg") or 2.0
        
        total_cards_pred = cards_model.predict_total_cards(
            home_cards_avg, away_cards_avg,
            home_sample_size=home_cards_ss,
            away_sample_size=away_cards_ss,
            home_xg=lambda_h,
            away_xg=lambda_a,
        )
        team_cards_pred = cards_model.predict_team_cards(
            home_cards_avg, away_cards_avg,
            home_sample_size=home_cards_ss,
            away_sample_size=away_cards_ss,
        )
        more_cards_pred = cards_model.predict_more_cards(
            home_cards_avg, away_cards_avg,
            home_sample_size=home_cards_ss,
            away_sample_size=away_cards_ss,
        )
        
        cards_market = {
            "available": True,
            "expected_total": total_cards_pred["expected_total"],
            "total_lines": total_cards_pred["lines"],
            "team_lines": team_cards_pred,
            "more_cards": more_cards_pred,
            "first_card": {"available": has_events_data} if has_events_data else {"available": False, "reason": "No event sequence data"},
            "sample_size": home_cards_ss + away_cards_ss,
            "effective_sample_size": total_cards_pred.get("effective_sample_size", home_cards_ss + away_cards_ss),
        }
    else:
        cards_market = {
            "available": False,
            "reason": f"Insufficient card data (home: {home_cards_ss}, away: {away_cards_ss})",
        }
    
    # Build shots on target market
    if has_sot_data:
        # Use SOT from team stats - need to compute from raw matches
        home_matches = team_loader.get_team_matches(home_team, max_matches=10)
        away_matches = team_loader.get_team_matches(away_team, max_matches=10)
        
        home_sot_avg = sum(m.get("shots_on_target") or 0 for m in home_matches if m.get("shots_on_target") is not None) / max(len(home_matches), 1)
        away_sot_avg = sum(m.get("shots_on_target") or 0 for m in away_matches if m.get("shots_on_target") is not None) / max(len(away_matches), 1)
        
        total_sot_pred = shots_model.predict_total_sot(home_sot_avg, away_sot_avg)
        team_sot_pred = shots_model.predict_team_sot(home_sot_avg, away_sot_avg)
        
        shots_market = {
            "available": True,
            "expected_total": total_sot_pred["expected_total"],
            "total_lines": total_sot_pred["lines"],
            "team_lines": team_sot_pred,
            "sample_size": home_stats.get("matches_with_sot", 0) + away_stats.get("matches_with_sot", 0),
        }
    else:
        shots_market = {
            "available": False,
            "reason": f"Insufficient SOT data (home: {home_stats.get('matches_with_sot', 0)}, away: {away_stats.get('matches_with_sot', 0)})",
        }
    
    # Build player props market with proper normalization
    home_players = player_loader.get_team_players(home_team, max_matches=10)
    away_players = player_loader.get_team_players(away_team, max_matches=10)
    
    has_player_data = len(home_players) > 0 or len(away_players) > 0
    
    if has_player_data:
        # Get scorer candidates for both teams
        home_scorers = player_loader.get_scorer_candidates(home_team, min_goals=0, max_matches=10)
        away_scorers = player_loader.get_scorer_candidates(away_team, min_goals=0, max_matches=10)
        
        # Build player data lists for normalized prediction
        home_players_list = []
        for scorer in home_scorers[:8]:
            pname = scorer["player_name"]
            pdata = home_players.get(pname)
            if pdata:
                pdata["player_name"] = pname
                pdata["team"] = home_team
                home_players_list.append(pdata)
        
        away_players_list = []
        for scorer in away_scorers[:8]:
            pname = scorer["player_name"]
            pdata = away_players.get(pname)
            if pdata:
                pdata["player_name"] = pname
                pdata["team"] = away_team
                away_players_list.append(pdata)
        
        # Use normalized anytime scorer prediction (sum <= team_xg)
        home_anytime = player_model.predict_anytime_scorer_normalized(
            home_players_list, lambda_h, team_total_shots=None
        )
        away_anytime = player_model.predict_anytime_scorer_normalized(
            away_players_list, lambda_a, team_total_shots=None
        )
        
        # Combine and sort
        anytime_scorers = home_anytime + away_anytime
        anytime_scorers.sort(key=lambda x: x["probability"], reverse=True)
        
        # Use normalized first scorer prediction (sum + P(no goal) = 1)
        # For first scorer, we need to consider both teams together
        all_anytime_for_first = home_anytime + away_anytime
        total_xg = lambda_h + lambda_a
        
        first_scorers = player_model.predict_first_scorer_normalized(
            all_anytime_for_first, total_xg
        )
        
        # Validation: check probability sums using INTERNAL decimal values (not percentages)
        # home_anytime_sum and away_anytime_sum should be comparable to lambda values (2-3 range)
        anytime_sum_home = sum(p.get("probability_decimal", p["probability"] / 100 if p["probability"] > 1 else p["probability"]) 
                               for p in home_anytime)
        anytime_sum_away = sum(p.get("probability_decimal", p["probability"] / 100 if p["probability"] > 1 else p["probability"]) 
                               for p in away_anytime)
        
        # For first scorer, use decimal values
        first_scorer_sum = sum(
            p.get("probability_decimal", p["probability"] / 100 if p["probability"] > 1 else p["probability"]) 
            for p in first_scorers if p["player_name"] != "[NO GOAL]"
        )
        
        logger.info(f"Player props validation: anytime_home_sum={anytime_sum_home:.3f} (λ_h={lambda_h:.2f}), anytime_away_sum={anytime_sum_away:.3f} (λ_a={lambda_a:.2f}), first_scorer_sum={first_scorer_sum:.3f}")
        
        player_props_market = {
            "available": True,
            "anytime_scorer": {
                "available": len(anytime_scorers) > 0,
                "top_candidates": anytime_scorers[:10],
                "validation": {
                    "home_anytime_sum": round(anytime_sum_home, 4),  # Sum of decimal probabilities
                    "away_anytime_sum": round(anytime_sum_away, 4),  # Sum of decimal probabilities
                    "home_lambda": round(lambda_h, 2),
                    "away_lambda": round(lambda_a, 2),
                },
            },
            "first_scorer": {
                "available": len(first_scorers) > 0,
                "top_candidates": [p for p in first_scorers if p["player_name"] != "[NO GOAL]"][:10],
                "no_goal_probability": next((p["probability"] for p in first_scorers if p["player_name"] == "[NO GOAL]"), None),
                "validation": {
                    "scorer_sum": round(first_scorer_sum * 100, 4),  # Express as percentage for display
                    "total_with_no_goal": round(sum(p.get("probability_decimal", p["probability"] / 100 if p["probability"] > 1 else p["probability"]) for p in first_scorers) * 100, 4),
                },
            },
            "assists": {"available": False, "reason": "Insufficient assist data for reliable predictions"},
            "player_shots_on_target": {"available": False, "reason": "Insufficient player-level SOT data"},
            "coverage_summary": {
                "home_players_tracked": len(home_players),
                "away_players_tracked": len(away_players),
                "home_scorers_identified": len(home_scorers),
                "away_scorers_identified": len(away_scorers),
            },
        }
    else:
        player_props_market = {
            "available": False,
            "reason": "No player-level data available for either team",
        }
    
    return {
        "corners": corners_market,
        "cards": cards_market,
        "shots_on_target": shots_market,
        "player_props": player_props_market,
    }


def predict_match_pipeline(
    home_team: str,
    away_team: str,
    match_date: str = None,
    neutral_venue: bool = False,
    include_markets: bool = True,
    competition_name: str = "International Friendly",
    competition_slug: str = None,
    **kwargs  # absorbe refresh_data, api_source, etc. para compatibilidad
) -> Dict[str, Any]:
    """
    Unified football prediction pipeline using ESPN World Cup data with static ratings fallback.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        match_date: Match date (YYYY-MM-DD), defaults to today
        neutral_venue: Whether match is at neutral venue
        include_markets: If True, include alternative markets (corners, cards, SOT, player props)
        competition_name: Name of the competition (e.g., "International Friendly", "FIFA World Cup")
        competition_slug: Slug for competition type (e.g., "fifa.world", "esp.1")
        
    Returns:
        Dict with match predictions and optional alternative markets
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
    
    # Load and run Lambda Recalibrator with context-aware compression
    recalibrator_path = "output/calibrators/lambda_recalibrator.pkl"
    try:
        recalibrator = LambdaRecalibrator.load(recalibrator_path, config=config)
        
        # Determine competition type for context-aware compression
        if 'world' in competition_name.lower() or 'cup' in competition_name.lower():
            comp_type = 'world_cup'
        elif 'friendly' in competition_name.lower():
            comp_type = 'friendly'
        elif any(league in competition_name.lower() for league in ['premier', 'la liga', 'serie a', 'bundesliga', 'ligue 1']):
            comp_type = 'league_top'
        else:
            comp_type = 'default'
        
        lambda_h, lambda_a = recalibrator.recalibrate(
            raw_lambda_h, raw_lambda_a,
            competition_type=comp_type,
            competition_slug=competition_slug
        )
        logger.info(
            f"Lambda recalibrated: ({raw_lambda_h:.3f}, {raw_lambda_a:.3f}) -> "
            f"({lambda_h:.3f}, {lambda_a:.3f}), total: {lambda_h + lambda_a:.3f}"
        )
    except FileNotFoundError:
        logger.warning("Lambda recalibrator not found, using raw lambdas")
        lambda_h, lambda_a = raw_lambda_h, raw_lambda_a

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

    # Step 8: Build alternative markets if requested
    alternative_markets = {}
    if include_markets:
        try:
            team_loader = TeamMatchStatsLoader()
            player_loader = PlayerMatchStatsLoader()
            events_loader = MatchEventsLoader()
            
            alternative_markets = _build_alternative_markets(
                home_team, away_team,
                lambda_h, lambda_a,
                team_loader, player_loader, events_loader
            )
            logger.info(f"Built alternative markets: corners={alternative_markets.get('corners', {}).get('available')}, cards={alternative_markets.get('cards', {}).get('available')}, sot={alternative_markets.get('shots_on_target', {}).get('available')}, player_props={alternative_markets.get('player_props', {}).get('available')}")
        except Exception as e:
            logger.warning(f"Failed to build alternative markets: {e}")
            alternative_markets = {
                "corners": {"available": False, "reason": str(e)},
                "cards": {"available": False, "reason": str(e)},
                "shots_on_target": {"available": False, "reason": str(e)},
                "player_props": {"available": False, "reason": str(e)},
            }

    sportsbook_odds = _fetch_match_sportsbook_odds(
        espn_client=espn_client,
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
    )

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
        "sportsbook_odds": sportsbook_odds,
        "markets": alternative_markets,  # Alternative markets block
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
            "fetched_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": overall_source,
            "home_data_quality": "high" if len(home_profile.data_warnings) == 0 else "medium" if len(home_profile.data_warnings) <= 2 else "low",
            "away_data_quality": "high" if len(away_profile.data_warnings) == 0 else "medium" if len(away_profile.data_warnings) <= 2 else "low",
            "warnings": all_warnings,
            "match_sample_home": len(home_profile.wc_form.get("matches", [])),
            "match_sample_away": len(away_profile.wc_form.get("matches", []))
        }
    }
    return response
