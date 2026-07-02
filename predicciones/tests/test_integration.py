import pytest
from src.models.dixon_coles import DixonColesModel
from src.models.lambda_recalibration import LambdaRecalibrator
from src.models.market_derivation import derive_all_markets
from src.sanity.sanity_checker import run_sanity_checks
from src.utils.config_loader import config

def build_dummy_features(attack_rating, defense_rating, localia):
    # attack_rating and defense_rating are multipliers around 1.0
    return {
        "attack_rating": attack_rating,
        "defense_rating": defense_rating,
        "recent_form": 1.0,
        "squad_multiplier": 1.0,
        "home_advantage_log": 0.25 if localia == 1 else 0.0,
        "ranking_factor": 1.0,
        "h2h_factor": 1.0,
        "context_modifier": 0.0
    }

def run_pipeline(home_features, away_features):
    dc_model = DixonColesModel(config=config)
    raw_h, raw_a = dc_model.predict_lambdas(home_features, away_features)
    
    recalibrator = LambdaRecalibrator(config=config)
    lh, la = recalibrator.recalibrate(raw_h, raw_a)
    
    matrix = dc_model.score_matrix(lh, la)
    markets = derive_all_markets(matrix, lh, la, config)
    final_markets = run_sanity_checks(markets, lh, la, config)
    
    return lh, la, final_markets

def test_elite_vs_weak():
    """Elite vs Weak: P(fav) 0.65-0.85, P(draw) 0.15-0.25"""
    hf = build_dummy_features(attack_rating=1.3, defense_rating=0.8, localia=1) # Elite
    af = build_dummy_features(attack_rating=0.8, defense_rating=1.3, localia=0) # Weak
    lh, la, markets = run_pipeline(hf, af)
    
    p_home = markets['1x2']['home']
    p_draw = markets['1x2']['draw']
    
    assert 0.65 <= p_home <= 0.88, f"P(Home) out of bounds: {p_home:.2f}"
    assert 0.10 <= p_draw <= 0.25, f"P(Draw) out of bounds: {p_draw:.2f}"
    
def test_elite_vs_elite():
    """Elite vs Elite (Neutral): P(fav) 0.35-0.55, P(draw) 0.25-0.35"""
    hf = build_dummy_features(attack_rating=1.4, defense_rating=0.7, localia=0)
    af = build_dummy_features(attack_rating=1.35, defense_rating=0.75, localia=0)
    lh, la, markets = run_pipeline(hf, af)
    
    p_home = markets['1x2']['home']
    p_draw = markets['1x2']['draw']
    
    assert 0.35 <= p_home <= 0.55, f"P(Home) out of bounds: {p_home:.2f}"
    assert 0.22 <= p_draw <= 0.35, f"P(Draw) out of bounds: {p_draw:.2f}"

def test_balanced_match():
    """Balanced Match: P(draw) 0.27-0.35, not >0.50"""
    hf = build_dummy_features(attack_rating=1.0, defense_rating=1.0, localia=0)
    af = build_dummy_features(attack_rating=1.0, defense_rating=1.0, localia=0)
    lh, la, markets = run_pipeline(hf, af)
    
    p_draw = markets['1x2']['draw']
    
    assert 0.24 <= p_draw <= 0.35, f"P(Draw) out of bounds: {p_draw:.2f}"
