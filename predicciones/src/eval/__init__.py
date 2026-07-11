"""
Evaluation module for probability calibration and backtesting.

This module provides tools for:
- Computing match probabilities from lambda parameters
- Evaluating calibration metrics (Brier, Log Loss, ECE)
- Running temporal backtests comparing model configurations
"""

from .probability_calibration import (
    compute_match_probabilities,
    compute_brier_score,
    compute_log_loss,
    compute_multiclass_log_loss,
    compute_reliability_curve,
    compute_ece,
    evaluate_market_predictions,
    ReliabilityCurveData,
    MarketMetrics,
)

__all__ = [
    # Probability computation
    'compute_match_probabilities',
    
    # Calibration metrics
    'compute_brier_score',
    'compute_log_loss',
    'compute_multiclass_log_loss',
    'compute_ece',
    
    # Reliability analysis
    'compute_reliability_curve',
    'ReliabilityCurveData',
    
    # Market evaluation
    'evaluate_market_predictions',
    'MarketMetrics',
]
