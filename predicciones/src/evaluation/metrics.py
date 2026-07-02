"""
Evaluation metrics module.
Calculates Log-loss, Brier Score, and reliability bounds for probability outputs.
Designed to be applied in layers (Base -> Post-Sanity -> Calibrated) to track
improvements across the pipeline.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List

import numpy as np
from sklearn.metrics import log_loss, brier_score_loss

logger = logging.getLogger(__name__)


def evaluate_layer(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    market_type: str = 'binary'
) -> Dict[str, float]:
    """
    Evaluates a set of predictions against truth for a single layer.
    
    Args:
        y_true: 1D array for binary (0,1), 2D one-hot for multi-class (1x2).
        y_pred: Probabilities (1D for binary, 2D for multi-class).
        market_type: 'binary' (BTTS, O/U) or 'multi' (1X2).
        
    Returns:
        Dict with metrics.
    """
    metrics = {}
    
    # Clip predictions to prevent infinite log loss
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    if market_type == 'binary':
        metrics['brier_score'] = brier_score_loss(y_true, y_pred)
        metrics['log_loss'] = log_loss(y_true, y_pred)
    elif market_type == 'multi':
        metrics['log_loss'] = log_loss(y_true, y_pred)
        # Brier for multi-class is sum of squared differences across all classes
        metrics['brier_score'] = np.mean(np.sum((y_pred - y_true)**2, axis=1))

    return metrics


def build_reliability_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Generates data for a reliability plot (calibration curve).
    Returns mean predicted value and true fraction of positives per bin.
    """
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
    
    return {
        'prob_true': prob_true.tolist(),
        'prob_pred': prob_pred.tolist()
    }


def compare_pipeline_layers(
    y_true: np.ndarray,
    preds_base: np.ndarray,
    preds_sanity: np.ndarray,
    preds_calibrated: np.ndarray,
    market_type: str = 'binary'
) -> Dict[str, Any]:
    """
    Compare log-loss and brier score across the 3 pipeline stages
    to prove that sanity and calibration layers improve (or at least don't degrade)
    the mathematical coherence.
    """
    return {
        'base': evaluate_layer(y_true, preds_base, market_type),
        'post_sanity': evaluate_layer(y_true, preds_sanity, market_type),
        'post_calibration': evaluate_layer(y_true, preds_calibrated, market_type)
    }
