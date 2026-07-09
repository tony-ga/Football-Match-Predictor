"""
Probability calibration and evaluation module.

Provides tools for:
- Computing match probabilities (1X2, O/U, BTTS) from lambda parameters
- Computing calibration metrics (Brier score, log loss, ECE)
- Generating reliability curve data for visualization

This module works with both baseline and Markov-aware models.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Probability Computation from Lambdas
# -----------------------------------------------------------------------------

def compute_match_probabilities(
    lambda_h: float,
    lambda_a: float,
    rho: float = -0.13,
    max_goals: int = 8,
) -> Dict[str, Any]:
    """
    Compute match outcome probabilities from Dixon-Coles corrected lambdas.
    
    Uses a Dixon-Coles corrected score matrix to derive probabilities for:
    - 1X2 (home win, draw, away win)
    - Over/Under 0.5, 1.5, 2.5, 3.5 goals
    - BTTS (both teams to score)
    
    Args:
        lambda_h: Expected home goals.
        lambda_a: Expected away goals.
        rho: Dixon-Coles correlation parameter (default -0.13).
        max_goals: Maximum goals per team to consider (default 8).
    
    Returns:
        Dictionary with probabilities:
        {
            "p_home_win": ...,
            "p_draw": ...,
            "p_away_win": ...,
            "p_over_0_5": ...,
            "p_under_0_5": ...,
            "p_over_1_5": ...,
            "p_under_1_5": ...,
            "p_over_2_5": ...,
            "p_under_2_5": ...,
            "p_over_3_5": ...,
            "p_under_3_5": ...,
            "p_btts_yes": ...,
            "p_btts_no": ...,
        }
    """
    # Import here to avoid circular dependency
    from ..models.dixon_coles import dc_score_matrix
    
    # Get DC-corrected score matrix
    score_matrix = dc_score_matrix(lambda_h, lambda_a, rho=rho, max_goals=max_goals)
    
    # Initialize result dict
    probs = {}
    
    # 1X2 probabilities
    p_home_win = 0.0
    p_draw = 0.0
    p_away_win = 0.0
    
    for i in range(max_goals + 1):  # home goals
        for j in range(max_goals + 1):  # away goals
            p = score_matrix[i, j]
            if i > j:
                p_home_win += p
            elif i == j:
                p_draw += p
            else:
                p_away_win += p
    
    probs["p_home_win"] = float(p_home_win)
    probs["p_draw"] = float(p_draw)
    probs["p_away_win"] = float(p_away_win)
    
    # Over/Under probabilities for multiple lines
    over_under_lines = [0.5, 1.5, 2.5, 3.5]
    for line in over_under_lines:
        total_goals = line
        p_over = 0.0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if i + j > total_goals:
                    p_over += score_matrix[i, j]
        
        p_under = 1.0 - p_over
        line_key = str(line).replace('.', '_')
        probs[f"p_over_{line_key}"] = float(p_over)
        probs[f"p_under_{line_key}"] = float(p_under)
    
    # BTTS (Both Teams To Score)
    p_btts_yes = 0.0
    for i in range(1, max_goals + 1):  # home >= 1
        for j in range(1, max_goals + 1):  # away >= 1
            p_btts_yes += score_matrix[i, j]
    
    p_btts_no = 1.0 - p_btts_yes
    probs["p_btts_yes"] = float(p_btts_yes)
    probs["p_btts_no"] = float(p_btts_no)
    
    return probs


# -----------------------------------------------------------------------------
# Calibration Metrics
# -----------------------------------------------------------------------------

def compute_brier_score(
    predictions: List[float],
    outcomes: List[int],
) -> float:
    """
    Compute Brier score for binary predictions.
    
    Brier score = mean((pred - actual)^2)
    Lower is better (0 = perfect, 1 = worst).
    
    Args:
        predictions: List of predicted probabilities [0, 1].
        outcomes: List of binary outcomes (0 or 1).
    
    Returns:
        Brier score (float).
    """
    if len(predictions) != len(outcomes):
        raise ValueError("Predictions and outcomes must have same length")
    
    return float(np.mean([(p - o) ** 2 for p, o in zip(predictions, outcomes)]))


def compute_log_loss(
    predictions: List[float],
    outcomes: List[int],
    epsilon: float = 1e-15,
) -> float:
    """
    Compute log loss (binary cross-entropy) for binary predictions.
    
    Log loss = -mean(y * log(p) + (1-y) * log(1-p))
    Lower is better (0 = perfect).
    
    Args:
        predictions: List of predicted probabilities [0, 1].
        outcomes: List of binary outcomes (0 or 1).
        epsilon: Small value to clip predictions for numerical stability.
    
    Returns:
        Log loss (float).
    """
    if len(predictions) != len(outcomes):
        raise ValueError("Predictions and outcomes must have same length")
    
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    outcomes = np.array(outcomes)
    
    return float(-np.mean(
        outcomes * np.log(predictions) + (1 - outcomes) * np.log(1 - predictions)
    ))


def compute_multiclass_log_loss(
    predictions: List[Dict[str, float]],
    outcomes: List[str],
    epsilon: float = 1e-15,
) -> float:
    """
    Compute log loss for multi-class predictions (e.g., 1X2).
    
    Args:
        predictions: List of dicts with class probabilities.
                     e.g., [{"home": 0.5, "draw": 0.3, "away": 0.2}, ...]
        outcomes: List of actual outcome keys.
                  e.g., ["home", "draw", "away", ...]
        epsilon: Small value for numerical stability.
    
    Returns:
        Multi-class log loss (float).
    """
    if len(predictions) != len(outcomes):
        raise ValueError("Predictions and outcomes must have same length")
    
    total_ll = 0.0
    for pred, outcome in zip(predictions, outcomes):
        p = pred.get(outcome, 0.0)
        p = np.clip(p, epsilon, 1 - epsilon)
        total_ll += -np.log(p)
    
    return float(total_ll / len(predictions))


@dataclass
class ReliabilityCurveData:
    """Data for plotting a reliability (calibration) curve."""
    bin_centers: List[float]       # Midpoint of each probability bin
    predicted_probs: List[float]   # Average predicted probability in bin
    actual_frequencies: List[float]  # Actual frequency of positive outcomes
    counts: List[int]              # Number of samples in each bin


def compute_reliability_curve(
    predictions: List[float],
    outcomes: List[int],
    n_bins: int = 10,
) -> ReliabilityCurveData:
    """
    Compute reliability curve data for calibration assessment.
    
    A reliability curve shows how well predicted probabilities match
    actual observed frequencies. Perfectly calibrated predictions fall
    on the diagonal (predicted = actual).
    
    Args:
        predictions: List of predicted probabilities [0, 1].
        outcomes: List of binary outcomes (0 or 1).
        n_bins: Number of bins for grouping predictions.
    
    Returns:
        ReliabilityCurveData with bin-level statistics.
    """
    if len(predictions) != len(outcomes):
        raise ValueError("Predictions and outcomes must have same length")
    
    predictions = np.array(predictions)
    outcomes = np.array(outcomes)
    
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    predicted_probs = []
    actual_frequencies = []
    counts = []
    
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        
        # Find predictions in this bin
        if i == n_bins - 1:
            mask = (predictions >= lower) & (predictions <= upper)
        else:
            mask = (predictions >= lower) & (predictions < upper)
        
        bin_preds = predictions[mask]
        bin_outcomes = outcomes[mask]
        
        count = len(bin_preds)
        counts.append(count)
        
        if count > 0:
            bin_centers.append((lower + upper) / 2)
            predicted_probs.append(float(np.mean(bin_preds)))
            actual_frequencies.append(float(np.mean(bin_outcomes)))
        else:
            # Empty bin: use center as predicted, NaN for actual
            bin_centers.append((lower + upper) / 2)
            predicted_probs.append(float((lower + upper) / 2))
            actual_frequencies.append(float('nan'))
    
    return ReliabilityCurveData(
        bin_centers=bin_centers,
        predicted_probs=predicted_probs,
        actual_frequencies=actual_frequencies,
        counts=counts
    )


def compute_ece(
    predictions: List[float],
    outcomes: List[int],
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE is a weighted average of the absolute difference between
    predicted and actual frequencies across bins.
    
    ECE = sum(n_samples_in_bin / total_samples * |avg_pred - avg_actual|)
    
    Args:
        predictions: List of predicted probabilities [0, 1].
        outcomes: List of binary outcomes (0 or 1).
        n_bins: Number of bins for grouping predictions.
    
    Returns:
        Expected Calibration Error (float, lower is better).
    """
    reliability_data = compute_reliability_curve(predictions, outcomes, n_bins)
    
    total_samples = len(predictions)
    ece = 0.0
    
    for i, count in enumerate(reliability_data.counts):
        if count > 0:
            pred = reliability_data.predicted_probs[i]
            actual = reliability_data.actual_frequencies[i]
            if not np.isnan(actual):
                ece += (count / total_samples) * abs(pred - actual)
    
    return float(ece)


# -----------------------------------------------------------------------------
# Market-Specific Evaluation Helpers
# -----------------------------------------------------------------------------

@dataclass
class MarketMetrics:
    """Metrics for a specific market (1X2, O/U, BTTS)."""
    market_name: str
    brier_score: float
    log_loss: float
    ece: float
    mae: Optional[float] = None  # For lambda-based metrics
    count: int = 0


def evaluate_market_predictions(
    predictions: List[Dict[str, float]],
    outcomes: List[Dict[str, int]],
    market_type: str,
) -> Dict[str, Any]:
    """
    Evaluate predictions for a specific market type.
    
    Args:
        predictions: List of prediction dicts.
        outcomes: List of outcome dicts (one-hot encoded).
        market_type: One of '1x2', 'over_under_25', 'btts'.
    
    Returns:
        Dictionary with metrics for this market.
    """
    metrics = {}
    
    if market_type == '1x2':
        # Multi-class evaluation
        pred_list = predictions
        outcome_keys = []
        for o in outcomes:
            if o.get('home', 0) == 1:
                outcome_keys.append('home')
            elif o.get('draw', 0) == 1:
                outcome_keys.append('draw')
            else:
                outcome_keys.append('away')
        
        metrics['log_loss'] = compute_multiclass_log_loss(pred_list, outcome_keys)
        
        # Also compute Brier for each outcome (one-vs-all)
        for outcome_key in ['home', 'draw', 'away']:
            preds = [p.get(outcome_key, 0.0) for p in pred_list]
            outs = [o.get(outcome_key, 0) for o in outcomes]
            metrics[f'brier_{outcome_key}'] = compute_brier_score(preds, outs)
        
        metrics['brier_avg'] = np.mean([
            metrics.get('brier_home', 0),
            metrics.get('brier_draw', 0),
            metrics.get('brier_away', 0)
        ])
        
    elif market_type == 'over_under_25':
        # Binary evaluation
        preds = [p.get('over', 0.0) for p in predictions]
        outs = [o.get('over', 0) for o in outcomes]
        
        metrics['brier_score'] = compute_brier_score(preds, outs)
        metrics['log_loss'] = compute_log_loss(preds, outs)
        metrics['ece'] = compute_ece(preds, outs)
        
    elif market_type == 'btts':
        # Binary evaluation
        preds = [p.get('yes', 0.0) for p in predictions]
        outs = [o.get('yes', 0) for o in outcomes]
        
        metrics['brier_score'] = compute_brier_score(preds, outs)
        metrics['log_loss'] = compute_log_loss(preds, outs)
        metrics['ece'] = compute_ece(preds, outs)
    
    metrics['count'] = len(predictions)
    
    return metrics
