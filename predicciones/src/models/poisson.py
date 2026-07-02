"""
Bivariate independent Poisson model for football match score prediction.

This is the simpler baseline model. Expected goals are estimated externally
and passed in; this module handles the score matrix generation.

For the main production model, see dixon_coles.py.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy.stats import poisson

logger = logging.getLogger(__name__)


def poisson_score_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 8,
) -> np.ndarray:
    """
    Generate a score probability matrix using independent Poisson distributions.

    P(home=i, away=j) = Poisson(i; lambda_home) * Poisson(j; lambda_away)

    Args:
        lambda_home: Expected goals for home team.
        lambda_away: Expected goals for away team.
        max_goals: Maximum goals per team to consider (matrix is (max+1) x (max+1)).

    Returns:
        2D numpy array of shape (max_goals+1, max_goals+1) where
        matrix[i][j] = P(home scores i, away scores j).
        Matrix sums to ~1.0 (small truncation error for high lambdas).
    """
    if lambda_home <= 0 or lambda_away <= 0:
        raise ValueError(
            f"Lambdas must be positive. Got lambda_home={lambda_home}, "
            f"lambda_away={lambda_away}"
        )

    goals_range = np.arange(max_goals + 1)

    # P(home = i) for i in [0, max_goals]
    p_home = poisson.pmf(goals_range, lambda_home)
    # P(away = j) for j in [0, max_goals]
    p_away = poisson.pmf(goals_range, lambda_away)

    # Outer product: matrix[i, j] = p_home[i] * p_away[j]
    matrix = np.outer(p_home, p_away)

    # Normalize to account for truncation
    total = matrix.sum()
    if total > 0:
        matrix = matrix / total

    logger.debug(
        f"Poisson matrix: lambda_h={lambda_home:.3f}, lambda_a={lambda_away:.3f}, "
        f"sum={matrix.sum():.6f}"
    )
    return matrix


def poisson_1x2(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 8,
) -> Tuple[float, float, float]:
    """
    Compute 1X2 probabilities from a Poisson score matrix.

    Returns:
        Tuple of (P_home_win, P_draw, P_away_win).
    """
    matrix = poisson_score_matrix(lambda_home, lambda_away, max_goals)

    p_home = float(np.sum(np.tril(matrix, k=-1)))   # i > j
    p_draw = float(np.sum(np.diag(matrix)))           # i == j
    p_away = float(np.sum(np.triu(matrix, k=1)))      # i < j

    # Renormalize
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    return p_home, p_draw, p_away


def expected_goals_from_1x2(
    p_home: float,
    p_draw: float,
    p_away: float,
    total_goals_prior: float = 2.5,
) -> Tuple[float, float]:
    """
    Rough inverse: estimate lambda_home and lambda_away from 1X2 probs.
    Useful for sanity checking or initializing optimization.

    Uses a simple heuristic based on odds-ratio and total goals prior.
    Not a replacement for proper MLE.
    """
    # Strength ratio from win probabilities
    if p_away <= 0:
        p_away = 0.01
    ratio = p_home / p_away  # home strength relative to away

    # Estimate lambdas such that lambda_h / lambda_a = ratio
    # and lambda_h + lambda_a ~ total_goals_prior
    lambda_a = total_goals_prior / (1 + np.sqrt(ratio))
    lambda_h = total_goals_prior - lambda_a

    return float(max(lambda_h, 0.1)), float(max(lambda_a, 0.1))
