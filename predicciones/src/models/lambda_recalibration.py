"""
Lambda recalibration module.
Learns a mapping from raw heuristic lambdas and context features to
actual match goals. Provides a fallback heuristic recalibration
if no trained model is available.

Implements context-aware lambda compression to ensure realistic goal expectations:
- World Cup / international matches: target total lambda ~2.4-2.8
- European top leagues: target total lambda ~2.6-3.0
- Friendly matches: target total lambda ~2.4-2.8
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


# Historical average goals by competition type (based on real data)
HISTORICAL_GOAL_AVERAGES = {
    # FIFA World Cup (recent tournaments): ~2.5-2.7 goals/match
    'world_cup': {'mean': 2.64, 'std': 1.3, 'min': 0.5, 'max': 4.5},
    # UEFA Euro: ~2.4-2.6 goals/match
    'euro': {'mean': 2.52, 'std': 1.2, 'min': 0.5, 'max': 4.0},
    # International friendlies: ~2.4-2.8 goals/match (can be higher due to experimental lineups)
    'friendly': {'mean': 2.65, 'std': 1.4, 'min': 0.5, 'max': 4.5},
    # Premier League: ~2.7-3.0 goals/match (higher scoring)
    'eng.1': {'mean': 2.85, 'std': 1.3, 'min': 0.5, 'max': 4.5},
    # La Liga: ~2.5-2.8 goals/match
    'esp.1': {'mean': 2.65, 'std': 1.2, 'min': 0.5, 'max': 4.0},
    # Serie A: ~2.6-2.9 goals/match
    'ita.1': {'mean': 2.75, 'std': 1.3, 'min': 0.5, 'max': 4.5},
    # Bundesliga: ~2.9-3.2 goals/match (highest scoring top league)
    'ger.1': {'mean': 3.05, 'std': 1.4, 'min': 0.5, 'max': 5.0},
    # Ligue 1: ~2.4-2.7 goals/match
    'fra.1': {'mean': 2.55, 'std': 1.2, 'min': 0.5, 'max': 4.0},
    # MLS: ~2.8-3.2 goals/match (high scoring)
    'usa.1': {'mean': 3.00, 'std': 1.4, 'min': 0.5, 'max': 5.0},
    # Brasileirão: ~2.4-2.7 goals/match
    'bra.1': {'mean': 2.55, 'std': 1.2, 'min': 0.5, 'max': 4.0},
    # Liga Profesional Argentina: ~2.3-2.6 goals/match (lower scoring)
    'arg.1': {'mean': 2.45, 'std': 1.1, 'min': 0.5, 'max': 3.8},
    # Default fallback
    'default': {'mean': 2.70, 'std': 1.3, 'min': 0.5, 'max': 4.5},
}


class LambdaRecalibrator:
    """
    Recalibrates raw expected goals (lambdas) from the base Dixon-Coles model
    to match empirical totals using historical regression and context-aware compression.
    
    Key features:
    - Context-aware lambda compression based on competition type
    - Soft clipping to prevent extreme total lambda values (>4.0-4.5)
    - Preserves relative team strength while adjusting absolute levels
    - Fallback heuristic when no trained model available
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.recal_config = self.config.get('lambda_recalibration', {})
        self.model_home: Optional[Ridge] = None
        self.model_away: Optional[Ridge] = None
        self.is_fitted = False
        
        # Fallback heuristic parameters (conservative scaling)
        self.fallback_a = self.recal_config.get('scale_a', 0.85)  # Reduced from 1.15
        self.fallback_b = self.recal_config.get('scale_b', 0.15)  # Reduced from 0.2
        
        # Context-aware compression settings
        self.compression_config = self.recal_config.get('compression', {})
        
        # Maximum total lambda by context (prevents 6+ goal expectations)
        self.max_total_lambda = {
            'world_cup': self.compression_config.get('world_cup_max', 3.8),
            'friendly': self.compression_config.get('friendly_max', 3.8),
            'league_top': self.compression_config.get('league_top_max', 4.2),
            'default': self.compression_config.get('default_max', 4.0),
        }
        
        # Compression strength (how aggressively to pull towards historical mean)
        self.compression_strength = self.compression_config.get('strength', 0.6)
        
    def fit(self, df: pd.DataFrame) -> 'LambdaRecalibrator':
        """
        Fit the recalibration regression model.
        Requires dataframe with columns:
        'raw_lambda_h', 'raw_lambda_a', 'home_score', 'away_score'
        """
        # Feature matrix: [raw_lambda_h, raw_lambda_a]
        X_h = np.column_stack([
            df['raw_lambda_h'].values,
            df['raw_lambda_a'].values,
        ])
        
        X_a = np.column_stack([
            df['raw_lambda_a'].values,
            df['raw_lambda_h'].values,
        ])

        y_h = df['home_score'].values
        y_a = df['away_score'].values

        self.model_home = Ridge(alpha=1.0)
        self.model_home.fit(X_h, y_h)

        self.model_away = Ridge(alpha=1.0)
        self.model_away.fit(X_a, y_a)

        self.is_fitted = True
        logger.info("LambdaRecalibrator fitted successfully.")
        return self
    
    def _get_context_max_lambda(self, competition_type: str) -> float:
        """Get maximum total lambda for given competition type."""
        if 'world' in competition_type.lower() or 'cup' in competition_type.lower():
            return self.max_total_lambda['world_cup']
        elif 'friendly' in competition_type.lower():
            return self.max_total_lambda['friendly']
        elif any(league in competition_type.lower() for league in ['eng', 'esp', 'ita', 'ger', 'fra']):
            return self.max_total_lambda['league_top']
        else:
            return self.max_total_lambda['default']
    
    def _get_historical_prior(self, competition_slug: Optional[str] = None) -> Dict[str, float]:
        """Get historical goal prior for competition."""
        if not competition_slug:
            return HISTORICAL_GOAL_AVERAGES['default']
        
        slug_lower = competition_slug.lower()
        for key in HISTORICAL_GOAL_AVERAGES:
            if key in slug_lower or slug_lower in key:
                return HISTORICAL_GOAL_AVERAGES[key]
        
        return HISTORICAL_GOAL_AVERAGES['default']
    
    def _compress_lambda(
        self,
        lambda_h: float,
        lambda_a: float,
        competition_type: str = 'default',
        competition_slug: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Apply context-aware compression to lambdas.
        
        This function ensures that:
        1. Total lambda doesn't exceed context-specific maximum
        2. Extreme lambdas are pulled towards historical mean
        3. Relative team strength (ratio) is preserved
        
        Args:
            lambda_h: Raw home lambda
            lambda_a: Raw away lambda
            competition_type: Type of competition ('world_cup', 'friendly', 'league')
            competition_slug: Specific league slug (e.g., 'fifa.world', 'eng.1')
        
        Returns:
            Tuple (compressed_lambda_h, compressed_lambda_a)
        """
        lambda_total = lambda_h + lambda_a
        
        # Get context-specific constraints
        max_total = self._get_context_max_lambda(competition_type)
        hist_prior = self._get_historical_prior(competition_slug)
        hist_mean = hist_prior['mean']
        
        # If total is within reasonable range, apply light adjustment only
        if lambda_total <= max_total:
            # Light compression towards historical mean
            strength = self.compression_strength * 0.5  # Weaker compression when already reasonable
            adjusted_total = lambda_total + strength * (hist_mean - lambda_total)
            scale_factor = adjusted_total / lambda_total if lambda_total > 0 else 1.0
        else:
            # Strong compression needed - use smooth sigmoid-like compression
            # This prevents hard clipping artifacts
            excess = lambda_total - max_total
            compression_factor = max_total / lambda_total
            
            # Smooth transition: blend between no compression and full compression
            # based on how much we exceed the limit
            excess_ratio = excess / max_total
            smooth_factor = 1.0 / (1.0 + excess_ratio)  # Goes from 1 to 0 as excess grows
            
            # Blend: more weight to compression as excess grows
            effective_scale = smooth_factor * 1.0 + (1 - smooth_factor) * compression_factor
            
            # Apply compression and pull towards historical mean
            adjusted_total = lambda_total * effective_scale
            adjusted_total = adjusted_total + self.compression_strength * (hist_mean - adjusted_total)
            scale_factor = adjusted_total / lambda_total if lambda_total > 0 else 1.0
        
        # Scale both lambdas proportionally (preserves ratio)
        lambda_h_adj = lambda_h * scale_factor
        lambda_a_adj = lambda_a * scale_factor
        
        # Ensure minimums
        min_lambda = self.config.get('dixon_coles', {}).get('min_lambda', 0.05)
        lambda_h_adj = max(lambda_h_adj, min_lambda)
        lambda_a_adj = max(lambda_a_adj, min_lambda)
        
        logger.debug(
            f"Lambda compression: ({lambda_h:.3f}, {lambda_a:.3f}) -> "
            f"({lambda_h_adj:.3f}, {lambda_a_adj:.3f}), total: {lambda_total:.3f} -> {lambda_h_adj + lambda_a_adj:.3f}"
        )
        
        return lambda_h_adj, lambda_a_adj

    def recalibrate(
        self,
        lambda_h: float,
        lambda_a: float,
        competition_type: str = 'default',
        competition_slug: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Recalibrate the raw lambdas. If no model is trained, applies
        the fallback soft transformation with context-aware compression.
        
        Args:
            lambda_h: Raw home lambda from Dixon-Coles model
            lambda_a: Raw away lambda from Dixon-Coles model
            competition_type: Type of competition for context-aware adjustment
            competition_slug: Specific league/competition slug
        
        Returns:
            Tuple (recalibrated_lambda_h, recalibrated_lambda_a)
        """
        # Step 1: Apply regression-based recalibration if model is fitted
        if self.is_fitted and self.model_home is not None and self.model_away is not None:
            # Predict using trained regression
            X_h = np.array([[lambda_h, lambda_a]])
            X_a = np.array([[lambda_a, lambda_h]])
            
            adj_h = float(self.model_home.predict(X_h)[0])
            adj_a = float(self.model_away.predict(X_a)[0])
        else:
            # Fallback heuristic scaling (conservative)
            adj_h = (self.fallback_a * lambda_h) + self.fallback_b
            adj_a = (self.fallback_a * lambda_a) + self.fallback_b
        
        # Step 2: Apply context-aware compression
        adj_h, adj_a = self._compress_lambda(
            adj_h, adj_a,
            competition_type=competition_type,
            competition_slug=competition_slug
        )
        
        # Ensure minimums based on config (prevent going negative)
        min_lambda = self.config.get('dixon_coles', {}).get('min_lambda', 0.05)
        adj_h = max(adj_h, min_lambda)
        adj_a = max(adj_a, min_lambda)
        
        return adj_h, adj_a

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model_home': self.model_home,
                'model_away': self.model_away,
                'is_fitted': self.is_fitted,
                'compression_config': self.compression_config,
            }, f)
        logger.debug(f"Saved LambdaRecalibrator to {path}")

    @classmethod
    def load(cls, path: str | Path, config: Optional[Dict[str, Any]] = None) -> 'LambdaRecalibrator':
        path = Path(path)
        cal = cls(config=config)
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
            cal.model_home = data['model_home']
            cal.model_away = data['model_away']
            cal.is_fitted = data['is_fitted']
            if 'compression_config' in data:
                cal.compression_config = data['compression_config']
            logger.info(f"Loaded trained LambdaRecalibrator from {path}")
        else:
            logger.warning(f"No trained LambdaRecalibrator found at {path}, will use fallback.")
        return cal
