"""
Lambda recalibration module.
Learns a mapping from raw heuristic lambdas and context features to
actual match goals. Provides a fallback heuristic recalibration
if no trained model is available.
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


class LambdaRecalibrator:
    """
    Recalibrates raw expected goals (lambdas) from the base Dixon-Coles model
    to match empirical totals using historical regression.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.recal_config = self.config.get('lambda_recalibration', {})
        self.model_home: Optional[Ridge] = None
        self.model_away: Optional[Ridge] = None
        self.is_fitted = False
        
        # Fallback heuristic parameters
        self.fallback_a = self.recal_config.get('scale_a', 1.15)
        self.fallback_b = self.recal_config.get('scale_b', 0.2)
        
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

    def recalibrate(self, lambda_h: float, lambda_a: float) -> Tuple[float, float]:
        """
        Recalibrate the raw lambdas. If no model is trained, applies
        the fallback soft transformation.
        """
        if self.is_fitted and self.model_home is not None and self.model_away is not None:
            # Predict using trained regression
            X_h = np.array([[lambda_h, lambda_a]])
            X_a = np.array([[lambda_a, lambda_h]])
            
            adj_h = float(self.model_home.predict(X_h)[0])
            adj_a = float(self.model_away.predict(X_a)[0])
        else:
            # Fallback heuristic scaling
            adj_h = (self.fallback_a * lambda_h) + self.fallback_b
            adj_a = (self.fallback_a * lambda_a) + self.fallback_b
            
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
            logger.info(f"Loaded trained LambdaRecalibrator from {path}")
        else:
            logger.warning(f"No trained LambdaRecalibrator found at {path}, will use fallback.")
        return cal
