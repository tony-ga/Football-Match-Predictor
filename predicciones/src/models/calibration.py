"""
Probability calibration module.
Calibrates model outputs (1X2, BTTS, O/U) to ensure probabilities match
real-world frequencies using Isotonic Regression or Platt Scaling.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class MarketCalibrator:
    """
    Calibrates probabilities for a specific market using historical data.
    Supported methods: 'isotonic' or 'platt' (LogisticRegression).
    """

    def __init__(self, market_name: str, method: str = 'isotonic'):
        self.market_name = market_name
        self.method = method.lower()
        if self.method not in ['isotonic', 'platt']:
            raise ValueError(f"Unsupported calibration method: {method}")

        self.models = {}  # Store models per outcome (e.g., 'home', 'draw', 'away')
        self.is_fitted = False

    def fit(self, probs: pd.DataFrame, y_true: pd.DataFrame) -> 'MarketCalibrator':
        """
        Fit calibrators on historical predictions vs actual outcomes.

        Args:
            probs: DataFrame where columns are outcomes (e.g., 'home', 'draw', 'away')
                   and values are uncalibrated probabilities [0..1].
            y_true: DataFrame where columns are the same outcomes
                    and values are binary indicators (1 if occurred, 0 otherwise).
        """
        outcomes = probs.columns.tolist()

        for outcome in outcomes:
            p = probs[outcome].values
            y = y_true[outcome].values

            # Ensure valid probabilities
            p = np.clip(p, 1e-5, 1 - 1e-5)

            if self.method == 'isotonic':
                model = IsotonicRegression(out_of_bounds='clip')
                model.fit(p, y)
            else:  # Platt scaling
                model = LogisticRegression(solver='lbfgs')
                # Logit transform for logistic regression
                logit_p = np.log(p / (1 - p)).reshape(-1, 1)
                model.fit(logit_p, y)

            self.models[outcome] = model

        self.is_fitted = True
        logger.info(f"Fitted {self.method} calibrator for market: {self.market_name}")
        return self

    def calibrate(self, probs: Dict[str, float]) -> Dict[str, float]:
        """
        Calibrate a single prediction.

        Args:
            probs: Dict of uncalibrated probabilities (e.g., {'home': 0.6, 'draw': 0.2, 'away': 0.2})

        Returns:
            Dict of calibrated and re-normalized probabilities.
        """
        if not self.is_fitted:
            logger.warning(f"Calibrator for {self.market_name} is not fitted. Returning uncalibrated.")
            return probs

        calibrated = {}
        for outcome, p in probs.items():
            if outcome not in self.models:
                calibrated[outcome] = p
                continue

            model = self.models[outcome]
            p_val = np.clip(p, 1e-5, 1 - 1e-5)

            if self.method == 'isotonic':
                c_p = model.predict([p_val])[0]
            else:  # Platt
                logit_p = np.log(p_val / (1 - p_val)).reshape(1, -1)
                c_p = model.predict_proba(logit_p)[0, 1]

            calibrated[outcome] = float(np.clip(c_p, 1e-5, 1 - 1e-5))

        # Re-normalize to ensure sum is exactly 1.0 (for mutually exclusive markets like 1X2, BTTS, O/U)
        total = sum(calibrated.values())
        if total > 0:
            for k in calibrated:
                calibrated[k] /= total

        return calibrated

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'market_name': self.market_name,
                'method': self.method,
                'models': self.models,
                'is_fitted': self.is_fitted,
            }, f)
        logger.debug(f"Saved calibrator for {self.market_name} to {path}")

    @classmethod
    def load(cls, path: str | Path) -> 'MarketCalibrator':
        path = Path(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)

        cal = cls(market_name=data['market_name'], method=data['method'])
        cal.models = data['models']
        cal.is_fitted = data['is_fitted']
        return cal


class CalibrationManager:
    """Manages calibrators for all markets."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        cal_config = self.config.get('calibration', {})
        self.method = cal_config.get('method', 'isotonic')

        # Dictionary to hold calibrators: e.g., '1x2', 'btts', 'over_under_25'
        self.calibrators: Dict[str, MarketCalibrator] = {}

    def add_calibrator(self, market_name: str, calibrator: MarketCalibrator) -> None:
        self.calibrators[market_name] = calibrator

    def load_from_config(self, base_dir: str | Path = ".") -> None:
        """Load pre-trained calibrators based on config paths."""
        base_dir = Path(base_dir)
        markets_cfg = self.config.get('calibration', {}).get('markets', [])

        for m in markets_cfg:
            name = m['name']
            filepath = base_dir / m['calibrator_file']
            if filepath.exists():
                self.calibrators[name] = MarketCalibrator.load(filepath)
                logger.info(f"Loaded calibrator for {name}")
            else:
                logger.warning(f"Calibrator file not found: {filepath}")

    def calibrate_markets(self, raw_markets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply calibration to all derived markets if a calibrator exists.
        Markets without a calibrator are passed through unchanged.
        """
        calibrated_markets = raw_markets.copy()

        # 1X2
        if '1x2' in self.calibrators and '1x2' in raw_markets:
            calibrated_markets['1x2'] = self.calibrators['1x2'].calibrate(raw_markets['1x2'])

        # BTTS
        if 'btts' in self.calibrators and 'btts' in raw_markets:
            calibrated_markets['btts'] = self.calibrators['btts'].calibrate(raw_markets['btts'])

        # Over/Under (split into individual lines)
        if 'over_under' in raw_markets:
            ou_raw = raw_markets['over_under']
            ou_calibrated = ou_raw.copy()

            for line in ['15', '25', '35']:
                cal_name = f'over_under_{line}'
                if cal_name in self.calibrators:
                    probs = {'over': ou_raw[f'over_{line}'], 'under': ou_raw[f'under_{line}']}
                    cal_probs = self.calibrators[cal_name].calibrate(probs)
                    ou_calibrated[f'over_{line}'] = cal_probs['over']
                    ou_calibrated[f'under_{line}'] = cal_probs['under']

            calibrated_markets['over_under'] = ou_calibrated

        return calibrated_markets
