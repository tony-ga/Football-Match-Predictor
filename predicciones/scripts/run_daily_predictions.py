#!/usr/bin/env python3
"""
Daily Predictions Script for Dixon-Coles + Markov Model

Generates match outcome predictions in CSV format for a given fixture file.

Usage:
    python scripts/run_daily_predictions.py --date 20250710
    python scripts/run_daily_predictions.py --fixture data/fixtures/20250710.csv

Outputs:
    - output/daily_predictions/YYYYMMDD_predictions.csv
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from predicciones.src.models.dixon_coles import DixonColesModel
from predicciones.src.eval.probability_calibration import compute_match_probabilities
from predicciones.src.features.markov_features import build_state_from_match_context
from predicciones.src.data.team_ratings_loader import TeamRatingsLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Default Configuration
# -----------------------------------------------------------------------------

DEFAULT_CONFIG = {
    'dixon_coles': {
        'use_markov_features': True,
        'markov_weight_schedule': {
            'early': 0.20,  # 0-30 min
            'mid': 0.15,    # 31-75 min
            'late': 0.10,   # 76+ min
        },
        'rho': -0.13,
        'home_advantage': 0.25,
    },
    'matrix': {
        'max_goals': 8,
    }
}


# -----------------------------------------------------------------------------
# Daily Prediction Runner
# -----------------------------------------------------------------------------

class DailyPredictionRunner:
    """
    Orchestrates daily prediction generation using Dixon-Coles + Markov model.
    
    Usage:
        runner = DailyPredictionRunner(config)
        df = runner.run_for_fixture(fixture_path)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, verbose: bool = False):
        """
        Initialize the prediction runner.
        
        Args:
            config: Model configuration dict. If None, uses DEFAULT_CONFIG.
            verbose: Enable detailed logging for ratings lookup
        """
        self.config = config or DEFAULT_CONFIG.copy()
        self.verbose = verbose
        
        # Initialize model
        self.model = DixonColesModel(self.config)
        
        # Initialize ratings loader
        self.ratings_loader = TeamRatingsLoader()
        
        # Load Markov tables if available
        markov_dir = project_root / "output" / "markov"
        event_probs_path = markov_dir / "state_event_probabilities.csv"
        baselines_path = markov_dir / "baseline_probabilities.csv"
        
        if event_probs_path.exists() and baselines_path.exists():
            self.model.load_markov_tables(
                str(event_probs_path),
                str(baselines_path)
            )
            logger.info("Markov tables loaded successfully")
        else:
            logger.warning(
                f"Markov tables not found at {markov_dir}. "
                "Running in baseline mode only."
            )
        
        # Store effective markov weight for reporting
        dc_config = self.config.get('dixon_coles', {})
        if dc_config.get('markov_weight_schedule'):
            schedule = dc_config['markov_weight_schedule']
            self.markov_weight_description = (
                f"piecewise: early={schedule.get('early', 0.20):.2f}, "
                f"mid={schedule.get('mid', 0.15):.2f}, "
                f"late={schedule.get('late', 0.10):.2f}"
            )
        else:
            weight = dc_config.get('markov_weight', 0.18)
            self.markov_weight_description = f"constant={weight:.2f}"
    
    def _build_team_features(self, team_name: str, is_home: bool) -> Dict[str, Any]:
        """
        Build feature dict for a team using real ratings from ratings_wc2026.json.
        
        Uses TeamRatingsLoader to fetch attack/defense ratings based on team name.
        Falls back to neutral defaults only if team is not found in ratings.
        
        Args:
            team_name: Team name from fixture
            is_home: Whether team is playing at home
            
        Returns:
            Feature dict compatible with DixonColesModel.predict_lambdas()
        """
        home_advantage = self.config.get('dixon_coles', {}).get('home_advantage', 0.25)
        
        features, rating_info = self.ratings_loader.build_team_features(
            team_name=team_name,
            is_home=is_home,
            home_advantage_log=home_advantage if is_home else 0.0,
            verbose=self.verbose
        )
        
        # Log detailed diagnostic info in verbose mode
        if self.verbose:
            logger.info(
                f"[RATINGS] {team_name}: "
                f"matched='{rating_info['matched_team_name']}', "
                f"attack={features['attack_rating']:.3f}, "
                f"defense={features['defense_rating']:.3f}, "
                f"rank={rating_info['fifa_rank']}, "
                f"ranking_factor={rating_info['ranking_factor']:.3f}, "
                f"fallback={rating_info['used_default_fallback']}, "
                f"source={rating_info['ratings_source']}"
            )
        
        return features
    
    def _compute_prediction(
        self,
        home_team: str,
        away_team: str,
        match_date: str,
        league: str,
    ) -> Dict[str, Any]:
        """
        Compute prediction for a single match.
        
        Returns:
            Dict with all prediction outputs.
        """
        # Build team features
        home_features = self._build_team_features(home_team, is_home=True)
        away_features = self._build_team_features(away_team, is_home=False)
        
        # Set pre-kickoff state (minute=0, score_diff=0, no cards)
        self.model.set_match_state(
            minute=0,
            score_diff=0,
            home_red_cards=0,
            away_red_cards=0,
            phase="regular_time"
        )
        
        # Compute baseline lambdas (without Markov)
        # Temporarily disable Markov to get baseline
        original_use_markov = self.model.use_markov_features
        self.model.use_markov_features = False
        
        lambda_home_base, lambda_away_base = self.model.predict_lambdas(
            home_features, away_features,
            match_state=self.model.match_state
        )
        
        # Compute Markov-aware lambdas
        self.model.use_markov_features = original_use_markov
        
        if self.model.use_markov_features and self.model.markov_event_probs is not None:
            lambda_home_markov, lambda_away_markov = self.model.predict_lambdas(
                home_features, away_features,
                match_state=self.model.match_state
            )
        else:
            # Fallback to baseline if Markov not available
            lambda_home_markov = lambda_home_base
            lambda_away_markov = lambda_away_base
        
        # Compute probabilities from Markov-aware lambdas
        max_goals = self.config.get('matrix', {}).get('max_goals', 8)
        rho = self.config.get('dixon_coles', {}).get('rho', -0.13)
        
        probs = compute_match_probabilities(
            lambda_h=lambda_home_markov,
            lambda_a=lambda_away_markov,
            rho=rho,
            max_goals=max_goals
        )
        
        # Build result dict
        result = {
            'date': match_date,
            'league': league,
            'home_team': home_team,
            'away_team': away_team,
            
            # Baseline lambdas
            'lambda_home_base': round(lambda_home_base, 4),
            'lambda_away_base': round(lambda_away_base, 4),
            
            # Markov-aware lambdas
            'lambda_home_markov': round(lambda_home_markov, 4),
            'lambda_away_markov': round(lambda_away_markov, 4),
            
            # 1X2 probabilities (Markov-aware)
            'p_home_win_markov': round(probs['p_home_win'], 4),
            'p_draw_markov': round(probs['p_draw'], 4),
            'p_away_win_markov': round(probs['p_away_win'], 4),
            
            # Over/Under 2.5 (Markov-aware)
            'p_over_2_5_markov': round(probs['p_over_2_5'], 4),
            'p_under_2_5_markov': round(probs['p_under_2_5'], 4),
            
            # BTTS (Markov-aware)
            'p_btts_yes_markov': round(probs['p_btts_yes'], 4),
            'p_btts_no_markov': round(probs['p_btts_no'], 4),
            
            # Additional O/U lines
            'p_over_1_5_markov': round(probs['p_over_1_5'], 4),
            'p_under_1_5_markov': round(probs['p_under_1_5'], 4),
            'p_over_3_5_markov': round(probs['p_over_3_5'], 4),
            'p_under_3_5_markov': round(probs['p_under_3_5'], 4),
            
            # Metadata
            'markov_weight_used': self.markov_weight_description,
            'markov_state_description': 'minute_bucket=0-15, score_diff_bucket=0, home_red_cards=0, away_red_cards=0',
        }
        
        return result
    
    def run_for_fixture(
        self,
        fixture_path: str,
    ) -> pd.DataFrame:
        """
        Generate predictions for all matches in a fixture file.
        
        Args:
            fixture_path: Path to CSV file with columns:
                          date, league, home_team, away_team[, kickoff_datetime]
        
        Returns:
            DataFrame with predictions for all matches.
        """
        fixture_path = Path(fixture_path)
        
        if not fixture_path.exists():
            raise FileNotFoundError(f"Fixture file not found: {fixture_path}")
        
        logger.info(f"Loading fixture from {fixture_path}")
        
        # Load fixture data
        df = pd.read_csv(fixture_path)
        
        # Validate required columns
        # Support both 'league' and 'competition' column names
        required_cols = ['date', 'home_team', 'away_team']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for league/competition column
        if 'league' not in df.columns and 'competition' not in df.columns:
            raise ValueError("Missing required column: 'league' or 'competition'")
        
        # Normalize column name to 'league' for downstream processing
        if 'competition' in df.columns and 'league' not in df.columns:
            df = df.rename(columns={'competition': 'league'})
        
        # Log ratings summary at start
        if self.verbose:
            ratings_summary = self.ratings_loader.get_ratings_summary()
            logger.info(
                f"Ratings loaded: {ratings_summary['total_teams']} teams, "
                f"avg_attack={ratings_summary['avg_attack']:.3f}, "
                f"avg_defense={ratings_summary['avg_defense']:.3f}"
            )
        
        # Generate predictions for each match
        predictions = []
        for idx, row in df.iterrows():
            try:
                pred = self._compute_prediction(
                    home_team=row['home_team'],
                    away_team=row['away_team'],
                    match_date=row['date'],
                    league=row['league'],
                )
                predictions.append(pred)
                logger.info(
                    f"Predicted: {row['home_team']} vs {row['away_team']} "
                    f"(H: {pred['lambda_home_markov']:.2f}, A: {pred['lambda_away_markov']:.2f})"
                )
            except Exception as e:
                logger.error(f"Error predicting match {idx}: {e}")
                # Add partial result with error info
                predictions.append({
                    'date': row['date'],
                    'league': row['league'],
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'error': str(e),
                })
        
        # Convert to DataFrame
        result_df = pd.DataFrame(predictions)
        
        return result_df


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def generate_sample_fixture(date_str: str, output_path: Path) -> Path:
    """
    Generate a sample fixture file for testing.
    
    Args:
        date_str: Date string in YYYYMMDD format.
        output_path: Directory to save fixture file.
    
    Returns:
        Path to generated fixture file.
    """
    # Parse date
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    date_formatted = date_obj.strftime("%Y-%m-%d")
    
    # Sample matches (example data)
    sample_data = {
        'date': [date_formatted] * 5,
        'league': [
            'Premier League',
            'La Liga',
            'Serie A',
            'Bundesliga',
            'Ligue 1'
        ],
        'home_team': [
            'Manchester City',
            'Real Madrid',
            'Inter Milan',
            'Bayern Munich',
            'PSG'
        ],
        'away_team': [
            'Liverpool',
            'Barcelona',
            'AC Milan',
            'Borussia Dortmund',
            'Marseille'
        ],
        'kickoff_datetime': [
            f"{date_formatted} 15:00",
            f"{date_formatted} 17:30",
            f"{date_formatted} 19:45",
            f"{date_formatted} 17:30",
            f"{date_formatted} 20:00"
        ]
    }
    
    fixture_df = pd.DataFrame(sample_data)
    fixture_path = output_path / f"{date_str}.csv"
    fixture_df.to_csv(fixture_path, index=False)
    
    logger.info(f"Generated sample fixture: {fixture_path}")
    return fixture_path


def run_for_date(
    date_str: str,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Convenience function to run predictions for a specific date.
    
    Creates/loads fixture file and generates predictions.
    
    Args:
        date_str: Date in YYYYMMDD format.
        config: Optional model configuration.
        verbose: Enable detailed logging for ratings lookup
    
    Returns:
        DataFrame with predictions.
    """
    # Ensure fixtures directory exists
    fixtures_dir = project_root / "data" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    
    fixture_path = fixtures_dir / f"{date_str}.csv"
    
    # Generate sample fixture if it doesn't exist
    if not fixture_path.exists():
        logger.warning(f"Fixture not found for {date_str}, generating sample...")
        generate_sample_fixture(date_str, fixtures_dir)
    
    # Run predictions
    runner = DailyPredictionRunner(config, verbose=verbose)
    predictions_df = runner.run_for_fixture(str(fixture_path))
    
    # Save output
    output_dir = project_root / "output" / "daily_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{date_str}_predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    
    logger.info(f"Predictions saved to {output_path}")
    
    return predictions_df


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate daily football match predictions using Dixon-Coles + Markov model"
    )
    parser.add_argument(
        "--date", "-d",
        type=str,
        help="Date in YYYYMMDD format (e.g., 20250710)"
    )
    parser.add_argument(
        "--fixture", "-f",
        type=str,
        help="Path to fixture CSV file"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to JSON config file (optional)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=str(project_root / "output" / "daily_predictions"),
        help="Output directory for predictions"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load custom config if provided
    config = DEFAULT_CONFIG.copy()
    if args.config:
        import json
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            # Deep merge
            for key, value in custom_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
    
    # Determine input source
    if args.date:
        # Run for specific date
        predictions_df = run_for_date(args.date, config, verbose=args.verbose)
    elif args.fixture:
        # Run for specific fixture file
        runner = DailyPredictionRunner(config, verbose=args.verbose)
        predictions_df = runner.run_for_fixture(args.fixture)
        
        # Extract date from fixture filename for output
        fixture_name = Path(args.fixture).stem
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{fixture_name}_predictions.csv"
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
    else:
        parser.error("Either --date or --fixture must be provided")
        return
    
    # Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"Total matches: {len(predictions_df)}")
    
    if 'p_home_win_markov' in predictions_df.columns:
        print(f"\nAverage probabilities:")
        print(f"  Home win: {predictions_df['p_home_win_markov'].mean():.2%}")
        print(f"  Draw:     {predictions_df['p_draw_markov'].mean():.2%}")
        print(f"  Away win: {predictions_df['p_away_win_markov'].mean():.2%}")
        print(f"  Over 2.5: {predictions_df['p_over_2_5_markov'].mean():.2%}")
        print(f"  BTTS Yes: {predictions_df['p_btts_yes_markov'].mean():.2%}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
