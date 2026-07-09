#!/usr/bin/env python3
"""
Lambda Distribution Validation Script

Validates the distribution of lambda values (expected goals) from the corrected pipeline.
This script:
- Takes a large sample of matches (20-50+)
- Runs pre-match predictions with the corrected pipeline
- Computes distribution statistics for lambda_home, lambda_away, lambda_total
- Reports percentiles, means, medians, and threshold exceedances

Outputs:
- output/lambda_validation/lambda_distribution_summary.csv
- output/lambda_validation/high_lambda_matches.csv
- output/lambda_validation/report.md
"""

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from predicciones.src.models.dixon_coles import DixonColesModel
from predicciones.src.features.team_features import extract_team_features
from predicciones.src.ingestion.schemas import TeamData

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class ThresholdConfig:
    """Threshold configuration for lambda warnings."""
    lambda_home_threshold: float = 3.0
    lambda_away_threshold: float = 3.0
    lambda_total_threshold: float = 5.0


DEFAULT_THRESHOLDS = ThresholdConfig()


# -----------------------------------------------------------------------------
# Test Match Generation
# -----------------------------------------------------------------------------

def generate_test_matches(n_matches: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate synthetic test matches with diverse team strengths.
    
    Creates matches between teams with varying attack/defense ratings
    to test the full range of lambda values.
    """
    np.random.seed(seed)
    
    # Load actual ratings from JSON
    ratings_path = project_root / "data" / "ratings_wc2026.json"
    with open(ratings_path, 'r') as f:
        ratings_data = json.load(f)
    
    teams_dict = ratings_data.get('teams', {})
    default_ratings = ratings_data.get('default', {'attack': 1.10, 'defense': 1.00, 'fifa_rank': 100})
    
    # Create team list with their ratings
    team_list = []
    for name, ratings in teams_dict.items():
        team_list.append({
            'name': name,
            'attack': ratings.get('attack', default_ratings['attack']),
            'defense': ratings.get('defense', default_ratings['defense']),
            'fifa_rank': ratings.get('fifa_rank', default_ratings['fifa_rank']),
        })
    
    logger.info(f"Loaded {len(team_list)} teams from ratings file")
    
    matches = []
    for i in range(n_matches):
        # Select two different teams
        home_idx = np.random.randint(0, len(team_list))
        away_idx = np.random.randint(0, len(team_list))
        while away_idx == home_idx:
            away_idx = np.random.randint(0, len(team_list))
        
        home_team = team_list[home_idx]
        away_team = team_list[away_idx]
        
        # Vary context modifiers
        is_neutral = np.random.random() < 0.3  # 30% neutral venue
        importance = np.random.uniform(3.0, 10.0)
        fatigue_home = np.random.uniform(2.0, 8.0)
        fatigue_away = np.random.uniform(2.0, 8.0)
        motivation_home = np.random.uniform(3.0, 10.0)
        motivation_away = np.random.uniform(3.0, 10.0)
        
        matches.append({
            'match_id': i,
            'home_team': home_team,
            'away_team': away_team,
            'is_neutral': is_neutral,
            'importance': importance,
            'fatigue_home': fatigue_home,
            'fatigue_away': fatigue_away,
            'motivation_home': motivation_home,
            'motivation_away': motivation_away,
        })
    
    return matches


def create_team_data(team_info: Dict[str, Any], is_home: bool, match_context: Dict[str, Any]) -> TeamData:
    """
    Create a TeamData object from team info and match context.
    
    This simulates the data structure that would come from the data ingestion pipeline.
    """
    # Build a realistic TeamData structure
    ranking = team_info['fifa_rank']
    attack = team_info['attack']
    defense = team_info['defense']
    
    # Derive historical stats from ratings (approximate)
    # Higher attack rating → more goals scored
    # Higher defense rating → fewer goals conceded
    base_goals = 1.35  # league average
    goals_scored_avg = base_goals * (attack / 1.0) * 0.9 + np.random.uniform(-0.3, 0.3)
    goals_conceded_avg = base_goals * (1.0 / defense) * 0.9 + np.random.uniform(-0.3, 0.3)
    goals_scored_avg = max(0.3, min(3.5, goals_scored_avg))
    goals_conceded_avg = max(0.3, min(3.5, goals_conceded_avg))
    
    # Estimate last 6 matches
    partidos_6 = 6
    goles_marcados_6 = round(goals_scored_avg * partidos_6)
    goles_recibidos_6 = round(goals_conceded_avg * partidos_6)
    
    # Form based on team strength
    win_prob = 0.3 + 0.4 * (attack / 2.0) - 0.2 * (ranking / 100.0)
    win_prob = max(0.1, min(0.8, win_prob))
    
    wins_6 = np.random.binomial(partidos_6, win_prob)
    draws_6 = np.random.binomial(partidos_6 - wins_6, 0.3)
    losses_6 = partidos_6 - wins_6 - draws_6
    
    # H2H (simplified)
    h2h_wins = np.random.randint(0, 4)
    h2h_draws = np.random.randint(0, 3)
    h2h_losses = np.random.randint(0, 4)
    
    # Context values
    fatigue = match_context.get('fatigue_home' if is_home else 'fatigue_away', 5.0)
    motivation = match_context.get('motivation_home' if is_home else 'motivation_away', 5.0)
    importancia = match_context.get('importance', 5.0)
    
    # Build TeamData dict structure
    team_data = {
        'nombre': team_info['name'],
        'CONTEXTO': {
            'ranking_fifa': ranking,
            'goles_marcados_ultimos_6': goles_marcados_6,
            'goles_recibidos_ultimos_6': goles_recibidos_6,
            'partidos_ultimos_6_meses': partidos_6,
            'victorias_ultimos_6': wins_6,
            'empates_ultimos_6': draws_6,
            'derrotas_ultimos_6': losses_6,
            'head_to_head_wins': h2h_wins,
            'head_to_head_draws': h2h_draws,
            'head_to_head_losses': h2h_losses,
            'dias_desde_ultimo_partido': np.random.randint(3, 10),
            'partidos_en_15_dias': np.random.randint(1, 4),
            'xg_promedio_favor': goals_scored_avg * 0.95,  # approximate
            'xg_promedio_contra': goals_conceded_avg * 1.05,  # approximate
        },
        'FACTORES_EXTERNOS': {
            'localía': 0.0 if match_context.get('is_neutral', False) else (1.0 if is_home else 0.0),
            'fatiga_acumulada': fatigue,
            'motivacion': motivation,
            'distancia_viaje_km': np.random.uniform(0, 5000) if not is_home else 0,
            'altitud_m': np.random.uniform(0, 500),
            'importancia_partido': importancia,
            'presion_mediatica': np.random.uniform(3.0, 9.0),
            'temperatura_c': np.random.uniform(15, 32),
            'humedad_pct': np.random.uniform(30, 80),
            'lluvia': np.random.random() < 0.2,
            'viento_kmh': np.random.uniform(0, 25),
        },
        'FACTORES_COLECTIVOS': {
            'eficiencia_finalizacion': 5.0 + (attack - 1.0) * 2.0,
            'creatividad_ofensiva': 5.0 + (attack - 1.0) * 1.5,
            'solidez_defensiva': 5.0 + (defense - 1.0) * 2.0,
            'discipline': 5.0,
            'cohesion_grupal': 5.0,
        },
        'FACTORES_TACTICOS': {
            'pressing_intensidad': 5.0,
            'bloque_defensivo': 5.0,
            'transiciones_rapidas': 5.0,
        },
        'JUGADORES': [],  # Empty squad for simplicity
    }
    
    return TeamData(**team_data)


# -----------------------------------------------------------------------------
# Lambda Analysis
# -----------------------------------------------------------------------------

class LambdaDistributionAnalyzer:
    """Analyzes lambda distributions from model predictions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.output_dir = project_root / "output" / "lambda_validation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize DC model with corrected config
        dc_config = {
            'dixon_coles': {
                'rho': self.config.get('dixon_coles', {}).get('rho', -0.13),
                'home_advantage': self.config.get('dixon_coles', {}).get('home_advantage', 0.25),
                'use_markov_features': False,  # Pre-match prediction
                'lambda_warning_threshold': 3.0,
                'lambda_total_warning_threshold': 5.0,
            }
        }
        self.model = DixonColesModel(dc_config)
        
        # Thresholds from model config
        self.thresholds = ThresholdConfig(
            lambda_home_threshold=self.model.lambda_warning_threshold,
            lambda_away_threshold=self.model.lambda_warning_threshold,
            lambda_total_threshold=self.model.lambda_total_warning_threshold,
        )
    
    def predict_match(self, match: Dict[str, Any]) -> Dict[str, Any]:
        """Run prediction for a single match."""
        home_info = match['home_team']
        away_info = match['away_team']
        context = {
            'is_neutral': match.get('is_neutral', False),
            'importance': match.get('importance', 5.0),
            'fatigue_home': match.get('fatigue_home', 5.0),
            'fatigue_away': match.get('fatigue_away', 5.0),
            'motivation_home': match.get('motivation_home', 5.0),
            'motivation_away': match.get('motivation_away', 5.0),
        }
        
        # Create TeamData objects
        home_data = create_team_data(home_info, is_home=True, match_context=context)
        away_data = create_team_data(away_info, is_home=False, match_context=context)
        
        # Extract features
        home_features = extract_team_features(home_data, is_home=True, opponent_ranking=away_info['fifa_rank'])
        away_features = extract_team_features(away_data, is_home=False, opponent_ranking=home_info['fifa_rank'])
        
        # Predict lambdas
        lambda_home, lambda_away = self.model.predict_lambdas(home_features, away_features)
        lambda_total = lambda_home + lambda_away
        
        return {
            'match_id': match['match_id'],
            'home_team': home_info['name'],
            'away_team': away_info['name'],
            'home_attack': home_info['attack'],
            'home_defense': home_info['defense'],
            'away_attack': away_info['attack'],
            'away_defense': away_info['defense'],
            'home_ranking': home_info['fifa_rank'],
            'away_ranking': away_info['fifa_rank'],
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'lambda_total': lambda_total,
            'is_neutral': context['is_neutral'],
        }
    
    def analyze_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute distribution statistics for lambda values."""
        df = pd.DataFrame(results)
        
        metrics = {}
        
        for col in ['lambda_home', 'lambda_away', 'lambda_total']:
            values = df[col].values
            
            metrics[f'{col}_mean'] = float(np.mean(values))
            metrics[f'{col}_median'] = float(np.median(values))
            metrics[f'{col}_std'] = float(np.std(values))
            metrics[f'{col}_min'] = float(np.min(values))
            metrics[f'{col}_max'] = float(np.max(values))
            
            # Percentiles
            percentiles = np.percentile(values, [10, 25, 50, 75, 90, 95])
            metrics[f'{col}_p10'] = float(percentiles[0])
            metrics[f'{col}_p25'] = float(percentiles[1])
            metrics[f'{col}_p50'] = float(percentiles[2])
            metrics[f'{col}_p75'] = float(percentiles[3])
            metrics[f'{col}_p90'] = float(percentiles[4])
            metrics[f'{col}_p95'] = float(percentiles[5])
        
        # Threshold exceedances
        n_matches = len(df)
        metrics['n_matches'] = n_matches
        
        metrics['n_home_above_threshold'] = int((df['lambda_home'] > self.thresholds.lambda_home_threshold).sum())
        metrics['pct_home_above_threshold'] = metrics['n_home_above_threshold'] / n_matches * 100
        
        metrics['n_away_above_threshold'] = int((df['lambda_away'] > self.thresholds.lambda_away_threshold).sum())
        metrics['pct_away_above_threshold'] = metrics['n_away_above_threshold'] / n_matches * 100
        
        metrics['n_total_above_threshold'] = int((df['lambda_total'] > self.thresholds.lambda_total_threshold).sum())
        metrics['pct_total_above_threshold'] = metrics['n_total_above_threshold'] / n_matches * 100
        
        # Sanity checks
        metrics['lambda_home_in_range'] = float(((df['lambda_home'] >= 0.05) & (df['lambda_home'] <= 4.0)).mean())
        metrics['lambda_away_in_range'] = float(((df['lambda_away'] >= 0.05) & (df['lambda_away'] <= 4.0)).mean())
        metrics['lambda_total_reasonable'] = float((df['lambda_total'] <= 8.0).mean())
        
        return metrics
    
    def get_high_lambda_matches(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract matches with high lambda values."""
        df = pd.DataFrame(results)
        
        high_lambda = df[
            (df['lambda_home'] > self.thresholds.lambda_home_threshold) |
            (df['lambda_away'] > self.thresholds.lambda_away_threshold) |
            (df['lambda_total'] > self.thresholds.lambda_total_threshold)
        ].copy()
        
        high_lambda = high_lambda.sort_values('lambda_total', ascending=False)
        
        return high_lambda
    
    def save_results(self, results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
        """Save analysis results to files."""
        df = pd.DataFrame(results)
        
        # Save distribution summary
        metrics_df = pd.DataFrame([metrics])
        metrics_path = self.output_dir / "lambda_distribution_summary.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved distribution summary to {metrics_path}")
        
        # Save high lambda matches
        high_lambda_df = self.get_high_lambda_matches(results)
        high_lambda_path = self.output_dir / "high_lambda_matches.csv"
        high_lambda_df.to_csv(high_lambda_path, index=False)
        logger.info(f"Saved {len(high_lambda_df)} high-lambda matches to {high_lambda_path}")
    
    def generate_report(self, results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> str:
        """Generate markdown report."""
        lines = []
        lines.append("# Lambda Distribution Validation Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Overview
        lines.append("## 1. Overview")
        lines.append("")
        lines.append(f"- **Matches analyzed:** {metrics['n_matches']}")
        lines.append(f"- **Thresholds:** lambda_home > {self.thresholds.lambda_home_threshold}, lambda_total > {self.thresholds.lambda_total_threshold}")
        lines.append("")
        
        # Distribution Summary
        lines.append("## 2. Lambda Distribution Summary")
        lines.append("")
        lines.append("### 2.1 lambda_home")
        lines.append("")
        lines.append(f"- Mean: {metrics['lambda_home_mean']:.4f}")
        lines.append(f"- Median: {metrics['lambda_home_median']:.4f}")
        lines.append(f"- Std Dev: {metrics['lambda_home_std']:.4f}")
        lines.append(f"- Range: [{metrics['lambda_home_min']:.4f}, {metrics['lambda_home_max']:.4f}]")
        lines.append("")
        lines.append("| P10 | P25 | P50 | P75 | P90 | P95 |")
        lines.append("|-----|-----|-----|-----|-----|-----|")
        lines.append(f"| {metrics['lambda_home_p10']:.4f} | {metrics['lambda_home_p25']:.4f} | {metrics['lambda_home_p50']:.4f} | {metrics['lambda_home_p75']:.4f} | {metrics['lambda_home_p90']:.4f} | {metrics['lambda_home_p95']:.4f} |")
        lines.append("")
        
        lines.append("### 2.2 lambda_away")
        lines.append("")
        lines.append(f"- Mean: {metrics['lambda_away_mean']:.4f}")
        lines.append(f"- Median: {metrics['lambda_away_median']:.4f}")
        lines.append(f"- Std Dev: {metrics['lambda_away_std']:.4f}")
        lines.append(f"- Range: [{metrics['lambda_away_min']:.4f}, {metrics['lambda_away_max']:.4f}]")
        lines.append("")
        lines.append("| P10 | P25 | P50 | P75 | P90 | P95 |")
        lines.append("|-----|-----|-----|-----|-----|-----|")
        lines.append(f"| {metrics['lambda_away_p10']:.4f} | {metrics['lambda_away_p25']:.4f} | {metrics['lambda_away_p50']:.4f} | {metrics['lambda_away_p75']:.4f} | {metrics['lambda_away_p90']:.4f} | {metrics['lambda_away_p95']:.4f} |")
        lines.append("")
        
        lines.append("### 2.3 lambda_total")
        lines.append("")
        lines.append(f"- Mean: {metrics['lambda_total_mean']:.4f}")
        lines.append(f"- Median: {metrics['lambda_total_median']:.4f}")
        lines.append(f"- Std Dev: {metrics['lambda_total_std']:.4f}")
        lines.append(f"- Range: [{metrics['lambda_total_min']:.4f}, {metrics['lambda_total_max']:.4f}]")
        lines.append("")
        lines.append("| P10 | P25 | P50 | P75 | P90 | P95 |")
        lines.append("|-----|-----|-----|-----|-----|-----|")
        lines.append(f"| {metrics['lambda_total_p10']:.4f} | {metrics['lambda_total_p25']:.4f} | {metrics['lambda_total_p50']:.4f} | {metrics['lambda_total_p75']:.4f} | {metrics['lambda_total_p90']:.4f} | {metrics['lambda_total_p95']:.4f} |")
        lines.append("")
        
        # Threshold Exceedances
        lines.append("## 3. Threshold Exceedances")
        lines.append("")
        lines.append("Matches exceeding configured warning thresholds:")
        lines.append("")
        lines.append(f"- **lambda_home > {self.thresholds.lambda_home_threshold}:** {metrics['n_home_above_threshold']} ({metrics['pct_home_above_threshold']:.1f}%)")
        lines.append(f"- **lambda_away > {self.thresholds.lambda_away_threshold}:** {metrics['n_away_above_threshold']} ({metrics['pct_away_above_threshold']:.1f}%)")
        lines.append(f"- **lambda_total > {self.thresholds.lambda_total_threshold}:** {metrics['n_total_above_threshold']} ({metrics['pct_total_above_threshold']:.1f}%)")
        lines.append("")
        
        # Sanity Checks
        lines.append("## 4. Sanity Checks")
        lines.append("")
        lines.append(f"- **lambda_home in [0.05, 4.0]:** {metrics['lambda_home_in_range']*100:.1f}%")
        lines.append(f"- **lambda_away in [0.05, 4.0]:** {metrics['lambda_away_in_range']*100:.1f}%")
        lines.append(f"- **lambda_total <= 8.0:** {metrics['lambda_total_reasonable']*100:.1f}%")
        lines.append("")
        
        # High Lambda Matches
        high_lambda_df = self.get_high_lambda_matches(results)
        lines.append("## 5. High Lambda Matches")
        lines.append("")
        if len(high_lambda_df) > 0:
            lines.append(f"Found {len(high_lambda_df)} matches exceeding thresholds:")
            lines.append("")
            lines.append("| Match | Home | Away | λ_home | λ_away | λ_total |")
            lines.append("|-------|------|------|--------|--------|---------|")
            for _, row in high_lambda_df.head(10).iterrows():
                lines.append(f"| {row['match_id']} | {row['home_team']} | {row['away_team']} | {row['lambda_home']:.3f} | {row['lambda_away']:.3f} | {row['lambda_total']:.3f} |")
            lines.append("")
        else:
            lines.append("No matches exceeded warning thresholds.")
            lines.append("")
        
        # Assessment
        lines.append("## 6. Assessment")
        lines.append("")
        
        # Check if distribution looks reasonable
        assessment_notes = []
        
        # Expected ranges for international football
        expected_total_mean_range = (2.2, 3.2)
        expected_home_range = (0.8, 2.2)
        expected_away_range = (0.6, 1.8)
        
        if expected_total_mean_range[0] <= metrics['lambda_total_mean'] <= expected_total_mean_range[1]:
            assessment_notes.append("✓ lambda_total mean is within expected range for international football")
        else:
            assessment_notes.append(f"⚠ lambda_total mean ({metrics['lambda_total_mean']:.2f}) outside expected range {expected_total_mean_range}")
        
        if expected_home_range[0] <= metrics['lambda_home_mean'] <= expected_home_range[1]:
            assessment_notes.append("✓ lambda_home mean is within expected range")
        else:
            assessment_notes.append(f"⚠ lambda_home mean ({metrics['lambda_home_mean']:.2f}) outside expected range {expected_home_range}")
        
        if expected_away_range[0] <= metrics['lambda_away_mean'] <= expected_away_range[1]:
            assessment_notes.append("✓ lambda_away mean is within expected range")
        else:
            assessment_notes.append(f"⚠ lambda_away mean ({metrics['lambda_away_mean']:.2f}) outside expected range {expected_away_range}")
        
        if metrics['pct_total_above_threshold'] < 10:
            assessment_notes.append(f"✓ Low percentage ({metrics['pct_total_above_threshold']:.1f}%) of extreme lambda_total values")
        else:
            assessment_notes.append(f"⚠ High percentage ({metrics['pct_total_above_threshold']:.1f}%) of extreme lambda_total values")
        
        if metrics['lambda_home_in_range'] > 0.95 and metrics['lambda_away_in_range'] > 0.95:
            assessment_notes.append("✓ Nearly all lambda values within clipping bounds")
        else:
            assessment_notes.append("⚠ Significant portion of lambda values at clipping bounds")
        
        for note in assessment_notes:
            lines.append(note)
        lines.append("")
        
        # Conclusion
        lines.append("## 7. Conclusion")
        lines.append("")
        
        is_operational = (
            expected_total_mean_range[0] <= metrics['lambda_total_mean'] <= expected_total_mean_range[1] and
            metrics['pct_total_above_threshold'] < 15 and
            metrics['lambda_home_in_range'] > 0.90 and
            metrics['lambda_away_in_range'] > 0.90
        )
        
        if is_operational:
            lines.append("**The lambda distribution appears reasonable for operational use.**")
            lines.append("")
            lines.append("Key observations:")
            lines.append(f"- Average total expected goals: {metrics['lambda_total_mean']:.2f} (typical for international football)")
            lines.append(f"- Only {metrics['pct_total_above_threshold']:.1f}% of matches exceed the warning threshold")
            lines.append("- Ratings normalization appears effective")
        else:
            lines.append("**Further calibration may be needed before operational deployment.**")
            lines.append("")
            lines.append("Areas to review:")
            if metrics['lambda_total_mean'] < expected_total_mean_range[0]:
                lines.append("- Lambda values may be too conservative")
            if metrics['lambda_total_mean'] > expected_total_mean_range[1]:
                lines.append("- Lambda values may be too aggressive")
            if metrics['pct_total_above_threshold'] >= 15:
                lines.append("- Too many extreme predictions; consider additional clipping")
        
        lines.append("")
        lines.append("---")
        lines.append("*End of Report*")
        
        return "\n".join(lines)
    
    def run_analysis(self, n_matches: int = 50) -> Dict[str, Any]:
        """Run complete lambda distribution analysis."""
        logger.info("=" * 60)
        logger.info("Lambda Distribution Validation")
        logger.info("=" * 60)
        
        # Generate test matches
        logger.info(f"Generating {n_matches} test matches...")
        matches = generate_test_matches(n_matches=n_matches)
        
        # Run predictions
        logger.info("Running predictions...")
        results = []
        for match in matches:
            result = self.predict_match(match)
            results.append(result)
        
        # Analyze distribution
        logger.info("Analyzing distribution...")
        metrics = self.analyze_distribution(results)
        
        # Save results
        self.save_results(results, metrics)
        
        # Generate report
        report = self.generate_report(results, metrics)
        report_path = self.output_dir / "report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Saved report to {report_path}")
        
        # Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("DISTRIBUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Matches analyzed: {metrics['n_matches']}")
        logger.info(f"lambda_home: mean={metrics['lambda_home_mean']:.4f}, median={metrics['lambda_home_median']:.4f}")
        logger.info(f"lambda_away: mean={metrics['lambda_away_mean']:.4f}, median={metrics['lambda_away_median']:.4f}")
        logger.info(f"lambda_total: mean={metrics['lambda_total_mean']:.4f}, median={metrics['lambda_total_median']:.4f}")
        logger.info(f"% above threshold (total): {metrics['pct_total_above_threshold']:.1f}%")
        logger.info("")
        logger.info(f"Full report saved to: {report_path}")
        logger.info("=" * 60)
        
        return metrics


def main():
    """Main entry point."""
    analyzer = LambdaDistributionAnalyzer()
    analyzer.run_analysis(n_matches=50)


if __name__ == "__main__":
    main()
