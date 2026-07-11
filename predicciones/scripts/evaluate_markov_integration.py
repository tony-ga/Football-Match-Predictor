#!/usr/bin/env python3
"""
Markov Integration A/B Evaluation Script

Compares baseline model (use_markov_features=False) vs Markov-aware model
(use_markov_features=True) across various match states.

Usage:
    python scripts/evaluate_markov_integration.py

Outputs:
    - output/markov_integration_eval/metrics_summary.csv
    - output/markov_integration_eval/state_level_comparison.csv
    - output/markov_integration_eval/report.md
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from predicciones.src.models.dixon_coles import DixonColesModel
from predicciones.src.features.markov_features import (
    load_markov_tables,
    get_markov_features,
    build_state_from_match_context,
    build_state_for_away_team,
    clear_cache,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Test States for Evaluation
# -----------------------------------------------------------------------------

@dataclass
class TestState:
    """Represents a test match state for evaluation."""
    name: str
    minute: int
    score_diff: int  # home - away
    home_red_cards: int = 0
    away_red_cards: int = 0
    phase: str = "regular_time"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'minute': self.minute,
            'score_diff': self.score_diff,
            'home_red_cards': self.home_red_cards,
            'away_red_cards': self.away_red_cards,
            'phase': self.phase
        }


# Define test states covering different scenarios
TEST_STATES = [
    TestState(name="0-15_empate", minute=8, score_diff=0),
    TestState(name="0-15_home_winning", minute=10, score_diff=1),
    TestState(name="0-15_away_winning", minute=12, score_diff=-1),
    
    TestState(name="46-60_empate", minute=55, score_diff=0),
    TestState(name="46-60_losing_by_1", minute=55, score_diff=-1),
    TestState(name="46-60_winning_by_1", minute=55, score_diff=1),
    TestState(name="46-60_losing_by_2", minute=55, score_diff=-2),
    
    TestState(name="76-90+_empate", minute=82, score_diff=0),
    TestState(name="76-90+_losing_by_1", minute=82, score_diff=-1),
    TestState(name="76-90+_winning_by_1", minute=82, score_diff=1),
    
    TestState(name="31-45+_empate", minute=40, score_diff=0),
    TestState(name="61-75_empate", minute=68, score_diff=0),
]


# -----------------------------------------------------------------------------
# Mock Features for Testing
# -----------------------------------------------------------------------------

def create_mock_team_features(team_name: str, is_home: bool = False) -> Dict[str, Any]:
    """Create mock team features for testing purposes."""
    base_features = {
        'nombre': team_name,
        'attack_rating': 1.0,
        'defense_rating': 1.0,
        'form_factor': 1.0,
        'ranking_factor': 1.0,
        'h2h_factor': 1.0,
        'squad_multiplier': 1.0,
        'context_modifier': 0.0,
    }
    
    if is_home:
        base_features['home_advantage_log'] = 0.25  # Typical home advantage
    
    return base_features


# -----------------------------------------------------------------------------
# Evaluation Metrics
# -----------------------------------------------------------------------------

@dataclass
class LambdaComparison:
    """Comparison of lambda predictions between baseline and Markov-aware models."""
    state_name: str
    minute: int
    score_diff: int
    lambda_home_baseline: float
    lambda_away_baseline: float
    lambda_home_markov: float
    lambda_away_markov: float
    delta_home: float
    delta_away: float
    total_goals_baseline: float
    total_goals_markov: float
    delta_total: float
    markov_warning: bool
    fallback_used: bool


@dataclass
class StateMetrics:
    """Metrics aggregated by state type."""
    state_name: str
    count: int
    avg_lambda_home_baseline: float
    avg_lambda_away_baseline: float
    avg_lambda_home_markov: float
    avg_lambda_away_markov: float
    avg_delta_home: float
    avg_delta_away: float
    avg_total_goals_baseline: float
    avg_total_goals_markov: float
    avg_delta_total: float
    std_delta_total: float
    warning_count: int
    fallback_count: int


def compute_brier_score(predictions: List[float], outcomes: List[int]) -> float:
    """Compute Brier score for binary predictions."""
    if len(predictions) != len(outcomes):
        raise ValueError("Predictions and outcomes must have same length")
    return np.mean([(p - o) ** 2 for p, o in zip(predictions, outcomes)])


def compute_log_loss(predictions: List[float], outcomes: List[int], epsilon: float = 1e-15) -> float:
    """Compute log loss for binary predictions."""
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    outcomes = np.array(outcomes)
    return -np.mean(outcomes * np.log(predictions) + (1 - outcomes) * np.log(1 - predictions))


# -----------------------------------------------------------------------------
# Main Evaluation Logic
# -----------------------------------------------------------------------------

class MarkovIntegrationEvaluator:
    """Evaluates the impact of Markov features on goal prediction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.output_dir = project_root / "output" / "markov_integration_eval"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Markov tables
        markov_dir = project_root / "output" / "markov"
        event_probs_path = markov_dir / "state_event_probabilities.csv"
        baselines_path = markov_dir / "baseline_probabilities.csv"
        
        self.event_probs_df, self.baselines = load_markov_tables(
            event_probs_path, baselines_path
        )
        
        logger.info(f"Loaded {len(self.event_probs_df)} Markov states")
        logger.info(f"Baseline global p_goal: {self.baselines['global']['p_goal']:.4f}")
    
    def create_model(self, use_markov: bool = False) -> DixonColesModel:
        """Create a Dixon-Coles model with optional Markov features."""
        config = {
            'dixon_coles': {
                'use_markov_features': use_markov,
                'rho': -0.13,
                'home_advantage': 0.25,
            }
        }
        
        model = DixonColesModel(config)
        
        if use_markov:
            markov_dir = project_root / "output" / "markov"
            model.load_markov_tables(
                str(markov_dir / "state_event_probabilities.csv"),
                str(markov_dir / "baseline_probabilities.csv")
            )
        
        return model
    
    def evaluate_state(
        self,
        state: TestState,
        model_baseline: DixonColesModel,
        model_markov: DixonColesModel,
    ) -> LambdaComparison:
        """Evaluate a single match state with both models."""
        # Create mock team features
        home_features = create_mock_team_features("HomeTeam", is_home=True)
        away_features = create_mock_team_features("AwayTeam", is_home=False)
        
        # Set match state
        match_state = state.to_dict()
        
        # Get baseline predictions
        lambda_h_base, lambda_a_base = model_baseline.predict_lambdas(
            home_features, away_features, match_state
        )
        
        # Get Markov-aware predictions
        lambda_h_markov, lambda_a_markov = model_markov.predict_lambdas(
            home_features, away_features, match_state
        )
        
        # Get Markov metadata for diagnostics
        home_state = build_state_from_match_context(
            minute=state.minute,
            score_diff=state.score_diff,
            home_red_cards=state.home_red_cards,
            away_red_cards=state.away_red_cards,
        )
        markov_features = get_markov_features(
            home_state, self.event_probs_df, self.baselines
        )
        metadata = markov_features.get('_markov_metadata', {})
        
        return LambdaComparison(
            state_name=state.name,
            minute=state.minute,
            score_diff=state.score_diff,
            lambda_home_baseline=round(lambda_h_base, 6),
            lambda_away_baseline=round(lambda_a_base, 6),
            lambda_home_markov=round(lambda_h_markov, 6),
            lambda_away_markov=round(lambda_a_markov, 6),
            delta_home=round(lambda_h_markov - lambda_h_base, 6),
            delta_away=round(lambda_a_markov - lambda_a_base, 6),
            total_goals_baseline=round(lambda_h_base + lambda_a_base, 6),
            total_goals_markov=round(lambda_h_markov + lambda_a_markov, 6),
            delta_total=round((lambda_h_markov + lambda_a_markov) - (lambda_h_base + lambda_a_base), 6),
            markov_warning=metadata.get('warning', False),
            fallback_used=metadata.get('fallback_used', False),
        )
    
    def run_state_level_evaluation(self) -> List[LambdaComparison]:
        """Run evaluation across all test states."""
        logger.info("Running state-level evaluation...")
        
        model_baseline = self.create_model(use_markov=False)
        model_markov = self.create_model(use_markov=True)
        
        results = []
        for state in TEST_STATES:
            comparison = self.evaluate_state(state, model_baseline, model_markov)
            results.append(comparison)
            logger.info(
                f"  {state.name}: baseline={comparison.total_goals_baseline:.3f}, "
                f"markov={comparison.total_goals_markov:.3f}, "
                f"delta={comparison.delta_total:+.3f}"
            )
        
        return results
    
    def aggregate_by_state_type(self, comparisons: List[LambdaComparison]) -> List[StateMetrics]:
        """Aggregate metrics by state type."""
        logger.info("Aggregating metrics by state type...")
        
        # Group by state name pattern (e.g., "0-15_empate", "46-60_losing_by_1")
        groups = {}
        for comp in comparisons:
            key = comp.state_name
            if key not in groups:
                groups[key] = []
            groups[key].append(comp)
        
        metrics = []
        for state_name, comps in groups.items():
            deltas = [c.delta_total for c in comps]
            metrics.append(StateMetrics(
                state_name=state_name,
                count=len(comps),
                avg_lambda_home_baseline=np.mean([c.lambda_home_baseline for c in comps]),
                avg_lambda_away_baseline=np.mean([c.lambda_away_baseline for c in comps]),
                avg_lambda_home_markov=np.mean([c.lambda_home_markov for c in comps]),
                avg_lambda_away_markov=np.mean([c.lambda_away_markov for c in comps]),
                avg_delta_home=np.mean([c.delta_home for c in comps]),
                avg_delta_away=np.mean([c.delta_away for c in comps]),
                avg_total_goals_baseline=np.mean([c.total_goals_baseline for c in comps]),
                avg_total_goals_markov=np.mean([c.total_goals_markov for c in comps]),
                avg_delta_total=np.mean(deltas),
                std_delta_total=np.std(deltas),
                warning_count=sum(1 for c in comps if c.markov_warning),
                fallback_count=sum(1 for c in comps if c.fallback_used),
            ))
        
        return metrics
    
    def generate_synthetic_outcomes(
        self,
        comparisons: List[LambdaComparison],
        seed: int = 42
    ) -> Tuple[List[int], List[int]]:
        """
        Generate synthetic match outcomes for metric computation.
        
        This simulates actual goals based on predicted lambdas using Poisson distribution.
        """
        np.random.seed(seed)
        
        outcomes_home = []
        outcomes_away = []
        
        for comp in comparisons:
            # Simulate goals from Poisson distribution
            goals_home = np.random.poisson(comp.lambda_home_markov)
            goals_away = np.random.poisson(comp.lambda_away_markov)
            
            outcomes_home.append(goals_home)
            outcomes_away.append(goals_away)
        
        return outcomes_home, outcomes_away
    
    def compute_metrics(
        self,
        comparisons: List[LambdaComparison],
        outcomes_home: List[int],
        outcomes_away: List[int],
    ) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        logger.info("Computing evaluation metrics...")
        
        # Convert predictions to probabilities for binary markets
        # Example: P(home team scores >= 1 goal)
        pred_home_scores = []
        pred_away_scores = []
        outcome_home_binary = []
        outcome_away_binary = []
        
        for comp, out_h, out_a in zip(comparisons, outcomes_home, outcomes_away):
            # P(score >= 1) = 1 - P(score = 0) = 1 - exp(-lambda)
            pred_home_scores.append(1 - np.exp(-comp.lambda_home_markov))
            pred_away_scores.append(1 - np.exp(-comp.lambda_away_markov))
            outcome_home_binary.append(1 if out_h >= 1 else 0)
            outcome_away_binary.append(1 if out_a >= 1 else 0)
        
        # Compute Brier scores
        brier_home = compute_brier_score(pred_home_scores, outcome_home_binary)
        brier_away = compute_brier_score(pred_away_scores, outcome_away_binary)
        brier_avg = (brier_home + brier_away) / 2
        
        # Compute log loss
        logloss_home = compute_log_loss(pred_home_scores, outcome_home_binary)
        logloss_away = compute_log_loss(pred_away_scores, outcome_away_binary)
        logloss_avg = (logloss_home + logloss_away) / 2
        
        # Goal prediction error (MAE)
        pred_total_goals = [c.total_goals_markov for c in comparisons]
        actual_total_goals = [h + a for h, a in zip(outcomes_home, outcomes_away)]
        mae_goals = np.mean([abs(p - a) for p, a in zip(pred_total_goals, actual_total_goals)])
        
        # Calibration: compare average predicted vs actual
        avg_pred_home = np.mean([c.lambda_home_markov for c in comparisons])
        avg_actual_home = np.mean(outcomes_home)
        avg_pred_away = np.mean([c.lambda_away_markov for c in comparisons])
        avg_actual_away = np.mean(outcomes_away)
        
        calibration_error_home = abs(avg_pred_home - avg_actual_home)
        calibration_error_away = abs(avg_pred_away - avg_actual_away)
        
        # Markov adjustment statistics
        deltas = [c.delta_total for c in comparisons]
        warnings = sum(1 for c in comparisons if c.markov_warning)
        fallbacks = sum(1 for c in comparisons if c.fallback_used)
        
        return {
            'brier_score_avg': round(brier_avg, 6),
            'brier_score_home': round(brier_home, 6),
            'brier_score_away': round(brier_away, 6),
            'log_loss_avg': round(logloss_avg, 6),
            'log_loss_home': round(logloss_home, 6),
            'log_loss_away': round(logloss_away, 6),
            'mae_goals': round(mae_goals, 6),
            'calibration_error_home': round(calibration_error_home, 6),
            'calibration_error_away': round(calibration_error_away, 6),
            'avg_delta_total': round(np.mean(deltas), 6),
            'std_delta_total': round(np.std(deltas), 6),
            'min_delta_total': round(np.min(deltas), 6),
            'max_delta_total': round(np.max(deltas), 6),
            'warning_count': warnings,
            'fallback_count': fallbacks,
            'total_states': len(comparisons),
        }
    
    def save_results(
        self,
        comparisons: List[LambdaComparison],
        state_metrics: List[StateMetrics],
        overall_metrics: Dict[str, Any],
    ) -> None:
        """Save evaluation results to files."""
        logger.info(f"Saving results to {self.output_dir}...")
        
        # Save state-level comparisons
        comparisons_df = pd.DataFrame([asdict(c) for c in comparisons])
        comparisons_path = self.output_dir / "state_level_comparison.csv"
        comparisons_df.to_csv(comparisons_path, index=False)
        logger.info(f"  Saved {comparisons_path}")
        
        # Save aggregated metrics
        metrics_df = pd.DataFrame([asdict(m) for m in state_metrics])
        metrics_path = self.output_dir / "metrics_summary.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"  Saved {metrics_path}")
        
        # Save overall metrics as JSON
        overall_path = self.output_dir / "overall_metrics.json"
        with open(overall_path, 'w') as f:
            json.dump(overall_metrics, f, indent=2)
        logger.info(f"  Saved {overall_path}")
    
    def generate_report(
        self,
        comparisons: List[LambdaComparison],
        state_metrics: List[StateMetrics],
        overall_metrics: Dict[str, Any],
    ) -> str:
        """Generate Markdown report."""
        report_lines = [
            "# Markov Integration A/B Evaluation Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            "This report evaluates the impact of integrating Markov state features into the",
            "Dixon-Coles goal prediction model. The evaluation compares:",
            "",
            "- **Baseline:** Dixon-Coles model without Markov features",
            "- **Markov-aware:** Dixon-Coles model with `use_markov_features=True`",
            "",
            "## Overall Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total states evaluated | {overall_metrics['total_states']} |",
            f"| Average delta (total goals) | {overall_metrics['avg_delta_total']:+.4f} |",
            f"| Std dev of delta | {overall_metrics['std_delta_total']:.4f} |",
            f"| Min delta | {overall_metrics['min_delta_total']:+.4f} |",
            f"| Max delta | {overall_metrics['max_delta_total']:+.4f} |",
            f"| States with warnings | {overall_metrics['warning_count']} |",
            f"| States using fallback | {overall_metrics['fallback_count']} |",
            "",
            "### Prediction Quality (Synthetic Outcomes)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Brier Score (avg) | {overall_metrics['brier_score_avg']:.4f} |",
            f"| Log Loss (avg) | {overall_metrics['log_loss_avg']:.4f} |",
            f"| MAE (goals) | {overall_metrics['mae_goals']:.4f} |",
            f"| Calibration Error (home) | {overall_metrics['calibration_error_home']:.4f} |",
            f"| Calibration Error (away) | {overall_metrics['calibration_error_away']:.4f} |",
            "",
            "## State-Level Analysis",
            "",
            "### By Match State",
            "",
        ]
        
        # Add state-level table
        report_lines.extend([
            "| State | Minute | Score Diff | Baseline Total | Markov Total | Delta | Warning | Fallback |",
            "|-------|--------|------------|----------------|--------------|-------|---------|----------|"
        ])
        
        for comp in comparisons:
            warning_flag = "⚠️" if comp.markov_warning else ""
            fallback_flag = "🔄" if comp.fallback_used else ""
            report_lines.append(
                f"| {comp.state_name} | {comp.minute} | {comp.score_diff:+d} | "
                f"{comp.total_goals_baseline:.3f} | {comp.total_goals_markov:.3f} | "
                f"{comp.delta_total:+.3f} | {warning_flag} | {fallback_flag} |"
            )
        
        report_lines.extend([
            "",
            "### Aggregated by State Type",
            "",
            "| State | Count | Avg Baseline | Avg Markov | Avg Delta | Std Delta | Warnings | Fallbacks |",
            "|-------|-------|--------------|------------|-----------|-----------|----------|-----------|"
        ])
        
        for m in sorted(state_metrics, key=lambda x: x.state_name):
            report_lines.append(
                f"| {m.state_name} | {m.count} | {m.avg_total_goals_baseline:.3f} | "
                f"{m.avg_total_goals_markov:.3f} | {m.avg_delta_total:+.3f} | "
                f"{m.std_delta_total:.3f} | {m.warning_count} | {m.fallback_count} |"
            )
        
        report_lines.extend([
            "",
            "## Analysis by Context",
            "",
            "### Early Game (0-15 min)",
            "",
        ])
        
        early_states = [c for c in comparisons if c.minute < 16]
        if early_states:
            early_delta = np.mean([c.delta_total for c in early_states])
            report_lines.append(f"- Average delta: {early_delta:+.4f}")
            report_lines.append(f"- Number of states: {len(early_states)}")
        
        report_lines.extend([
            "",
            "### Mid Game (46-60 min)",
            "",
        ])
        
        mid_states = [c for c in comparisons if 46 <= c.minute < 61]
        if mid_states:
            mid_delta = np.mean([c.delta_total for c in mid_states])
            report_lines.append(f"- Average delta: {mid_delta:+.4f}")
            report_lines.append(f"- Number of states: {len(mid_states)}")
        
        report_lines.extend([
            "",
            "### Late Game (76-90+ min)",
            "",
        ])
        
        late_states = [c for c in comparisons if c.minute >= 76]
        if late_states:
            late_delta = np.mean([c.delta_total for c in late_states])
            report_lines.append(f"- Average delta: {late_delta:+.4f}")
            report_lines.append(f"- Number of states: {len(late_states)}")
        
        report_lines.extend([
            "",
            "## Interpretation",
            "",
            "### Markov Adjustment Weight",
            "",
            "The current implementation uses a 30% weight for Markov adjustments:",
            "",
            "```python",
            "adjustment = 1.0 + 0.3 * (ratio - 1.0)",
            "```",
            "",
            "This means:",
            "",
            "- If `markov_p_goal` equals baseline, no adjustment is made",
            "- If `markov_p_goal` is 50% higher than baseline, lambda increases by ~15%",
            "- If `markov_p_goal` is 50% lower than baseline, lambda decreases by ~15%",
            "",
            "### Recommendations",
            "",
        ])
        
        # Generate recommendations based on results
        avg_abs_delta = np.mean([abs(c.delta_total) for c in comparisons])
        max_delta = max([abs(c.delta_total) for c in comparisons])
        
        if avg_abs_delta < 0.05:
            report_lines.append(
                "1. **Weight Assessment:** The average adjustment magnitude is small "
                f"(|{avg_abs_delta:.3f}|). The 30% weight may be appropriate or could potentially "
                "be increased if stronger state-based signals are desired."
            )
        elif avg_abs_delta > 0.2:
            report_lines.append(
                "1. **Weight Assessment:** The average adjustment magnitude is large "
                f"(|{avg_abs_delta:.3f}|). Consider reducing the 30% weight to 15-20% to avoid "
                "overriding the base model too aggressively."
            )
        else:
            report_lines.append(
                "1. **Weight Assessment:** The average adjustment magnitude is moderate "
                f"(|{avg_abs_delta:.3f}|). The 30% weight appears reasonable for balancing "
                "base model and state-based signals."
            )
        
        if overall_metrics['fallback_count'] > len(comparisons) * 0.3:
            report_lines.append(
                "2. **Coverage:** A significant portion of states use fallback baselines. "
                "Consider expanding the Markov training data to cover more states."
            )
        else:
            report_lines.append(
                "2. **Coverage:** Most states have direct Markov estimates (low fallback rate). "
                "The transition matrix provides good coverage."
            )
        
        if overall_metrics['warning_count'] > len(comparisons) * 0.2:
            report_lines.append(
                "3. **Sample Size:** Many states have low sample sizes. The smoothing mechanism "
                "is working correctly, but interpret these adjustments with caution."
            )
        else:
            report_lines.append(
                "3. **Sample Size:** Most states have adequate sample sizes. Adjustments are "
                "based on reliable data."
            )
        
        report_lines.extend([
            "",
            "4. **Feature Selection:** Currently only `markov_p_goal_next_window` is used for",
            "   lambda adjustment. Future work could incorporate:",
            "   - `markov_expected_shots_next_window` for shot-based intensity",
            "   - `markov_expected_corners_next_window` for pressure measurement",
            "   - Combined multi-feature adjustment formula",
            "",
            "5. **Activation Recommendation:** Based on this evaluation:",
            "",
        ])
        
        # Final recommendation
        if avg_abs_delta < 0.03 and overall_metrics['fallback_count'] > 5:
            report_lines.append(
                "   **→ Leave experimental:** The signal is weak and coverage is limited. "
                "Keep the integration for testing but don't enable by default."
            )
        elif avg_abs_delta > 0.15:
            report_lines.append(
                "   **→ Enable with reduced weight:** The signal is strong but may be too "
                "aggressive. Reduce weight from 30% to 15-20% and enable by default."
            )
        else:
            report_lines.append(
                "   **→ Enable by default:** The adjustments are moderate and improve "
                "state-awareness without overwhelming the base model. Consider enabling "
                "`use_markov_features=True` as default for in-play prediction."
            )
        
        report_lines.extend([
            "",
            "## Files Generated",
            "",
            f"- `{self.output_dir / 'metrics_summary.csv'}` - Aggregated metrics by state",
            f"- `{self.output_dir / 'state_level_comparison.csv'}` - Detailed state comparisons",
            f"- `{self.output_dir / 'overall_metrics.json'}` - Overall metrics in JSON format",
            "",
            "---",
            "",
            "*End of Report*",
        ])
        
        return "\n".join(report_lines)
    
    def run_full_evaluation(self) -> None:
        """Run complete evaluation pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Markov Integration A/B Evaluation")
        logger.info("=" * 60)
        
        # Run state-level evaluation
        comparisons = self.run_state_level_evaluation()
        
        # Aggregate by state type
        state_metrics = self.aggregate_by_state_type(comparisons)
        
        # Generate synthetic outcomes for metric computation
        outcomes_home, outcomes_away = self.generate_synthetic_outcomes(comparisons)
        
        # Compute overall metrics
        overall_metrics = self.compute_metrics(comparisons, outcomes_home, outcomes_away)
        
        # Save results
        self.save_results(comparisons, state_metrics, overall_metrics)
        
        # Generate and save report
        report = self.generate_report(comparisons, state_metrics, overall_metrics)
        report_path = self.output_dir / "report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"  Saved {report_path}")
        
        # Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total states evaluated: {overall_metrics['total_states']}")
        logger.info(f"Average delta (total goals): {overall_metrics['avg_delta_total']:+.4f}")
        logger.info(f"Std dev of delta: {overall_metrics['std_delta_total']:.4f}")
        logger.info(f"Brier Score (avg): {overall_metrics['brier_score_avg']:.4f}")
        logger.info(f"Log Loss (avg): {overall_metrics['log_loss_avg']:.4f}")
        logger.info(f"MAE (goals): {overall_metrics['mae_goals']:.4f}")
        logger.info(f"Warnings: {overall_metrics['warning_count']}, Fallbacks: {overall_metrics['fallback_count']}")
        logger.info("")
        logger.info(f"Full report saved to: {report_path}")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    evaluator = MarkovIntegrationEvaluator()
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
