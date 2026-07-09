#!/usr/bin/env python3
"""
Generate Daily Report Script

Creates human-readable daily prediction reports in Markdown and CSV summary format.

Usage:
    python scripts/generate_daily_report.py --date 2025-07-15
    python scripts/generate_daily_report.py --predictions output/daily_predictions/20250715_predictions.csv

Outputs:
    - output/daily_reports/YYYYMMDD_report.md
    - output/daily_reports/YYYYMMDD_summary.csv
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> str:
    """
    Parse and normalize date string to YYYY-MM-DD format.
    
    Accepts:
        - YYYYMMDD (e.g., 20250715)
        - YYYY-MM-DD (e.g., 2025-07-15)
    
    Returns:
        Date string in YYYY-MM-DD format.
    """
    if len(date_str) == 8 and date_str.isdigit():
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}")
    
    if len(date_str) == 10:
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}")
    
    raise ValueError(f"Unrecognized date format: {date_str}")


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    """
    Load predictions from CSV file.
    
    Args:
        predictions_path: Path to predictions CSV.
    
    Returns:
        DataFrame with predictions. Empty DataFrame if file is empty.
    """
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    
    try:
        df = pd.read_csv(predictions_path)
    except pd.errors.EmptyDataError:
        logger.warning(f"Predictions file is empty: {predictions_path}")
        return pd.DataFrame()
    
    logger.info(f"Loaded {len(df)} predictions from {predictions_path}")
    return df


def compute_delta_lambda(row: pd.Series) -> float:
    """
    Compute total lambda delta between Markov and baseline.
    
    delta_lambda_total = (lambda_home_markov + lambda_away_markov) - (lambda_home_base + lambda_away_base)
    """
    lambda_home_base = row.get('lambda_home_base', 0) or 0
    lambda_away_base = row.get('lambda_away_base', 0) or 0
    lambda_home_markov = row.get('lambda_home_markov', 0) or 0
    lambda_away_markov = row.get('lambda_away_markov', 0) or 0
    
    return (lambda_home_markov + lambda_away_markov) - (lambda_home_base + lambda_away_base)


def compute_max_confidence_1x2(row: pd.Series) -> float:
    """
    Compute max confidence for 1X2 outcome.
    """
    p_home = row.get('p_home_win_markov', 0) or 0
    p_draw = row.get('p_draw_markov', 0) or 0
    p_away = row.get('p_away_win_markov', 0) or 0
    return max(p_home, p_draw, p_away)


def generate_summary_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate summary CSV with all relevant columns.
    
    Columns:
        - date, league, home_team, away_team
        - lambda_home_base, lambda_away_base
        - lambda_home_markov, lambda_away_markov
        - p_home_win_markov, p_draw_markov, p_away_win_markov
        - p_over_2_5_markov, p_btts_yes_markov
        - delta_lambda_total
        - markov_weight_used
    """
    # Compute delta_lambda_total
    df['delta_lambda_total'] = df.apply(compute_delta_lambda, axis=1)
    
    # Select columns for summary
    summary_cols = [
        'date', 'league', 'home_team', 'away_team',
        'lambda_home_base', 'lambda_away_base',
        'lambda_home_markov', 'lambda_away_markov',
        'p_home_win_markov', 'p_draw_markov', 'p_away_win_markov',
        'p_over_2_5_markov', 'p_btts_yes_markov',
        'delta_lambda_total', 'markov_weight_used'
    ]
    
    # Only include columns that exist
    available_cols = [col for col in summary_cols if col in df.columns]
    
    summary_df = df[available_cols].copy()
    summary_df.to_csv(output_path, index=False)
    logger.info(f"Summary CSV saved to {output_path}")


def generate_markdown_report(df: pd.DataFrame, date_str: str, output_path: Path) -> None:
    """
    Generate Markdown report with:
        a) Resumen general
        b) Top partidos por confianza 1X2
        c) Top partidos por probabilidad de over 2.5
        d) Top partidos por probabilidad de BTTS yes
        e) Partidos donde baseline vs Markov cambian más
    """
    # Build report
    lines = []
    
    # Header
    normalized_date = parse_date(date_str)
    date_obj = datetime.strptime(normalized_date, "%Y-%m-%d")
    formatted_date = date_obj.strftime("%A, %B %d, %Y")
    
    lines.append(f"# Daily Prediction Report")
    lines.append(f"")
    lines.append(f"**Date:** {formatted_date}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"")
    
    # Handle empty DataFrame
    if len(df) == 0:
        lines.append(f"## 📊 Summary Overview")
        lines.append(f"")
        lines.append(f"**No matches found for this date.**")
        lines.append(f"")
        lines.append(f"This could be due to:")
        lines.append(f"- No fixtures scheduled for {formatted_date}")
        lines.append(f"- API sources unavailable (no API keys configured)")
        lines.append(f"- Off-season period")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"*Report generated by Dixon-Coles + Markov prediction pipeline*")
        
        content = "\n".join(lines)
        output_path.write_text(content, encoding='utf-8')
        logger.info(f"Markdown report saved to {output_path}")
        return
    
    # Compute additional metrics
    df['max_confidence_1x2'] = df.apply(compute_max_confidence_1x2, axis=1)
    df['delta_lambda_total'] = df.apply(compute_delta_lambda, axis=1)
    df['delta_lambda_abs'] = df['delta_lambda_total'].abs()
    
    # Get best outcome for each match
    def get_best_outcome(row):
        p_home = row.get('p_home_win_markov', 0) or 0
        p_draw = row.get('p_draw_markov', 0) or 0
        p_away = row.get('p_away_win_markov', 0) or 0
        
        if p_home >= p_draw and p_home >= p_away:
            return 'Home Win', p_home
        elif p_draw >= p_home and p_draw >= p_away:
            return 'Draw', p_draw
        else:
            return 'Away Win', p_away
    
    df[['best_outcome', 'best_outcome_prob']] = df.apply(
        lambda row: pd.Series(get_best_outcome(row)), axis=1
    )
    
    # Section a) Resumen general
    lines.append(f"## 📊 Summary Overview")
    lines.append(f"")
    lines.append(f"- **Total matches processed:** {len(df)}")
    
    leagues = df['league'].unique() if 'league' in df.columns else []
    lines.append(f"- **Leagues included:** {len(leagues)}")
    for league in sorted(leagues):
        count = len(df[df['league'] == league])
        lines.append(f"  - {league}: {count} match(es)")
    
    lines.append(f"")
    
    # Section b) Top partidos por confianza 1X2
    lines.append(f"## 🎯 Top Matches by 1X2 Confidence")
    lines.append(f"")
    lines.append(f"*Ranked by maximum probability among Home Win / Draw / Away Win*")
    lines.append(f"")
    
    top_confidence = df.nlargest(5, 'max_confidence_1x2')
    
    lines.append(f"| Rank | Match | League | Best Outcome | Confidence |")
    lines.append(f"|------|-------|--------|--------------|------------|")
    
    for idx, (_, row) in enumerate(top_confidence.iterrows(), 1):
        match = f"{row['home_team']} vs {row['away_team']}"
        league = row.get('league', 'N/A')
        outcome = row.get('best_outcome', 'N/A')
        conf = row.get('best_outcome_prob', 0)
        lines.append(f"| {idx} | {match} | {league} | {outcome} | {conf:.1%} |")
    
    lines.append(f"")
    
    # Section c) Top partidos por probabilidad de over 2.5
    lines.append(f"## ⚽ Top Matches by Over 2.5 Probability")
    lines.append(f"")
    
    if 'p_over_2_5_markov' in df.columns:
        top_over = df.nlargest(5, 'p_over_2_5_markov')
        
        lines.append(f"| Rank | Match | League | P(Over 2.5) |")
        lines.append(f"|------|-------|--------|-------------|")
        
        for idx, (_, row) in enumerate(top_over.iterrows(), 1):
            match = f"{row['home_team']} vs {row['away_team']}"
            league = row.get('league', 'N/A')
            prob = row.get('p_over_2_5_markov', 0)
            lines.append(f"| {idx} | {match} | {league} | {prob:.1%} |")
    else:
        lines.append(f"*Over 2.5 probabilities not available*")
    
    lines.append(f"")
    
    # Section d) Top partidos por probabilidad de BTTS yes
    lines.append(f"## 🔄 Top Matches by BTTS Yes Probability")
    lines.append(f"")
    
    if 'p_btts_yes_markov' in df.columns:
        top_btts = df.nlargest(5, 'p_btts_yes_markov')
        
        lines.append(f"| Rank | Match | League | P(BTTS Yes) |")
        lines.append(f"|------|-------|--------|-------------|")
        
        for idx, (_, row) in enumerate(top_btts.iterrows(), 1):
            match = f"{row['home_team']} vs {row['away_team']}"
            league = row.get('league', 'N/A')
            prob = row.get('p_btts_yes_markov', 0)
            lines.append(f"| {idx} | {match} | {league} | {prob:.1%} |")
    else:
        lines.append(f"*BTTS probabilities not available*")
    
    lines.append(f"")
    
    # Section e) Partidos donde baseline vs Markov cambian más
    lines.append(f"## 📈 Largest Baseline vs Markov Lambda Changes")
    lines.append(f"")
    lines.append(f"*Ranked by absolute change in total expected goals (λ_home + λ_away)*")
    lines.append(f"")
    
    top_delta = df.nlargest(5, 'delta_lambda_abs')
    
    lines.append(f"| Rank | Match | League | Δλ Total | Baseline λ | Markov λ | Direction |")
    lines.append(f"|------|-------|--------|----------|------------|----------|-----------|")
    
    for idx, (_, row) in enumerate(top_delta.iterrows(), 1):
        match = f"{row['home_team']} vs {row['away_team']}"
        league = row.get('league', 'N/A')
        delta = row.get('delta_lambda_total', 0)
        
        lambda_home_base = row.get('lambda_home_base', 0) or 0
        lambda_away_base = row.get('lambda_away_base', 0) or 0
        lambda_home_markov = row.get('lambda_home_markov', 0) or 0
        lambda_away_markov = row.get('lambda_away_markov', 0) or 0
        
        baseline_total = lambda_home_base + lambda_away_base
        markov_total = lambda_home_markov + lambda_away_markov
        
        direction = "↑ Increase" if delta > 0 else ("↓ Decrease" if delta < 0 else "→ No change")
        
        lines.append(f"| {idx} | {match} | {league} | {delta:+.3f} | {baseline_total:.3f} | {markov_total:.3f} | {direction} |")
    
    lines.append(f"")
    
    # Footer
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"*Report generated by Dixon-Coles + Markov prediction pipeline*")
    
    # Write file
    content = "\n".join(lines)
    output_path.write_text(content, encoding='utf-8')
    logger.info(f"Markdown report saved to {output_path}")


def generate_daily_report(
    predictions_path: Path,
    date_str: str,
    output_dir: Optional[Path] = None
) -> tuple:
    """
    Generate both Markdown report and summary CSV.
    
    Args:
        predictions_path: Path to predictions CSV.
        date_str: Date string for report.
        output_dir: Output directory. If None, uses project_root/output/daily_reports.
    
    Returns:
        Tuple of (markdown_path, csv_path).
    """
    # Load predictions
    df = load_predictions(predictions_path)
    
    # Setup output directory
    if output_dir is None:
        output_dir = project_root / "output" / "daily_reports"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize date for filenames
    normalized_date = parse_date(date_str)
    date_obj = datetime.strptime(normalized_date, "%Y-%m-%d")
    filename_date = date_obj.strftime("%Y%m%d")
    
    # Generate outputs
    md_path = output_dir / f"{filename_date}_report.md"
    csv_path = output_dir / f"{filename_date}_summary.csv"
    
    generate_markdown_report(df, date_str, md_path)
    generate_summary_csv(df, csv_path)
    
    return md_path, csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate daily prediction report in Markdown and CSV formats"
    )
    parser.add_argument(
        "--date", "-d",
        type=str,
        help="Date in YYYYMMDD or YYYY-MM-DD format (used to find predictions file)"
    )
    parser.add_argument(
        "--predictions", "-p",
        type=str,
        help="Direct path to predictions CSV file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for reports (default: output/daily_reports/)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine predictions file path
    if args.predictions:
        predictions_path = Path(args.predictions)
    elif args.date:
        # Construct path from date
        normalized_date = parse_date(args.date)
        date_obj = datetime.strptime(normalized_date, "%Y-%m-%d")
        filename_date = date_obj.strftime("%Y%m%d")
        predictions_path = project_root / "output" / "daily_predictions" / f"{filename_date}_predictions.csv"
    else:
        parser.error("Either --date or --predictions must be provided")
    
    # Generate report
    md_path, csv_path = generate_daily_report(
        predictions_path,
        args.date or args.predictions,
        Path(args.output_dir) if args.output_dir else None
    )
    
    print(f"\n✓ Report generated successfully!")
    print(f"   Markdown: {md_path}")
    print(f"   Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
