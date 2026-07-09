#!/usr/bin/env python3
"""
Daily Pipeline Orchestrator

Orchestrates the complete daily prediction pipeline:
1. Fetch fixtures for a given date
2. Run predictions on those fixtures
3. Generate daily report

Usage:
    python scripts/run_daily_pipeline.py --date 2025-07-15
    python scripts/run_daily_pipeline.py --date 20250715 --verbose

Outputs:
    - data/fixtures/YYYYMMDD.csv
    - output/daily_predictions/YYYYMMDD_predictions.csv
    - output/daily_reports/YYYYMMDD_report.md
    - output/daily_reports/YYYYMMDD_summary.csv
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

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


def run_fetch_fixtures(date_str: str, verbose: bool = False) -> tuple:
    """
    Step 1: Fetch fixtures for the given date.
    
    Args:
        date_str: Date string.
        verbose: Enable verbose logging.
    
    Returns:
        Tuple of (Path to fixtures CSV file, DataFrame with fixtures).
    """
    from scripts.fetch_daily_fixtures import fetch_fixtures_for_date, save_fixtures
    
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching fixtures")
    logger.info("=" * 60)
    
    df = fetch_fixtures_for_date(date_str)
    output_path = save_fixtures(df, date_str)
    
    if len(df) == 0:
        logger.warning(f"No fixtures found for {date_str}")
    else:
        logger.info(f"Fetched {len(df)} fixtures")
    
    return output_path, df


def run_predictions(fixture_path: Path, date_str: str, config: Optional[Dict[str, Any]] = None, fixtures_df=None) -> tuple:
    """
    Step 2: Run predictions on fixtures.
    
    Args:
        fixture_path: Path to fixtures CSV.
        date_str: Date string.
        config: Optional model configuration.
        fixtures_df: Optional DataFrame with fixtures (to check if empty).
    
    Returns:
        Tuple of (Path to predictions CSV file, predictions DataFrame).
    """
    from scripts.run_daily_predictions import DailyPredictionRunner
    
    logger.info("=" * 60)
    logger.info("STEP 2: Running predictions")
    logger.info("=" * 60)
    
    # Check if fixtures are empty
    if fixtures_df is not None and len(fixtures_df) == 0:
        logger.warning("No fixtures to process - skipping predictions")
        # Return empty DataFrame and path
        output_dir = project_root / "output" / "daily_predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        normalized_date = parse_date(date_str)
        date_obj = datetime.strptime(normalized_date, "%Y-%m-%d")
        filename_date = date_obj.strftime("%Y%m%d")
        
        output_path = output_dir / f"{filename_date}_predictions.csv"
        empty_df = pd.DataFrame(columns=['home_team', 'away_team', 'prediction'])
        empty_df.to_csv(output_path, index=False)
        return output_path, empty_df
    
    # Initialize runner
    runner = DailyPredictionRunner(config)
    
    # Run predictions
    predictions_df = runner.run_for_fixture(str(fixture_path))
    
    # Save output
    output_dir = project_root / "output" / "daily_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize date for filename
    normalized_date = parse_date(date_str)
    date_obj = datetime.strptime(normalized_date, "%Y-%m-%d")
    filename_date = date_obj.strftime("%Y%m%d")
    
    output_path = output_dir / f"{filename_date}_predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    
    logger.info(f"Predictions saved to {output_path} ({len(predictions_df)} matches)")
    
    return output_path, predictions_df


def run_generate_report(predictions_path: Path, date_str: str) -> tuple:
    """
    Step 3: Generate daily report.
    
    Args:
        predictions_path: Path to predictions CSV.
        date_str: Date string.
    
    Returns:
        Tuple of (markdown_path, csv_path).
    """
    from scripts.generate_daily_report import generate_daily_report
    
    logger.info("=" * 60)
    logger.info("STEP 3: Generating report")
    logger.info("=" * 60)
    
    md_path, csv_path = generate_daily_report(predictions_path, date_str)
    
    logger.info(f"Report generated: {md_path}")
    logger.info(f"Summary CSV: {csv_path}")
    
    return md_path, csv_path


def run_daily_pipeline(
    date_str: str,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run the complete daily prediction pipeline.
    
    Args:
        date_str: Date in YYYYMMDD or YYYY-MM-DD format.
        config: Optional model configuration.
        verbose: Enable verbose logging.
    
    Returns:
        Dict with paths to all generated files and status information.
        If no fixtures found, returns early with 'status': 'no_fixtures'.
    """
    import pandas as pd
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Normalize date
    normalized_date = parse_date(date_str)
    date_obj = datetime.strptime(normalized_date, "%Y-%m-%d")
    filename_date = date_obj.strftime("%Y%m%d")
    
    logger.info(f"Starting daily pipeline for {normalized_date}")
    logger.info(f"Filename date: {filename_date}")
    
    # Step 1: Fetch fixtures
    fixture_path, fixtures_df = run_fetch_fixtures(date_str, verbose)
    
    # Check if no fixtures were found - abort early
    if fixtures_df is None or len(fixtures_df) == 0:
        logger.warning("=" * 60)
        logger.warning("PIPELINE ABORTED: No fixtures found")
        logger.warning("=" * 60)
        logger.warning("Reason: No API keys configured or no fixtures returned from APIs")
        logger.warning("Solution: Configure FOOTBALL_DATA_TOKEN or API_FOOTBALL_KEY,")
        logger.warning("         or use an existing dated fixture with matches.")
        logger.warning("=" * 60)
        
        return {
            'status': 'no_fixtures',
            'fixtures': fixture_path,
            'predictions': None,
            'report_md': None,
            'report_csv': None,
            'message': 'No fixtures found for selected date'
        }
    
    # Step 2: Run predictions
    predictions_path, predictions_df = run_predictions(fixture_path, date_str, config, fixtures_df)
    
    # Check if predictions are empty
    if predictions_df is None or len(predictions_df) == 0:
        logger.warning("=" * 60)
        logger.warning("PIPELINE COMPLETED WITH NO PREDICTIONS")
        logger.warning("=" * 60)
        logger.warning("Fixtures were found but no predictions could be generated.")
        logger.warning("=" * 60)
        
        return {
            'status': 'no_predictions',
            'fixtures': fixture_path,
            'predictions': predictions_path,
            'report_md': None,
            'report_csv': None,
            'message': 'No predictions generated - fixtures had no processable matches',
            'fixtures_count': len(fixtures_df),
            'predictions_count': 0
        }
    
    # Step 3: Generate report
    md_path, csv_path = run_generate_report(predictions_path, date_str)
    
    # Summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    outputs = {
        'status': 'success',
        'fixtures': fixture_path,
        'predictions': predictions_path,
        'report_md': md_path,
        'report_csv': csv_path,
        'fixtures_count': len(fixtures_df),
        'predictions_count': len(predictions_df)
    }
    
    for name, path in outputs.items():
        if name not in ['status', 'fixtures_count', 'predictions_count', 'message']:
            logger.info(f"  {name}: {path}")
    
    logger.info(f"  fixtures_count: {len(fixtures_df)}")
    logger.info(f"  predictions_count: {len(predictions_df)}")
    
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Run complete daily prediction pipeline"
    )
    parser.add_argument(
        "--date", "-d",
        type=str,
        required=True,
        help="Date in YYYYMMDD or YYYY-MM-DD format (e.g., 2025-07-15 or 20250715)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to JSON config file (optional)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Load custom config if provided
    config = None
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Run pipeline
    outputs = run_daily_pipeline(args.date, config, args.verbose)
    
    # Print summary based on status
    print("\n" + "=" * 60)
    
    status = outputs.get('status', 'unknown')
    
    if status == 'no_fixtures':
        print("PIPELINE ABORTED: No fixtures found")
        print("=" * 60)
        print(f"\n⚠️  {outputs.get('message', 'No fixtures found for selected date')}")
        print(f"\n📋 Fixtures file created (empty): {outputs['fixtures']}")
        print("\nReason: Missing API keys or no fixtures returned from APIs")
        print("\nTo fix this:")
        print("  1. Set FOOTBALL_DATA_TOKEN environment variable, OR")
        print("  2. Set API_FOOTBALL_KEY environment variable, OR")
        print("  3. Use an existing dated fixture file with matches")
        print()
    elif status == 'no_predictions':
        print("PIPELINE COMPLETED WITH NO PREDICTIONS")
        print("=" * 60)
        print(f"\n⚠️  {outputs.get('message', 'No predictions could be generated')}")
        print(f"\n📋 Fixtures found: {outputs.get('fixtures_count', 0)}")
        print(f"📊 Predictions:   {outputs.get('predictions_count', 0)}")
        print(f"📋 Fixtures file: {outputs['fixtures']}")
        print(f"📊 Predictions file (empty): {outputs['predictions']}")
        print()
    else:
        print("DAILY PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"\nGenerated files:")
        print(f"  📋 Fixtures:      {outputs['fixtures']} ({outputs.get('fixtures_count', 0)} matches)")
        print(f"  📊 Predictions:   {outputs['predictions']} ({outputs.get('predictions_count', 0)} matches)")
        print(f"  📝 Report (MD):   {outputs['report_md']}")
        print(f"  📈 Summary (CSV): {outputs['report_csv']}")
        print()


if __name__ == "__main__":
    main()
