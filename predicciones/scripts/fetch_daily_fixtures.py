#!/usr/bin/env python3
"""
Fetch Daily Fixtures Script

Obtains real fixtures for a given date from available data sources.
Uses cascading fallback strategy:
1. Primary APIs (football-data.org, API-Football)
2. ESPN fallback (no API key required)
3. Local existing fixture files

Usage:
    python scripts/fetch_daily_fixtures.py --date 2025-07-15
    python scripts/fetch_daily_fixtures.py --date 20250715

Output:
    - data/fixtures/YYYYMMDD.csv
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from predicciones.src.data.fixture_resolver import FixtureResolver

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
    # Try YYYYMMDD format
    if len(date_str) == 8 and date_str.isdigit():
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}")
    
    # Try YYYY-MM-DD format
    if len(date_str) == 10:
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}")
    
    raise ValueError(f"Unrecognized date format: {date_str}. Use YYYYMMDD or YYYY-MM-DD")


def fetch_fixtures_for_date(date_str: str, leagues: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fetch fixtures for a specific date using cascading fallback strategy.
    
    Args:
        date_str: Date in YYYY-MM-DD or YYYYMMDD format.
        leagues: Optional list of ESPN league slugs to try.
        
    Returns:
        DataFrame with columns: match_id, home_team, away_team, competition, 
                               date, kickoff_datetime, neutral_venue
        Empty DataFrame with valid columns if no fixtures found.
    """
    # Normalize date
    normalized_date = parse_date(date_str)
    logger.info(f"Fetching fixtures for date: {normalized_date}")
    
    # Use the new fixture resolver with cascading fallback
    resolver = FixtureResolver()
    df, source = resolver.resolve_fixtures_for_date(normalized_date, leagues)
    
    if len(df) > 0:
        logger.info(f"Fixtures obtained from source: {source}")
    else:
        logger.warning(f"No fixtures found for date {normalized_date} from any source")
    
    return df


def save_fixtures(df: pd.DataFrame, date_str: str, output_dir: Optional[Path] = None) -> Path:
    """
    Save fixtures DataFrame to CSV file.
    
    Args:
        df: Fixtures DataFrame.
        date_str: Date string (will be normalized to YYYYMMDD for filename).
        output_dir: Output directory. If None, uses project_root/data/fixtures.
    
    Returns:
        Path to saved CSV file.
    """
    if output_dir is None:
        output_dir = project_root / "data" / "fixtures"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize date for filename
    normalized_date = parse_date(date_str)
    date_obj = datetime.strptime(normalized_date, "%Y-%m-%d")
    filename_date = date_obj.strftime("%Y%m%d")
    
    output_path = output_dir / f"{filename_date}.csv"
    
    df.to_csv(output_path, index=False)
    logger.info(f"Fixtures saved to {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Fetch daily football fixtures from API sources"
    )
    parser.add_argument(
        "--date", "-d",
        type=str,
        required=True,
        help="Date in YYYYMMDD or YYYY-MM-DD format (e.g., 2025-07-15 or 20250715)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for fixtures CSV (default: data/fixtures/)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Fetch fixtures
    df = fetch_fixtures_for_date(args.date)
    
    # Save to file
    output_dir = Path(args.output_dir) if args.output_dir else None
    output_path = save_fixtures(df, args.date, output_dir)
    
    # Report results
    if len(df) == 0:
        print(f"\n⚠️  No fixtures found for {args.date}")
        print(f"   Empty CSV created at: {output_path}")
    else:
        print(f"\n✓ Fetched {len(df)} fixtures for {args.date}")
        print(f"   Saved to: {output_path}")
        print(f"\nLeagues included:")
        # Use 'competition' column instead of 'league'
        col_name = 'competition' if 'competition' in df.columns else 'league'
        for league in df[col_name].unique():
            count = len(df[df[col_name] == league])
            print(f"   - {league}: {count} match(es)")


if __name__ == "__main__":
    main()
