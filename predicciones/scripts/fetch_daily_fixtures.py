#!/usr/bin/env python3
"""
Fetch Daily Fixtures Script

Obtains real fixtures for a given date from available data sources.

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
from typing import Optional, List, Dict, Any

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from predicciones.src.data.api_client import FootballAPIClient

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


def fetch_fixtures_for_date(date_str: str, api_client: Optional[FootballAPIClient] = None) -> pd.DataFrame:
    """
    Fetch fixtures for a specific date from available API sources.
    
    Args:
        date_str: Date in YYYY-MM-DD or YYYYMMDD format.
        api_client: Optional FootballAPIClient instance. If None, creates new one.
    
    Returns:
        DataFrame with columns: date, league, home_team, away_team, kickoff_datetime
        Empty DataFrame with valid columns if no fixtures found.
    """
    # Normalize date
    normalized_date = parse_date(date_str)
    logger.info(f"Fetching fixtures for date: {normalized_date}")
    
    # Initialize API client if not provided
    if api_client is None:
        api_client = FootballAPIClient()
    
    fixtures = []
    
    # Strategy 1: Try football-data.org daily matches endpoint
    try:
        fixtures_from_api = _fetch_from_football_data(api_client, normalized_date)
        if fixtures_from_api:
            fixtures.extend(fixtures_from_api)
            logger.info(f"Fetched {len(fixtures_from_api)} fixtures from football-data.org")
    except Exception as e:
        logger.warning(f"football-data.org failed: {e}")
    
    # Strategy 2: Try API-Football fixtures by date
    try:
        fixtures_from_api = _fetch_from_api_football(api_client, normalized_date)
        if fixtures_from_api:
            fixtures.extend(fixtures_from_api)
            logger.info(f"Fetched {len(fixtures_from_api)} fixtures from API-Football")
    except Exception as e:
        logger.warning(f"API-Football failed: {e}")
    
    # Strategy 3: Try open football sources for major leagues
    if not fixtures:
        try:
            fixtures_from_api = _fetch_from_open_sources(normalized_date)
            if fixtures_from_api:
                fixtures.extend(fixtures_from_api)
                logger.info(f"Fetched {len(fixtures_from_api)} fixtures from open sources")
        except Exception as e:
            logger.warning(f"Open sources failed: {e}")
    
    # Create DataFrame
    if fixtures:
        df = pd.DataFrame(fixtures)
        # Ensure column order
        required_cols = ['date', 'league', 'home_team', 'away_team', 'kickoff_datetime']
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        df = df[required_cols]
        logger.info(f"Total fixtures fetched: {len(df)}")
        return df
    else:
        # Return empty DataFrame with valid columns
        logger.warning(f"No fixtures found for date {normalized_date}")
        return pd.DataFrame(columns=['date', 'league', 'home_team', 'away_team', 'kickoff_datetime'])


def _fetch_from_football_data(api_client: FootballAPIClient, date_str: str) -> List[Dict[str, Any]]:
    """
    Fetch fixtures from football-data.org for a specific date.
    
    Uses the /matches endpoint with date range filtering.
    """
    fixtures = []
    
    if not api_client.football_data_token:
        logger.debug("No FOOTBALL_DATA_TOKEN available, skipping football-data.org")
        return []
    
    headers = {"X-Auth-Token": api_client.football_data_token}
    
    # football-data.org allows filtering by date range
    # We'll query matches for the specific date
    url = "https://api.football-data.org/v4/matches"
    params = {
        "dateFrom": date_str,
        "dateTo": date_str,
        "limit": 100
    }
    
    response = api_client._make_request(url, headers, params)
    matches = response.get("matches", [])
    
    for match in matches:
        status = match.get("status", "")
        # Include scheduled and finished matches
        if status in ["SCHEDULED", "FINISHED", "IN_PLAY", "PAUSED"]:
            home_team = match.get("homeTeam", {}).get("name")
            away_team = match.get("awayTeam", {}).get("name")
            competition = match.get("competition", {}).get("name", "Unknown")
            utc_date = match.get("utcDate")
            
            if home_team and away_team and utc_date:
                # Extract kickoff time from utcDate
                try:
                    kickoff_dt = datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
                    kickoff_str = kickoff_dt.strftime("%Y-%m-%d %H:%M")
                except:
                    kickoff_str = utc_date
                
                fixtures.append({
                    "date": date_str,
                    "league": competition,
                    "home_team": home_team,
                    "away_team": away_team,
                    "kickoff_datetime": kickoff_str
                })
    
    return fixtures


def _fetch_from_api_football(api_client: FootballAPIClient, date_str: str) -> List[Dict[str, Any]]:
    """
    Fetch fixtures from API-Football for a specific date.
    """
    fixtures = []
    
    if not api_client.api_football_key:
        logger.debug("No API_FOOTBALL_KEY available, skipping API-Football")
        return []
    
    headers = {"x-apisports-key": api_client.api_football_key}
    url = "https://v3.football.api-sports.io/fixtures"
    params = {"date": date_str}
    
    response = api_client._make_request(url, headers, params)
    fixture_list = response.get("response", [])
    
    for fixture in fixture_list:
        home_team = fixture.get("teams", {}).get("home", {}).get("name")
        away_team = fixture.get("teams", {}).get("away", {}).get("name")
        league = fixture.get("league", {}).get("name", "Unknown")
        fixture_date = fixture.get("fixture", {}).get("date")
        
        if home_team and away_team and fixture_date:
            # Extract kickoff time
            try:
                kickoff_dt = datetime.fromisoformat(fixture_date.replace("Z", "+00:00"))
                kickoff_str = kickoff_dt.strftime("%Y-%m-%d %H:%M")
            except:
                kickoff_str = fixture_date
            
            fixtures.append({
                "date": date_str,
                "league": league,
                "home_team": home_team,
                "away_team": away_team,
                "kickoff_datetime": kickoff_str
            })
    
    return fixtures


def _fetch_from_open_sources(date_str: str) -> List[Dict[str, Any]]:
    """
    Fetch fixtures from open/static sources as fallback.
    
    This uses known league schedules and can be extended with more sources.
    For now, returns empty list as we prefer live API data.
    """
    # In production, this could query additional open sources
    # For now, we rely on the API clients above
    return []


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
        for league in df['league'].unique():
            count = len(df[df['league'] == league])
            print(f"   - {league}: {count} match(es)")


if __name__ == "__main__":
    main()
