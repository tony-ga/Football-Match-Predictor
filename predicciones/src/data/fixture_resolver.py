#!/usr/bin/env python3
"""
Fixture Resolver with Cascading Fallback Strategy

This module implements a multi-source fixture resolution strategy:
1. Primary API sources (football-data.org, API-Football)
2. ESPN fallback (no API key required)
3. Local existing fixture files

Usage:
    from predicciones.src.data.fixture_resolver import resolve_fixtures_for_date
    
    fixtures_df = resolve_fixtures_for_date("2025-07-15")
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import sys

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from predicciones.src.data.api_client import FootballAPIClient
from predicciones.src.data.espn_client import EspnWorldCupClient

logger = logging.getLogger(__name__)

# ESPN soccer league slugs for different competitions
ESPN_SOCCER_LEAGUES = {
    "fifa.world": "FIFA World Cup",
    "uefa.champions": "UEFA Champions League",
    "eng.1": "Premier League",
    "esp.1": "La Liga",
    "ita.1": "Serie A",
    "ger.1": "Bundesliga",
    "fra.1": "Ligue 1",
    "usa.1": "MLS",
    "bra.1": "Brasileirão",
    "arg.1": "Liga Profesional Argentina",
}


class FixtureResolver:
    """
    Resolves fixtures for a given date using cascading fallback strategy.
    
    Order of sources:
    1. Primary APIs (football-data.org, API-Football) - requires API keys
    2. ESPN fallback - no API key required
    3. Local existing fixture files
    """
    
    def __init__(self):
        self.api_client = FootballAPIClient()
        self.espn_client = EspnWorldCupClient()
        
    def resolve_fixtures_for_date(
        self, 
        date_str: str, 
        leagues: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, str]:
        """
        Resolve fixtures for a given date using cascading fallback strategy.
        
        Args:
            date_str: Date in YYYY-MM-DD or YYYYMMDD format
            leagues: Optional list of league slugs to query (e.g., ["fifa.world", "eng.1"])
                    If None, uses default leagues based on date/context
            
        Returns:
            Tuple of (DataFrame with fixtures, source_name)
            DataFrame columns: match_id, home_team, away_team, competition, date, 
                              kickoff_datetime, neutral_venue
            source_name: One of "primary_api", "espn", "local_file", "none"
        """
        # Normalize date
        normalized_date = self._normalize_date(date_str)
        filename_date = normalized_date.replace("-", "")
        
        logger.info(f"Resolving fixtures for date: {normalized_date}")
        
        # Strategy A: Try primary API sources
        logger.info("Trying primary fixture source...")
        fixtures_df, source = self._try_primary_apis(normalized_date)
        
        if len(fixtures_df) > 0:
            logger.info(f"Primary source returned {len(fixtures_df)} fixtures")
            return self._normalize_dataframe(fixtures_df, filename_date), "primary_api"
        
        logger.info("No fixtures from primary source, trying ESPN fallback...")
        
        # Strategy B: Try ESPN fallback
        fixtures_df, source = self._try_espn(normalized_date, leagues)
        
        if len(fixtures_df) > 0:
            logger.info(f"ESPN returned {len(fixtures_df)} fixtures for {filename_date}")
            return self._normalize_dataframe(fixtures_df, filename_date), "espn"
        
        logger.info("ESPN returned no fixtures, checking local files...")
        
        # Strategy C: Try local existing fixture files
        fixtures_df, source = self._try_local_file(normalized_date)
        
        if len(fixtures_df) > 0:
            logger.info(f"Using existing local fixture as fallback ({len(fixtures_df)} matches)")
            return self._normalize_dataframe(fixtures_df, filename_date), "local_file"
        
        # Strategy D: All sources failed
        logger.warning("Aborting: no fixtures found in any source")
        
        # Return empty DataFrame with proper columns
        empty_df = self._create_empty_dataframe()
        return empty_df, "none"
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string to YYYY-MM-DD format."""
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
        
        raise ValueError(f"Unrecognized date format: {date_str}. Use YYYYMMDD or YYYY-MM-DD")
    
    def _try_primary_apis(self, date_str: str) -> Tuple[pd.DataFrame, str]:
        """
        Try primary API sources (football-data.org, API-Football).
        
        Returns DataFrame and source name, or empty DataFrame if no fixtures found.
        """
        fixtures = []
        
        # Try football-data.org
        try:
            fixtures_from_api = self._fetch_from_football_data(date_str)
            if fixtures_from_api:
                fixtures.extend(fixtures_from_api)
                logger.info(f"Fetched {len(fixtures_from_api)} fixtures from football-data.org")
        except Exception as e:
            logger.debug(f"football-data.org failed: {e}")
        
        # Try API-Football
        try:
            fixtures_from_api = self._fetch_from_api_football(date_str)
            if fixtures_from_api:
                fixtures.extend(fixtures_from_api)
                logger.info(f"Fetched {len(fixtures_from_api)} fixtures from API-Football")
        except Exception as e:
            logger.debug(f"API-Football failed: {e}")
        
        if fixtures:
            df = pd.DataFrame(fixtures)
            return df, "primary_api"
        
        return pd.DataFrame(), "none"
    
    def _fetch_from_football_data(self, date_str: str) -> List[Dict[str, Any]]:
        """Fetch fixtures from football-data.org."""
        fixtures = []
        
        if not self.api_client.football_data_token:
            logger.debug("No FOOTBALL_DATA_TOKEN available, skipping football-data.org")
            return []
        
        headers = {"X-Auth-Token": self.api_client.football_data_token}
        url = "https://api.football-data.org/v4/matches"
        params = {
            "dateFrom": date_str,
            "dateTo": date_str,
            "limit": 100
        }
        
        response = self.api_client._make_request(url, headers, params)
        matches = response.get("matches", [])
        
        for match in matches:
            status = match.get("status", "")
            if status in ["SCHEDULED", "FINISHED", "IN_PLAY", "PAUSED"]:
                home_team = match.get("homeTeam", {}).get("name")
                away_team = match.get("awayTeam", {}).get("name")
                competition = match.get("competition", {}).get("name", "Unknown")
                utc_date = match.get("utcDate")
                
                if home_team and away_team and utc_date:
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
                        "kickoff_datetime": kickoff_str,
                        "match_id": str(match.get("id", "")),
                        "neutral_venue": False
                    })
        
        return fixtures
    
    def _fetch_from_api_football(self, date_str: str) -> List[Dict[str, Any]]:
        """Fetch fixtures from API-Football."""
        fixtures = []
        
        if not self.api_client.api_football_key:
            logger.debug("No API_FOOTBALL_KEY available, skipping API-Football")
            return []
        
        headers = {"x-apisports-key": self.api_client.api_football_key}
        url = "https://v3.football.api-sports.io/fixtures"
        params = {"date": date_str}
        
        response = self.api_client._make_request(url, headers, params)
        fixture_list = response.get("response", [])
        
        for fixture in fixture_list:
            home_team = fixture.get("teams", {}).get("home", {}).get("name")
            away_team = fixture.get("teams", {}).get("away", {}).get("name")
            league = fixture.get("league", {}).get("name", "Unknown")
            fixture_date = fixture.get("fixture", {}).get("date")
            
            if home_team and away_team and fixture_date:
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
                    "kickoff_datetime": kickoff_str,
                    "match_id": str(fixture.get("fixture", {}).get("id", "")),
                    "neutral_venue": fixture.get("fixture", {}).get("venue", {}).get("neutral", False)
                })
        
        return fixtures
    
    def _try_espn(self, date_str: str, leagues: Optional[List[str]] = None) -> Tuple[pd.DataFrame, str]:
        """
        Try ESPN as fallback source.
        
        Uses ESPN's public scoreboard API for soccer.
        No API key required.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            leagues: Optional list of league slugs. If None, tries default leagues.
            
        Returns:
            Tuple of (DataFrame, source_name)
        """
        # Convert date to ESPN format (YYYYMMDD)
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            espn_date = date_obj.strftime("%Y%m%d")
        except ValueError:
            logger.warning(f"Invalid date for ESPN: {date_str}")
            return pd.DataFrame(), "none"
        
        # Determine which leagues to try
        if leagues is None:
            # Default: try FIFA World Cup first, then other major leagues
            leagues_to_try = ["fifa.world", "uefa.champions", "eng.1", "esp.1"]
        else:
            leagues_to_try = leagues
        
        all_fixtures = []
        
        for league_slug in leagues_to_try:
            try:
                fixtures = self._fetch_from_espn(espn_date, league_slug)
                if fixtures:
                    logger.debug(f"ESPN {league_slug} returned {len(fixtures)} fixtures")
                    all_fixtures.extend(fixtures)
            except Exception as e:
                logger.debug(f"ESPN {league_slug} failed: {e}")
        
        if all_fixtures:
            df = pd.DataFrame(all_fixtures)
            return df, "espn"
        
        return pd.DataFrame(), "none"
    
    def _fetch_from_espn(self, espn_date: str, league_slug: str) -> List[Dict[str, Any]]:
        """
        Fetch fixtures from ESPN for a specific date and league.
        
        ESPN endpoint pattern:
        https://site.api.espn.com/apis/site/v2/sports/soccer/{league_slug}/scoreboard?dates={YYYYMMDD}
        
        Args:
            espn_date: Date in YYYYMMDD format
            league_slug: ESPN league slug (e.g., "fifa.world", "eng.1")
            
        Returns:
            List of fixture dicts
        """
        fixtures = []
        
        # Build ESPN URL
        base_url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league_slug}/scoreboard"
        params = {"dates": espn_date, "limit": 100}
        
        logger.debug(f"ESPN API request: {base_url} dates={espn_date}")
        
        scoreboard = self.espn_client.get_scoreboard(dates=espn_date, limit=100)
        events = scoreboard.get("events", [])
        
        if not events:
            return []
        
        competition_name = ESPN_SOCCER_LEAGUES.get(league_slug, league_slug)
        
        for event in events:
            norm = self.espn_client._normalize_event(event)
            if not norm:
                continue
            
            home_team = norm.get("home_team", "")
            away_team = norm.get("away_team", "")
            
            if not home_team or not away_team:
                continue
            
            # Extract kickoff datetime
            event_date = norm.get("date", "")
            kickoff_str = event_date
            if event_date:
                try:
                    # ESPN date format is typically ISO 8601
                    dt = datetime.fromisoformat(event_date.replace("Z", "+00:00"))
                    kickoff_str = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            
            fixtures.append({
                "date": espn_date[:4] + "-" + espn_date[4:6] + "-" + espn_date[6:],
                "league": competition_name,
                "home_team": home_team,
                "away_team": away_team,
                "kickoff_datetime": kickoff_str,
                "match_id": str(norm.get("event_id", "")),
                "neutral_venue": norm.get("neutral_venue", False)
            })
        
        return fixtures
    
    def _try_local_file(self, date_str: str) -> Tuple[pd.DataFrame, str]:
        """
        Try to load existing local fixture file.
        
        Checks both possible locations:
        - data/fixtures/YYYYMMDD.csv
        - predicciones/data/fixtures/YYYYMMDD.csv
        
        Args:
            date_str: Date in YYYY-MM-DD format
            
        Returns:
            Tuple of (DataFrame, source_name)
        """
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            filename_date = date_obj.strftime("%Y%m%d")
        except ValueError:
            return pd.DataFrame(), "none"
        
        # Check both possible locations
        fixture_paths = [
            project_root / "data" / "fixtures" / f"{filename_date}.csv",
            project_root / "predicciones" / "data" / "fixtures" / f"{filename_date}.csv",
        ]
        
        for fixture_path in fixture_paths:
            if fixture_path.exists():
                try:
                    df = pd.read_csv(fixture_path)
                    if len(df) > 0:
                        logger.debug(f"Found existing fixture file: {fixture_path}")
                        return df, "local_file"
                except Exception as e:
                    logger.debug(f"Error reading {fixture_path}: {e}")
                    continue
        
        return pd.DataFrame(), "none"
    
    def _normalize_dataframe(self, df: pd.DataFrame, filename_date: str) -> pd.DataFrame:
        """
        Normalize DataFrame to ensure required columns exist.
        
        Required columns:
        - match_id
        - home_team
        - away_team
        - competition
        - date
        - kickoff_datetime
        - neutral_venue
        """
        # Ensure all required columns exist
        required_cols = [
            "match_id", "home_team", "away_team", "competition",
            "date", "kickoff_datetime", "neutral_venue"
        ]
        
        # Map common column names to standard names
        column_mapping = {
            "league": "competition",
            "home": "home_team",
            "away": "away_team",
            "home_team_name": "home_team",
            "away_team_name": "away_team",
            "kickoff": "kickoff_datetime",
            "datetime": "kickoff_datetime",
            "neutral": "neutral_venue",
            "event_id": "match_id",
            "id": "match_id",
            "fixture_id": "match_id"
        }
        
        # Rename columns if they match the mapping
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Add missing columns with default values
        defaults = {
            "match_id": "",
            "home_team": "",
            "away_team": "",
            "competition": "Unknown",
            "date": filename_date[:4] + "-" + filename_date[4:6] + "-" + filename_date[6:],
            "kickoff_datetime": "",
            "neutral_venue": False
        }
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = defaults.get(col)
        
        # Select only required columns in order
        df = df[required_cols]
        
        return df
    
    def _create_empty_dataframe(self) -> pd.DataFrame:
        """Create empty DataFrame with required columns."""
        return pd.DataFrame(columns=[
            "match_id", "home_team", "away_team", "competition",
            "date", "kickoff_datetime", "neutral_venue"
        ])


def resolve_fixtures_for_date(
    date_str: str,
    leagues: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Convenience function to resolve fixtures for a date.
    
    Args:
        date_str: Date in YYYY-MM-DD or YYYYMMDD format
        leagues: Optional list of ESPN league slugs to try
        
    Returns:
        Tuple of (DataFrame with fixtures, source_name)
        source_name: One of "primary_api", "espn", "local_file", "none"
    """
    resolver = FixtureResolver()
    return resolver.resolve_fixtures_for_date(date_str, leagues)


if __name__ == "__main__":
    # Test the resolver
    logging.basicConfig(level=logging.INFO)
    
    import argparse
    parser = argparse.ArgumentParser(description="Test fixture resolver")
    parser.add_argument("--date", "-d", type=str, required=True, help="Date to test")
    args = parser.parse_args()
    
    df, source = resolve_fixtures_for_date(args.date)
    
    print(f"\nSource: {source}")
    print(f"Fixtures found: {len(df)}")
    if len(df) > 0:
        print(df.to_string())
