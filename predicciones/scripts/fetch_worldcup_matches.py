#!/usr/bin/env python
"""
Fetch World Cup Matches from ESPN API.

This script downloads match summaries from ESPN for FIFA World Cup matches
and caches them in the standard directory structure so they can be used by
analyze_match_state_patterns.py and other analysis tools.

Features:
- Fetch match list from scoreboard endpoint
- Download summary for each match
- Cache in data/cache/espn/ with standard naming convention
- Validate cached files before marking as complete
- Support for date ranges, max matches, and force refresh
- Rate limiting and error handling

Usage:
    python scripts/fetch_worldcup_matches.py --league fifa.world --from-date 2025-01-01 --to-date 2026-07-08 --verbose
    python scripts/fetch_worldcup_matches.py --league fifa.world --max-matches 200 --verbose
    python scripts/fetch_worldcup_matches.py --league fifa.world --from-date 2026-06-01 --to-date 2026-07-08 --force-refresh --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.espn_client import EspnWorldCupClient

logger = logging.getLogger(__name__)


class MatchFetcher:
    """Fetches and caches World Cup matches from ESPN API."""
    
    def __init__(
        self,
        league: str = "fifa.world",
        cache_dir: str = "data/cache/espn",
        timeout: int = 30,
        rate_limit_delay: float = 0.5,
    ):
        """
        Initialize the match fetcher.
        
        Args:
            league: League identifier (default: fifa.world)
            cache_dir: Directory to cache downloaded summaries
            timeout: Request timeout in seconds
            rate_limit_delay: Delay between API requests to avoid rate limiting
        """
        self.league = league
        self.cache_dir = Path(cache_dir)
        self.rate_limit_delay = rate_limit_delay
        self.client = EspnWorldCupClient(timeout=timeout)
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "total_found": 0,
            "already_cached": 0,
            "downloaded": 0,
            "failed": 0,
            "invalid": 0,
        }
    
    def _get_cache_path(self, event_id: str) -> Path:
        """Generate cache file path for an event ID."""
        # Match the existing naming convention:
        # https:__site.api.espn.com_apis_site_v2_sports_soccer_{league}_summary_{"event": "{event_id}"}.json
        filename = f'https:__site.api.espn.com_apis_site_v2_sports_soccer_{self.league}_summary_{{"event": "{event_id}"}}.json'
        return self.cache_dir / filename
    
    def _is_already_cached(self, event_id: str, force_refresh: bool = False) -> bool:
        """Check if a match is already cached."""
        if force_refresh:
            return False
        
        cache_path = self._get_cache_path(event_id)
        return cache_path.exists()
    
    def _validate_summary(self, summary_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that a summary contains required data.
        
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        if not summary_data:
            return False, "Empty summary"
        
        # Check for response wrapper
        if "response" not in summary_data:
            return False, "Missing 'response' key"
        
        response = summary_data["response"]
        
        # Check for header with basic match info
        header = response.get("header", {})
        if not header:
            return False, "Missing header"
        
        # Check for competitions
        competitions = response.get("competitions", [])
        if not competitions:
            return False, "Missing competitions"
        
        comp = competitions[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            return False, "Missing competitors"
        
        # Check for commentary or keyEvents
        commentary = response.get("commentary", [])
        key_events = response.get("keyEvents", [])
        
        # At least some events should be present
        if len(commentary) < 5 and len(key_events) < 3:
            return False, f"Too few events (commentary={len(commentary)}, keyEvents={len(key_events)})"
        
        return True, ""
    
    def _download_summary(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Download summary for a specific event."""
        try:
            logger.info(f"Downloading summary for event {event_id}...")
            summary = self.client.get_summary(event_id)
            
            if not summary:
                logger.warning(f"Empty response for event {event_id}")
                return None
            
            # Wrap in response key if needed (match existing format)
            if "response" not in summary:
                summary = {"response": summary}
            
            return summary
            
        except Exception as e:
            logger.error(f"Error downloading event {event_id}: {e}")
            return None
    
    def _save_to_cache(self, event_id: str, summary_data: Dict[str, Any]) -> bool:
        """Save summary to cache file."""
        try:
            cache_path = self._get_cache_path(event_id)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Cached event {event_id} to {cache_path}")
            return True
        except Exception as e:
            logger.error(f"Error caching event {event_id}: {e}")
            return False
    
    def fetch_scoreboard_matches(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        max_matches: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch match list from scoreboard endpoint.
        
        Args:
            from_date: Start date for range
            to_date: End date for range
            max_matches: Maximum number of matches to return
            
        Returns:
            List of match info dictionaries with event_id, date, teams, etc.
        """
        logger.info("Fetching match list from scoreboard...")
        
        # Build dates parameter
        dates_param = None
        if from_date and to_date:
            # Format: YYYYMMDD-YYYYMMDD
            dates_param = f"{from_date.strftime('%Y%m%d')}-{to_date.strftime('%Y%m%d')}"
        elif from_date:
            dates_param = from_date.strftime('%Y%m%d')
        elif to_date:
            dates_param = to_date.strftime('%Y%m%d')
        
        # Fetch scoreboard
        scoreboard_data = self.client.get_scoreboard(dates=dates_param, limit=max_matches or 500)
        
        if not scoreboard_data:
            logger.warning("Empty scoreboard response")
            return []
        
        # Extract events
        events = scoreboard_data.get("events", [])
        self.stats["total_found"] = len(events)
        
        logger.info(f"Found {len(events)} matches in scoreboard")
        
        # Normalize and filter events
        matches = []
        for event in events:
            try:
                normalized = self.client._normalize_event(event)
                if normalized and normalized.get("event_id"):
                    matches.append(normalized)
            except Exception as e:
                logger.warning(f"Error normalizing event: {e}")
        
        # Apply max_matches limit
        if max_matches and len(matches) > max_matches:
            matches = matches[:max_matches]
        
        # Sort by date
        matches.sort(key=lambda x: x.get("date", ""), reverse=True)
        
        return matches
    
    def fetch_and_cache_matches(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        max_matches: Optional[int] = None,
        force_refresh: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Main method to fetch and cache all matches.
        
        Args:
            from_date: Start date for range
            to_date: End date for range
            max_matches: Maximum number of matches to process
            force_refresh: Re-download even if already cached
            verbose: Print detailed progress
            
        Returns:
            Dictionary with statistics
        """
        # Fetch match list
        matches = self.fetch_scoreboard_matches(
            from_date=from_date,
            to_date=to_date,
            max_matches=max_matches,
        )
        
        if not matches:
            logger.warning("No matches found to process")
            return self.stats
        
        # Process each match
        for i, match in enumerate(matches):
            event_id = match.get("event_id")
            if not event_id:
                continue
            
            # Check if already cached
            if self._is_already_cached(event_id, force_refresh):
                self.stats["already_cached"] += 1
                if verbose:
                    print(f"[{i+1}/{len(matches)}] Event {event_id} already cached, skipping")
                continue
            
            # Download summary
            summary = self._download_summary(event_id)
            
            if not summary:
                self.stats["failed"] += 1
                logger.warning(f"Failed to download event {event_id}")
                continue
            
            # Validate summary
            is_valid, reason = self._validate_summary(summary)
            
            if not is_valid:
                self.stats["invalid"] += 1
                logger.warning(f"Invalid summary for event {event_id}: {reason}")
                # Still cache invalid summaries but log the issue
                # This allows manual inspection later
            
            # Save to cache
            if self._save_to_cache(event_id, summary):
                self.stats["downloaded"] += 1
                if verbose:
                    status = "✓" if is_valid else "⚠"
                    teams = f"{match.get('home_team', '?')} vs {match.get('away_team', '?')}"
                    print(f"[{i+1}/{len(matches)}] {status} Event {event_id}: {teams}")
            
            # Rate limiting
            if i < len(matches) - 1:
                time.sleep(self.rate_limit_delay)
        
        # Print summary
        print("\n=== Fetch Complete ===")
        print(f"Total matches found: {self.stats['total_found']}")
        print(f"Already cached: {self.stats['already_cached']}")
        print(f"Newly downloaded: {self.stats['downloaded']}")
        print(f"Failed downloads: {self.stats['failed']}")
        print(f"Invalid summaries: {self.stats['invalid']}")
        print(f"Cache directory: {self.cache_dir}")
        
        return self.stats


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch World Cup matches from ESPN API and cache them for analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/fetch_worldcup_matches.py --league fifa.world --from-date 2025-01-01 --to-date 2026-07-08 --verbose
  python scripts/fetch_worldcup_matches.py --league fifa.world --max-matches 200 --verbose
  python scripts/fetch_worldcup_matches.py --league fifa.world --from-date 2026-06-01 --to-date 2026-07-08 --force-refresh --verbose
        """
    )
    
    parser.add_argument(
        "--league",
        type=str,
        default="fifa.world",
        help="League identifier (default: fifa.world)"
    )
    parser.add_argument(
        "--from-date",
        type=parse_date,
        default=None,
        help="Start date for match range (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--to-date",
        type=parse_date,
        default=None,
        help="End date for match range (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Maximum number of matches to fetch"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download matches even if already cached"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache/espn",
        help="Directory to cache downloaded summaries (default: data/cache/espn)"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Delay between API requests in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create fetcher and run
    fetcher = MatchFetcher(
        league=args.league,
        cache_dir=args.cache_dir,
        timeout=args.timeout,
        rate_limit_delay=args.rate_limit,
    )
    
    stats = fetcher.fetch_and_cache_matches(
        from_date=args.from_date,
        to_date=args.to_date,
        max_matches=args.max_matches,
        force_refresh=args.force_refresh,
        verbose=args.verbose,
    )
    
    # Exit with error code if no matches were processed
    if stats["downloaded"] == 0 and stats["already_cached"] == 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
