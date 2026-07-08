#!/usr/bin/env python
"""
Build Market Dataset Script.

Downloads ESPN data for historical matches and builds derived datasets
for corners, cards, shots, and player props markets.

Usage:
    python scripts/build_market_dataset.py --days-back 180 --league fifa.world
    python scripts/build_market_dataset.py --start-date 2025-01-01 --end-date 2025-06-01
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.espn_client import EspnWorldCupClient
from src.data.cache_manager import EspnCacheManager
from src.data.espn_stats_parsers import (
    extract_team_stats_from_summary,
    extract_player_stats_from_summary,
    extract_events_from_summary,
    parse_commentary_events_with_stats,
)
from src.utils.config_paths import DERIVED_DIR, ESPN_CACHE_DIR, PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# PATH CONSTANTS (from config_paths for canonical paths)
# =============================================================================
# DERIVED_DIR and CACHE_DIR are now imported from config_paths
# This ensures all parts of the project use the same canonical locations
CACHE_DIR = ESPN_CACHE_DIR


# =============================================================================
# WRITER HELPERS
# =============================================================================
def write_jsonl(path: Path, rows: List[Dict[str, Any]], create_empty: bool = True) -> int:
    """
    Write list of dicts to JSONL file.
    
    Args:
        path: Output file path
        rows: List of dict records to write
        create_empty: If True, create empty file when rows is empty.
                     If False, skip file creation when rows is empty.
    
    Returns:
        Number of rows written (0 if skipped)
    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if not rows:
        if create_empty:
            # Create empty file
            with open(path, "w", encoding="utf-8") as f:
                pass
            logger.info(f"Created empty file: {path}")
            return 0
        else:
            # Skip file creation
            logger.info(f"Skipping file creation (0 rows): {path}")
            return 0
    
    # Write rows
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    
    logger.info(f"Wrote {len(rows)} rows to {path}")
    return len(rows)


class MarketDatasetBuilder:
    """
    Builds market prediction datasets from ESPN data.
    
    Downloads match data, extracts team/player stats,
    and saves derived datasets for model training.
    """
    
    def __init__(self, league: str = "fifa.world"):
        self.league = league
        self.espn_client = EspnWorldCupClient()
        self.cache_manager = EspnCacheManager()
        
        # Counters
        self.matches_processed = 0
        self.matches_with_corners = 0
        self.matches_with_cards = 0
        self.matches_with_sot = 0
        self.matches_with_players = 0
        
        # Event logging counters
        self.total_commentary_count = 0
        self.total_parsed_event_count = 0
        self.aggregated_event_type_counts: Dict[str, int] = {}
        
        # Accumulators for batch writing
        self.team_rows: List[Dict[str, Any]] = []
        self.player_rows: List[Dict[str, Any]] = []
        self.event_rows: List[Dict[str, Any]] = []
    
    def build_dataset(
        self,
        days_back: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_matches: int = 500,
    ) -> Dict[str, int]:
        """
        Build dataset for specified date range.
        
        Args:
            days_back: Number of days back from today
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_matches: Maximum matches to process
            
        Returns:
            Dict with processing statistics
        """
        # Determine date range
        if days_back is not None:
            end = datetime.now(UTC).date()
            start = end - timedelta(days=days_back)
            date_range = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
            logger.info(f"Building dataset for last {days_back} days: {date_range}")
        elif start_date and end_date:
            date_range = f"{start_date.replace('-', '')}-{end_date.replace('-', '')}"
            logger.info(f"Building dataset for {start_date} to {end_date}")
        else:
            # Default: last 90 days
            days_back = 90
            end = datetime.now(UTC).date()
            start = end - timedelta(days=days_back)
            date_range = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
            logger.info(f"Using default date range: last {days_back} days")
        
        # Get scoreboard for date range
        logger.info("Fetching scoreboard from ESPN...")
        scoreboard = self.espn_client.get_scoreboard(dates=date_range, limit=max_matches)
        events = scoreboard.get("events", [])
        
        logger.info(f"Found {len(events)} events in date range")
        
        # Process each event
        for i, event in enumerate(events):
            event_id = str(event.get("id", ""))
            
            # Check cache first
            cached_response = self.cache_manager.get_raw_response(
                "summary",
                {"event": event_id},
                ttl_hours=24 * 7,  # 1 week TTL for historical data
            )
            
            if cached_response:
                summary = cached_response
                logger.debug(f"Using cached summary for event {event_id}")
            else:
                # Fetch summary
                summary = self.espn_client.get_summary(event_id)
                if summary:
                    self.cache_manager.set_raw_response(
                        "summary",
                        {"event": event_id},
                        summary,
                    )
            
            if not summary:
                logger.warning(f"No summary available for event {event_id}")
                continue
            
            # Process match
            self._process_match(event, summary)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(events)} matches...")
        
        # Log summary
        logger.info("=" * 50)
        logger.info("Dataset Build Complete")
        logger.info(f"Total matches processed: {self.matches_processed}")
        logger.info(f"Matches with corners: {self.matches_with_corners}")
        logger.info(f"Matches with cards: {self.matches_with_cards}")
        logger.info(f"Matches with SOT: {self.matches_with_sot}")
        logger.info(f"Matches with player stats: {self.matches_with_players}")
        logger.info("-" * 50)
        logger.info("Match Events Logging")
        logger.info(f"Total commentary entries processed: {self.total_commentary_count}")
        logger.info(f"Total parsed events: {self.total_parsed_event_count}")
        logger.info(f"Event type counts: {self.aggregated_event_type_counts}")
        logger.info("=" * 50)
        
        # Write all accumulated data to files
        self._write_all_data()
        
        return {
            "total_matches": self.matches_processed,
            "with_corners": self.matches_with_corners,
            "with_cards": self.matches_with_cards,
            "with_sot": self.matches_with_sot,
            "with_players": self.matches_with_players,
            "commentary_count": self.total_commentary_count,
            "parsed_event_count": self.total_parsed_event_count,
            "event_type_counts": self.aggregated_event_type_counts,
        }
    
    def _write_all_data(self) -> None:
        """Write all accumulated data to derived files."""
        # Team match stats
        team_path = DERIVED_DIR / "team_match_stats.jsonl"
        team_count = write_jsonl(team_path, self.team_rows, create_empty=True)
        
        # Player match stats
        player_path = DERIVED_DIR / "player_match_stats.jsonl"
        player_count = write_jsonl(player_path, self.player_rows, create_empty=True)
        
        # Match events
        event_path = DERIVED_DIR / "match_events.jsonl"
        event_count = write_jsonl(event_path, self.event_rows, create_empty=True)
        
        logger.info("=" * 50)
        logger.info("Data Files Written")
        logger.info(f"  - {team_path} ({team_count} rows)")
        logger.info(f"  - {player_path} ({player_count} rows)")
        logger.info(f"  - {event_path} ({event_count} rows)")
        logger.info(f"  - {CACHE_DIR}/*.json (raw API responses)")
        logger.info("=" * 50)
    
    def _process_match(self, event: Dict[str, Any], summary: Dict[str, Any]) -> None:
        """Process a single match and save derived stats."""
        event_id = str(event.get("id", ""))
        
        # Extract basic match info
        competitions = event.get("competitions", [])
        if not competitions:
            return
        
        comp = competitions[0]
        competitors = comp.get("competitors", [])
        
        home_comp = None
        away_comp = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home_comp = c
            elif c.get("homeAway") == "away":
                away_comp = c
        
        if not home_comp or not away_comp:
            return
        
        home_team = home_comp.get("team", {}).get("displayName", "")
        away_team = away_comp.get("team", {}).get("displayName", "")
        
        # Extract team stats
        team_stats = extract_team_stats_from_summary(summary)
        
        # Check what data is available
        has_corners = team_stats["home_corners"] is not None or team_stats["away_corners"] is not None
        has_cards = any([
            team_stats["home_yellow_cards"] is not None,
            team_stats["away_yellow_cards"] is not None,
            team_stats["home_total_cards"] is not None,
        ])
        has_sot = team_stats["home_shots_on_target"] is not None or team_stats["away_shots_on_target"] is not None
        
        # Build match stats record
        match_stats = {
            "event_id": event_id,
            "date": event.get("date", ""),
            "home_team": home_team,
            "away_team": away_team,
            "home_score": team_stats["home_goals"],
            "away_score": team_stats["away_goals"],
            "status": "completed",  # We only process completed matches
            
            # Corners
            "home_corners": team_stats["home_corners"],
            "away_corners": team_stats["away_corners"],
            "total_corners": (
                (team_stats["home_corners"] or 0) + (team_stats["away_corners"] or 0)
                if has_corners else None
            ),
            
            # Cards
            "home_yellow_cards": team_stats["home_yellow_cards"],
            "away_yellow_cards": team_stats["away_yellow_cards"],
            "home_red_cards": team_stats["home_red_cards"],
            "away_red_cards": team_stats["away_red_cards"],
            "home_total_cards": team_stats["home_total_cards"],
            "away_total_cards": team_stats["away_total_cards"],
            "total_cards": (
                (team_stats["home_total_cards"] or 0) + (team_stats["away_total_cards"] or 0)
                if has_cards else None
            ),
            
            # Shots
            "home_shots": team_stats["home_shots"],
            "away_shots": team_stats["away_shots"],
            "home_shots_on_target": team_stats["home_shots_on_target"],
            "away_shots_on_target": team_stats["away_shots_on_target"],
            "total_sot": (
                (team_stats["home_shots_on_target"] or 0) + (team_stats["away_shots_on_target"] or 0)
                if has_sot else None
            ),
            
            # Other
            "home_possession": team_stats["home_possession"],
            "away_possession": team_stats["away_possession"],
            "home_fouls": team_stats["home_fouls"],
            "away_fouls": team_stats["away_fouls"],
        }
        
        # Accumulate for batch writing (instead of appending directly to file)
        self.team_rows.append(match_stats)
        self.matches_processed += 1
        
        if has_corners:
            self.matches_with_corners += 1
        if has_cards:
            self.matches_with_cards += 1
        if has_sot:
            self.matches_with_sot += 1
        
        # Extract and accumulate player stats
        players = extract_player_stats_from_summary(summary)
        if players:
            for player in players:
                player_record = {
                    "event_id": event_id,
                    "date": event.get("date", ""),
                    "player_id": player["player_id"],
                    "player_name": player["display_name"],
                    "team": home_team if player["team_home_away"] == "home" else away_team,
                    "position": player["position"],
                    "is_starter": player["is_starter"],
                    "minutes": player["minutes"],
                    "goals": player["goals"],
                    "assists": player["assists"],
                    "shots": player["shots"],
                    "shots_on_target": player["shots_on_target"],
                    "yellow_cards": player["yellow_cards"],
                    "red_cards": player["red_cards"],
                    "total_cards": player["total_cards"],
                }
                self.player_rows.append(player_record)
            
            self.matches_with_players += 1
        
        # Extract and accumulate events (for first/last corner/card analysis)
        # Use the new function that returns both events and stats
        commentary = summary.get("commentary", [])
        match_date = event.get("date", "")
        
        if commentary and isinstance(commentary, list):
            self.total_commentary_count += len(commentary)
            events_list, event_type_counts = parse_commentary_events_with_stats(commentary)
            self.total_parsed_event_count += len(events_list)
            
            # Aggregate event type counts
            for etype, count in event_type_counts.items():
                self.aggregated_event_type_counts[etype] = \
                    self.aggregated_event_type_counts.get(etype, 0) + count
            
            # UNWIND: Write each event as a separate row (event-per-row format)
            for seq_idx, evt in enumerate(events_list):
                event_record = {
                    "event_id": event_id,
                    "match_date": match_date,
                    "sequence_index": seq_idx,
                    "minute": evt.get("minute"),
                    "period": evt.get("period"),
                    "event_type": evt.get("event_type"),
                    "team_name": evt.get("team_name"),
                    "team_abbr": None,  # Will be populated if available
                    "player_name": evt.get("player_name"),
                    "description": evt.get("description"),
                    "raw_event": evt.get("raw_event"),
                }
                
                # Try to extract team abbreviation from raw_event
                raw_evt = evt.get("raw_event", {})
                if isinstance(raw_evt, dict):
                    play = raw_evt.get("play", {})
                    if isinstance(play, dict):
                        team = play.get("team", {})
                        if isinstance(team, dict):
                            event_record["team_abbr"] = team.get("abbreviation") or team.get("abbr")
                
                self.event_rows.append(event_record)
        else:
            # Fallback to old method for non-soccer sports
            events_list = extract_events_from_summary(summary)
            if events_list:
                self.total_parsed_event_count += len(events_list)
                # Also unwind fallback events
                for seq_idx, evt in enumerate(events_list):
                    event_record = {
                        "event_id": event_id,
                        "match_date": match_date,
                        "sequence_index": seq_idx,
                        "minute": evt.get("minute"),
                        "period": evt.get("period"),
                        "event_type": evt.get("event_type"),
                        "team_name": evt.get("team_name"),
                        "team_abbr": evt.get("team_abbr"),
                        "player_name": evt.get("player_name"),
                        "description": evt.get("description"),
                        "raw_event": evt.get("raw_event"),
                    }
                    self.event_rows.append(event_record)


def main():
    parser = argparse.ArgumentParser(
        description="Build market prediction dataset from ESPN data"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=None,
        help="Number of days back from today (default: 90)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--league",
        type=str,
        default="fifa.world",
        help="League identifier (default: fifa.world)"
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=500,
        help="Maximum matches to process (default: 500)"
    )
    
    args = parser.parse_args()
    
    builder = MarketDatasetBuilder(league=args.league)
    stats = builder.build_dataset(
        days_back=args.days_back,
        start_date=args.start_date,
        end_date=args.end_date,
        max_matches=args.max_matches,
    )
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total matches processed: {stats['total_matches']}")
    print(f"Matches with corners data: {stats['with_corners']} ({stats['with_corners']/max(stats['total_matches'],1)*100:.1f}%)")
    print(f"Matches with cards data: {stats['with_cards']} ({stats['with_cards']/max(stats['total_matches'],1)*100:.1f}%)")
    print(f"Matches with SOT data: {stats['with_sot']} ({stats['with_sot']/max(stats['total_matches'],1)*100:.1f}%)")
    print(f"Matches with player stats: {stats['with_players']} ({stats['with_players']/max(stats['total_matches'],1)*100:.1f}%)")
    print("=" * 50)
    
    # Print accurate file status
    print(f"\nData saved to:")
    team_file = DERIVED_DIR / "team_match_stats.jsonl"
    player_file = DERIVED_DIR / "player_match_stats.jsonl"
    event_file = DERIVED_DIR / "match_events.jsonl"
    
    # Use PROJECT_ROOT for relative path calculations
    if team_file.exists():
        with open(team_file, "r") as f:
            team_rows = sum(1 for _ in f)
        try:
            rel_path = team_file.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = team_file
        print(f"  - {rel_path} ({team_rows} rows)")
    else:
        try:
            rel_path = team_file.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = team_file
        print(f"  - {rel_path} (not created, 0 rows)")
    
    if player_file.exists():
        with open(player_file, "r") as f:
            player_rows = sum(1 for _ in f)
        try:
            rel_path = player_file.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = player_file
        print(f"  - {rel_path} ({player_rows} rows)")
    else:
        try:
            rel_path = player_file.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = player_file
        print(f"  - {rel_path} (not created, 0 rows)")
    
    if event_file.exists():
        with open(event_file, "r") as f:
            event_rows = sum(1 for _ in f)
        try:
            rel_path = event_file.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = event_file
        print(f"  - {rel_path} ({event_rows} rows)")
    else:
        try:
            rel_path = event_file.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = event_file
        print(f"  - {rel_path} (not created, 0 rows)")
    
    try:
        cache_rel = CACHE_DIR.relative_to(PROJECT_ROOT)
    except ValueError:
        cache_rel = CACHE_DIR
    print(f"  - {cache_rel}/*.json (raw API responses)")


if __name__ == "__main__":
    main()
