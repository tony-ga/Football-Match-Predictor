#!/usr/bin/env python3
"""
Regenerate match_events.jsonl for all event_ids present in player_match_stats.jsonl.

This script:
1. Reads all unique event_ids from player_match_stats.jsonl
2. Fetches match events from ESPN API for each event_id
3. Writes them to match_events.jsonl
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "predicciones"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DERIVED_DIR = PROJECT_ROOT / "data" / "derived"
PLAYER_STATS_PATH = DERIVED_DIR / "player_match_stats.jsonl"
MATCH_EVENTS_PATH = DERIVED_DIR / "match_events.jsonl"
MATCH_EVENTS_BACKUP_PATH = DERIVED_DIR / "match_events.jsonl.bak"


def get_unique_event_ids_from_player_stats() -> List[str]:
    """Extract all unique event_ids from player_match_stats.jsonl."""
    if not PLAYER_STATS_PATH.exists():
        logger.error(f"Player stats file not found: {PLAYER_STATS_PATH}")
        return []
    
    event_ids = set()
    with open(PLAYER_STATS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    eid = record.get("event_id")
                    if eid:
                        event_ids.add(eid)
                except json.JSONDecodeError:
                    continue
    
    return sorted(list(event_ids))


def fetch_match_events_for_event_id(event_id: str, league: str = "fifa.world") -> Optional[Dict[str, Any]]:
    """
    Fetch match events for a single event_id from ESPN API.
    
    Returns a dict with:
    - event_id
    - date
    - competition
    - home_team
    - away_team
    - events: list of normalized events
    """
    try:
        from predicciones.src.data.espn_match_events import (
            fetch_match_summary,
            extract_commentary_events,
            extract_key_events,
            normalize_match_events,
        )
        
        # Fetch summary
        summary = fetch_match_summary(event_id, league=league)
        
        if not summary:
            logger.warning(f"No summary data for event {event_id}")
            return None
        
        # Extract basic match info
        competitions = summary.get("competitions", [])
        if not competitions:
            logger.warning(f"No competitions found for event {event_id}")
            return None
        
        competition_data = competitions[0]
        competition_name = competition_data.get("conference", {}).get("name", "")
        
        # Get teams
        competitors = competition_data.get("competitors", [])
        home_team = ""
        away_team = ""
        for comp in competitors:
            team_data = comp.get("team", {})
            team_name = team_data.get("displayName", "") or team_data.get("name", "")
            if comp.get("homeAway") == "home":
                home_team = team_name
            else:
                away_team = team_name
        
        # Get date
        event_date = competition_data.get("date", "")
        
        # Extract events from commentary
        commentary_events = extract_commentary_events(summary)
        key_events = extract_key_events(summary)
        
        # Normalize and merge events
        all_events = normalize_match_events(
            commentary_events=commentary_events,
            key_events=key_events,
            core_plays_events=[],
            event_id=event_id
        )
        
        # Build result
        result = {
            "event_id": event_id,
            "date": event_date,
            "competition": competition_name,
            "league_slug": league,
            "home_team": home_team,
            "away_team": away_team,
            "events": all_events,
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching events for {event_id}: {e}")
        return None


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Regenerating match_events.jsonl for all event_ids")
    logger.info("=" * 60)
    
    # Step 1: Get all unique event_ids from player_match_stats.jsonl
    logger.info(f"Reading event_ids from: {PLAYER_STATS_PATH}")
    event_ids = get_unique_event_ids_from_player_stats()
    
    if not event_ids:
        logger.error("No event_ids found in player_match_stats.jsonl")
        sys.exit(1)
    
    logger.info(f"Found {len(event_ids)} unique event_ids")
    logger.info(f"First 10: {event_ids[:10]}")
    
    # Step 2: Backup existing match_events.jsonl if it exists
    if MATCH_EVENTS_PATH.exists():
        logger.info(f"Backing up existing match_events.jsonl to {MATCH_EVENTS_BACKUP_PATH}")
        MATCH_EVENTS_BACKUP_PATH.write_bytes(MATCH_EVENTS_PATH.read_bytes())
    
    # Step 3: Fetch events for each event_id
    successful = 0
    failed = 0
    failed_ids = []
    
    # Open output file
    with open(MATCH_EVENTS_PATH, 'w', encoding='utf-8') as f:
        for i, event_id in enumerate(event_ids, 1):
            logger.info(f"[{i}/{len(event_ids)}] Fetching events for event_id: {event_id}")
            
            # Try fifa.world first, then try other common leagues if needed
            leagues_to_try = ["fifa.world", "uefa.champions", "eng.1", "esp.1", "ita.1", "ger.1", "fra.1"]
            result = None
            
            for league in leagues_to_try:
                result = fetch_match_events_for_event_id(event_id, league=league)
                if result:
                    logger.info(f"  Success with league: {league}")
                    break
            
            if result:
                # Write as JSONL
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                successful += 1
            else:
                failed += 1
                failed_ids.append(event_id)
                logger.warning(f"  Failed to fetch events for {event_id}")
            
            # Progress update every 10 events
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(event_ids)} ({successful} successful, {failed} failed)")
    
    # Step 4: Summary
    logger.info("=" * 60)
    logger.info("REGENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total event_ids processed: {len(event_ids)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    if failed_ids:
        logger.warning(f"Failed event_ids: {failed_ids[:20]}{'...' if len(failed_ids) > 20 else ''}")
    
    logger.info(f"Output written to: {MATCH_EVENTS_PATH}")
    
    # Verify output
    if MATCH_EVENTS_PATH.exists():
        with open(MATCH_EVENTS_PATH, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        logger.info(f"Lines in output file: {line_count}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
