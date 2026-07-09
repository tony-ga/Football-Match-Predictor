#!/usr/bin/env python3
"""
Enrich player_match_stats.jsonl with:
1. Actual minutes played derived from match_events.jsonl substitutions
2. Competition info (competition name, league_slug)
3. Opponent and home_or_away info

This script reads existing player_match_stats.jsonl and match_events.jsonl,
derives missing data, and writes an enriched version.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DERIVED_DIR = Path("/workspace/data/derived")
PLAYER_STATS_PATH = DERIVED_DIR / "player_match_stats.jsonl"
MATCH_EVENTS_PATH = DERIVED_DIR / "match_events.jsonl"
OUTPUT_PATH = DERIVED_DIR / "player_match_stats_enriched.jsonl"


def load_match_events() -> Dict[str, Dict[str, Any]]:
    """Load match events indexed by event_id."""
    events = {}
    if not MATCH_EVENTS_PATH.exists():
        logger.warning(f"Match events file not found: {MATCH_EVENTS_PATH}")
        return events
    
    with open(MATCH_EVENTS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    event_id = str(record.get("event_id", ""))
                    if event_id:
                        events[event_id] = record
                except json.JSONDecodeError:
                    continue
    
    logger.info(f"Loaded {len(events)} match events")
    return events


def extract_substitutions_and_duration(match_event: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, int]], int]:
    """
    Extract substitution timings and match duration from match event.
    
    Returns:
        - substitutions: Dict[athlete_id] -> {subbed_in_minute, subbed_out_minute}
        - match_duration: Total match duration in minutes
    """
    substitutions = defaultdict(lambda: {"subbed_in_minute": None, "subbed_out_minute": None})
    match_duration = 90
    
    events_list = match_event.get("events", [])
    
    # Find final whistle to determine match duration
    for evt in events_list:
        desc = (evt.get("description") or "").lower()
        raw_event = evt.get("raw_event", {})
        clock = raw_event.get("clock", {}) or raw_event.get("time", {})
        clock_value = clock.get("value", 0) or 0
        
        # Detect match end
        if "ends" in desc and ("match" in desc or "half extra time" in desc or "second half" in desc):
            if clock_value > 0:
                match_duration = max(match_duration, int(clock_value // 60))
        
        # Detect substitution events
        event_type = (evt.get("event_type") or "").lower()
        if event_type == "substitution" or "substitution" in desc or "replaces" in desc:
            # Extract players from raw_event.participants
            play_info = raw_event.get("play", {})
            participants = play_info.get("participants", [])
            
            minute = int(clock_value // 60) if clock_value else None
            
            # First participant is usually the player coming IN
            # Second participant is usually the player going OUT
            if len(participants) >= 2:
                player_in = participants[0].get("athlete", {}).get("id", "")
                player_out = participants[1].get("athlete", {}).get("id", "")
                
                if player_in:
                    substitutions[str(player_in)]["subbed_in_minute"] = minute
                if player_out:
                    substitutions[str(player_out)]["subbed_out_minute"] = minute
            elif len(participants) == 1:
                # Try to extract from description: "Player A replaces Player B"
                desc_text = evt.get("description", "")
                if "replaces" in desc_text:
                    parts = desc_text.split(" replaces ")
                    if len(parts) == 2:
                        player_in_name = parts[0].split("(")[0].strip()
                        player_out_name = parts[1].split("(")[0].strip()
                        # We'd need name-to-id mapping, skip for now
                        pass
    
    return dict(substitutions), match_duration


def derive_minutes_from_events(
    event_id: str,
    player_id: str,
    is_starter: bool,
    match_events: Dict[str, Dict[str, Any]]
) -> Optional[int]:
    """
    Derive minutes played for a player from match events.
    
    Logic:
    - Starter who wasn't subbed out: full match duration
    - Starter who was subbed out: subbed_out_minute
    - Sub who came in: match_duration - subbed_in_minute
    - Sub who never came in: 0
    """
    if event_id not in match_events:
        return None
    
    match_event = match_events[event_id]
    substitutions, match_duration = extract_substitutions_and_duration(match_event)
    
    if player_id in substitutions:
        sub_info = substitutions[player_id]
        subbed_in = sub_info.get("subbed_in_minute")
        subbed_out = sub_info.get("subbed_out_minute")
        
        if is_starter:
            if subbed_out is not None:
                return subbed_out
            else:
                return match_duration
        else:
            if subbed_in is not None:
                return max(0, match_duration - subbed_in)
            else:
                # Sub who never entered
                return 0
    
    # No substitution data found
    if is_starter:
        return match_duration
    else:
        return 0  # Assume sub who didn't play


def enrich_player_stats():
    """Main enrichment function."""
    logger.info("Loading match events...")
    match_events = load_match_events()
    
    if not match_events:
        logger.error("No match events available. Cannot enrich player stats.")
        return False
    
    logger.info("Loading player stats...")
    player_stats = []
    if not PLAYER_STATS_PATH.exists():
        logger.error(f"Player stats file not found: {PLAYER_STATS_PATH}")
        return False
    
    with open(PLAYER_STATS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    player_stats.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    logger.info(f"Loaded {len(player_stats)} player stat records")
    
    # Enrich records
    enriched = []
    null_minutes_before = sum(1 for p in player_stats if p.get("minutes") is None)
    null_minutes_after = 0
    
    for record in player_stats:
        event_id = str(record.get("event_id", ""))
        player_id = str(record.get("player_id", ""))
        is_starter = record.get("is_starter", False)
        
        # Get competition info from match_events
        competition = ""
        league_slug = ""
        opponent = ""
        home_or_away = ""
        
        if event_id in match_events:
            match_event = match_events[event_id]
            competition = match_event.get("competition", "")
            
            # Derive league_slug from competition
            if "fifa.world" in competition.lower() or "world cup" in competition.lower():
                league_slug = "fifa.world"
            elif "friendly" in competition.lower():
                league_slug = "international.friendly"
            elif "uefa" in competition.lower():
                league_slug = "uefa.champions"
            else:
                league_slug = competition.lower().replace(" ", ".") if competition else ""
            
            # Determine opponent and home/away
            home_team = match_event.get("home_team", "")
            away_team = match_event.get("away_team", "")
            player_team = record.get("team", "")
            
            if player_team.lower() == home_team.lower():
                opponent = away_team
                home_or_away = "home"
            elif player_team.lower() == away_team.lower():
                opponent = home_team
                home_or_away = "away"
        
        # Derive minutes if null
        minutes = record.get("minutes")
        if minutes is None:
            derived_minutes = derive_minutes_from_events(
                event_id, player_id, is_starter, match_events
            )
            if derived_minutes is not None:
                minutes = derived_minutes
            else:
                null_minutes_after += 1
        
        # Build enriched record
        enriched_record = {
            "event_id": event_id,
            "date": record.get("date", ""),
            "player_id": player_id,
            "player_name": record.get("player_name", ""),
            "team": record.get("team", ""),
            "position": record.get("position", ""),
            "is_starter": is_starter,
            "minutes": minutes,
            "goals": record.get("goals"),
            "assists": record.get("assists"),
            "shots": record.get("shots"),
            "shots_on_target": record.get("shots_on_target"),
            "yellow_cards": record.get("yellow_cards"),
            "red_cards": record.get("red_cards"),
            "total_cards": record.get("total_cards"),
            # New fields
            "competition": competition,
            "league_slug": league_slug,
            "opponent": opponent,
            "home_or_away": home_or_away,
        }
        enriched.append(enriched_record)
    
    # Write enriched file
    logger.info(f"Writing {len(enriched)} enriched records to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for record in enriched:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # Backup original and replace
    backup_path = PLAYER_STATS_PATH.with_suffix(".jsonl.bak")
    logger.info(f"Backing up original to {backup_path}")
    PLAYER_STATS_PATH.rename(backup_path)
    
    logger.info(f"Moving enriched to {PLAYER_STATS_PATH}")
    OUTPUT_PATH.rename(PLAYER_STATS_PATH)
    
    # Report
    logger.info("=" * 60)
    logger.info("ENRICHMENT COMPLETE")
    logger.info(f"Records processed: {len(enriched)}")
    logger.info(f"Null minutes before: {null_minutes_before}")
    logger.info(f"Null minutes after: {null_minutes_after}")
    logger.info(f"Success rate: {(1 - null_minutes_after/max(len(enriched),1))*100:.1f}%")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    success = enrich_player_stats()
    exit(0 if success else 1)
