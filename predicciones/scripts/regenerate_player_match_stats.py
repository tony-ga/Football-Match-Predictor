#!/usr/bin/env python3
"""
Build data/derived/player_match_stats.jsonl from ESPN match summaries.

Layer contract:
- roster/base: summary.rosters[].roster[] is the source of participating players.
- boxscore stats: each roster player's stats[] is the source of goals/shots/cards.
- events: keyEvents/plays enrich cards and substitution timing only as fallback.
- summary views: espn_player_stats.py consumes the generated JSONL.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DERIVED_DIR = PROJECT_ROOT / "data" / "derived"
DEFAULT_OUTPUT = DERIVED_DIR / "player_match_stats.jsonl"
TEAM_STATS_PATH = DERIVED_DIR / "team_match_stats.jsonl"

sys.path.insert(0, str(PROJECT_ROOT))

from predicciones.src.data.espn_stats_parsers import _parse_player_stat_array

LOGGER = logging.getLogger(__name__)
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary"


def _num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _team_context(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    competitors = (
        summary.get("header", {})
        .get("competitions", [{}])[0]
        .get("competitors", [])
    )
    context: Dict[str, Dict[str, Any]] = {}
    for comp in competitors:
        team = comp.get("team", {})
        home_away = comp.get("homeAway", "")
        name = team.get("displayName", "")
        context[home_away] = {
            "team_id": str(team.get("id") or ""),
            "team": name,
            "score": _num(comp.get("score")),
        }
    for side, other in (("home", "away"), ("away", "home")):
        if side in context and other in context:
            context[side]["opponent"] = context[other]["team"]
    return context


def _competition(summary: Dict[str, Any]) -> Tuple[str, str]:
    league = summary.get("header", {}).get("league", {})
    competition = league.get("name") or league.get("displayName") or "FIFA World Cup"
    slug = league.get("slug") or "fifa.world"
    return competition, slug


def _event_id(summary: Dict[str, Any], fallback: str) -> str:
    return str(summary.get("header", {}).get("id") or fallback)


def _event_date(summary: Dict[str, Any]) -> str:
    return summary.get("header", {}).get("competitions", [{}])[0].get("date", "")


def _event_minute(event: Dict[str, Any]) -> Optional[int]:
    clock = event.get("clock", {})
    if isinstance(clock, dict) and clock.get("value") is not None:
        try:
            return int(float(clock["value"]) // 60)
        except (TypeError, ValueError):
            return None
    return None


def _participants(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    participants = event.get("participants", [])
    if not isinstance(participants, list):
        return []
    normalized = []
    for participant in participants:
        if not isinstance(participant, dict):
            continue
        athlete = participant.get("athlete") if isinstance(participant.get("athlete"), dict) else participant
        if athlete.get("id"):
            normalized.append(athlete)
    return normalized


def _event_fallbacks(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    by_player: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "goals": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "subbed_in_minute": None,
        "subbed_out_minute": None,
    })
    for event in summary.get("keyEvents", []) or summary.get("plays", []) or []:
        if not isinstance(event, dict):
            continue
        text = (event.get("text") or "").lower()
        type_info = event.get("type", {})
        type_text = ""
        if isinstance(type_info, dict):
            type_text = f"{type_info.get('text', '')} {type_info.get('type', '')}".lower()
        event_text = f"{text} {type_text}"
        participants = _participants(event)
        minute = _event_minute(event)

        if "yellow card" in event_text:
            for athlete in participants:
                by_player[str(athlete["id"])]["yellow_cards"] += 1
        elif "red card" in event_text:
            for athlete in participants:
                by_player[str(athlete["id"])]["red_cards"] += 1
        elif "goal" in event_text and "own goal" not in event_text:
            for athlete in participants[:1]:
                by_player[str(athlete["id"])]["goals"] += 1
        elif "substitut" in event_text and len(participants) >= 2:
            by_player[str(participants[0]["id"])]["subbed_in_minute"] = minute
            by_player[str(participants[1]["id"])]["subbed_out_minute"] = minute
    return dict(by_player)


def _minutes(player: Dict[str, Any], player_events: Dict[str, Any], match_duration: int = 90) -> int:
    is_starter = bool(player.get("starter") or player.get("starterFlag"))
    subbed_in = bool(player.get("subbedIn"))
    subbed_out = bool(player.get("subbedOut"))
    sub_in_min = player_events.get("subbed_in_minute")
    sub_out_min = player_events.get("subbed_out_minute")
    if is_starter:
        return int(sub_out_min) if subbed_out and sub_out_min is not None else match_duration
    if subbed_in:
        return max(0, match_duration - int(sub_in_min)) if sub_in_min is not None else 25
    return 0


def build_player_match_rows(summary: Dict[str, Any], event_id: str) -> List[Dict[str, Any]]:
    competition, league_slug = _competition(summary)
    team_context = _team_context(summary)
    events_by_player = _event_fallbacks(summary)
    resolved_event_id = _event_id(summary, event_id)
    date = _event_date(summary)
    rows: List[Dict[str, Any]] = []

    for roster_block in summary.get("rosters", []):
        if not isinstance(roster_block, dict):
            continue
        home_away = roster_block.get("homeAway", "")
        team_info = roster_block.get("team", {})
        team_name = team_info.get("displayName") or team_context.get(home_away, {}).get("team", "")
        opponent = team_context.get(home_away, {}).get("opponent", "")
        for player in roster_block.get("roster", []) or []:
            athlete = player.get("athlete") if isinstance(player.get("athlete"), dict) else {}
            player_id = str(athlete.get("id") or player.get("id") or "")
            if not player_id:
                continue
            parsed = _parse_player_stat_array(player.get("stats", []))
            fallback = events_by_player.get(player_id, {})
            position = player.get("position", {})
            if isinstance(position, dict):
                position = position.get("abbreviation") or position.get("name") or ""
            goals = parsed.get("goals")
            yellow = parsed.get("yellow_cards")
            red = parsed.get("red_cards")
            if goals is None:
                goals = fallback.get("goals", 0)
            if yellow is None:
                yellow = fallback.get("yellow_cards", 0)
            if red is None:
                red = fallback.get("red_cards", 0)
            rows.append({
                "event_id": resolved_event_id,
                "date": date,
                "player_id": player_id,
                "player_name": athlete.get("displayName") or player.get("displayName") or "",
                "team": team_name,
                "position": position,
                "is_starter": bool(player.get("starter") or player.get("starterFlag")),
                "minutes": _minutes(player, fallback),
                "goals": _num(goals),
                "assists": _num(parsed.get("assists")),
                "shots": _num(parsed.get("shots")),
                "shots_on_target": _num(parsed.get("shots_on_target")),
                "yellow_cards": _num(yellow),
                "red_cards": _num(red),
                "total_cards": _num(parsed.get("total_cards"), _num(yellow) + _num(red)),
                "competition": competition,
                "league_slug": league_slug,
                "opponent": opponent,
                "home_or_away": home_away,
            })
    return rows


def load_event_ids_from_team_stats(path: Path = TEAM_STATS_PATH) -> List[str]:
    event_ids: List[str] = []
    if not path.exists():
        return event_ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            event_id = str(record.get("event_id") or record.get("match_id") or "")
            if event_id and event_id not in event_ids:
                event_ids.append(event_id)
    return event_ids


def fetch_summary(event_id: str, session: requests.Session) -> Dict[str, Any]:
    response = session.get(ESPN_SUMMARY_URL, params={"event": event_id}, timeout=30)
    response.raise_for_status()
    return response.json()


def write_jsonl_atomic(rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp_path, output_path)


def regenerate(event_ids: List[str], output_path: Path = DEFAULT_OUTPUT) -> List[Dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    all_rows: List[Dict[str, Any]] = []
    for event_id in event_ids:
        LOGGER.info("Fetching ESPN summary for event %s", event_id)
        summary = fetch_summary(event_id, session)
        rows = build_player_match_rows(summary, event_id)
        if not rows:
            LOGGER.warning("No player rows generated for event %s", event_id)
            continue
        all_rows.extend(rows)
    all_rows.sort(key=lambda r: (r["date"], r["event_id"], r["team"], not r["is_starter"], r["player_name"]))
    write_jsonl_atomic(all_rows, output_path)
    return all_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate player_match_stats.jsonl from ESPN summaries")
    parser.add_argument("--event-id", action="append", dest="event_ids", help="ESPN event id. Repeatable.")
    parser.add_argument("--all-from-team-stats", action="store_true", help="Use event ids from data/derived/team_match_stats.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")

    event_ids = args.event_ids or []
    if args.all_from_team_stats:
        event_ids = load_event_ids_from_team_stats()
    if not event_ids:
        parser.error("Pass --event-id at least once or --all-from-team-stats")

    rows = regenerate(event_ids, args.output)
    print(f"Wrote {len(rows)} player-match rows to {args.output}")
    print(f"File size: {args.output.stat().st_size} bytes")
    print(f"Updated mtime: {args.output.stat().st_mtime_ns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
