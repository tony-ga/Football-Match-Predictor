"""
Guided Match Timeline CLI.

This module keeps timeline responsibilities separate:
- discover teams
- discover matches for a team
- load timeline events for a selected internal event_id
- render a terminal-friendly timeline
- run the interactive menu controller
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_DIR = PROJECT_ROOT / "data" / "derived"
PLAYER_STATS_PATH = DERIVED_DIR / "player_match_stats.jsonl"
MATCH_EVENTS_PATH = DERIVED_DIR / "match_events.jsonl"
PRIMARY_EVENT_TYPES = {
    "goal",
    "own_goal",
    "yellow_card",
    "red_card",
    "substitution",
    "kickoff",
    "halftime",
    "half_time",
    "fulltime",
    "full_time",
    "penalty",
    "penalty_goal",
    "penalty_missed",
}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _fmt_date(date_value: Optional[str]) -> str:
    if not date_value:
        return "N/A"
    return str(date_value)[:10]


def _fmt_score(value: Any) -> str:
    if value is None:
        return "?"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric == int(numeric):
        return str(int(numeric))
    return str(numeric)


def _event_sort_key(event: Dict[str, Any]) -> Tuple[float, int]:
    clock_value = event.get("clock_value")
    if clock_value is None:
        minute = event.get("minute")
        clock_value = float(minute or 0) * 60
    event_type = (event.get("event_type") or event.get("type") or "").lower()
    if event_type in {"halftime", "half_time"} and not clock_value:
        clock_value = 45 * 60
    elif event_type in {"fulltime", "full_time"} and not clock_value:
        clock_value = 999 * 60
    return (float(clock_value or 0), int(event.get("sequence_index") or 0))


def _event_label(event_type: str) -> str:
    labels = {
        "goal": "GOAL",
        "own_goal": "OWN GOAL",
        "yellow_card": "YELLOW CARD",
        "red_card": "RED CARD",
        "substitution": "SUBSTITUTION",
        "kickoff": "KICKOFF",
        "halftime": "HALF TIME",
        "fulltime": "FULL TIME",
        "penalty": "PENALTY",
        "penalty_goal": "PENALTY GOAL",
        "penalty_missed": "PENALTY MISSED",
    }
    return labels.get((event_type or "event").lower(), (event_type or "event").replace("_", " ").upper())


def get_available_timeline_teams() -> List[str]:
    """Return all teams with available match rows, sorted alphabetically."""
    teams = {row.get("team") for row in _load_jsonl(PLAYER_STATS_PATH) if row.get("team")}
    if not teams:
        for row in _load_jsonl(MATCH_EVENTS_PATH):
            if row.get("home_team"):
                teams.add(row["home_team"])
            if row.get("away_team"):
                teams.add(row["away_team"])
    return sorted(teams)


def get_matches_for_team(team_name: str) -> List[Dict[str, Any]]:
    """Return readable match metadata for a team. event_id remains internal."""
    rows = _load_jsonl(PLAYER_STATS_PATH)
    by_event: Dict[str, Dict[str, Any]] = {}
    scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for row in rows:
        event_id = str(row.get("event_id") or "")
        team = row.get("team") or ""
        if not event_id or not team:
            continue
        scores[event_id][team] += float(row.get("goals") or 0)

        entry = by_event.setdefault(event_id, {
            "event_id": event_id,
            "date": row.get("date") or "",
            "competition": row.get("competition") or "",
            "league_slug": row.get("league_slug") or "fifa.world",
            "home_team": "",
            "away_team": "",
            "home_score": None,
            "away_score": None,
        })
        if row.get("home_or_away") == "home":
            entry["home_team"] = team
        elif row.get("home_or_away") == "away":
            entry["away_team"] = team

    matches = []
    for event_id, match in by_event.items():
        if team_name.lower() not in {
            (match.get("home_team") or "").lower(),
            (match.get("away_team") or "").lower(),
        }:
            continue
        home = match.get("home_team") or "Home"
        away = match.get("away_team") or "Away"
        match["home_score"] = scores[event_id].get(home)
        match["away_score"] = scores[event_id].get(away)
        matches.append(match)

    if not matches:
        for row in _load_jsonl(MATCH_EVENTS_PATH):
            if team_name not in {row.get("home_team"), row.get("away_team")}:
                continue
            matches.append({
                "event_id": str(row.get("event_id") or ""),
                "date": row.get("date") or "",
                "competition": row.get("competition") or "",
                "league_slug": row.get("league_slug") or "fifa.world",
                "home_team": row.get("home_team") or "",
                "away_team": row.get("away_team") or "",
                "home_score": None,
                "away_score": None,
            })

    matches.sort(key=lambda m: (m.get("date") or "", m.get("home_team") or "", m.get("away_team") or ""))
    return matches


def _load_derived_timeline(match: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    event_id = str(match.get("event_id") or "")
    for record in _load_jsonl(MATCH_EVENTS_PATH):
        if str(record.get("event_id") or "") != event_id:
            continue
        events = record.get("events") or []
        return {
            "event_id": event_id,
            "league": match.get("league_slug") or record.get("league_slug") or "fifa.world",
            "match": {
                "short_name": f"{match.get('home_team')} vs {match.get('away_team')}",
                "date": match.get("date") or record.get("date"),
                "status": "STATUS_FINAL",
                "home_team": match.get("home_team") or record.get("home_team"),
                "away_team": match.get("away_team") or record.get("away_team"),
                "home_score": match.get("home_score"),
                "away_score": match.get("away_score"),
                "competition": match.get("competition") or record.get("competition"),
            },
            "sources": {
                "used_source": "derived_jsonl",
                "total_events": len(events),
            },
            "events": events,
        }
    return None


def _fetch_espn_timeline(match: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        from predicciones.src.data.espn_match_events import get_match_event_timeline

        timeline = get_match_event_timeline(
            event_id=str(match["event_id"]),
            league=match.get("league_slug") or "fifa.world",
            prefer_source="commentary",
        )
        timeline.setdefault("match", {})
        timeline["match"]["competition"] = match.get("competition") or timeline.get("league")
        return timeline
    except Exception:
        return None


def load_timeline_for_match(match: Dict[str, Any]) -> Dict[str, Any]:
    """Load timeline events; use derived JSONL first, ESPN fallback when needed."""
    derived = _load_derived_timeline(match)
    score_total = float(match.get("home_score") or 0) + float(match.get("away_score") or 0)
    derived_events = derived.get("events", []) if derived else []
    has_goal_events = any((event.get("event_type") or "").lower() in {"goal", "own_goal"} for event in derived_events)

    if derived and (has_goal_events or score_total == 0):
        return derived

    fetched = _fetch_espn_timeline(match)
    if fetched and fetched.get("events"):
        fetched["match"].update({
            "date": match.get("date") or fetched.get("match", {}).get("date"),
            "home_team": match.get("home_team") or fetched.get("match", {}).get("home_team"),
            "away_team": match.get("away_team") or fetched.get("match", {}).get("away_team"),
            "home_score": match.get("home_score"),
            "away_score": match.get("away_score"),
            "competition": match.get("competition") or fetched.get("league"),
        })
        return fetched

    return derived or {
        "event_id": match.get("event_id"),
        "league": match.get("league_slug") or "fifa.world",
        "match": match,
        "sources": {"used_source": "none", "total_events": 0},
        "events": [],
    }


def render_timeline(timeline: Dict[str, Any]) -> str:
    """Render a readable plain-text timeline for terminal output."""
    match = timeline.get("match", {})
    competition = match.get("competition") or timeline.get("league") or "Competition N/A"
    date = _fmt_date(match.get("date"))
    home = match.get("home_team") or "Home"
    away = match.get("away_team") or "Away"
    home_score = _fmt_score(match.get("home_score"))
    away_score = _fmt_score(match.get("away_score"))

    lines = [
        "-" * 72,
        "Match Timeline",
        f"{competition} - {date}",
        f"{home} vs {away}",
        f"Final Score: {home} {home_score}-{away_score} {away}",
        "",
    ]

    raw_events = timeline.get("events") or []
    primary_events = [
        event for event in raw_events
        if (event.get("event_type") or event.get("type") or "").lower() in PRIMARY_EVENT_TYPES
    ]
    events = sorted(primary_events or raw_events, key=_event_sort_key)
    deduped_events = []
    seen = set()
    for event in events:
        event_type = (event.get("event_type") or event.get("type") or "").lower()
        if (
            event_type in {"fulltime", "full_time"}
            and not event.get("minute")
            and any((other.get("event_type") or other.get("type") or "").lower() in {"fulltime", "full_time"} and other.get("minute") for other in events)
        ):
            continue
        signature = (
            event.get("clock_display") or event.get("minute"),
            event.get("event_type") or event.get("type"),
            event.get("team_name") or event.get("team_abbr"),
            event.get("player_name"),
            event.get("description") or event.get("text"),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped_events.append(event)
    events = deduped_events
    if not events:
        lines.append("No timeline events available for this match")
    for event in events:
        minute = event.get("clock_display") or (f"{event.get('minute')}'" if event.get("minute") is not None else "")
        event_type = _event_label(event.get("event_type") or event.get("type") or "")
        team = event.get("team_name") or event.get("team_abbr") or ""
        player = event.get("player_name") or ""
        description = event.get("description") or event.get("text") or ""
        detail = player or description
        if event_type == "SUBSTITUTION" and description:
            detail = description
        lines.append(f"{minute:>7}  {event_type:<13} {team:<14} {detail}")

    lines.append("-" * 72)
    return "\n".join(lines)


def _ask_navigation(prompt: str, max_index: int, allow_back: bool = True) -> str:
    while True:
        choice = Prompt.ask(prompt).strip()
        upper = choice.upper()
        if upper == "Q":
            return "Q"
        if allow_back and upper == "B":
            return "B"
        if choice.isdigit() and 1 <= int(choice) <= max_index:
            return choice
        back_text = "B to go back, " if allow_back else ""
        print(f"Invalid option. Enter 1-{max_index}, {back_text}or Q to quit.")


def run_match_timeline_menu(console: Optional[Console] = None) -> None:
    """Interactive controller for module 4."""
    active_console = console or Console()

    while True:
        active_console.print()
        active_console.print(Panel(
            "[bold]Match Timeline[/bold]\n\n"
            "Select a team, choose one of its available matches, then view a clean timeline.\n\n"
            "  [cyan]1.[/cyan] Choose team\n"
            "  [cyan]Q.[/cyan] Exit",
            title="Match Timeline",
            border_style="blue",
        ))
        choice = Prompt.ask("[cyan]Choose an option[/cyan]").strip().upper()
        if choice == "Q":
            return
        if choice != "1":
            active_console.print("[yellow]Invalid option. Please choose 1 or Q.[/yellow]")
            continue

        teams = get_available_timeline_teams()
        if not teams:
            active_console.print("[yellow]No teams found with timeline data.[/yellow]")
            continue

        while True:
            table = Table(title="Available Teams", show_header=True, header_style="bold magenta")
            table.add_column("#", justify="right", style="cyan")
            table.add_column("Team", style="white")
            for idx, team in enumerate(teams, 1):
                table.add_row(str(idx), team)
            active_console.print(table)
            active_console.print("[dim]B = back, Q = exit[/dim]")

            team_choice = _ask_navigation(f"Select team (1-{len(teams)})", len(teams))
            if team_choice == "Q":
                return
            if team_choice == "B":
                break

            selected_team = teams[int(team_choice) - 1]
            matches = get_matches_for_team(selected_team)
            if not matches:
                active_console.print(f"[yellow]No matches found for {selected_team}[/yellow]")
                continue

            while True:
                table = Table(title=f"{selected_team} Matches", show_header=True, header_style="bold magenta")
                table.add_column("#", justify="right", style="cyan")
                table.add_column("Match", style="white")
                table.add_column("Date", style="dim")
                table.add_column("Competition", style="dim")
                table.add_column("Score", justify="center")
                for idx, match in enumerate(matches, 1):
                    home = match.get("home_team") or "Home"
                    away = match.get("away_team") or "Away"
                    score = f"{_fmt_score(match.get('home_score'))}-{_fmt_score(match.get('away_score'))}"
                    table.add_row(
                        str(idx),
                        f"{home} vs {away}",
                        _fmt_date(match.get("date")),
                        match.get("competition") or "N/A",
                        score,
                    )
                active_console.print(table)
                active_console.print("[dim]B = back to teams, Q = exit[/dim]")

                match_choice = _ask_navigation(f"Select match (1-{len(matches)})", len(matches))
                if match_choice == "Q":
                    return
                if match_choice == "B":
                    break

                selected_match = matches[int(match_choice) - 1]
                timeline = load_timeline_for_match(selected_match)
                active_console.print()
                active_console.print(render_timeline(timeline))
                while True:
                    nav = Prompt.ask("[cyan]B = back to matches, Q = exit[/cyan]").strip().upper()
                    if nav == "Q":
                        return
                    if nav == "B":
                        break
                    active_console.print("[yellow]Invalid option. Enter B or Q.[/yellow]")
