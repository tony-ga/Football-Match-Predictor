"""
ESPN Match Predictor and Win Probability Extraction Module.

This module provides functions to extract match win probabilities from ESPN soccer summaries.
It handles the `predictor` node (pre-match probabilities) and `winProbability` node 
(historical/in-play probability flow).

Supports:
- Extraction of homeTeamWinPercentage, awayTeamWinPercentage, tiePercentage from predictor
- Extraction of win probability flow with clock, period, play text
- Robust handling when predictor/winProbability are absent
- Normalization of probability values from various formats
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def normalize_probability(value: Any) -> Optional[float]:
    """
    Normalize a probability value to float.
    
    Handles various input formats:
    - String "54.2" or "54.2%"
    - Integer or float 54.2
    - None or empty string
    
    Args:
        value: Raw probability value from ESPN API
        
    Returns:
        Float probability value or None if invalid/missing
        
    Examples:
        >>> normalize_probability("54.2")
        54.2
        >>> normalize_probability("54.2%")
        54.2
        >>> normalize_probability(54.2)
        54.2
        >>> normalize_probability(None)
        None
    """
    if value is None:
        return None
    
    if isinstance(value, (int, float)):
        # Already numeric
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    if isinstance(value, str):
        # Remove percentage sign and whitespace
        cleaned = value.strip().replace("%", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    return None


def fetch_match_summary(event_id: str, league: str = "fifa.world") -> Dict[str, Any]:
    """
    Fetch match summary from ESPN API.
    
    Args:
        event_id: The ESPN event ID for the match
        league: League slug (e.g., 'fifa.world', 'eng.1')
        
    Returns:
        Raw JSON response from ESPN summary endpoint
        
    Raises:
        ValueError: If event_id is invalid
        Exception: If API request fails
    """
    from .espn_client_v2 import EspnClient
    
    if not event_id or not event_id.strip():
        raise ValueError("Event ID cannot be empty")
    
    client = EspnClient(league=league)
    summary = client.get_summary(event_id)
    
    if not summary:
        logger.warning(f"Empty response from ESPN for event {event_id}")
        return {}
    
    # Check for error indicators in response
    if isinstance(summary, dict) and summary.get("error"):
        logger.warning(f"ESPN returned error for event {event_id}: {summary.get('message', 'Unknown error')}")
        return {}
    
    return summary


def extract_match_context(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract basic match context from summary.
    
    Args:
        summary: Raw ESPN summary JSON
        
    Returns:
        Dict with match context: short_name, date, status, home_team, away_team, 
        home_score, away_score
    """
    if not summary:
        return {
            "short_name": "",
            "date": "",
            "status": "",
            "home_team": "",
            "away_team": "",
            "home_score": None,
            "away_score": None,
        }
    
    competitions = summary.get("competitions", [])
    if not competitions:
        return {
            "short_name": "",
            "date": "",
            "status": "",
            "home_team": "",
            "away_team": "",
            "home_score": None,
            "away_score": None,
        }
    
    comp = competitions[0]
    competitors = comp.get("competitors", [])
    
    home_comp = None
    away_comp = None
    for c in competitors:
        if c.get("homeAway") == "home":
            home_comp = c
        elif c.get("homeAway") == "away":
            away_comp = c
    
    # Fallback
    if not home_comp or not away_comp:
        if len(competitors) >= 2:
            home_comp = competitors[0]
            away_comp = competitors[1]
        else:
            home_comp = competitors[0] if competitors else {}
            away_comp = {}
    
    # Extract team info
    home_team_data = home_comp.get("team", {}) if home_comp else {}
    away_team_data = away_comp.get("team", {}) if away_comp else {}
    
    home_team = home_team_data.get("displayName", "") or home_team_data.get("name", "") or ""
    away_team = away_team_data.get("displayName", "") or away_team_data.get("name", "") or ""
    
    # Extract scores
    home_score_raw = home_comp.get("score") if home_comp else None
    away_score_raw = away_comp.get("score") if away_comp else None
    
    home_score = None
    away_score = None
    try:
        home_score = int(home_score_raw) if home_score_raw is not None else None
    except (ValueError, TypeError):
        pass
    try:
        away_score = int(away_score_raw) if away_score_raw is not None else None
    except (ValueError, TypeError):
        pass
    
    # Extract status
    status_block = comp.get("status", {})
    status_type = status_block.get("type", {})
    status_name = status_type.get("name", "") or status_type.get("state", "") or ""
    
    # Extract date
    event_date = summary.get("date", "") or summary.get("commenceDate", "")
    
    # Build short name
    short_name = f"{home_team} vs {away_team}" if home_team and away_team else ""
    
    return {
        "short_name": short_name,
        "date": event_date,
        "status": status_name,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
    }


def extract_predictor(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract predictor data from summary.
    
    Args:
        summary: Raw ESPN summary JSON
        
    Returns:
        Dict with home_team_win_percentage, away_team_win_percentage, tie_percentage
        or None if predictor doesn't exist
    """
    if not summary:
        return None
    
    predictor = summary.get("predictor")
    if not predictor:
        return None
    
    # Extract percentages
    home_win_pct = normalize_probability(predictor.get("homeTeamWinPercentage"))
    away_win_pct = normalize_probability(predictor.get("awayTeamWinPercentage"))
    tie_pct = normalize_probability(predictor.get("tiePercentage"))
    
    return {
        "home_team_win_percentage": home_win_pct,
        "away_team_win_percentage": away_win_pct,
        "tie_percentage": tie_pct,
    }


def extract_win_probability_flow(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract win probability flow from summary.
    
    Args:
        summary: Raw ESPN summary JSON
        
    Returns:
        List of dicts with sequence_index, clock_display, clock_value, period,
        home_win_percentage, away_win_percentage, tie_percentage, play_text, raw_event
    """
    if not summary:
        return []
    
    win_prob = summary.get("winProbability")
    if not win_prob or not isinstance(win_prob, list):
        return []
    
    items = []
    for idx, entry in enumerate(win_prob):
        if not isinstance(entry, dict):
            continue
        
        # Extract play info
        play = entry.get("play", {})
        
        # Clock info
        clock = play.get("clock", {}) if play else {}
        clock_display = clock.get("displayValue") if clock else None
        clock_value = clock.get("value") if clock else None
        
        # Period info
        period = None
        if play:
            period_data = play.get("period")
            if period_data:
                period = period_data.get("number") if isinstance(period_data, dict) else period_data
        
        # Probability values
        home_win_pct = normalize_probability(entry.get("homeWinPercentage"))
        away_win_pct = normalize_probability(entry.get("awayWinPercentage"))
        tie_pct = normalize_probability(entry.get("tiePercentage"))
        
        # Play text/description
        play_text = play.get("text") if play else None
        if not play_text:
            play_text = entry.get("text")
        
        items.append({
            "sequence_index": idx,
            "clock_display": clock_display,
            "clock_value": clock_value,
            "period": period,
            "home_win_percentage": home_win_pct,
            "away_win_percentage": away_win_pct,
            "tie_percentage": tie_pct,
            "play_text": play_text,
            "raw_event": entry,
        })
    
    return items


def build_match_probability_report(
    summary: Dict[str, Any],
    event_id: str,
    league: str,
    include_flow: bool = False
) -> Dict[str, Any]:
    """
    Build a complete match probability report.
    
    Args:
        summary: Raw ESPN summary JSON
        event_id: Event ID
        league: League slug
        include_flow: Whether to include win probability flow
        
    Returns:
        Structured report dict with match info, predictor, and optionally win_probability_flow
    """
    # Extract match context
    match_context = extract_match_context(summary)
    
    # Extract predictor
    predictor_data = extract_predictor(summary)
    
    # Build base report
    report = {
        "event_id": event_id,
        "league": league,
        "match": match_context,
        "predictor": {
            "available": predictor_data is not None,
            "home_team_win_percentage": predictor_data["home_team_win_percentage"] if predictor_data else None,
            "away_team_win_percentage": predictor_data["away_team_win_percentage"] if predictor_data else None,
            "tie_percentage": predictor_data["tie_percentage"] if predictor_data else None,
        } if predictor_data else {
            "available": False,
            "home_team_win_percentage": None,
            "away_team_win_percentage": None,
            "tie_percentage": None,
        },
    }
    
    # Include win probability flow if requested
    if include_flow:
        flow_items = extract_win_probability_flow(summary)
        report["win_probability_flow"] = {
            "available": len(flow_items) > 0,
            "count": len(flow_items),
            "items": flow_items,
        }
    
    return report


def print_match_probability_report(
    report: Dict[str, Any],
    include_flow: bool = False,
    limit: Optional[int] = None
) -> None:
    """
    Print a human-readable match probability report.
    
    Args:
        report: Report dict from build_match_probability_report
        include_flow: Whether to print win probability flow
        limit: Maximum number of flow items to print
    """
    match_info = report.get("match", {})
    predictor = report.get("predictor", {})
    
    # Header
    print("=" * 60)
    print("MATCH WIN PROBABILITY REPORT")
    print(match_info.get("short_name", "Unknown Match"))
    print(f"Event ID: {report.get('event_id', 'N/A')}")
    print(f"League: {report.get('league', 'N/A')}")
    print(f"Status: {match_info.get('status', 'N/A')}")
    
    # Score if available
    home_score = match_info.get("home_score")
    away_score = match_info.get("away_score")
    home_team = match_info.get("home_team", "")
    away_team = match_info.get("away_team", "")
    
    if home_score is not None and away_score is not None:
        print(f"Score: {home_team} {home_score} - {away_score} {away_team}")
    
    print("=" * 60)
    print()
    
    # Predictor section
    print("Predictor")
    if predictor.get("available"):
        home_pct = predictor.get("home_team_win_percentage")
        away_pct = predictor.get("away_team_win_percentage")
        tie_pct = predictor.get("tie_percentage")
        
        if home_pct is not None:
            print(f"  {home_team} win: {home_pct:.1f}%")
        if tie_pct is not None:
            print(f"  Draw: {tie_pct:.1f}%")
        if away_pct is not None:
            print(f"  {away_team} win: {away_pct:.1f}%")
    else:
        print("  Predictor: not available for this match")
    
    # Win probability flow section
    if include_flow:
        flow_data = report.get("win_probability_flow", {})
        print()
        print("-" * 60)
        print("Win Probability Flow")
        print("-" * 60)
        
        if flow_data.get("available") and flow_data.get("items"):
            items = flow_data["items"]
            if limit:
                items = items[:limit]
            
            for item in items:
                clock = item.get("clock_display", "")
                home_pct = item.get("home_win_percentage")
                away_pct = item.get("away_win_percentage")
                tie_pct = item.get("tie_percentage")
                play_text = item.get("play_text", "")
                
                # Format percentages
                home_str = f"{home_pct:.1f}%" if home_pct is not None else "N/A"
                away_str = f"{away_pct:.1f}%" if away_pct is not None else "N/A"
                tie_str = f"{tie_pct:.1f}%" if tie_pct is not None else "N/A"
                
                # Truncate play text if too long
                if play_text and len(play_text) > 60:
                    play_text = play_text[:57] + "..."
                
                print(f"{clock:<6} Home: {home_str:>6} | Draw: {tie_str:>6} | Away: {away_str:>6} | {play_text}")
            
            if limit and flow_data["count"] > limit:
                print(f"... and {flow_data['count'] - limit} more entries")
        else:
            print("  Win probability flow: not available for this match")
        
        print("-" * 60)


def save_report(report: Dict[str, Any], output_path: str) -> None:
    """
    Save report to JSON file.
    
    Args:
        report: Report dict to save
        output_path: Path to output file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Report saved to {output_path}")
