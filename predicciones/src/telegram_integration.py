from __future__ import annotations

import csv
import html
import json
from datetime import date, timedelta
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from predicciones.src.cli.match_timeline import (
    load_timeline_for_match,
    render_timeline,
)
from predicciones.src.data.espn_client import EspnWorldCupClient
from predicciones.src.data.espn_stats_parsers import (
    extract_events_from_summary,
    extract_player_stats_from_summary,
    extract_team_stats_from_summary,
)
from predicciones.src.models.calibration import CalibrationManager, MarketCalibrator
from predicciones.src.models.market_table import build_market_table
from predicciones.src.pipeline.predict import predict_match_pipeline
from predicciones.scripts.espn_player_stats import (
    fetch_extended_player_stats,
    format_card_timeline_csv,
    format_card_timeline_table,
    format_output_csv,
    format_output_json,
    format_output_table,
)
from predicciones.src.utils.team_normalization import (
    get_unique_teams_for_menu,
    get_jsonl_team_name,
    normalize_team_name,
)

MAX_TELEGRAM_CHARS = 3900
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DERIVED_DIR = PROJECT_ROOT / "data" / "derived"
MATCH_EVENTS_PATH = DERIVED_DIR / "match_events.jsonl"
PLAYER_STATS_PATH = DERIVED_DIR / "player_match_stats.jsonl"
STAGE_CHOICES = [
    {"label": "Fase de Grupos", "value": "group"},
    {"label": "Dieciseisavos de Final (Ronda de 32)", "value": "round_of_32"},
    {"label": "Octavos de Final (Ronda de 16)", "value": "round_of_16"},
    {"label": "Cuartos de Final", "value": "quarter_final"},
    {"label": "Semifinales (por definir)", "value": "semi_final"},
    {"label": "Partido por el Tercer Puesto (por definir)", "value": "third_place"},
    {"label": "Final (por definir)", "value": "final"},
]


@dataclass(frozen=True)
class CompetitionMatch:
    competition: str
    home_team: str
    away_team: str
    date: str = ""
    kickoff_datetime: str = ""
    neutral_venue: bool = False
    source_path: str = ""


@dataclass(frozen=True)
class TeamPlayer:
    player_name: str
    player_id: str
    position: str = ""
    games_played: int = 0
    minutes_played: int = 0
    goals: int = 0
    assists: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    total_cards: int = 0
    shots: int = 0
    shots_on_target: int = 0


def escape_html_text(text: Any) -> str:
    return html.escape("" if text is None else str(text))


@lru_cache(maxsize=8)
def _load_jsonl_cached(path_str: str) -> Tuple[Dict[str, Any], ...]:
    path = Path(path_str)
    if not path.exists():
        return tuple()
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return tuple(rows)


def _load_match_rows() -> List[Dict[str, Any]]:
    return list(_load_jsonl_cached(str(MATCH_EVENTS_PATH)))


def _load_player_rows() -> List[Dict[str, Any]]:
    return list(_load_jsonl_cached(str(PLAYER_STATS_PATH)))


def _date_range_token(days_back: int = 30, days_forward: int = 7) -> str:
    start = date.today() - timedelta(days=days_back)
    end = date.today() + timedelta(days=days_forward)
    return f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"


def build_stage_items() -> List[Dict[str, Any]]:
    return list(STAGE_CHOICES)


def _get_espn_client() -> EspnWorldCupClient:
    return EspnWorldCupClient(timeout=15)


def load_worldcup_matches(
    *,
    days_back: int = 60,
    days_forward: int = 7,
    stage: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    client = _get_espn_client()
    matches = client.get_world_cup_matches(dates=_date_range_token(days_back, days_forward), limit=500)
    filtered: List[Dict[str, Any]] = []
    stage_lower = stage.lower().strip() if stage else ""
    status_lower = status.lower().strip() if status else ""

    for match in matches:
        match_stage = str(match.get("stage") or "").lower()
        match_status = str(match.get("status") or "").lower()
        if stage_lower and stage_lower != match_stage:
            continue
        if status_lower and status_lower != match_status:
            continue
        filtered.append(match)

    filtered.sort(key=lambda m: (str(m.get("date") or ""), str(m.get("stage") or ""), str(m.get("home_team") or ""), str(m.get("away_team") or "")))
    return filtered


def build_match_items(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for idx, match in enumerate(matches):
        stage = str(match.get("stage") or "").replace("_", " ").title()
        date_value = str(match.get("date") or "")[:10]
        label = f"{match.get('home_team', 'Home')} vs {match.get('away_team', 'Away')}"
        if date_value:
            label += f" | {date_value}"
        if stage:
            label += f" | {stage}"
        items.append({"label": label, "value": idx, "match": match})
    return items


def _stage_title(stage: str) -> str:
    labels = {
        "group": "Fase de Grupos",
        "round_of_32": "Dieciseisavos de Final (Ronda de 32)",
        "round_of_16": "Octavos de Final (Ronda de 16)",
        "quarter_final": "Cuartos de Final",
        "semi_final": "Semifinales (por definir)",
        "third_place": "Partido por el Tercer Puesto (por definir)",
        "final": "Final (por definir)",
    }
    return labels.get(stage, stage.replace("_", " ").title())


def format_match_analysis_text(match: Dict[str, Any], summary: Dict[str, Any]) -> str:
    team_stats = extract_team_stats_from_summary(summary)
    events = extract_events_from_summary(summary)
    players = extract_player_stats_from_summary(summary)
    home = match.get("home_team", "Home")
    away = match.get("away_team", "Away")
    stage = _stage_title(str(match.get("stage") or ""))
    home_goals = team_stats.get("home_goals")
    away_goals = team_stats.get("away_goals")
    if home_goals is None:
        home_goals = match.get("home_score")
    if away_goals is None:
        away_goals = match.get("away_score")

    def _count_events(period: int, event_types: set[str]) -> int:
        return sum(
            1
            for event in events
            if int(event.get("period") or 0) == period and str(event.get("event_type") or "").lower() in event_types
        )

    first_half = {
        "goals": _count_events(1, {"goal", "own_goal"}),
        "yellow_cards": _count_events(1, {"yellow_card"}),
        "red_cards": _count_events(1, {"red_card"}),
        "corners": _count_events(1, {"corner"}),
    }
    second_half = {
        "goals": _count_events(2, {"goal", "own_goal"}),
        "yellow_cards": _count_events(2, {"yellow_card"}),
        "red_cards": _count_events(2, {"red_card"}),
        "corners": _count_events(2, {"corner"}),
    }

    top_scorers = [
        p for p in sorted(players, key=lambda p: float(p.get("goals") or 0), reverse=True)
        if str(p.get("player_name") or "").strip() and float(p.get("goals") or 0) > 0
    ][:5]
    top_shots = [
        p for p in sorted(players, key=lambda p: float(p.get("shots") or 0), reverse=True)
        if str(p.get("player_name") or "").strip() and float(p.get("shots") or 0) > 0
    ][:5]
    top_cards = [
        p for p in sorted(players, key=lambda p: float(p.get("total_cards") or 0), reverse=True)
        if str(p.get("player_name") or "").strip() and float(p.get("total_cards") or 0) > 0
    ][:5]

    lines = [
        f"📊 Análisis de Partidos",
        f"{stage}",
        f"{home} vs {away}",
        f"Fecha: {str(match.get('date') or '')[:10]}",
        "",
        "Resumen de equipo",
        f"  Goles         : {home_goals if home_goals is not None else 'N/A'} - {away_goals if away_goals is not None else 'N/A'}",
        f"  Tiros         : {team_stats.get('home_shots', 'N/A')} - {team_stats.get('away_shots', 'N/A')}",
        f"  A puerta      : {team_stats.get('home_shots_on_target', 'N/A')} - {team_stats.get('away_shots_on_target', 'N/A')}",
        f"  Corners       : {team_stats.get('home_corners', 'N/A')} - {team_stats.get('away_corners', 'N/A')}",
        f"  Tarjetas amarillas: {team_stats.get('home_yellow_cards', 'N/A')} - {team_stats.get('away_yellow_cards', 'N/A')}",
        f"  Tarjetas rojas    : {team_stats.get('home_red_cards', 'N/A')} - {team_stats.get('away_red_cards', 'N/A')}",
        f"  Faltas        : {team_stats.get('home_fouls', 'N/A')} - {team_stats.get('away_fouls', 'N/A')}",
        f"  Offsides      : {team_stats.get('home_offsides', 'N/A')} - {team_stats.get('away_offsides', 'N/A')}",
        f"  Posesion      : {team_stats.get('home_possession', 'N/A')} - {team_stats.get('away_possession', 'N/A')}",
        "",
        "Desglose por tiempo",
        f"  1T goles      : {first_half['goals']}",
        f"  1T corners    : {first_half['corners']}",
        f"  1T amarillas  : {first_half['yellow_cards']}",
        f"  1T rojas      : {first_half['red_cards']}",
        f"  2T goles      : {second_half['goals']}",
        f"  2T corners    : {second_half['corners']}",
        f"  2T amarillas  : {second_half['yellow_cards']}",
        f"  2T rojas      : {second_half['red_cards']}",
    ]

    if top_scorers:
        lines.extend(["", "Top anotadores"])
        for player in top_scorers:
            lines.append(f"  - {player.get('player_name', 'N/A')}: {int(player.get('goals') or 0)}")

    if top_shots:
        lines.extend(["", "Top tiros"])
        for player in top_shots:
            lines.append(f"  - {player.get('player_name', 'N/A')}: {int(player.get('shots') or 0)}")

    if top_cards:
        lines.extend(["", "Top tarjetas"])
        for player in top_cards:
            lines.append(f"  - {player.get('player_name', 'N/A')}: {int(player.get('total_cards') or 0)}")

    return "\n".join(lines)


def split_telegram_message(text: str, limit: int = MAX_TELEGRAM_CHARS) -> List[str]:
    """Split long plain-text messages on line boundaries."""
    if not text:
        return [""]

    chunks: List[str] = []
    current: List[str] = []
    current_size = 0

    for line in text.splitlines():
        line_with_newline = line + "\n"
        if current and current_size + len(line_with_newline) > limit:
            chunks.append("".join(current).rstrip())
            current = []
            current_size = 0

        if len(line_with_newline) > limit:
            if current:
                chunks.append("".join(current).rstrip())
                current = []
                current_size = 0
            for start in range(0, len(line_with_newline), limit):
                chunks.append(line_with_newline[start:start + limit].rstrip())
            continue

        current.append(line_with_newline)
        current_size += len(line_with_newline)

    if current:
        chunks.append("".join(current).rstrip())

    return chunks or [text[:limit]]


def available_team_names() -> List[str]:
    teams = get_unique_teams_for_menu()
    return [display for display, _ in teams] if teams else []


def available_competitions() -> List[str]:
    competitions = {
        str(row.get("competition") or "").strip()
        for row in _load_match_rows()
        if str(row.get("competition") or "").strip()
    }
    if not competitions:
        competitions = {
            str(row.get("competition") or "").strip()
            for row in _load_player_rows()
            if str(row.get("competition") or "").strip()
        }
    return sorted(competitions)


def list_available_timeline_matches() -> List[CompetitionMatch]:
    matches: List[CompetitionMatch] = []
    seen: set[Tuple[str, str, str, str]] = set()

    for match in _load_match_rows():
        competition = str(match.get("competition") or "").strip()
        home = str(match.get("home_team") or "").strip()
        away = str(match.get("away_team") or "").strip()
        date = str(match.get("date") or "").strip()
        key = (competition, home, away, date)
        if key in seen:
            continue
        seen.add(key)
        matches.append(
            CompetitionMatch(
                competition=competition,
                home_team=home,
                away_team=away,
                date=date,
                kickoff_datetime=str(match.get("kickoff_datetime") or ""),
                source_path="derived",
            )
        )
    matches.sort(key=lambda x: (x.competition, x.date, x.home_team, x.away_team))
    return matches


def get_matches_for_competition(competition: str) -> List[CompetitionMatch]:
    normalized = competition.strip().lower()
    matches: List[CompetitionMatch] = []
    seen: set[Tuple[str, str, str, str]] = set()

    for row in _load_match_rows():
        row_competition = str(row.get("competition") or "").strip()
        if not row_competition or normalized not in row_competition.lower():
            continue
        home = str(row.get("home_team") or "").strip()
        away = str(row.get("away_team") or "").strip()
        date = str(row.get("date") or "").strip()
        key = (row_competition, home, away, date)
        if key in seen:
            continue
        seen.add(key)
        matches.append(
            CompetitionMatch(
                competition=row_competition,
                home_team=home,
                away_team=away,
                date=date,
                kickoff_datetime=str(row.get("kickoff_datetime") or ""),
                neutral_venue=bool(row.get("neutral_venue", False)),
                source_path="derived",
            )
        )

    matches.sort(key=lambda x: (x.date, x.home_team, x.away_team))
    return matches


def get_teams_for_competition(competition: str) -> List[str]:
    teams = set()
    for match in get_matches_for_competition(competition):
        if match.home_team:
            teams.add(match.home_team)
        if match.away_team:
            teams.add(match.away_team)
    return sorted(teams)


def get_player_rows_for_team(team_name: str, max_matches: int = 10) -> List[Dict[str, Any]]:
    canonical_team = normalize_team_name(team_name)
    return fetch_extended_player_stats(canonical_team, mode="summary", max_matches=max_matches)


def get_player_cards_for_team(team_name: str, max_matches: int = 10) -> List[Dict[str, Any]]:
    canonical_team = normalize_team_name(team_name)
    return fetch_extended_player_stats(canonical_team, mode="cards", max_matches=max_matches)


def get_player_timeline_rows_for_team(team_name: str, max_matches: int = 10) -> List[Dict[str, Any]]:
    canonical_team = normalize_team_name(team_name)
    return fetch_extended_player_stats(canonical_team, mode="timeline", max_matches=max_matches)


def get_team_matches(team_name: str) -> List[Dict[str, Any]]:
    canonical_team = normalize_team_name(team_name)
    jsonl_team = get_jsonl_team_name(canonical_team)
    candidates = {
        str(team_name or "").strip().lower(),
        canonical_team.lower(),
        jsonl_team.lower(),
    }
    matches: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, str]] = set()

    for row in _load_match_rows():
        home = str(row.get("home_team") or "").strip()
        away = str(row.get("away_team") or "").strip()
        if not candidates.intersection({home.lower(), away.lower()}):
            continue
        competition = str(row.get("competition") or "").strip()
        date = str(row.get("date") or "").strip()
        key = (competition, home, away, date)
        if key in seen:
            continue
        seen.add(key)
        matches.append(dict(row))

    if not matches:
        for row in _load_player_rows():
            row_team = str(row.get("team") or "").strip().lower()
            if row_team not in candidates:
                continue
            event_id = str(row.get("event_id") or "")
            if not event_id:
                continue
            # Player rows do not carry the full match payload, so keep a minimal fallback.
            matches.append(
                {
                    "event_id": event_id,
                    "date": row.get("date") or "",
                    "competition": row.get("competition") or "",
                    "league_slug": row.get("league_slug") or "fifa.world",
                    "home_team": row.get("team") or "",
                    "away_team": row.get("opponent") or "",
                    "home_score": None,
                    "away_score": None,
                }
            )

    matches.sort(key=lambda m: (m.get("date") or "", m.get("home_team") or "", m.get("away_team") or ""))
    return matches


def get_competition_teams_and_matches(competition: str) -> Tuple[List[str], List[CompetitionMatch]]:
    matches = get_matches_for_competition(competition)
    teams = sorted({m.home_team for m in matches if m.home_team} | {m.away_team for m in matches if m.away_team})
    return teams, matches


def load_calibration_manager() -> CalibrationManager:
    manager = CalibrationManager()
    calib_dir = PROJECT_ROOT / "output" / "calibrators"
    expected_calibrators = {
        "1x2": "cal_1x2.pkl",
        "btts": "cal_btts.pkl",
        "over_under_15": "cal_ou15.pkl",
        "over_under_25": "cal_ou25.pkl",
        "over_under_35": "cal_ou35.pkl",
    }

    for market_name, filename in expected_calibrators.items():
        calib_path = calib_dir / filename
        if not calib_path.exists():
            continue
        try:
            manager.add_calibrator(market_name, MarketCalibrator.load(calib_path))
        except Exception:
            continue
    return manager


def format_prediction_text(response: Dict[str, Any]) -> str:
    home = response.get("home_team", "Home")
    away = response.get("away_team", "Away")
    predictions = response.get("predictions", {})
    x2 = predictions.get("1x2", {})
    btts = predictions.get("btts", {})
    ou = predictions.get("over_under", {})
    expected = predictions.get("expected_goals", {})
    team_context = response.get("team_context", {})
    odds = response.get("sportsbook_odds", {})
    warnings = response.get("data_freshness", {}).get("warnings") or []
    correct_scores = predictions.get("correct_scores", []) or []
    halftime = predictions.get("halftime", {}) or {}
    first_half_goals = predictions.get("first_half_goals", {}) or {}
    second_half_goals = predictions.get("second_half_goals", {}) or {}
    corners = response.get("markets", {}).get("corners", {}) or {}
    player_props = response.get("markets", {}).get("player_props", {}) or {}
    anytime = player_props.get("anytime_scorer", {}) if isinstance(player_props, dict) else {}
    first = player_props.get("first_scorer", {}) if isinstance(player_props, dict) else {}

    def _fmt_distribution(distribution: Any, label: str) -> List[str]:
        rows: List[str] = []
        if not isinstance(distribution, list):
            return rows
        rows.append(label)
        for item in distribution[:4]:
            goal = item.get("goals", "?")
            prob = float(item.get("probability", 0))
            rows.append(f"  - {goal}: {prob:.1%}")
        return rows

    lines = [
        f"🔮 Prediccion: {home} vs {away}",
        "",
        "Resumen",
        f"  1X2 home   : {float(x2.get('home', 0)):.1%}",
        f"  1X2 draw   : {float(x2.get('draw', 0)):.1%}",
        f"  1X2 away   : {float(x2.get('away', 0)):.1%}",
        f"  BTTS yes   : {float(btts.get('yes', 0)):.1%}",
        f"  Over 2.5   : {float(ou.get('over_2_5', 0)):.1%}",
        f"  Under 2.5  : {float(ou.get('under_2_5', 0)):.1%}",
        f"  xG total   : {float(expected.get('total', 0)):.2f}",
    ]

    if halftime:
        lines.extend([
            "",
            "Descanso",
            f"  Home HT  : {float(halftime.get('home', 0)):.1%}",
            f"  Draw HT  : {float(halftime.get('draw', 0)):.1%}",
            f"  Away HT  : {float(halftime.get('away', 0)):.1%}",
        ])

    if first_half_goals.get("expected_goals"):
        fh = first_half_goals["expected_goals"]
        lines.extend([
            "",
            "Primer tiempo",
            f"  xG total : {float(fh.get('total', 0)):.2f}",
            f"  Over 1.5 : {float(first_half_goals.get('over_under', {}).get('over_1_5', 0)):.1%}",
            f"  Under 1.5: {float(first_half_goals.get('over_under', {}).get('under_1_5', 0)):.1%}",
        ])

    if second_half_goals.get("expected_goals"):
        sh = second_half_goals["expected_goals"]
        lines.extend([
            "",
            "Segundo tiempo",
            f"  xG total : {float(sh.get('total', 0)):.2f}",
            f"  Over 1.5 : {float(second_half_goals.get('over_under', {}).get('over_1_5', 0)):.1%}",
            f"  Under 1.5: {float(second_half_goals.get('over_under', {}).get('under_1_5', 0)):.1%}",
        ])

    if corners.get("available"):
        periods = corners.get("periods", {}) or {}
        lines.extend([
            "",
            "Corners",
            f"  FT esperado : {float(corners.get('expected_total', 0)):.2f}",
            f"  1T esperado : {float(periods.get('expected_first_half_total', 0)):.2f}",
            f"  2T esperado : {float(periods.get('expected_second_half_total', 0)):.2f}",
            f"  Home 1T     : {float(periods.get('team_expected', {}).get('home', {}).get('first_half', 0)):.2f}",
            f"  Away 1T     : {float(periods.get('team_expected', {}).get('away', {}).get('first_half', 0)):.2f}",
        ])

    if correct_scores:
        lines.extend(["", "Marcadores probables"])
        for score in correct_scores[:5]:
            lines.append(
                f"  - {score.get('score', 'N/A'):<5} {float(score.get('probability', 0)):.1%}"
            )

    home_dist = predictions.get("home_goals_distribution") or []
    away_dist = predictions.get("away_goals_distribution") or []
    if home_dist:
        lines.extend(["", "Distribucion de goles - local"])
        for row in home_dist[:5]:
            lines.append(f"  - {row.get('goals', '?')}: {float(row.get('probability', 0)):.1%}")
    if away_dist:
        lines.extend(["", "Distribucion de goles - visitante"])
        for row in away_dist[:5]:
            lines.append(f"  - {row.get('goals', '?')}: {float(row.get('probability', 0)):.1%}")

    if team_context:
        lines.extend(["", "Contexto"])
        for side in ("home", "away"):
            side_data = team_context.get(side, {})
            lines.append(
                f"  {side_data.get('team') or side.title():<10}: "
                f"lambda_attack={float(side_data.get('lambda_attack', 0)):.3f} "
                f"lambda_defense={float(side_data.get('lambda_defense', 0)):.3f} "
                f"source={side_data.get('data_source', 'N/A')}"
            )

    if odds.get("notes"):
        lines.extend(["", "Notas"])
        for note in odds["notes"]:
            lines.append(f"  - {note}")

    if warnings:
        lines.extend(["", "Alertas"])
        for warning in warnings:
            lines.append(f"  - {warning}")

    scorer_rows: List[str] = []
    if anytime.get("available") and anytime.get("top_candidates"):
        scorer_rows.append("Posibles anotadores")
        for candidate in anytime["top_candidates"][:5]:
            name = candidate.get("player_name", "N/A")
            prob = candidate.get("probability_pct", candidate.get("probability", 0))
            if isinstance(prob, (int, float)) and prob <= 1:
                prob = prob * 100
            scorer_rows.append(f"  - {name:<24} {float(prob):.1f}%")
    if first.get("available") and first.get("top_candidates"):
        scorer_rows.extend(["", "Primer anotador"])
        for candidate in first["top_candidates"][:5]:
            name = candidate.get("player_name", "N/A")
            prob = candidate.get("probability_pct", candidate.get("probability", 0))
            if isinstance(prob, (int, float)) and prob <= 1:
                prob = prob * 100
            scorer_rows.append(f"  - {name:<24} {float(prob):.1f}%")
    if scorer_rows:
        lines.extend([""] + scorer_rows)

    if response.get("markets"):
        lines.extend(["", "Mercados alternativos"])
        for key in ("corners", "cards", "shots_on_target", "player_props"):
            market = response["markets"].get(key, {})
            if market.get("available"):
                lines.append(f"  - {key.replace('_', ' ').title()}: disponible")
            else:
                lines.append(f"  - {key.replace('_', ' ').title()}: {market.get('reason', 'no disponible')}")

    return "\n".join(lines)


def format_parlay_text(
    home_team: str,
    away_team: str,
    pred_data: Dict[str, Any],
    parlays: List[Any],
) -> str:
    lines = [f"🧩 Mercados: {home_team} vs {away_team}", ""]

    table_rows = build_market_table(pred_data, limit=18)
    if not table_rows:
        lines.append("No se encontraron mercados válidos.")
        return "\n".join(lines)

    lines.append("Mercado | Prob. | Riesgo | Periodo")
    lines.append("-" * 46)
    for row in table_rows:
        lines.append(
            f"{row.label[:24]:<24} | "
            f"{row.probability:>5.1%} | "
            f"{row.risk:<8} | "
            f"{row.period}"
        )
        if row.rationale:
            lines.append(f"  -> {row.rationale}")
        if row.edge is not None or row.ev is not None:
            edge = f"{row.edge:+.1%}" if row.edge is not None else "-"
            ev = f"{row.ev:+.3f}" if row.ev is not None else "-"
            lines.append(f"  -> edge={edge} ev={ev}")
    return "\n".join(lines).strip()


def format_timeline_text(team_name: str, match: Dict[str, Any]) -> str:
    timeline = load_timeline_for_match(match)
    return render_timeline(timeline)


def format_player_team_summary(team_name: str, max_matches: int = 10) -> str:
    canonical_team = normalize_team_name(team_name)
    data = fetch_extended_player_stats(canonical_team, mode="summary", max_matches=max_matches)
    if not data:
        return f"No player data found for {canonical_team}"
    title = f"Datos de jugadores: {canonical_team}"
    return title + "\n\n" + format_output_table(data, mode="roster")


def format_player_detail(team_name: str, player_name: str, max_matches: int = 10) -> str:
    canonical_team = normalize_team_name(team_name)
    summary_rows = fetch_extended_player_stats(canonical_team, mode="summary", max_matches=max_matches)
    if not summary_rows:
        return f"No player data found for {canonical_team}"

    player = None
    for row in summary_rows:
        if str(row.get("player_name", "")).lower() == player_name.lower():
            player = row
            break
    if not player:
        return f"No data found for player: {player_name}"

    lines = [
        f"Jugador: {player.get('player_name', 'N/A')}",
        f"Equipo: {player.get('team_name', canonical_team)}",
        f"Posicion: {player.get('position', 'N/A')}",
        "",
        "Resumen",
        f"  GP      : {player.get('games_played', 0)}",
        f"  Min     : {player.get('minutes_played', 0)}",
        f"  Goles   : {player.get('goals', 0)}",
        f"  Asist   : {player.get('assists', 0)}",
        f"  YC      : {player.get('yellow_cards', 0)}",
        f"  RC      : {player.get('red_cards', 0)}",
        f"  Tiros   : {player.get('shots', 0)}",
        f"  A puerta: {player.get('shots_on_target', 0)}",
    ]

    cards = fetch_extended_player_stats(canonical_team, mode="cards", max_matches=max_matches)
    detail_cards = [card for card in cards if str(card.get("player_name", "")).lower() == player_name.lower()]
    if detail_cards:
        lines.extend(["", "Tarjetas"])
        lines.append(f"  Amarillas: {sum(int(c.get('yellow_cards', 0)) for c in detail_cards)}")
        lines.append(f"  Rojas    : {sum(int(c.get('red_cards', 0)) for c in detail_cards)}")

    return "\n".join(lines)


def render_player_summary_table(data: List[Dict[str, Any]]) -> str:
    return format_output_table(data, mode="roster")


def render_player_cards_table(data: List[Dict[str, Any]]) -> str:
    return format_card_timeline_table(data)


def render_player_cards_csv(data: List[Dict[str, Any]]) -> str:
    return format_card_timeline_csv(data)


def render_player_json(data: List[Dict[str, Any]]) -> str:
    return format_output_json(data)


def render_team_summary_csv(data: List[Dict[str, Any]]) -> str:
    return format_output_csv(data, mode="roster")
