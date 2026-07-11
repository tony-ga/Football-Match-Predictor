from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from rich.console import Console
from rich import box
from rich.table import Table

from predicciones.src.cli.commands import list_available_fixtures
from predicciones.src.cli.match_timeline import (
    get_matches_for_team,
    load_timeline_for_match,
    render_timeline,
)
from predicciones.src.models.calibration import CalibrationManager, MarketCalibrator
from predicciones.src.models.parlay_builder import (
    RiskLevel,
    build_all_same_game_parlays,
    check_calibration,
)
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
    normalize_team_name,
)

MAX_TELEGRAM_CHARS = 3900
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    competitions = set()
    for fixture in list_available_fixtures():
        path = PROJECT_ROOT / fixture["path"]
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        comp_col = "competition" if "competition" in df.columns else "league" if "league" in df.columns else None
        if not comp_col:
            continue
        for raw in df[comp_col].dropna().astype(str).tolist():
            if raw.strip():
                competitions.add(raw.strip())
    for match in list_available_timeline_matches():
        if match.competition.strip():
            competitions.add(match.competition.strip())
    return sorted(competitions)


def list_available_timeline_matches() -> List[CompetitionMatch]:
    matches: List[CompetitionMatch] = []
    seen: set[Tuple[str, str, str, str]] = set()

    for team in available_team_names():
        try:
            team_matches = get_matches_for_team(team)
        except Exception:
            continue
        for match in team_matches:
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
                    source_path="",
                )
            )
    matches.sort(key=lambda x: (x.competition, x.date, x.home_team, x.away_team))
    return matches


def get_matches_for_competition(competition: str) -> List[CompetitionMatch]:
    normalized = competition.strip().lower()
    matches: List[CompetitionMatch] = []
    seen: set[Tuple[str, str, str, str]] = set()

    for fixture in list_available_fixtures():
        path = PROJECT_ROOT / fixture["path"]
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        comp_col = "competition" if "competition" in df.columns else "league" if "league" in df.columns else None
        if not comp_col:
            continue

        filtered = df[df[comp_col].astype(str).str.lower().str.contains(normalized, na=False, regex=False)]
        for _, row in filtered.iterrows():
            home = str(row.get("home_team") or "").strip()
            away = str(row.get("away_team") or "").strip()
            date = str(row.get("date") or fixture.get("date") or "").strip()
            key = (competition, home, away, date)
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                CompetitionMatch(
                    competition=str(row.get(comp_col) or competition),
                    home_team=home,
                    away_team=away,
                    date=date,
                    kickoff_datetime=str(row.get("kickoff_datetime") or ""),
                    neutral_venue=bool(row.get("neutral_venue", False)),
                    source_path=str(path),
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
    return get_matches_for_team(canonical_team)


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

    lines = [
        f"Prediccion: {home} vs {away}",
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
    selected_risk: Optional[str] = None,
) -> str:
    risk_map = {
        "low": RiskLevel.LOW,
        "medium": RiskLevel.MEDIUM,
        "high": RiskLevel.HIGH,
    }
    selected = [p for p in parlays if selected_risk is None or p.risk_level == risk_map[selected_risk]]

    lines = [f"Parlay: {home_team} vs {away_team}"]
    if selected_risk:
        lines.append(f"Riesgo: {selected_risk}")
    lines.append("")

    if not selected:
        lines.append("No se encontraron parlays válidos.")
        return "\n".join(lines)

    for parlay in selected:
        lines.append(f"[{parlay.risk_level.value.upper()}] ticket_score={parlay.ticket_score:.2f} combined={parlay.combined_probability:.1%}")
        if parlay.structure_evaluation is not None:
            structure = parlay.structure_evaluation
            lines.append(
                "  structure: "
                f"compatibility={structure.compatibility_score:.2f} "
                f"info_gain={structure.information_gain_score:.2f} "
                f"diversity={structure.family_diversity_score:.2f}"
            )
        for idx, pick in enumerate(parlay.picks, 1):
            evaluation = pick.market_evaluation
            if pick.calibrated_probability is not None:
                lines.append(
                    f"  {idx}. {pick.market_name} | raw={pick.model_probability:.1%} "
                    f"calib={pick.calibrated_probability:.1%}"
                )
            else:
                lines.append(f"  {idx}. {pick.market_name} | raw={pick.model_probability:.1%}")
            if evaluation and evaluation.edge is not None:
                lines.append(f"     edge={evaluation.edge:+.1%} ev={evaluation.ev:+.3f}" if evaluation.ev is not None else f"     edge={evaluation.edge:+.1%}")
        lines.append(f"  rationale: {parlay.game_script_rationale}")
        lines.append("")

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
