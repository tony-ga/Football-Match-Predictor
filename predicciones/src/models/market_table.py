from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .market_catalog import build_market_catalog
from .market_evaluation import EvaluatedMarket


@dataclass(frozen=True)
class MarketRow:
    market_key: str
    label: str
    probability: float
    risk: str
    theme: str
    period: str
    family: str
    rationale: str
    edge: Optional[float] = None
    ev: Optional[float] = None


def _risk_band(probability: float, theme: str) -> Tuple[str, str]:
    if theme in {"correct_score", "more_corners", "more_cards"}:
        if probability >= 0.48:
            return "🟡", "Medio"
        return "🔴", "Alto"
    if probability >= 0.70:
        return "🟢", "Bajo"
    if probability >= 0.56:
        return "🟡", "Medio"
    return "🔴", "Alto"


def _theme_for_market_key(market_key: str) -> Tuple[str, str, str]:
    if market_key.startswith("1x2_"):
        return "1x2", "result", "FT"
    if market_key.startswith("double_chance_"):
        return "double_chance", "result", "FT"
    if market_key.startswith("btts_"):
        return "btts", "btts", "FT"
    if market_key.startswith("first_half_"):
        return "goals_first_half", "totals", "1T"
    if market_key.startswith("second_half_"):
        return "goals_second_half", "totals", "2T"
    if market_key.startswith("corners_first_half_"):
        return "corners_first_half", "corners", "1T"
    if market_key.startswith("corners_second_half_"):
        return "corners_second_half", "corners", "2T"
    if market_key.startswith("corners_"):
        return "corners_full_time", "corners", "FT"
    if market_key.startswith("cards_"):
        return "cards_full_time", "cards", "FT"
    if market_key.startswith("shots_"):
        return "shots_full_time", "shots", "FT"
    if market_key.startswith("player_shots_"):
        return "player_shots", "player", "FT"
    if market_key.startswith("over_") or market_key.startswith("under_"):
        return "goals_full_time", "totals", "FT"
    return "other", "other", "FT"


def _pretty_label(market_key: str, fallback: str) -> str:
    replacements = {
        "1x2_home": "Victoria local",
        "1x2_draw": "Empate",
        "1x2_away": "Victoria visitante",
        "double_chance_home_or_draw": "Local o empate",
        "double_chance_away_or_draw": "Visitante o empate",
        "double_chance_home_or_away": "Sin empate",
        "btts_yes": "Ambos anotan: Sí",
        "btts_no": "Ambos anotan: No",
    }
    if market_key in replacements:
        return replacements[market_key]
    return fallback


def _find_evaluation(evaluations: List[EvaluatedMarket], market_key: str) -> Optional[EvaluatedMarket]:
    for evaluation in evaluations:
        if evaluation.market_key == market_key:
            return evaluation
    return None


def _line_value(label: str) -> str:
    for token in ("0.5", "1.5", "2.5", "3.5", "4.5", "5.5", "6.5", "7.5", "8.5", "9.5", "10.5"):
        if token in label:
            return token
    return ""


def _add_candidate(
    rows: List[MarketRow],
    seen: set,
    *,
    market_key: str,
    label: str,
    probability: Optional[float],
    theme: str,
    family: str,
    period: str,
    rationale: str,
    edge: Optional[float] = None,
    ev: Optional[float] = None,
):
    if probability is None:
        return
    key = (theme, period, _line_value(label), market_key)
    if key in seen:
        return
    seen.add(key)
    risk_emoji, risk_label = _risk_band(float(probability), theme)
    rows.append(
        MarketRow(
            market_key=market_key,
            label=label,
            probability=float(probability),
            risk=f"{risk_emoji} {risk_label}",
            theme=theme,
            period=period,
            family=family,
            rationale=rationale,
            edge=edge,
            ev=ev,
        )
    )


def build_market_table(response: Dict[str, Any], limit: int = 18) -> List[MarketRow]:
    predictions = response.get("predictions", {}) or {}
    markets = response.get("markets", {}) or {}
    evaluations = response.get("market_evaluations", []) or []
    catalog = build_market_catalog()
    rows: List[MarketRow] = []
    seen: set = set()

    def eval_for(key: str) -> Optional[EvaluatedMarket]:
        return _find_evaluation(evaluations, key)

    x2 = predictions.get("1x2", {}) or {}
    for key, label in [
        ("1x2_home", "Victoria local"),
        ("1x2_draw", "Empate"),
        ("1x2_away", "Victoria visitante"),
    ]:
        ev = eval_for(key)
        _add_candidate(
            rows,
            seen,
            market_key=key,
            label=label,
            probability=x2.get(key.split("_", 1)[1]),
            theme="1x2",
            family="result",
            period="FT",
            rationale=(catalog.get(key).interpretation if key in catalog else "Resultado global"),
            edge=ev.edge if ev else None,
            ev=ev.ev if ev else None,
        )

    dc = predictions.get("double_chance", {}) or {}
    for key, label in [
        ("double_chance_home_or_draw", "Local o empate"),
        ("double_chance_away_or_draw", "Visitante o empate"),
        ("double_chance_home_or_away", "Sin empate"),
    ]:
        ev = eval_for(key)
        _add_candidate(
            rows,
            seen,
            market_key=key,
            label=label,
            probability=dc.get(key.split("double_chance_", 1)[1]),
            theme="double_chance",
            family="result",
            period="FT",
            rationale=(catalog.get(key).interpretation if key in catalog else "Cobertura de resultado"),
            edge=ev.edge if ev else None,
            ev=ev.ev if ev else None,
        )

    ou = predictions.get("over_under", {}) or {}
    goal_keys = [
        ("over_1_5", "Más de 1.5 goles"),
        ("under_1_5", "Menos de 1.5 goles"),
        ("over_2_5", "Más de 2.5 goles"),
        ("under_2_5", "Menos de 2.5 goles"),
        ("over_3_5", "Más de 3.5 goles"),
        ("under_3_5", "Menos de 3.5 goles"),
        ("over_4_5", "Más de 4.5 goles"),
        ("under_4_5", "Menos de 4.5 goles"),
    ]
    for key, label in goal_keys:
        ev = eval_for(key)
        _add_candidate(
            rows,
            seen,
            market_key=key,
            label=label,
            probability=ou.get(key),
            theme="goals_full_time",
            family="totals",
            period="FT",
            rationale=(catalog.get(key).interpretation if key in catalog else "Total de goles"),
            edge=ev.edge if ev else None,
            ev=ev.ev if ev else None,
        )

    halftime = predictions.get("halftime", {}) or {}
    for key, label in [
        ("halftime_home", "Descanso: local"),
        ("halftime_draw", "Descanso: empate"),
        ("halftime_away", "Descanso: visitante"),
    ]:
        _add_candidate(
            rows,
            seen,
            market_key=key,
            label=label,
            probability=halftime.get(key.split("_", 1)[1]),
            theme="halftime",
            family="totals",
            period="1T",
            rationale="Distribución de primer tiempo",
        )

    first_half_goals = predictions.get("first_half_goals", {}) or {}
    first_half_ou = first_half_goals.get("over_under", {}) or {}
    for key, label in [
        ("first_half_over_0_5", "1T más de 0.5 goles"),
        ("first_half_over_1_5", "1T más de 1.5 goles"),
        ("first_half_under_1_5", "1T menos de 1.5 goles"),
        ("first_half_over_2_5", "1T más de 2.5 goles"),
        ("first_half_under_2_5", "1T menos de 2.5 goles"),
        ("first_half_over_3_5", "1T más de 3.5 goles"),
        ("first_half_under_3_5", "1T menos de 3.5 goles"),
    ]:
        _add_candidate(
            rows,
            seen,
            market_key=key,
            label=label,
            probability=first_half_ou.get(key.replace("first_half_", "")),
            theme="goals_first_half",
            family="totals",
            period="1T",
            rationale="Modelo de goles del primer tiempo",
        )

    second_half_goals = predictions.get("second_half_goals", {}) or {}
    second_half_ou = second_half_goals.get("over_under", {}) or {}
    for key, label in [
        ("second_half_over_0_5", "2T más de 0.5 goles"),
        ("second_half_over_1_5", "2T más de 1.5 goles"),
        ("second_half_under_1_5", "2T menos de 1.5 goles"),
        ("second_half_over_2_5", "2T más de 2.5 goles"),
        ("second_half_under_2_5", "2T menos de 2.5 goles"),
        ("second_half_over_3_5", "2T más de 3.5 goles"),
        ("second_half_under_3_5", "2T menos de 3.5 goles"),
    ]:
        _add_candidate(
            rows,
            seen,
            market_key=key,
            label=label,
            probability=second_half_ou.get(key.replace("second_half_", "")),
            theme="goals_second_half",
            family="totals",
            period="2T",
            rationale="Modelo de goles del segundo tiempo",
        )

    corners = markets.get("corners", {}) or {}
    if corners.get("available"):
        total_lines = corners.get("total_lines", {}) or {}
        for key, label in [
            ("corners_over_6_5", "Corners FT más de 6.5"),
            ("corners_over_7_5", "Corners FT más de 7.5"),
            ("corners_over_8_5", "Corners FT más de 8.5"),
            ("corners_over_9_5", "Corners FT más de 9.5"),
            ("corners_under_8_5", "Corners FT menos de 8.5"),
            ("corners_under_10_5", "Corners FT menos de 10.5"),
        ]:
            _add_candidate(
                rows,
                seen,
                market_key=key,
                label=label,
                probability=total_lines.get(key.split("corners_", 1)[1]),
                theme="corners_full_time",
                family="corners",
                period="FT",
                rationale="Corners globales estimados por Poisson regularizado",
            )

        periods = corners.get("periods", {}) or {}
        for key, label, theme, period in [
            ("first_half_over_1_5", "1T corners más de 1.5", "corners_first_half", "1T"),
            ("first_half_over_2_5", "1T corners más de 2.5", "corners_first_half", "1T"),
            ("first_half_under_3_5", "1T corners menos de 3.5", "corners_first_half", "1T"),
            ("second_half_over_2_5", "2T corners más de 2.5", "corners_second_half", "2T"),
            ("second_half_over_3_5", "2T corners más de 3.5", "corners_second_half", "2T"),
            ("second_half_under_4_5", "2T corners menos de 4.5", "corners_second_half", "2T"),
        ]:
            source = periods.get("first_half_lines" if period == "1T" else "second_half_lines", {})
            _add_candidate(
                rows,
                seen,
                market_key=key,
                label=label,
                probability=source.get(key.replace("first_half_", "").replace("second_half_", "")),
                theme=theme,
                family="corners",
                period=period,
                rationale="Corners por mitad basados en la mezcla ESPN + Poisson",
            )

    cards = markets.get("cards", {}) or {}
    if cards.get("available"):
        total_lines = cards.get("total_lines", {}) or {}
        for key, label in [
            ("cards_over_3_5", "Tarjetas FT más de 3.5"),
            ("cards_over_4_5", "Tarjetas FT más de 4.5"),
            ("cards_under_4_5", "Tarjetas FT menos de 4.5"),
            ("cards_under_6_5", "Tarjetas FT menos de 6.5"),
        ]:
            _add_candidate(
                rows,
                seen,
                market_key=key,
                label=label,
                probability=total_lines.get(key.split("cards_", 1)[1].replace("_5", "")),
                theme="cards_full_time",
                family="cards",
                period="FT",
                rationale="Tarjetas estimadas desde intensidad competitiva",
            )

    shots = markets.get("shots_on_target", {}) or {}
    if shots.get("available"):
        total_lines = shots.get("total_lines", {}) or {}
        for key, label in [
            ("shots_over_7_5", "Tiros a puerta FT más de 7.5"),
            ("shots_over_8_5", "Tiros a puerta FT más de 8.5"),
            ("shots_under_9_5", "Tiros a puerta FT menos de 9.5"),
        ]:
            _add_candidate(
                rows,
                seen,
                market_key=key,
                label=label,
                probability=total_lines.get(key.split("shots_", 1)[1].replace("_5", "")),
                theme="shots_full_time",
                family="shots",
                period="FT",
                rationale="Tiros a puerta derivados del volumen ofensivo",
            )

    player_props = markets.get("player_props", {}) or {}
    anytime = player_props.get("anytime_scorer", {}) if isinstance(player_props, dict) else {}
    if anytime.get("available"):
        for candidate in anytime.get("top_candidates", [])[:3]:
            name = candidate.get("player_name") or "Jugador"
            prob = candidate.get("probability_decimal")
            if prob is None:
                prob = candidate.get("probability_pct")
                if isinstance(prob, (int, float)) and prob > 1:
                    prob = prob / 100.0
            safe = str(name).lower().replace(" ", "_")
            _add_candidate(
                rows,
                seen,
                market_key=f"player_shots_over_1_5_{safe}",
                label=f"{name} anota",
                probability=prob,
                theme="player_shots",
                family="player",
                period="FT",
                rationale="Selección individual de mayor probabilidad",
            )

    rows.sort(key=lambda row: (-row.probability, row.family, row.period, row.label))

    final_rows: List[MarketRow] = []
    family_caps = {
        "result": 4,
        "totals": 8,
        "corners": 6,
        "cards": 3,
        "shots": 3,
        "player": 3,
    }
    family_counts: Dict[str, int] = {}
    theme_counts: Dict[str, int] = {}

    for row in rows:
        if len(final_rows) >= limit:
            break
        if family_counts.get(row.family, 0) >= family_caps.get(row.family, 3):
            continue
        if theme_counts.get(row.theme, 0) >= 2:
            continue
        family_counts[row.family] = family_counts.get(row.family, 0) + 1
        theme_counts[row.theme] = theme_counts.get(row.theme, 0) + 1
        final_rows.append(row)

    return final_rows
