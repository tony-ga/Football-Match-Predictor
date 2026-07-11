from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from predicciones.src.data.espn_client import EspnWorldCupClient
from predicciones.src.data.espn_stats_parsers import extract_events_from_summary

logger = logging.getLogger(__name__)


def _poisson_sf(k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0
    from math import exp, factorial

    cdf = 0.0
    for i in range(k + 1):
        cdf += exp(-lam) * (lam ** i) / factorial(i)
    return max(0.0, 1.0 - cdf)


def _poisson_half_line_prob(expected: float, line: float, side: str) -> float:
    line_int = int(line + 0.5)
    if side == "over":
        return _poisson_sf(line_int - 1, expected)
    return 1.0 - _poisson_sf(line_int - 1, expected)


def _safe_mean(values: List[float], default: float) -> float:
    return float(np.mean(values)) if values else default


@dataclass(frozen=True)
class CornerHalfProfile:
    sample_size: int
    first_half_for_avg: float
    first_half_against_avg: float
    second_half_for_avg: float
    second_half_against_avg: float
    first_half_share: float
    second_half_share: float


@lru_cache(maxsize=1)
def _get_client() -> EspnWorldCupClient:
    return EspnWorldCupClient(timeout=15)


def _count_corner_halves(summary: Dict[str, Any], team_name: str) -> Tuple[int, int, int, int, int, int]:
    events = extract_events_from_summary(summary)
    home_team = ""
    away_team = ""
    competitions = summary.get("competitions") or []
    if competitions:
        competitors = competitions[0].get("competitors") or []
        for competitor in competitors:
            if competitor.get("homeAway") == "home":
                home_team = competitor.get("team", {}).get("displayName", "") or ""
            elif competitor.get("homeAway") == "away":
                away_team = competitor.get("team", {}).get("displayName", "") or ""

    home_first = home_second = away_first = away_second = 0
    first_total = second_total = 0
    team_norm = str(team_name).strip().lower()

    for event in events:
        if (event.get("event_type") or "").lower() != "corner":
            continue
        period = int(event.get("period") or 1)
        event_team = str(event.get("team_name") or "").strip().lower()
        is_team = event_team == team_norm
        if period <= 1:
            first_total += 1
            if is_team:
                if team_norm == str(home_team).strip().lower():
                    home_first += 1
                elif team_norm == str(away_team).strip().lower():
                    away_first += 1
        else:
            second_total += 1
            if is_team:
                if team_norm == str(home_team).strip().lower():
                    home_second += 1
                elif team_norm == str(away_team).strip().lower():
                    away_second += 1

    team_first_for = home_first if team_norm == str(home_team).strip().lower() else away_first
    team_second_for = home_second if team_norm == str(home_team).strip().lower() else away_second
    opp_first = away_first if team_norm == str(home_team).strip().lower() else home_first
    opp_second = away_second if team_norm == str(home_team).strip().lower() else home_second

    return team_first_for, team_second_for, opp_first, opp_second, first_total, second_total


@lru_cache(maxsize=128)
def get_team_corner_half_profile(team_name: str, max_matches: int = 10) -> CornerHalfProfile:
    client = _get_client()
    matches = client.get_recent_team_matches(team_name, days_back=180, max_matches=max_matches)
    first_for: List[float] = []
    first_against: List[float] = []
    second_for: List[float] = []
    second_against: List[float] = []

    for match in matches:
        if not match.get("completed") or not match.get("event_id"):
            continue
        try:
            summary = client.get_summary(str(match["event_id"]))
        except Exception as exc:  # pragma: no cover - network fallback
            logger.debug("Failed to fetch summary for %s: %s", match.get("event_id"), exc)
            continue
        team_first, team_second, opp_first, opp_second, _, _ = _count_corner_halves(summary, team_name)
        if team_first or opp_first or team_second or opp_second:
            first_for.append(float(team_first))
            first_against.append(float(opp_first))
            second_for.append(float(team_second))
            second_against.append(float(opp_second))

    sample_size = len(first_for)
    total_for = [a + b for a, b in zip(first_for, second_for)]
    total_for_mean = _safe_mean(total_for, 0.0)
    first_for_mean = _safe_mean(first_for, 0.0)
    second_for_mean = _safe_mean(second_for, 0.0)

    if total_for_mean <= 0:
        first_share = 0.45
    else:
        first_share = float(np.clip(first_for_mean / max(first_for_mean + second_for_mean, 0.1), 0.32, 0.58))

    return CornerHalfProfile(
        sample_size=sample_size,
        first_half_for_avg=round(first_for_mean, 2),
        first_half_against_avg=round(_safe_mean(first_against, 0.0), 2),
        second_half_for_avg=round(second_for_mean, 2),
        second_half_against_avg=round(_safe_mean(second_against, 0.0), 2),
        first_half_share=round(first_share, 3),
        second_half_share=round(1.0 - first_share, 3),
    )


def _blend_share(team_share: float, global_share: float, sample_size: int) -> float:
    weight = float(np.clip(sample_size / 8.0, 0.0, 1.0))
    return float(np.clip((team_share * weight) + (global_share * (1.0 - weight)), 0.28, 0.62))


@lru_cache(maxsize=1)
def get_global_corner_half_share() -> float:
    client = _get_client()
    matches = client.get_world_cup_matches(limit=500, season_type=3)
    first = 0
    second = 0
    for match in matches[:24]:
        if not match.get("event_id") or not match.get("completed"):
            continue
        try:
            summary = client.get_summary(str(match["event_id"]))
        except Exception:
            continue
        events = extract_events_from_summary(summary)
        for event in events:
            if (event.get("event_type") or "").lower() != "corner":
                continue
            if int(event.get("period") or 1) <= 1:
                first += 1
            else:
                second += 1
    total = first + second
    if total <= 0:
        return 0.45
    return float(np.clip(first / total, 0.34, 0.56))


def build_corner_period_projection(
    home_team: str,
    away_team: str,
    home_expected_corners: float,
    away_expected_corners: float,
    home_sample_size: int = 0,
    away_sample_size: int = 0,
) -> Dict[str, Any]:
    global_share = get_global_corner_half_share()
    home_profile = get_team_corner_half_profile(home_team)
    away_profile = get_team_corner_half_profile(away_team)

    home_first_share = _blend_share(home_profile.first_half_share, global_share, home_profile.sample_size)
    away_first_share = _blend_share(away_profile.first_half_share, global_share, away_profile.sample_size)
    home_second_share = 1.0 - home_first_share
    away_second_share = 1.0 - away_first_share

    expected_home_first = home_expected_corners * home_first_share
    expected_home_second = home_expected_corners * home_second_share
    expected_away_first = away_expected_corners * away_first_share
    expected_away_second = away_expected_corners * away_second_share

    first_total = expected_home_first + expected_away_first
    second_total = expected_home_second + expected_away_second
    total = first_total + second_total

    def build_lines(expected: float, lines: List[float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for line in lines:
            key = str(line).replace(".", "_")
            out[f"over_{key}"] = round(_poisson_half_line_prob(expected, line, "over"), 4)
            out[f"under_{key}"] = round(_poisson_half_line_prob(expected, line, "under"), 4)
        return out

    return {
        "available": True,
        "global_first_half_share": round(global_share, 3),
        "home_first_half_share": round(home_first_share, 3),
        "away_first_half_share": round(away_first_share, 3),
        "expected_total": round(total, 2),
        "expected_first_half_total": round(first_total, 2),
        "expected_second_half_total": round(second_total, 2),
        "team_expected": {
            "home": {
                "first_half": round(expected_home_first, 2),
                "second_half": round(expected_home_second, 2),
            },
            "away": {
                "first_half": round(expected_away_first, 2),
                "second_half": round(expected_away_second, 2),
            },
        },
        "total_lines": build_lines(total, [6.5, 7.5, 8.5, 9.5, 10.5]),
        "first_half_lines": build_lines(first_total, [1.5, 2.5, 3.5, 4.5]),
        "second_half_lines": build_lines(second_total, [2.5, 3.5, 4.5, 5.5]),
        "team_lines": {
            "home_first_half": build_lines(expected_home_first, [0.5, 1.5, 2.5, 3.5]),
            "away_first_half": build_lines(expected_away_first, [0.5, 1.5, 2.5, 3.5]),
            "home_second_half": build_lines(expected_home_second, [0.5, 1.5, 2.5, 3.5]),
            "away_second_half": build_lines(expected_away_second, [0.5, 1.5, 2.5, 3.5]),
        },
        "profiles": {
            "home": home_profile.__dict__,
            "away": away_profile.__dict__,
        },
    }
