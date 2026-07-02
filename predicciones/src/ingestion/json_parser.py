"""
JSON parser and validator for football match prediction inputs.
Handles loading, validation, defaults filling, and scale normalization.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

from pydantic import ValidationError

from .schemas import (
    Contexto,
    FactoresColectivos,
    FactoresTacticos,
    FactoresExternos,
    Jugador,
    MatchInput,
    TeamData,
)

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Raised when a match JSON cannot be parsed or validated."""
    pass


def load_match_json(source: Union[str, Path, Dict[str, Any]]) -> MatchInput:
    """
    Load and validate a match JSON from a file path, JSON string, or dict.

    Args:
        source: File path (str/Path), JSON string, or already-parsed dict.

    Returns:
        Validated MatchInput object.

    Raises:
        ParseError: If the JSON is invalid or fails schema validation.
    """
    raw: Dict[str, Any]

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                try:
                    raw = json.load(f)
                except json.JSONDecodeError as e:
                    raise ParseError(f"Invalid JSON in {path}: {e}") from e
        else:
            # Try as JSON string
            try:
                raw = json.loads(str(source))
            except json.JSONDecodeError as e:
                raise ParseError(f"Invalid JSON string: {e}") from e
    elif isinstance(source, dict):
        raw = source
    else:
        raise ParseError(f"Unsupported source type: {type(source)}")

    # Normalize top-level keys
    raw = _normalize_top_level_keys(raw)

    try:
        match_input = MatchInput.model_validate(raw)
    except ValidationError as e:
        raise ParseError(f"Schema validation failed:\n{e}") from e

    logger.info(
        "Parsed match: %s vs %s",
        match_input.metadata.home_team,
        match_input.metadata.away_team,
    )
    return match_input


def _normalize_top_level_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize top-level key names to the expected schema."""
    normalized: Dict[str, Any] = {}
    key_map: Dict[str, str] = {
        "meta": "metadata",
        "match_metadata": "metadata",
        "equipo1": "team1",
        "equipo2": "team2",
        "home": "team1",
        "away": "team2",
    }
    for k, v in data.items():
        normalized_key = key_map.get(k.lower(), k)
        normalized[normalized_key] = v
    return normalized


def match_input_to_dict(match: MatchInput) -> Dict[str, Any]:
    """Serialize a MatchInput back to a plain dict."""
    return match.model_dump()


def extract_team_summary(team: TeamData) -> Dict[str, Any]:
    """
    Extract a flat summary dict from a TeamData object for logging/debugging.
    """
    ctx = team.CONTEXTO
    ext = team.FACTORES_EXTERNOS
    colect = team.FACTORES_COLECTIVOS
    tacticos = team.FACTORES_TACTICOS

    n_jugadores = len(team.JUGADORES)
    titulares = [j for j in team.JUGADORES if j.titular]

    return {
        "nombre": team.nombre,
        "ranking_fifa": ctx.ranking_fifa,
        "forma_reciente": _calc_form_ratio(ctx),
        "xg_favor": ctx.xg_promedio_favor,
        "xg_contra": ctx.xg_promedio_contra,
        "localía": ext.localía,
        "fatiga": ext.fatiga_acumulada,
        "motivacion": ext.motivacion,
        "cohesion": colect.cohesion_grupal,
        "creatividad_ofensiva": colect.creatividad_ofensiva,
        "solidez_defensiva": colect.solidez_defensiva,
        "pressing": tacticos.pressing_intensidad,
        "n_jugadores": n_jugadores,
        "n_titulares": len(titulares),
    }


def _calc_form_ratio(ctx: Contexto) -> float:
    """Calculate win ratio from last 6 months matches."""
    total = ctx.partidos_ultimos_6_meses or 0
    if total == 0:
        return 0.5  # neutral if no data
    wins = ctx.victorias_ultimos_6 or 0
    draws = ctx.empates_ultimos_6 or 0
    # Points-based form ratio
    points = wins * 3 + draws * 1
    max_points = total * 3
    return points / max_points if max_points > 0 else 0.5
