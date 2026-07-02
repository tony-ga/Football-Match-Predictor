"""
Pydantic v2 schemas for football match prediction input validation.
Covers all JSON fields: metadata, team1/team2, FACTORES_TACTICOS,
FACTORES_COLECTIVOS, CONTEXTO, FACTORES_EXTERNOS, JUGADORES.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class CompetitionType(str, Enum):
    WORLD_CUP = "world_cup"
    CONTINENTAL = "continental"
    FRIENDLY = "friendly"
    QUALIFIER = "qualifier"
    LEAGUE = "league"


class MatchStage(str, Enum):
    GROUP = "group"
    ROUND_OF_16 = "round_of_16"
    QUARTER_FINAL = "quarter_final"
    SEMI_FINAL = "semi_final"
    FINAL = "final"
    REGULAR = "regular"


class Metadata(BaseModel):
    match_id: str = Field(default="unknown", description="Unique match identifier")
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    competition: str = Field(default="unknown")
    competition_type: CompetitionType = Field(default=CompetitionType.FRIENDLY)
    stage: MatchStage = Field(default=MatchStage.REGULAR)
    date: Optional[str] = None
    neutral_venue: bool = Field(default=False)
    venue: Optional[str] = None
    referee: Optional[str] = None


class FactoresTacticos(BaseModel):
    """Tactical factors on a 0-10 scale."""

    pressing_intensidad: float = Field(default=5.0, ge=0, le=10)
    bloque_defensivo: float = Field(default=5.0, ge=0, le=10)
    transiciones_rapidas: float = Field(default=5.0, ge=0, le=10)
    posesion_preferida: float = Field(default=5.0, ge=0, le=10)
    linea_defensiva_alta: float = Field(default=5.0, ge=0, le=10)
    variabilidad_tactica: float = Field(default=5.0, ge=0, le=10)
    juego_aereo: float = Field(default=5.0, ge=0, le=10)
    duelos_individuales: float = Field(default=5.0, ge=0, le=10)

    @field_validator("*", mode="before")
    @classmethod
    def coerce_to_float(cls, v: Any) -> float:
        if v is None:
            return 5.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 5.0


class FactoresColectivos(BaseModel):
    """Collective team factors on a 0-10 scale."""

    cohesion_grupal: float = Field(default=5.0, ge=0, le=10)
    experiencia_internacional: float = Field(default=5.0, ge=0, le=10)
    liderazgo_cancha: float = Field(default=5.0, ge=0, le=10)
    mentalidad_competitiva: float = Field(default=5.0, ge=0, le=10)
    solidez_defensiva: float = Field(default=5.0, ge=0, le=10)
    creatividad_ofensiva: float = Field(default=5.0, ge=0, le=10)
    eficiencia_finalizacion: float = Field(default=5.0, ge=0, le=10)
    ritmo_juego: float = Field(default=5.0, ge=0, le=10)
    discipline: float = Field(default=5.0, ge=0, le=10)

    @field_validator("*", mode="before")
    @classmethod
    def coerce_to_float(cls, v: Any) -> float:
        if v is None:
            return 5.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 5.0


class Contexto(BaseModel):
    """Match context factors."""

    ranking_fifa: Optional[float] = Field(default=50.0, ge=1, le=211)
    partidos_ultimos_6_meses: Optional[int] = Field(default=6, ge=0)
    victorias_ultimos_6: Optional[int] = Field(default=3, ge=0)
    derrotas_ultimos_6: Optional[int] = Field(default=1, ge=0)
    empates_ultimos_6: Optional[int] = Field(default=2, ge=0)
    goles_marcados_ultimos_6: Optional[float] = Field(default=1.5, ge=0)
    goles_recibidos_ultimos_6: Optional[float] = Field(default=1.0, ge=0)
    xg_promedio_favor: Optional[float] = Field(default=1.5, ge=0, le=5)
    xg_promedio_contra: Optional[float] = Field(default=1.0, ge=0, le=5)
    head_to_head_wins: Optional[int] = Field(default=0, ge=0)
    head_to_head_draws: Optional[int] = Field(default=0, ge=0)
    head_to_head_losses: Optional[int] = Field(default=0, ge=0)
    dias_desde_ultimo_partido: Optional[int] = Field(default=7, ge=0)
    partidos_en_15_dias: Optional[int] = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_h2h(self) -> "Contexto":
        total = (
            (self.head_to_head_wins or 0)
            + (self.head_to_head_draws or 0)
            + (self.head_to_head_losses or 0)
        )
        if total > 50:
            # Normalize if too many H2H games reported — could clamp or warn
            pass
        return self


class FactoresExternos(BaseModel):
    """External/environmental factors."""

    localía: float = Field(
        default=0.0,
        ge=-1,
        le=1,
        description="1=home, 0=neutral, -1=away",
    )
    distancia_viaje_km: Optional[float] = Field(default=0.0, ge=0)
    diferencia_horaria_h: Optional[float] = Field(default=0.0)
    temperatura_c: Optional[float] = Field(default=20.0)
    humedad_pct: Optional[float] = Field(default=60.0, ge=0, le=100)
    altitud_m: Optional[float] = Field(default=0.0, ge=0)
    lluvia: Optional[bool] = Field(default=False)
    viento_kmh: Optional[float] = Field(default=10.0, ge=0)
    presion_mediatica: float = Field(default=5.0, ge=0, le=10)
    motivacion: float = Field(default=5.0, ge=0, le=10)
    fatiga_acumulada: float = Field(default=3.0, ge=0, le=10)
    importancia_partido: float = Field(default=5.0, ge=0, le=10)

    @field_validator("temperatura_c", mode="before")
    @classmethod
    def default_temp(cls, v: Any) -> float:
        if v is None:
            return 20.0
        return float(v)


class Jugador(BaseModel):
    """Individual player data."""

    nombre: str = Field(default="Unknown")
    posicion: str = Field(default="MID", description="GK, DEF, MID, FWD")
    rating: float = Field(default=6.0, ge=0, le=10)
    forma_actual: float = Field(default=6.0, ge=0, le=10)
    titular: bool = Field(default=True)
    goles_temporada: Optional[float] = Field(default=0.0, ge=0)
    asistencias_temporada: Optional[float] = Field(default=0.0, ge=0)
    xg_temporada: Optional[float] = Field(default=0.0, ge=0)
    disponibilidad: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="1=fully fit, 0=unavailable",
    )
    impacto_equipo: float = Field(
        default=5.0,
        ge=0,
        le=10,
        description="How much team depends on this player",
    )

    @field_validator("posicion", mode="before")
    @classmethod
    def normalize_position(cls, v: Any) -> str:
        if v is None:
            return "MID"
        mapping: Dict[str, str] = {
            "portero": "GK",
            "goalkeeper": "GK",
            "arquero": "GK",
            "defensa": "DEF",
            "defender": "DEF",
            "lateral": "DEF",
            "central": "DEF",
            "mediocampista": "MID",
            "midfielder": "MID",
            "medio": "MID",
            "delantero": "FWD",
            "forward": "FWD",
            "atacante": "FWD",
            "extremo": "FWD",
        }
        return mapping.get(str(v).lower(), str(v).upper())


class TeamData(BaseModel):
    """Complete data for one team."""

    nombre: str = Field(..., description="Team name")
    FACTORES_TACTICOS: FactoresTacticos = Field(default_factory=FactoresTacticos)
    FACTORES_COLECTIVOS: FactoresColectivos = Field(default_factory=FactoresColectivos)
    CONTEXTO: Contexto = Field(default_factory=Contexto)
    FACTORES_EXTERNOS: FactoresExternos = Field(default_factory=FactoresExternos)
    JUGADORES: List[Jugador] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_keys(cls, data: Any) -> Any:
        """Handle both camelCase and UPPER_CASE key variants."""
        if isinstance(data, dict):
            key_map: Dict[str, str] = {
                "factores_tacticos": "FACTORES_TACTICOS",
                "tacticos": "FACTORES_TACTICOS",
                "factores_colectivos": "FACTORES_COLECTIVOS",
                "colectivos": "FACTORES_COLECTIVOS",
                "contexto": "CONTEXTO",
                "factores_externos": "FACTORES_EXTERNOS",
                "externos": "FACTORES_EXTERNOS",
                "jugadores": "JUGADORES",
                "players": "JUGADORES",
            }
            for old_key, new_key in key_map.items():
                if old_key in data and new_key not in data:
                    data[new_key] = data.pop(old_key)
        return data


class MatchInput(BaseModel):
    """Top-level match input schema."""

    metadata: Metadata
    team1: TeamData
    team2: TeamData

    model_config = {
        "json_schema_extra": {
            "example": {
                "metadata": {
                    "match_id": "wc2022_g1_arg_qat",
                    "home_team": "Argentina",
                    "away_team": "Qatar",
                    "competition": "FIFA World Cup 2022",
                    "competition_type": "world_cup",
                    "stage": "group",
                    "neutral_venue": True,
                },
                "team1": {"nombre": "Argentina"},
                "team2": {"nombre": "Qatar"},
            }
        }
    }
