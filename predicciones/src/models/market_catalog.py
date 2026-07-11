from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class MarketFamily(Enum):
    RESULT = "result"
    TOTALS = "totals"
    BTTS = "btts"
    CORNERS = "corners"
    PLAYER_SHOTS = "player_shots"


class RiskProfile(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class MarketDefinition:
    market_key: str
    name: str
    family: MarketFamily
    calculation_source: str
    risk_fit: List[RiskProfile]
    interpretation: str
    requires_team_history: bool = False
    requires_player_history: bool = False
    enabled: bool = True
    
    @property
    def is_low_risk_compatible(self) -> bool:
        return RiskProfile.LOW in self.risk_fit

    @property
    def is_medium_risk_compatible(self) -> bool:
        return RiskProfile.MEDIUM in self.risk_fit
        
    @property
    def is_high_risk_compatible(self) -> bool:
        return RiskProfile.HIGH in self.risk_fit


def build_market_catalog() -> Dict[str, MarketDefinition]:
    """
    Builds and returns the full catalog of supported markets.
    """
    catalog = {}
    
    def add(m: MarketDefinition):
        catalog[m.market_key] = m

    # RESULT
    add(MarketDefinition(
        market_key="1x2_home",
        name="Local",
        family=MarketFamily.RESULT,
        calculation_source="poisson_matrix",
        risk_fit=[RiskProfile.HIGH],
        interpretation="El modelo ve ventaja clara del local, script estrecho donde se exige victoria.",
    ))
    add(MarketDefinition(
        market_key="1x2_draw",
        name="Empate",
        family=MarketFamily.RESULT,
        calculation_source="poisson_matrix",
        risk_fit=[RiskProfile.HIGH],
        interpretation="El modelo ve un partido muy equilibrado sin ganador claro.",
    ))
    add(MarketDefinition(
        market_key="1x2_away",
        name="Visitante",
        family=MarketFamily.RESULT,
        calculation_source="poisson_matrix",
        risk_fit=[RiskProfile.HIGH],
        interpretation="El modelo ve ventaja clara del visitante, script estrecho donde se exige victoria.",
    ))
    add(MarketDefinition(
        market_key="double_chance_home_or_draw",
        name="Home or Draw",
        family=MarketFamily.RESULT,
        calculation_source="poisson_matrix",
        risk_fit=[RiskProfile.LOW, RiskProfile.MEDIUM],
        interpretation="Inclinación al local sin separación suficiente para exigir victoria. Mucho margen de error.",
    ))
    add(MarketDefinition(
        market_key="double_chance_away_or_draw",
        name="Away or Draw",
        family=MarketFamily.RESULT,
        calculation_source="poisson_matrix",
        risk_fit=[RiskProfile.LOW, RiskProfile.MEDIUM],
        interpretation="Inclinación al visitante sin separación suficiente para exigir victoria. Mucho margen de error.",
    ))
    add(MarketDefinition(
        market_key="double_chance_home_or_away",
        name="Home or Away",
        family=MarketFamily.RESULT,
        calculation_source="poisson_matrix",
        risk_fit=[RiskProfile.LOW, RiskProfile.MEDIUM],
        interpretation="Partido que no debería terminar en empate, pero no hay un claro favorito.",
    ))

    # TOTALS
    add(MarketDefinition(
        market_key="over_1_5",
        name="Over 1.5 Goals",
        family=MarketFamily.TOTALS,
        calculation_source="poisson_matrix",
        risk_fit=[RiskProfile.LOW],
        interpretation="Partido abierto mínimo. Script ancho con mucho margen de error.",
    ))
    add(MarketDefinition(
        market_key="over_2_5",
        name="Over 2.5 Goals",
        family=MarketFamily.TOTALS,
        calculation_source="poisson_matrix",
        risk_fit=[RiskProfile.MEDIUM],
        interpretation="Partido abierto que exige volumen estándar de goles.",
    ))
    add(MarketDefinition(
        market_key="over_3_5",
        name="Over 3.5 Goals",
        family=MarketFamily.TOTALS,
        calculation_source="poisson_matrix",
        risk_fit=[RiskProfile.HIGH],
        interpretation="Partido muy abierto, script estrecho y agresivo de alta puntuación.",
    ))
    add(MarketDefinition(
        market_key="over_4_5",
        name="Over 4.5 Goals",
        family=MarketFamily.TOTALS,
        calculation_source="poisson_matrix",
        risk_fit=[RiskProfile.HIGH],
        interpretation="Caos extremo. Margen de error mínimo para que se cumpla.",
    ))
    add(MarketDefinition(
        market_key="under_1_5",
        name="Under 1.5 Goals",
        family=MarketFamily.TOTALS,
        calculation_source="poisson_matrix",
        risk_fit=[RiskProfile.HIGH],
        interpretation="Partido sumamente cerrado, margen de error mínimo.",
    ))
    add(MarketDefinition(
        market_key="under_2_5",
        name="Under 2.5 Goals",
        family=MarketFamily.TOTALS,
        calculation_source="poisson_matrix",
        risk_fit=[RiskProfile.MEDIUM, RiskProfile.HIGH],
        interpretation="Partido cerrado, se espera poco volumen ofensivo.",
    ))
    add(MarketDefinition(
        market_key="under_3_5",
        name="Under 3.5 Goals",
        family=MarketFamily.TOTALS,
        calculation_source="poisson_matrix",
        risk_fit=[RiskProfile.MEDIUM],
        interpretation="Partido controlado donde no se espera descontrol ofensivo.",
    ))
    add(MarketDefinition(
        market_key="under_4_5",
        name="Under 4.5 Goals",
        family=MarketFamily.TOTALS,
        calculation_source="poisson_matrix",
        risk_fit=[RiskProfile.LOW],
        interpretation="El modelo no espera caos extremo. Script muy ancho con mucho margen de error.",
    ))

    # BTTS
    add(MarketDefinition(
        market_key="btts_yes",
        name="BTTS: Yes",
        family=MarketFamily.BTTS,
        calculation_source="poisson_matrix",
        risk_fit=[RiskProfile.MEDIUM, RiskProfile.HIGH],
        interpretation="Ambos equipos tienen capacidad ofensiva y vulnerabilidad defensiva.",
    ))
    add(MarketDefinition(
        market_key="btts_no",
        name="BTTS: No",
        family=MarketFamily.BTTS,
        calculation_source="poisson_matrix",
        risk_fit=[RiskProfile.MEDIUM, RiskProfile.HIGH],
        interpretation="Al menos un equipo debería mantener su portería a cero o no tener capacidad de anotar.",
    ))

    # CORNERS
    add(MarketDefinition(
        market_key="corners_over_6_5",
        name="Corners Over 6.5",
        family=MarketFamily.CORNERS,
        calculation_source="historical_averages",
        risk_fit=[RiskProfile.LOW],
        requires_team_history=True,
        interpretation="Ritmo base muy accesible. Script ancho con alto margen de error.",
    ))
    add(MarketDefinition(
        market_key="corners_over_7_5",
        name="Corners Over 7.5",
        family=MarketFamily.CORNERS,
        calculation_source="historical_averages",
        risk_fit=[RiskProfile.LOW, RiskProfile.MEDIUM],
        requires_team_history=True,
        interpretation="Ritmo y promedios sostienen volumen estándar alto.",
    ))
    add(MarketDefinition(
        market_key="corners_over_8_5",
        name="Corners Over 8.5",
        family=MarketFamily.CORNERS,
        calculation_source="historical_averages",
        risk_fit=[RiskProfile.MEDIUM],
        requires_team_history=True,
        interpretation="Exige buen volumen ofensivo de ambos equipos.",
    ))
    add(MarketDefinition(
        market_key="corners_over_9_5",
        name="Corners Over 9.5",
        family=MarketFamily.CORNERS,
        calculation_source="historical_averages",
        risk_fit=[RiskProfile.HIGH],
        requires_team_history=True,
        interpretation="Script estrecho que requiere flujo constante de ataques y saques de esquina.",
    ))
    add(MarketDefinition(
        market_key="corners_under_10_5",
        name="Corners Under 10.5",
        family=MarketFamily.CORNERS,
        calculation_source="historical_averages",
        risk_fit=[RiskProfile.MEDIUM],
        requires_team_history=True,
        interpretation="Partido más pausado o centrado en el medio campo.",
    ))
    add(MarketDefinition(
        market_key="corners_under_8_5",
        name="Corners Under 8.5",
        family=MarketFamily.CORNERS,
        calculation_source="historical_averages",
        risk_fit=[RiskProfile.HIGH],
        requires_team_history=True,
        interpretation="Se espera un juego muy trabado con mínimas llegadas a línea de fondo.",
    ))

    # PLAYER SHOTS (dynamic templates)
    add(MarketDefinition(
        market_key="player_shots_over_1_5",
        name="Player Shots Over 1.5",
        family=MarketFamily.PLAYER_SHOTS,
        calculation_source="player_stats",
        risk_fit=[RiskProfile.LOW],
        requires_player_history=True,
        interpretation="Volumen individual accesible. Jugador con participación estándar.",
    ))
    add(MarketDefinition(
        market_key="player_shots_over_2_5",
        name="Player Shots Over 2.5",
        family=MarketFamily.PLAYER_SHOTS,
        calculation_source="player_stats",
        risk_fit=[RiskProfile.MEDIUM],
        requires_player_history=True,
        interpretation="Volumen individual suficiente para línea agresiva pero plausible.",
    ))
    add(MarketDefinition(
        market_key="player_shots_over_3_5",
        name="Player Shots Over 3.5",
        family=MarketFamily.PLAYER_SHOTS,
        calculation_source="player_stats",
        risk_fit=[RiskProfile.HIGH],
        requires_player_history=True,
        interpretation="Script estrecho. El jugador debe ser el eje ofensivo dominante.",
    ))

    return catalog
