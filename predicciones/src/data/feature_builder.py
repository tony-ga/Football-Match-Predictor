import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Ruta al archivo de ratings estáticos del Mundial 2026
RATINGS_PATH = Path(__file__).parent.parent.parent / "data" / "ratings_wc2026.json"


class TeamProfile:
    def __init__(
        self,
        team_name: str,
        lambda_attack: float,
        lambda_defense: float,
        recent_form: Dict[str, Any],
        wc_form: Dict[str, Any],
        corners_lambda: float,
        cards_lambda: float,
        effective_weight_matches: float,
        data_warnings: Optional[List[str]] = None
    ):
        self.team_name = team_name
        self.lambda_attack = lambda_attack
        self.lambda_defense = lambda_defense
        self.recent_form = recent_form
        self.wc_form = wc_form
        self.corners_lambda = corners_lambda
        self.cards_lambda = cards_lambda
        self.effective_weight_matches = effective_weight_matches
        self.data_warnings = data_warnings or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team_name": self.team_name,
            "lambda_attack": self.lambda_attack,
            "lambda_defense": self.lambda_defense,
            "recent_form": self.recent_form,
            "wc_form": self.wc_form,
            "corners_lambda": self.corners_lambda,
            "cards_lambda": self.cards_lambda,
            "effective_weight_matches": self.effective_weight_matches,
            "data_warnings": self.data_warnings
        }


class MatchFeatureBuilder:
    def __init__(self, api_client=None):
        # api_client se mantiene en firma para compatibilidad pero NO SE USA
        self._ratings = self._load_ratings()

    def _load_ratings(self) -> dict:
        """Carga los ratings estáticos desde el archivo JSON."""
        try:
            with open(RATINGS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"ratings_wc2026.json not found at {RATINGS_PATH}")
            return {"teams": {}, "default": {"attack": 1.10, "defense": 1.00, "fifa_rank": 100}}

    def build_team_profile(
        self,
        team_name: str,
        match_date: str,
        include_wc_matches: bool = True
    ) -> TeamProfile:
        """
        Construye el perfil del equipo usando únicamente ratings estáticos.
        NO llama a ninguna API externa.
        """
        teams = self._ratings.get("teams", {})
        default = self._ratings.get("default", {"attack": 1.10, "defense": 1.00, "fifa_rank": 100})
        
        rating = teams.get(team_name, default)
        if team_name not in teams:
            logger.warning(f"Team '{team_name}' not in ratings_wc2026.json, using default.")

        lambda_attack = rating["attack"]
        lambda_defense = rating["defense"]
        fifa_rank = rating.get("fifa_rank", 100)

        return TeamProfile(
            team_name=team_name,
            lambda_attack=lambda_attack,
            lambda_defense=lambda_defense,
            recent_form={
                "record": "N/A",
                "goals_scored_avg": round(lambda_attack * 1.35, 2),
                "goals_conceded_avg": round(lambda_defense * 1.35, 2),
                "btts_rate": 0.50,
                "corners_avg": 5.0,
                "cards_avg": 2.0,
                "clean_sheets": 0,
                "form": "N/A",
                "data_source": "ratings_wc2026",
                "fifa_rank": fifa_rank
            },
            wc_form={
                "played": 0, "record": "N/A",
                "goals_scored": 0, "goals_conceded": 0, "matches": []
            },
            corners_lambda=5.0,
            cards_lambda=2.0,
            effective_weight_matches=0.0,
            data_warnings=[]
        )
