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
        data_warnings: Optional[List[str]] = None,
        data_source: str = "static_ratings"
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
        self.data_source = data_source

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
            "data_warnings": self.data_warnings,
            "data_source": self.data_source
        }


class MatchFeatureBuilder:
    def __init__(self, api_client=None, espn_client=None):
        # api_client se mantiene en firma para compatibilidad pero NO SE USA como fuente principal
        # espn_client es el cliente de ESPN World Cup
        self.api = api_client
        self.espn = espn_client
        self._ratings = self._load_ratings()
        
        # Stage weights para weighting por etapa
        self.stage_weights = {
            "group": 1.0,
            "round_of_32": 1.1,
            "round_of_16": 1.2,
            "quarter_final": 1.3,
            "semi_final": 1.4,
            "final": 1.5,
            "third_place": 1.2,
        }

    def _load_ratings(self) -> dict:
        """Carga los ratings estáticos desde el archivo JSON."""
        try:
            with open(RATINGS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"ratings_wc2026.json not found at {RATINGS_PATH}")
            return {"teams": {}, "default": {"attack": 1.10, "defense": 1.00, "fifa_rank": 100}}

    def _get_stage_weight(self, stage: str) -> float:
        """Obtiene el weight para una etapa dada."""
        return self.stage_weights.get(stage, 1.0)

    def _compute_espn_profile(
        self,
        team_name: str,
        matches: List[Dict[str, Any]],
        base_rating: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Computa perfil basado en partidos ESPN.
        
        Returns dict con:
        - lambda_attack, lambda_defense ajustados
        - recent_form stats
        - warnings
        """
        warnings = []
        
        # Filtrar solo partidos completados
        completed = [m for m in matches if m.get("completed", False)]
        
        if not completed:
            warnings.append(f"ESPN returned 0 completed matches for team {team_name}")
            return None
        
        # Calcular estadísticas agregadas
        total_goals_scored = 0
        total_goals_conceded = 0
        total_shots_on_target_for = 0
        total_shots_on_target_against = 0
        total_corners_for = 0
        total_fouls_for = 0
        wins = 0
        draws = 0
        losses = 0
        clean_sheets = 0
        
        weighted_attack_sum = 0
        weighted_defense_sum = 0
        weight_total = 0
        
        for match in completed:
            stage = match.get("stage", "group")
            weight = self._get_stage_weight(stage)
            
            # Determinar si es home o away
            home_team = match.get("home_team", "")
            away_team = match.get("away_team", "")
            is_home = (team_name == home_team) or (team_name.lower() == home_team.lower())
            
            # Scores
            if is_home:
                goals_for = match.get("home_score") or 0
                goals_against = match.get("away_score") or 0
                winner = match.get("home_winner")
                stats = match.get("stats", {})
                shots_on_target = stats.get("home_shots_on_target") or 0
                shots_on_target_allowed = stats.get("away_shots_on_target") or 0
                corners = stats.get("home_corners") or 0
                fouls = stats.get("home_fouls") or 0
            else:
                goals_for = match.get("away_score") or 0
                goals_against = match.get("home_score") or 0
                winner = match.get("away_winner")
                stats = match.get("stats", {})
                shots_on_target = stats.get("away_shots_on_target") or 0
                shots_on_target_allowed = stats.get("home_shots_on_target") or 0
                corners = stats.get("away_corners") or 0
                fouls = stats.get("away_fouls") or 0
            
            total_goals_scored += goals_for
            total_goals_conceded += goals_against
            total_shots_on_target_for += shots_on_target
            total_shots_on_target_against += shots_on_target_allowed
            total_corners_for += corners
            total_fouls_for += fouls
            
            if winner is True:
                wins += 1
            elif winner is False:
                losses += 1
            else:
                draws += 1
            
            if goals_against == 0:
                clean_sheets += 1
            
            # Weighted sums para attack/defense signal
            weighted_attack_sum += weight * goals_for
            weighted_defense_sum += weight * goals_against
            weight_total += weight
        
        n_matches = len(completed)
        avg_goals_scored = total_goals_scored / n_matches if n_matches > 0 else 0
        avg_goals_conceded = total_goals_conceded / n_matches if n_matches > 0 else 0
        avg_sot_for = total_shots_on_target_for / n_matches if n_matches > 0 else 0
        avg_sot_against = total_shots_on_target_against / n_matches if n_matches > 0 else 0
        avg_corners = total_corners_for / n_matches if n_matches > 0 else 0
        avg_fouls = total_fouls_for / n_matches if n_matches > 0 else 0
        
        # Fórmula conservadora para attack/defense signals
        # attack_signal = 0.7 * goals_scored_avg + 0.3 * max(shots_on_target_avg, 0.5)
        # defense_signal = 0.7 * goals_conceded_avg + 0.3 * max(shots_on_target_allowed_avg, 0.5)
        attack_signal = 0.7 * avg_goals_scored + 0.3 * max(avg_sot_for, 0.5)
        defense_signal = 0.7 * avg_goals_conceded + 0.3 * max(avg_sot_against, 0.5)
        
        # Normalizar contra promedio base razonable (~1.35 goles por equipo por partido)
        base_avg = 1.35
        lambda_attack_espn = attack_signal / base_avg if base_avg > 0 else 1.0
        lambda_defense_espn = defense_signal / base_avg if base_avg > 0 else 1.0
        
        # Clippear a rangos razonables
        lambda_attack_espn = max(0.7, min(2.4, lambda_attack_espn))
        lambda_defense_espn = max(0.7, min(2.2, lambda_defense_espn))
        
        # Shrinkage hacia el rating/base cuando haya pocos partidos
        base_attack = base_rating.get("attack", 1.10)
        base_defense = base_rating.get("defense", 1.00)
        
        if n_matches < 3:
            # Mezclar: más peso al base rating
            shrink_factor = n_matches / 3.0  # 0 a 1
            lambda_attack_espn = (1 - shrink_factor) * base_attack + shrink_factor * lambda_attack_espn
            lambda_defense_espn = (1 - shrink_factor) * base_defense + shrink_factor * lambda_defense_espn
            warnings.append(f"Using blended ESPN/base profile for team {team_name} (only {n_matches} matches)")
        
        # Form string
        form_str = f"W{wins} D{draws} L{losses}"
        
        recent_form = {
            "record": form_str,
            "goals_scored_avg": round(avg_goals_scored, 2),
            "goals_conceded_avg": round(avg_goals_conceded, 2),
            "btts_rate": round((n_matches - clean_sheets) / n_matches, 2) if n_matches > 0 else 0.0,
            "corners_avg": round(avg_corners, 1),
            "cards_avg": round(avg_fouls / 2.0, 1),  # proxy
            "clean_sheets": clean_sheets,
            "form": form_str,
            "data_source": "espn_world_cup",
            "fifa_rank": base_rating.get("fifa_rank", 100),
            "matches_played": n_matches,
            "weighted_attack_signal": round(attack_signal, 3),
            "weighted_defense_signal": round(defense_signal, 3),
        }
        
        wc_form = {
            "played": n_matches,
            "record": form_str,
            "goals_scored": total_goals_scored,
            "goals_conceded": total_goals_conceded,
            "matches": completed[-5:]  # últimos 5
        }
        
        return {
            "lambda_attack": lambda_attack_espn,
            "lambda_defense": lambda_defense_espn,
            "recent_form": recent_form,
            "wc_form": wc_form,
            "warnings": warnings,
            "effective_weight_matches": weight_total,
        }

    def build_team_profile(
        self,
        team_name: str,
        match_date: str,
        include_wc_matches: bool = True
    ) -> TeamProfile:
        """
        Construye el perfil del equipo usando prioridad de fuentes:
        1. ESPN World Cup matches recientes del equipo
        2. dataset/rating estático existente
        3. prior por confederación/default
        
        Si hay >= 3 partidos completados de ESPN, usar principalmente ESPN.
        Si hay 1-2 partidos, mezclar 60% rating/base + 40% ESPN.
        Si hay 0 partidos, fallback al sistema actual.
        """
        teams = self._ratings.get("teams", {})
        default = self._ratings.get("default", {"attack": 1.10, "defense": 1.00, "fifa_rank": 100})
        
        rating = teams.get(team_name, default)
        if team_name not in teams:
            logger.warning(f"Team '{team_name}' not in ratings_wc2026.json, using default.")
        
        base_lambda_attack = rating["attack"]
        base_lambda_defense = rating["defense"]
        fifa_rank = rating.get("fifa_rank", 100)
        
        data_source = "static_ratings"
        warnings = []
        lambda_attack = base_lambda_attack
        lambda_defense = base_lambda_defense
        recent_form = None
        wc_form = None
        effective_weight = 0.0
        
        # Intentar ESPN primero si está disponible
        if self.espn and include_wc_matches:
            try:
                matches = self.espn.get_recent_team_matches(team_name, days_back=60, max_matches=8)
                completed_matches = [m for m in matches if m.get("completed", False)]
                
                if len(completed_matches) >= 3:
                    # Usar ESPN principalmente
                    espn_data = self._compute_espn_profile(team_name, matches, rating)
                    if espn_data:
                        lambda_attack = espn_data["lambda_attack"]
                        lambda_defense = espn_data["lambda_defense"]
                        recent_form = espn_data["recent_form"]
                        wc_form = espn_data["wc_form"]
                        warnings.extend(espn_data["warnings"])
                        effective_weight = espn_data["effective_weight_matches"]
                        data_source = "espn_world_cup"
                        logger.info(f"Using ESPN profile for {team_name} ({len(completed_matches)} matches)")
                
                elif len(completed_matches) > 0:
                    # Mezclar ESPN + base
                    espn_data = self._compute_espn_profile(team_name, matches, rating)
                    if espn_data:
                        # Blend 60% base + 40% ESPN
                        blend_espn = 0.4
                        blend_base = 0.6
                        lambda_attack = blend_base * base_lambda_attack + blend_espn * espn_data["lambda_attack"]
                        lambda_defense = blend_base * base_lambda_defense + blend_espn * espn_data["lambda_defense"]
                        recent_form = espn_data["recent_form"]
                        wc_form = espn_data["wc_form"]
                        warnings.extend(espn_data["warnings"])
                        effective_weight = espn_data["effective_weight_matches"]
                        data_source = "espn_world_cup_blended_static"
                        logger.info(f"Using blended ESPN/base profile for {team_name}")
                
                else:
                    warnings.append(f"ESPN returned 0 completed matches for team {team_name}")
                    
            except Exception as e:
                logger.warning(f"ESPN client failed for {team_name}: {e}. Falling back to static ratings.")
                warnings.append(f"ESPN error: {str(e)}")
        
        # Fallback: construir recent_form desde ratings si no hay ESPN
        if recent_form is None:
            recent_form = {
                "record": "N/A",
                "goals_scored_avg": round(base_lambda_attack * 1.35, 2),
                "goals_conceded_avg": round(base_lambda_defense * 1.35, 2),
                "btts_rate": 0.50,
                "corners_avg": 5.0,
                "cards_avg": 2.0,
                "clean_sheets": 0,
                "form": "N/A",
                "data_source": data_source,
                "fifa_rank": fifa_rank
            }
            warnings.append(f"Falling back to static ratings for team {team_name}")
        
        if wc_form is None:
            wc_form = {
                "played": 0, "record": "N/A",
                "goals_scored": 0, "goals_conceded": 0, "matches": []
            }
        
        return TeamProfile(
            team_name=team_name,
            lambda_attack=lambda_attack,
            lambda_defense=lambda_defense,
            recent_form=recent_form,
            wc_form=wc_form,
            corners_lambda=5.0,
            cards_lambda=2.0,
            effective_weight_matches=effective_weight,
            data_warnings=warnings,
            data_source=data_source
        )
