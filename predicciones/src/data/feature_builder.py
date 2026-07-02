import math
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Confederation priors for fallback when no match data is available
# (lambda_attack, lambda_defense) relative to league average
CONFEDERATION_PRIORS = {
    "UEFA":      (1.45, 1.10),  # Europa — alto ataque y defensa
    "CONMEBOL":  (1.40, 1.05),  # Sudamérica
    "CONCACAF":  (1.20, 0.95),  # Norteamérica/Centroamérica
    "CAF":       (1.15, 0.90),  # África
    "AFC":       (1.10, 0.88),  # Asia
    "OFC":       (1.00, 0.85),  # Oceanía
    "UNKNOWN":   (1.25, 1.00),  # Default si no se puede inferir
}

# Mapping de equipos conocidos a confederación
TEAM_CONFEDERATION = {
    # UEFA
    "Inglaterra": "UEFA", "Francia": "UEFA", "Alemania": "UEFA",
    "España": "UEFA", "Italia": "UEFA", "Portugal": "UEFA",
    "Países Bajos": "UEFA", "Bélgica": "UEFA", "Croatia": "UEFA",
    "Serbia": "UEFA", "Dinamarca": "UEFA", "Austria": "UEFA",
    "Suiza": "UEFA", "Turquía": "UEFA", "Polonia": "UEFA",
    "Escocia": "UEFA", "Ucrania": "UEFA", "Hungría": "UEFA",
    "República Checa": "UEFA", "Rumania": "UEFA", "Eslovaquia": "UEFA",
    "Eslovenia": "UEFA", "Albania": "UEFA", "Georgia": "UEFA",
    "England": "UEFA", "France": "UEFA", "Germany": "UEFA",
    "Spain": "UEFA", "Italy": "UEFA", "Portugal": "UEFA",
    "Netherlands": "UEFA", "Belgium": "UEFA", "Croatia": "UEFA",
    "Serbia": "UEFA", "Denmark": "UEFA", "Austria": "UEFA",
    "Switzerland": "UEFA", "Turkey": "UEFA", "Poland": "UEFA",
    "Scotland": "UEFA", "Ukraine": "UEFA", "Hungary": "UEFA",
    "Czech Republic": "UEFA", "Romania": "UEFA", "Slovakia": "UEFA",
    "Slovenia": "UEFA", "Albania": "UEFA", "Georgia": "UEFA",

    # CONMEBOL
    "Brasil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL", "Perú": "CONMEBOL", "Bolivia": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL", "Peru": "CONMEBOL", "Bolivia": "CONMEBOL",
    "Venezuela": "CONMEBOL",

    # CONCACAF
    "México": "CONCACAF", "Estados Unidos": "CONCACAF", "Canadá": "CONCACAF",
    "Costa Rica": "CONCACAF", "Honduras": "CONCACAF", "Jamaica": "CONCACAF",
    "Panamá": "CONCACAF", "El Salvador": "CONCACAF", "Guatemala": "CONCACAF",
    "Trinidad y Tobago": "CONCACAF",
    "Mexico": "CONCACAF", "United States": "CONCACAF", "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF", "Honduras": "CONCACAF", "Jamaica": "CONCACAF",
    "Panama": "CONCACAF", "El Salvador": "CONCACAF", "Guatemala": "CONCACAF",
    "Trinidad and Tobago": "CONCACAF", "USA": "CONCACAF",

    # CAF
    "Senegal": "CAF", "Marruecos": "CAF", "Nigeria": "CAF",
    "Egipto": "CAF", "Costa de Marfil": "CAF", "Ghana": "CAF",
    "Camerún": "CAF", "Mali": "CAF", "Argelia": "CAF", "Túnez": "CAF",
    "Sudáfrica": "CAF", "Congo": "CAF",
    "Republica Democratica del Congo": "CAF",
    "República Democrática del Congo": "CAF",
    "RD Congo": "CAF", "DR Congo": "CAF",
    "DRC": "CAF",
    "Senegal": "CAF", "Morocco": "CAF", "Nigeria": "CAF",
    "Egypt": "CAF", "Ivory Coast": "CAF", "Ghana": "CAF",
    "Cameroon": "CAF", "Mali": "CAF", "Algeria": "CAF", "Tunisia": "CAF",
    "South Africa": "CAF", "Congo": "CAF",
    "Democratic Republic of the Congo": "CAF",

    # AFC
    "Japón": "AFC", "Corea del Sur": "AFC", "Arabia Saudita": "AFC",
    "Irán": "AFC", "Australia": "AFC", "Qatar": "AFC",
    "China": "AFC", "Irak": "AFC", "Emiratos Árabes": "AFC",
    "Uzbekistán": "AFC",
    "Japan": "AFC", "South Korea": "AFC", "Saudi Arabia": "AFC",
    "Iran": "AFC", "Australia": "AFC", "Qatar": "AFC",
    "China": "AFC", "Iraq": "AFC", "United Arab Emirates": "AFC",
    "Uzbekistan": "AFC",
}

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
    def __init__(self, api_client):
        self.api = api_client

    def _parse_date(self, date_str: Any) -> datetime:
        if isinstance(date_str, datetime):
            return date_str
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(str(date_str).split(".")[0], fmt)
            except ValueError:
                continue
        # Fallback to today
        return datetime.today()

    def _apply_time_decay(
        self, 
        matches: List[Dict[str, Any]], 
        reference_date: datetime,
        phi: float = 0.0065
    ) -> List[Dict[str, Any]]:
        """Adds field 'weight' to each match according to exponential decay."""
        decayed_matches = []
        for match in matches:
            match_date = self._parse_date(match.get("date"))
            days_ago = (reference_date - match_date).days
            
            # Exclude matches older than 36 months (1095 days)
            if days_ago <= 1095:
                # Ensure days_ago isn't negative for future scheduled/today matches
                days_ago = max(0, days_ago)
                weight = math.exp(-phi * days_ago)
                m_copy = match.copy()
                m_copy["weight"] = weight
                m_copy["days_ago"] = days_ago
                decayed_matches.append(m_copy)
        return decayed_matches

    def _boost_wc_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        World Cup matches get a weight multiplier based on stage.
        - Group stage: weight *= 1.3
        - Octavos/Cuartos: weight *= 1.5
        - Semi/Final: weight *= 1.8
        """
        boosted = []
        for match in matches:
            m_copy = match.copy()
            comp = str(m_copy.get("competition", "")).lower()
            if "world cup" in comp or "mundial" in comp:
                stage = str(m_copy.get("stage", "group")).lower()
                if "group" in stage:
                    m_copy["weight"] = m_copy.get("weight", 1.0) * 1.3
                elif any(x in stage for x in ("16", "octavos", "quarter", "cuartos", "qf")):
                    m_copy["weight"] = m_copy.get("weight", 1.0) * 1.5
                elif any(x in stage for x in ("semi", "final", "sf", "third")):
                    m_copy["weight"] = m_copy.get("weight", 1.0) * 1.8
                else:
                    # Default WC boost
                    m_copy["weight"] = m_copy.get("weight", 1.0) * 1.4
            boosted.append(m_copy)
        return boosted

    def build_team_profile(
        self, 
        team_name: str,
        match_date: str,
        include_wc_matches: bool = True
    ) -> TeamProfile:
        ref_date = self._parse_date(match_date)
        
        # Accumulate warnings for data quality reporting
        data_warnings = []
        
        # 1. Fetch matches
        wc_matches = []
        if include_wc_matches:
            try:
                all_wc = self.api.get_world_cup_fixtures()
                wc_matches = [
                    m for m in all_wc 
                    if m.get("home_team") == team_name or m.get("away_team") == team_name
                ]
            except Exception as e:
                msg = f"API fetch failed for '{team_name}' (WC fixtures): {type(e).__name__}: {e}"
                logger.warning(msg)
                data_warnings.append(msg)

        external_matches = []
        try:
            external_matches = self.api.get_team_last_matches(team_name, n=10)
        except Exception as e:
            msg = f"API fetch failed for '{team_name}' (external matches): {type(e).__name__}: {e}"
            logger.warning(msg)
            data_warnings.append(msg)

        # Combine
        all_matches = []
        for m in wc_matches:
            m["is_wc"] = True
            all_matches.append(m)
        for m in external_matches:
            m["is_wc"] = False
            all_matches.append(m)

        # 2. Process metrics
        # If no matches found at all, use confederation-based prior
        if not all_matches:
            confederation = TEAM_CONFEDERATION.get(team_name, "UNKNOWN")
            prior_attack, prior_defense = CONFEDERATION_PRIORS[confederation]
            logger.warning(
                f"No match history for '{team_name}'. "
                f"Using confederation prior: {confederation} "
                f"(λ_att={prior_attack}, λ_def={prior_defense})"
            )
            return TeamProfile(
                team_name=team_name,
                lambda_attack=prior_attack,
                lambda_defense=prior_defense,
                recent_form={
                    "record": "N/A",
                    "goals_scored_avg": prior_attack * 1.35,
                    "goals_conceded_avg": prior_defense * 1.35,
                    "btts_rate": 0.5,
                    "corners_avg": 5.0,
                    "cards_avg": 2.0,
                    "clean_sheets": 0,
                    "form": "?????",
                    "data_source": f"confederation_prior_{confederation}"
                },
                wc_form={
                    "played": 0,
                    "record": "N/A",
                    "goals_scored": 0,
                    "goals_conceded": 0,
                    "matches": []
                },
                corners_lambda=5.0,
                cards_lambda=2.0,
                effective_weight_matches=0.0,
                data_warnings=data_warnings
            )

        # Decay & Boost weights
        decayed = self._apply_time_decay(all_matches, ref_date)
        boosted = self._boost_wc_matches(decayed)
        
        # Calculate weighted metrics
        total_weight = 0.0
        weighted_scored = 0.0
        weighted_conceded = 0.0
        weighted_corners = 0.0
        weighted_cards = 0.0
        
        total_corners_weight = 0.0
        total_cards_weight = 0.0
        
        # We also need unweighted recent stats for reporting
        recent_10 = [m for m in all_matches if not m.get("is_wc", False)][:10]
        if not recent_10:
            recent_10 = all_matches[:10] # fallback if only WC matches exist
            
        wc_played = [m for m in all_matches if m.get("is_wc", False)]

        # Calculate weighted averages
        for m in boosted:
            w = m["weight"]
            is_home = m["home_team"] == team_name
            scored = m["home_score"] if is_home else m["away_score"]
            conceded = m["away_score"] if is_home else m["home_score"]
            
            weighted_scored += scored * w
            weighted_conceded += conceded * w
            total_weight += w
            
            # Optional corners and cards
            if "corners" in m:
                weighted_corners += m["corners"] * w
                total_corners_weight += w
            if "cards" in m:
                weighted_cards += m["cards"] * w
                total_cards_weight += w

        if total_weight > 0:
            avg_scored = weighted_scored / total_weight
            avg_conceded = weighted_conceded / total_weight
        else:
            avg_scored = 1.35
            avg_conceded = 1.35

        corners_lambda = (weighted_corners / total_corners_weight) if total_corners_weight > 0 else 5.0
        cards_lambda = (weighted_cards / total_cards_weight) if total_cards_weight > 0 else 2.0

        # Calculate recent_form stats
        r_wins = 0
        r_draws = 0
        r_losses = 0
        r_scored = 0
        r_conceded = 0
        r_btts = 0
        r_corners = 0
        r_cards = 0
        r_clean_sheets = 0
        form_letters = []

        for m in recent_10:
            is_home = m["home_team"] == team_name
            scored = m["home_score"] if is_home else m["away_score"]
            conceded = m["away_score"] if is_home else m["home_score"]
            
            r_scored += scored
            r_conceded += conceded
            
            if scored > conceded:
                r_wins += 1
                form_letters.append("W")
            elif scored == conceded:
                r_draws += 1
                form_letters.append("D")
            else:
                r_losses += 1
                form_letters.append("L")
                
            if scored > 0 and conceded > 0:
                r_btts += 1
            if conceded == 0:
                r_clean_sheets += 1
                
            r_corners += m.get("corners", 5)
            r_cards += m.get("cards", 2)

        n_recent = len(recent_10) or 1
        recent_form_dict = {
            "record": f"W{r_wins} D{r_draws} L{r_losses}",
            "goals_scored_avg": round(r_scored / n_recent, 2),
            "goals_conceded_avg": round(r_conceded / n_recent, 2),
            "btts_rate": round(r_btts / n_recent, 2),
            "corners_avg": round(r_corners / n_recent, 2),
            "cards_avg": round(r_cards / n_recent, 2),
            "clean_sheets": r_clean_sheets,
            "form": "".join(form_letters[:5])
        }

        # WC performance stats
        wc_wins = 0
        wc_draws = 0
        wc_losses = 0
        wc_scored = 0
        wc_conceded = 0
        wc_matches_list = []

        for m in wc_played:
            is_home = m["home_team"] == team_name
            scored = m["home_score"] if is_home else m["away_score"]
            conceded = m["away_score"] if is_home else m["home_score"]
            opp = m["away_team"] if is_home else m["home_team"]
            
            wc_scored += scored
            wc_conceded += conceded
            
            if scored > conceded:
                wc_wins += 1
                res = "W"
            elif scored == conceded:
                wc_draws += 1
                res = "D"
            else:
                wc_losses += 1
                res = "L"
                
            wc_matches_list.append({
                "vs": opp,
                "result": f"{scored}-{conceded}",
                "xg_for": m.get("xg_for", scored),
                "xg_against": m.get("xg_against", conceded)
            })

        wc_form_dict = {
            "played": len(wc_played),
            "record": f"W{wc_wins} D{wc_draws} L{wc_losses}",
            "goals_scored": wc_scored,
            "goals_conceded": wc_conceded,
            "matches": wc_matches_list
        }

        # Conversion to ratings
        LEAGUE_AVG_GOALS = 1.35
        # lambda_attack/lambda_defense represent strength relative to average
        lambda_attack = max(0.1, avg_scored / LEAGUE_AVG_GOALS)
        lambda_defense = max(0.1, avg_conceded / LEAGUE_AVG_GOALS)

        # Calculate effective weight (sum of weights, not just count)
        effective_weight = sum(m.get("weight", 1.0) for m in boosted)

        return TeamProfile(
            team_name=team_name,
            lambda_attack=lambda_attack,
            lambda_defense=lambda_defense,
            recent_form=recent_form_dict,
            wc_form=wc_form_dict,
            corners_lambda=corners_lambda,
            cards_lambda=cards_lambda,
            effective_weight_matches=effective_weight,
            data_warnings=data_warnings
        )
