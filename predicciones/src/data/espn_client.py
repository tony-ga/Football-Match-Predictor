"""
ESPN World Cup API Client.

Integración con la API pública no documentada de ESPN para el Mundial.
Endpoint principal: https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world

Soporta:
- scoreboard: eventos/partidos del Mundial, competidores, estado, venue
- summary: detalle del partido y boxscore si está disponible
- dates=YYYYMMDD o rango YYYYMMDD-YYYYMMDD y limit
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests

logger = logging.getLogger(__name__)


class EspnWorldCupClient:
    """Cliente para la API de ESPN FIFA World Cup."""
    
    BASE_SITE = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"
    
    def __init__(self, timeout: int = 20):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        })
        
        # Cargar mappings de equipos
        self.mappings_path = Path(__file__).parent / "team_mappings.json"
        self.aliases = {}
        if self.mappings_path.exists():
            try:
                with open(self.mappings_path, "r", encoding="utf-8") as f:
                    mappings = json.load(f)
                    self.aliases = mappings.get("aliases", {})
            except Exception as e:
                logger.warning(f"Failed to load team mappings: {e}")
        
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
    
    def _make_request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Realiza una petición HTTP con retries y backoff exponencial.
        Maneja timeouts, 429, 500s y JSON inválido.
        """
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                logger.info(f"ESPN API request: {url} params={params}")
                response = self.session.get(url, params=params, timeout=self.timeout)
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", base_delay * (2 ** attempt)))
                    logger.warning(f"ESPN rate limited (429). Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                
                if response.status_code >= 500:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"ESPN server error ({response.status_code}). Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                
                response.raise_for_status()
                
                try:
                    return response.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from ESPN: {e}")
                    return None
                    
            except requests.Timeout:
                logger.warning(f"Request timeout after {self.timeout}s (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
            except requests.RequestException as e:
                logger.error(f"Request failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                else:
                    return None
        
        return None
    
    def get_scoreboard(self, dates: Optional[str] = None, limit: int = 200, **extra_params) -> Dict[str, Any]:
        """
        GET /scoreboard con params opcionales dates y limit.
        
        Args:
            dates: Formato YYYYMMDD o rango YYYYMMDD-YYYYMMDD
            limit: Número máximo de eventos a retornar
        
        Returns:
            JSON crudo del scoreboard o dict vacío si falla.
        """
        url = f"{self.BASE_SITE}/scoreboard"
        params = {"limit": limit}
        if dates:
            params["dates"] = dates
        for key, value in extra_params.items():
            if value is not None:
                params[key] = value
        
        result = self._make_request(url, params)
        return result if result else {}
    
    def get_summary(self, event_id: str) -> Dict[str, Any]:
        """
        GET /summary?event={event_id}
        
        Args:
            event_id: ID del evento/partido
        
        Returns:
            JSON crudo del summary o dict vacío si falla.
        """
        url = f"{self.BASE_SITE}/summary"
        params = {"event": event_id}
        
        result = self._make_request(url, params)
        return result if result else {}
    
    def normalize_team_name(self, team_name: str) -> str:
        """
        Normaliza aliases: Mexico/México, Inglaterra/England, DR Congo/Republica Democratica del Congo, etc.
        
        Args:
            team_name: Nombre del equipo tal como viene de ESPN
        
        Returns:
            Nombre normalizado según mappings.
        """
        if not team_name:
            return ""
        
        # Primero buscar coincidencia directa en aliases
        if team_name in self.aliases:
            return self.aliases[team_name]
        
        # Búsqueda case-insensitive
        for alias, canonical in self.aliases.items():
            if alias.lower() == team_name.lower():
                return canonical
        
        # Si no hay alias, retornar el nombre original
        return team_name
    
    def _normalize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convierte el JSON crudo ESPN a formato interno uniforme.
        
        El dict normalizado incluye:
        - event_id, date, competition, stage, status, completed
        - neutral_venue, venue
        - home_team, away_team, home_score, away_score
        - home_winner, away_winner
        - stats: shots, SOT, possession, corners, fouls
        - odds: home, draw, away
        """
        competitions = event.get("competitions", [])
        if not competitions:
            return {}
        
        comp = competitions[0]
        competitors = comp.get("competitors", [])
        
        # Identificar home/away
        home_comp = None
        away_comp = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home_comp = c
            elif c.get("homeAway") == "away":
                away_comp = c
        
        if not home_comp or not away_comp:
            # Intentar usar los dos primeros si no hay homeAway claro
            if len(competitors) >= 2:
                home_comp = competitors[0]
                away_comp = competitors[1]
            else:
                return {}
        
        # Extraer información básica
        event_id = str(event.get("id", ""))
        event_date = event.get("date", "")
        
        # Status
        status = comp.get("status", {})
        status_type = status.get("type", {})
        status_name = status_type.get("name", "").lower()
        status_state = status_type.get("state", "").lower()  # some APIs use "state"
        
        # Determinar estado: pre, in, post
        # ESPN usa STATUS_FINAL, STATUS_IN_PROGRESS, STATUS_SCHEDULED, etc.
        if "final" in status_name or "ended" in status_name or status_state == "post":
            normalized_status = "post"
            completed = True
        elif "progress" in status_name or "live" in status_name or "active" in status_name or status_state == "in":
            normalized_status = "in"
            completed = False
        elif "pre" in status_name or "scheduled" in status_name or status_state == "pre":
            normalized_status = "pre"
            completed = False
        else:
            # Default basado en si hay scores
            if home_score is not None and away_score is not None:
                normalized_status = "post"
                completed = True
            else:
                normalized_status = "pre"
                completed = False
        
        # Scores
        home_score = home_comp.get("score")
        away_score = away_comp.get("score")
        
        try:
            home_score = int(home_score) if home_score is not None else None
            away_score = int(away_score) if away_score is not None else None
        except (ValueError, TypeError):
            home_score = None
            away_score = None
        
        # Winners
        home_winner = home_comp.get("winner", False) if completed else None
        away_winner = away_comp.get("winner", False) if completed else None
        
        # Venue
        venue_info = comp.get("venue", {})
        venue_name = venue_info.get("fullName") or venue_info.get("address", {}).get("city")
        neutral_venue = venue_info.get("neutral", None)
        
        # Competition y stage
        league_info = event.get("league", {})
        competition = league_info.get("name", "FIFA World Cup")
        
        # Inferir stage desde season.slug, week, notes, type.shortDetail
        season = event.get("season", {})
        season_slug = season.get("slug", "")
        week = event.get("week", "")
        notes = comp.get("notes", "")
        type_info = comp.get("type", {})
        short_detail = type_info.get("shortDetail", "")
        
        # Determinar stage
        stage = "group"  # default
        stage_lower = f"{season_slug} {week} {notes} {short_detail}".lower()
        
        if "third" in stage_lower or "tercer" in stage_lower:
            stage = "third_place"
        elif "semi" in stage_lower:
            stage = "semi_final"
        elif "quarter" in stage_lower or "cuartos" in stage_lower:
            stage = "quarter_final"
        elif "round of 16" in stage_lower or "octavos" in stage_lower:
            stage = "round_of_16"
        elif "round of 32" in stage_lower:
            stage = "round_of_32"
        elif "final" in stage_lower:
            stage = "final"
        
        # Stats desde scoreboard (si existen)
        stats = self._extract_stats_from_competition(comp)
        
        # Odds (si existen)
        odds = self._extract_odds(comp)
        
        # Nombres de equipos normalizados
        home_team_raw = home_comp.get("team", {}).get("displayName", "")
        away_team_raw = away_comp.get("team", {}).get("displayName", "")
        home_team = self.normalize_team_name(home_team_raw)
        away_team = self.normalize_team_name(away_team_raw)
        
        return {
            "event_id": event_id,
            "date": event_date,
            "competition": competition,
            "stage": stage,
            "status": normalized_status,
            "completed": completed,
            "neutral_venue": neutral_venue,
            "venue": venue_name,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "home_winner": home_winner,
            "away_winner": away_winner,
            "stats": stats,
            "odds": odds,
            "_raw_home_name": home_team_raw,
            "_raw_away_name": away_team_raw,
        }
    
    def _extract_stats_from_competition(self, comp: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae estadísticas desde el bloque de competition del scoreboard."""
        stats = {
            "home_shots": None,
            "away_shots": None,
            "home_shots_on_target": None,
            "away_shots_on_target": None,
            "home_possession": None,
            "away_possession": None,
            "home_corners": None,
            "away_corners": None,
            "home_fouls": None,
            "away_fouls": None,
        }
        
        statistics = comp.get("statistics", [])
        if not statistics:
            return stats
        
        # Buscar competidores para mapear stats
        competitors = comp.get("competitors", [])
        home_idx = None
        away_idx = None
        for i, c in enumerate(competitors):
            if c.get("homeAway") == "home":
                home_idx = i
            elif c.get("homeAway") == "away":
                away_idx = i
        
        for stat_block in statistics:
            name = stat_block.get("name", "").lower()
            displays = stat_block.get("displayValue", [])
            
            # Mapeo de nombres de estadísticas
            if "shots" in name and "on target" not in name and "goal" not in name:
                if len(displays) >= 2:
                    stats["home_shots"] = self._parse_float(displays[home_idx]) if home_idx is not None else None
                    stats["away_shots"] = self._parse_float(displays[away_idx]) if away_idx is not None else None
            elif "shot" in name and "target" in name:
                if len(displays) >= 2:
                    stats["home_shots_on_target"] = self._parse_float(displays[home_idx]) if home_idx is not None else None
                    stats["away_shots_on_target"] = self._parse_float(displays[away_idx]) if away_idx is not None else None
            elif "possession" in name or "posesión" in name:
                if len(displays) >= 2:
                    stats["home_possession"] = self._parse_float(displays[home_idx]) if home_idx is not None else None
                    stats["away_possession"] = self._parse_float(displays[away_idx]) if away_idx is not None else None
            elif "corner" in name:
                if len(displays) >= 2:
                    stats["home_corners"] = self._parse_float(displays[home_idx]) if home_idx is not None else None
                    stats["away_corners"] = self._parse_float(displays[away_idx]) if away_idx is not None else None
            elif "foul" in name:
                if len(displays) >= 2:
                    stats["home_fouls"] = self._parse_float(displays[home_idx]) if home_idx is not None else None
                    stats["away_fouls"] = self._parse_float(displays[away_idx]) if away_idx is not None else None
        
        return stats
    
    def _extract_stats_from_summary(self, summary_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intenta sacar shots, SOT, possession, corners, fouls desde summary.
        
        Busca en bloques de lineups, boxscore, formations, rosters o referencias a Core API.
        """
        stats = {
            "home_shots": None,
            "away_shots": None,
            "home_shots_on_target": None,
            "away_shots_on_target": None,
            "home_possession": None,
            "away_possession": None,
            "home_corners": None,
            "away_corners": None,
            "home_fouls": None,
            "away_fouls": None,
        }
        
        if not summary_json:
            return stats
        
        # Buscar en boxscore
        boxscore = summary_json.get("boxscore", {})
        teams_stats = boxscore.get("teams", [])
        
        home_stats = None
        away_stats = None
        for t in teams_stats:
            if t.get("homeAway") == "home":
                home_stats = t.get("statistics", [])
            elif t.get("homeAway") == "away":
                away_stats = t.get("statistics", [])
        
        # Parsear estadísticas desde arrays de stats
        for stat_list, is_home in [(home_stats, True), (away_stats, False)]:
            if not stat_list:
                continue
            
            for stat in stat_list:
                name = stat.get("name", "").lower() if isinstance(stat, dict) else ""
                value = stat.get("displayValue") or stat.get("value") if isinstance(stat, dict) else None
                value = self._parse_float(value)
                
                if "shots" in name and "on target" not in name:
                    if is_home:
                        stats["home_shots"] = value
                    else:
                        stats["away_shots"] = value
                elif "shot" in name and "target" in name:
                    if is_home:
                        stats["home_shots_on_target"] = value
                    else:
                        stats["away_shots_on_target"] = value
                elif "possession" in name:
                    if is_home:
                        stats["home_possession"] = value
                    else:
                        stats["away_possession"] = value
                elif "corner" in name:
                    if is_home:
                        stats["home_corners"] = value
                    else:
                        stats["away_corners"] = value
                elif "foul" in name:
                    if is_home:
                        stats["home_fouls"] = value
                    else:
                        stats["away_fouls"] = value
        
        # También intentar desde competitions[0].statistics en summary
        competitions = summary_json.get("competitions", [])
        if competitions:
            comp_stats = self._extract_stats_from_competition(competitions[0])
            for key, val in comp_stats.items():
                if val is not None:
                    stats[key] = val
        
        return stats
    
    def _extract_odds(self, comp: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae odds si están disponibles en el JSON."""
        odds = {
            "home": None,
            "draw": None,
            "away": None,
        }
        
        # Buscar en odds block
        odds_data = comp.get("odds", [])
        if not odds_data:
            return odds
        
        # ESPN puede tener múltiples bookmakers, tomar el primero
        if isinstance(odds_data, list) and len(odds_data) > 0:
            first_odds = odds_data[0]
            if first_odds is None or not isinstance(first_odds, dict):
                return odds
            
            # Intentar extraer desde moneyline (formato moderno de ESPN)
            moneyline = first_odds.get("moneyline", {})
            if moneyline:
                home_ml = moneyline.get("home", {})
                away_ml = moneyline.get("away", {})
                draw_ml = moneyline.get("draw", {})
                
                # Extraer odds actuales (pueden ser strings como "+155" o "-200")
                home_val = home_ml.get("current", {}).get("odds")
                away_val = away_ml.get("current", {}).get("odds")
                draw_val = draw_ml.get("current", {}).get("odds")
                
                if home_val:
                    odds["home"] = self._parse_moneyline_to_decimal(home_val)
                if away_val:
                    odds["away"] = self._parse_moneyline_to_decimal(away_val)
                if draw_val:
                    odds["draw"] = self._parse_moneyline_to_decimal(draw_val)
            
            # Fallback al formato antiguo con details
            if odds["home"] is None:
                details = first_odds.get("details", [])
                if details and isinstance(details, list):
                    for detail in details:
                        if not isinstance(detail, dict):
                            continue
                        name = detail.get("name", "").lower()
                        value = detail.get("value") or detail.get("decimalValue")
                        value = self._parse_float(value)
                        
                        if value is None:
                            continue
                        
                        if "home" in name or "local" in name:
                            odds["home"] = value
                        elif "draw" in name or "tie" in name or "empate" in name:
                            odds["draw"] = value
                        elif "away" in name or "visitor" in name or "visitante" in name:
                            odds["away"] = value
        
        return odds
    
    def _parse_moneyline_to_decimal(self, ml_value: str) -> Optional[float]:
        """Convierte moneyline americano (+155, -200) a decimal."""
        try:
            val = int(ml_value.replace("+", ""))
            if val > 0:
                return round((val / 100) + 1, 2)
            else:
                return round((100 / abs(val)) + 1, 2)
        except (ValueError, TypeError):
            return None
    
    def _parse_float(self, value: Any) -> Optional[float]:
        """Parsea un valor a float, retornando None si falla."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                # Remover % si existe
                value = value.replace("%", "").strip()
                return float(value)
            except ValueError:
                return None
        return None
    
    def get_world_cup_matches(self, dates: Optional[str] = None, limit: int = 200, season_type: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Devuelve todos los partidos normalizados del scoreboard.
        
        Args:
            dates: Rango de fechas opcional
            limit: Límite de eventos
        
        Returns:
            Lista de dicts normalizados.
        """
        extra_params = {}
        if season_type is not None:
            extra_params["seasontype.seasontype"] = season_type
        scoreboard = self.get_scoreboard(dates=dates, limit=limit, **extra_params)
        events = scoreboard.get("events", [])
        
        normalized = []
        for event in events:
            norm = self._normalize_event(event)
            if norm:
                normalized.append(norm)
        
        return normalized
    
    def get_recent_team_matches(
        self,
        team_name: str,
        days_back: int = 60,
        max_matches: int = 8,
        limit_per_request: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Busca partidos recientes de un equipo en una ventana temporal.
        
        Consulta el scoreboard de ESPN con un rango de fechas hacia atrás
        y filtra los eventos donde participe el equipo especificado.
        
        Args:
            team_name: Nombre del equipo (se normaliza automáticamente)
            days_back: Días hacia atrás para buscar (default: 60)
            max_matches: Máximo número de partidos a retornar (default: 8)
            limit_per_request: Límite de eventos por request (default: 500)
        
        Returns:
            Lista de partidos normalizados donde participa el equipo,
            ordenados del más reciente al más antiguo. Nunca devuelve None;
            devuelve [] si no hay resultados.
        """
        import datetime
        
        normalized_name = self.normalize_team_name(team_name)
        logger.info(f"Buscando partidos recientes para: {team_name} (normalizado: {normalized_name})")
        
        # Calcular rango de fechas
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=days_back)
        date_range = f"{start_date.strftime('%Y%m%d')}-{today.strftime('%Y%m%d')}"
        
        logger.debug(f"Consultando ESPN con rango de fechas: {date_range}")
        
        scoreboard = self.get_scoreboard(dates=date_range, limit=limit_per_request)
        events = scoreboard.get("events", [])
        
        if not events:
            logger.warning(f"No events returned from ESPN for {team_name} in last {days_back} days")
            return []
        
        team_matches = []
        for event in events:
            norm = self._normalize_event(event)
            if not norm:
                continue
            
            # Verificar si el equipo participa
            home_team = norm.get("home_team", "")
            away_team = norm.get("away_team", "")
            
            # Comparar con aliases
            if normalized_name not in [home_team, away_team]:
                # Intentar matching con raw names
                raw_home = norm.get("_raw_home_name", "")
                raw_away = norm.get("_raw_away_name", "")
                raw_normalized_home = self.normalize_team_name(raw_home)
                raw_normalized_away = self.normalize_team_name(raw_away)
                
                if normalized_name not in [raw_normalized_home, raw_normalized_away]:
                    continue
            
            # Enriquecer con summary si está completo y faltan stats
            if norm.get("completed") and norm.get("event_id"):
                stats = norm.get("stats", {})
                has_key_stats = any([
                    stats.get("home_shots"),
                    stats.get("home_shots_on_target"),
                    stats.get("home_possession"),
                ])
                
                if not has_key_stats:
                    logger.debug(f"Enriching event {norm['event_id']} with summary...")
                    summary = self.get_summary(norm["event_id"])
                    if summary:
                        summary_stats = self._extract_stats_from_summary(summary)
                        # Merge stats
                        for key, val in summary_stats.items():
                            if val is not None and stats.get(key) is None:
                                stats[key] = val
                        norm["stats"] = stats
            
            team_matches.append(norm)
            
            # Limitar número de partidos
            if len(team_matches) >= max_matches:
                break
        
        # Ordenar por fecha (más reciente primero)
        team_matches.sort(key=lambda x: x.get("date", ""), reverse=True)
        
        logger.info(f"Encontrados {len(team_matches)} partidos para {team_name} en últimos {days_back} días")
        return team_matches
