import os
import json
import logging
import datetime
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from fuzzywuzzy import fuzz, process

logger = logging.getLogger(__name__)

class FootballAPIClient:
    def __init__(self, api_key: str = None, source: str = "auto"):
        # source: "api_football" | "football_data" | "open_football" | "auto"
        self.source = source
        self.api_football_key = api_key or os.getenv("API_FOOTBALL_KEY") or ""
        self.football_data_token = os.getenv("FOOTBALL_DATA_TOKEN") or ""
        
        self.session = requests.Session()
        # In-memory cache for 1 hour
        self.cache = TTLCache(maxsize=200, ttl=3600)
        
        # Determine cache dir
        self.cache_dir = Path(os.getenv("DATA_CACHE_DIR", "./data/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load mappings
        self.mappings_path = Path(__file__).parent / "team_mappings.json"
        if self.mappings_path.exists():
            with open(self.mappings_path, "r", encoding="utf-8") as f:
                self.mappings = json.load(f)
        else:
            self.mappings = {"api_football": {}, "football_data": {}}

    def _get_cache_filename(self, endpoint: str, params: Dict[str, Any]) -> Path:
        date_str = datetime.date.today().isoformat()
        param_str = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
        clean_endpoint = endpoint.replace("/", "_").replace("?", "_").replace("&", "_")
        filename = f"{date_str}_{clean_endpoint}_{param_str}.json"
        # Remove any invalid chars for windows path
        for char in ['\\', ':', '*', '?', '"', '<', '>', '|']:
            filename = filename.replace(char, "_")
        return self.cache_dir / filename

    def _save_to_disk_cache(self, filename: Path, data: Any):
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save disk cache: {e}")

    def _load_from_disk_cache(self, filename: Path) -> Optional[Any]:
        if filename.exists():
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load disk cache: {e}")
        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True
    )
    def _make_request(self, url: str, headers: Dict[str, str], params: Dict[str, Any] = None) -> Dict[str, Any]:
        params = params or {}
        # In-memory cache check
        cache_key = (url, frozenset(params.items()))
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Disk cache check
        endpoint_part = url.split("://")[-1].replace("/", "_")
        disk_file = self._get_cache_filename(endpoint_part, params)
        cached_data = self._load_from_disk_cache(disk_file)
        if cached_data is not None:
            self.cache[cache_key] = cached_data
            return cached_data

        logger.info(f"Making HTTP request to: {url} with params {params}")
        response = self.session.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Cache response
        self.cache[cache_key] = data
        self._save_to_disk_cache(disk_file, data)
        return data

    def resolve_team_id(self, team_name: str, api_type: str = "api_football") -> Optional[int]:
        """Maps common team name to API ID using fuzzy matching."""
        api_mappings = self.mappings.get(api_type, {})
        
        # Direct check
        if team_name in api_mappings:
            return api_mappings[team_name]
            
        # Try case-insensitive matching
        for name, tid in api_mappings.items():
            if name.lower() == team_name.lower():
                return tid
                
        # Fuzzy matching
        choices = list(api_mappings.keys())
        if not choices:
            return None
            
        best_match, score = process.extractOne(team_name, choices, scorer=fuzz.token_sort_ratio)
        if score >= 80:
            resolved_id = api_mappings[best_match]
            logger.info(f"Fuzzy matched: '{team_name}' -> '{best_match}' (score: {score}, ID: {resolved_id})")
            return resolved_id
            
        logger.warning(f"Could not resolve team ID for '{team_name}' with high confidence. Best match: '{best_match}' (score: {score})")
        return None

    def get_world_cup_fixtures(self, season: int = 2026) -> List[Dict[str, Any]]:
        """Obtains World Cup fixtures with results using fallbacks."""
        if self.source == "api_football" or (self.source == "auto" and self.api_football_key):
            try:
                headers = {"x-apisports-key": self.api_football_key}
                # League 1 is World Cup in API-Football
                data = self._make_request("https://v3.football.api-sports.io/fixtures", headers=headers, params={"league": 1, "season": season})
                fixtures = data.get("response", [])
                if fixtures:
                    return self._normalize_api_football_fixtures(fixtures)
            except Exception as e:
                logger.warning(f"API-Football failed for WC fixtures: {e}. Trying next source.")

        if self.source == "football_data" or self.source == "auto":
            try:
                headers = {"X-Auth-Token": self.football_data_token} if self.football_data_token else {}
                # WC is the competition code for World Cup
                data = self._make_request(f"https://api.football-data.org/v4/competitions/WC/matches", headers=headers, params={"season": season})
                matches = data.get("matches", [])
                if matches:
                    return self._normalize_football_data_fixtures(matches)
            except Exception as e:
                logger.warning(f"Football-Data failed for WC fixtures: {e}. Trying next source.")

        # Priority 3: Open-Football static JSON URL on GitHub
        try:
            logger.info("Trying Open-Football static URL...")
            url = f"https://raw.githubusercontent.com/openfootball/world-cup.json/master/{season}/cup.json"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return self._normalize_open_football_fixtures(data)
        except Exception as e:
            logger.warning(f"Open-Football failed: {e}")

        logger.warning("All live API sources failed to retrieve World Cup fixtures. Returning empty list.")
        return []

    def get_team_last_matches(self, team_name: str, n: int = 10, exclude_competition: str = "World Cup") -> List[Dict[str, Any]]:
        """Gets last N matches outside of World Cup for the given team."""
        if self.source == "api_football" or (self.source == "auto" and self.api_football_key):
            team_id = self.resolve_team_id(team_name, "api_football")
            if team_id:
                try:
                    headers = {"x-apisports-key": self.api_football_key}
                    data = self._make_request("https://v3.football.api-sports.io/fixtures", headers=headers, params={"team": team_id, "last": n * 2})
                    fixtures = data.get("response", [])
                    normalized = self._normalize_api_football_fixtures(fixtures)
                    # Filter out World Cup matches
                    filtered = [m for m in normalized if exclude_competition.lower() not in m.get("competition", "").lower()]
                    return filtered[:n]
                except Exception as e:
                    logger.warning(f"API-Football failed for team {team_name} matches: {e}")

        if self.source == "football_data" or self.source == "auto":
            team_id = self.resolve_team_id(team_name, "football_data")
            if team_id:
                try:
                    headers = {"X-Auth-Token": self.football_data_token} if self.football_data_token else {}
                    data = self._make_request(f"https://api.football-data.org/v4/teams/{team_id}/matches", headers=headers, params={"status": "FINISHED", "limit": n * 2})
                    matches = data.get("matches", [])
                    normalized = self._normalize_football_data_fixtures(matches)
                    filtered = [m for m in normalized if exclude_competition.lower() not in m.get("competition", "").lower()]
                    return filtered[:n]
                except Exception as e:
                    logger.warning(f"Football-Data failed for team {team_name} matches: {e}")

        logger.warning(f"All live API sources failed for team last matches: {team_name}. Returning empty list.")
        return []

    def get_fixture_stats(self, fixture_id: int) -> Dict[str, Any]:
        """Obtains fixture stats from API-Football."""
        if not self.api_football_key:
            return {}
        try:
            headers = {"x-apisports-key": self.api_football_key}
            data = self._make_request(f"https://v3.football.api-sports.io/fixtures/statistics", headers=headers, params={"fixture": fixture_id})
            return data.get("response", [])
        except Exception as e:
            logger.warning(f"Failed to get fixture stats: {e}")
            return {}

    def _normalize_api_football_fixtures(self, response_fixtures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for item in response_fixtures:
            fixture = item.get("fixture", {})
            league = item.get("league", {})
            teams = item.get("teams", {})
            goals = item.get("goals", {})
            
            home_team = teams.get("home", {}).get("name")
            away_team = teams.get("away", {}).get("name")
            home_score = goals.get("home")
            away_score = goals.get("away")
            
            if home_score is None or away_score is None:
                continue # Skip unplayed matches
                
            normalized.append({
                "fixture_id": fixture.get("id"),
                "date": fixture.get("date"),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": int(home_score),
                "away_score": int(away_score),
                "competition": league.get("name"),
                "neutral": fixture.get("neutral", False),
                "source": "api_football"
            })
        return normalized

    def _normalize_football_data_fixtures(self, response_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for match in response_matches:
            home_team = match.get("homeTeam", {}).get("name")
            away_team = match.get("awayTeam", {}).get("name")
            score = match.get("score", {})
            full_time = score.get("fullTime", {})
            home_score = full_time.get("home")
            away_score = full_time.get("away")
            
            if home_score is None or away_score is None:
                continue
                
            normalized.append({
                "fixture_id": match.get("id"),
                "date": match.get("utcDate"),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": int(home_score),
                "away_score": int(away_score),
                "competition": match.get("competition", {}).get("name"),
                "neutral": False, # default value
                "source": "football_data"
            })
        return normalized

    def _normalize_open_football_fixtures(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        normalized = []
        rounds = data.get("rounds", [])
        for rnd in rounds:
            matches = rnd.get("matches", [])
            for match in matches:
                home_team = match.get("team1", {}).get("name") or match.get("team1")
                away_team = match.get("team2", {}).get("name") or match.get("team2")
                score1 = match.get("score1")
                score2 = match.get("score2")
                
                if score1 is None or score2 is None:
                    continue
                    
                normalized.append({
                    "fixture_id": None,
                    "date": match.get("date"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": int(score1),
                    "away_score": int(score2),
                    "competition": "FIFA World Cup",
                    "neutral": True,
                    "source": "open_football"
                })
        return normalized
