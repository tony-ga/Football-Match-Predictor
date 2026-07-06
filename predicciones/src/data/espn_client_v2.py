"""
ESPN API client with caching support.

Provides cached access to ESPN endpoints to avoid rate limits and improve performance.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..domain.exceptions import EspnApiError

logger = logging.getLogger(__name__)


class EspnCache:
    """Simple file-based cache for ESPN responses."""
    
    def __init__(self, cache_dir: str, ttl_seconds: int = 300):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_seconds: Time-to-live for cached entries (default 5 minutes)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        # Sanitize key for filename
        safe_key = key.replace("/", "_").replace("?", "_").replace("=", "_")
        return self.cache_dir / f"{safe_key}.json"
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if valid."""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Check TTL
            cached_at = datetime.fromisoformat(data["_cached_at"])
            if datetime.utcnow() - cached_at > timedelta(seconds=self.ttl_seconds):
                # Expired
                cache_path.unlink(missing_ok=True)
                return None
            
            return data["response"]
        except Exception as e:
            logger.warning(f"Cache read error for {key}: {e}")
            return None
    
    def set(self, key: str, response: Dict[str, Any]) -> None:
        """Cache a response."""
        cache_path = self._get_cache_path(key)
        
        try:
            data = {
                "_cached_at": datetime.utcnow().isoformat(),
                "response": response
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Cache write error for {key}: {e}")
    
    def clear(self) -> None:
        """Clear all cached entries."""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")


class EspnClient:
    """
    HTTP client for ESPN API with caching and error handling.
    
    Supports:
    - Scoreboard endpoint for upcoming matches
    - Summary endpoint for match details
    - Teams endpoint for team listings
    - Configurable sport and league
    """
    
    DEFAULT_CONFIG = {
        "sport": "soccer",
        "league": "fifa.world",
        "timeout": 20,
        "max_retries": 3,
        "base_delay": 1.0,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    
    def __init__(
        self,
        sport: Optional[str] = None,
        league: Optional[str] = None,
        timeout: Optional[int] = None,
        cache_enabled: bool = True,
        cache_dir: Optional[str] = None,
        cache_ttl: int = 300,
    ):
        """
        Initialize ESPN client.
        
        Args:
            sport: Sport slug (default: soccer)
            league: League slug (default: fifa.world)
            timeout: Request timeout in seconds
            cache_enabled: Whether to use caching
            cache_dir: Cache directory path
            cache_ttl: Cache TTL in seconds
        """
        self.config = {
            **self.DEFAULT_CONFIG,
            **{k: v for k, v in {
                "sport": sport,
                "league": league,
                "timeout": timeout,
            }.items() if v is not None}
        }
        
        self.base_url = f"https://site.api.espn.com/apis/site/v2/sports/{self.config['sport']}/{self.config['league']}"
        self.timeout = self.config["timeout"]
        self.max_retries = self.config["max_retries"]
        self.base_delay = self.config["base_delay"]
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.config["user_agent"],
            "Accept": "application/json",
        })
        
        # Initialize cache
        if cache_enabled:
            if cache_dir is None:
                cache_dir = str(Path(__file__).parent.parent.parent / "data" / "cache" / "espn")
            self.cache = EspnCache(cache_dir, cache_ttl)
        else:
            self.cache = None
    
    def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retries and caching.
        
        Args:
            url: Request URL
            params: Query parameters
            use_cache: Whether to use cache
            
        Returns:
            JSON response as dict
            
        Raises:
            EspnApiError: If request fails after retries
        """
        # Build cache key
        cache_key = f"{url}?{json.dumps(params, sort_keys=True)}" if params else url
        
        # Check cache first
        if use_cache and self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached
        
        # Make request with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.info(f"ESPN API request (attempt {attempt + 1}/{self.max_retries}): {url}")
                response = self.session.get(url, params=params, timeout=self.timeout)
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", self.base_delay * (2 ** attempt)))
                    logger.warning(f"ESPN rate limited (429). Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                
                # Handle server errors
                if response.status_code >= 500:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"ESPN server error ({response.status_code}). Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                
                # Raise for other errors
                response.raise_for_status()
                
                # Parse JSON
                try:
                    result = response.json()
                except json.JSONDecodeError as e:
                    raise EspnApiError(
                        f"Invalid JSON response from ESPN",
                        status_code=response.status_code,
                        url=url
                    ) from e
                
                # Cache successful response
                if self.cache and use_cache:
                    self.cache.set(cache_key, result)
                
                return result
                
            except requests.Timeout as e:
                last_error = e
                logger.warning(f"Request timeout after {self.timeout}s (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.base_delay * (2 ** attempt))
                    
            except requests.RequestException as e:
                last_error = e
                logger.error(f"Request failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.base_delay * (2 ** attempt))
                else:
                    raise EspnApiError(
                        f"Request failed after {self.max_retries} attempts: {e}",
                        url=url
                    ) from e
        
        # All retries exhausted
        raise EspnApiError(
            f"Request failed after {self.max_retries} attempts",
            url=url
        )
    
    def get_scoreboard(
        self,
        dates: Optional[str] = None,
        limit: int = 200
    ) -> Dict[str, Any]:
        """
        Get scoreboard with upcoming/recent matches.
        
        Args:
            dates: Date range in YYYYMMDD or YYYYMMDD-YYYYMMDD format
            limit: Maximum number of events to return
            
        Returns:
            Scoreboard JSON response
        """
        url = f"{self.base_url}/scoreboard"
        params = {"limit": limit}
        if dates:
            params["dates"] = dates
        
        return self._make_request(url, params)
    
    def get_summary(self, event_id: str) -> Dict[str, Any]:
        """
        Get match summary/details.
        
        Args:
            event_id: ESPN event ID
            
        Returns:
            Summary JSON response
        """
        url = f"{self.base_url}/summary"
        params = {"event": event_id}
        
        return self._make_request(url, params)
    
    def get_teams(self) -> Dict[str, Any]:
        """
        Get list of teams in the league.
        
        Returns:
            Teams JSON response
        """
        url = f"{self.base_url}/teams"
        return self._make_request(url, {})
    
    def get_team_schedule(
        self,
        team_id: str,
        season: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get team schedule.
        
        Args:
            team_id: ESPN team ID
            season: Season year (optional)
            
        Returns:
            Team schedule JSON response
        """
        # Note: This uses the sports.core.api.espn.com endpoint
        base_core_url = "https://sports.core.api.espn.com/apis/site/v2/sports"
        url = f"{base_core_url}/{self.config['sport']}/{self.config['league']}/teams/{team_id}/schedule"
        params = {}
        if season:
            params["season"] = season
        
        return self._make_request(url, params)
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self.cache:
            self.cache.clear()
            logger.info("ESPN cache cleared")
