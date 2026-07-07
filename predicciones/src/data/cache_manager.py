"""
ESPN Cache Manager.

Handles caching of ESPN API responses with Windows/Linux-safe filenames.
Uses SHA256 hashing of URLs/params for cache keys.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class EspnCacheManager:
    """
    Manages caching of ESPN API responses.
    
    Features:
    - SHA256-based cache keys (Windows/Linux safe)
    - TTL-based expiration
    - Separate caches for raw responses and derived data
    """
    
    DEFAULT_TTL_HOURS = 24
    
    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "data" / "cache" / "espn"
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Derived data cache
        self.derived_dir = Path(__file__).parent.parent.parent / "data" / "derived"
        self.derived_dir.mkdir(parents=True, exist_ok=True)
    
    def _make_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """
        Create a SHA256-based cache key from endpoint and params.
        
        This ensures filenames are safe on all platforms.
        """
        key_string = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get_raw_response(
        self,
        endpoint: str,
        params: Dict[str, Any],
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached raw API response if not expired.
        
        Args:
            endpoint: API endpoint name
            params: Request parameters
            ttl_hours: Time-to-live in hours
            
        Returns:
            Cached response or None if not found/expired
        """
        cache_key = self._make_cache_key(endpoint, params)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            
            # Check expiration
            cached_at = datetime.fromisoformat(cached_data.get("_cached_at", ""))
            if datetime.utcnow() - cached_at > timedelta(hours=ttl_hours):
                logger.debug(f"Cache expired for {endpoint}")
                return None
            
            return cached_data.get("data")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to read cache file {cache_file}: {e}")
            return None
    
    def set_raw_response(
        self,
        endpoint: str,
        params: Dict[str, Any],
        data: Dict[str, Any],
    ) -> Path:
        """
        Cache a raw API response.
        
        Args:
            endpoint: API endpoint name
            params: Request parameters
            data: Response data to cache
            
        Returns:
            Path to cache file
        """
        cache_key = self._make_cache_key(endpoint, params)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        cached_data = {
            "_cached_at": datetime.utcnow().isoformat(),
            "_endpoint": endpoint,
            "_params": params,
            "data": data,
        }
        
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cached_data, f, indent=2)
        
        logger.debug(f"Cached response to {cache_file}")
        return cache_file
    
    def append_derived_match_stats(
        self,
        match_stats: Dict[str, Any],
        filename: str = "team_match_stats.jsonl",
    ) -> None:
        """
        Append derived match statistics to JSONL file.
        
        Args:
            match_stats: Dict with match-level stats
            filename: Output filename
        """
        output_file = self.derived_dir / filename
        
        # Add timestamp
        match_stats["_created_at"] = datetime.utcnow().isoformat()
        
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(match_stats) + "\n")
    
    def append_derived_player_stats(
        self,
        player_stats: Dict[str, Any],
        filename: str = "player_match_stats.jsonl",
    ) -> None:
        """
        Append derived player statistics to JSONL file.
        
        Args:
            player_stats: Dict with player-level stats
            filename: Output filename
        """
        output_file = self.derived_dir / filename
        
        # Add timestamp
        player_stats["_created_at"] = datetime.utcnow().isoformat()
        
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(player_stats) + "\n")
    
    def load_derived_stats(
        self,
        filename: str = "team_match_stats.jsonl",
        limit: int = None,
    ) -> list:
        """
        Load derived statistics from JSONL file.
        
        Args:
            filename: File to load from
            limit: Maximum number of records to return
            
        Returns:
            List of stat dicts
        """
        input_file = self.derived_dir / filename
        
        if not input_file.exists():
            return []
        
        records = []
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    if limit and len(records) >= limit:
                        break
                    try:
                        records.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        except IOError as e:
            logger.warning(f"Failed to load derived stats: {e}")
        
        return records
    
    def clear_expired(self, ttl_hours: int = DEFAULT_TTL_HOURS) -> int:
        """
        Clear expired cache files.
        
        Args:
            ttl_hours: TTL threshold
            
        Returns:
            Number of files cleared
        """
        cleared = 0
        cutoff = datetime.utcnow() - timedelta(hours=ttl_hours)
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                
                cached_at = datetime.fromisoformat(cached_data.get("_cached_at", ""))
                if cached_at < cutoff:
                    cache_file.unlink()
                    cleared += 1
            except (json.JSONDecodeError, KeyError, ValueError):
                # Delete corrupted files
                cache_file.unlink()
                cleared += 1
        
        logger.info(f"Cleared {cleared} expired cache files")
        return cleared
