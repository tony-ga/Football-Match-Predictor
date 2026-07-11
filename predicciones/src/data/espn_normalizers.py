"""
Team name normalization and matching.

Handles aliases, accents, fuzzy matching for team names.
"""
from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..domain.models import TeamNormalizationResult

logger = logging.getLogger(__name__)


class TeamNormalizer:
    """
    Normalizes and matches team names using aliases and fuzzy matching.
    
    Supports:
    - Direct alias lookup (México -> Mexico)
    - Case-insensitive matching
    - Accent removal
    - Fuzzy matching for close matches
    """
    
    def __init__(self, mappings_path: Optional[str] = None):
        """
        Initialize the normalizer with team mappings.
        
        Args:
            mappings_path: Path to team_mappings.json. If None, uses default location.
        """
        self.aliases: Dict[str, str] = {}
        self._reverse_aliases: Dict[str, List[str]] = {}
        self._canonical_names: set = set()
        
        if mappings_path is None:
            mappings_path = str(Path(__file__).parent / "team_mappings.json")
        
        self._load_mappings(mappings_path)
    
    def _load_mappings(self, path: str) -> None:
        """Load aliases from JSON file."""
        import json
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.aliases = data.get("aliases", {})
                
                # Build reverse index
                for alias, canonical in self.aliases.items():
                    if canonical not in self._reverse_aliases:
                        self._reverse_aliases[canonical] = []
                    self._reverse_aliases[canonical].append(alias)
                    self._canonical_names.add(canonical)
                    
        except Exception as e:
            logger.warning(f"Failed to load team mappings from {path}: {e}")
    
    def normalize(self, team_name: str) -> str:
        """
        Normalize a team name to its canonical form.
        
        Args:
            team_name: Raw team name
            
        Returns:
            Normalized/canonical team name
        """
        if not team_name:
            return ""
        
        # Direct lookup first
        if team_name in self.aliases:
            return self.aliases[team_name]
        
        # Case-insensitive lookup
        team_lower = team_name.lower()
        for alias, canonical in self.aliases.items():
            if alias.lower() == team_lower:
                return canonical
        
        # Remove accents and try again
        team_no_accents = self._remove_accents(team_name)
        if team_no_accents in self.aliases:
            return self.aliases[team_no_accents]
        
        for alias, canonical in self.aliases.items():
            alias_no_accents = self._remove_accents(alias)
            if alias_no_accents.lower() == team_no_accents.lower():
                return canonical
        
        # If no match found, return original (capitalized)
        return team_name
    
    def find_team(
        self,
        query: str,
        available_teams: Optional[List[str]] = None
    ) -> TeamNormalizationResult:
        """
        Find a team by name with fuzzy matching.
        
        Args:
            query: Team name to search for
            available_teams: Optional list of valid team names to restrict results
            
        Returns:
            TeamNormalizationResult with match info
        """
        if not query:
            return TeamNormalizationResult(found=False)
        
        # First try exact normalization
        normalized = self.normalize(query)
        
        # Check if it's in available teams
        if available_teams:
            for team in available_teams:
                if team.lower() == normalized.lower():
                    return TeamNormalizationResult(
                        found=True,
                        normalized_name=team,
                        confidence=1.0
                    )
            
            # Try fuzzy match against available teams
            best_match, confidence = self._fuzzy_match(normalized, available_teams)
            if best_match and confidence > 0.7:
                return TeamNormalizationResult(
                    found=True,
                    normalized_name=best_match,
                    confidence=confidence
                )
            
            return TeamNormalizationResult(
                found=False,
                alternatives=available_teams[:5]  # Return some alternatives
            )
        
        # No restriction, just return normalized
        return TeamNormalizationResult(
            found=True,
            normalized_name=normalized,
            confidence=0.9 if normalized != query else 1.0
        )
    
    def _remove_accents(self, text: str) -> str:
        """Remove accents from text."""
        normalized = unicodedata.normalize('NFKD', text)
        return ''.join(c for c in normalized if not unicodedata.combining(c))
    
    def _fuzzy_match(
        self,
        query: str,
        candidates: List[str]
    ) -> Tuple[Optional[str], float]:
        """
        Perform fuzzy matching against candidates.
        
        Uses simple ratio based on common substring matching.
        Returns (best_match, confidence) or (None, 0.0) if no good match.
        """
        if not candidates:
            return None, 0.0
        
        query_lower = query.lower()
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            cand_lower = candidate.lower()
            
            # Exact match after normalization
            if query_lower == cand_lower:
                return candidate, 1.0
            
            # Check if one contains the other
            if query_lower in cand_lower:
                score = len(query_lower) / len(cand_lower)
                if score > best_score:
                    best_score = score
                    best_match = candidate
            
            if cand_lower in query_lower:
                score = len(cand_lower) / len(query_lower)
                if score > best_score:
                    best_score = score
                    best_match = candidate
            
            # Simple character-based similarity
            common = sum(1 for c in query_lower if c in cand_lower)
            score = common / max(len(query_lower), len(cand_lower))
            if score > best_score:
                best_score = score
                best_match = candidate
        
        return best_match, best_score
    
    def get_all_canonical_names(self) -> List[str]:
        """Return list of all canonical team names."""
        return sorted(self._canonical_names)
    
    def get_aliases_for(self, canonical_name: str) -> List[str]:
        """Return all aliases for a canonical name."""
        return self._reverse_aliases.get(canonical_name, [])
