"""
Centralized team name normalization module.

Provides consistent team name aliasing across all modules:
- feature_builder (ratings lookup)
- espn_client (API queries)
- jsonl_loader (player/team stats filtering)
- market_derivation (market building)
"""
import json
from pathlib import Path
from typing import Dict, Optional

# Path to team mappings file
MAPPINGS_PATH = Path(__file__).parent.parent / "data" / "team_mappings.json"

# Additional aliases for dataset consistency
# These map common input names to the names used in derived JSONL files
DATASET_ALIASES = {
    # Input -> Dataset name (player_match_stats.jsonl uses these)
    "USA": "United States",
    "US": "United States",
    "USMNT": "United States",
    "America": "United States",
    "EEUU": "United States",
    "EE.UU.": "United States",
    
    # Belgium variants - map to "Bélgica" which is in ratings_wc2026.json
    "Belgium": "Bélgica",
    "Belgique": "Bélgica",
    "BEL": "Bélgica",
    
    # Other common variants that might appear
    "Brasil": "Brazil",
    "Holland": "Netherlands",
    "Nederlands": "Netherlands",
    "UK": "England",
    "Korea": "South Korea",
    "S. Korea": "South Korea",
    "Republic of Korea": "South Korea",
    
    # Spanish to English mappings for international compatibility
    "Francia": "France",
    "Alemania": "Germany",
    "España": "Spain",
    "Inglaterra": "England",
    "Marruecos": "Morocco",
    "Suiza": "Switzerland",
    "Países Bajos": "Netherlands",
    "Holanda": "Netherlands",
    "Corea del Sur": "South Korea",
    "Estados Unidos": "United States",
}


def _load_aliases() -> Dict[str, str]:
    """Load aliases from team_mappings.json and merge with dataset aliases."""
    aliases = {}
    
    # Load from file if exists
    if MAPPINGS_PATH.exists():
        try:
            with open(MAPPINGS_PATH, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
                aliases = mappings.get("aliases", {})
        except (json.JSONDecodeError, IOError) as e:
            pass  # Use empty aliases if file can't be loaded
    
    # Merge with dataset-specific aliases (dataset aliases take precedence)
    aliases.update(DATASET_ALIASES)
    
    return aliases


def normalize_team_name(team_name: str, context: str = "general") -> str:
    """
    Normalize team name using centralized alias mapping.
    
    Args:
        team_name: Raw team name from user input, API, or dataset
        context: Context for normalization ('general', 'dataset', 'ratings')
                 - 'general': Standard normalization
                 - 'dataset': For matching against JSONL datasets
                 - 'ratings': For matching against ratings_wc2026.json
    
    Returns:
        Normalized team name for consistent lookups
    
    Examples:
        >>> normalize_team_name("USA")
        'United States'
        >>> normalize_team_name("Belgique")
        'Belgium'
        >>> normalize_team_name("United States")
        'United States'
    """
    if not team_name:
        return ""
    
    # Strip whitespace
    team_name = team_name.strip()
    
    # Load all aliases
    aliases = _load_aliases()
    
    # Direct match
    if team_name in aliases:
        return aliases[team_name]
    
    # Case-insensitive match
    team_name_lower = team_name.lower()
    for alias, canonical in aliases.items():
        if alias.lower() == team_name_lower:
            return canonical
    
    # No alias found, return original
    return team_name


def get_canonical_team_name(team_name: str) -> str:
    """
    Get the canonical team name used across all datasets.
    
    This is an alias for normalize_team_name with default context.
    
    Args:
        team_name: Raw team name
    
    Returns:
        Canonical team name
    """
    return normalize_team_name(team_name, context="general")


def reverse_alias(canonical_name: str) -> str:
    """
    Given a canonical name, return a common short form if available.
    
    Useful for display purposes where shorter names are preferred.
    
    Args:
        canonical_name: Full canonical team name
    
    Returns:
        Short form or original name if no short form exists
    """
    # Common short forms
    short_forms = {
        "United States": "USA",
        "Brazil": "BRA",
        "Argentina": "ARG",
        "Germany": "GER",
        "France": "FRA",
        "Spain": "ESP",
        "England": "ENG",
        "Italy": "ITA",
        "Netherlands": "NED",
        "Belgium": "BEL",
        "Portugal": "POR",
        "Mexico": "MEX",
        "Japan": "JPN",
        "South Korea": "KOR",
    }
    
    return short_forms.get(canonical_name, canonical_name)


def is_valid_team(team_name: str) -> bool:
    """
    Check if a team name is recognized (either as alias or canonical).
    
    Args:
        team_name: Team name to validate
    
    Returns:
        True if team is recognized, False otherwise
    """
    if not team_name:
        return False
    
    aliases = _load_aliases()
    normalized = normalize_team_name(team_name)
    
    # Check if it normalizes to something different (meaning it's an alias)
    if normalized != team_name:
        return True
    
    # Check if it's a known canonical name (appears as a value in aliases)
    if team_name in aliases.values():
        return True
    
    # Check direct membership in known teams
    known_teams = set(aliases.values())
    return team_name in known_teams
