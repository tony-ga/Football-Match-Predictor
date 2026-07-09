"""
Centralized team name normalization module.

Provides consistent team name aliasing across all modules:
- feature_builder (ratings lookup)
- espn_client (API queries)
- jsonl_loader (player/team stats filtering)
- market_derivation (market building)
- CLI menu (team selection display and lookup)
"""
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# Path to team mappings file
MAPPINGS_PATH = Path(__file__).parent.parent / "data" / "team_mappings.json"

# Path to ratings file which contains Spanish names as primary keys
RATINGS_PATH = Path(__file__).parent.parent.parent / "data" / "ratings_wc2026.json"

# Additional aliases for dataset consistency
# These map common input names to the names used in derived JSONL files
# Note: Spanish names are preferred as canonical when they exist in ratings_wc2026.json
DATASET_ALIASES = {
    # Input -> Dataset name (player_match_stats.jsonl uses these)
    "USA": "Estados Unidos",
    "US": "Estados Unidos",
    "USMNT": "Estados Unidos",
    "America": "Estados Unidos",
    "EEUU": "Estados Unidos",
    "EE.UU.": "Estados Unidos",
    "United States": "Estados Unidos",
    
    # Belgium variants - map to "Bélgica" which is in ratings_wc2026.json
    "Belgium": "Bélgica",
    "Belgique": "Bélgica",
    "BEL": "Bélgica",
    
    # Other common variants that might appear
    "Brasil": "Brasil",  # Keep Brasil as canonical (same in both)
    "Holland": "Países Bajos",
    "Nederlands": "Países Bajos",
    "Netherlands": "Países Bajos",
    "UK": "Inglaterra",
    "Korea": "Corea del Sur",
    "S. Korea": "Corea del Sur",
    "Republic of Korea": "Corea del Sur",
    "South Korea": "Corea del Sur",
    
    # English to Spanish mappings - prefer Spanish as canonical
    # Note: We do NOT include Spanish->English mappings since ratings uses Spanish
    "France": "Francia",
    "Germany": "Alemania",
    "Spain": "España",
    "England": "Inglaterra",
    "Morocco": "Marruecos",
    "Switzerland": "Suiza",
    "Japan": "Japón",
    "Denmark": "Dinamarca",
    "Turkey": "Turquía",
    "Poland": "Polonia",
    "Ukraine": "Ucrania",
    "Mexico": "México",
    "Canada": "Canadá",
    "Panama": "Panamá",
    "Italy": "Italia",
    "Croatia": "Croacia",
    "Portugal": "Portugal",
    "Uruguay": "Uruguay",
    "Colombia": "Colombia",
    "Chile": "Chile",
    "Ecuador": "Ecuador",
    "Peru": "Perú",
    "Paraguay": "Paraguay",
    "Bolivia": "Bolivia",
    "Venezuela": "Venezuela",
    "Senegal": "Senegal",
    "Nigeria": "Nigeria",
    "Ghana": "Ghana",
    "Cameroon": "Camerún",
    "Egypt": "Egipto",
    "Tunisia": "Túnez",
    "Algeria": "Argelia",
    "Saudi Arabia": "Arabia Saudita",
    "Iran": "Irán",
    "Australia": "Australia",
    "Costa Rica": "Costa Rica",
    "Honduras": "Honduras",
    "New Zealand": "Nueva Zelanda",
    "Serbia": "Serbia",
    "Austria": "Austria",
    "DR Congo": "República Democrática del Congo",
    "Ivory Coast": "Costa de Marfil",
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


def _load_ratings_teams() -> Dict[str, dict]:
    """Load teams from ratings_wc2026.json file."""
    if RATINGS_PATH.exists():
        try:
            with open(RATINGS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('teams', {})
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def get_team_canonical_mapping() -> Dict[str, str]:
    """
    Build a canonical mapping of team names.
    
    Returns a dict where:
    - Keys are display names (Spanish preferred for national teams)
    - Values are canonical names used internally for lookups
    
    This function deduplicates teams that have both Spanish and English names
    in the ratings file, preferring the Spanish name as the display name.
    
    Returns:
        Dict mapping display_name -> canonical_name
    """
    ratings_teams = _load_ratings_teams()
    aliases = _load_aliases()
    
    # Build reverse alias map: canonical -> list of aliases
    canonical_to_aliases: Dict[str, List[str]] = {}
    for alias, canonical in aliases.items():
        if canonical not in canonical_to_aliases:
            canonical_to_aliases[canonical] = []
        canonical_to_aliases[canonical].append(alias)
    
    # Define Spanish preference pairs (spanish_name, english_name)
    spanish_preference = [
        ("Francia", "France"),
        ("Alemania", "Germany"),
        ("España", "Spain"),
        ("Inglaterra", "England"),
        ("Marruecos", "Morocco"),
        ("Suiza", "Switzerland"),
        ("Países Bajos", "Netherlands"),
        ("Holanda", "Netherlands"),
        ("Corea del Sur", "South Korea"),
        ("Estados Unidos", "United States"),
        ("Japón", "Japan"),
        ("Bélgica", "Belgium"),
        ("Dinamarca", "Denmark"),
        ("Turquía", "Turkey"),
        ("Polonia", "Poland"),
        ("Ucrania", "Ukraine"),
        ("México", "Mexico"),
        ("Canadá", "Canada"),
        ("Panamá", "Panama"),
        ("Brasil", "Brazil"),
        ("Argentina", "Argentina"),  # Same in both
        ("Italia", "Italy"),
        ("Croacia", "Croatia"),
        ("Portugal", "Portugal"),
        ("Uruguay", "Uruguay"),
        ("Colombia", "Colombia"),
        ("Chile", "Chile"),
        ("Ecuador", "Ecuador"),
        ("Perú", "Peru"),
        ("Paraguay", "Paraguay"),
        ("Bolivia", "Bolivia"),
        ("Venezuela", "Venezuela"),
        ("Senegal", "Senegal"),
        ("Nigeria", "Nigeria"),
        ("Ghana", "Ghana"),
        ("Camerún", "Cameroon"),
        ("Egipto", "Egypt"),
        ("Túnez", "Tunisia"),
        ("Argelia", "Algeria"),
        ("Arabia Saudita", "Saudi Arabia"),
        ("Irán", "Iran"),
        ("Australia", "Australia"),
        ("Costa Rica", "Costa Rica"),
        ("Honduras", "Honduras"),
        ("Nueva Zelanda", "New Zealand"),
        ("Serbia", "Serbia"),
        ("Austria", "Austria"),
        ("República Democrática del Congo", "DR Congo"),
    ]
    
    # Build set of known canonical names from aliases values
    known_canonicals = set(aliases.values())
    
    # Build the mapping: prefer Spanish names as display, map to canonical
    display_to_canonical: Dict[str, str] = {}
    
    for spanish, english in spanish_preference:
        # Check which variant exists in ratings
        spanish_exists = spanish in ratings_teams
        english_exists = english in ratings_teams
        
        if spanish_exists:
            # Prefer Spanish as display name, canonical is also Spanish
            display_to_canonical[spanish] = spanish
            # Map English alias to Spanish canonical
            if english_exists or english in known_canonicals:
                display_to_canonical[english] = spanish
        elif english_exists:
            # Only English exists, use it as canonical
            display_to_canonical[english] = english
            display_to_canonical[spanish] = english
    
    # Add any remaining teams from ratings that aren't in the preference list
    for team_name in ratings_teams.keys():
        if team_name not in display_to_canonical:
            display_to_canonical[team_name] = team_name
    
    return display_to_canonical


def get_unique_teams_for_menu() -> List[Tuple[str, str]]:
    """
    Get a list of unique teams for menu display.
    
    Returns:
        List of tuples (display_name, canonical_name)
        where display_name is what the user sees (preferably Spanish)
        and canonical_name is what should be used for lookups
    """
    mapping = get_team_canonical_mapping()
    
    # Deduplicate: only keep one entry per canonical name
    canonical_to_display: Dict[str, str] = {}
    
    # Prefer Spanish names as display
    spanish_names = {
        "Francia", "Alemania", "España", "Inglaterra", "Marruecos",
        "Suiza", "Países Bajos", "Corea del Sur", "Estados Unidos",
        "Japón", "Bélgica", "Dinamarca", "Turquía", "Polonia", "Ucrania",
        "México", "Canadá", "Panamá", "Brasil", "Italia", "Croacia",
        "Perú", "Nueva Zelanda", "República Democrática del Congo",
        "Camerún", "Egipto", "Túnez", "Argelia", "Arabia Saudita", "Irán"
    }
    
    for display, canonical in mapping.items():
        if canonical not in canonical_to_display:
            canonical_to_display[canonical] = display
        elif display in spanish_names:
            # Prefer Spanish names
            canonical_to_display[canonical] = display
    
    # Return sorted list
    result = [(display, canonical) for canonical, display in canonical_to_display.items()]
    result.sort(key=lambda x: x[0])  # Sort by display name
    
    return result


def normalize_team_name(team_name: str, context: str = "general") -> str:
    """
    Normalize team name using centralized alias mapping.
    
    This function normalizes team names to their canonical form used in the
    ratings_wc2026.json file (which uses Spanish names as primary keys).
    
    Args:
        team_name: Raw team name from user input, API, or dataset
        context: Context for normalization ('general', 'dataset', 'ratings')
                 - 'general': Standard normalization
                 - 'dataset': For matching against JSONL datasets
                 - 'ratings': For matching against ratings_wc2026.json
    
    Returns:
        Normalized team name for consistent lookups
    
    Examples:
        >>> normalize_team_name("France")
        'Francia'
        >>> normalize_team_name("England")
        'Inglaterra'
        >>> normalize_team_name("Francia")
        'Francia'
    """
    if not team_name:
        return ""
    
    # Strip whitespace
    team_name = team_name.strip()
    
    # Load all aliases
    aliases = _load_aliases()
    
    # Direct match in aliases
    if team_name in aliases:
        return aliases[team_name]
    
    # Case-insensitive match in aliases
    team_name_lower = team_name.lower()
    for alias, canonical in aliases.items():
        if alias.lower() == team_name_lower:
            return canonical
    
    # Check ratings file for direct match (it has Spanish names as primary keys)
    ratings_teams = _load_ratings_teams()
    if team_name in ratings_teams:
        return team_name
    
    # Case-insensitive match in ratings
    for rating_team in ratings_teams.keys():
        if rating_team.lower() == team_name_lower:
            return rating_team
    
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
    ratings_teams = _load_ratings_teams()
    known_teams.update(ratings_teams.keys())
    return team_name in known_teams
