"""
Team Ratings Loader for Daily Predictions Pipeline.

Provides centralized loading of team ratings from ratings_wc2026.json
and builds feature dicts compatible with Dixon-Coles model.

This module bridges the gap between static ratings files and the
feature expectations of the prediction model.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Default project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RATINGS_PATH = PROJECT_ROOT / "data" / "ratings_wc2026.json"

# League-average goals per game (consistent with src/features/ratings.py)
LEAGUE_AVG_GOALS = 1.35


class TeamRatingsLoader:
    """
    Loads team ratings from JSON file and provides lookup functionality.
    
    Supports:
    - Loading from ratings_wc2026.json
    - Team name normalization via team_normalization module
    - Fallback to default ratings for unknown teams
    - Detailed logging for debugging
    """
    
    def __init__(self, ratings_path: Optional[str] = None):
        """
        Initialize the ratings loader.
        
        Args:
            ratings_path: Optional custom path to ratings JSON file.
                          If None, uses default RATINGS_PATH.
        """
        self.ratings_path = Path(ratings_path) if ratings_path else RATINGS_PATH
        self.ratings_data = self._load_ratings()
        self.teams = self.ratings_data.get("teams", {})
        self.default = self.ratings_data.get(
            "default", 
            {"attack": 1.10, "defense": 1.00, "fifa_rank": 100}
        )
        
        # Compute normalization factors to center ratings around 1.0
        # This ensures lambda calculations are properly scaled for Poisson/DC model
        self._attack_mean, self._defense_mean = self._compute_rating_means()
        logger.info(
            f"Rating normalization factors: attack_mean={self._attack_mean:.4f}, "
            f"defense_mean={self._defense_mean:.4f}"
        )
        
        # Try to import team normalization for alias resolution
        self._normalizer = None
        try:
            from src.utils.team_normalization import normalize_team_name
            self._normalizer = normalize_team_name
            logger.debug("Team normalization module loaded successfully")
        except ImportError:
            logger.warning(
                "Team normalization module not available. "
                "Team name matching will be exact only."
            )
    
    def _load_ratings(self) -> Dict[str, Any]:
        """Load ratings from JSON file."""
        if not self.ratings_path.exists():
            logger.error(f"Ratings file not found at {self.ratings_path}")
            return {"teams": {}, "default": {"attack": 1.10, "defense": 1.00, "fifa_rank": 100}}
        
        try:
            with open(self.ratings_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loaded ratings from {self.ratings_path}: {len(data.get('teams', {}))} teams")
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load ratings: {e}")
            return {"teams": {}, "default": {"attack": 1.10, "defense": 1.00, "fifa_rank": 100}}
    
    def _compute_rating_means(self) -> Tuple[float, float]:
        """
        Compute mean attack and defense ratings across all teams.
        
        Returns:
            Tuple of (mean_attack, mean_defense)
        """
        if not self.teams:
            return 1.0, 1.0
        
        attacks = [r.get('attack', 1.0) for r in self.teams.values()]
        defenses = [r.get('defense', 1.0) for r in self.teams.values()]
        
        mean_attack = sum(attacks) / len(attacks) if attacks else 1.0
        mean_defense = sum(defenses) / len(defenses) if defenses else 1.0
        
        return mean_attack, mean_defense
    
    def normalize_team_name(self, team_name: str) -> str:
        """
        Normalize team name using alias mapping if available.
        
        Args:
            team_name: Raw team name
            
        Returns:
            Normalized team name for ratings lookup
        """
        if not self._normalizer:
            # Fallback: simple whitespace stripping
            return team_name.strip() if team_name else ""
        
        return self._normalizer(team_name, context="ratings")
    
    def get_team_rating(
        self, 
        team_name: str,
        verbose: bool = False
    ) -> Tuple[Dict[str, Any], str, bool]:
        """
        Get rating dict for a team.
        
        Args:
            team_name: Team name to look up
            verbose: Enable detailed logging
            
        Returns:
            Tuple of (rating_dict, matched_name, used_fallback)
            - rating_dict: Dict with 'attack', 'defense', 'fifa_rank'
            - matched_name: The name key used in ratings (may differ from input)
            - used_fallback: True if default rating was used
        """
        if not team_name:
            if verbose:
                logger.warning("Empty team name provided, using default rating")
            return self.default.copy(), "", True
        
        # Normalize team name
        normalized = self.normalize_team_name(team_name)
        
        # Direct lookup
        if normalized in self.teams:
            if verbose:
                logger.info(f"Found team '{team_name}' -> '{normalized}' in ratings")
            return self.teams[normalized].copy(), normalized, False
        
        # Try case-insensitive match as fallback
        normalized_lower = normalized.lower()
        for key, rating in self.teams.items():
            if key.lower() == normalized_lower:
                if verbose:
                    logger.info(f"Found team '{team_name}' via case-insensitive match to '{key}'")
                return rating.copy(), key, False
        
        # No match found - use default
        if verbose:
            logger.warning(
                f"Team '{team_name}' (normalized: '{normalized}') not found in ratings. "
                f"Using default rating: attack={self.default['attack']}, defense={self.default['defense']}"
            )
        return self.default.copy(), normalized, True
    
    def build_team_features(
        self,
        team_name: str,
        is_home: bool,
        home_advantage_log: float = 0.25,
        verbose: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Build complete feature dict for a team suitable for Dixon-Coles model.
        
        Args:
            team_name: Team name
            is_home: Whether team is playing at home
            home_advantage_log: Log-scale home advantage (default 0.25)
            verbose: Enable detailed logging
            
        Returns:
            Tuple of (features_dict, rating_info_dict)
            - features_dict: Complete feature dict for model
            - rating_info_dict: Diagnostic info about ratings used
        """
        # Get team rating
        rating, matched_name, used_fallback = self.get_team_rating(team_name, verbose)
        
        # Extract rating components
        attack_rating_raw = rating.get("attack", self.default["attack"])
        defense_rating_raw = rating.get("defense", self.default["defense"])
        fifa_rank = rating.get("fifa_rank", self.default.get("fifa_rank", 100))
        
        # Normalize ratings to be centered around 1.0
        # This is critical for proper Poisson/Dixon-Coles lambda scaling
        # Raw ratings may have means != 1.0 (e.g., avg_attack=1.28, avg_defense=1.17)
        # Normalized rating = raw_rating / mean_rating
        # This ensures that average teams produce lambdas near LEAGUE_AVG_GOALS
        if not used_fallback:
            attack_rating = attack_rating_raw / self._attack_mean
            defense_rating = defense_rating_raw / self._defense_mean
        else:
            # For fallback teams, also normalize the default rating
            attack_rating = self.default["attack"] / self._attack_mean
            defense_rating = self.default["defense"] / self._defense_mean
        
        # IMPORTANT: When using explicit attack/defense ratings from a file,
        # we set ranking_factor, form_factor, h2h_factor, squad_multiplier to 1.0
        # to avoid double-counting team strength. The ratings already encode
        # the team's quality based on FIFA rank and performance.
        # These factors are only meaningful when computing features from scratch
        # without pre-computed ratings.
        ranking_factor = 1.0
        
        if verbose:
            logger.debug(
                f"Rating normalization for {team_name}: "
                f"attack {attack_rating_raw:.3f} -> {attack_rating:.3f}, "
                f"defense {defense_rating_raw:.3f} -> {defense_rating:.3f}, "
                f"ranking_factor set to 1.0 (avoid double-counting)"
            )
        
        # IMPORTANT: When using explicit attack/defense ratings from a file,
        # we set ranking_factor to 1.0 to avoid double-counting team strength.
        # The ratings already encode the team's quality based on FIFA rank and performance.
        # The old ranking_factor calculation is kept below for reference but overridden.
        # OLD CODE (for reference only - DO NOT USE with pre-computed ratings):
        #   reference_rank = 50.0
        #   normalized = max(0, (reference_rank - fifa_rank) / reference_rank)
        #   ranking_factor = 0.65 + 0.70 * normalized
        #   ranking_factor = float(max(0.65, min(1.35, ranking_factor)))
        ranking_factor = 1.0  # Overridden to prevent double-counting
        
        # Build feature dict (compatible with DixonColesModel.predict_lambdas)
        features = {
            'nombre': matched_name or team_name,
            'attack_rating': float(attack_rating),
            'defense_rating': float(defense_rating),
            'form_factor': 1.0,  # Can be extended with form data
            'ranking_factor': float(ranking_factor),  # Always 1.0 when using pre-computed ratings
            'h2h_factor': 1.0,   # Can be extended with H2H data
            'squad_multiplier': 1.0,  # Can be extended with squad data
            'home_advantage_log': home_advantage_log if is_home else 0.0,
            'context_modifier': 0.0,  # Can be extended with context data
        }
        
        # Rating info for diagnostics (include both raw and normalized)
        rating_info = {
            'input_team_name': team_name,
            'matched_team_name': matched_name,
            'home_attack_rating': float(attack_rating) if is_home else None,
            'home_defense_rating': float(defense_rating) if is_home else None,
            'away_attack_rating': float(attack_rating) if not is_home else None,
            'away_defense_rating': float(defense_rating) if not is_home else None,
            'home_attack_rating_raw': float(attack_rating_raw) if is_home else None,
            'home_defense_rating_raw': float(defense_rating_raw) if is_home else None,
            'away_attack_rating_raw': float(attack_rating_raw) if not is_home else None,
            'away_defense_rating_raw': float(defense_rating_raw) if not is_home else None,
            'fifa_rank': fifa_rank,
            'ranking_factor': ranking_factor,
            'ratings_source': 'ratings_wc2026.json',
            'used_default_fallback': used_fallback,
            'normalization_applied': True,
            'attack_mean': self._attack_mean,
            'defense_mean': self._defense_mean,
        }
        
        if verbose:
            logger.info(
                f"Built features for {team_name}: "
                f"attack={attack_rating:.3f}, defense={defense_rating:.3f}, "
                f"rank={fifa_rank}, ranking_factor={ranking_factor:.3f}, "
                f"fallback={used_fallback}"
            )
        
        return features, rating_info
    
    def get_ratings_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about loaded ratings.
        
        Returns:
            Dict with summary info
        """
        if not self.teams:
            return {
                'total_teams': 0,
                'avg_attack': self.default['attack'],
                'avg_defense': self.default['defense'],
                'min_rank': self.default.get('fifa_rank', 100),
                'max_rank': self.default.get('fifa_rank', 100),
            }
        
        attacks = [r.get('attack', 1.0) for r in self.teams.values()]
        defenses = [r.get('defense', 1.0) for r in self.teams.values()]
        ranks = [r.get('fifa_rank', 100) for r in self.teams.values()]
        
        return {
            'total_teams': len(self.teams),
            'avg_attack': sum(attacks) / len(attacks),
            'avg_defense': sum(defenses) / len(defenses),
            'min_rank': min(ranks),
            'max_rank': max(ranks),
            'ratings_file': str(self.ratings_path),
        }


# Convenience function for direct use
def load_ratings_and_build_features(
    home_team: str,
    away_team: str,
    home_advantage: float = 0.25,
    verbose: bool = False,
    ratings_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Convenience function to load ratings and build features for both teams.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        home_advantage: Home advantage log value
        verbose: Enable detailed logging
        ratings_path: Optional custom ratings file path
        
    Returns:
        Tuple of (home_features, away_features, diagnostic_info)
    """
    loader = TeamRatingsLoader(ratings_path)
    
    home_features, home_info = loader.build_team_features(
        home_team, is_home=True, 
        home_advantage_log=home_advantage,
        verbose=verbose
    )
    
    away_features, away_info = loader.build_team_features(
        away_team, is_home=False,
        home_advantage_log=0.0,
        verbose=verbose
    )
    
    diagnostic_info = {
        'home': home_info,
        'away': away_info,
        'ratings_summary': loader.get_ratings_summary(),
    }
    
    return home_features, away_features, diagnostic_info
