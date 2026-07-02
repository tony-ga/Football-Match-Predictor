"""
CSV loader for historical football match data.
Supports public FIFA/international results datasets.
Expected columns: date, home_team, away_team, home_score, away_score,
                  tournament, neutral (optional).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard column names after loading
STANDARD_COLS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
]

# Column name aliases for common public datasets
COLUMN_ALIASES: dict[str, str] = {
    "home": "home_team",
    "away": "away_team",
    "home_goals": "home_score",
    "away_goals": "away_score",
    "hg": "home_score",
    "ag": "away_score",
    "fthg": "home_score",
    "ftag": "away_score",
    "comp": "tournament",
    "competition": "tournament",
}


def load_historical_csv(
    filepath: str | Path,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    tournaments: Optional[List[str]] = None,
    min_year: int = 2000,
) -> pd.DataFrame:
    """
    Load and clean a historical match results CSV.

    Args:
        filepath: Path to CSV file.
        min_date: Filter matches after this date (YYYY-MM-DD).
        max_date: Filter matches before this date (YYYY-MM-DD).
        tournaments: If provided, filter to these tournament names.
        min_year: Minimum year to include (default 2000).

    Returns:
        Cleaned DataFrame with standardized columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, parse_dates=["date"], dayfirst=False)
    logger.info("Loaded %d rows from %s", len(df), path)

    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns=COLUMN_ALIASES)

    # Ensure required columns exist
    required = ["date", "home_team", "away_team", "home_score", "away_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Add optional columns with defaults
    if "tournament" not in df.columns:
        df["tournament"] = "unknown"
    if "neutral" not in df.columns:
        df["neutral"] = False

    # Type coercion
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = df["neutral"].fillna(False).astype(bool)

    # Drop rows with missing scores or dates
    n_before = len(df)
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    logger.info("Dropped %d rows with missing values", n_before - len(df))

    # Year filter
    df = df[df["date"].dt.year >= min_year]

    # Date filters
    if min_date:
        df = df[df["date"] >= pd.to_datetime(min_date)]
    if max_date:
        df = df[df["date"] <= pd.to_datetime(max_date)]

    # Tournament filter
    if tournaments:
        pattern = "|".join(tournaments)
        df = df[df["tournament"].str.contains(pattern, case=False, na=False)]

    # Add derived columns
    df["result"] = df.apply(_get_result, axis=1)       # H, D, A
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["goal_diff"] = df["home_score"] - df["away_score"]

    df = df.sort_values("date").reset_index(drop=True)
    logger.info(
        "Final dataset: %d matches from %s to %s",
        len(df),
        df["date"].min(),
        df["date"].max(),
    )
    return df


def _get_result(row: pd.Series) -> str:
    """Return 'H' (home win), 'D' (draw), or 'A' (away win)."""
    if row["home_score"] > row["away_score"]:
        return "H"
    elif row["home_score"] < row["away_score"]:
        return "A"
    else:
        return "D"


def get_team_stats(
    df: pd.DataFrame,
    team: str,
    n_recent: int = 20,
    reference_date: Optional[str] = None,
) -> dict:
    """
    Compute rolling team stats from historical data.

    Args:
        df: Historical matches DataFrame.
        team: Team name to compute stats for.
        n_recent: Number of most recent games to consider.
        reference_date: Only consider matches up to this date (YYYY-MM-DD).

    Returns:
        Dict with keys: goals_scored_avg, goals_conceded_avg,
                        win_rate, draw_rate, loss_rate, form_ppg, n_games.
    """
    if reference_date:
        df = df[df["date"] <= pd.to_datetime(reference_date)]

    home_games = df[df["home_team"] == team].copy()
    away_games = df[df["away_team"] == team].copy()

    home_games["scored"] = home_games["home_score"]
    home_games["conceded"] = home_games["away_score"]
    home_games["points"] = home_games["result"].map({"H": 3, "D": 1, "A": 0})

    away_games["scored"] = away_games["away_score"]
    away_games["conceded"] = away_games["home_score"]
    away_games["points"] = away_games["result"].map({"A": 3, "D": 1, "H": 0})

    all_games = pd.concat(
        [
            home_games[["date", "scored", "conceded", "points"]],
            away_games[["date", "scored", "conceded", "points"]],
        ]
    ).sort_values("date").tail(n_recent)

    if len(all_games) == 0:
        return {
            "goals_scored_avg": 1.2,
            "goals_conceded_avg": 1.0,
            "win_rate": 0.40,
            "draw_rate": 0.25,
            "loss_rate": 0.35,
            "form_ppg": 1.5,
            "n_games": 0,
        }

    n = len(all_games)
    wins = int((all_games["points"] == 3).sum())
    draws = int((all_games["points"] == 1).sum())
    losses = n - wins - draws

    return {
        "goals_scored_avg": float(all_games["scored"].mean()),
        "goals_conceded_avg": float(all_games["conceded"].mean()),
        "win_rate": wins / n,
        "draw_rate": draws / n,
        "loss_rate": losses / n,
        "form_ppg": float(all_games["points"].mean()),
        "n_games": n,
    }


def generate_synthetic_dataset(n_matches: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic historical dataset for testing when no real data is available.
    Simulates international football matches with realistic distributions
    using a Dixon-Coles-inspired Poisson model.

    Args:
        n_matches: Number of synthetic matches to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with the same schema as load_historical_csv output.
    """
    rng = np.random.default_rng(seed)

    teams = [
        "Argentina", "Brazil", "France", "Germany", "England", "Spain",
        "Portugal", "Netherlands", "Belgium", "Italy", "Croatia", "Uruguay",
        "Mexico", "Colombia", "Chile", "Ecuador", "Peru", "Paraguay",
        "USA", "Canada", "Senegal", "Morocco", "Nigeria", "Ghana",
        "Japan", "South Korea", "Australia", "Iran", "Saudi Arabia",
        "Qatar", "Costa Rica", "Panama", "Honduras", "Bolivia",
    ]

    # Team strength priors (higher = stronger)
    strength: dict[str, float] = {
        "Argentina": 8.5, "Brazil": 8.5, "France": 8.3, "Germany": 8.0,
        "England": 7.8, "Spain": 7.8, "Portugal": 7.8, "Netherlands": 7.5,
        "Belgium": 7.3, "Italy": 7.5, "Croatia": 7.0, "Uruguay": 7.0,
        "Mexico": 6.5, "Colombia": 6.5, "Chile": 6.3, "Ecuador": 6.0,
        "Peru": 5.8, "Paraguay": 5.5, "USA": 6.2, "Canada": 5.8,
        "Senegal": 6.5, "Morocco": 6.3, "Nigeria": 6.2, "Ghana": 5.8,
        "Japan": 6.5, "South Korea": 6.3, "Australia": 5.8, "Iran": 5.5,
        "Saudi Arabia": 5.5, "Qatar": 5.0, "Costa Rica": 5.5,
        "Panama": 5.0, "Honduras": 4.8, "Bolivia": 4.5,
    }

    dates = pd.date_range("2010-01-01", "2024-01-01", periods=n_matches)
    records = []

    for i in range(n_matches):
        home_team, away_team = rng.choice(teams, size=2, replace=False)
        s_h = strength.get(home_team, 5.0)
        s_a = strength.get(away_team, 5.0)

        # Dixon-Coles-like Poisson lambdas
        lambda_h = float(np.exp(0.3 + 0.15 * s_h - 0.10 * s_a + 0.2))  # home adv
        lambda_a = float(np.exp(0.3 + 0.15 * s_a - 0.10 * s_h))
        lambda_h = float(np.clip(lambda_h, 0.3, 4.5))
        lambda_a = float(np.clip(lambda_a, 0.3, 4.5))

        home_score = int(rng.poisson(lambda_h))
        away_score = int(rng.poisson(lambda_a))

        records.append(
            {
                "date": dates[i],
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "tournament": rng.choice(
                    [
                        "FIFA World Cup",
                        "Copa America",
                        "UEFA Nations League",
                        "Friendly",
                        "FIFA World Cup qualifier",
                    ]
                ),
                "neutral": bool(rng.random() > 0.7),
            }
        )

    df = pd.DataFrame(records)
    df["result"] = df.apply(_get_result, axis=1)
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["goal_diff"] = df["home_score"] - df["away_score"]
    return df
