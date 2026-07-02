"""
Dataset builder: converts historical match records into the same
feature schema used in production prediction.
This is critical for ensuring training and inference use identical features.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_training_dataset(
    df: pd.DataFrame,
    lookback_games: int = 20,
    min_games_required: int = 5,
    target_markets: bool = True,
) -> pd.DataFrame:
    """
    Convert a raw historical match DataFrame into a training dataset
    where each row has the same features used in production.

    For each match, computes rolling pre-match features for both teams
    (using only data BEFORE the match date to avoid leakage).

    Args:
        df: Historical matches DataFrame (from csv_loader).
        lookback_games: Rolling window for team stats.
        min_games_required: Skip matches where a team has fewer than this many games.
        target_markets: If True, add target columns for all markets.

    Returns:
        DataFrame with features and targets for model training.
    """
    df = df.sort_values('date').reset_index(drop=True)
    records = []

    for idx, row in df.iterrows():
        match_date = row['date']
        home_team = row['home_team']
        away_team = row['away_team']

        # Historical data BEFORE this match (strict)
        prior = df[(df['date'] < match_date)]

        home_stats = _rolling_stats(prior, home_team, lookback_games)
        away_stats = _rolling_stats(prior, away_team, lookback_games)

        if (home_stats['n_games'] < min_games_required or
                away_stats['n_games'] < min_games_required):
            continue

        h2h_stats = _h2h_stats(prior, home_team, away_team)

        record = {
            'match_id': f"{match_date.strftime('%Y%m%d')}_{home_team[:3]}_{away_team[:3]}",
            'date': match_date,
            'home_team': home_team,
            'away_team': away_team,
            'tournament': row.get('tournament', 'unknown'),
            'neutral': row.get('neutral', False),

            # Home team features
            'home_goals_scored_avg': home_stats['goals_scored_avg'],
            'home_goals_conceded_avg': home_stats['goals_conceded_avg'],
            'home_form_ppg': home_stats['form_ppg'],
            'home_win_rate': home_stats['win_rate'],
            'home_n_games': home_stats['n_games'],

            # Away team features
            'away_goals_scored_avg': away_stats['goals_scored_avg'],
            'away_goals_conceded_avg': away_stats['goals_conceded_avg'],
            'away_form_ppg': away_stats['form_ppg'],
            'away_win_rate': away_stats['win_rate'],
            'away_n_games': away_stats['n_games'],

            # H2H
            'h2h_home_wins': h2h_stats['home_wins'],
            'h2h_draws': h2h_stats['draws'],
            'h2h_away_wins': h2h_stats['away_wins'],
            'h2h_n_games': h2h_stats['n_games'],

            # Derived features
            'attack_ratio': _safe_ratio(
                home_stats['goals_scored_avg'],
                away_stats['goals_scored_avg']
            ),
            'defense_ratio': _safe_ratio(
                away_stats['goals_conceded_avg'],
                home_stats['goals_conceded_avg']
            ),
            'form_ratio': _safe_ratio(
                home_stats['form_ppg'],
                away_stats['form_ppg']
            ),
        }

        if target_markets:
            home_score = row['home_score']
            away_score = row['away_score']
            total = home_score + away_score

            record.update({
                'home_score': home_score,
                'away_score': away_score,
                'total_goals': total,

                # 1X2: 0=home, 1=draw, 2=away
                'target_1x2': (
                    0 if home_score > away_score else
                    1 if home_score == away_score else 2
                ),
                'target_home_win': int(home_score > away_score),
                'target_draw': int(home_score == away_score),
                'target_away_win': int(home_score < away_score),

                # Binary markets
                'target_btts': int(home_score > 0 and away_score > 0),
                'target_over_15': int(total > 1.5),
                'target_over_25': int(total > 2.5),
                'target_over_35': int(total > 3.5),
                'target_cs_home': int(away_score == 0),
                'target_cs_away': int(home_score == 0),
            })

        records.append(record)

    result = pd.DataFrame(records)
    logger.info(f"Built training dataset: {len(result)} rows from {len(df)} matches")
    return result


def _rolling_stats(df: pd.DataFrame, team: str, n: int) -> dict:
    """Compute rolling stats for a team from historical data."""
    home = df[df['home_team'] == team][['date', 'home_score', 'away_score', 'result']].copy()
    away = df[df['away_team'] == team][['date', 'home_score', 'away_score', 'result']].copy()

    home['scored'] = home['home_score']
    home['conceded'] = home['away_score']
    home['points'] = home['result'].map({'H': 3, 'D': 1, 'A': 0})

    away['scored'] = away['away_score']
    away['conceded'] = away['home_score']
    away['points'] = away['result'].map({'A': 3, 'D': 1, 'H': 0})

    games = pd.concat(
        [home[['date', 'scored', 'conceded', 'points']],
         away[['date', 'scored', 'conceded', 'points']]]
    ).sort_values('date').tail(n)

    if len(games) == 0:
        return {'goals_scored_avg': 1.2, 'goals_conceded_avg': 1.0,
                'win_rate': 0.4, 'draw_rate': 0.25, 'form_ppg': 1.5, 'n_games': 0}

    ng = len(games)
    wins = (games['points'] == 3).sum()
    draws = (games['points'] == 1).sum()

    return {
        'goals_scored_avg': float(games['scored'].mean()),
        'goals_conceded_avg': float(games['conceded'].mean()),
        'win_rate': float(wins / ng),
        'draw_rate': float(draws / ng),
        'form_ppg': float(games['points'].mean()),
        'n_games': ng,
    }


def _h2h_stats(df: pd.DataFrame, home: str, away: str) -> dict:
    """Head-to-head statistics between two teams."""
    h2h = df[
        ((df['home_team'] == home) & (df['away_team'] == away)) |
        ((df['home_team'] == away) & (df['away_team'] == home))
    ]

    if len(h2h) == 0:
        return {'home_wins': 0, 'draws': 0, 'away_wins': 0, 'n_games': 0}

    home_wins = len(h2h[
        ((h2h['home_team'] == home) & (h2h['result'] == 'H')) |
        ((h2h['home_team'] == away) & (h2h['result'] == 'A'))
    ])
    draws = (h2h['result'] == 'D').sum()
    away_wins = len(h2h) - home_wins - draws

    return {
        'home_wins': int(home_wins),
        'draws': int(draws),
        'away_wins': int(away_wins),
        'n_games': len(h2h),
    }


def _safe_ratio(a: float, b: float, eps: float = 0.1) -> float:
    """Safe ratio with epsilon to avoid division by zero."""
    return a / (b + eps)


def split_train_val(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    temporal: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/validation sets.
    If temporal=True, uses time-based split (no shuffling) to prevent leakage.
    """
    if temporal:
        df = df.sort_values('date')
        split_idx = int(len(df) * (1 - val_fraction))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    else:
        from sklearn.model_selection import train_test_split
        return train_test_split(df, test_size=val_fraction, random_state=42)
