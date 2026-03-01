"""
build_nhl_goalie_ratings.py
============================
Computes per-season goalie quality z-scores from NHL API stats.
Analogous to build_qb_ratings.py for NFL.

Formula:
    goalie_score = sv_pct_z * 0.50 + (-gaa_z) * 0.30 + gsaa_z * 0.20
    (only SV% + GAA for pre-2010 seasons where GSAA is unavailable)

Saves:
  - nhl_goalie_ratings.csv          (player_id, season, goalie_score, sv_pct, gaa, name)
  - nhl_goalie_team_ratings.csv     (team, goalie_score, starter_name, sv_pct, gaa)
                                     current season only, one row per team

Usage:
    python build_nhl_goalie_ratings.py
"""

import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apis.nhl import NHLClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIN_GP = 15          # Minimum games played to qualify for rating
SEASONS = list(range(2000, 2026))  # 2000-01 through 2025-26
CURRENT_SEASON = "20242025"
GSAA_AVAILABLE_FROM = 2010  # Before this, only SV% + GAA used


def compute_goalie_score(df_season: pd.DataFrame, season_year: int) -> pd.DataFrame:
    """
    Compute z-score goalie quality rating for one season.
    df_season must have: player_id, name, team, gp, sv_pct, gaa, shots_against
    """
    df = df_season[df_season['gp'] >= MIN_GP].copy()
    if len(df) < 5:
        return pd.DataFrame()

    # Z-score each metric (within this season's qualifying goalies)
    def zscore(series):
        mu = series.mean()
        sigma = series.std()
        if sigma == 0:
            return pd.Series(0.0, index=series.index)
        return (series - mu) / sigma

    df['sv_pct_z'] = zscore(df['sv_pct'].fillna(df['sv_pct'].mean()))
    df['gaa_z']    = zscore(df['gaa'].fillna(df['gaa'].mean()))

    # GSAA (Goals Saved Above Average) — approximate as:
    # gsaa = (league_avg_sv_pct - sv_pct) * shots_against * (-1)
    # Positive = saved more goals than average
    if season_year >= GSAA_AVAILABLE_FROM and 'shots_against' in df.columns:
        lg_sv_pct = df['sv_pct'].mean()
        df['gsaa'] = (df['sv_pct'] - lg_sv_pct) * df['shots_against'].fillna(0)
        df['gsaa_z'] = zscore(df['gsaa'])
        df['goalie_score'] = df['sv_pct_z'] * 0.50 + df['gaa_z'] * (-1.0) * 0.30 + df['gsaa_z'] * 0.20
    else:
        df['gsaa_z'] = 0.0
        df['goalie_score'] = df['sv_pct_z'] * 0.65 + df['gaa_z'] * (-1.0) * 0.35

    # Re-z-score final goalie_score (so scale is consistent across seasons)
    df['goalie_score'] = zscore(df['goalie_score'])

    df['season'] = season_year
    return df[['player_id', 'name', 'team', 'season', 'gp', 'sv_pct', 'gaa',
               'sv_pct_z', 'gaa_z', 'gsaa_z', 'goalie_score']].copy()


def main():
    logger.info("=== NHL Goalie Ratings Build ===")
    client = NHLClient(sleep_between=0.2)

    all_seasons = []
    current_ratings = []

    for season_year in SEASONS:
        season_id = f"{season_year}{season_year + 1}"
        logger.info(f"Fetching {season_year}-{season_year+1} goalie stats...")

        goalies = client.get_goalie_season_stats(season_id)
        if not goalies:
            logger.warning(f"  No data for {season_year}")
            continue

        df = pd.DataFrame(goalies)
        required_cols = ['player_id', 'name', 'team', 'gp', 'sv_pct', 'gaa']
        if not all(c in df.columns for c in required_cols):
            logger.warning(f"  Missing columns for {season_year}: {df.columns.tolist()}")
            continue

        df['sv_pct'] = pd.to_numeric(df['sv_pct'], errors='coerce')
        df['gaa'] = pd.to_numeric(df['gaa'], errors='coerce')
        df['shots_against'] = pd.to_numeric(df.get('shots_against', 0), errors='coerce')
        df['gp'] = pd.to_numeric(df['gp'], errors='coerce').fillna(0).astype(int)

        # Drop rows with missing sv_pct
        df = df.dropna(subset=['sv_pct', 'gaa'])

        rated = compute_goalie_score(df, season_year)
        if rated.empty:
            logger.warning(f"  No qualifying goalies for {season_year}")
            continue

        logger.info(f"  {len(rated)} qualified goalies | top: {rated.nlargest(1,'goalie_score')['name'].values}")
        all_seasons.append(rated)

        # Current season: build team-level ratings
        if season_id == CURRENT_SEASON:
            logger.info("  Building current season team ratings...")
            for _, row in rated.iterrows():
                # Handle multi-team goalies (traded; take last team if comma-separated)
                team_str = str(row.get('team', ''))
                team = team_str.split(',')[-1].strip() if ',' in team_str else team_str
                current_ratings.append({
                    'team':          team,
                    'player_id':     row['player_id'],
                    'starter_name':  row['name'],
                    'sv_pct':        row['sv_pct'],
                    'gaa':           row['gaa'],
                    'gp':            row['gp'],
                    'goalie_score':  row['goalie_score'],
                })

    # Save historical ratings
    if all_seasons:
        ratings_df = pd.concat(all_seasons, ignore_index=True)
        ratings_df.to_csv("nhl_goalie_ratings.csv", index=False)
        logger.info(f"\nSaved {len(ratings_df)} goalie-seasons to nhl_goalie_ratings.csv")
    else:
        logger.error("No goalie ratings computed!")
        return

    # Save team-level current ratings
    # For each team, take the goalie with the most GP (the starter)
    if current_ratings:
        curr_df = pd.DataFrame(current_ratings)
        # Pick starter per team = highest GP
        team_ratings = (
            curr_df.sort_values('gp', ascending=False)
            .drop_duplicates(subset=['team'])
            .reset_index(drop=True)
        )
        team_ratings.to_csv("nhl_goalie_team_ratings.csv", index=False)
        logger.info(f"Saved {len(team_ratings)} team goalie ratings to nhl_goalie_team_ratings.csv")
        logger.info("\nTop goalies by score:")
        for _, row in team_ratings.nlargest(5, 'goalie_score').iterrows():
            logger.info(f"  {row['team']}: {row['starter_name']} (SV%={row['sv_pct']:.3f}, GAA={row['gaa']:.2f}, score={row['goalie_score']:.2f})")
    else:
        # Fallback: create neutral ratings for all teams
        logger.warning("No current season data found. Creating neutral team ratings.")
        from apis.nhl import NHL_TEAMS
        neutral = pd.DataFrame([
            {'team': t, 'goalie_score': 0.0, 'starter_name': 'Unknown', 'sv_pct': 0.906, 'gaa': 2.90, 'gp': 0}
            for t in NHL_TEAMS
        ])
        neutral.to_csv("nhl_goalie_team_ratings.csv", index=False)


if __name__ == "__main__":
    main()
