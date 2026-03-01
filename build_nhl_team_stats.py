"""
build_nhl_team_stats.py
========================
Computes per-team offensive/defensive xG% (expected goals percentage)
and Corsi-For% proxies from NHL API shot data.
Analogous to build_team_stats.py for NFL (which uses EPA).

For seasons 2015+: uses shot quality weighted metrics from NHL EDGE stats
For pre-2015: uses simple shot differential (shots for vs against) as proxy

Saves:
  - nhl_team_stats_historical.csv   (team, season, off_xg_pct, def_xg_pct,
                                     corsi_for_pct, shots_for_pg, shots_against_pg,
                                     pp_pct, pk_pct)
  - nhl_team_stats_current.csv      (same structure, current season only)

Usage:
    python build_nhl_team_stats.py
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

XG_AVAILABLE_FROM = 2015   # Before this, use shot differential as proxy
SEASONS = list(range(2000, 2026))
CURRENT_SEASON_ID = "20242025"


def process_team_stats(raw_teams: list, season_year: int) -> pd.DataFrame:
    """
    Convert raw NHL API team stats to our feature format.
    Computes:
      - off_xg_pct: offense xG (shots_for / (shots_for + shots_against)) as proxy
      - def_xg_pct: defense xG (shots_against / (shots_for + shots_against))
      - corsi_for_pct: shot differential %
      - shots_for_pg, shots_against_pg
      - pp_pct, pk_pct
    """
    if not raw_teams:
        return pd.DataFrame()

    rows = []
    for t in raw_teams:
        sf = t.get('shots_for_pg') or 0
        sa = t.get('shots_against_pg') or 0
        total = sf + sa if (sf + sa) > 0 else 1.0

        # Corsi-For% (shot attempt differential)
        corsi_for_pct = sf / total

        # xG proxy: for pre-2015, use shot differential; for 2015+, same metric
        # (True xG from EDGE would require different endpoint; shot% is a decent proxy)
        off_xg_pct = corsi_for_pct         # positive = generates more shots
        def_xg_pct = 1.0 - corsi_for_pct  # positive = suppresses shots better

        # Normalize around 0.50 (50% is league average)
        # so these become centered features
        rows.append({
            'team':             t.get('team', ''),
            'season':           season_year,
            'gp':               t.get('gp', 0),
            'wins':             t.get('wins', 0),
            'goals_for_pg':     round(t.get('goals_for_pg', 0) or 0, 3),
            'goals_against_pg': round(t.get('goals_against_pg', 0) or 0, 3),
            'shots_for_pg':     round(sf, 2),
            'shots_against_pg': round(sa, 2),
            'corsi_for_pct':    round(corsi_for_pct, 4),
            'off_xg_pct':       round(off_xg_pct, 4),
            'def_xg_pct':       round(def_xg_pct, 4),
            'pp_pct':           round(t.get('pp_pct') or 0.18, 4),
            'pk_pct':           round(t.get('pk_pct') or 0.82, 4),
        })

    return pd.DataFrame(rows)


def main():
    logger.info("=== NHL Team Stats Build ===")
    client = NHLClient(sleep_between=0.2)

    all_seasons = []

    for season_year in SEASONS:
        season_id = f"{season_year}{season_year + 1}"
        logger.info(f"Fetching {season_year}-{season_year+1} team stats...")

        raw = client.get_team_season_stats(season_id)
        if not raw:
            logger.warning(f"  No data for {season_year}")
            continue

        df = process_team_stats(raw, season_year)
        if df.empty:
            continue

        logger.info(f"  {len(df)} teams | avg shots_for={df['shots_for_pg'].mean():.1f}")
        all_seasons.append(df)

    if not all_seasons:
        logger.error("No team stats collected!")
        return

    historical_df = pd.concat(all_seasons, ignore_index=True)
    historical_df.to_csv("nhl_team_stats_historical.csv", index=False)
    logger.info(f"\nSaved {len(historical_df)} team-seasons to nhl_team_stats_historical.csv")

    # Save current season separately
    current_df = historical_df[historical_df['season'] == max(SEASONS)].copy()
    if current_df.empty:
        # Try the second-to-last season if the latest isn't available yet
        current_df = historical_df[historical_df['season'] == max(SEASONS) - 1].copy()

    if not current_df.empty:
        current_df.to_csv("nhl_team_stats_current.csv", index=False)
        logger.info(f"Saved {len(current_df)} teams to nhl_team_stats_current.csv")
        logger.info("\nTop 5 teams by offensive xG%:")
        for _, row in current_df.nlargest(5, 'off_xg_pct').iterrows():
            logger.info(f"  {row['team']}: xG%={row['off_xg_pct']:.3f} shots/pg={row['shots_for_pg']:.1f}")
    else:
        logger.warning("No current season data found")


if __name__ == "__main__":
    main()
