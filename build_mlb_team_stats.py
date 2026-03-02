"""
build_mlb_team_stats.py
========================
Computes per-team offensive and pitching stats from FanGraphs via pybaseball.
Analogous to build_team_stats.py (NFL EPA) and build_nhl_team_stats.py (xG%).

Stats collected (FanGraphs team batting + team pitching):
  Batting: wOBA, wRC+, OPS, BB%, K%, ISO
  Pitching: ERA-, FIP-, WHIP, K/9, BB/9, HR/9

Saves:
  - mlb_team_stats_historical.csv  (team, season, batting + pitching stats)
  - mlb_team_stats_current.csv     (current season only, same structure)

Usage:
    python build_mlb_team_stats.py
"""

import os, sys, logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

try:
    from pybaseball import team_batting, team_pitching, cache as pb_cache
    pb_cache.enable()
except ImportError:
    logger.error("pybaseball not installed. Run: pip install pybaseball")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────
SEASONS         = list(range(2000, 2026))
CURRENT_SEASON  = 2025
OUT_HIST        = "mlb_team_stats_historical.csv"
OUT_CURRENT     = "mlb_team_stats_current.csv"

# Normalize Baseball Reference / FanGraphs team names → EdgeIQ abbreviations
# FanGraphs uses full team names; we map from common abbreviations after parsing
TEAM_ABBREV_MAP = {
    'Angels':           'LAA', 'Astros':           'HOU', 'Athletics':        'OAK',
    'Blue Jays':        'TOR', 'Braves':           'ATL', 'Brewers':          'MIL',
    'Cardinals':        'STL', 'Cubs':             'CHC', 'Diamondbacks':     'ARI',
    'Dodgers':          'LAD', 'Giants':           'SF',  'Guardians':        'CLE',
    'Indians':          'CLE', 'Mariners':         'SEA', 'Marlins':          'MIA',
    'Mets':             'NYM', 'Nationals':        'WSH', 'Orioles':          'BAL',
    'Padres':           'SD',  'Phillies':         'PHI', 'Pirates':          'PIT',
    'Rangers':          'TEX', 'Rays':             'TB',  'Red Sox':          'BOS',
    'Reds':             'CIN', 'Rockies':          'COL', 'Royals':           'KC',
    'Tigers':           'DET', 'Twins':            'MIN', 'White Sox':        'CWS',
    'Yankees':          'NYY', 'Expos':            'MTL', 'Devil Rays':       'TB',
    'Florida Marlins':  'MIA',
}

def normalize_team(name):
    if not name:
        return name
    for key, abbrev in TEAM_ABBREV_MAP.items():
        if key.lower() in str(name).lower():
            return abbrev
    return str(name)


def fetch_batting_stats(year):
    try:
        df = team_batting(year, year)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df['season'] = year
        # team_batting returns FanGraphs team batting: columns include
        # Team, G, PA, HR, R, RBI, wOBA, wRC+, OBP, SLG, OPS, BB%, K%, ISO
        batting_cols = {
            'Team': 'team_name',
            'wOBA': 'woba',
            'wRC+': 'wrc_plus',
            'OPS':  'ops',
            'BB%':  'bb_pct',
            'K%':   'k_pct',
            'ISO':  'iso',
        }
        available = {k: v for k, v in batting_cols.items() if k in df.columns}
        df = df.rename(columns=available)
        df['team'] = df['team_name'].apply(normalize_team)
        keep_cols = ['season', 'team', 'team_name'] + [v for k, v in available.items() if v != 'team_name']
        return df[[c for c in keep_cols if c in df.columns]]
    except Exception as e:
        logger.warning(f"  {year} batting error: {e}")
        return pd.DataFrame()


def fetch_pitching_stats(year):
    try:
        df = team_pitching(year, year)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df['season'] = year
        # team_pitching returns FanGraphs team pitching: columns include
        # Team, ERA, FIP, WHIP, K/9, BB/9, HR/9, ERA-, FIP-, xFIP
        pitching_cols = {
            'Team':  'team_name',
            'ERA-':  'era_minus',
            'FIP-':  'fip_minus',
            'xFIP':  'xfip',
            'WHIP':  'whip',
            'K/9':   'k_per_9',
            'BB/9':  'bb_per_9',
            'HR/9':  'hr_per_9',
            'ERA':   'era',
            'FIP':   'fip',
        }
        available = {k: v for k, v in pitching_cols.items() if k in df.columns}
        df = df.rename(columns=available)
        df['team'] = df['team_name'].apply(normalize_team)
        keep_cols = ['season', 'team', 'team_name'] + [v for k, v in available.items() if v != 'team_name']
        return df[[c for c in keep_cols if c in df.columns]]
    except Exception as e:
        logger.warning(f"  {year} pitching error: {e}")
        return pd.DataFrame()


def main():
    all_batting  = []
    all_pitching = []

    logger.info(f"Fetching MLB team stats {SEASONS[0]}–{SEASONS[-1]} via FanGraphs (pybaseball)...")

    for year in SEASONS:
        logger.info(f"  Season {year}...")

        bat = fetch_batting_stats(year)
        if not bat.empty:
            all_batting.append(bat)
            logger.info(f"    batting: {len(bat)} teams")
        else:
            logger.warning(f"    batting: no data")

        pit = fetch_pitching_stats(year)
        if not pit.empty:
            all_pitching.append(pit)
            logger.info(f"    pitching: {len(pit)} teams")
        else:
            logger.warning(f"    pitching: no data")

    if not all_batting and not all_pitching:
        logger.error("No data collected. Check pybaseball and network.")
        sys.exit(1)

    # Merge batting + pitching on season + team
    if all_batting:
        bat_df = pd.concat(all_batting, ignore_index=True)
    else:
        bat_df = pd.DataFrame(columns=['season', 'team'])

    if all_pitching:
        pit_df = pd.concat(all_pitching, ignore_index=True)
    else:
        pit_df = pd.DataFrame(columns=['season', 'team'])

    if not bat_df.empty and not pit_df.empty:
        # Drop duplicate 'team_name' column if present in both
        if 'team_name' in pit_df.columns:
            pit_df = pit_df.drop(columns=['team_name'], errors='ignore')
        combined = bat_df.merge(pit_df, on=['season', 'team'], how='outer')
    elif not bat_df.empty:
        combined = bat_df
    else:
        combined = pit_df

    combined = combined.sort_values(['season', 'team']).reset_index(drop=True)

    # Save historical
    combined.to_csv(OUT_HIST, index=False)
    logger.info(f"\nSaved {OUT_HIST} ({len(combined)} rows, {len(combined.columns)} cols)")

    # Save current season
    current = combined[combined['season'] == CURRENT_SEASON].copy()
    if current.empty:
        # Fall back to most recent available season
        latest = combined['season'].max()
        current = combined[combined['season'] == latest].copy()
        logger.warning(f"  No {CURRENT_SEASON} data; using {latest} as current")

    current.to_csv(OUT_CURRENT, index=False)
    logger.info(f"Saved {OUT_CURRENT} ({len(current)} teams)")

    # Print sample
    print_cols = ['season', 'team', 'woba', 'wrc_plus', 'era_minus', 'fip_minus']
    show_cols = [c for c in print_cols if c in combined.columns]
    if show_cols and len(combined) > 0:
        logger.info(f"\nSample (most recent season with full data):")
        sample = combined.sort_values('season', ascending=False).head(10)
        logger.info(f"\n{sample[show_cols].to_string(index=False)}")

    logger.info("Done!")


if __name__ == '__main__':
    main()
