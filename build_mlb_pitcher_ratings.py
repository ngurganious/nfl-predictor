"""
build_mlb_pitcher_ratings.py
=============================
Computes per-season starting pitcher quality z-scores from FanGraphs via pybaseball.
Analogous to build_qb_ratings.py (NFL) and build_nhl_goalie_ratings.py (NHL).

Formula:
    pitcher_score = (-era_minus_z) * 0.35 + (-fip_minus_z) * 0.35 + kbb_z * 0.20 + (-whip_z) * 0.10
    (re-z-scored for cross-season consistency)

Where:
  - era_minus_z  = z-score of ERA- within season (lower ERA- is better → negate)
  - fip_minus_z  = z-score of FIP- within season (lower is better → negate)
  - kbb_z        = z-score of K/BB ratio (strikeouts per walk, higher is better)
  - whip_z       = z-score of WHIP (lower is better → negate)

ERA- and FIP- are park-adjusted (100 = league average, < 100 = better than average).
K/BB captures command and swing-and-miss simultaneously.

Qualification: minimum 15 starts (or 80 innings pitched) per season.

Saves:
  - mlb_pitcher_ratings.csv        (player_id, season, pitcher_score, era_minus, fip_minus,
                                     k_bb, whip, name, team)
  - mlb_pitcher_team_ratings.csv   (team, pitcher_score, starter_name, era_minus, fip_minus)
                                    current season only, one row per team

Usage:
    python build_mlb_pitcher_ratings.py
"""

import os, sys, logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

try:
    from pybaseball import pitching_stats, cache as pb_cache
    pb_cache.enable()
except ImportError:
    logger.error("pybaseball not installed. Run: pip install pybaseball")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────
SEASONS        = list(range(2000, 2026))
CURRENT_SEASON = 2025
MIN_IP         = 50     # minimum innings pitched to qualify (~ 8-9 starts)
MIN_GS         = 10     # minimum games started to qualify
OUT_HIST       = "mlb_pitcher_ratings.csv"
OUT_CURRENT    = "mlb_pitcher_team_ratings.csv"

# Team name → abbreviation (FanGraphs uses team names)
TEAM_ABBREV_MAP = {
    'Angels':        'LAA', 'Astros':       'HOU', 'Athletics':    'OAK',
    'Blue Jays':     'TOR', 'Braves':       'ATL', 'Brewers':      'MIL',
    'Cardinals':     'STL', 'Cubs':         'CHC', 'Diamondbacks': 'ARI',
    'Dodgers':       'LAD', 'Giants':       'SF',  'Guardians':    'CLE',
    'Indians':       'CLE', 'Mariners':     'SEA', 'Marlins':      'MIA',
    'Mets':          'NYM', 'Nationals':    'WSH', 'Orioles':      'BAL',
    'Padres':        'SD',  'Phillies':     'PHI', 'Pirates':      'PIT',
    'Rangers':       'TEX', 'Rays':         'TB',  'Red Sox':      'BOS',
    'Reds':          'CIN', 'Rockies':      'COL', 'Royals':       'KC',
    'Tigers':        'DET', 'Twins':        'MIN', 'White Sox':    'CWS',
    'Yankees':       'NYY', 'Expos':        'MTL', 'Devil Rays':   'TB',
}

def normalize_team(name):
    if pd.isna(name):
        return 'UNK'
    for key, abbrev in TEAM_ABBREV_MAP.items():
        if key.lower() in str(name).lower():
            return abbrev
    return str(name)


def z_score(series):
    mu = series.mean()
    sd = series.std()
    if sd < 1e-8:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sd


def fetch_pitching_season(year):
    """Fetch individual pitcher stats for a season via pybaseball (FanGraphs)."""
    try:
        # pitching_stats returns FanGraphs individual pitcher stats
        # qual=1 means minimum 1 IP (we filter afterward)
        df = pitching_stats(year, year, qual=1)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df['season'] = year
        return df
    except Exception as e:
        logger.warning(f"  {year}: {e}")
        return pd.DataFrame()


def compute_pitcher_score(df_season, year):
    """Compute z-score composite for qualified starters in a season."""
    # FanGraphs columns we need:
    #   Name, Team, IP, GS (games started), ERA-, FIP-, K/BB, WHIP
    # Some older seasons may not have ERA- or FIP-; fall back to raw ERA/FIP

    required_any = ['Name', 'IP']
    if not all(c in df_season.columns for c in required_any):
        logger.warning(f"  {year}: missing required columns, skipping")
        return pd.DataFrame()

    df = df_season.copy()

    # Qualify by games started or innings pitched
    ip_col = 'IP' if 'IP' in df.columns else None
    gs_col = 'GS' if 'GS' in df.columns else None

    if ip_col:
        df = df[df[ip_col] >= MIN_IP]
    if gs_col:
        df = df[df[gs_col] >= MIN_GS]

    if len(df) < 5:
        logger.warning(f"  {year}: only {len(df)} qualified starters, skipping")
        return pd.DataFrame()

    # Determine which ERA/FIP metrics are available
    has_era_minus = 'ERA-' in df.columns
    has_fip_minus = 'FIP-' in df.columns
    has_kbb       = 'K/BB' in df.columns
    has_whip      = 'WHIP' in df.columns
    has_era       = 'ERA' in df.columns
    has_fip       = 'FIP' in df.columns

    # ERA component (lower is better → negate z-score)
    if has_era_minus:
        df['era_minus'] = pd.to_numeric(df['ERA-'], errors='coerce').fillna(100)
        era_z = z_score(df['era_minus'])
    elif has_era:
        df['era_minus'] = pd.to_numeric(df['ERA'], errors='coerce').fillna(df['ERA'].median() if has_era else 4.5)
        era_z = z_score(df['era_minus'])
    else:
        era_z = pd.Series(0.0, index=df.index)
        df['era_minus'] = np.nan

    # FIP component (lower is better → negate z-score)
    if has_fip_minus:
        df['fip_minus'] = pd.to_numeric(df['FIP-'], errors='coerce').fillna(100)
        fip_z = z_score(df['fip_minus'])
    elif has_fip:
        df['fip_minus'] = pd.to_numeric(df['FIP'], errors='coerce').fillna(df['FIP'].median() if has_fip else 4.5)
        fip_z = z_score(df['fip_minus'])
    else:
        fip_z = pd.Series(0.0, index=df.index)
        df['fip_minus'] = np.nan

    # K/BB component (higher is better, keep positive)
    if has_kbb:
        df['k_bb'] = pd.to_numeric(df['K/BB'], errors='coerce').fillna(2.0)
        kbb_z = z_score(df['k_bb'])
    else:
        kbb_z = pd.Series(0.0, index=df.index)
        df['k_bb'] = np.nan

    # WHIP component (lower is better → negate)
    if has_whip:
        df['whip_val'] = pd.to_numeric(df['WHIP'], errors='coerce').fillna(1.3)
        whip_z = z_score(df['whip_val'])
    else:
        whip_z = pd.Series(0.0, index=df.index)
        df['whip_val'] = np.nan

    # Composite: lower ERA-/FIP-/WHIP is better (negate), higher K/BB is better
    raw_score = (-era_z * 0.35) + (-fip_z * 0.35) + (kbb_z * 0.20) + (-whip_z * 0.10)
    df['pitcher_score'] = z_score(raw_score)   # re-z-score for cross-season consistency

    # Normalize team
    team_col = 'Team' if 'Team' in df.columns else None
    df['team'] = df[team_col].apply(normalize_team) if team_col else 'UNK'

    # playerid if available
    id_col = 'IDfg' if 'IDfg' in df.columns else None

    out_cols = ['season', 'Name', 'team', 'pitcher_score', 'era_minus', 'fip_minus', 'k_bb', 'whip_val', 'IP']
    if id_col:
        df = df.rename(columns={id_col: 'player_id'})
        out_cols = ['player_id'] + out_cols
    if gs_col:
        out_cols.append('GS')

    out_cols = [c for c in out_cols if c in df.columns]
    result = df[out_cols].rename(columns={'Name': 'name', 'IP': 'ip', 'GS': 'gs'})
    result = result.sort_values('pitcher_score', ascending=False)
    return result


def main():
    all_records = []

    logger.info(f"Computing MLB starting pitcher quality ratings {SEASONS[0]}–{SEASONS[-1]}...")

    for year in SEASONS:
        logger.info(f"  Season {year}...")
        raw = fetch_pitching_season(year)
        if raw.empty:
            logger.warning(f"  {year}: no data")
            continue

        scored = compute_pitcher_score(raw, year)
        if not scored.empty:
            all_records.append(scored)
            logger.info(f"  → {len(scored)} qualified starters")

    if not all_records:
        logger.error("No pitcher ratings computed. Check pybaseball installation.")
        sys.exit(1)

    hist_df = pd.concat(all_records, ignore_index=True)
    hist_df = hist_df.sort_values(['season', 'pitcher_score'], ascending=[True, False])

    # Save historical ratings
    hist_df.to_csv(OUT_HIST, index=False)
    logger.info(f"\nSaved {OUT_HIST} ({len(hist_df)} pitcher-seasons, {hist_df['season'].nunique()} seasons)")

    # Build current-season team ratings (best starter per team)
    current_df = hist_df[hist_df['season'] == hist_df['season'].max()].copy()
    if current_df.empty:
        logger.warning("No current season data; using most recent season available")
        current_df = hist_df[hist_df['season'] == hist_df['season'].max()].copy()

    # One row per team — best starter (may not reflect current rotation; use top-3 average for robustness)
    team_ratings = []
    for team, grp in current_df.groupby('team'):
        grp = grp.sort_values('pitcher_score', ascending=False)
        top3 = grp.head(3)
        team_ratings.append({
            'team':              team,
            'pitcher_score':     top3['pitcher_score'].mean(),      # rotation quality (top 3 avg)
            'ace_pitcher_score': grp.iloc[0]['pitcher_score'],       # best starter
            'ace_name':          grp.iloc[0]['name'],
            'era_minus':         grp.iloc[0].get('era_minus', np.nan),
            'fip_minus':         grp.iloc[0].get('fip_minus', np.nan),
        })

    team_df = pd.DataFrame(team_ratings).sort_values('pitcher_score', ascending=False)
    team_df.to_csv(OUT_CURRENT, index=False)
    logger.info(f"Saved {OUT_CURRENT} ({len(team_df)} teams)")

    # Print top/bottom starters in most recent season
    season = current_df['season'].iloc[0]
    print_cols = ['name', 'team', 'pitcher_score', 'era_minus', 'fip_minus', 'k_bb']
    show_cols = [c for c in print_cols if c in current_df.columns]
    logger.info(f"\nTop 10 starters ({season}):")
    logger.info(f"\n{current_df.head(10)[show_cols].to_string(index=False)}")
    logger.info(f"\nBottom 10 starters ({season}):")
    logger.info(f"\n{current_df.tail(10)[show_cols].to_string(index=False)}")

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
