"""
build_mlb_games.py
==================
Fetches all MLB regular-season game results 2000–2025 from the official
MLB Stats API (statsapi.mlb.com, no auth required), computes ELO ratings,
and saves:

  - mlb_games_processed.csv  (all regular-season games with ELO + rolling form)
  - mlb_elo_ratings.pkl      (dict: team_abbrev → current ELO)

Run once (~5 min including cache warming):
    python build_mlb_games.py

ELO system:
  - Base K = 12 (162-game season — less per-game signal than NFL K=20)
  - Home field advantage: +35 ELO points
  - Season regression: elo = prev * 0.67 + 1500 * 0.33 at season start
  - New/unknown team starts at 1500
"""

import os, sys, pickle, time, logging
import pandas as pd
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apis.mlb import MLBClient, normalize_team

# ── Constants ──────────────────────────────────────────────────────────────────
SEASONS      = list(range(2000, 2026))
ELO_K        = 12
HOME_ADV     = 35
ELO_DEFAULT  = 1500
SEASON_REGRESS = 0.67       # elo = prev*0.67 + 1500*0.33 at each new season

OUT_CSV = "mlb_games_processed.csv"
OUT_ELO = "mlb_elo_ratings.pkl"

# ── ELO helpers ────────────────────────────────────────────────────────────────
def expected_win(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

def regress_elos(elo_dict):
    return {t: elo * SEASON_REGRESS + ELO_DEFAULT * (1 - SEASON_REGRESS)
            for t, elo in elo_dict.items()}

# ── ELO computation ────────────────────────────────────────────────────────────
def compute_elos(games_df):
    games_df = games_df.sort_values('game_date').reset_index(drop=True)
    elo_ratings = {}
    current_season = None
    rows = []

    for _, row in games_df.iterrows():
        season    = row['season']
        home      = row['home_team']
        away      = row['away_team']
        home_win  = row['home_win']

        if season != current_season:
            elo_ratings = regress_elos(elo_ratings)
            current_season = season

        home_elo = elo_ratings.get(home, ELO_DEFAULT)
        away_elo = elo_ratings.get(away, ELO_DEFAULT)

        adj_home = home_elo + HOME_ADV
        exp_h    = expected_win(adj_home, away_elo)

        home_elo_new = home_elo + ELO_K * (home_win - exp_h)
        away_elo_new = away_elo + ELO_K * ((1 - home_win) - (1 - exp_h))

        rows.append({
            'home_elo_before': home_elo,
            'away_elo_before': away_elo,
            'home_elo_after':  home_elo_new,
            'away_elo_after':  away_elo_new,
            'elo_diff':        home_elo + HOME_ADV - away_elo,
        })

        elo_ratings[home] = home_elo_new
        elo_ratings[away] = away_elo_new

    elo_df = pd.DataFrame(rows)
    for col in elo_df.columns:
        games_df[col] = elo_df[col].values

    return games_df, elo_ratings


# ── Rolling form features ──────────────────────────────────────────────────────
def add_rolling_form(games_df, window=5):
    games_df = games_df.sort_values('game_date').reset_index(drop=True)

    home_rows = games_df[['game_date','home_team','home_score','away_score','season']].copy()
    home_rows.columns = ['game_date','team','runs_for','runs_against','season']
    home_rows['win'] = (games_df['home_win'] == 1).astype(int).values

    away_rows = games_df[['game_date','away_team','away_score','home_score','season']].copy()
    away_rows.columns = ['game_date','team','runs_for','runs_against','season']
    away_rows['win'] = (games_df['home_win'] == 0).astype(int).values

    team_games = pd.concat([home_rows, away_rows]).sort_values(['team','game_date']).reset_index(drop=True)

    roll_lookup = {}
    for team, grp in team_games.groupby('team'):
        grp = grp.sort_values('game_date').reset_index(drop=True)
        roll_lookup[team] = grp[['game_date','runs_for','runs_against','win']].copy()

    def get_rolling(team, date, col, w):
        if team not in roll_lookup:
            return np.nan
        g = roll_lookup[team]
        past = g[g['game_date'] < date]
        if past.empty:
            return np.nan
        return past.tail(w)[col].mean()

    logger.info("Computing rolling form features (this may take a few minutes)...")
    dates_arr  = games_df['game_date'].values
    home_teams = games_df['home_team'].values
    away_teams = games_df['away_team'].values

    h_rf, h_ra, h_w = [], [], []
    a_rf, a_ra, a_w = [], [], []

    for i in range(len(games_df)):
        d = dates_arr[i]
        h = home_teams[i]
        a = away_teams[i]
        h_rf.append(get_rolling(h, d, 'runs_for',     window))
        h_ra.append(get_rolling(h, d, 'runs_against', window))
        h_w.append( get_rolling(h, d, 'win',          window))
        a_rf.append(get_rolling(a, d, 'runs_for',     window))
        a_ra.append(get_rolling(a, d, 'runs_against', window))
        a_w.append( get_rolling(a, d, 'win',          window))

    games_df[f'home_l{window}_runs_for']     = h_rf
    games_df[f'home_l{window}_runs_against'] = h_ra
    games_df[f'home_l{window}_wins']         = h_w
    games_df[f'away_l{window}_runs_for']     = a_rf
    games_df[f'away_l{window}_runs_against'] = a_ra
    games_df[f'away_l{window}_wins']         = a_w

    games_df[f'home_l{window}_run_diff'] = (
        games_df[f'home_l{window}_runs_for'] - games_df[f'home_l{window}_runs_against']
    )
    games_df[f'away_l{window}_run_diff'] = (
        games_df[f'away_l{window}_runs_for'] - games_df[f'away_l{window}_runs_against']
    )

    return games_df


# ── ELO trend (4-game rolling ELO change) ─────────────────────────────────────
def add_elo_trend(games_df, window=4):
    home_rows = games_df[['game_date','home_team','home_elo_after','home_elo_before']].copy()
    home_rows.columns = ['game_date','team','elo_after','elo_before']
    away_rows = games_df[['game_date','away_team','away_elo_after','away_elo_before']].copy()
    away_rows.columns = ['game_date','team','elo_after','elo_before']

    team_elos = pd.concat([home_rows, away_rows]).sort_values(['team','game_date'])
    trend_lookup = {}
    for team, grp in team_elos.groupby('team'):
        grp = grp.sort_values('game_date').reset_index(drop=True)
        grp['elo_change'] = grp['elo_after'] - grp['elo_before']
        trend_lookup[team] = grp[['game_date','elo_change']].copy()

    def get_trend(team, date, w):
        if team not in trend_lookup:
            return 0.0
        g = trend_lookup[team]
        past = g[g['game_date'] < date]
        if past.empty:
            return 0.0
        return past.tail(w)['elo_change'].sum()

    dates_arr  = games_df['game_date'].values
    home_teams = games_df['home_team'].values
    away_teams = games_df['away_team'].values

    games_df['home_elo_trend'] = [get_trend(home_teams[i], dates_arr[i], window) for i in range(len(games_df))]
    games_df['away_elo_trend'] = [get_trend(away_teams[i], dates_arr[i], window) for i in range(len(games_df))]
    return games_df


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    client = MLBClient()
    all_records = []

    logger.info(f"Fetching MLB game results {SEASONS[0]}–{SEASONS[-1]} from MLB Stats API...")
    logger.info("(Cache-enabled — first run fetches from API, subsequent runs are instant)")

    for year in tqdm(SEASONS, desc="Seasons"):
        records = client.get_season_schedule(year)
        if records:
            all_records.extend(records)
            logger.info(f"  {year}: {len(records)} games")
        else:
            logger.warning(f"  {year}: no data")
        time.sleep(0.1)

    if not all_records:
        logger.error("No game records collected. Check network/API.")
        sys.exit(1)

    logger.info(f"\nTotal records: {len(all_records)}")
    games_df = pd.DataFrame(all_records)
    games_df['game_date'] = pd.to_datetime(games_df['game_date'])
    games_df = games_df.dropna(subset=['home_score','away_score','home_win'])
    games_df = games_df.drop_duplicates(subset=['game_date','home_team','away_team'], keep='first')
    games_df = games_df.sort_values('game_date').reset_index(drop=True)

    logger.info(f"Games after dedup: {len(games_df)}")
    logger.info(f"Seasons: {games_df['season'].min()}–{games_df['season'].max()}")
    logger.info(f"Teams: {sorted(games_df['home_team'].unique())}")

    # ELO
    logger.info("Computing ELO ratings...")
    games_df, elo_ratings = compute_elos(games_df)

    # Rolling form
    games_df = add_rolling_form(games_df)

    # ELO trend
    games_df = add_elo_trend(games_df)

    # Save
    games_df.to_csv(OUT_CSV, index=False)
    logger.info(f"\nSaved {OUT_CSV} ({len(games_df)} rows, {len(games_df.columns)} cols)")

    with open(OUT_ELO, 'wb') as f:
        pickle.dump(elo_ratings, f)
    logger.info(f"Saved {OUT_ELO} ({len(elo_ratings)} teams)")

    sorted_elos = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    logger.info("\nTop 5 ELO teams:")
    for team, elo in sorted_elos[:5]:
        logger.info(f"  {team}: {elo:.1f}")
    logger.info("Bottom 5 ELO teams:")
    for team, elo in sorted_elos[-5:]:
        logger.info(f"  {team}: {elo:.1f}")

    total = len(games_df)
    home_wins = games_df['home_win'].sum()
    logger.info(f"\nHome win rate: {home_wins/total:.1%} ({int(home_wins)}/{total})")
    logger.info("Done!")


if __name__ == '__main__':
    main()
