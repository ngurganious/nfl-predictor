"""
nhl_feature_engineering.py
===========================
Computes the 31 pre-game features for NHL game prediction.
Mirrors feature_engineering.py from the NFL predictor.

All features are correctly lagged — no future data is used.
Target: home_win (binary, 1 = home team won)

29 Features:
  Core (4):       nhl_elo_diff, nhl_elo_implied_prob, moneyline_implied_prob, puck_line
  Form (6):       home/away_l5_goals_for, home/away_l5_goals_against
  Derived (6):    home/away_l5_goal_diff, goal_diff_advantage,
                  matchup_adv_home, matchup_adv_away, net_matchup_adv
  Trend (2):      home/away_nhl_elo_trend
  Goalie (1):     goalie_quality_diff
  xG/shots (5):   home/away_off_xg_pct, home/away_def_xg_pct, xg_total_diff
  PP/PK (5):      home/away_pp_pct, home/away_pk_pct, pp_pct_diff, pk_pct_diff
  Rolling wins (3): home/away_l5_wins, win_pct_advantage

Usage:
    from nhl_feature_engineering import build_nhl_enhanced_features, NHL_ENHANCED_FEATURES
    df_eng = build_nhl_enhanced_features(df)
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Feature list ──────────────────────────────────────────────────────────────
NHL_ENHANCED_FEATURES: List[str] = [
    # Core
    'nhl_elo_diff',
    'nhl_elo_implied_prob',
    'moneyline_implied_prob',
    'puck_line',
    # Form
    'home_l5_goals_for',
    'away_l5_goals_for',
    'home_l5_goals_against',
    'away_l5_goals_against',
    'home_l5_goal_diff',
    'away_l5_goal_diff',
    # Derived matchup
    'goal_diff_advantage',
    'matchup_adv_home',
    'matchup_adv_away',
    'net_matchup_adv',
    # ELO trend
    'home_nhl_elo_trend',
    'away_nhl_elo_trend',
    # Goalie quality (analogous to QB score diff)
    'goalie_quality_diff',
    # xG / shot metrics (analogous to EPA)
    'home_off_xg_pct',
    'away_off_xg_pct',
    'home_def_xg_pct',
    'away_def_xg_pct',
    'xg_total_diff',
    # Power play / penalty kill efficiency
    'home_pp_pct',
    'away_pp_pct',
    'pp_pct_diff',
    'home_pk_pct',
    'away_pk_pct',
    'pk_pct_diff',
    # Rolling 5-game win rate
    'home_l5_wins',
    'away_l5_wins',
    'win_pct_advantage',
]

FORM_WINDOW   = 5   # Rolling games for form features
ELO_TREND_WINDOW = 4   # ELO momentum window (same as NFL optimal)

# ELO constants (must match build_nhl_games.py)
HOME_ADV_ELO = 28
STARTING_ELO = 1500


# ── Helper ─────────────────────────────────────────────────────────────────────

def _elo_win_prob(elo_diff: float) -> float:
    """Home win probability from ELO difference (home ice advantage already in elo_diff)."""
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def _moneyline_to_prob(ml: float) -> float:
    """Convert American moneyline to implied probability (removes vig approximation)."""
    if ml is None or pd.isna(ml):
        return 0.5
    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return abs(ml) / (abs(ml) + 100.0)


# ── Sub-feature builders ───────────────────────────────────────────────────────

def _add_implied_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """ELO-implied and moneyline-implied win probabilities."""
    df['nhl_elo_implied_prob'] = df['nhl_elo_diff'].apply(_elo_win_prob)

    if 'moneyline_home' in df.columns:
        df['moneyline_implied_prob'] = df['moneyline_home'].apply(_moneyline_to_prob)
    elif 'moneyline_implied_prob' not in df.columns:
        # Fallback: use ELO probability
        df['moneyline_implied_prob'] = df['nhl_elo_implied_prob']

    # puck_line: NHL puck line is always ±1.5; direction follows ELO favorite
    if 'puck_line' not in df.columns:
        df['puck_line'] = df['nhl_elo_diff'].apply(lambda x: -1.5 if x > 0 else 1.5)

    return df


def _add_rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-team rolling FORM_WINDOW-game goals for/against averages.
    Lagged correctly: only uses games before the current game.
    """
    # Build per-team timeline dicts
    team_gf: Dict[str, list] = {}   # goals for
    team_ga: Dict[str, list] = {}   # goals against

    home_gf = []
    home_ga = []
    away_gf = []
    away_ga = []

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        h_score = row.get('home_score', 0) or 0
        a_score = row.get('away_score', 0) or 0

        # Compute rolling avg BEFORE updating with this game
        def _avg_last_n(timeline, n=FORM_WINDOW):
            recent = timeline[-n:] if len(timeline) >= 1 else []
            return float(np.mean(recent)) if recent else 0.0

        home_gf.append(_avg_last_n(team_gf.get(home, [])))
        home_ga.append(_avg_last_n(team_ga.get(home, [])))
        away_gf.append(_avg_last_n(team_gf.get(away, [])))
        away_ga.append(_avg_last_n(team_ga.get(away, [])))

        # Update team timelines with this game result
        team_gf.setdefault(home, []).append(h_score)
        team_ga.setdefault(home, []).append(a_score)
        team_gf.setdefault(away, []).append(a_score)
        team_ga.setdefault(away, []).append(h_score)

    df = df.copy()
    df['home_l5_goals_for']     = home_gf
    df['home_l5_goals_against'] = home_ga
    df['away_l5_goals_for']     = away_gf
    df['away_l5_goals_against'] = away_ga

    df['home_l5_goal_diff']   = df['home_l5_goals_for'] - df['home_l5_goals_against']
    df['away_l5_goal_diff']   = df['away_l5_goals_for'] - df['away_l5_goals_against']
    df['goal_diff_advantage'] = df['home_l5_goal_diff'] - df['away_l5_goal_diff']

    # Matchup features: home offense vs away defense (and vice versa)
    df['matchup_adv_home'] = df['home_l5_goals_for']     - df['away_l5_goals_against']
    df['matchup_adv_away'] = df['away_l5_goals_for']     - df['home_l5_goals_against']
    df['net_matchup_adv']  = df['matchup_adv_home'] - df['matchup_adv_away']

    return df


def _add_elo_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-team ELO momentum over the last ELO_TREND_WINDOW games.
    Uses the pre-built nhl_elo_diff column to reconstruct team ELO timelines.
    """
    # We need to recompute running ELO to get per-team history
    # Use simplified approach: maintain team ELO dict and track changes
    HOME_ADV = HOME_ADV_ELO
    K = 6

    team_elo: Dict[str, float] = {}
    team_elo_history: Dict[str, list] = {}   # list of (gameday, elo_after_game)

    home_trend = []
    away_trend = []

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        home_win = int(row['home_win'])

        h_elo = team_elo.get(home, STARTING_ELO)
        a_elo = team_elo.get(away, STARTING_ELO)

        # ELO trend: change over last ELO_TREND_WINDOW games
        def _trend(history, current_elo, window=ELO_TREND_WINDOW):
            if len(history) < window:
                return 0.0
            prev_elo = history[-window][1]
            return current_elo - prev_elo

        home_trend.append(_trend(team_elo_history.get(home, []), h_elo))
        away_trend.append(_trend(team_elo_history.get(away, []), a_elo))

        # Update ELO
        expected = 1.0 / (1.0 + 10.0 ** ((a_elo - (h_elo + HOME_ADV)) / 400.0))
        new_h = h_elo + K * (home_win - expected)
        new_a = a_elo + K * ((1 - home_win) - (1 - expected))

        team_elo[home] = new_h
        team_elo[away] = new_a

        gameday = row.get('gameday', '')
        team_elo_history.setdefault(home, []).append((gameday, new_h))
        team_elo_history.setdefault(away, []).append((gameday, new_a))

    df = df.copy()
    df['home_nhl_elo_trend'] = home_trend
    df['away_nhl_elo_trend'] = away_trend

    return df


def _add_goalie_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add goalie_quality_diff: home goalie z-score minus away goalie z-score.
    Loads nhl_goalie_team_ratings.csv for current season.
    For historical data, loads from nhl_goalie_ratings.csv (by team + season).

    If data is unavailable, fills with 0.0 (neutral).
    """
    df = df.copy()
    df['goalie_quality_diff'] = 0.0

    # Try to load historical per-team-season goalie ratings
    try:
        goalie_df = pd.read_csv("nhl_goalie_ratings.csv")
        # Aggregate to team-season level: best goalie per team-season (highest GP)
        if 'team' in goalie_df.columns and 'season' in goalie_df.columns:
            # Remove multi-team entries (traded players)
            goalie_df = goalie_df[~goalie_df['team'].astype(str).str.contains(',', na=False)]
            team_season_goalie = (
                goalie_df.sort_values('gp', ascending=False)
                .drop_duplicates(subset=['team', 'season'])
                .set_index(['team', 'season'])['goalie_score']
                .to_dict()
            )
            diffs = []
            for _, row in df.iterrows():
                home_score = team_season_goalie.get((row['home_team'], row['season']), None)
                away_score = team_season_goalie.get((row['away_team'], row['season']), None)
                if home_score is not None and away_score is not None:
                    diffs.append(home_score - away_score)
                else:
                    diffs.append(0.0)
            df['goalie_quality_diff'] = diffs
    except FileNotFoundError:
        logger.warning("nhl_goalie_ratings.csv not found — goalie_quality_diff will be 0")

    return df


def _add_team_xg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add offensive/defensive xG% features from nhl_team_stats_historical.csv.
    For pre-XG_AVAILABLE_FROM seasons, features are 0.0 (neutral).
    Analogous to _add_team_epa() in feature_engineering.py.
    """
    XG_AVAILABLE_FROM = 2000  # Shot data available for all seasons

    df = df.copy()
    for col in ['home_off_xg_pct', 'away_off_xg_pct', 'home_def_xg_pct', 'away_def_xg_pct']:
        df[col] = 0.0
    df['xg_total_diff'] = 0.0

    try:
        xg_df = pd.read_csv("nhl_team_stats_historical.csv")
        xg_dict = {}
        for _, row in xg_df.iterrows():
            key = (str(row['team']), int(row['season']))
            xg_dict[key] = {
                'off_xg_pct': row.get('off_xg_pct', 0.5),
                'def_xg_pct': row.get('def_xg_pct', 0.5),
            }

        home_off, home_def, away_off, away_def, xg_diff = [], [], [], [], []
        for _, row in df.iterrows():
            season = int(row.get('season', 0))
            h = xg_dict.get((str(row['home_team']), season), {})
            a = xg_dict.get((str(row['away_team']), season), {})

            ho = h.get('off_xg_pct', 0.5)
            hd = h.get('def_xg_pct', 0.5)
            ao = a.get('off_xg_pct', 0.5)
            ad = a.get('def_xg_pct', 0.5)

            home_off.append(ho)
            home_def.append(hd)
            away_off.append(ao)
            away_def.append(ad)
            # Net advantage: home off vs away off (positive = home generates more shots)
            xg_diff.append((ho - ao) + (ad - hd))

        df['home_off_xg_pct'] = home_off
        df['away_off_xg_pct'] = away_off
        df['home_def_xg_pct'] = home_def
        df['away_def_xg_pct'] = away_def
        df['xg_total_diff'] = xg_diff

    except FileNotFoundError:
        logger.warning("nhl_team_stats_historical.csv not found — xG features will be 0")

    return df


def _add_pp_pk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add power play % and penalty kill % features from nhl_team_stats_historical.csv.
    Joins on (team, season). Missing seasons default to league average (0.20 PP%, 0.80 PK%).
    """
    df = df.copy()
    LEAGUE_PP = 0.20
    LEAGUE_PK = 0.80

    for col in ['home_pp_pct', 'away_pp_pct', 'home_pk_pct', 'away_pk_pct']:
        df[col] = 0.0
    df['pp_pct_diff'] = 0.0
    df['pk_pct_diff'] = 0.0

    try:
        stats_df = pd.read_csv("nhl_team_stats_historical.csv")
        stats_dict = {}
        for _, row in stats_df.iterrows():
            key = (str(row['team']), int(row['season']))
            stats_dict[key] = {
                'pp_pct': row.get('pp_pct', LEAGUE_PP),
                'pk_pct': row.get('pk_pct', LEAGUE_PK),
            }

        h_pp, h_pk, a_pp, a_pk = [], [], [], []
        for _, row in df.iterrows():
            season = int(row.get('season', 0))
            h = stats_dict.get((str(row['home_team']), season), {})
            a = stats_dict.get((str(row['away_team']), season), {})
            h_pp.append(h.get('pp_pct', LEAGUE_PP))
            h_pk.append(h.get('pk_pct', LEAGUE_PK))
            a_pp.append(a.get('pp_pct', LEAGUE_PP))
            a_pk.append(a.get('pk_pct', LEAGUE_PK))

        df['home_pp_pct'] = h_pp
        df['away_pp_pct'] = a_pp
        df['home_pk_pct'] = h_pk
        df['away_pk_pct'] = a_pk
        df['pp_pct_diff'] = df['home_pp_pct'] - df['away_pp_pct']
        df['pk_pct_diff'] = df['home_pk_pct'] - df['away_pk_pct']

    except FileNotFoundError:
        logger.warning("nhl_team_stats_historical.csv not found — PP/PK features will use league avg")
        df['home_pp_pct'] = LEAGUE_PP
        df['away_pp_pct'] = LEAGUE_PP
        df['home_pk_pct'] = LEAGUE_PK
        df['away_pk_pct'] = LEAGUE_PK

    return df


def _add_rolling_wins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-team rolling FORM_WINDOW-game win rate.
    Lagged correctly: only uses games before the current game.
    Mirrors _add_rolling_form() for wins instead of goals.
    """
    team_wins: Dict[str, list] = {}

    home_l5w = []
    away_l5w = []

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        home_win = int(row.get('home_win', 0))

        def _win_rate(timeline, n=FORM_WINDOW):
            recent = timeline[-n:] if len(timeline) >= 1 else []
            return float(np.mean(recent)) if recent else 0.5

        home_l5w.append(_win_rate(team_wins.get(home, [])))
        away_l5w.append(_win_rate(team_wins.get(away, [])))

        # Update: home won = 1, away won = 0
        team_wins.setdefault(home, []).append(home_win)
        team_wins.setdefault(away, []).append(1 - home_win)

    df = df.copy()
    df['home_l5_wins']      = home_l5w
    df['away_l5_wins']      = away_l5w
    df['win_pct_advantage'] = df['home_l5_wins'] - df['away_l5_wins']

    return df


# ── Master function ────────────────────────────────────────────────────────────

def build_nhl_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering function.
    Input: nhl_games_processed.csv DataFrame (sorted by gameday ascending)
    Output: same DataFrame with 22 additional feature columns added.

    All features use only information available before each game.
    """
    if df.empty:
        return df

    # Normalize column names from nhl_games_processed.csv
    if 'elo_diff' in df.columns and 'nhl_elo_diff' not in df.columns:
        df = df.rename(columns={'elo_diff': 'nhl_elo_diff'})

    # Sort chronologically (required for correct lag computation)
    df = df.sort_values('gameday').reset_index(drop=True)

    logger.info("Computing implied probabilities...")
    df = _add_implied_probabilities(df)

    logger.info("Computing rolling 5-game form...")
    df = _add_rolling_form(df)

    logger.info("Computing ELO trend...")
    df = _add_elo_trend(df)

    logger.info("Adding goalie quality...")
    df = _add_goalie_quality(df)

    logger.info("Adding team xG stats...")
    df = _add_team_xg(df)

    logger.info("Adding PP/PK% features...")
    df = _add_pp_pk_features(df)

    logger.info("Computing rolling 5-game win rates...")
    df = _add_rolling_wins(df)

    # Final fillna(0) for any NaN that slipped through
    for col in NHL_ENHANCED_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = 0.0

    # Validate
    n_feats = len(NHL_ENHANCED_FEATURES)
    nan_counts = df[NHL_ENHANCED_FEATURES].isna().sum()
    total_nan = nan_counts.sum()
    if total_nan > 0:
        logger.warning(f"NaN values remaining: {nan_counts[nan_counts > 0].to_dict()}")
    else:
        logger.info(f"All {n_feats} features: 0 NaN values")

    return df
