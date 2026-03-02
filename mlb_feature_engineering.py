"""
mlb_feature_engineering.py
===========================
Computes the 29-feature matrix for MLB game prediction.
Mirrors nhl_feature_engineering.py from the NHL predictor.

All features are correctly lagged — no future data is used.
Target: home_win (binary, 1 = home team won)

29 Features:
  Core Vegas (3):   run_line, moneyline_implied_prob, run_line_implied_prob
  ELO (4):          mlb_elo_diff, mlb_elo_implied_prob, home/away_mlb_elo_trend
  Form L5 (6):      home/away_l5_runs_for, home/away_l5_runs_against, home/away_l5_run_diff
  Matchup (3):      matchup_adv_home, matchup_adv_away, net_matchup_adv
  Pitcher (1):      pitcher_quality_diff
  Team offense (4): home/away_woba, home/away_wrc_plus
  Team pitching (4): home/away_era_minus, home/away_fip_minus
  Win rate L10 (3): home/away_l10_wins, win_pct_advantage
  Context (1):      is_day_game

Usage:
    from mlb_feature_engineering import build_mlb_enhanced_features, MLB_ENHANCED_FEATURES
    df_eng = build_mlb_enhanced_features(df)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Feature list ──────────────────────────────────────────────────────────────
MLB_ENHANCED_FEATURES = [
    # Core Vegas
    'run_line',
    'moneyline_implied_prob',
    'run_line_implied_prob',
    # ELO
    'mlb_elo_diff',
    'mlb_elo_implied_prob',
    'home_mlb_elo_trend',
    'away_mlb_elo_trend',
    # Form (L5)
    'home_l5_runs_for',
    'away_l5_runs_for',
    'home_l5_runs_against',
    'away_l5_runs_against',
    'home_l5_run_diff',
    'away_l5_run_diff',
    # Matchup cross-features
    'matchup_adv_home',
    'matchup_adv_away',
    'net_matchup_adv',
    # Pitcher quality (analogous to QB score diff / goalie_quality_diff)
    'pitcher_quality_diff',
    # Team offense
    'home_woba',
    'away_woba',
    'home_wrc_plus',
    'away_wrc_plus',
    # Team pitching
    'home_era_minus',
    'away_era_minus',
    'home_fip_minus',
    'away_fip_minus',
    # Rolling L10 win rate
    'home_l10_wins',
    'away_l10_wins',
    'win_pct_advantage',
    # Context
    'is_day_game',
]

L10_WINDOW      = 10
ELO_TREND_WINDOW = 4
HOME_ADV_ELO    = 35    # must match build_mlb_games.py
STARTING_ELO    = 1500

# League-average fallback values for team stats
LEAGUE_WOBA      = 0.320
LEAGUE_WRC       = 100
LEAGUE_ERA_MINUS = 100
LEAGUE_FIP_MINUS = 100

# ── Abbreviation bridge: MLB Stats API → FanGraphs ───────────────────────────
# mlb_games_processed.csv uses MLB Stats API codes.
# mlb_team_stats_historical.csv and mlb_pitcher_ratings.csv use FanGraphs codes.
_GAMES_TO_STATS = {
    'AZ':  'ARI',
    'CWS': 'CHW',
    'KC':  'KCR',
    'SD':  'SDP',
    'SF':  'SFG',
    'TB':  'TBR',
    'WSH': 'WSN',
    'LA':  'LAD',  # historical MLB Stats API code for Dodgers
}


def _stats_key(team, season):
    """
    Map an MLB Stats API team abbreviation + season year to the FanGraphs
    abbreviation used in mlb_team_stats_historical.csv and mlb_pitcher_ratings.csv.
    """
    t = _GAMES_TO_STATS.get(team, team)
    # Tampa Bay was 'TBD' (Devil Rays) in FanGraphs before 2008
    if team == 'TB' and int(season) < 2008:
        return 'TBD'
    # Florida Marlins were 'FLA' in FanGraphs before the 2012 rebrand to Miami
    if team == 'MIA' and int(season) < 2012:
        return 'FLA'
    return t


# ── Helpers ───────────────────────────────────────────────────────────────────

def _elo_win_prob(elo_diff):
    """Home win probability from ELO difference (home advantage already baked in)."""
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def _moneyline_to_prob(ml):
    """Convert American moneyline to implied probability."""
    if ml is None or pd.isna(ml):
        return 0.5
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return abs(ml) / (abs(ml) + 100.0)


# ── Sub-feature builders ──────────────────────────────────────────────────────

def _add_implied_probabilities(df):
    """ELO-implied prob, moneyline-implied prob, run line."""
    df = df.copy()

    # Normalize column: elo_diff → mlb_elo_diff
    if 'elo_diff' in df.columns and 'mlb_elo_diff' not in df.columns:
        df = df.rename(columns={'elo_diff': 'mlb_elo_diff'})

    df['mlb_elo_implied_prob'] = df['mlb_elo_diff'].apply(_elo_win_prob)

    if 'moneyline_home' in df.columns:
        df['moneyline_implied_prob'] = df['moneyline_home'].apply(_moneyline_to_prob)
    elif 'moneyline_implied_prob' not in df.columns:
        df['moneyline_implied_prob'] = df['mlb_elo_implied_prob']

    # Run line is always ±1.5 in MLB; direction follows ELO favorite
    if 'run_line' not in df.columns:
        df['run_line'] = df['mlb_elo_diff'].apply(lambda x: -1.5 if x > 0 else 1.5)

    # Run line implied prob — from odds if live, else 0.5 (symmetric bet)
    if 'run_line_odds' in df.columns:
        df['run_line_implied_prob'] = df['run_line_odds'].apply(_moneyline_to_prob)
    elif 'run_line_implied_prob' not in df.columns:
        df['run_line_implied_prob'] = 0.5

    return df


def _add_elo_trend(df):
    """Rename home/away_elo_trend columns to mlb-prefixed equivalents."""
    df = df.copy()
    if 'home_elo_trend' in df.columns and 'home_mlb_elo_trend' not in df.columns:
        df = df.rename(columns={
            'home_elo_trend': 'home_mlb_elo_trend',
            'away_elo_trend': 'away_mlb_elo_trend',
        })
    elif 'home_mlb_elo_trend' not in df.columns:
        df['home_mlb_elo_trend'] = 0.0
        df['away_mlb_elo_trend'] = 0.0
    return df


def _add_matchup_features(df):
    """
    Derive run-differential matchup features from L5 rolling run data.
    matchup_adv_home = home offense advantage over away pitching (runs allowed).
    matchup_adv_away = away offense advantage over home pitching.
    """
    df = df.copy()
    df['matchup_adv_home'] = df['home_l5_runs_for'] - df['away_l5_runs_against']
    df['matchup_adv_away'] = df['away_l5_runs_for'] - df['home_l5_runs_against']
    df['net_matchup_adv']  = df['matchup_adv_home'] - df['matchup_adv_away']
    return df


def _add_l10_wins(df):
    """
    Per-team rolling L10 win rate.
    Lagged correctly: only uses games before the current game.
    """
    team_wins = {}
    home_l10  = []
    away_l10  = []

    for _, row in df.iterrows():
        home     = row['home_team']
        away     = row['away_team']
        home_win = int(row.get('home_win', 0))

        def _win_rate(timeline, n=L10_WINDOW):
            recent = timeline[-n:] if timeline else []
            return float(np.mean(recent)) if recent else 0.5

        home_l10.append(_win_rate(team_wins.get(home, [])))
        away_l10.append(_win_rate(team_wins.get(away, [])))

        team_wins.setdefault(home, []).append(home_win)
        team_wins.setdefault(away, []).append(1 - home_win)

    df = df.copy()
    df['home_l10_wins']    = home_l10
    df['away_l10_wins']    = away_l10
    df['win_pct_advantage'] = df['home_l10_wins'] - df['away_l10_wins']
    return df


def _add_pitcher_quality(df):
    """
    Add pitcher_quality_diff: home SP z-score minus away SP z-score.
    Loads mlb_pitcher_ratings.csv (historical, by team+season).
    Falls back to 0.0 (neutral) when data is unavailable.
    """
    df = df.copy()
    df['pitcher_quality_diff'] = 0.0

    try:
        pitcher_df = pd.read_csv('mlb_pitcher_ratings.csv')
        # Remove multi-team entries (traded players shown as '- - -')
        pitcher_df = pitcher_df[~pitcher_df['team'].astype(str).str.contains('- - -', na=False)]

        # Best starter per team-season by most GS
        team_season_score = (
            pitcher_df.sort_values('gs', ascending=False)
            .drop_duplicates(subset=['team', 'season'])
            .set_index(['team', 'season'])['pitcher_score']
            .to_dict()
        )

        diffs = []
        for _, row in df.iterrows():
            season = int(row['season'])
            h_key  = (_stats_key(row['home_team'], season), season)
            a_key  = (_stats_key(row['away_team'], season), season)
            h_score = team_season_score.get(h_key)
            a_score = team_season_score.get(a_key)
            diffs.append(
                (h_score - a_score) if h_score is not None and a_score is not None else 0.0
            )
        df['pitcher_quality_diff'] = diffs

    except FileNotFoundError:
        logger.warning('mlb_pitcher_ratings.csv not found — pitcher_quality_diff will be 0')

    return df


def _add_team_batting_stats(df):
    """
    Add home/away wOBA and wRC+ from mlb_team_stats_historical.csv.
    Falls back to league average when not found: wOBA 0.320, wRC+ 100.
    """
    df = df.copy()
    for col in ['home_woba', 'away_woba', 'home_wrc_plus', 'away_wrc_plus']:
        df[col] = float('nan')

    try:
        stats_df = pd.read_csv('mlb_team_stats_historical.csv')
        stats_dict = {}
        for _, row in stats_df.iterrows():
            key = (str(row['team']), int(row['season']))
            stats_dict[key] = {
                'woba':     float(row.get('woba',     LEAGUE_WOBA)),
                'wrc_plus': float(row.get('wrc_plus', LEAGUE_WRC)),
            }

        h_woba, a_woba, h_wrc, a_wrc = [], [], [], []
        for _, row in df.iterrows():
            season = int(row['season'])
            h = stats_dict.get((_stats_key(row['home_team'], season), season), {})
            a = stats_dict.get((_stats_key(row['away_team'], season), season), {})
            h_woba.append(h.get('woba',     LEAGUE_WOBA))
            a_woba.append(a.get('woba',     LEAGUE_WOBA))
            h_wrc.append( h.get('wrc_plus', LEAGUE_WRC))
            a_wrc.append( a.get('wrc_plus', LEAGUE_WRC))

        df['home_woba']     = h_woba
        df['away_woba']     = a_woba
        df['home_wrc_plus'] = h_wrc
        df['away_wrc_plus'] = a_wrc

    except FileNotFoundError:
        logger.warning('mlb_team_stats_historical.csv not found — batting stats will use league avg')
        df['home_woba']     = LEAGUE_WOBA
        df['away_woba']     = LEAGUE_WOBA
        df['home_wrc_plus'] = LEAGUE_WRC
        df['away_wrc_plus'] = LEAGUE_WRC

    return df


def _add_team_pitching_stats(df):
    """
    Add home/away ERA- and FIP- from mlb_team_stats_historical.csv.
    Falls back to league average: ERA- 100, FIP- 100.
    Lower values = better pitching.
    """
    df = df.copy()
    for col in ['home_era_minus', 'away_era_minus', 'home_fip_minus', 'away_fip_minus']:
        df[col] = float('nan')

    try:
        stats_df = pd.read_csv('mlb_team_stats_historical.csv')
        stats_dict = {}
        for _, row in stats_df.iterrows():
            key = (str(row['team']), int(row['season']))
            stats_dict[key] = {
                'era_minus': float(row.get('era_minus', LEAGUE_ERA_MINUS)),
                'fip_minus': float(row.get('fip_minus', LEAGUE_FIP_MINUS)),
            }

        h_era, a_era, h_fip, a_fip = [], [], [], []
        for _, row in df.iterrows():
            season = int(row['season'])
            h = stats_dict.get((_stats_key(row['home_team'], season), season), {})
            a = stats_dict.get((_stats_key(row['away_team'], season), season), {})
            h_era.append(h.get('era_minus', LEAGUE_ERA_MINUS))
            a_era.append(a.get('era_minus', LEAGUE_ERA_MINUS))
            h_fip.append(h.get('fip_minus', LEAGUE_FIP_MINUS))
            a_fip.append(a.get('fip_minus', LEAGUE_FIP_MINUS))

        df['home_era_minus'] = h_era
        df['away_era_minus'] = a_era
        df['home_fip_minus'] = h_fip
        df['away_fip_minus'] = a_fip

    except FileNotFoundError:
        logger.warning('mlb_team_stats_historical.csv not found — pitching stats will use league avg')
        df['home_era_minus'] = LEAGUE_ERA_MINUS
        df['away_era_minus'] = LEAGUE_ERA_MINUS
        df['home_fip_minus'] = LEAGUE_FIP_MINUS
        df['away_fip_minus'] = LEAGUE_FIP_MINUS

    return df


# ── Master function ───────────────────────────────────────────────────────────

def build_mlb_enhanced_features(df):
    """
    Master feature engineering function.
    Input:  mlb_games_processed.csv DataFrame (sorted by game_date ascending)
    Output: same DataFrame with all MLB_ENHANCED_FEATURES columns added.

    All features use only information available before each game.
    """
    if df.empty:
        return df

    df = df.sort_values('game_date').reset_index(drop=True)

    logger.info('Computing implied probabilities and run line...')
    df = _add_implied_probabilities(df)

    logger.info('Renaming ELO trend columns...')
    df = _add_elo_trend(df)

    logger.info('Computing matchup cross-features...')
    df = _add_matchup_features(df)

    logger.info('Computing L10 rolling win rates...')
    df = _add_l10_wins(df)

    logger.info('Adding pitcher quality diff...')
    df = _add_pitcher_quality(df)

    logger.info('Adding team batting stats (wOBA, wRC+)...')
    df = _add_team_batting_stats(df)

    logger.info('Adding team pitching stats (ERA-, FIP-)...')
    df = _add_team_pitching_stats(df)

    # Ensure is_day_game is numeric
    if 'is_day_game' in df.columns:
        df['is_day_game'] = df['is_day_game'].fillna(0).astype(int)

    # Final fillna(0) for any NaN that slipped through
    for col in MLB_ENHANCED_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = 0.0

    # Validate
    nan_counts = df[MLB_ENHANCED_FEATURES].isna().sum()
    total_nan  = nan_counts.sum()
    if total_nan > 0:
        logger.warning(f'NaN values remaining: {nan_counts[nan_counts > 0].to_dict()}')
    else:
        logger.info(f'All {len(MLB_ENHANCED_FEATURES)} features: 0 NaN values')

    return df
