"""
nhl_app.py
==========
NHL Predictor Pro — 2-tab Streamlit module.
Called from app.py as render_nhl_app().

Tabs:
  1. Game Predictor (This Week's Games / Manual Entry)
  2. Backtesting (accuracy, $10 ML sim, Kelly criterion)

Mirrors the NFL section in final_app.py.
"""

import os
import pickle
import logging
import warnings
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings('ignore')

try:
    import prediction_history as _ph
    _PH_OK = True
except ImportError:
    _PH_OK = False
logger = logging.getLogger(__name__)

# NHL lineup position helpers (from nhl_game_week.py)
try:
    from nhl_game_week import (
        get_nhl_team_depth_chart,
        get_nhl_players_for_slot,
        NHL_FWD_SLOTS, NHL_DEF_SLOTS, NHL_GOALIE_SLOTS,
    )
    _LINEUP_MODULES_OK = True
except ImportError:
    _LINEUP_MODULES_OK = False

# ── NHL team list ─────────────────────────────────────────────────────────────
NHL_TEAMS = sorted([
    'ANA', 'BOS', 'BUF', 'CGY', 'CAR', 'CHI', 'COL', 'CBJ',
    'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NSH',
    'NJD', 'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'SEA', 'SJS',
    'STL', 'TBL', 'TOR', 'UTA', 'VAN', 'VGK', 'WSH', 'WPG',
])

NHL_TEAM_NAMES = {
    'ANA': 'Anaheim Ducks',    'BOS': 'Boston Bruins',
    'BUF': 'Buffalo Sabres',   'CGY': 'Calgary Flames',
    'CAR': 'Carolina Hurricanes', 'CHI': 'Chicago Blackhawks',
    'COL': 'Colorado Avalanche',  'CBJ': 'Columbus Blue Jackets',
    'DAL': 'Dallas Stars',     'DET': 'Detroit Red Wings',
    'EDM': 'Edmonton Oilers',  'FLA': 'Florida Panthers',
    'LAK': 'Los Angeles Kings','MIN': 'Minnesota Wild',
    'MTL': 'Montreal Canadiens', 'NSH': 'Nashville Predators',
    'NJD': 'New Jersey Devils','NYI': 'New York Islanders',
    'NYR': 'New York Rangers', 'OTT': 'Ottawa Senators',
    'PHI': 'Philadelphia Flyers', 'PIT': 'Pittsburgh Penguins',
    'SEA': 'Seattle Kraken',   'SJS': 'San Jose Sharks',
    'STL': 'St. Louis Blues',  'TBL': 'Tampa Bay Lightning',
    'TOR': 'Toronto Maple Leafs', 'UTA': 'Utah Hockey Club',
    'VAN': 'Vancouver Canucks','VGK': 'Vegas Golden Knights',
    'WSH': 'Washington Capitals', 'WPG': 'Winnipeg Jets',
}

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_nhl_model():
    """Load game prediction model and ELO ratings."""
    try:
        with open("model_nhl_enhanced.pkl", "rb") as f:
            pkg = pickle.load(f)
        model    = pkg['model']
        features = pkg['features']
        accuracy = pkg.get('accuracy', 0.0)
        # Smoke test
        test_fv = pd.DataFrame([{feat: 0.0 for feat in features}])
        model.predict_proba(test_fv)
    except Exception as e:
        logger.error(f"NHL model load failed: {e}")
        model, features, accuracy = None, [], 0.0

    try:
        with open("nhl_elo_ratings.pkl", "rb") as f:
            elo = pickle.load(f)
    except Exception:
        elo = {}

    return model, features, accuracy, elo


@st.cache_resource
def load_nhl_total_model():
    """Load O/U (total goals) model."""
    try:
        with open("model_nhl_total.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


@st.cache_data
def load_nhl_games():
    """Load historical NHL game data."""
    try:
        return pd.read_csv("nhl_games_processed.csv")
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_nhl_goalie_ratings():
    """Load current season goalie team ratings."""
    try:
        df = pd.read_csv("nhl_goalie_team_ratings.csv")
        return df.set_index('team') if 'team' in df.columns else df
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_nhl_team_stats():
    """Load current NHL team xG/shot stats."""
    try:
        return pd.read_csv("nhl_team_stats_current.csv").set_index('team')
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_nhl_full_goalie_ratings():
    """Load full goalie ratings (all players, current season) for name-based lookup."""
    try:
        df = pd.read_csv("nhl_goalie_ratings.csv")
        max_season = df['season'].max()
        return df[df['season'] == max_season].copy()
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_nhl_historical_features():
    """Load and engineer features for backtesting (cached)."""
    from nhl_feature_engineering import build_nhl_enhanced_features, NHL_ENHANCED_FEATURES
    games = load_nhl_games()
    if games.empty:
        return games, []
    try:
        df_eng = build_nhl_enhanced_features(games)
        return df_eng, NHL_ENHANCED_FEATURES
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return games, []


# ── ELO helpers ───────────────────────────────────────────────────────────────

def get_nhl_elo(team: str, elo_ratings: dict) -> float:
    return elo_ratings.get(team, 1500.0)


def nhl_elo_win_prob(home_elo: float, away_elo: float, home_adv: float = 28.0) -> float:
    return 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + home_adv)) / 400.0))


# ── Depth chart cache ─────────────────────────────────────────────────────────

def _get_nhl_depth_chart(team: str, nhl_client) -> dict:
    """Fetch and cache a team's depth chart in session state."""
    cache_key = f'nhl_dc_{team}'
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    if not _LINEUP_MODULES_OK or nhl_client is None:
        return {'C': [], 'LW': [], 'RW': [], 'D': [], 'G': []}
    try:
        chart = get_nhl_team_depth_chart(team, nhl_client)
    except Exception:
        chart = {'C': [], 'LW': [], 'RW': [], 'D': [], 'G': []}
    st.session_state[cache_key] = chart
    return chart


def _lookup_goalie_score(
    goalie_name: str,
    team: str,
    goalie_ratings: pd.DataFrame,
    full_goalie_ratings: pd.DataFrame,
) -> float:
    """
    Return a goalie_score for the named goalie.
    First checks team-level ratings (starter), then full season ratings.
    Returns 0.0 if not found.
    """
    if not goalie_name or 'N/A' in goalie_name:
        return 0.0
    # Check current team starter first
    if goalie_ratings is not None and not goalie_ratings.empty:
        if team in goalie_ratings.index:
            row = goalie_ratings.loc[team]
            if str(row.get('starter_name', '')).strip() == goalie_name.strip():
                return float(row.get('goalie_score', 0.0))
    # Fall back to full ratings pool
    if full_goalie_ratings is not None and not full_goalie_ratings.empty:
        # Column is 'name' (not 'player_name') in nhl_goalie_ratings.csv
        name_col = 'name' if 'name' in full_goalie_ratings.columns else 'player_name'
        match = full_goalie_ratings[
            full_goalie_ratings[name_col].str.strip() == goalie_name.strip()
        ]
        if not match.empty:
            return float(match.iloc[0]['goalie_score'])
    return 0.0


# ── Feature vector builder ────────────────────────────────────────────────────

def build_prediction_features(
    home_team: str,
    away_team: str,
    features: list,
    elo_ratings: dict,
    goalie_ratings: pd.DataFrame,
    team_stats: pd.DataFrame,
    moneyline_home: float = None,
    puck_line: float = None,
    nhl_games: pd.DataFrame = None,
    h_goalie_name: str = None,
    a_goalie_name: str = None,
    full_goalie_ratings: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Build a feature vector for the given home/away matchup.
    Mirrors the feature vector assembly in final_app.py Tab 1.
    """
    from nhl_feature_engineering import _elo_win_prob, _moneyline_to_prob

    home_elo = get_nhl_elo(home_team, elo_ratings)
    away_elo = get_nhl_elo(away_team, elo_ratings)
    elo_diff = home_elo - away_elo

    elo_prob = nhl_elo_win_prob(home_elo, away_elo)
    ml_prob  = _moneyline_to_prob(moneyline_home) if moneyline_home else elo_prob

    if puck_line is None:
        puck_line = -1.5 if elo_diff > 0 else 1.5

    # Rolling form: look up last 5 games from nhl_games_processed.csv
    def _team_l5(team, col):
        if nhl_games is None or nhl_games.empty:
            return 3.0  # league average goals per game
        # Find last 5 games where team was home or away
        home_games = nhl_games[nhl_games['home_team'] == team].tail(5)
        away_games = nhl_games[nhl_games['away_team'] == team].tail(5)
        all_games  = pd.concat([home_games, away_games]).sort_values('gameday').tail(5)
        if all_games.empty:
            return 3.0
        gf_list, ga_list = [], []
        for _, row in all_games.iterrows():
            if row['home_team'] == team:
                gf_list.append(row.get('home_score', 3))
                ga_list.append(row.get('away_score', 3))
            else:
                gf_list.append(row.get('away_score', 3))
                ga_list.append(row.get('home_score', 3))
        if col == 'gf':
            return float(np.mean(gf_list)) if gf_list else 3.0
        return float(np.mean(ga_list)) if ga_list else 3.0

    h_gf = _team_l5(home_team, 'gf')
    h_ga = _team_l5(home_team, 'ga')
    a_gf = _team_l5(away_team, 'gf')
    a_ga = _team_l5(away_team, 'ga')

    h_diff = h_gf - h_ga
    a_diff = a_gf - a_ga

    # Goalie quality — use name override if provided, else team default
    def _goalie_score_default(team):
        if goalie_ratings is None or goalie_ratings.empty:
            return 0.0
        if team in goalie_ratings.index:
            return float(goalie_ratings.loc[team, 'goalie_score'])
        return 0.0

    if h_goalie_name:
        h_goalie = _lookup_goalie_score(h_goalie_name, home_team, goalie_ratings, full_goalie_ratings)
    else:
        h_goalie = _goalie_score_default(home_team)

    if a_goalie_name:
        a_goalie = _lookup_goalie_score(a_goalie_name, away_team, goalie_ratings, full_goalie_ratings)
    else:
        a_goalie = _goalie_score_default(away_team)

    goalie_diff = h_goalie - a_goalie

    # xG / shot stats
    def _xg_stat(team, col, default=0.5):
        if team_stats is None or team_stats.empty:
            return default
        if team in team_stats.index and col in team_stats.columns:
            return float(team_stats.loc[team, col])
        return default

    h_off_xg = _xg_stat(home_team, 'off_xg_pct')
    h_def_xg = _xg_stat(home_team, 'def_xg_pct')
    a_off_xg = _xg_stat(away_team, 'off_xg_pct')
    a_def_xg = _xg_stat(away_team, 'def_xg_pct')
    xg_total_diff = (h_off_xg - a_off_xg) + (a_def_xg - h_def_xg)

    # ELO trends (approximate from recent ELO history)
    # For live predictions, we use 0 (neutral) when we don't have per-game history
    h_elo_trend = 0.0
    a_elo_trend = 0.0

    # PP/PK% from team_stats (historical CSV — same as _add_pp_pk_features)
    LEAGUE_PP = 0.20
    LEAGUE_PK = 0.80

    def _pp_pk_stat(team, col, default):
        if team_stats is None or team_stats.empty:
            return default
        if team in team_stats.index and col in team_stats.columns:
            return float(team_stats.loc[team, col])
        return default

    h_pp = _pp_pk_stat(home_team, 'pp_pct', LEAGUE_PP)
    a_pp = _pp_pk_stat(away_team, 'pp_pct', LEAGUE_PP)
    h_pk = _pp_pk_stat(home_team, 'pk_pct', LEAGUE_PK)
    a_pk = _pp_pk_stat(away_team, 'pk_pct', LEAGUE_PK)

    # Rolling win rate from last 5 games in nhl_games
    def _team_l5_wins(team):
        if nhl_games is None or nhl_games.empty:
            return 0.5
        home_games = nhl_games[nhl_games['home_team'] == team][['gameday', 'home_win']].rename(
            columns={'home_win': 'won'})
        away_games = nhl_games[nhl_games['away_team'] == team][['gameday', 'home_win']].copy()
        away_games['won'] = 1 - away_games['home_win']
        away_games = away_games[['gameday', 'won']]
        all_g = pd.concat([home_games, away_games]).sort_values('gameday').tail(5)
        if all_g.empty:
            return 0.5
        return float(all_g['won'].mean())

    h_l5w = _team_l5_wins(home_team)
    a_l5w = _team_l5_wins(away_team)

    fv = {
        'nhl_elo_diff':          elo_diff,
        'nhl_elo_implied_prob':  elo_prob,
        'moneyline_implied_prob': ml_prob,
        'puck_line':             puck_line,
        'home_l5_goals_for':     h_gf,
        'away_l5_goals_for':     a_gf,
        'home_l5_goals_against': h_ga,
        'away_l5_goals_against': a_ga,
        'home_l5_goal_diff':     h_diff,
        'away_l5_goal_diff':     a_diff,
        'goal_diff_advantage':   h_diff - a_diff,
        'matchup_adv_home':      h_gf - a_ga,
        'matchup_adv_away':      a_gf - h_ga,
        'net_matchup_adv':       (h_gf - a_ga) - (a_gf - h_ga),
        'home_nhl_elo_trend':    h_elo_trend,
        'away_nhl_elo_trend':    a_elo_trend,
        'goalie_quality_diff':   goalie_diff,
        'home_off_xg_pct':       h_off_xg,
        'away_off_xg_pct':       a_off_xg,
        'home_def_xg_pct':       h_def_xg,
        'away_def_xg_pct':       a_def_xg,
        'xg_total_diff':         xg_total_diff,
        'home_pp_pct':           h_pp,
        'away_pp_pct':           a_pp,
        'pp_pct_diff':           h_pp - a_pp,
        'home_pk_pct':           h_pk,
        'away_pk_pct':           a_pk,
        'pk_pct_diff':           h_pk - a_pk,
        'home_l5_wins':          h_l5w,
        'away_l5_wins':          a_l5w,
        'win_pct_advantage':     h_l5w - a_l5w,
    }

    # Ensure all requested features are present
    for f in features:
        if f not in fv:
            fv[f] = 0.0

    return pd.DataFrame([{f: fv.get(f, 0.0) for f in features}])


# ── Prediction + display ──────────────────────────────────────────────────────

def run_nhl_prediction(
    home_team: str,
    away_team: str,
    model,
    features: list,
    elo_ratings: dict,
    goalie_ratings: pd.DataFrame,
    team_stats: pd.DataFrame,
    total_model_pkg: dict,
    moneyline_home: float = None,
    vegas_total: float = None,
    h_goalie_name: str = None,
    a_goalie_name: str = None,
    full_goalie_ratings: pd.DataFrame = None,
    nhl_games: pd.DataFrame = None,
) -> dict:
    """Run game prediction and return results dict."""
    if model is None:
        return {'error': 'Model not loaded'}

    fv = build_prediction_features(
        home_team, away_team, features, elo_ratings,
        goalie_ratings, team_stats,
        moneyline_home=moneyline_home,
        nhl_games=nhl_games,
        h_goalie_name=h_goalie_name,
        a_goalie_name=a_goalie_name,
        full_goalie_ratings=full_goalie_ratings,
    )

    prob_home = float(model.predict_proba(fv)[0][1])
    prob_away = 1.0 - prob_home

    # O/U prediction
    ou_pred = None
    ou_diff = None
    ou_mae  = None
    if total_model_pkg:
        ou_features = total_model_pkg.get('features', [])
        ou_fv = fv[[f for f in ou_features if f in fv.columns]].reindex(columns=ou_features, fill_value=0.0)
        league_avg = total_model_pkg.get('league_avg_total', 6.0)
        try:
            residual = total_model_pkg['model'].predict(ou_fv)[0]
            ou_pred = league_avg + residual
            ou_diff = ou_pred - (vegas_total or league_avg)
            ou_mae  = total_model_pkg.get('mae', 1.1)
        except Exception:
            pass

    # Vegas moneyline implied prob
    ml_implied = None
    if moneyline_home is not None:
        if moneyline_home > 0:
            ml_implied = 100.0 / (moneyline_home + 100.0)
        else:
            ml_implied = abs(moneyline_home) / (abs(moneyline_home) + 100.0)

    return {
        'home_team':       home_team,
        'away_team':       away_team,
        'home_win_prob':   prob_home,
        'away_win_prob':   prob_away,
        'ou_pred':         ou_pred,
        'ou_diff':         ou_diff,
        'ou_mae':          ou_mae,
        'vegas_total':     vegas_total,
        'ml_implied':      ml_implied,
        'moneyline_home':  moneyline_home,
        'elo_diff':        fv['nhl_elo_diff'].values[0] if 'nhl_elo_diff' in fv.columns else 0.0,
    }


def render_nhl_prediction_result(result: dict, prefix: str = "", game_date: str = None):
    """Display the prediction result in the Streamlit UI."""
    if 'error' in result:
        st.error(result['error'])
        return

    home = result['home_team']
    away = result['away_team']
    prob_h = result['home_win_prob']
    prob_a = result['away_win_prob']

    col1, col2 = st.columns(2)
    winner = home if prob_h > 0.5 else away
    conf   = max(prob_h, prob_a)

    with col1:
        st.metric(f"🏠 {home} Win Prob", f"{prob_h*100:.1f}%")
        st.progress(float(prob_h))

    with col2:
        st.metric(f"✈️ {away} Win Prob", f"{prob_a*100:.1f}%")
        st.progress(float(prob_a))

    # Verdict
    if conf > 0.75:
        label, _css_class = "LOCK", "signal-lock"
    elif conf > 0.65:
        label, _css_class = "HIGH CONFIDENCE", "signal-strong"
    elif conf > 0.58:
        label, _css_class = "MODERATE", "signal-lean"
    else:
        label, _css_class = "TOSS-UP", "signal-pass"
    st.markdown(
        f'<div class="signal-badge {_css_class}">{label}: {winner}</div>',
        unsafe_allow_html=True)

    # Vegas comparison
    if result.get('ml_implied') is not None:
        ml_imp = result['ml_implied']
        diff   = (prob_h - ml_imp) * 100
        if abs(diff) > 2:
            arrow = "↑" if diff > 0 else "↓"
            st.caption(f"Model: {prob_h*100:.1f}% | Vegas ML implied: {ml_imp*100:.1f}% ({arrow}{abs(diff):.1f}%)")

    # ── Kelly Bet Sizing ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📐 Kelly Bet Sizing")
    _pick_home_k  = prob_h >= 0.5
    _pick_prob_k  = prob_h if _pick_home_k else prob_a
    _pick_label_k = home if _pick_home_k else away
    _ml_home      = result.get('moneyline_home')
    if _ml_home is not None:
        _pick_ml_k = _ml_home if _pick_home_k else -_ml_home
    else:
        _pick_ml_k = -110  # fallback: standard line
    _strategy_k   = st.session_state.get('nhl_bet_strategy', 'Kelly Criterion')
    _risk_tol_k   = st.session_state.get('nhl_risk_tolerance', 'Moderate')
    _kelly_frac_k = {'Conservative': 0.25, 'Moderate': 0.5, 'Aggressive': 1.0}[_risk_tol_k]
    _fixed_pct_k  = float(st.session_state.get('nhl_fixed_pct', 2.0))
    _fixed_dol_k  = int(st.session_state.get('nhl_fixed_dollar', 50))
    def _nhl_kelly(p, ml, frac=0.5):
        try:
            b = (100.0 / abs(ml)) if ml < 0 else (ml / 100.0)
            full_k = (b * p - (1.0 - p)) / b
            pct = max(0.0, min(full_k * frac, 0.10)) * 100
        except Exception:
            return 0.0, 'signal-pass', 'PASS'
        if pct >= 4.0: return pct, 'signal-strong', 'STRONG EDGE'
        if pct >= 2.0: return pct, 'signal-lean',   'LEAN'
        if pct >= 1.0: return pct, 'signal-lean',   'SMALL EDGE'
        return pct, 'signal-pass', 'PASS'
    _kpct_k, _ktier_k, _kbadge_k = _nhl_kelly(_pick_prob_k, _pick_ml_k, frac=_kelly_frac_k)
    _bankroll_val = int(st.session_state.get('nhl_bankroll', 1000))
    if _pick_ml_k < 0:
        _vegas_impl_k = abs(_pick_ml_k) / (abs(_pick_ml_k) + 100)
    else:
        _vegas_impl_k = 100 / (_pick_ml_k + 100)
    _edge_k = (_pick_prob_k - _vegas_impl_k) * 100
    # Bet amount + label based on strategy
    if _strategy_k == 'Fixed %':
        _bet_amt_k   = _bankroll_val * _fixed_pct_k / 100
        _pct_label_k = "Fixed %"
        _pct_val_k   = f"{_fixed_pct_k:.1f}%"
    elif _strategy_k == 'Fixed $':
        _bet_amt_k   = float(_fixed_dol_k)
        _pct_label_k = "Fixed $"
        _pct_val_k   = f"${_fixed_dol_k}"
    else:  # Kelly Criterion or Fractional Kelly
        _bet_amt_k   = _bankroll_val * _kpct_k / 100
        _pct_label_k = "Kelly %"
        _pct_val_k   = f"{_kpct_k:.1f}%"
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Model Edge",  f"{_edge_k:+.1f}%",
               help="Model win prob minus Vegas implied prob for the predicted winner")
    kc2.metric(_pct_label_k, _pct_val_k,
               help="Bet size based on selected strategy — adjust in sidebar")
    kc3.metric("Bet Amount",  f"${_bet_amt_k:.0f}",
               help=f"Of your ${_bankroll_val:,} bankroll — adjust bankroll in sidebar")
    kc4.markdown(
        f'<p style="font-size:0.8em;color:gray;margin-bottom:4px">Signal</p>'
        f'<span class="signal-badge {_ktier_k}">{_kbadge_k}</span>',
        unsafe_allow_html=True)
    _ml_display = f"{_pick_ml_k:+.0f}" if _ml_home is not None else "-110 (est.)"
    if _strategy_k == 'Fixed %':
        _caption_extra = f"Fixed {_fixed_pct_k:.1f}% per game regardless of edge."
    elif _strategy_k == 'Fixed $':
        _caption_extra = f"Fixed ${_fixed_dol_k} per game regardless of edge."
    else:
        _caption_extra = f"{_risk_tol_k} Kelly ({_kelly_frac_k:.2g}×). Kelly caps at 10% of bankroll to limit volatility."
    st.caption(
        f"Betting on **{_pick_label_k}** at {_pick_prob_k*100:.1f}% model confidence. "
        f"Moneyline: {_ml_display}. Vegas implied: {_vegas_impl_k*100:.1f}%. {_caption_extra}"
    )

    # ── Log prediction to history (once per session per game) ────────────────
    if _PH_OK:
        _log_key = f'{prefix}_logged' if prefix else f'nhl_{home}_{away}_logged'
        if not st.session_state.get(_log_key, False):
            try:
                _gdate = game_date or date.today().isoformat()
                _ou_pred_log = result.get('ou_pred')
                _vegas_total_log = result.get('vegas_total')
                _ou_diff_log = result.get('ou_diff', 0)
                _rec = {
                    'id': f"nhl_{home}_{away}_{_gdate}",
                    'sport': 'nhl',
                    'game_date': _gdate,
                    'home_team': home,
                    'away_team': away,
                    'predicted_at': datetime.now().isoformat(),
                    'model_home_prob': round(float(prob_h), 4),
                    'vegas_ml_home': result.get('moneyline_home'),
                    'vegas_implied_prob': round(float(_vegas_impl_k), 4),
                    'model_edge_pct': round(float(_edge_k), 2),
                    'kelly_signal': _ktier_k,
                    'kelly_pct': round(float(_kpct_k), 2),
                    'ou_line': float(_vegas_total_log) if _vegas_total_log is not None else None,
                    'model_total': float(_ou_pred_log) if _ou_pred_log is not None else None,
                    'ou_lean': ('OVER' if _ou_diff_log > 0 else 'UNDER') if _ou_pred_log is not None else None,
                    'actual_winner': None,
                    'actual_score_home': None,
                    'actual_score_away': None,
                    'actual_total': None,
                    'prediction_correct': None,
                    'ou_correct': None,
                }
                _ph.log_prediction(_rec)
                st.session_state[_log_key] = True
            except Exception:
                pass

    # O/U prediction
    ou_pred = result.get('ou_pred')
    if ou_pred is not None:
        st.divider()
        vegas_total = result.get('vegas_total')
        ou_mae      = result.get('ou_mae', 1.1)
        ou_diff     = result.get('ou_diff', 0)

        st.markdown("**Over/Under Prediction**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Model Total", f"{ou_pred:.1f} goals")
        if vegas_total:
            c2.metric("Vegas O/U", f"{vegas_total:.1f}")
            lean = "OVER" if ou_diff > 0 else "UNDER"
            c3.metric("Lean", lean, delta=f"{abs(ou_diff):.1f} goals")
        c1.caption(f"MAE ±{ou_mae:.1f} goals")


# ── Game expander (weekly schedule) ───────────────────────────────────────────

def _render_nhl_game_expander(
    game: dict,
    idx: int,
    model,
    features: list,
    elo_ratings: dict,
    goalie_ratings: pd.DataFrame,
    team_stats: pd.DataFrame,
    total_model_pkg: dict,
    nhl_games: pd.DataFrame,
    nhl_client=None,
    full_goalie_ratings: pd.DataFrame = None,
):
    """Render a single NHL game as a Streamlit expander with full lineup cards."""
    home = game['home_team']
    away = game['away_team']
    home_name = NHL_TEAM_NAMES.get(home, home)
    away_name  = NHL_TEAM_NAMES.get(away, away)

    label = f"{away_name} @ {home_name}  |  {game.get('game_time_et','TBD')}"
    if game.get('is_outdoor'):
        label += "  🌨️ OUTDOOR"

    pfx = f"nhl_g{idx}"

    # Extract ISO game date for prediction logging
    _dt_et = game.get('datetime_et')
    try:
        _game_date = _dt_et.date().isoformat() if _dt_et else date.today().isoformat()
    except Exception:
        _game_date = date.today().isoformat()

    # Append pre-calculated prediction badge to collapsed label
    _pre = st.session_state.get(f'{pfx}_pred')
    if _pre and 'error' not in _pre:
        _h_prob = _pre.get('home_win_prob', 0.5)
        _winner = home if _h_prob >= 0.5 else away
        _conf   = max(_h_prob, 1.0 - _h_prob)
        _emoji  = "🔥" if _conf >= 0.65 else ("✅" if _conf >= 0.58 else "⚪")
        label  += f"  |  {_emoji} {_winner} {_conf*100:.0f}%"

    with st.expander(label, expanded=st.session_state.get(f'{pfx}_expanded', False)):

        # ── Vegas Lines ──────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            moneyline_home = st.number_input(
                "Home ML (American)", value=None, format="%.0f",
                key=f"{pfx}_ml_home", placeholder="e.g. -150"
            )
        with c2:
            vegas_total = st.number_input(
                "Vegas O/U (total goals)", value=None, format="%.1f",
                key=f"{pfx}_ou_total", placeholder="e.g. 6.0"
            )
        with c3:
            if game.get('is_outdoor'):
                st.slider("Temp (°F)", 0, 50, 32, key=f"{pfx}_temp")
                st.slider("Wind (mph)", 0, 30, 5, key=f"{pfx}_wind")

        # ── Starting Lineups ─────────────────────────────────────────────────
        st.divider()
        st.markdown("##### Starting Lineups")

        # Fetch depth charts
        home_dc = _get_nhl_depth_chart(home, nhl_client) if _LINEUP_MODULES_OK and nhl_client else {'C': [], 'LW': [], 'RW': [], 'D': [], 'G': []}
        away_dc = _get_nhl_depth_chart(away, nhl_client) if _LINEUP_MODULES_OK and nhl_client else {'C': [], 'LW': [], 'RW': [], 'D': [], 'G': []}

        def _slot_players(dc, pos_key, slot_depth):
            if _LINEUP_MODULES_OK:
                return get_nhl_players_for_slot(dc, pos_key, slot_depth)
            return [f'{pos_key}{slot_depth} (N/A)']

        # Header row
        hdr_h, hdr_v, hdr_a = st.columns([5, 1, 5])
        hdr_h.markdown(f"**{home_name}**")
        hdr_v.markdown("")
        hdr_a.markdown(f"**{away_name}**")

        h_fwd_sel = {}
        a_fwd_sel = {}

        # Forwards
        st.markdown("**Forwards**")
        prev_line = None
        for lbl, pos_key, depth in NHL_FWD_SLOTS if _LINEUP_MODULES_OK else []:
            # Draw a subtle separator between lines
            line_num = depth
            if prev_line and line_num != prev_line:
                st.caption("― Line 2 ―")
            prev_line = line_num
            col_h, col_v, col_a = st.columns([5, 1, 5])
            with col_h:
                opts_h = _slot_players(home_dc, pos_key, depth)
                h_fwd_sel[lbl] = st.selectbox(
                    lbl, opts_h, index=0,
                    key=f"{pfx}_h_fwd_{lbl}", label_visibility='collapsed'
                )
            with col_v:
                st.markdown(f"<div style='text-align:center;padding-top:8px'><small>{lbl}</small></div>",
                            unsafe_allow_html=True)
            with col_a:
                opts_a = _slot_players(away_dc, pos_key, depth)
                a_fwd_sel[lbl] = st.selectbox(
                    lbl, opts_a, index=0,
                    key=f"{pfx}_a_fwd_{lbl}", label_visibility='collapsed'
                )

        # Defense
        st.markdown("**Defense**")
        h_def_sel = {}
        a_def_sel = {}
        prev_pair = None
        for lbl, pos_key, depth in NHL_DEF_SLOTS if _LINEUP_MODULES_OK else []:
            pair_num = (depth + 1) // 2
            if prev_pair and pair_num != prev_pair:
                st.caption("― Pair 2 ―")
            prev_pair = pair_num
            col_h, col_v, col_a = st.columns([5, 1, 5])
            with col_h:
                opts_h = _slot_players(home_dc, pos_key, depth)
                h_def_sel[lbl] = st.selectbox(
                    lbl, opts_h, index=0,
                    key=f"{pfx}_h_def_{lbl}", label_visibility='collapsed'
                )
            with col_v:
                st.markdown(f"<div style='text-align:center;padding-top:8px'><small>{lbl}</small></div>",
                            unsafe_allow_html=True)
            with col_a:
                opts_a = _slot_players(away_dc, pos_key, depth)
                a_def_sel[lbl] = st.selectbox(
                    lbl, opts_a, index=0,
                    key=f"{pfx}_a_def_{lbl}", label_visibility='collapsed'
                )

        # Goalie row
        st.markdown("**Goalie**")
        def _goalie_display(team, goalie_name):
            if goalie_ratings is not None and not goalie_ratings.empty and team in goalie_ratings.index:
                row = goalie_ratings.loc[team]
                if str(row.get('starter_name', '')).strip() == goalie_name.strip():
                    return f"{goalie_name}  ·  SV% {row.get('sv_pct',0):.3f}  ·  GAA {row.get('gaa',0):.2f}  ·  Score {row.get('goalie_score',0):.2f}"
            return goalie_name

        g_col_h, g_col_v, g_col_a = st.columns([5, 1, 5])
        with g_col_h:
            h_goalie_opts = _slot_players(home_dc, 'G', 1)
            h_goalie_sel = st.selectbox(
                "G", h_goalie_opts, index=0,
                key=f"{pfx}_h_goalie", label_visibility='collapsed'
            )
            st.caption(_goalie_display(home, h_goalie_sel))
        with g_col_v:
            st.markdown("<div style='text-align:center;padding-top:8px'><small>G</small></div>",
                        unsafe_allow_html=True)
        with g_col_a:
            a_goalie_opts = _slot_players(away_dc, 'G', 1)
            a_goalie_sel = st.selectbox(
                "G", a_goalie_opts, index=0,
                key=f"{pfx}_a_goalie", label_visibility='collapsed'
            )
            st.caption(_goalie_display(away, a_goalie_sel))

        # ── Predict button ────────────────────────────────────────────────────
        st.divider()
        pred_key   = f"{pfx}_pred"
        fprint_key = f"{pfx}_pred_fprint"

        # Fingerprint of inputs that affect the prediction (goalie + Vegas lines)
        current_fprint = (home, away,
                          h_goalie_sel if _LINEUP_MODULES_OK else None,
                          a_goalie_sel if _LINEUP_MODULES_OK else None,
                          moneyline_home, vegas_total)

        if st.button(f"🏒 Predict {away} @ {home}", key=f"{pfx}_predict", type="primary",
                     use_container_width=True):
            result = run_nhl_prediction(
                home_team=home, away_team=away,
                model=model, features=features,
                elo_ratings=elo_ratings,
                goalie_ratings=goalie_ratings,
                team_stats=team_stats,
                total_model_pkg=total_model_pkg,
                moneyline_home=moneyline_home,
                vegas_total=vegas_total,
                nhl_games=nhl_games,
                h_goalie_name=h_goalie_sel if _LINEUP_MODULES_OK else None,
                a_goalie_name=a_goalie_sel if _LINEUP_MODULES_OK else None,
                full_goalie_ratings=full_goalie_ratings,
            )
            st.session_state[pred_key]   = result
            st.session_state[fprint_key] = current_fprint
            st.rerun()

        if pred_key in st.session_state:
            if st.session_state.get(fprint_key) != current_fprint:
                st.caption("⚠️ Goalie or Vegas lines changed — click Predict to update")
            render_nhl_prediction_result(st.session_state[pred_key], prefix=pfx, game_date=_game_date)


# ── Manual entry tab ──────────────────────────────────────────────────────────

def _render_nhl_manual_entry(
    model, features, elo_ratings, goalie_ratings, team_stats,
    total_model_pkg, nhl_games, full_goalie_ratings=None, nhl_client=None,
):
    """Manual entry form for NHL game prediction."""
    st.subheader("Game Setup")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**🏠 Home Team**")
        home = st.selectbox("Home", NHL_TEAMS, index=NHL_TEAMS.index('BOS'), key='nhl_home')

    with c2:
        st.markdown("**✈️ Away Team**")
        away = st.selectbox("Away", NHL_TEAMS, index=NHL_TEAMS.index('TOR'), key='nhl_away')

    with c3:
        st.markdown("**📊 Vegas Lines**")
        moneyline_home = st.number_input(
            "Home Moneyline (American)", value=None, format="%.0f",
            placeholder="e.g. -150 or +130", key='nhl_ml'
        )
        vegas_total = st.number_input(
            "O/U Total Goals", value=None, format="%.1f",
            placeholder="e.g. 5.5 or 6.0", key='nhl_ou'
        )

    if home == away:
        st.warning("Home and away teams must be different.")
        return

    # ── Starting Lineups ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("##### Starting Lineups")

    home_name = NHL_TEAM_NAMES.get(home, home)
    away_name  = NHL_TEAM_NAMES.get(away, away)

    home_dc = _get_nhl_depth_chart(home, nhl_client) if _LINEUP_MODULES_OK and nhl_client else {'C': [], 'LW': [], 'RW': [], 'D': [], 'G': []}
    away_dc = _get_nhl_depth_chart(away, nhl_client) if _LINEUP_MODULES_OK and nhl_client else {'C': [], 'LW': [], 'RW': [], 'D': [], 'G': []}

    def _slot_players_m(dc, pos_key, slot_depth):
        if _LINEUP_MODULES_OK:
            return get_nhl_players_for_slot(dc, pos_key, slot_depth)
        return [f'{pos_key}{slot_depth} (N/A)']

    # Header
    hdr_h, hdr_v, hdr_a = st.columns([5, 1, 5])
    hdr_h.markdown(f"**{home_name}**")
    hdr_v.markdown("")
    hdr_a.markdown(f"**{away_name}**")

    # Forwards
    st.markdown("**Forwards**")
    h_fwd_sel_m = {}
    a_fwd_sel_m = {}
    prev_line_m = None
    for lbl, pos_key, depth in (NHL_FWD_SLOTS if _LINEUP_MODULES_OK else []):
        if prev_line_m and depth != prev_line_m:
            st.caption("― Line 2 ―")
        prev_line_m = depth
        col_h, col_v, col_a = st.columns([5, 1, 5])
        with col_h:
            h_fwd_sel_m[lbl] = st.selectbox(
                lbl, _slot_players_m(home_dc, pos_key, depth), index=0,
                key=f"nhl_m_h_fwd_{lbl}", label_visibility='collapsed'
            )
        with col_v:
            st.markdown(f"<div style='text-align:center;padding-top:8px'><small>{lbl}</small></div>",
                        unsafe_allow_html=True)
        with col_a:
            a_fwd_sel_m[lbl] = st.selectbox(
                lbl, _slot_players_m(away_dc, pos_key, depth), index=0,
                key=f"nhl_m_a_fwd_{lbl}", label_visibility='collapsed'
            )

    # Defense
    st.markdown("**Defense**")
    h_def_sel_m = {}
    a_def_sel_m = {}
    prev_pair_m = None
    for lbl, pos_key, depth in (NHL_DEF_SLOTS if _LINEUP_MODULES_OK else []):
        pair_num = (depth + 1) // 2
        if prev_pair_m and pair_num != prev_pair_m:
            st.caption("― Pair 2 ―")
        prev_pair_m = pair_num
        col_h, col_v, col_a = st.columns([5, 1, 5])
        with col_h:
            h_def_sel_m[lbl] = st.selectbox(
                lbl, _slot_players_m(home_dc, pos_key, depth), index=0,
                key=f"nhl_m_h_def_{lbl}", label_visibility='collapsed'
            )
        with col_v:
            st.markdown(f"<div style='text-align:center;padding-top:8px'><small>{lbl}</small></div>",
                        unsafe_allow_html=True)
        with col_a:
            a_def_sel_m[lbl] = st.selectbox(
                lbl, _slot_players_m(away_dc, pos_key, depth), index=0,
                key=f"nhl_m_a_def_{lbl}", label_visibility='collapsed'
            )

    # Goalie
    st.markdown("**Goalie**")
    def _goalie_display_m(team, goalie_name):
        if goalie_ratings is not None and not goalie_ratings.empty and team in goalie_ratings.index:
            row = goalie_ratings.loc[team]
            if str(row.get('starter_name', '')).strip() == goalie_name.strip():
                return f"{goalie_name}  ·  SV% {row.get('sv_pct',0):.3f}  ·  GAA {row.get('gaa',0):.2f}  ·  Score {row.get('goalie_score',0):.2f}"
        return goalie_name

    g_col_h, g_col_v, g_col_a = st.columns([5, 1, 5])
    with g_col_h:
        h_g_opts = _slot_players_m(home_dc, 'G', 1)
        h_goalie_sel_m = st.selectbox(
            "G", h_g_opts, index=0,
            key="nhl_m_h_goalie", label_visibility='collapsed'
        )
        st.caption(_goalie_display_m(home, h_g_opts[0] if h_g_opts else ''))
    with g_col_v:
        st.markdown("<div style='text-align:center;padding-top:8px'><small>G</small></div>",
                    unsafe_allow_html=True)
    with g_col_a:
        a_g_opts = _slot_players_m(away_dc, 'G', 1)
        a_goalie_sel_m = st.selectbox(
            "G", a_g_opts, index=0,
            key="nhl_m_a_goalie", label_visibility='collapsed'
        )
        st.caption(_goalie_display_m(away, a_g_opts[0] if a_g_opts else ''))

    # ── Predict ───────────────────────────────────────────────────────────────
    st.divider()
    if st.button("🏒 PREDICT", type="primary", use_container_width=True, key='nhl_predict_btn'):
        result = run_nhl_prediction(
            home_team=home, away_team=away,
            model=model, features=features,
            elo_ratings=elo_ratings,
            goalie_ratings=goalie_ratings,
            team_stats=team_stats,
            total_model_pkg=total_model_pkg,
            moneyline_home=moneyline_home,
            vegas_total=vegas_total,
            nhl_games=nhl_games,
            h_goalie_name=h_goalie_sel_m if _LINEUP_MODULES_OK else None,
            a_goalie_name=a_goalie_sel_m if _LINEUP_MODULES_OK else None,
            full_goalie_ratings=full_goalie_ratings,
        )
        st.session_state['nhl_manual_result'] = result
        st.rerun()

    if 'nhl_manual_result' in st.session_state:
        st.divider()
        render_nhl_prediction_result(st.session_state['nhl_manual_result'],
                                      game_date=date.today().isoformat())

        # ELO sidebar info
        home_elo = get_nhl_elo(home, elo_ratings)
        away_elo = get_nhl_elo(away, elo_ratings)
        with st.expander("📊 Prediction Details"):
            d1, d2 = st.columns(2)
            d1.metric(f"{home} ELO", f"{home_elo:.0f}")
            d2.metric(f"{away} ELO", f"{away_elo:.0f}")


# ── Weekly schedule tab ────────────────────────────────────────────────────────

def _render_nhl_weekly_schedule(
    model, features, elo_ratings, goalie_ratings, team_stats,
    total_model_pkg, nhl_games, full_goalie_ratings=None,
):
    """Fetch and render this week's NHL schedule."""
    try:
        from apis.nhl import NHLClient
        from nhl_game_week import fetch_nhl_weekly_schedule
    except ImportError as e:
        st.error(f"Could not load NHL API: {e}")
        return

    if st.button("🔄 Refresh Schedule", key='nhl_refresh_sched'):
        # Clear schedule, depth charts, and pre-calc cache
        for k in list(st.session_state.keys()):
            if (k.startswith('nhl_weekly_schedule') or k.startswith('nhl_dc_')
                    or k == 'nhl_precalc_done' or k == 'nhl_total_games'
                    or (k.startswith('nhl_g') and ('_pred' in k or '_expanded' in k))):
                del st.session_state[k]
        st.rerun()

    # Create a single NHLClient instance for all depth chart lookups
    client_key = 'nhl_client_instance'
    if client_key not in st.session_state:
        st.session_state[client_key] = NHLClient()
    nhl_client = st.session_state[client_key]

    schedule_key = 'nhl_weekly_schedule'
    if schedule_key not in st.session_state:
        with st.spinner("Fetching this week's NHL schedule..."):
            st.session_state[schedule_key] = fetch_nhl_weekly_schedule(nhl_client)

    schedule = st.session_state.get(schedule_key, {})

    if not schedule:
        st.info("No NHL games found for this week. Try refreshing or use Manual Entry.")
        return

    total_games = sum(len(v) for v in schedule.values())
    st.session_state['nhl_total_games'] = total_games

    # ── Pre-calculate all predictions (once per schedule load) ────────────────
    if 'nhl_precalc_done' not in st.session_state and model is not None:
        with st.spinner(f"Pre-calculating predictions for {total_games} games..."):
            _idx = 0
            for _day, _games in schedule.items():
                for _game in _games:
                    _pred_key = f"nhl_g{_idx}_pred"
                    if _pred_key not in st.session_state:
                        try:
                            _r = run_nhl_prediction(
                                _game['home_team'], _game['away_team'],
                                model, features, elo_ratings, goalie_ratings,
                                team_stats, total_model_pkg,
                                nhl_games=nhl_games,
                                full_goalie_ratings=full_goalie_ratings,
                            )
                            if _r and 'error' not in _r:
                                st.session_state[_pred_key] = _r
                        except Exception:
                            pass
                    _idx += 1
        st.session_state['nhl_precalc_done'] = True

    st.caption(f"Showing {total_games} games  ·  Predictions pre-calculated  ·  Expand a card for lineups")

    # ── Expand All / Collapse All ─────────────────────────────────────────────
    ec1, _, ec2 = st.columns([2, 6, 2])
    with ec1:
        if st.button("⬇ Expand All", key='nhl_expand_all', use_container_width=True):
            for i in range(total_games):
                st.session_state[f'nhl_g{i}_expanded'] = True
            st.rerun()
    with ec2:
        if st.button("⬆ Collapse All", key='nhl_collapse_all', use_container_width=True):
            for i in range(total_games):
                st.session_state[f'nhl_g{i}_expanded'] = False
            st.rerun()

    game_idx = 0
    for day, games in schedule.items():
        if not games:
            continue
        # day key is "DayName Mon DD" (e.g. "Saturday Mar 01") — split for display
        day_parts = day.split(' ', 1)
        day_display = day_parts[0]
        date_lbl = day_parts[1] if len(day_parts) > 1 else games[0].get('game_date_label', '')
        st.markdown(f"### {day_display}  <small style='color:gray'>({date_lbl})</small>",
                    unsafe_allow_html=True)
        for game in games:
            _render_nhl_game_expander(
                game, game_idx,
                model, features, elo_ratings, goalie_ratings,
                team_stats, total_model_pkg, nhl_games,
                nhl_client=nhl_client,
                full_goalie_ratings=full_goalie_ratings,
            )
            game_idx += 1


# ── Tab 1: Game Predictor ─────────────────────────────────────────────────────

def _render_tab1(model, features, accuracy, elo_ratings, goalie_ratings, team_stats, total_model_pkg, nhl_games, full_goalie_ratings=None):
    """NHL Game Predictor tab."""
    if model is None:
        st.error("NHL model not loaded. Run `python build_nhl_model.py` first.")
        with st.expander("Quick setup instructions"):
            st.code("""
# From the nfl_predictor directory:
python build_nhl_games.py          # ~20 min (fetches 25 seasons)
python build_nhl_goalie_ratings.py # ~5 min
python build_nhl_team_stats.py     # ~5 min
python build_nhl_model.py          # ~10 min (trains ensemble)
""")
        return

    # Mode selector
    mode = st.radio(
        "Input mode",
        ["This Week's Games", "Manual Entry"],
        horizontal=True,
        key='nhl_input_mode',
    )

    st.divider()

    if mode == "This Week's Games":
        _render_nhl_weekly_schedule(
            model, features, elo_ratings, goalie_ratings,
            team_stats, total_model_pkg, nhl_games,
            full_goalie_ratings=full_goalie_ratings,
        )
    else:
        # Create NHLClient for manual entry depth chart lookups
        nhl_client_m = None
        if _LINEUP_MODULES_OK:
            try:
                from apis.nhl import NHLClient
                client_key = 'nhl_client_instance'
                if client_key not in st.session_state:
                    st.session_state[client_key] = NHLClient()
                nhl_client_m = st.session_state[client_key]
            except Exception:
                pass
        _render_nhl_manual_entry(
            model, features, elo_ratings, goalie_ratings,
            team_stats, total_model_pkg, nhl_games,
            full_goalie_ratings=full_goalie_ratings,
            nhl_client=nhl_client_m,
        )


# ── Tab 2: Backtesting ────────────────────────────────────────────────────────

def _ml_pnl(odds: float, won: bool, stake: float = 10.0) -> float:
    """Calculate P&L for a moneyline bet."""
    if won:
        if odds > 0:
            return stake * odds / 100.0
        else:
            return stake * 100.0 / abs(odds)
    return -stake


def _implied_prob(odds: float) -> float:
    if odds is None:
        return 0.5
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _prob_to_ml(p: float, vig: float = 0.0455) -> float:
    """Convert win probability to American moneyline with standard vig applied.

    A 60% model probability → vigged ~62.3% → roughly -165 ML.
    A 35% model probability → vigged ~37.3% underdog → roughly +168 ML.
    """
    vigged_p = min(max(p + vig / 2, 0.03), 0.97)
    if vigged_p >= 0.5:
        return -(vigged_p / (1 - vigged_p)) * 100
    return (1 - vigged_p) / vigged_p * 100


def _render_tab2(model, features, accuracy):
    """NHL Backtesting tab."""
    st.subheader("Model Backtesting")

    if model is None:
        st.error("NHL model not loaded. Run `python build_nhl_model.py` first.")
        return

    with st.spinner("Loading historical data and computing features..."):
        df_eng, feat_list = load_nhl_historical_features()

    if df_eng.empty:
        st.error("No NHL historical data found. Run `python build_nhl_games.py` first.")
        return

    # Use last 5 seasons for backtesting
    max_season = df_eng['season'].max()
    min_bt_season = max_season - 4
    data5 = df_eng[df_eng['season'] >= min_bt_season].copy()

    if feat_list and model:
        available_feats = [f for f in feat_list if f in data5.columns]
        X = data5[available_feats].fillna(0.0)
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        data5 = data5.copy()
        data5['pred'] = preds
        data5['prob'] = probs
        data5['correct'] = (data5['pred'] == data5['home_win']).astype(int)

        # ── Overall Accuracy ─────────────────────────────────────────────────
        overall_acc = data5['correct'].mean()
        baseline    = data5['home_win'].mean()

        st.markdown("### Overall Accuracy")
        c1, c2, c3 = st.columns(3)
        c1.metric("Model Accuracy", f"{overall_acc*100:.1f}%", delta=f"+{(overall_acc-baseline)*100:.1f}% vs baseline")
        c2.metric("Home Win Rate (baseline)", f"{baseline*100:.1f}%")
        c3.metric("Seasons analyzed", f"{min_bt_season}-{max_season}")
        if accuracy:
            st.caption(f"Holdout accuracy (at training time): {accuracy*100:.1f}%")

        # ── By Season ────────────────────────────────────────────────────────
        st.markdown("### Accuracy by Season")
        season_acc = data5.groupby('season').agg(
            accuracy=('correct', 'mean'),
            games=('correct', 'count'),
            home_win_rate=('home_win', 'mean'),
        ).reset_index()
        season_acc['season_label'] = season_acc['season'].astype(str) + '-' + (season_acc['season']+1).astype(str)
        season_acc['accuracy_pct'] = (season_acc['accuracy'] * 100).round(1)
        season_acc['baseline_pct'] = (season_acc['home_win_rate'] * 100).round(1)
        st.dataframe(
            season_acc[['season_label', 'accuracy_pct', 'baseline_pct', 'games']].rename(columns={
                'season_label': 'Season', 'accuracy_pct': 'Model Acc %',
                'baseline_pct': 'Baseline %', 'games': 'Games',
            }),
            use_container_width=True,
            hide_index=True,
        )

        # ── Game-by-Game Results ──────────────────────────────────────────────
        st.markdown("### Game-by-Game Results")
        season_options = sorted(data5['season'].unique(), reverse=True)
        selected_season = st.selectbox(
            "Select season",
            season_options,
            format_func=lambda s: f"{s}-{s+1}",
            key='nhl_bt_season',
        )
        season_data = data5[data5['season'] == selected_season].copy()
        season_data = season_data.sort_values('gameday')
        season_data['predicted_winner'] = season_data.apply(
            lambda r: r['home_team'] if r['pred'] == 1 else r['away_team'], axis=1
        )
        season_data['actual_winner'] = season_data.apply(
            lambda r: r['home_team'] if r['home_win'] == 1 else r['away_team'], axis=1
        )
        season_data['confidence'] = (season_data['prob'].apply(lambda p: max(p, 1-p)) * 100).round(1)
        season_data['result'] = season_data['correct'].apply(lambda c: '✅' if c else '❌')

        display_cols = ['gameday', 'home_team', 'home_score', 'away_score', 'away_team',
                        'predicted_winner', 'confidence', 'result']
        available_display = [c for c in display_cols if c in season_data.columns]
        st.dataframe(
            season_data[available_display].rename(columns={
                'gameday': 'Date', 'home_team': 'Home', 'home_score': 'H Sc',
                'away_score': 'A Sc', 'away_team': 'Away',
                'predicted_winner': 'Predicted', 'confidence': 'Conf %', 'result': '✓',
            }),
            use_container_width=True,
            hide_index=True,
        )

        # ── $10 Moneyline Simulation ──────────────────────────────────────────
        st.markdown("### $10 Moneyline Simulation")
        st.caption(
            "Simulates betting $10 on each game at standard -110 odds. "
            "Historical NHL moneylines aren't stored, so -110 is used as a flat proxy "
            "(break-even = 52.4% accuracy)."
        )

        FLAT_ODDS = -110
        data5_bt = data5.copy()

        model_pnl  = 0.0
        home_pnl   = 0.0
        model_pnls = []
        home_pnls  = []

        for _, row in data5_bt.iterrows():
            model_won = bool(row['correct'])
            model_pnl += _ml_pnl(FLAT_ODDS, model_won)
            model_pnls.append(model_pnl)

            home_won = bool(row['home_win'] == 1)
            home_pnl += _ml_pnl(FLAT_ODDS, home_won)
            home_pnls.append(home_pnl)

        n_games = len(data5_bt)
        total_wagered = n_games * 10.0

        bt_c1, bt_c2, bt_c3 = st.columns(3)
        bt_c1.metric("Model Net P&L", f"${model_pnl:.2f}",
                     delta=f"{model_pnl/total_wagered*100:.1f}% ROI")
        bt_c2.metric("Always-Home Net P&L", f"${home_pnl:.2f}",
                     delta=f"{home_pnl/total_wagered*100:.1f}% ROI")
        bt_c3.metric("Games Simulated", f"{n_games:,}")
        st.caption(f"Total wagered: ${total_wagered:,.0f} | Odds: {FLAT_ODDS} flat (historical lines not stored)")

        # Cumulative P&L chart
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=model_pnls,  name='Model',      line=dict(color='#2ecc71', width=2)))
            fig.add_trace(go.Scatter(y=home_pnls,   name='Always Home', line=dict(color='#3498db', width=1.5, dash='dot')))
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(
                title=f"Cumulative P&L: {min_bt_season}-{max_season} Seasons",
                xaxis_title="Game",
                yaxis_title="Net P&L ($)",
                height=350, margin=dict(l=40, r=20, t=40, b=40),
                legend=dict(orientation='h', y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

        # ── Kelly Criterion ───────────────────────────────────────────────────
        st.markdown("### Kelly Criterion Simulation")

        bankroll_start = int(st.session_state.get('nhl_bankroll', 1000))
        with st.sidebar:
            st.markdown("---")
            st.markdown("**NHL Kelly Settings**")
            kelly_frac = st.select_slider(
                "Kelly Fraction",
                options=[0.25, 0.5, 1.0],
                value=0.5,
                format_func=lambda x: {0.25: 'Quarter-Kelly', 0.5: 'Half-Kelly', 1.0: 'Full Kelly'}[x],
                key='nhl_kelly_frac',
            )

        bankroll = float(bankroll_start)
        flat_bankroll = float(bankroll_start)
        flat_stake = bankroll_start * 0.005  # 0.5% flat bet

        kelly_history = [100.0]
        flat_history  = [100.0]

        KELLY_ODDS = -110
        b = 100.0 / 110.0  # net profit per $1 at -110 (10/11)

        for _, row in data5_bt.iterrows():
            p = float(row['prob']) if float(row['prob']) >= 0.5 else (1.0 - float(row['prob']))
            # Kelly formula: f* = (b*p - (1-p)) / b
            kelly_pct = max(0, (b * p - (1.0 - p)) / b) * kelly_frac
            kelly_pct = min(kelly_pct, 0.005)  # cap at 0.5% per game
            bet = bankroll * kelly_pct
            won = bool(row['correct'])
            pnl = _ml_pnl(KELLY_ODDS, won, stake=bet)
            bankroll += pnl
            bankroll  = max(bankroll, 0.01)

            flat_pnl = _ml_pnl(KELLY_ODDS, won, stake=flat_stake)
            flat_bankroll += flat_pnl
            flat_bankroll  = max(flat_bankroll, 0.01)

            kelly_history.append(bankroll / bankroll_start * 100)
            flat_history.append(flat_bankroll / bankroll_start * 100)

        k_c1, k_c2, k_c3 = st.columns(3)
        k_c1.metric("Kelly Final Bankroll", f"${bankroll:,.2f}",
                    delta=f"{(bankroll/bankroll_start - 1)*100:.1f}%")
        k_c2.metric("Flat Bet Final Bankroll", f"${flat_bankroll:,.2f}",
                    delta=f"{(flat_bankroll/bankroll_start - 1)*100:.1f}%")
        k_c3.metric("Kelly Fraction", f"{kelly_frac}×")
        st.caption(
            "Odds: -110 flat (historical NHL lines not stored — break-even = 52.4% accuracy). "
            "Kelly bets when model confidence exceeds 52.4%."
        )

        try:
            import plotly.graph_objects as go
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=kelly_history, name=f'Kelly ({kelly_frac}×)',
                                      line=dict(color='#9b59b6', width=2)))
            fig2.add_trace(go.Scatter(y=flat_history, name='Flat 2%',
                                      line=dict(color='#e67e22', width=1.5, dash='dot')))
            fig2.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
            fig2.update_layout(
                title="Bankroll Index (100 = starting bankroll)",
                xaxis_title="Game", yaxis_title="Index",
                height=350, margin=dict(l=40, r=20, t=40, b=40),
                legend=dict(orientation='h', y=1.02),
            )
            st.plotly_chart(fig2, use_container_width=True)
        except ImportError:
            pass
    else:
        st.warning("Feature engineering failed. Check that nhl_games_processed.csv exists.")


# ── Track Record tab ───────────────────────────────────────────────────────────

def _render_nhl_track_record():
    """NHL Track Record tab — Prediction History, Bet Tracker, Export/Import."""
    st.header("📋 Track Record")
    st.caption("Auto-logged predictions from this session · Log bets you placed · Export your data")
    st.divider()

    if not _PH_OK:
        st.error("prediction_history.py not found — Track Record unavailable.")
        return

    import json as _json

    _tr_h, _tr_b, _tr_e = st.tabs(["📊 Prediction History", "💰 Bet Tracker", "📤 Export / Import"])

    # ── Prediction History ────────────────────────────────────────────────────
    with _tr_h:
        _tr_c1, _tr_c2 = st.columns([6, 2])
        with _tr_c2:
            if st.button("🔄 Refresh Results", key='nhl_tr_refresh', use_container_width=True,
                         help="Check NHL API for completed games and fill in actual results"):
                with st.spinner("Backfilling results..."):
                    _n = _ph.backfill_results()
                if _n:
                    st.success(f"Updated {_n} game result(s)")
                else:
                    st.info("No new results found")

        _all_preds = _ph.load_predictions()
        _nhl_preds = [p for p in _all_preds if p.get('sport') == 'nhl']

        if not _nhl_preds:
            st.info("No NHL predictions logged yet. Load the schedule and predictions will appear here automatically.")
        else:
            _fc1, _fc2, _fc3 = st.columns(3)
            with _fc1:
                _sig_opts = ["All Signals", "💎 STRONG", "📈 LEAN", "👀 SMALL", "⚪ PASS"]
                _sig_filter = st.selectbox("Signal", _sig_opts, key='nhl_tr_sig_filter')
            with _fc2:
                _out_opts = ["All Outcomes", "✅ Correct", "❌ Incorrect", "⏳ Pending"]
                _out_filter = st.selectbox("Outcome", _out_opts, key='nhl_tr_out_filter')
            with _fc3:
                _sort_opts = ["Newest First", "Oldest First"]
                _sort_sel = st.selectbox("Sort", _sort_opts, key='nhl_tr_sort')

            _sig_map = {"💎 STRONG": "STRONG", "📈 LEAN": "LEAN", "👀 SMALL": "SMALL", "⚪ PASS": "PASS"}
            _fp = _nhl_preds[:]
            if _sig_filter != "All Signals":
                _fp = [p for p in _fp if p.get('kelly_signal') == _sig_map.get(_sig_filter)]
            if _out_filter == "✅ Correct":
                _fp = [p for p in _fp if p.get('prediction_correct') is True]
            elif _out_filter == "❌ Incorrect":
                _fp = [p for p in _fp if p.get('prediction_correct') is False]
            elif _out_filter == "⏳ Pending":
                _fp = [p for p in _fp if p.get('prediction_correct') is None]

            _fp.sort(key=lambda x: x.get('predicted_at', ''), reverse=(_sort_sel == "Newest First"))

            _rows = []
            for _p in _fp:
                _winner_pred = _p['home_team'] if _p.get('model_home_prob', 0.5) >= 0.5 else _p['away_team']
                _correct = _p.get('prediction_correct')
                _outcome = "✅" if _correct is True else ("❌" if _correct is False else "⏳")
                _actual = _p.get('actual_winner', '—') or '—'
                _score = (f"{_p.get('actual_score_home','')}-{_p.get('actual_score_away','')}"
                          if _p.get('actual_score_home') is not None else '—')
                _badge_map = {'STRONG': '💎', 'LEAN': '📈', 'SMALL': '👀', 'PASS': '⚪'}
                _badge = _badge_map.get(_p.get('kelly_signal', 'PASS'), '⚪')
                _prob_disp = (_p.get('model_home_prob', 0.5) if _winner_pred == _p['home_team']
                              else 1 - _p.get('model_home_prob', 0.5))
                _rows.append({
                    'Date': _p.get('game_date', ''),
                    'Matchup': f"{_p['away_team']} @ {_p['home_team']}",
                    'Pick': _winner_pred,
                    'Win Prob': f"{_prob_disp*100:.1f}%",
                    'Edge': f"{_p.get('model_edge_pct', 0):+.1f}%",
                    'Signal': f"{_badge} {_p.get('kelly_signal', 'PASS')}",
                    'Result': _actual,
                    'Score': _score,
                    'O/U': '✅' if _p.get('ou_correct') is True else ('❌' if _p.get('ou_correct') is False else '—'),
                    '': _outcome,
                })

            st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)

            _settled = [p for p in _fp if p.get('prediction_correct') is not None]
            if _settled:
                st.markdown("---")
                _sc1, _sc2, _sc3, _sc4 = st.columns(4)
                _hit_rate = sum(1 for p in _settled if p.get('prediction_correct')) / len(_settled)
                _sc1.metric("Overall Hit Rate", f"{_hit_rate*100:.1f}%", f"{len(_settled)} settled")
                _strong = [p for p in _settled if p.get('kelly_signal') == 'STRONG']
                if _strong:
                    _strong_hr = sum(1 for p in _strong if p.get('prediction_correct')) / len(_strong)
                    _sc2.metric("💎 STRONG Hit Rate", f"{_strong_hr*100:.1f}%", f"{len(_strong)} bets")
                _correct_edges = [p.get('model_edge_pct', 0) for p in _settled if p.get('prediction_correct')]
                if _correct_edges:
                    _sc3.metric("Avg Edge (wins)", f"{sum(_correct_edges)/len(_correct_edges):.1f}%")
                _ou_settled = [p for p in _fp if p.get('ou_correct') is not None]
                if _ou_settled:
                    _ou_hr = sum(1 for p in _ou_settled if p.get('ou_correct')) / len(_ou_settled)
                    _sc4.metric("O/U Hit Rate", f"{_ou_hr*100:.1f}%", f"{len(_ou_settled)} settled")

    # ── Bet Tracker ───────────────────────────────────────────────────────────
    with _tr_b:
        _all_preds_bt = _ph.load_predictions()
        _nhl_preds_bt = [p for p in _all_preds_bt if p.get('sport') == 'nhl']
        _all_bets = [b for b in _ph.load_bets() if b.get('sport') == 'nhl']

        with st.expander("➕ Log a Bet", expanded=not bool(_all_bets)):
            if not _nhl_preds_bt:
                st.info("No NHL predictions logged yet — load the schedule first.")
            else:
                _pred_options = {
                    f"{p['away_team']} @ {p['home_team']} ({p.get('game_date','')})": p['id']
                    for p in sorted(_nhl_preds_bt, key=lambda x: x.get('game_date',''), reverse=True)
                }
                with st.form("nhl_log_bet_form"):
                    _sel_label = st.selectbox("Game", list(_pred_options.keys()), key='nhl_bt_game_sel')
                    _sel_pred_id = _pred_options[_sel_label]
                    _sel_pred = next((p for p in _nhl_preds_bt if p['id'] == _sel_pred_id), {})
                    _home_t = _sel_pred.get('home_team', '')
                    _away_t = _sel_pred.get('away_team', '')
                    _bt_c1, _bt_c2 = st.columns(2)
                    with _bt_c1:
                        _pick_team = st.selectbox("Bet On", [_home_t, _away_t] if _home_t else ["—"])
                        _bet_amount = st.number_input("Bet Amount ($)", min_value=1.0, max_value=100_000.0,
                                                      value=50.0, step=5.0)
                    with _bt_c2:
                        _odds_taken = st.number_input("Odds (American)", value=-110, format="%d")
                        _sportsbook = st.text_input("Sportsbook", value="DraftKings", max_chars=30)
                    _submit_bet = st.form_submit_button("Log Bet", type="primary", use_container_width=True)
                    if _submit_bet:
                        _bet_rec = {
                            'id': f"bet_nhl_{_sel_pred_id}_{int(datetime.now().timestamp())}",
                            'prediction_id': _sel_pred_id,
                            'sport': 'nhl',
                            'game_date': _sel_pred.get('game_date', ''),
                            'home_team': _home_t,
                            'away_team': _away_t,
                            'pick': _pick_team,
                            'amount': float(_bet_amount),
                            'odds': int(_odds_taken),
                            'sportsbook': _sportsbook,
                            'logged_at': datetime.now().isoformat(),
                        }
                        _ph.log_bet(_bet_rec)
                        st.success(f"Logged: ${_bet_amount:.0f} on {_pick_team} at {_odds_taken:+d}")
                        st.rerun()

        if _all_bets:
            _pred_lookup = {p['id']: p for p in _nhl_preds_bt}
            _bet_rows = []
            for _b in sorted(_all_bets, key=lambda x: x.get('logged_at', ''), reverse=True):
                _pred = _pred_lookup.get(_b.get('prediction_id', ''), {})
                _actual_w = _pred.get('actual_winner')
                _result_str = '⏳'
                _pnl = None
                if _actual_w is not None:
                    _od = _b.get('odds', -110)
                    _dec = (1 + 100.0 / abs(_od)) if _od < 0 else (1 + _od / 100.0)
                    if _b.get('pick') == _actual_w:
                        _pnl = _b['amount'] * (_dec - 1)
                        _result_str = '✅ Win'
                    else:
                        _pnl = -_b['amount']
                        _result_str = '❌ Loss'
                _bet_rows.append({
                    'Date': _b.get('game_date', ''),
                    'Game': f"{_b.get('away_team','')} @ {_b.get('home_team','')}",
                    'Pick': _b.get('pick', ''),
                    'Amount': f"${_b.get('amount', 0):.0f}",
                    'Odds': f"{_b.get('odds', 0):+d}",
                    'Book': _b.get('sportsbook', ''),
                    'Result': _result_str,
                    'P&L': f"${_pnl:+.0f}" if _pnl is not None else '—',
                })
            st.dataframe(pd.DataFrame(_bet_rows), use_container_width=True, hide_index=True)

            _settled_bets = [(_b, _pred_lookup.get(_b.get('prediction_id', ''), {}))
                             for _b in _all_bets
                             if _pred_lookup.get(_b.get('prediction_id', ''), {}).get('actual_winner') is not None]
            if _settled_bets:
                _total_staked = sum(_b['amount'] for _b, _ in _settled_bets)
                _total_pnl = 0.0
                for _b, _pred in _settled_bets:
                    _od = _b.get('odds', -110)
                    _dec = (1 + 100.0 / abs(_od)) if _od < 0 else (1 + _od / 100.0)
                    if _b.get('pick') == _pred.get('actual_winner'):
                        _total_pnl += _b['amount'] * (_dec - 1)
                    else:
                        _total_pnl -= _b['amount']
                _roi = (_total_pnl / _total_staked * 100) if _total_staked > 0 else 0
                _sm1, _sm2, _sm3 = st.columns(3)
                _sm1.metric("Total Staked", f"${_total_staked:.0f}")
                _sm2.metric("Net P&L", f"${_total_pnl:+.0f}")
                _sm3.metric("ROI", f"{_roi:+.1f}%")

            _cum_pnl = []
            _running = 0.0
            for _b in sorted(_all_bets, key=lambda x: x.get('logged_at', '')):
                _pred = _pred_lookup.get(_b.get('prediction_id', ''), {})
                _aw = _pred.get('actual_winner')
                if _aw is not None:
                    _od = _b.get('odds', -110)
                    _dec = (1 + 100.0 / abs(_od)) if _od < 0 else (1 + _od / 100.0)
                    _running += _b['amount'] * (_dec - 1) if _b.get('pick') == _aw else -_b['amount']
                    _cum_pnl.append({
                        'Bet #': len(_cum_pnl) + 1,
                        'Cumulative P&L': round(_running, 2),
                        'Game': f"{_b.get('away_team','')} @ {_b.get('home_team','')}",
                    })
            if _cum_pnl:
                import plotly.express as px
                _fig_pnl = px.line(pd.DataFrame(_cum_pnl), x='Bet #', y='Cumulative P&L',
                                   hover_data=['Game'],
                                   title='Cumulative P&L (settled bets only)')
                _fig_pnl.add_hline(y=0, line_dash='dash', line_color='gray')
                _fig_pnl.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(_fig_pnl, use_container_width=True)
        else:
            st.info("No bets logged yet. Expand **Log a Bet** above to record your first bet.")

    # ── Export / Import ───────────────────────────────────────────────────────
    with _tr_e:
        _exp_c1, _exp_c2 = st.columns(2)
        with _exp_c1:
            st.markdown("**📥 Export Data**")
            _all_p_exp = _ph.load_predictions()
            _all_b_exp = _ph.load_bets()
            _export_pkg = {'predictions': _all_p_exp, 'bets': _all_b_exp,
                           'exported_at': datetime.now().isoformat()}
            st.download_button(
                "Download All Data (JSON)",
                data=_json.dumps(_export_pkg, indent=2, default=str),
                file_name=f"edgeiq_track_record_{date.today().isoformat()}.json",
                mime="application/json",
                key='nhl_export_all',
                use_container_width=True,
            )
            st.download_button(
                "Download Predictions Only (JSON)",
                data=_json.dumps(_all_p_exp, indent=2, default=str),
                file_name=f"edgeiq_predictions_{date.today().isoformat()}.json",
                mime="application/json",
                key='nhl_export_preds',
                use_container_width=True,
            )
            st.download_button(
                "Download Bets Only (JSON)",
                data=_json.dumps(_all_b_exp, indent=2, default=str),
                file_name=f"edgeiq_bets_{date.today().isoformat()}.json",
                mime="application/json",
                key='nhl_export_bets',
                use_container_width=True,
            )
        with _exp_c2:
            st.markdown("**📤 Import Data**")
            _up = st.file_uploader("Upload EdgeIQ JSON backup", type=["json"], key='nhl_import_upload')
            if _up is not None:
                try:
                    _imported = _json.load(_up)
                    _imp_preds = (_imported.get('predictions', _imported)
                                  if isinstance(_imported, dict) else _imported)
                    _imp_bets = _imported.get('bets', []) if isinstance(_imported, dict) else []
                    if st.button("Confirm Import", key='nhl_confirm_import', type="primary"):
                        _imp_count_p = 0
                        for _ip in (_imp_preds if isinstance(_imp_preds, list) else []):
                            if _ip.get('id'):
                                _ph.log_prediction(_ip)
                                _imp_count_p += 1
                        _imp_count_b = 0
                        for _ib in (_imp_bets if isinstance(_imp_bets, list) else []):
                            if _ib.get('id'):
                                _ph.log_bet(_ib)
                                _imp_count_b += 1
                        st.success(f"Imported {_imp_count_p} predictions, {_imp_count_b} bets")
                        st.rerun()
                except Exception as _imp_err:
                    st.error(f"Import failed: {_imp_err}")


# ── Main render function ───────────────────────────────────────────────────────

def render_nhl_app():
    """Entry point called from app.py."""
    # Back to Home button
    back_col, title_col = st.columns([1, 8])
    with back_col:
        if st.button("🏠 Home", key="nhl_back_home"):
            st.session_state['sport'] = None
            st.rerun()
    with title_col:
        st.markdown('<div class="edgeiq-logo"><span class="edgeiq-icon">⚡</span> NHL Terminal</div>', unsafe_allow_html=True)

    # Load all data
    model, features, accuracy, elo_ratings = load_nhl_model()
    total_model_pkg      = load_nhl_total_model()
    nhl_games            = load_nhl_games()
    goalie_ratings       = load_nhl_goalie_ratings()
    team_stats           = load_nhl_team_stats()
    full_goalie_ratings  = load_nhl_full_goalie_ratings()

    # Sidebar — ELO rankings
    with st.sidebar:
        st.markdown("### 💰 Bankroll Settings")
        st.number_input(
            "My bankroll ($)", min_value=100, max_value=100_000,
            value=min(st.session_state.get('nhl_bankroll', 1000), 100_000), step=100,
            key='nhl_bankroll',
            help="Used for Kelly bet amount recommendations in game predictions.",
        )
        st.markdown("---")
        st.markdown("**🎯 Betting Settings**")
        _nhl_strat = st.selectbox(
            "Betting Strategy",
            options=["Kelly Criterion", "Fixed %", "Fixed $", "Fractional Kelly"],
            key='nhl_bet_strategy',
            help="Kelly: model edge × risk tolerance · Fixed %: set % of bankroll · Fixed $: set dollar amount · Fractional Kelly: custom fraction"
        )
        st.selectbox(
            "Risk Tolerance",
            options=["Conservative", "Moderate", "Aggressive"],
            index=1,
            key='nhl_risk_tolerance',
            help="Conservative 0.25× · Moderate 0.5× · Aggressive 1.0× Kelly multiplier"
        )
        if _nhl_strat == "Fixed %":
            st.number_input(
                "Fixed bet (%)", min_value=0.5, max_value=10.0,
                value=float(st.session_state.get('nhl_fixed_pct', 2.0)), step=0.5,
                key='nhl_fixed_pct',
                help="Bet this % of bankroll per game regardless of model edge"
            )
        elif _nhl_strat == "Fixed $":
            st.number_input(
                "Fixed bet ($)", min_value=1, max_value=10_000,
                value=int(st.session_state.get('nhl_fixed_dollar', 50)), step=10,
                key='nhl_fixed_dollar',
                help="Bet this fixed dollar amount per game regardless of model edge"
            )
        # ── Daily Summary ─────────────────────────────────────────────────
        if _PH_OK:
            st.markdown("---")
            st.markdown("**📊 Today's Summary**")
            try:
                _daily = _ph.get_daily_recs('nhl')
                if _daily:
                    _br = int(st.session_state.get('nhl_bankroll', 1000))
                    _st = st.session_state.get('nhl_bet_strategy', 'Kelly Criterion')
                    _fp = float(st.session_state.get('nhl_fixed_pct', 2.0))
                    _fd = int(st.session_state.get('nhl_fixed_dollar', 50))
                    _rt = st.session_state.get('nhl_risk_tolerance', 'Moderate')
                    _frac = {'Conservative': 0.25, 'Moderate': 0.5, 'Aggressive': 1.0}[_rt]
                    _total_stake = 0.0
                    _total_win = 0.0
                    _total_ev = 0.0
                    for _dr in _daily:
                        _kp = _dr.get('kelly_pct', 0.0)
                        if _st == 'Fixed %':
                            _amt = _br * _fp / 100
                        elif _st == 'Fixed $':
                            _amt = float(_fd)
                        else:
                            _amt = _br * _kp / 100
                        _ml = _dr.get('vegas_ml_home', -110) or -110
                        _dec = (1 + 100.0 / abs(_ml)) if _ml < 0 else (1 + _ml / 100.0)
                        _total_stake += _amt
                        _total_win += _amt * (_dec - 1)
                        _total_ev += _amt * (_dr.get('model_edge_pct', 0.0) / 100)
                    _ds1, _ds2 = st.columns(2)
                    _ds1.metric("Bets", f"{len(_daily)}")
                    _ds2.metric("Stake", f"${_total_stake:.0f}")
                    _ds1.metric("Pot. Win", f"${_total_win:.0f}")
                    _ds2.metric("EV", f"${_total_ev:+.0f}")
                else:
                    st.caption("No picks logged yet today")
            except Exception:
                pass

        st.markdown("---")
        st.markdown("**📊 NHL ELO Rankings**")
        if elo_ratings:
            from apis.nhl import NHL_TEAMS
            elo_series = pd.Series(elo_ratings)
            elo_series = elo_series[elo_series.index.isin(NHL_TEAMS)].sort_values(ascending=False)
            elo_df = elo_series.reset_index()
            elo_df.columns = ['Team', 'ELO']
            elo_df['ELO'] = elo_df['ELO'].round(0).astype(int)
            elo_df.index += 1
            st.dataframe(elo_df, use_container_width=True, height=350)
        else:
            st.caption("ELO ratings not loaded yet")
        if accuracy:
            st.success(f"Model accuracy: {accuracy*100:.1f}%")

    # Model info bar
    if model is not None:
        st.markdown(f"*31-feature stacking ensemble · {accuracy*100:.1f}% holdout accuracy · 25 seasons of NHL data*")
        st.divider()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🏒 Game Predictor", "📅 Backtesting", "📋 Track Record"])

    with tab1:
        _render_tab1(model, features, accuracy, elo_ratings, goalie_ratings, team_stats, total_model_pkg, nhl_games, full_goalie_ratings=full_goalie_ratings)

    with tab2:
        _render_tab2(model, features, accuracy)

    with tab3:
        _render_nhl_track_record()
