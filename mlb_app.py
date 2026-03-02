"""
mlb_app.py
==========
MLB Predictor — 3-tab Streamlit module.
Called from app.py as render_mlb_app().

Tabs:
  1. Game Predictor (This Week's Games / Manual Entry)
  2. Backtesting (accuracy, $10 ML sim, Kelly criterion)
  3. Track Record (prediction history, bet tracker, export)

Mirrors nhl_app.py / final_app.py structure.
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

# ── MLB team constants ─────────────────────────────────────────────────────────
# Uses MLB Stats API abbreviations (same as mlb_elo_ratings.pkl / mlb_games_processed.csv)
MLB_TEAMS = sorted([
    'AZ',  'ATL', 'BAL', 'BOS', 'CHC', 'CIN', 'CLE', 'COL',
    'CWS', 'DET', 'HOU', 'KC',  'LAA', 'LAD', 'MIA', 'MIL',
    'MIN', 'NYM', 'NYY', 'ATH', 'PHI', 'PIT', 'SD',  'SEA',
    'SF',  'STL', 'TB',  'TEX', 'TOR', 'WSH',
])

MLB_TEAM_NAMES = {
    'AZ':  'Arizona Diamondbacks',  'ATL': 'Atlanta Braves',
    'BAL': 'Baltimore Orioles',     'BOS': 'Boston Red Sox',
    'CHC': 'Chicago Cubs',          'CIN': 'Cincinnati Reds',
    'CLE': 'Cleveland Guardians',   'COL': 'Colorado Rockies',
    'CWS': 'Chicago White Sox',     'DET': 'Detroit Tigers',
    'HOU': 'Houston Astros',        'KC':  'Kansas City Royals',
    'LAA': 'Los Angeles Angels',    'LAD': 'Los Angeles Dodgers',
    'MIA': 'Miami Marlins',         'MIL': 'Milwaukee Brewers',
    'MIN': 'Minnesota Twins',       'NYM': 'New York Mets',
    'NYY': 'New York Yankees',      'ATH': 'Athletics',
    'PHI': 'Philadelphia Phillies', 'PIT': 'Pittsburgh Pirates',
    'SD':  'San Diego Padres',      'SEA': 'Seattle Mariners',
    'SF':  'San Francisco Giants',  'STL': 'St. Louis Cardinals',
    'TB':  'Tampa Bay Rays',        'TEX': 'Texas Rangers',
    'TOR': 'Toronto Blue Jays',     'WSH': 'Washington Nationals',
}

# Abbreviation bridge: MLB Stats API → FanGraphs (for team stats + pitcher ratings lookup)
_GAMES_TO_STATS = {
    'AZ':  'ARI', 'CWS': 'CHW', 'KC':  'KCR',
    'SD':  'SDP', 'SF':  'SFG', 'TB':  'TBR',
    'WSH': 'WSN', 'LA':  'LAD',
}

def _stats_key(team: str, season: int = 2025) -> str:
    t = _GAMES_TO_STATS.get(team, team)
    if team == 'TB' and season < 2008:
        return 'TBD'
    if team == 'MIA' and season < 2012:
        return 'FLA'
    return t


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_mlb_model():
    try:
        with open("model_mlb_enhanced.pkl", "rb") as f:
            pkg = pickle.load(f)
        model    = pkg['model']
        features = pkg['features']
        accuracy = pkg.get('accuracy', 0.0)
        test_fv  = pd.DataFrame([{feat: 0.0 for feat in features}])
        model.predict_proba(test_fv)
    except Exception as e:
        logger.error(f"MLB model load failed: {e}")
        model, features, accuracy = None, [], 0.0

    try:
        with open("mlb_elo_ratings.pkl", "rb") as f:
            elo = pickle.load(f)
    except Exception:
        elo = {}

    return model, features, accuracy, elo


@st.cache_resource
def load_mlb_total_model():
    try:
        with open("model_mlb_total.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


@st.cache_data
def load_mlb_games():
    try:
        return pd.read_csv("mlb_games_processed.csv")
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_mlb_pitcher_ratings():
    """Current-season pitcher quality by team (FanGraphs codes)."""
    try:
        df = pd.read_csv("mlb_pitcher_team_ratings.csv")
        return df.set_index('team') if 'team' in df.columns else df
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_mlb_full_pitcher_ratings():
    """All pitcher ratings (name-based lookup for manual entry)."""
    try:
        df = pd.read_csv("mlb_pitcher_ratings.csv")
        max_season = df['season'].max()
        return df[df['season'] == max_season].copy()
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_mlb_team_stats():
    """Current-season team batting + pitching stats (FanGraphs codes)."""
    try:
        df = pd.read_csv("mlb_team_stats_current.csv")
        max_season = df['season'].max()
        return df[df['season'] == max_season].set_index('team')
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_mlb_historical_features():
    from mlb_feature_engineering import build_mlb_enhanced_features, MLB_ENHANCED_FEATURES
    games = load_mlb_games()
    if games.empty:
        return games, []
    try:
        df_eng = build_mlb_enhanced_features(games)
        return df_eng, MLB_ENHANCED_FEATURES
    except Exception as e:
        logger.error(f"MLB feature engineering failed: {e}")
        return games, []


# ── ELO helpers ───────────────────────────────────────────────────────────────

def get_mlb_elo(team: str, elo_ratings: dict) -> float:
    return float(elo_ratings.get(team, 1500.0))


def mlb_elo_win_prob(home_elo: float, away_elo: float, home_adv: float = 35.0) -> float:
    return 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + home_adv)) / 400.0))


# ── Rolling form from historical data ────────────────────────────────────────

def _team_l5_runs(team: str, col: str, mlb_games: pd.DataFrame) -> float:
    """Return team's L5 avg runs for ('rf') or against ('ra')."""
    if mlb_games is None or mlb_games.empty:
        return 4.5  # league avg ~4.5 per team per game
    home_g = mlb_games[mlb_games['home_team'] == team].tail(10)
    away_g = mlb_games[mlb_games['away_team'] == team].tail(10)
    all_g  = pd.concat([home_g, away_g]).sort_values('game_date').tail(5)
    if all_g.empty:
        return 4.5
    rf_list, ra_list = [], []
    for _, row in all_g.iterrows():
        if row['home_team'] == team:
            rf_list.append(row.get('home_score', 4))
            ra_list.append(row.get('away_score', 4))
        else:
            rf_list.append(row.get('away_score', 4))
            ra_list.append(row.get('home_score', 4))
    if col == 'rf':
        return float(np.mean(rf_list)) if rf_list else 4.5
    return float(np.mean(ra_list)) if ra_list else 4.5


def _team_l10_wins(team: str, mlb_games: pd.DataFrame) -> float:
    if mlb_games is None or mlb_games.empty:
        return 0.5
    home_g = mlb_games[mlb_games['home_team'] == team][['game_date', 'home_win']].rename(
        columns={'home_win': 'won'})
    away_g = mlb_games[mlb_games['away_team'] == team][['game_date', 'home_win']].copy()
    away_g['won'] = 1 - away_g['home_win']
    away_g = away_g[['game_date', 'won']]
    all_g = pd.concat([home_g, away_g]).sort_values('game_date').tail(10)
    if all_g.empty:
        return 0.5
    return float(all_g['won'].mean())


# ── Pitcher quality helpers ────────────────────────────────────────────────────

def _get_pitcher_score(
    team: str,
    pitcher_name: str = None,
    pitcher_ratings: pd.DataFrame = None,
    full_pitcher_ratings: pd.DataFrame = None,
) -> float:
    fg_key = _stats_key(team)
    if pitcher_name and full_pitcher_ratings is not None and not full_pitcher_ratings.empty:
        match = full_pitcher_ratings[
            full_pitcher_ratings['name'].str.strip() == pitcher_name.strip()
        ]
        if not match.empty:
            return float(match.iloc[0]['pitcher_score'])
    if pitcher_ratings is not None and not pitcher_ratings.empty and fg_key in pitcher_ratings.index:
        return float(pitcher_ratings.loc[fg_key, 'pitcher_score'])
    return 0.0


# ── Feature vector builder ────────────────────────────────────────────────────

def build_mlb_prediction_features(
    home_team: str,
    away_team: str,
    features: list,
    elo_ratings: dict,
    pitcher_ratings: pd.DataFrame,
    team_stats: pd.DataFrame,
    moneyline_home: float = None,
    mlb_games: pd.DataFrame = None,
    h_sp_name: str = None,
    a_sp_name: str = None,
    full_pitcher_ratings: pd.DataFrame = None,
    is_day_game: int = 0,
) -> pd.DataFrame:
    from mlb_feature_engineering import _elo_win_prob, _moneyline_to_prob

    home_elo = get_mlb_elo(home_team, elo_ratings)
    away_elo = get_mlb_elo(away_team, elo_ratings)
    elo_diff = home_elo - away_elo

    elo_prob = mlb_elo_win_prob(home_elo, away_elo)
    ml_prob  = _moneyline_to_prob(moneyline_home) if moneyline_home else elo_prob

    run_line = -1.5 if elo_diff > 0 else 1.5

    # Rolling form from historical data
    h_rf = _team_l5_runs(home_team, 'rf', mlb_games)
    h_ra = _team_l5_runs(home_team, 'ra', mlb_games)
    a_rf = _team_l5_runs(away_team, 'rf', mlb_games)
    a_ra = _team_l5_runs(away_team, 'ra', mlb_games)

    # L10 win rates
    h_l10w = _team_l10_wins(home_team, mlb_games)
    a_l10w = _team_l10_wins(away_team, mlb_games)

    # Pitcher quality
    h_sp = _get_pitcher_score(home_team, h_sp_name, pitcher_ratings, full_pitcher_ratings)
    a_sp = _get_pitcher_score(away_team, a_sp_name, pitcher_ratings, full_pitcher_ratings)
    pitcher_diff = h_sp - a_sp

    # Team batting + pitching stats from current season CSV
    LEAGUE_WOBA      = 0.320
    LEAGUE_WRC       = 100
    LEAGUE_ERA_MINUS = 100
    LEAGUE_FIP_MINUS = 100

    def _ts(team, col, default):
        fg = _stats_key(team)
        if team_stats is not None and not team_stats.empty and fg in team_stats.index:
            val = team_stats.loc[fg, col] if col in team_stats.columns else default
            return float(val) if not pd.isna(val) else default
        return default

    fv = {
        'run_line':               run_line,
        'moneyline_implied_prob': ml_prob,
        'run_line_implied_prob':  0.5,
        'mlb_elo_diff':           elo_diff,
        'mlb_elo_implied_prob':   elo_prob,
        'home_mlb_elo_trend':     0.0,
        'away_mlb_elo_trend':     0.0,
        'home_l5_runs_for':       h_rf,
        'away_l5_runs_for':       a_rf,
        'home_l5_runs_against':   h_ra,
        'away_l5_runs_against':   a_ra,
        'home_l5_run_diff':       h_rf - h_ra,
        'away_l5_run_diff':       a_rf - a_ra,
        'matchup_adv_home':       h_rf - a_ra,
        'matchup_adv_away':       a_rf - h_ra,
        'net_matchup_adv':        (h_rf - a_ra) - (a_rf - h_ra),
        'pitcher_quality_diff':   pitcher_diff,
        'home_woba':              _ts(home_team, 'woba',     LEAGUE_WOBA),
        'away_woba':              _ts(away_team, 'woba',     LEAGUE_WOBA),
        'home_wrc_plus':          _ts(home_team, 'wrc_plus', LEAGUE_WRC),
        'away_wrc_plus':          _ts(away_team, 'wrc_plus', LEAGUE_WRC),
        'home_era_minus':         _ts(home_team, 'era_minus', LEAGUE_ERA_MINUS),
        'away_era_minus':         _ts(away_team, 'era_minus', LEAGUE_ERA_MINUS),
        'home_fip_minus':         _ts(home_team, 'fip_minus', LEAGUE_FIP_MINUS),
        'away_fip_minus':         _ts(away_team, 'fip_minus', LEAGUE_FIP_MINUS),
        'home_l10_wins':          h_l10w,
        'away_l10_wins':          a_l10w,
        'win_pct_advantage':      h_l10w - a_l10w,
        'is_day_game':            is_day_game,
    }

    for f in features:
        if f not in fv:
            fv[f] = 0.0

    return pd.DataFrame([{f: fv.get(f, 0.0) for f in features}])


# ── Prediction runner ─────────────────────────────────────────────────────────

def run_mlb_prediction(
    home_team: str,
    away_team: str,
    model,
    features: list,
    elo_ratings: dict,
    pitcher_ratings: pd.DataFrame,
    team_stats: pd.DataFrame,
    total_model_pkg: dict,
    moneyline_home: float = None,
    vegas_total: float = None,
    h_sp_name: str = None,
    a_sp_name: str = None,
    full_pitcher_ratings: pd.DataFrame = None,
    mlb_games: pd.DataFrame = None,
    is_day_game: int = 0,
) -> dict:
    if model is None:
        return {'error': 'Model not loaded'}

    fv = build_mlb_prediction_features(
        home_team, away_team, features, elo_ratings,
        pitcher_ratings, team_stats,
        moneyline_home=moneyline_home,
        mlb_games=mlb_games,
        h_sp_name=h_sp_name,
        a_sp_name=a_sp_name,
        full_pitcher_ratings=full_pitcher_ratings,
        is_day_game=is_day_game,
    )

    prob_home = float(model.predict_proba(fv)[0][1])
    prob_away = 1.0 - prob_home

    ou_pred = ou_diff = ou_mae = None
    if total_model_pkg:
        ou_features = total_model_pkg.get('features', [])
        ou_fv = fv.reindex(columns=ou_features, fill_value=0.0)
        league_avg = total_model_pkg.get('league_avg_total', 9.1)
        try:
            residual = total_model_pkg['model'].predict(ou_fv)[0]
            ou_pred  = league_avg + residual
            ou_diff  = ou_pred - (vegas_total or league_avg)
            ou_mae   = total_model_pkg.get('mae', 3.44)
        except Exception:
            pass

    ml_implied = None
    if moneyline_home is not None:
        if moneyline_home > 0:
            ml_implied = 100.0 / (moneyline_home + 100.0)
        else:
            ml_implied = abs(moneyline_home) / (abs(moneyline_home) + 100.0)

    return {
        'home_team':      home_team,
        'away_team':      away_team,
        'home_win_prob':  prob_home,
        'away_win_prob':  prob_away,
        'ou_pred':        ou_pred,
        'ou_diff':        ou_diff,
        'ou_mae':         ou_mae,
        'vegas_total':    vegas_total,
        'ml_implied':     ml_implied,
        'moneyline_home': moneyline_home,
        'elo_diff':       fv['mlb_elo_diff'].values[0] if 'mlb_elo_diff' in fv.columns else 0.0,
    }


# ── Prediction display ────────────────────────────────────────────────────────

def _mlb_kelly(p: float, ml: float, frac: float = 0.5):
    try:
        b    = (100.0 / abs(ml)) if ml < 0 else (ml / 100.0)
        full = (b * p - (1.0 - p)) / b
        pct  = max(0.0, min(full * frac, 0.10)) * 100
    except Exception:
        return 0.0, 'signal-pass', 'PASS'
    if pct >= 4.0: return pct, 'signal-strong', 'STRONG EDGE'
    if pct >= 2.0: return pct, 'signal-lean',   'LEAN'
    if pct >= 1.0: return pct, 'signal-lean',   'SMALL EDGE'
    return pct, 'signal-pass', 'PASS'


def render_mlb_prediction_result(result: dict, prefix: str = "", game_date: str = None):
    if 'error' in result:
        st.error(result['error'])
        return

    home   = result['home_team']
    away   = result['away_team']
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

    if conf > 0.75:
        label, css = "🔒 LOCK", "signal-lock"
    elif conf > 0.65:
        label, css = "HIGH CONFIDENCE", "signal-strong"
    elif conf > 0.58:
        label, css = "MODERATE", "signal-lean"
    else:
        label, css = "TOSS-UP", "signal-pass"
    st.markdown(
        f'<div class="signal-badge {css}">{label}: {winner}</div>',
        unsafe_allow_html=True)

    if result.get('ml_implied') is not None:
        ml_imp = result['ml_implied']
        diff   = (prob_h - ml_imp) * 100
        if abs(diff) > 2:
            arrow = "↑" if diff > 0 else "↓"
            st.caption(f"Model: {prob_h*100:.1f}% | Vegas ML implied: {ml_imp*100:.1f}% ({arrow}{abs(diff):.1f}%)")

    # O/U prediction
    if result.get('ou_pred') is not None:
        st.markdown("---")
        ou_pred = result['ou_pred']
        ou_mae  = result.get('ou_mae', 3.44)
        ou_diff = result.get('ou_diff', 0)
        vt      = result.get('vegas_total')
        lean    = "OVER" if ou_diff > 0 else "UNDER"
        oc1, oc2 = st.columns(2)
        oc1.metric("Predicted Total", f"{ou_pred:.1f} runs",
                   help=f"±{ou_mae:.1f} run MAE")
        if vt:
            oc2.metric("vs Vegas Total", f"{lean} {vt}", f"{ou_diff:+.1f}")

    # ── Kelly Bet Sizing ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📐 Kelly Bet Sizing")
    _pick_home  = prob_h >= 0.5
    _pick_prob  = prob_h if _pick_home else prob_a
    _pick_label = home if _pick_home else away
    _ml_home    = result.get('moneyline_home')
    _pick_ml    = (_ml_home if _pick_home else -_ml_home) if _ml_home is not None else -110

    _strategy    = st.session_state.get('mlb_bet_strategy', 'Kelly Criterion')
    _risk_tol    = st.session_state.get('mlb_risk_tolerance', 'Moderate')
    _kelly_frac  = {'Conservative': 0.25, 'Moderate': 0.5, 'Aggressive': 1.0}[_risk_tol]
    _fixed_pct   = float(st.session_state.get('mlb_fixed_pct', 2.0))
    _fixed_dol   = int(st.session_state.get('mlb_fixed_dollar', 50))
    _bankroll    = int(st.session_state.get('mlb_bankroll', 1000))

    _kpct, _ktier, _kbadge = _mlb_kelly(_pick_prob, _pick_ml, frac=_kelly_frac)

    if _pick_ml < 0:
        _vegas_impl = abs(_pick_ml) / (abs(_pick_ml) + 100)
    else:
        _vegas_impl = 100 / (_pick_ml + 100)
    _edge = (_pick_prob - _vegas_impl) * 100

    if _strategy == 'Fixed %':
        _bet_amt    = _bankroll * _fixed_pct / 100
        _pct_label  = "Fixed %"
        _pct_val    = f"{_fixed_pct:.1f}%"
    elif _strategy == 'Fixed $':
        _bet_amt    = float(_fixed_dol)
        _pct_label  = "Fixed $"
        _pct_val    = f"${_fixed_dol}"
    else:
        _bet_amt    = _bankroll * _kpct / 100
        _pct_label  = "Kelly %"
        _pct_val    = f"{_kpct:.1f}%"

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Model Edge",  f"{_edge:+.1f}%",
               help="Model win prob minus Vegas implied prob for predicted winner")
    kc2.metric(_pct_label, _pct_val,
               help="Bet size based on selected strategy")
    kc3.metric("Bet Amount", f"${_bet_amt:.0f}",
               help=f"Of your ${_bankroll:,} bankroll")
    kc4.markdown(
        f'<p style="font-size:0.8em;color:gray;margin-bottom:4px">Signal</p>'
        f'<span class="signal-badge {_ktier}">{_kbadge}</span>',
        unsafe_allow_html=True)

    _ml_display = f"{_pick_ml:+.0f}" if _ml_home is not None else "-110 (est.)"
    if _strategy == 'Fixed %':
        _caption_extra = f"Fixed {_fixed_pct:.1f}% per game regardless of edge."
    elif _strategy == 'Fixed $':
        _caption_extra = f"Fixed ${_fixed_dol} per game regardless of edge."
    else:
        _caption_extra = f"{_risk_tol} Kelly ({_kelly_frac:.2g}×). Kelly caps at 10% of bankroll to limit volatility."
    st.caption(
        f"Betting on **{_pick_label}** at {_pick_prob*100:.1f}% model confidence. "
        f"Moneyline: {_ml_display}. Vegas implied: {_vegas_impl*100:.1f}%. {_caption_extra}"
    )

    # ── Log prediction history ─────────────────────────────────────────────────
    if _PH_OK:
        _log_key = f'{prefix}_logged' if prefix else f'mlb_{home}_{away}_logged'
        if not st.session_state.get(_log_key, False):
            try:
                _gdate  = game_date or date.today().isoformat()
                _ou_p   = result.get('ou_pred')
                _vt     = result.get('vegas_total')
                _rec = {
                    'id':            f"mlb_{home}_{away}_{_gdate}",
                    'sport':         'mlb',
                    'game_date':     _gdate,
                    'home_team':     home,
                    'away_team':     away,
                    'predicted_at':  datetime.now().isoformat(),
                    'home_win_prob': round(prob_h, 4),
                    'away_win_prob': round(prob_a, 4),
                    'predicted_winner': winner,
                    'confidence_tier':  label,
                    'kelly_pct':     round(_kpct, 2),
                    'bet_amount':    round(_bet_amt, 2),
                    'moneyline':     _ml_home,
                    'ou_prediction': round(_ou_p, 2) if _ou_p else None,
                    'vegas_total':   _vt,
                    'actual_winner': None,
                    'result':        None,
                }
                _ph.log_prediction(_rec)
                st.session_state[_log_key] = True
            except Exception:
                pass


# ── SP info panel ─────────────────────────────────────────────────────────────

def _render_sp_panel(team: str, sp_info: dict, pitcher_ratings: pd.DataFrame):
    fg_key = _stats_key(team)
    sp_name = None
    if sp_info and sp_info.get('name') and sp_info['name'] != 'TBD':
        sp_name = sp_info['name']
    elif pitcher_ratings is not None and not pitcher_ratings.empty and fg_key in pitcher_ratings.index:
        sp_name = str(pitcher_ratings.loc[fg_key, 'ace_name'])

    row = pitcher_ratings.loc[fg_key] if (pitcher_ratings is not None and not pitcher_ratings.empty and fg_key in pitcher_ratings.index) else None

    sp_display = sp_name or "TBD"
    era_m = int(row['era_minus']) if row is not None and not pd.isna(row.get('era_minus', float('nan'))) else "—"
    fip_m = int(row['fip_minus']) if row is not None and not pd.isna(row.get('fip_minus', float('nan'))) else "—"
    score = round(float(row['pitcher_score']), 2) if row is not None else "—"

    st.markdown(f"**⚾ SP:** {sp_display}")
    sm1, sm2, sm3 = st.columns(3)
    sm1.metric("ERA-", era_m, help="ERA- vs league avg (100 = avg, <100 = better)")
    sm2.metric("FIP-", fip_m, help="FIP- vs league avg (100 = avg, <100 = better)")
    sm3.metric("SP Score", score, help="Composite z-score (higher = better)")


# ── Weekly schedule section ───────────────────────────────────────────────────

def _render_weekly_games(model, features, elo_ratings, pitcher_ratings, team_stats,
                         total_model_pkg, mlb_games, full_pitcher_ratings, mlb_client):
    from mlb_game_week import fetch_mlb_weekly_schedule, get_mlb_sp_display_name

    try:
        schedule = fetch_mlb_weekly_schedule(mlb_client)
    except Exception as e:
        st.warning(f"Could not load schedule: {e}")
        schedule = {}

    if not schedule:
        st.info("No scheduled games found for this week. MLB regular season runs April – October.")
        return

    # Expand All / Collapse All
    ec1, ec2 = st.columns([1, 1])
    with ec1:
        if st.button("🔽 Expand All", key="mlb_expand_all"):
            for day, games in schedule.items():
                for idx, _ in enumerate(games):
                    st.session_state[f"mlb_g{day}{idx}_expanded"] = True
    with ec2:
        if st.button("🔼 Collapse All", key="mlb_collapse_all"):
            for day, games in schedule.items():
                for idx, _ in enumerate(games):
                    st.session_state[f"mlb_g{day}{idx}_expanded"] = False

    game_counter = 0
    for day_key, games in schedule.items():
        st.markdown(f"### {day_key}")
        for idx, game in enumerate(games):
            home       = game['home_team']
            away       = game['away_team']
            home_name  = MLB_TEAM_NAMES.get(home, home)
            away_name  = MLB_TEAM_NAMES.get(away, away)
            home_sp    = game.get('home_sp')
            away_sp    = game.get('away_sp')
            game_time  = game.get('game_time_et', 'TBD')
            venue      = game.get('venue', '')
            game_date  = game.get('game_date_label', date.today().isoformat())
            day_night  = game.get('day_night', 'night')

            # Pre-compute result and determine predicted winner label
            _sess_key = f"mlb_g{day_key}{idx}_pred"
            if _sess_key not in st.session_state:
                try:
                    _res = run_mlb_prediction(
                        home, away, model, features, elo_ratings,
                        pitcher_ratings, team_stats, total_model_pkg,
                        mlb_games=mlb_games,
                        h_sp_name=get_mlb_sp_display_name(home_sp) if home_sp else None,
                        a_sp_name=get_mlb_sp_display_name(away_sp) if away_sp else None,
                        full_pitcher_ratings=full_pitcher_ratings,
                        is_day_game=1 if day_night == 'day' else 0,
                    )
                    st.session_state[_sess_key] = _res
                except Exception:
                    st.session_state[_sess_key] = {}

            _res = st.session_state.get(_sess_key, {})
            ph = _res.get('home_win_prob', 0.5)
            pa = 1.0 - ph
            _winner = home if ph >= 0.5 else away
            _conf   = max(ph, pa)

            if _conf > 0.75:
                _conf_icon = "🔒"
            elif _conf > 0.65:
                _conf_icon = "🔥"
            elif _conf > 0.58:
                _conf_icon = "✅"
            else:
                _conf_icon = "⚠️"

            h_sp_label = get_mlb_sp_display_name(home_sp) if home_sp else "TBD"
            a_sp_label = get_mlb_sp_display_name(away_sp) if away_sp else "TBD"

            exp_label  = (
                f"{_conf_icon} {away_name} @ {home_name} — {game_time}  |  "
                f"Pred: **{_winner}** ({_conf*100:.0f}%)  |  "
                f"SP: {a_sp_label} vs {h_sp_label}"
            )
            exp_key    = f"mlb_g{day_key}{idx}_expanded"
            expanded   = st.session_state.get(exp_key, False)

            with st.expander(exp_label, expanded=expanded):
                st.session_state[exp_key] = True
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.markdown(f"**🏠 {home_name}** ({home})")
                    _render_sp_panel(home, home_sp, pitcher_ratings)
                with sc2:
                    st.markdown(f"**✈️ {away_name}** ({away})")
                    _render_sp_panel(away, away_sp, pitcher_ratings)

                if venue:
                    st.caption(f"📍 {venue}")

                if _res:
                    render_mlb_prediction_result(
                        _res,
                        prefix=f"mlb_wk_{day_key}{idx}",
                        game_date=game_date,
                    )

            game_counter += 1


# ── Manual entry section ──────────────────────────────────────────────────────

def _render_manual_entry(model, features, elo_ratings, pitcher_ratings, team_stats,
                         total_model_pkg, mlb_games, full_pitcher_ratings):
    st.markdown("### Manual Game Entry")
    mc1, mc2 = st.columns(2)
    with mc1:
        home_team = st.selectbox("Home Team", MLB_TEAMS,
                                 index=MLB_TEAMS.index('NYY') if 'NYY' in MLB_TEAMS else 0,
                                 key="mlb_manual_home")
    with mc2:
        away_team = st.selectbox("Away Team", MLB_TEAMS,
                                 index=MLB_TEAMS.index('BOS') if 'BOS' in MLB_TEAMS else 1,
                                 key="mlb_manual_away")

    with st.expander("⚾ Starting Pitchers (optional)", expanded=False):
        pc1, pc2 = st.columns(2)
        with pc1:
            h_sp = st.text_input(f"{home_team} SP", placeholder="e.g. Gerrit Cole", key="mlb_manual_h_sp")
        with pc2:
            a_sp = st.text_input(f"{away_team} SP", placeholder="e.g. Chris Sale", key="mlb_manual_a_sp")

    with st.expander("📊 Vegas Odds (optional)", expanded=False):
        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            ml_home = st.number_input(f"{home_team} Moneyline", value=0, step=5,
                                      help="Enter American odds (e.g. -140 or +120). 0 = not set.",
                                      key="mlb_manual_ml")
            ml_home_val = ml_home if ml_home != 0 else None
        with vc2:
            run_line_odds = st.number_input("Run Line Odds (home -1.5)", value=0, step=5,
                                            help="Odds for home -1.5. 0 = not set.",
                                            key="mlb_manual_rl_odds")
        with vc3:
            vegas_total = st.number_input("O/U Total", value=0.0, step=0.5,
                                          help="Vegas run total. 0 = use model average.",
                                          key="mlb_manual_total")
            vegas_total_val = vegas_total if vegas_total > 0 else None

    is_day = st.checkbox("Day game?", value=False, key="mlb_manual_daynight")

    if st.button("⚾ Run Prediction", type="primary", key="mlb_manual_run"):
        if home_team == away_team:
            st.error("Home and away teams must be different.")
            return
        with st.spinner("Computing prediction..."):
            result = run_mlb_prediction(
                home_team, away_team, model, features, elo_ratings,
                pitcher_ratings, team_stats, total_model_pkg,
                moneyline_home=ml_home_val,
                vegas_total=vegas_total_val,
                h_sp_name=h_sp or None,
                a_sp_name=a_sp or None,
                full_pitcher_ratings=full_pitcher_ratings,
                mlb_games=mlb_games,
                is_day_game=1 if is_day else 0,
            )
        st.session_state['mlb_manual_result'] = result

    if 'mlb_manual_result' in st.session_state:
        st.markdown("---")
        render_mlb_prediction_result(st.session_state['mlb_manual_result'], prefix="mlb_manual")


# ── Tab 1: Game Predictor ─────────────────────────────────────────────────────

def _render_tab1(model, features, accuracy, elo_ratings, pitcher_ratings, team_stats,
                 total_model_pkg, mlb_games, full_pitcher_ratings=None, mlb_client=None):
    st.markdown(f"**Model accuracy:** {accuracy*100:.1f}%  |  **Features:** {len(features)}  |  **Training games:** 56,000+")

    mode = st.radio("Mode", ["📅 This Week's Games", "✏️ Manual Entry"],
                    horizontal=True, key="mlb_tab1_mode")

    if mode == "📅 This Week's Games":
        _render_weekly_games(model, features, elo_ratings, pitcher_ratings, team_stats,
                             total_model_pkg, mlb_games, full_pitcher_ratings, mlb_client)
    else:
        _render_manual_entry(model, features, elo_ratings, pitcher_ratings, team_stats,
                             total_model_pkg, mlb_games, full_pitcher_ratings)


# ── Tab 2: Backtesting ────────────────────────────────────────────────────────

@st.cache_data
def _mlb_backtest_results(seasons: list):
    import plotly.graph_objects as go
    from mlb_feature_engineering import MLB_ENHANCED_FEATURES

    df_eng, feat_list = load_mlb_historical_features()
    if df_eng.empty or not feat_list:
        return None, None, None

    df = df_eng[df_eng['season'].isin(seasons)].copy()
    if df.empty:
        return None, None, None

    model_pkg = None
    try:
        with open("model_mlb_enhanced.pkl", "rb") as f:
            model_pkg = pickle.load(f)
    except Exception:
        return None, None, None

    model    = model_pkg['model']
    features = model_pkg['features']

    X = df[features].fillna(0.0)
    df = df.copy()
    df['prob_home'] = model.predict_proba(X)[:, 1]
    df['pred_home'] = (df['prob_home'] >= 0.5).astype(int)
    df['correct']   = (df['pred_home'] == df['home_win']).astype(int)

    accuracy = df['correct'].mean()

    # $10 per game moneyline simulation (flat bet, model side)
    df['bet_return'] = df.apply(
        lambda r: 10 * (100/110) if r['correct'] else -10, axis=1
    )
    df['cumulative_return'] = df['bet_return'].cumsum()

    # Half-Kelly simulation
    LEAGUE_AVG_ML = -110
    df['kelly_pct'] = df.apply(
        lambda r: max(0.0, min(
            ((100/110) * r['prob_home'] - (1 - r['prob_home'])) / (100/110) * 0.5,
            0.10
        )) if r['prob_home'] >= 0.5 else max(0.0, min(
            ((100/110) * (1 - r['prob_home']) - r['prob_home']) / (100/110) * 0.5,
            0.10
        )),
        axis=1
    )
    bankroll = 1000.0
    bankroll_series = [bankroll]
    for _, row in df.iterrows():
        bet = bankroll * row['kelly_pct']
        if row['correct']:
            bankroll += bet * (100/110)
        else:
            bankroll -= bet
        bankroll_series.append(bankroll)

    df['bankroll'] = bankroll_series[1:]

    return df, accuracy, bankroll_series


def _render_tab2(model, features, accuracy):
    import plotly.graph_objects as go

    st.markdown("### MLB Backtesting")
    st.markdown(
        f"Model accuracy on holdout (2024–2025): **{accuracy*100:.1f}%**  |  "
        f"Baseline (always home): ~53.4%"
    )

    df_eng, feat_list = load_mlb_historical_features()
    if df_eng.empty:
        st.error("Historical MLB game data not found.")
        return

    available_seasons = sorted(df_eng['season'].unique().tolist(), reverse=True)
    default_seasons   = available_seasons[:5]
    selected = st.multiselect(
        "Seasons to backtest", available_seasons, default=default_seasons,
        key="mlb_bt_seasons"
    )

    if not selected:
        st.info("Select at least one season above.")
        return

    with st.spinner("Running backtest..."):
        df_bt, bt_acc, bankroll_series = _mlb_backtest_results(tuple(sorted(selected)))

    if df_bt is None:
        st.error("Backtest failed — model not loaded.")
        return

    # Summary metrics
    bm1, bm2, bm3, bm4 = st.columns(4)
    bm1.metric("Backtest Accuracy",   f"{bt_acc*100:.1f}%")
    bm2.metric("Games",               f"{len(df_bt):,}")
    bm3.metric("$10/game P&L",        f"${df_bt['cumulative_return'].iloc[-1]:+.0f}")
    bm4.metric("Kelly Final Bankroll", f"${bankroll_series[-1]:,.0f}", f"from $1,000")

    # Accuracy by season
    st.markdown("#### Accuracy by Season")
    season_acc = df_bt.groupby('season')['correct'].mean().reset_index()
    season_acc.columns = ['Season', 'Accuracy']
    season_acc['Accuracy'] = (season_acc['Accuracy'] * 100).round(1)
    st.dataframe(season_acc, use_container_width=True, hide_index=True)

    # Cumulative P&L chart
    st.markdown("#### Cumulative P&L — $10 Flat Bet (Model Side)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(df_bt))),
        y=df_bt['cumulative_return'].tolist(),
        mode='lines',
        name='$10 Flat Bet',
        line=dict(color='#22c55e', width=2),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        xaxis_title="Game #",
        yaxis_title="Cumulative P&L ($)",
        height=350,
        margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

    # Kelly bankroll chart
    st.markdown("#### Kelly Bankroll Growth (Half-Kelly, 10% cap)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=list(range(len(bankroll_series))),
        y=bankroll_series,
        mode='lines',
        name='Half-Kelly Bankroll',
        line=dict(color='#eab308', width=2),
    ))
    fig2.add_hline(y=1000, line_dash="dash", line_color="gray", opacity=0.5)
    fig2.update_layout(
        xaxis_title="Game #",
        yaxis_title="Bankroll ($)",
        height=350,
        margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig2, use_container_width=True)


# ── Tab 3: Track Record ───────────────────────────────────────────────────────

def _render_tab3():
    if not _PH_OK:
        st.info("Prediction history module not available.")
        return
    st.markdown("### MLB Track Record")
    _tr_h, _tr_b, _tr_e = st.tabs(["📊 Prediction History", "💰 Bet Tracker", "📤 Export / Import"])

    with _tr_h:
        try:
            preds = _ph.get_predictions(sport='mlb')
            if not preds:
                st.info("No MLB predictions logged yet. Run predictions in the Game Predictor tab.")
            else:
                df_p = pd.DataFrame(preds)
                st.dataframe(df_p, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load predictions: {e}")

    with _tr_b:
        st.info("Bet Tracker: log your placed bets and track P&L.")
        try:
            bets = _ph.get_bets(sport='mlb')
        except Exception:
            bets = []
        with st.form("mlb_log_bet_form"):
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                bt_team = st.text_input("Team Bet On", key="mlb_bt_team")
            with bc2:
                bt_ml   = st.number_input("Moneyline", value=-110, key="mlb_bt_ml")
            with bc3:
                bt_amt  = st.number_input("Bet ($)", value=10.0, min_value=1.0, key="mlb_bt_amt")
            bt_date = st.date_input("Game Date", key="mlb_bt_date")
            bt_result = st.selectbox("Result", ["Win", "Loss", "Push", "Pending"], key="mlb_bt_result")
            if st.form_submit_button("Log Bet"):
                try:
                    _ph.log_bet({
                        'sport': 'mlb', 'team': bt_team,
                        'moneyline': bt_ml, 'amount': bt_amt,
                        'game_date': str(bt_date), 'result': bt_result,
                    })
                    st.success("Bet logged!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Could not log bet: {ex}")
        if bets:
            df_b = pd.DataFrame(bets)
            st.dataframe(df_b, use_container_width=True)

    with _tr_e:
        st.markdown("##### Export")
        try:
            preds = _ph.get_predictions(sport='mlb')
            bets  = _ph.get_bets(sport='mlb')
            if preds:
                st.download_button(
                    "📥 Download Predictions (JSON)",
                    data=str(preds),
                    file_name=f"mlb_predictions_{date.today()}.json",
                    mime="application/json",
                    key="mlb_dl_preds",
                )
            if bets:
                st.download_button(
                    "📥 Download Bets (JSON)",
                    data=str(bets),
                    file_name=f"mlb_bets_{date.today()}.json",
                    mime="application/json",
                    key="mlb_dl_bets",
                )
            if not preds and not bets:
                st.info("No data to export yet.")
        except Exception as e:
            st.warning(f"Export error: {e}")


# ── Main entry point ──────────────────────────────────────────────────────────

def render_mlb_app():
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚾ MLB Settings")

        st.markdown("### 💰 Bankroll")
        bankroll = st.number_input(
            "Bankroll ($)", min_value=100, max_value=100000,
            value=int(st.session_state.get('mlb_bankroll', 1000)),
            step=100, key="mlb_bankroll_input",
        )
        st.session_state['mlb_bankroll'] = bankroll

        st.markdown("### 🎯 Betting Strategy")
        strategy = st.selectbox(
            "Strategy",
            ["Kelly Criterion", "Fractional Kelly", "Fixed %", "Fixed $"],
            key="mlb_bet_strategy",
        )
        risk_tol = st.selectbox(
            "Risk Tolerance",
            ["Conservative", "Moderate", "Aggressive"],
            index=1, key="mlb_risk_tolerance",
        )
        if strategy == 'Fixed %':
            st.number_input("Fixed bet (%)", min_value=0.5, max_value=10.0,
                            value=2.0, step=0.5, key="mlb_fixed_pct")
        elif strategy == 'Fixed $':
            st.number_input("Fixed bet ($)", min_value=5, max_value=int(bankroll),
                            value=50, step=5, key="mlb_fixed_dollar")

        st.markdown("---")
        if st.button("🏠 Back to Home", key="mlb_back_home"):
            st.session_state['sport'] = None
            st.rerun()

    # Inject badge CSS (same as NFL/NHL)
    st.markdown("""
    <style>
    .signal-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: 600;
        margin: 4px 0;
    }
    .signal-lock   { background-color: #7c3aed; color: white; }
    .signal-strong { background-color: #22c55e; color: white; }
    .signal-lean   { background-color: #eab308; color: black; }
    .signal-pass   { background-color: #94a3b8; color: white; }
    </style>
    """, unsafe_allow_html=True)

    st.title("⚾ MLB Predictor")

    # Load models and data
    model, features, accuracy, elo_ratings = load_mlb_model()
    total_model_pkg  = load_mlb_total_model()
    mlb_games        = load_mlb_games()
    pitcher_ratings  = load_mlb_pitcher_ratings()
    full_pitcher_ratings = load_mlb_full_pitcher_ratings()
    team_stats       = load_mlb_team_stats()

    if model is None:
        st.error("MLB model not found. Run `python build_mlb_model.py` first.")
        return

    # MLB client for live schedule
    mlb_client = None
    try:
        from apis.mlb import MLBClient
        mlb_client = MLBClient()
    except Exception:
        pass

    # Today's sidebar summary
    with st.sidebar:
        st.markdown("### 📊 Today's Summary")
        if _PH_OK:
            try:
                today_preds = [
                    p for p in _ph.get_predictions(sport='mlb')
                    if p.get('game_date') == date.today().isoformat()
                ]
                if today_preds:
                    total_stake   = sum(p.get('bet_amount', 0) for p in today_preds)
                    total_pot_win = sum(
                        p.get('bet_amount', 0) * (100 / 110)
                        for p in today_preds
                    )
                    st.metric("Games Predicted", len(today_preds))
                    st.metric("Total at Stake",  f"${total_stake:.0f}")
                    st.metric("Potential Win",   f"${total_pot_win:.0f}")
                else:
                    st.caption("No MLB predictions logged today.")
            except Exception:
                pass

    tab1, tab2, tab3 = st.tabs(["⚾ Game Predictor", "📅 Backtesting", "📋 Track Record"])

    with tab1:
        _render_tab1(model, features, accuracy, elo_ratings, pitcher_ratings, team_stats,
                     total_model_pkg, mlb_games, full_pitcher_ratings, mlb_client)

    with tab2:
        _render_tab2(model, features, accuracy)

    with tab3:
        _render_tab3()
