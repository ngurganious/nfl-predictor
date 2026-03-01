# ================================================
# NFL PREDICTOR - Final App with Lineup Builder
# Self-healing: rebuilds models if pickle fails
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime, date
warnings.filterwarnings('ignore')

try:
    import prediction_history as _ph
    _PH_OK = True
except ImportError:
    _PH_OK = False

# New weekly-schedule helpers (game_week.py + defensive_matchup.py)
try:
    from game_week import (
        fetch_weekly_schedule, get_team_full_depth_chart,
        get_players_for_position, get_starter_for_position,
        build_lineup_dict, calc_offense_score, get_weather_flag,
        OFF_POSITIONS, DEF_POSITIONS,
    )
    from defensive_matchup import calc_matchup_adj, format_breakdown_table
    _WEEKLY_MODULES_OK = True
except ImportError:
    _WEEKLY_MODULES_OK = False

# Load .env keys (RAPIDAPI_KEY, ODDS_API_KEY) if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Streamlit Cloud: inject st.secrets into os.environ so data_pipeline.py picks them up
try:
    import os
    for _sk, _sv in st.secrets.items():
        os.environ.setdefault(str(_sk), str(_sv))
except Exception:
    pass  # local dev uses .env; ignore if st.secrets not available


NFL_TEAMS = sorted([
    'ARI','ATL','BAL','BUF','CAR','CHI','CIN','CLE',
    'DAL','DEN','DET','GB', 'HOU','IND','JAX','KC',
    'LA', 'LAC','LV', 'MIA','MIN','NE', 'NO', 'NYG',
    'NYJ','PHI','PIT','SEA','SF', 'TB', 'TEN','WAS'
])

# ── Model rebuild function ────────────────────────
def rebuild_models():
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    games = pd.read_csv('games_processed.csv')
    games = games[games['game_type'] == 'REG'].copy()
    games = games.dropna(subset=['home_score','away_score'])
    games = games.sort_values('gameday').reset_index(drop=True)
    games['temp']     = games['temp'].fillna(games['temp'].median())
    games['wind']     = games['wind'].fillna(0)
    games['is_dome']  = (games['roof'] == 'dome').astype(int)
    games['is_grass'] = (games['surface'].str.contains('grass', na=False)).astype(int)
    games['home_win'] = (games['home_score'] > games['away_score']).astype(int)

    feature_cols = ['elo_diff','spread_line','home_rest','away_rest',
                    'temp','wind','is_dome','is_grass','div_game']
    model_data = games[feature_cols + ['home_win']].dropna()
    X = model_data[feature_cols]
    y = model_data['home_win']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False)
    game_model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=42)
    game_model.fit(X_train, y_train)

    # ELO ratings
    elo = {}
    K = 20
    def get_e(t): return elo.get(t, 1500)
    def upd(w, l):
        ew, el = get_e(w), get_e(l)
        ex = 1 / (1 + 10 ** ((el - ew) / 400))
        elo[w] = ew + K * (1 - ex)
        elo[l] = el + K * (0 - (1 - ex))
    for _, g in games.iterrows():
        if g['home_score'] > g['away_score']:   upd(g['home_team'], g['away_team'])
        elif g['away_score'] > g['home_score']: upd(g['away_team'], g['home_team'])

    # Player models
    passing   = pd.read_csv('passing_stats.csv')
    rushing   = pd.read_csv('rushing_stats.csv')
    receiving = pd.read_csv('receiving_stats.csv')

    def train_reg(df, target, features):
        d = df[features + [target]].dropna()
        Xr, yr = d[features], d[target]
        Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42, shuffle=False)
        m = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
        m.fit(Xtr, ytr)
        return m

    pass_model = train_reg(passing,  'pass_yards',
        ['avg_pass_yards_l4','avg_pass_attempts_l4','avg_completions_l4',
         'avg_pass_tds_l4','temp','wind','is_dome','is_home','spread_line'])
    rush_model = train_reg(rushing,  'rush_yards',
        ['avg_rush_yards_l4','avg_rush_attempts_l4','avg_rush_tds_l4',
         'temp','wind','is_dome','is_home','spread_line'])
    rec_model  = train_reg(receiving,'rec_yards',
        ['avg_rec_yards_l4','avg_targets_l4','avg_receptions_l4',
         'avg_rec_tds_l4','temp','wind','is_dome','is_home','spread_line'])

    try:
        with open('player_lookup.pkl','rb') as f:
            players = pickle.load(f)
    except:
        players = {}

    with open('model.pkl','wb') as f:          pickle.dump(game_model, f)
    with open('elo_ratings.pkl','wb') as f:    pickle.dump(elo, f)
    with open('pass_yards_model.pkl','wb') as f: pickle.dump(pass_model, f)
    with open('rush_yards_model.pkl','wb') as f: pickle.dump(rush_model, f)
    with open('rec_yards_model.pkl','wb') as f:  pickle.dump(rec_model, f)

    return game_model, elo, pass_model, rush_model, rec_model, players

# ── Load everything ───────────────────────────────
@st.cache_resource
def load_all():
    try:
        # Try enhanced model first, fall back to original
        try:
            with open('model_enhanced.pkl','rb') as f:
                pkg = pickle.load(f)
            game_model       = pkg['model']
            enhanced_features = pkg['features']
            enhanced_acc     = pkg.get('accuracy', 0)
        except Exception:
            with open('model.pkl','rb') as f:
                game_model = pickle.load(f)
            enhanced_features = None
            enhanced_acc     = None

        with open('elo_ratings.pkl','rb') as f:      elo        = pickle.load(f)

        def _unpack_prop(raw):
            """Handle both old (raw model) and new ({'model':...,'features':...}) formats."""
            if isinstance(raw, dict):
                return raw['model'], raw.get('features', [])
            return raw, []

        with open('pass_yards_model.pkl','rb') as f: _p = pickle.load(f)
        with open('rush_yards_model.pkl','rb') as f: _r = pickle.load(f)
        with open('rec_yards_model.pkl','rb') as f:  _rc = pickle.load(f)
        pass_model, pass_feat = _unpack_prop(_p)
        rush_model, rush_feat = _unpack_prop(_r)
        rec_model,  rec_feat  = _unpack_prop(_rc)
        with open('player_lookup.pkl','rb') as f:    players    = pickle.load(f)
        # Smoke test — use the model's actual feature set (all zeros = neutral)
        _test_feats = enhanced_features if enhanced_features else \
                      ['elo_diff','spread_line','home_rest','away_rest',
                       'temp','wind','is_dome','is_grass','div_game']
        test = pd.DataFrame([{f: 0 for f in _test_feats}])
        game_model.predict_proba(test)
        return (game_model, elo, pass_model, rush_model, rec_model, players,
                enhanced_features, enhanced_acc, pass_feat, rush_feat, rec_feat)
    except Exception as e:
        st.info("Setting up for first launch — building models (takes ~2 minutes)...")
        result = rebuild_models()
        return (result[0], result[1], result[2], result[3], result[4], result[5],
                None, None, [], [], [])


@st.cache_resource
def load_pipeline():
    """Load the data pipeline (wraps all 5 APIs). Cached for the session."""
    try:
        from data_pipeline import DataPipeline
        return DataPipeline()
    except Exception:
        return None


@st.cache_data
def load_team_rolling_stats():
    """Load per-team rolling stats saved by retrain_model.py."""
    try:
        return pd.read_csv('team_rolling_stats.csv', index_col='team')
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_team_stats_current():
    """Load current-season team EPA/scoring stats from team_stats_current.csv."""
    try:
        return pd.read_csv('team_stats_current.csv', index_col='team')
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_total_model():
    """Load over/under regression model saved by build_total_model.py."""
    try:
        with open('model_total.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_data
def load_qb_ratings():
    """Load current-season QB quality ratings saved by build_qb_ratings.py."""
    try:
        return pd.read_csv('qb_team_ratings.csv', index_col='team')
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_def_stats():
    """Load per-team per-season defensive stats saved by build_player_model.py."""
    try:
        dp = pd.read_csv('def_pass_stats.csv')
        dr = pd.read_csv('def_rush_stats.csv')
        # Index by (team, season) for fast lookup
        dp = dp.set_index(['team', 'season'])
        dr = dr.set_index(['team', 'season'])
        return dp, dr
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

def get_opp_pass_defense(team, def_pass_df, current_season=2024):
    """Return opponent pass-defense features; falls back to league average (0.0)."""
    defaults = {'opp_def_epa_per_play': 0.0, 'opp_def_pass_epa': 0.05, 'opp_def_pass_yards': 4.8}
    if def_pass_df.empty:
        return defaults
    for season in [current_season, current_season - 1]:
        try:
            row = def_pass_df.loc[(team, season)]
            return {
                'opp_def_epa_per_play': float(row.get('opp_def_epa_per_play', 0.0)),
                'opp_def_pass_epa':     float(row.get('opp_def_pass_epa', 0.05)),
                'opp_def_pass_yards':   float(row.get('opp_def_pass_yards', 4.8)),
            }
        except KeyError:
            continue
    return defaults

def get_opp_rush_defense(team, def_rush_df, current_season=2024):
    """Return opponent rush-defense features; falls back to league average (0.0)."""
    defaults = {'opp_def_epa_per_play': 0.0, 'opp_def_rush_epa': -0.05, 'opp_def_rush_yards': 4.3}
    if def_rush_df.empty:
        return defaults
    for season in [current_season, current_season - 1]:
        try:
            row = def_rush_df.loc[(team, season)]
            return {
                'opp_def_epa_per_play': float(row.get('opp_def_epa_per_play', 0.0)),
                'opp_def_rush_epa':     float(row.get('opp_def_rush_epa', -0.05)),
                'opp_def_rush_yards':   float(row.get('opp_def_rush_yards', 4.3)),
            }
        except KeyError:
            continue
    return defaults

@st.cache_data
def load_data():
    games     = pd.read_csv('games_processed.csv')
    passing   = pd.read_csv('passing_stats.csv')
    rushing   = pd.read_csv('rushing_stats.csv')
    receiving = pd.read_csv('receiving_stats.csv')
    lineup_df = pd.read_csv('lineup_summary.csv')
    return games, passing, rushing, receiving, lineup_df

@st.cache_data(show_spinner="Computing enhanced features for backtesting (one-time, ~90 sec)...")
def load_enhanced_games():
    """Runs full feature engineering on games_processed.csv for the backtest tab.
    Cached per session — only recomputes on first load."""
    try:
        from feature_engineering import build_enhanced_features
        g = pd.read_csv('games_processed.csv')
        g = g[g['game_type'] == 'REG'].copy()
        g = g.dropna(subset=['home_score', 'away_score'])
        g['is_dome']  = (g['roof'] == 'dome').astype(int)
        g['is_grass'] = g['surface'].str.contains('grass', na=False).astype(int)
        g['home_win'] = (g['home_score'] > g['away_score']).astype(int)
        return build_enhanced_features(g)
    except Exception:
        return pd.DataFrame()

game_model, elo_ratings, pass_model, rush_model, rec_model, player_lookup, \
    enhanced_features, enhanced_acc, pass_feat, rush_feat, rec_feat = load_all()
games, passing, rushing, receiving, lineup_df = load_data()
pipeline        = load_pipeline()
team_roll_stats   = load_team_rolling_stats()
team_stats_curr   = load_team_stats_current()
total_model_pkg   = load_total_model()
qb_ratings      = load_qb_ratings()
def_pass_stats, def_rush_stats = load_def_stats()

# ── Helpers (defined after load so elo_ratings exists) ────────────
def get_elo(team):
    return elo_ratings.get(team, 1500)

def elo_win_prob(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def get_starters(team):
    result = {}
    for pos in ['QB','RB','WR','TE']:
        players = player_lookup.get(team, {}).get(pos, [])
        result[pos]            = players[0]['name']  if players else 'Unknown'
        result[f'{pos}_list']  = [p['name'] for p in players] if players else ['Unknown']
        result[f'{pos}_score'] = players[0]['score'] if players else 50.0
    return result

def calc_lineup_score(team, qb, rb, wr, te):
    def find_score(pos, name):
        for p in player_lookup.get(team, {}).get(pos, []):
            if p['name'] == name:
                return p['score']
        return 50.0
    qb_s = find_score('QB', qb)
    rb_s = find_score('RB', rb)
    wr_s = find_score('WR', wr)
    te_s = find_score('TE', te)
    score = qb_s*0.40 + wr_s*0.25 + rb_s*0.20 + te_s*0.15
    return score, qb_s, rb_s, wr_s, te_s

def lineup_adjustment(home_score, away_score):
    return (home_score - away_score) * 0.005

def get_player_recent(df, name_col, name, stat_cols, n=4):
    rows = df[df[name_col] == name].sort_values(['season','week']).tail(n)
    if len(rows) == 0:
        return None
    result = {}
    for col in stat_cols:
        avg_col = f'avg_{col}_l4'
        if avg_col in rows.columns:
            result[avg_col] = rows[avg_col].iloc[-1]
    return result

def render_nfl_app():
    # ── Back to Home ──────────────────────────────────
    _back_col, _title_col = st.columns([1, 8])
    with _back_col:
        if st.button("🏠 Home", key="nfl_back_home"):
            st.session_state['sport'] = None
            st.rerun()
    with _title_col:
        st.markdown('<div class="edgeiq-logo"><span class="edgeiq-icon">⚡</span> NFL Terminal</div>', unsafe_allow_html=True)
    # ── Header ────────────────────────────────────────
    st.markdown("*25 years of data • ELO ratings • Lineup-adjusted ML predictions • Player props*")
    st.divider()
    
    # ── Sidebar ───────────────────────────────────────
    with st.sidebar:
        # Model version badge
        if enhanced_features:
            st.success(f"Enhanced Model — {enhanced_acc*100:.1f}% accuracy ({len(enhanced_features)} features)")
        else:
            st.info("Original Model — 60-61% accuracy")
    
        # API status
        if pipeline:
            with st.expander("API Status", expanded=False):
                for api, status in pipeline.get_api_status().items():
                    st.caption(f"{api}: {status}")
    
        st.header("ELO Power Rankings")
        elo_df = pd.Series(elo_ratings).reset_index()
        elo_df.columns = ['Team','ELO']
        elo_df = elo_df[elo_df['Team'].isin(NFL_TEAMS)]\
            .sort_values('ELO', ascending=False).reset_index(drop=True)
        elo_df.index += 1
        elo_df['ELO'] = elo_df['ELO'].round(0).astype(int)
        st.dataframe(elo_df, use_container_width=True)
    
    # ══════════════════════════════════════════════════
    # KELLY CRITERION BET-SIZING HELPERS
    # ══════════════════════════════════════════════════
    
    def _kelly_rec(model_prob: float, moneyline_odds: float, fraction: float = 0.5):
        """
        Half-Kelly bet-sizing recommendation.
    
        Args:
            model_prob:     model's win probability for the picked team (0–1)
            moneyline_odds: American odds (negative = favorite, positive = underdog)
            fraction:       Kelly fraction to apply (0.5 = Half-Kelly, default)
    
        Returns:
            (kelly_pct, tier, badge_str)
            kelly_pct  — % of bankroll to bet (0–10)
            tier       — 'signal-strong' | 'signal-lean' | 'signal-pass'
            badge_str  — Text label
        """
        if moneyline_odds is None or moneyline_odds == 0 or model_prob is None:
            return 0.0, 'signal-pass', 'PASS'
        try:
            b = (100.0 / abs(moneyline_odds)) if moneyline_odds < 0 else (moneyline_odds / 100.0)
            q = 1.0 - float(model_prob)
            full_kelly = (b * float(model_prob) - q) / b
            kelly = max(0.0, min(full_kelly * fraction, 0.10))  # cap at 10% bankroll
            pct = kelly * 100.0
        except Exception:
            return 0.0, 'signal-pass', 'PASS'
        if pct >= 4.0:
            return pct, 'signal-strong', 'STRONG EDGE'
        if pct >= 2.0:
            return pct, 'signal-lean',   'LEAN'
        if pct >= 1.0:
            return pct, 'signal-lean',   'SMALL EDGE'
        return pct, 'signal-pass', 'PASS'
    
    
    def _spread_to_ml(spread_for_team: float) -> float:
        """
        Approximate American moneyline from a point spread (team's perspective).
    
        Spread is negative when the team is favored (e.g. -7 = 7-point fav).
        Returns negative odds for favorites, positive for underdogs.
        """
        import math
        try:
            p = 1.0 / (1.0 + math.exp(float(spread_for_team) * 0.0541))
            if p > 0.5:
                return -(100.0 * p / (1.0 - p))
            return 100.0 * (1.0 - p) / p
        except Exception:
            return -110.0  # default even-ish line
    
    
    # ══════════════════════════════════════════════════
    # PARLAY LADDER HELPERS
    # ══════════════════════════════════════════════════

    def _full_to_abbr(full_name):
        parts = full_name.strip().split()
        if len(parts) >= 2:
            return f"{parts[0][0]}.{parts[-1]}"
        return full_name

    def _prop_confidence(prediction, vegas_line, mae):
        from scipy.stats import norm
        if vegas_line is None or vegas_line <= 0 or mae <= 0:
            return 0.5, 'OVER' if prediction > 0 else 'UNDER', 0.0, 0.5
        sigma = mae * 1.253
        edge = prediction - vegas_line
        z = edge / sigma
        over_prob = norm.cdf(z)
        under_prob = 1.0 - over_prob
        if over_prob >= under_prob:
            return over_prob, 'OVER', edge, over_prob
        return under_prob, 'UNDER', edge, over_prob

    def _rpl_add_selection(leg_dict):
        if 'rpl_selections' not in st.session_state:
            st.session_state['rpl_selections'] = {}
        st.session_state['rpl_selections'][leg_dict['leg_id']] = leg_dict

    def _rpl_remove_selection(leg_id):
        sels = st.session_state.get('rpl_selections', {})
        sels.pop(leg_id, None)

    # ══════════════════════════════════════════════════
    # WEEKLY SCHEDULE HELPERS  (used by Tab 1)
    # ══════════════════════════════════════════════════
    
    def _get_depth_chart(team: str) -> dict:
        """
        Return a full depth chart for *team*, cached in session state for 6 hours.
        Falls back to player_lookup only if all API calls fail.
        """
        cache_key = f'dc_{team}'
        cached = st.session_state.get(cache_key)
        if cached is not None:
            return cached
    
        chart = {}
        if _WEEKLY_MODULES_OK and pipeline:
            chart = get_team_full_depth_chart(
                team,
                pipeline.espn,
                getattr(pipeline, 'tank01', None),
                player_lookup,
            )
        elif _WEEKLY_MODULES_OK:
            # No pipeline — try ESPN only (no key needed)
            try:
                from apis.espn import ESPNClient
                _espn = ESPNClient()
                chart = get_team_full_depth_chart(team, _espn, None, player_lookup)
            except Exception:
                pass
    
        # If still empty, synthesise from player_lookup
        if not chart:
            for pos in ['QB', 'RB', 'WR', 'TE']:
                known = player_lookup.get(team, {}).get(pos, [])
                chart[pos] = [{'name': p['name'], 'score': p.get('score', 55.0),
                                'depth': i + 1, 'position': pos}
                               for i, p in enumerate(known)]
    
        st.session_state[cache_key] = chart
        return chart
    
    
    def _depth_list(depth_chart: dict, pos_label: str) -> list:
        """Wrapper around get_players_for_position with graceful fallback."""
        if _WEEKLY_MODULES_OK:
            try:
                return get_players_for_position(depth_chart, pos_label)
            except Exception:
                pass
        return ['Unknown']
    
    
    def run_game_prediction(
        home_team: str, away_team: str,
        conditions: dict,
        home_off: dict, away_off: dict,
        home_def_chart: dict, away_def_chart: dict,
        live_ctx: dict = None,
    ) -> dict:
        """
        Run the full prediction pipeline for one game.
    
        Args:
            home_team / away_team  — 3-letter abbrevs
            conditions  — {spread, total_ou, temp, wind, is_dome, is_grass,
                           home_rest, away_rest, is_div}
            home_off    — lineup dict from build_lineup_dict() for home offense
            away_off    — lineup dict from build_lineup_dict() for away offense
            home_def_chart / away_def_chart — depth chart dicts for each team's defense
            live_ctx    — optional pipeline context from Fetch Live Data
    
        Returns a dict with all result fields needed by render_prediction_result().
        """
        from feature_engineering import _spread_to_prob
    
        home_elo = get_elo(home_team)
        away_elo = get_elo(away_team)
        elo_diff = home_elo - away_elo
    
        # Resolve effective conditions (live API overrides manual sliders)
        _live_wx   = (live_ctx or {}).get('weather', {})
        _live_odds = (live_ctx or {}).get('odds', {})
        _live_inj  = (live_ctx or {}).get('injuries', {})
    
        eff_temp    = _live_wx.get('temp',    conditions['temp'])    if _live_wx else conditions['temp']
        eff_wind    = _live_wx.get('wind',    conditions['wind'])    if _live_wx else conditions['wind']
        eff_is_dome = _live_wx.get('is_dome', conditions['is_dome']) if _live_wx else conditions['is_dome']
        eff_spread  = _live_odds.get('spread_home', conditions['spread']) \
                      if _live_odds.get('spread_home') is not None else conditions['spread']
    
        base_fv = {
            'elo_diff':    elo_diff,
            'spread_line': eff_spread,
            'home_rest':   conditions['home_rest'],
            'away_rest':   conditions['away_rest'],
            'temp':        eff_temp,
            'wind':        eff_wind,
            'is_dome':     eff_is_dome,
            'is_grass':    conditions['is_grass'],
            'div_game':    int(conditions.get('is_div', 0)),
        }
    
        ext_fv = {}
        if enhanced_features:
            def _roll(team, col):
                if team in team_roll_stats.index:
                    v = team_roll_stats.loc[team, col] if col in team_roll_stats.columns else np.nan
                    return float(v) if pd.notna(v) else np.nan
                return np.nan
    
            def _epa(team, col):
                if not team_stats_curr.empty and team in team_stats_curr.index:
                    v = team_stats_curr.loc[team, col] if col in team_stats_curr.columns else np.nan
                    return float(v) if pd.notna(v) else 0.0
                return 0.0
    
            def _qb(team):
                if not qb_ratings.empty and team in qb_ratings.index:
                    v = qb_ratings.loc[team, 'qb_score'] if 'qb_score' in qb_ratings.columns else np.nan
                    return float(v) if pd.notna(v) else 0.0
                return 0.0
    
            h_l5_diff = _roll(home_team, 'l5_pts_diff')
            a_l5_diff = _roll(away_team, 'l5_pts_diff')
            h_l5_for  = _roll(home_team, 'l5_pts_for')
            a_l5_for  = _roll(away_team, 'l5_pts_for')
            h_l5_ag   = _roll(home_team, 'l5_pts_against')
            a_l5_ag   = _roll(away_team, 'l5_pts_against')
            h_elo_tr  = _roll(home_team, 'elo_trend')
            a_elo_tr  = _roll(away_team, 'elo_trend')
            h_off_epa = _epa(home_team, 'off_epa_per_play')
            h_def_epa = _epa(home_team, 'def_epa_per_play')
            a_off_epa = _epa(away_team, 'off_epa_per_play')
            a_def_epa = _epa(away_team, 'def_epa_per_play')
            h_qb = _qb(home_team)
            a_qb = _qb(away_team)
    
            pts_diff_adv = (h_l5_diff - a_l5_diff) if pd.notna(h_l5_diff) and pd.notna(a_l5_diff) else np.nan
            matchup_h    = (h_l5_for - a_l5_ag)    if pd.notna(h_l5_for)  and pd.notna(a_l5_ag)   else np.nan
            matchup_a    = (a_l5_for - h_l5_ag)    if pd.notna(a_l5_for)  and pd.notna(h_l5_ag)   else np.nan
            net_matchup  = (matchup_h - matchup_a)  if pd.notna(matchup_h) and pd.notna(matchup_a)  else np.nan
            epa_off_diff = h_off_epa - a_off_epa
            epa_def_diff = a_def_epa - h_def_epa
            epa_total    = epa_off_diff + epa_def_diff
    
            ext_fv = {
                'home_l5_win_pct':     _roll(home_team, 'l5_win_pct'),
                'away_l5_win_pct':     _roll(away_team, 'l5_win_pct'),
                'home_l5_pts_diff':    h_l5_diff,
                'away_l5_pts_diff':    a_l5_diff,
                'pts_diff_advantage':  pts_diff_adv,
                'home_l5_pts_for':     h_l5_for,
                'away_l5_pts_for':     a_l5_for,
                'home_l5_pts_against': h_l5_ag,
                'away_l5_pts_against': a_l5_ag,
                'matchup_adv_home':    matchup_h,
                'matchup_adv_away':    matchup_a,
                'net_matchup_adv':     net_matchup,
                'home_elo_trend':      h_elo_tr,
                'away_elo_trend':      a_elo_tr,
                'rest_advantage':      conditions['home_rest'] - conditions['away_rest'],
                'spread_implied_prob': _spread_to_prob(eff_spread),
                'elo_implied_prob':    1 / (1 + 10 ** (-elo_diff / 400)),
                'qb_score_diff':       h_qb - a_qb,
                'home_off_epa':        h_off_epa,
                'away_off_epa':        a_off_epa,
                'home_def_epa':        h_def_epa,
                'away_def_epa':        a_def_epa,
                'epa_off_diff':        epa_off_diff,
                'epa_def_diff':        epa_def_diff,
                'epa_total_diff':      epa_total,
                'net_injury_adj':      _live_inj.get('away_impact', 0) - _live_inj.get('home_impact', 0),
            }
    
        combined_fv = {**base_fv, **ext_fv}
        if enhanced_features:
            feat_row = {col: combined_fv.get(col, np.nan) for col in enhanced_features}
            features = pd.DataFrame([feat_row])
        else:
            features = pd.DataFrame([base_fv])
    
        base_prob_home = game_model.predict_proba(features)[0][1]
    
        # --- Lineup adjustment (existing logic, using extended offense score) ---
        if _WEEKLY_MODULES_OK:
            home_off_score = calc_offense_score(home_off)
            away_off_score = calc_offense_score(away_off)
        else:
            # Fallback to 4-position score
            home_qb_s = home_off.get('qb_score', 65.0)
            home_rb_s = home_off.get('rb1_score', 65.0)
            home_wr_s = home_off.get('wr1_score', 65.0)
            home_te_s = home_off.get('te1_score', 60.0)
            away_qb_s = away_off.get('qb_score', 65.0)
            away_rb_s = away_off.get('rb1_score', 65.0)
            away_wr_s = away_off.get('wr1_score', 65.0)
            away_te_s = away_off.get('te1_score', 60.0)
            home_off_score = home_qb_s * 0.40 + home_wr_s * 0.25 + home_rb_s * 0.20 + home_te_s * 0.15
            away_off_score = away_qb_s * 0.40 + away_wr_s * 0.25 + away_rb_s * 0.20 + away_te_s * 0.15
    
        _h_impact = _live_inj.get('home_impact', 0)
        _a_impact = _live_inj.get('away_impact', 0)
        home_off_adj = home_off_score * (1 - _h_impact * 0.5)
        away_off_adj = away_off_score * (1 - _a_impact * 0.5)
        lineup_adj = (home_off_adj - away_off_adj) * 0.005
    
        # --- Defensive matchup adjustment (new) ---
        matchup_adj_home = 0.0
        matchup_adj_away = 0.0
        home_matchup_breakdown = []
        away_matchup_breakdown = []
        if _WEEKLY_MODULES_OK and away_def_chart and home_def_chart:
            try:
                matchup_adj_home, home_matchup_breakdown = calc_matchup_adj(
                    home_off, away_def_chart, away_team, team_stats_curr)
                matchup_adj_away, away_matchup_breakdown = calc_matchup_adj(
                    away_off, home_def_chart, home_team, team_stats_curr)
            except Exception:
                pass
    
        net_matchup_adj = matchup_adj_home - matchup_adj_away
    
        final_prob_home = float(np.clip(
            base_prob_home + lineup_adj + net_matchup_adj, 0.05, 0.95))
        final_prob_away = 1.0 - final_prob_home
        elo_prob = elo_win_prob(home_elo, away_elo)
    
        # --- Over/Under prediction ---
        # The model predicts the *residual* (actual_total - total_line).
        # All EPA features must use correct values (near 0.0), NOT the pts-scale _avg=22.
        ou_result = None
        ou_total_line = float((_live_odds.get('total') or conditions.get('total_ou', 44.5)))
        if total_model_pkg:
            _avg = 22.0
            _h_diff = (ext_fv.get('home_l5_pts_diff', 0) if ext_fv else 0) or 0
            _a_diff = (ext_fv.get('away_l5_pts_diff', 0) if ext_fv else 0) or 0
            _h_for  = _avg + _h_diff / 2;  _h_ag = _avg - _h_diff / 2
            _a_for  = _avg + _a_diff / 2;  _a_ag = _avg - _a_diff / 2
            # Pull EPA values that are already computed in ext_fv (0.0 if not available)
            _h_oe = float((ext_fv or {}).get('home_off_epa', 0.0) or 0.0)
            _a_oe = float((ext_fv or {}).get('away_off_epa', 0.0) or 0.0)
            _h_de = float((ext_fv or {}).get('home_def_epa', 0.0) or 0.0)
            _a_de = float((ext_fv or {}).get('away_def_epa', 0.0) or 0.0)
            _qb_diff = float((ext_fv or {}).get('qb_score_diff', 0.0) or 0.0)
            _sip  = float((ext_fv or {}).get('spread_implied_prob', 0.5) or 0.5)
            # Features whose correct scale is ~0, not ~22 — must NOT fall back to _avg
            _epa_zero_keys = {
                'home_off_epa', 'away_off_epa', 'home_def_epa', 'away_def_epa',
                'scoring_env_off', 'scoring_env_def', 'qb_score_diff', 'elo_diff',
                'div_game', 'spread_implied_prob',
            }
            ou_fv = {
                # Scoring trends
                'total_line':          ou_total_line,
                'home_l5_pts_for':     _h_for,   'away_l5_pts_for':     _a_for,
                'home_l5_pts_against': _h_ag,    'away_l5_pts_against': _a_ag,
                'home_l5_pts_diff':    _h_diff,  'away_l5_pts_diff':    _a_diff,
                'matchup_adv_home':    _h_for - _a_ag,
                'matchup_adv_away':    _a_for - _h_ag,
                # EPA features — correct per-play scale (~0.0 to 0.20)
                'home_off_epa':        _h_oe,    'away_off_epa':        _a_oe,
                'home_def_epa':        _h_de,    'away_def_epa':        _a_de,
                'scoring_env_off':     _h_oe + _a_oe,
                'scoring_env_def':     _h_de + _a_de,
                # QB / team quality
                'qb_score_diff':       _qb_diff,
                'elo_diff':            elo_diff,
                'abs_spread':          abs(eff_spread),
                # Game context
                'div_game':            int(conditions.get('is_div', 0)),
                'spread_implied_prob': _sip,
                # Weather / rest
                'wind': eff_wind, 'temp': eff_temp, 'is_dome': eff_is_dome,
                'home_rest': conditions['home_rest'], 'away_rest': conditions['away_rest'],
            }
            ou_feats = total_model_pkg['features']
            ou_row   = {f: ou_fv.get(f, 0.0 if f in _epa_zero_keys else _avg)
                        for f in ou_feats}
            # Model returns residual; convert to actual predicted total
            pred_residual = float(total_model_pkg['model'].predict(pd.DataFrame([ou_row]))[0])
            pred_total    = ou_total_line + pred_residual
            ou_result = {
                'pred_total': pred_total,
                'line':       ou_total_line,
                'edge':       pred_residual,   # residual = edge from Vegas line
                'mae':        total_model_pkg.get('mae', 10.2),
            }
    
        return {
            'base_prob_home':        base_prob_home,
            'final_prob_home':       final_prob_home,
            'final_prob_away':       final_prob_away,
            'elo_prob':              elo_prob,
            'lineup_adj':            lineup_adj,
            'net_matchup_adj':       net_matchup_adj,
            'home_off_score':        home_off_adj,
            'away_off_score':        away_off_adj,
            'home_matchup_breakdown': home_matchup_breakdown,
            'away_matchup_breakdown': away_matchup_breakdown,
            'ou_result':             ou_result,
            'eff_spread':            eff_spread,
            'eff_wind':              eff_wind,
            'eff_temp':              eff_temp,
            'eff_is_dome':           eff_is_dome,
            'live_odds':             _live_odds,
            'live_inj':              _live_inj,
            'data_source':           'Live API' if _live_wx else 'Manual',
        }
    
    
    def render_prediction_result(result: dict, home_team: str, away_team: str, pfx: str, game_date: str = None):
        """Render win-probability metrics, matchup breakdown, and O/U for one game."""
        fph  = result['final_prob_home']
        fpaw = result['final_prob_away']
        bph  = result['base_prob_home']
        la   = result['lineup_adj']
        ma   = result['net_matchup_adj']
    
        st.subheader("Prediction")
        src_label = result['data_source']
        if enhanced_features:
            src_label += f" | Enhanced model ({enhanced_acc*100:.1f}%)"
        st.caption(src_label)
    
        r1, r2 = st.columns(2)
        with r1:
            st.metric(f"{home_team} Win Prob", f"{fph*100:.1f}%",
                      delta=f"{(fph-bph)*100:+.1f}% adj")
            st.progress(fph)
        with r2:
            st.metric(f"{away_team} Win Prob", f"{fpaw*100:.1f}%",
                      delta=f"ELO base: {(1-result['elo_prob'])*100:.1f}%")
            st.progress(fpaw)
    
        winner     = home_team if fph > 0.5 else away_team
        confidence = max(fph, fpaw)
        if confidence > 0.75:
            label, _css_class = "LOCK", "signal-lock"
        elif confidence > 0.70:
            label, _css_class = "HIGH CONFIDENCE", "signal-strong"
        elif confidence > 0.60:
            label, _css_class = "MODERATE", "signal-lean"
        else:
            label, _css_class = "TOSS-UP", "signal-pass"
        st.markdown(
            f'<div class="signal-badge {_css_class}">{label}: {winner}</div>',
            unsafe_allow_html=True)
    
        _lo = result.get('live_odds', {})
        if _lo.get('formatted') and _lo['formatted'] != 'Lines not available':
            vp = _lo.get('vegas_ml_prob')
            vstr = f"Vegas: {_lo['formatted']}"
            if vp:
                vstr += f"  |  Market: {vp*100:.1f}% {home_team}"
            st.info(vstr)
    
        # ── Kelly Bet Sizing ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📐 Kelly Bet Sizing")
        _bankroll_val  = int(st.session_state.get('bankroll', 1000))
        _strategy_k    = st.session_state.get('nfl_bet_strategy', 'Kelly Criterion')
        _risk_tol_k    = st.session_state.get('nfl_risk_tolerance', 'Moderate')
        _kelly_frac_k  = {'Conservative': 0.25, 'Moderate': 0.5, 'Aggressive': 1.0}[_risk_tol_k]
        _fixed_pct_k   = float(st.session_state.get('nfl_fixed_pct', 2.0))
        _fixed_dol_k   = int(st.session_state.get('nfl_fixed_dollar', 50))
        _pick_home_k   = fph >= 0.5
        _pick_prob_k   = fph if _pick_home_k else fpaw
        _pick_label_k  = home_team if _pick_home_k else away_team
        # Moneyline: prefer live odds, derive from spread if absent
        _odds_k    = result.get('live_odds') or {}
        _pick_ml_k = _odds_k.get('home_ml') if _pick_home_k else _odds_k.get('away_ml')
        if _pick_ml_k is None:
            _sprd_k    = result.get('spread_line', 0) or 0
            _pick_ml_k = _spread_to_ml(float(_sprd_k) if _pick_home_k else -float(_sprd_k))
        _kpct_k, _ktier_k, _kbadge_k = _kelly_rec(_pick_prob_k, _pick_ml_k, fraction=_kelly_frac_k)
        _vegas_impl_k = (abs(_pick_ml_k) / (abs(_pick_ml_k) + 100)
                         if _pick_ml_k < 0 else 100 / (_pick_ml_k + 100))
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
        kc1.metric("Model Edge", f"{_edge_k:+.1f}%",
                   help="Model win prob minus Vegas implied prob for the predicted winner")
        kc2.metric(_pct_label_k, _pct_val_k,
                   help="Bet size based on selected strategy — adjust in sidebar")
        kc3.metric("Bet Amount", f"${_bet_amt_k:.0f}",
                   help=f"Of your ${_bankroll_val:,} bankroll — adjust bankroll in sidebar")
        kc4.markdown(
            f'<p style="font-size:0.8em;color:gray;margin-bottom:4px">Signal</p>'
            f'<span class="signal-badge {_ktier_k}">{_kbadge_k}</span>',
            unsafe_allow_html=True)
        if _strategy_k == 'Fixed %':
            _caption_extra = f"Fixed {_fixed_pct_k:.1f}% per game regardless of edge."
        elif _strategy_k == 'Fixed $':
            _caption_extra = f"Fixed ${_fixed_dol_k} per game regardless of edge."
        else:
            _caption_extra = f"{_risk_tol_k} Kelly ({_kelly_frac_k:.2g}×). Kelly caps at 10% of bankroll to limit volatility."
        st.caption(
            f"Betting on **{_pick_label_k}** at {_pick_prob_k*100:.1f}% model confidence. "
            f"Vegas implied: {_vegas_impl_k*100:.1f}%. {_caption_extra}"
        )

        # ── Log prediction to history (once per session per game) ────────────
        if _PH_OK:
            _log_key = f'{pfx}_logged'
            if not st.session_state.get(_log_key, False):
                try:
                    _ou_res = result.get('ou_result')
                    _gdate = game_date or date.today().isoformat()
                    _rec = {
                        'id': f"nfl_{home_team}_{away_team}_{_gdate}",
                        'sport': 'nfl',
                        'game_date': _gdate,
                        'home_team': home_team,
                        'away_team': away_team,
                        'predicted_at': datetime.now().isoformat(),
                        'model_home_prob': round(float(fph), 4),
                        'vegas_ml_home': _odds_k.get('home_ml'),
                        'vegas_implied_prob': round(float(_vegas_impl_k), 4),
                        'model_edge_pct': round(float(_edge_k), 2),
                        'kelly_signal': _ktier_k,
                        'kelly_pct': round(float(_kpct_k), 2),
                        'ou_line': float(_ou_res['line']) if _ou_res else None,
                        'model_total': float(_ou_res['pred_total']) if _ou_res else None,
                        'ou_lean': ('OVER' if _ou_res['edge'] > 0 else 'UNDER') if _ou_res else None,
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

        with st.expander("Lineup + Matchup Breakdown"):
            lc1, lc2 = st.columns(2)
            with lc1:
                st.markdown(f"**{home_team} Offense: {result['home_off_score']:.1f}/100**")
                _li = result.get('live_inj', {})
                if _li.get('home_impact', 0) > 0:
                    st.caption(f"Injury adj: -{_li['home_impact']:.0%}")
            with lc2:
                st.markdown(f"**{away_team} Offense: {result['away_off_score']:.1f}/100**")
                if _li.get('away_impact', 0) > 0:
                    st.caption(f"Injury adj: -{_li['away_impact']:.0%}")
    
            st.caption(f"Lineup adj: {la*100:+.2f}%  |  Matchup adj: {ma*100:+.2f}%  |  "
                       f"Base: {bph*100:.1f}%  |  Final: {fph*100:.1f}%")
    
            if result.get('home_matchup_breakdown') and _WEEKLY_MODULES_OK:
                st.divider()
                st.markdown(f"**{home_team} Offense vs {away_team} Defense**")
                df_h = format_breakdown_table(result['home_matchup_breakdown'])
                st.dataframe(df_h, use_container_width=True, hide_index=True)
            if result.get('away_matchup_breakdown') and _WEEKLY_MODULES_OK:
                st.markdown(f"**{away_team} Offense vs {home_team} Defense**")
                df_a = format_breakdown_table(result['away_matchup_breakdown'])
                st.dataframe(df_a, use_container_width=True, hide_index=True)
    
        # --- Over/Under ---
        ou = result.get('ou_result')
        if ou:
            st.divider()
            st.subheader("Over/Under")
            oc1, oc2, oc3 = st.columns(3)
            edge = ou['edge']
            lean = "OVER" if edge > 0 else "UNDER"
            oc1.metric("Model Total", f"{ou['pred_total']:.1f} pts")
            oc2.metric("Vegas Line",  f"{ou['line']:.1f} pts")
            oc3.metric("Lean", lean,  delta=f"{abs(edge):.1f} pts {lean}")
            if abs(edge) >= 4:
                st.success(f"Strong {lean} — {abs(edge):.1f} pts edge (MAE {ou['mae']:.1f} pts)")
            elif abs(edge) >= 2:
                st.info(f"Moderate {lean} — {abs(edge):.1f} pts edge")
            else:
                st.caption(f"Slight {lean} by {abs(edge):.1f} pts — within model uncertainty")
    
        # Conditions warnings
        ew = result.get('eff_wind', 0)
        et = result.get('eff_temp', 65)
        ed = result.get('eff_is_dome', 0)
        if ew >= 20:
            st.warning(f"High wind ({ew} mph) — expect lower scoring / passing suppressed")
        if et < 32:
            st.warning(f"Freezing ({et}°F) — typically reduces total scoring ~4 pts")
        if ed:
            st.info("Dome game — weather neutralised, passing typically boosted")
    
    
    def _render_game_expander(game_info: dict, game_idx: int):
        """
        Render a single collapsible game card inside its day section.
    
        Collapsed label shows: AWAY @ HOME | spread | O/U | time | weather flag
        Expanded shows: conditions, lineups (full offense + defense), predict button.
        """
        pfx        = f'g{game_idx}_'
        home_team  = game_info.get('home_team', 'HOME')
        away_team  = game_info.get('away_team', 'AWAY')
        game_time  = game_info.get('game_time_et', 'TBD')
        venue      = game_info.get('venue', '')

        # Extract ISO game date for prediction logging
        _dt_et = game_info.get('datetime_et')
        try:
            _game_date = _dt_et.date().isoformat() if _dt_et else date.today().isoformat()
        except Exception:
            _game_date = date.today().isoformat()
    
        # Pull cached conditions (spread/O/U/weather) for the collapsed label
        cached_cond = st.session_state.get(f'{pfx}cond', {})
        spread_disp = cached_cond.get('spread', '?')
        ou_disp     = cached_cond.get('total_ou', '?')
        temp_disp   = cached_cond.get('temp', None)
        wind_disp   = cached_cond.get('wind', None)
        dome_disp   = cached_cond.get('is_dome', 0)
        wx_flag     = (get_weather_flag(temp_disp, wind_disp, dome_disp)
                       if _WEEKLY_MODULES_OK else None)
    
        # Build collapsed label
        spread_str = f"Sprd {spread_disp:+.1f}" if isinstance(spread_disp, (int, float)) else "Sprd —"
        ou_str     = f"O/U {ou_disp}" if ou_disp != '?' else "O/U —"
        label_parts = [f"{away_team} @ {home_team}", spread_str, ou_str, game_time]
        if wx_flag:
            label_parts.append(f"⚠️ {wx_flag}")
    
        # Append winner + win% + Kelly badge if a prediction has already been run
        _pred_for_badge = st.session_state.get(f'{pfx}pred')
        if _pred_for_badge:
            _badge_prob_home = float(_pred_for_badge.get('final_prob_home', 0.5))
            _badge_pick_home = _badge_prob_home >= 0.5
            _badge_p  = _badge_prob_home if _badge_pick_home else (1.0 - _badge_prob_home)
            _badge_winner = home_team if _badge_pick_home else away_team
            label_parts.append(f"🏈 {_badge_winner} {_badge_p*100:.0f}%")
            # Try live moneyline first, then derive from spread
            _live_ctx_b = st.session_state.get(f'{pfx}live_ctx') or {}
            _odds_b     = _live_ctx_b.get('odds') or {}
            _badge_ml   = _odds_b.get('home_ml') if _badge_pick_home else _odds_b.get('away_ml')
            if _badge_ml is None:
                _sprd_b   = spread_disp if isinstance(spread_disp, (int, float)) else 0.0
                _badge_ml = _spread_to_ml(_sprd_b if _badge_pick_home else -_sprd_b)
            _bkpct, _bktier, _bkbadge = _kelly_rec(_badge_p, _badge_ml)
            if _bktier != 'signal-pass':
                label_parts.append(f"💎 {_bkpct:.1f}%")
            else:
                label_parts.append("⚪ PASS")
    
        expander_label = "  |  ".join(label_parts)
    
        # Keep expander open once any button inside has been clicked
        _stay_open = st.session_state.get(f'{pfx}stay_open', False)
        with st.expander(expander_label, expanded=_stay_open):
    
            # ── Conditions ──────────────────────────────────────────────────────
            st.markdown("##### Game Conditions")
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.caption(f"Venue: {venue[:35] or 'TBD'}")
                home_rest = st.slider("Home Rest (days)", 3, 14, 7, key=f'{pfx}hr')
                away_rest = st.slider("Away Rest (days)", 3, 14, 7, key=f'{pfx}ar')
                is_div    = st.checkbox("Divisional game?", key=f'{pfx}div')
            with cc2:
                roof    = st.selectbox("Stadium", ['outdoors','dome','closed','open'],
                                       key=f'{pfx}roof')
                is_dome = 1 if roof == 'dome' else 0
                outdoor = roof in ['outdoors', 'open']
                temp    = st.slider("Temp (°F)", 20, 100, 65, key=f'{pfx}tmp') if outdoor else 72
                wind    = st.slider("Wind (mph)", 0, 40, 5, key=f'{pfx}wnd')   if outdoor else 0
                surface = st.selectbox("Surface", ['grass', 'turf'], key=f'{pfx}srf')
                is_grass= 1 if surface == 'grass' else 0
            with cc3:
                spread   = st.slider("Vegas Spread (neg = home fav)", -28.0, 28.0, -3.0, 0.5,
                                      key=f'{pfx}sprd')
                total_ou = st.number_input("Vegas O/U", min_value=20.0, max_value=80.0,
                                            value=44.5, step=0.5, key=f'{pfx}ou')
    
            # Save current conditions so collapsed label updates
            st.session_state[f'{pfx}cond'] = {
                'spread': spread, 'total_ou': total_ou,
                'temp': temp, 'wind': wind, 'is_dome': is_dome,
            }
    
            conditions = {
                'spread': spread, 'total_ou': total_ou,
                'temp': temp, 'wind': wind, 'is_dome': is_dome,
                'is_grass': is_grass, 'home_rest': home_rest,
                'away_rest': away_rest, 'is_div': int(is_div),
            }
    
            # ── Live Data fetch ──────────────────────────────────────────────────
            st.divider()
            ld1, ld2, ld3 = st.columns([2, 2, 1])
            with ld1:
                gdate = st.date_input("Game Date", value=date.today(), key=f'{pfx}gdate')
            with ld2:
                gtime_sel = st.selectbox("Kickoff (ET)",
                    ["13:00","16:05","16:25","20:15","20:20","21:15"], key=f'{pfx}gtime')
            with ld3:
                st.write(""); st.write("")
                fetch_btn = st.button("Fetch Live Data", key=f'{pfx}fetch',
                                       use_container_width=True)
    
            live_ctx_key = f'{pfx}live_ctx'
            if fetch_btn:
                st.session_state[f'{pfx}stay_open'] = True  # pin expander open
                if pipeline:
                    with st.spinner("Fetching live data..."):
                        _ctx = pipeline.get_game_context(
                            home_team, away_team,
                            f"{gdate}T{gtime_sel}",
                            elo_diff=get_elo(home_team) - get_elo(away_team),
                            home_rest=home_rest, away_rest=away_rest,
                            div_game=int(is_div), surface=surface,
                        )
                        st.session_state[live_ctx_key] = _ctx
                st.rerun()  # re-render with expander kept open
    
            live_ctx = st.session_state.get(live_ctx_key, {})
            if live_ctx:
                _wx  = live_ctx.get('weather', {})
                _ods = live_ctx.get('odds', {})
                _inj = live_ctx.get('injuries', {})
                _api = live_ctx.get('api_status', {})
                i1, i2, i3 = st.columns(3)
                with i1:
                    if _wx:
                        dl = "DOME" if _wx.get('is_dome') else "Outdoor"
                        st.info(f"{_wx.get('stadium','')[:20]}: {_wx.get('temp',72)}F | "
                                f"{_wx.get('wind',0)} mph | {dl}")
                with i2:
                    fmt = _ods.get('formatted', '')
                    if fmt and fmt != 'Lines not available':
                        st.info(f"Vegas: {fmt}")
                    elif not _api.get('odds'):
                        st.caption("Vegas: add ODDS_API_KEY")
                with i3:
                    h_imp = _inj.get('home_impact', 0)
                    a_imp = _inj.get('away_impact', 0)
                    if h_imp > 0.10 or a_imp > 0.10:
                        st.warning(f"Injuries: {home_team} {h_imp:.0%} | {away_team} {a_imp:.0%}")
                    else:
                        st.success("No significant injuries")
    
                h_out = {n: s for n, s in _inj.get('home_flags', {}).items()
                         if s in ('OUT','IR','PUP','DNR','DOUBTFUL')}
                a_out = {n: s for n, s in _inj.get('away_flags', {}).items()
                         if s in ('OUT','IR','PUP','DNR','DOUBTFUL')}
                if h_out or a_out:
                    with st.expander("Injury Detail"):
                        ic1, ic2 = st.columns(2)
                        with ic1:
                            st.markdown(f"**{home_team}**")
                            for n, s in h_out.items():
                                st.write(f"  {n}: {s}")
                        with ic2:
                            st.markdown(f"**{away_team}**")
                            for n, s in a_out.items():
                                st.write(f"  {n}: {s}")
    
            # ── Depth charts (cached) ────────────────────────────────────────────
            home_dc = _get_depth_chart(home_team)
            away_dc = _get_depth_chart(away_team)
    
            # ── Starting Lineups ────────────────────────────────────────────────
            st.divider()
            st.markdown("##### Starting Lineups")
    
            # --- Offense ---
            st.markdown("**Offense**")
            off_cols_h, off_vs, off_cols_a = st.columns([5, 1, 5])
            off_labels = ['QB','WR1','WR2','WR3','WR4','RB1','RB2','TE1','TE2',
                          'LT','LG','C','RG','RT']
            h_off_sel = {}
            a_off_sel = {}
    
            with off_cols_h:
                st.markdown(f"**{home_team}**")
            with off_vs:
                st.markdown("<br>**VS**", unsafe_allow_html=True)
            with off_cols_a:
                st.markdown(f"**{away_team}**")
    
            for lbl in off_labels:
                col_h, col_v, col_a = st.columns([5, 1, 5])
                with col_h:
                    opts_h = _depth_list(home_dc, lbl)
                    h_off_sel[lbl] = st.selectbox(
                        lbl, opts_h, index=0, key=f'{pfx}h_off_{lbl}', label_visibility='collapsed')
                    # Show label in a more compact way
                with col_v:
                    st.markdown(f"<div style='text-align:center;padding-top:8px'><small>{lbl}</small></div>",
                                unsafe_allow_html=True)
                with col_a:
                    opts_a = _depth_list(away_dc, lbl)
                    a_off_sel[lbl] = st.selectbox(
                        lbl, opts_a, index=0, key=f'{pfx}a_off_{lbl}', label_visibility='collapsed')
    
            # --- Defense ---
            st.markdown("**Defense**")
            def_labels = ['DE1','DE2','DT','LB1','LB2','LB3','CB1','CB2','FS','SS']
            h_def_sel = {}
            a_def_sel = {}
    
            col_hd, col_dv, col_ad = st.columns([5, 1, 5])
            with col_hd:
                st.markdown(f"**{home_team}**")
            with col_dv:
                st.markdown("<br>**VS**", unsafe_allow_html=True)
            with col_ad:
                st.markdown(f"**{away_team}**")
    
            for lbl in def_labels:
                col_h, col_v, col_a = st.columns([5, 1, 5])
                with col_h:
                    opts_h = _depth_list(home_dc, lbl)
                    h_def_sel[lbl] = st.selectbox(
                        lbl, opts_h, index=0, key=f'{pfx}h_def_{lbl}', label_visibility='collapsed')
                with col_v:
                    st.markdown(f"<div style='text-align:center;padding-top:8px'><small>{lbl}</small></div>",
                                unsafe_allow_html=True)
                with col_a:
                    opts_a = _depth_list(away_dc, lbl)
                    a_def_sel[lbl] = st.selectbox(
                        lbl, opts_a, index=0, key=f'{pfx}a_def_{lbl}', label_visibility='collapsed')
    
            # ── Predict button ───────────────────────────────────────────────────
            st.divider()
            if st.button("Predict This Game", type="primary",
                          use_container_width=True, key=f'{pfx}predict'):
                st.session_state[f'{pfx}stay_open'] = True  # pin expander open
                h_lineup = build_lineup_dict(home_team, home_dc, h_off_sel) if _WEEKLY_MODULES_OK else \
                           {'qb_score': 65.0, 'wr1_score': 65.0, 'rb1_score': 65.0,
                            'te1_score': 60.0, 'wr2_score': 52.0}
                a_lineup = build_lineup_dict(away_team, away_dc, a_off_sel) if _WEEKLY_MODULES_OK else \
                           {'qb_score': 65.0, 'wr1_score': 65.0, 'rb1_score': 65.0,
                            'te1_score': 60.0, 'wr2_score': 52.0}
                result = run_game_prediction(
                    home_team, away_team, conditions,
                    h_lineup, a_lineup,
                    home_dc, away_dc,
                    live_ctx=live_ctx,
                )
                st.session_state[f'{pfx}pred'] = result
                st.rerun()  # re-render with expander kept open + results shown
    
            pred_result = st.session_state.get(f'{pfx}pred')
            if pred_result:
                st.divider()
                render_prediction_result(pred_result, home_team, away_team, pfx, game_date=_game_date)
    
    
    # ── Tabs ──────────────────────────────────────────
    tab1, tab2, tab_ladder, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Game Predictor + Lineups",
        "🏃 Player Props",
        "🪜 Parlay Ladder",
        "📈 Head-to-Head",
        "🏆 Super Bowl Predictor",
        "📅 Backtesting",
        "📋 Track Record",
    ])
    
    def _render_manual_entry_tab():
        """Original manual game-entry form (preserved as-is for Manual Entry mode)."""
        st.caption("Pre-populated with likely starters — swap anyone out to see how it affects the prediction")
        st.divider()
    
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("🏠 Home Team")
            home_team = st.selectbox("Home Team", NFL_TEAMS, index=NFL_TEAMS.index('KC'))
            home_rest = st.slider("Days of Rest", 3, 14, 7, key='hr')
            spread    = st.slider("Vegas Spread (neg = home favored)", -28.0, 28.0, -3.0, 0.5)
        with c2:
            st.subheader("✈️ Away Team")
            away_team = st.selectbox("Away Team", NFL_TEAMS, index=NFL_TEAMS.index('BUF'))
            away_rest = st.slider("Days of Rest", 3, 14, 7, key='ar')
            is_div    = st.checkbox("Divisional Game?")
        with c3:
            st.subheader("🌤️ Conditions")
            roof      = st.selectbox("Stadium", ['outdoors','dome','closed','open'])
            is_dome   = 1 if roof == 'dome' else 0
            outdoor   = roof in ['outdoors','open']
            temp      = st.slider("Temp (°F)", 20, 100, 65) if outdoor else 72
            wind      = st.slider("Wind (mph)", 0, 40, 5)   if outdoor else 0
            surface   = st.selectbox("Surface", ['grass','turf'])
            is_grass  = 1 if surface == 'grass' else 0
            total_ou  = st.number_input("Vegas O/U (total)", min_value=20.0, max_value=80.0,
                                         value=44.5, step=0.5, key='total_ou')
    
        # ── Live Data Section ────────────────────────────────────────────
        st.divider()
        st.subheader("Live Data")
        st.caption("Auto-fetches real weather, Vegas lines, and injury report for the selected matchup")
    
        ld1, ld2, ld3 = st.columns([2, 2, 1])
        with ld1:
            game_date = st.date_input("Game Date", value=date.today(), key='gdate')
        with ld2:
            game_time_sel = st.selectbox(
                "Kickoff (ET)", ["13:00","16:05","16:25","20:15","20:20","21:15"], key='gtime')
        with ld3:
            st.write("")
            st.write("")
            fetch_clicked = st.button("Fetch Live Data", use_container_width=True)
    
        if fetch_clicked:
            if pipeline:
                with st.spinner("Fetching weather, Vegas lines, injuries..."):
                    _home_elo_tmp = get_elo(home_team)
                    _away_elo_tmp = get_elo(away_team)
                    _ctx = pipeline.get_game_context(
                        home_team, away_team,
                        f"{game_date}T{game_time_sel}",
                        elo_diff=_home_elo_tmp - _away_elo_tmp,
                        home_rest=home_rest, away_rest=away_rest,
                        div_game=int(is_div), surface=surface,
                    )
                    st.session_state['live_ctx'] = _ctx
                    st.session_state['live_ctx_teams'] = (home_team, away_team)
            else:
                st.warning("Pipeline not loaded — check that data_pipeline.py is present.")
    
        # Display live data if fetched for these teams
        _live_ctx = st.session_state.get('live_ctx', {})
        _ctx_teams = st.session_state.get('live_ctx_teams', (None, None))
        _ctx_valid = (_live_ctx and _ctx_teams == (home_team, away_team))
    
        if _ctx_valid:
            _wx   = _live_ctx.get('weather', {})
            _odds = _live_ctx.get('odds', {})
            _inj  = _live_ctx.get('injuries', {})
            _api  = _live_ctx.get('api_status', {})
    
            info_cols = st.columns(3)
            with info_cols[0]:
                if _wx:
                    dome_label = "DOME" if _wx.get('is_dome') else "Outdoor"
                    st.info(f"Weather ({_wx.get('stadium','')[:20]}): "
                            f"{_wx.get('temp',72)}F | {_wx.get('wind',0)} mph | {dome_label}")
            with info_cols[1]:
                _fmt = _odds.get('formatted', '')
                if _fmt and _fmt != 'Lines not available':
                    st.info(f"Vegas: {_fmt}")
                elif not _api.get('odds'):
                    st.caption("Vegas lines: add ODDS_API_KEY to .env")
            with info_cols[2]:
                _h_imp = _inj.get('home_impact', 0)
                _a_imp = _inj.get('away_impact', 0)
                if _h_imp > 0.10 or _a_imp > 0.10:
                    st.warning(
                        f"Injury alert: {home_team} {_h_imp:.0%} | {away_team} {_a_imp:.0%}")
                else:
                    st.success("No significant injuries detected")
    
            # Show injured players detail
            _h_flags = _inj.get('home_flags', {})
            _a_flags = _inj.get('away_flags', {})
            key_statuses = ("OUT", "IR", "PUP", "DNR", "DOUBTFUL")
            _h_out = {n: s for n, s in _h_flags.items() if s in key_statuses}
            _a_out = {n: s for n, s in _a_flags.items() if s in key_statuses}
            if _h_out or _a_out:
                with st.expander("Injury Detail"):
                    ic1, ic2 = st.columns(2)
                    with ic1:
                        st.markdown(f"**{home_team}**")
                        for name, status in _h_out.items():
                            st.write(f"  {name}: {status}")
                    with ic2:
                        st.markdown(f"**{away_team}**")
                        for name, status in _a_out.items():
                            st.write(f"  {name}: {status}")
    
        st.divider()
        st.subheader("Starting Lineups")
        st.caption("Auto-populated with projected starters — change any player to see the impact")
    
        home_starters = get_starters(home_team)
        lu_col1, lu_spacer, lu_col2 = st.columns([5,1,5])
    
        with lu_col1:
            st.markdown(f"**🏠 {home_team} Offense**")
            home_qb = st.selectbox("QB",  home_starters['QB_list'],  index=0, key='hqb')
            home_rb = st.selectbox("RB",  home_starters['RB_list'],  index=0, key='hrb')
            home_wr = st.selectbox("WR1", home_starters['WR_list'],  index=0, key='hwr')
            home_te = st.selectbox("TE",  home_starters['TE_list'],  index=0, key='hte')
    
        with lu_spacer:
            st.markdown("<br><br><br><br><br><br>**VS**", unsafe_allow_html=True)
    
        with lu_col2:
            st.markdown(f"**✈️ {away_team} Offense**")
            away_starters = get_starters(away_team)
            away_qb = st.selectbox("QB",  away_starters['QB_list'],  index=0, key='aqb')
            away_rb = st.selectbox("RB",  away_starters['RB_list'],  index=0, key='arb')
            away_wr = st.selectbox("WR1", away_starters['WR_list'],  index=0, key='awr')
            away_te = st.selectbox("TE",  away_starters['TE_list'],  index=0, key='ate')
    
        st.divider()
    
        if st.button("Predict Game", type="primary", use_container_width=True):
            home_elo = get_elo(home_team)
            away_elo = get_elo(away_team)
            elo_diff = home_elo - away_elo
    
            # Resolve effective game conditions:
            # Live data takes precedence over manual sliders when fetched for this matchup
            _lctx       = st.session_state.get('live_ctx', {})
            _lctx_valid = (st.session_state.get('live_ctx_teams') == (home_team, away_team))
            _live_wx    = _lctx.get('weather', {}) if _lctx_valid else {}
            _live_odds  = _lctx.get('odds',    {}) if _lctx_valid else {}
            _live_inj   = _lctx.get('injuries',{}) if _lctx_valid else {}
    
            eff_temp    = _live_wx.get('temp',    temp)    if _live_wx else temp
            eff_wind    = _live_wx.get('wind',    wind)    if _live_wx else wind
            eff_is_dome = _live_wx.get('is_dome', is_dome) if _live_wx else is_dome
            eff_spread  = _live_odds.get('spread_home', spread) if _live_odds.get('spread_home') is not None else spread
            eff_spread  = eff_spread if eff_spread is not None else spread
            data_source = "Live API data" if _lctx_valid and _live_wx else "Manual inputs"
    
            # Base feature vector (9 original features — always valid)
            base_fv = {
                'elo_diff':    elo_diff,
                'spread_line': eff_spread,
                'home_rest':   home_rest,
                'away_rest':   away_rest,
                'temp':        eff_temp,
                'wind':        eff_wind,
                'is_dome':     eff_is_dome,
                'is_grass':    is_grass,
                'div_game':    int(is_div),
            }
    
            # Extended features for enhanced model
            if enhanced_features:
                from feature_engineering import _spread_to_prob
    
                # ── Rolling form stats (team_rolling_stats.csv) ──────────────
                def _roll(team, col):
                    if team in team_roll_stats.index:
                        v = team_roll_stats.loc[team, col] if col in team_roll_stats.columns else np.nan
                        return float(v) if pd.notna(v) else np.nan
                    return np.nan
    
                h_l5_win  = _roll(home_team, 'l5_win_pct')
                a_l5_win  = _roll(away_team, 'l5_win_pct')
                h_l5_diff = _roll(home_team, 'l5_pts_diff')
                a_l5_diff = _roll(away_team, 'l5_pts_diff')
                h_elo_tr  = _roll(home_team, 'elo_trend')
                a_elo_tr  = _roll(away_team, 'elo_trend')
                h_l5_for  = _roll(home_team, 'l5_pts_for')
                a_l5_for  = _roll(away_team, 'l5_pts_for')
                h_l5_ag   = _roll(home_team, 'l5_pts_against')
                a_l5_ag   = _roll(away_team, 'l5_pts_against')
    
                # ── Current-season EPA (team_stats_current.csv) ───────────────
                def _epa(team, col):
                    if not team_stats_curr.empty and team in team_stats_curr.index:
                        v = team_stats_curr.loc[team, col] if col in team_stats_curr.columns else np.nan
                        return float(v) if pd.notna(v) else 0.0
                    return 0.0
    
                h_off_epa = _epa(home_team, 'off_epa_per_play')
                h_def_epa = _epa(home_team, 'def_epa_per_play')
                a_off_epa = _epa(away_team, 'off_epa_per_play')
                a_def_epa = _epa(away_team, 'def_epa_per_play')
    
                # ── QB quality differential (qb_team_ratings.csv) ────────────
                def _qb(team):
                    if not qb_ratings.empty and team in qb_ratings.index:
                        v = qb_ratings.loc[team, 'qb_score'] if 'qb_score' in qb_ratings.columns else np.nan
                        return float(v) if pd.notna(v) else 0.0
                    return 0.0
    
                h_qb = _qb(home_team)
                a_qb = _qb(away_team)
    
                # ── Derived features ──────────────────────────────────────────
                pts_diff_adv  = (h_l5_diff - a_l5_diff) if pd.notna(h_l5_diff) and pd.notna(a_l5_diff) else np.nan
                matchup_h     = (h_l5_for - a_l5_ag)    if pd.notna(h_l5_for)  and pd.notna(a_l5_ag)   else np.nan
                matchup_a     = (a_l5_for - h_l5_ag)    if pd.notna(a_l5_for)  and pd.notna(h_l5_ag)   else np.nan
                net_matchup   = (matchup_h - matchup_a)  if pd.notna(matchup_h) and pd.notna(matchup_a)  else np.nan
                epa_off_diff  = h_off_epa - a_off_epa
                epa_def_diff  = a_def_epa - h_def_epa   # positive = home D is better
                epa_total     = epa_off_diff + epa_def_diff
    
                ext_fv = {
                    'home_l5_win_pct':    h_l5_win,
                    'away_l5_win_pct':    a_l5_win,
                    'home_l5_pts_diff':   h_l5_diff,
                    'away_l5_pts_diff':   a_l5_diff,
                    'pts_diff_advantage': pts_diff_adv,
                    'home_l5_pts_for':    h_l5_for,
                    'away_l5_pts_for':    a_l5_for,
                    'home_l5_pts_against':h_l5_ag,
                    'away_l5_pts_against':a_l5_ag,
                    'matchup_adv_home':   matchup_h,
                    'matchup_adv_away':   matchup_a,
                    'net_matchup_adv':    net_matchup,
                    'home_elo_trend':     h_elo_tr,
                    'away_elo_trend':     a_elo_tr,
                    'rest_advantage':     home_rest - away_rest,
                    'spread_implied_prob':_spread_to_prob(eff_spread),
                    'elo_implied_prob':   1 / (1 + 10 ** (-elo_diff / 400)),
                    'qb_score_diff':      h_qb - a_qb,
                    'home_off_epa':       h_off_epa,
                    'away_off_epa':       a_off_epa,
                    'home_def_epa':       h_def_epa,
                    'away_def_epa':       a_def_epa,
                    'epa_off_diff':       epa_off_diff,
                    'epa_def_diff':       epa_def_diff,
                    'epa_total_diff':     epa_total,
                    # Live data extras (used if fetched)
                    'net_injury_adj':     _live_inj.get('away_impact', 0) - _live_inj.get('home_impact', 0),
                }
    
                combined_fv = {**base_fv, **ext_fv}
                # Build DataFrame with exactly the columns the enhanced model expects
                feat_row = {col: combined_fv.get(col, np.nan) for col in enhanced_features}
                features = pd.DataFrame([feat_row])
            else:
                features = pd.DataFrame([base_fv])
    
            base_prob_home = game_model.predict_proba(features)[0][1]
            home_off, home_qb_s, home_rb_s, home_wr_s, home_te_s = \
                calc_lineup_score(home_team, home_qb, home_rb, home_wr, home_te)
            away_off, away_qb_s, away_rb_s, away_wr_s, away_te_s = \
                calc_lineup_score(away_team, away_qb, away_rb, away_wr, away_te)
    
            # Injury adjustment on lineup scores
            _h_impact = _live_inj.get('home_impact', 0)
            _a_impact = _live_inj.get('away_impact', 0)
            home_off_adj = home_off * (1 - _h_impact * 0.5)
            away_off_adj = away_off * (1 - _a_impact * 0.5)
    
            adj = lineup_adjustment(home_off_adj, away_off_adj)
            final_prob_home = float(np.clip(base_prob_home + adj, 0.05, 0.95))
            final_prob_away = 1 - final_prob_home
            elo_prob = elo_win_prob(home_elo, away_elo)
    
            st.subheader("Prediction Results")
            st.caption(f"Using: {data_source}" + (f" | Enhanced model ({enhanced_acc*100:.1f}% trained accuracy)" if enhanced_features else " | Original model"))
    
            r1, r2 = st.columns(2)
            with r1:
                st.metric(f"{home_team} Win Probability", f"{final_prob_home*100:.1f}%",
                          delta=f"{(final_prob_home-base_prob_home)*100:+.1f}% lineup adj")
                st.progress(final_prob_home)
            with r2:
                st.metric(f"{away_team} Win Probability", f"{final_prob_away*100:.1f}%",
                          delta=f"ELO baseline: {(1-elo_prob)*100:.1f}%")
                st.progress(final_prob_away)
    
            winner     = home_team if final_prob_home > 0.5 else away_team
            confidence = max(final_prob_home, final_prob_away)
            if confidence > 0.75:
                label, _css_class = "LOCK", "signal-lock"
            elif confidence > 0.70:
                label, _css_class = "HIGH CONFIDENCE", "signal-strong"
            elif confidence > 0.60:
                label, _css_class = "MODERATE", "signal-lean"
            else:
                label, _css_class = "TOSS-UP", "signal-pass"
            st.markdown(
                f'<div class="signal-badge {_css_class}">{label}: {winner}</div>',
                unsafe_allow_html=True)
    
            # Vegas comparison
            if _live_odds.get('formatted') and _live_odds['formatted'] != 'Lines not available':
                _vp = _live_odds.get('vegas_ml_prob')
                vegas_str = f"Vegas: {_live_odds['formatted']}"
                if _vp:
                    vegas_str += f"  |  Market implied: {_vp*100:.1f}% {home_team}"
                st.info(vegas_str)
    
            with st.expander("Lineup Strength Breakdown"):
                lc1, lc2 = st.columns(2)
                with lc1:
                    st.markdown(f"**{home_team} Offense: {home_off_adj:.1f}/100**")
                    st.write(f"QB  {home_qb} — {home_qb_s:.0f}/100")
                    st.write(f"RB  {home_rb} — {home_rb_s:.0f}/100")
                    st.write(f"WR  {home_wr} — {home_wr_s:.0f}/100")
                    st.write(f"TE  {home_te} — {home_te_s:.0f}/100")
                    if _h_impact > 0:
                        st.caption(f"Injury adj: -{_h_impact:.0%} to lineup score")
                with lc2:
                    st.markdown(f"**{away_team} Offense: {away_off_adj:.1f}/100**")
                    st.write(f"QB  {away_qb} — {away_qb_s:.0f}/100")
                    st.write(f"RB  {away_rb} — {away_rb_s:.0f}/100")
                    st.write(f"WR  {away_wr} — {away_wr_s:.0f}/100")
                    st.write(f"TE  {away_te} — {away_te_s:.0f}/100")
                    if _a_impact > 0:
                        st.caption(f"Injury adj: -{_a_impact:.0%} to lineup score")
                st.caption(f"Lineup adj: {adj*100:+.1f}% | Base: {base_prob_home*100:.1f}% | Final: {final_prob_home*100:.1f}%")
    
                # ── QB Quality Comparison ─────────────────────────────────
                if not qb_ratings.empty:
                    st.divider()
                    st.markdown("**QB Quality Ratings (2024 season)**")
                    qb_c1, qb_c2 = st.columns(2)
                    for col, team, qb_name in [(qb_c1, home_team, home_qb), (qb_c2, away_team, away_qb)]:
                        with col:
                            if team in qb_ratings.index:
                                row = qb_ratings.loc[team]
                                rated_name = str(row.get('player_name', '?'))
                                score      = row.get('qb_score', 0)
                                pct        = row.get('qb_pct', 50)
                                cmp        = row.get('completion_pct', 0)
                                ypa        = row.get('yards_per_att', 0)
                                tdint      = row.get('td_int_ratio', 0)
                                is_backup  = (qb_name != rated_name and rated_name != 'Unknown')
                                st.markdown(f"**{team}**: {qb_name}")
                                if is_backup:
                                    st.warning(f"Backup QB! Rated: {rated_name}")
                                pct_label = ("Elite" if pct >= 90 else "Above avg" if pct >= 65
                                             else "Average" if pct >= 40 else "Below avg")
                                st.write(f"Rating: **{pct:.0f}th pct** ({pct_label})")
                                st.write(f"Cmp%: {cmp:.1%}  |  Y/A: {ypa:.1f}  |  TD/INT: {tdint:.1f}x")
                            else:
                                st.write(f"**{team}**: {qb_name}")
                                st.caption("No rating data available")
    
            # ── Over/Under Prediction ─────────────────────────────────────
            # Model predicts residual (actual - Vegas line). EPA features use ~0.0 scale.
            if total_model_pkg:
                _ou_total_line = float(_live_odds.get('total') or total_ou)
                _avg = 22.0
                _h_diff = ext_fv.get('home_l5_pts_diff', 0) if enhanced_features else 0
                _a_diff = ext_fv.get('away_l5_pts_diff', 0) if enhanced_features else 0
                _h_diff = 0 if pd.isna(_h_diff) else _h_diff
                _a_diff = 0 if pd.isna(_a_diff) else _a_diff
                _h_for  = _avg + _h_diff / 2;  _h_ag = _avg - _h_diff / 2
                _a_for  = _avg + _a_diff / 2;  _a_ag = _avg - _a_diff / 2
                _h_oe = float(ext_fv.get('home_off_epa', 0.0) or 0.0) if enhanced_features else 0.0
                _a_oe = float(ext_fv.get('away_off_epa', 0.0) or 0.0) if enhanced_features else 0.0
                _h_de = float(ext_fv.get('home_def_epa', 0.0) or 0.0) if enhanced_features else 0.0
                _a_de = float(ext_fv.get('away_def_epa', 0.0) or 0.0) if enhanced_features else 0.0
                _qbd  = float(ext_fv.get('qb_score_diff', 0.0) or 0.0) if enhanced_features else 0.0
                _sip  = float(ext_fv.get('spread_implied_prob', 0.5) or 0.5) if enhanced_features else 0.5
                _epa_zero = {'home_off_epa','away_off_epa','home_def_epa','away_def_epa',
                             'scoring_env_off','scoring_env_def','qb_score_diff','elo_diff',
                             'div_game','spread_implied_prob'}
                ou_fv_dict = {
                    'total_line':          _ou_total_line,
                    'home_l5_pts_for':     _h_for,  'away_l5_pts_for':     _a_for,
                    'home_l5_pts_against': _h_ag,   'away_l5_pts_against': _a_ag,
                    'home_l5_pts_diff':    _h_diff, 'away_l5_pts_diff':    _a_diff,
                    'matchup_adv_home':    _h_for - _a_ag,
                    'matchup_adv_away':    _a_for - _h_ag,
                    'home_off_epa':        _h_oe,   'away_off_epa':        _a_oe,
                    'home_def_epa':        _h_de,   'away_def_epa':        _a_de,
                    'scoring_env_off':     _h_oe + _a_oe,
                    'scoring_env_def':     _h_de + _a_de,
                    'qb_score_diff':       _qbd,
                    'elo_diff':            elo_diff,
                    'abs_spread':          abs(eff_spread),
                    'div_game':            int(is_div),
                    'spread_implied_prob': _sip,
                    'wind':     eff_wind,  'temp': eff_temp,  'is_dome': eff_is_dome,
                    'home_rest': home_rest, 'away_rest': away_rest,
                }
                ou_feats = total_model_pkg['features']
                ou_row   = {f: ou_fv_dict.get(f, 0.0 if f in _epa_zero else _avg) for f in ou_feats}
                pred_residual = float(total_model_pkg['model'].predict(pd.DataFrame([ou_row]))[0])
                pred_total    = _ou_total_line + pred_residual   # line + residual = projected total
                edge          = pred_residual                    # residual IS the edge
                lean          = "OVER" if edge > 0 else "UNDER"
                ou_mae        = total_model_pkg.get('mae', 10.2)
    
                st.divider()
                st.subheader("Over/Under Prediction")
                ou_c1, ou_c2, ou_c3 = st.columns(3)
                with ou_c1:
                    st.metric("Model Predicted Total", f"{pred_total:.1f} pts")
                with ou_c2:
                    st.metric("Vegas O/U Line", f"{_ou_total_line:.1f} pts")
                with ou_c3:
                    st.metric("Model Lean", lean, delta=f"{abs(edge):.1f} pts {lean}")
    
                edge_abs = abs(edge)
                if edge_abs >= 4:
                    st.success(f"Strong lean **{lean}** by {edge_abs:.1f} pts (model MAE: {ou_mae:.1f} pts)")
                elif edge_abs >= 2:
                    st.info(f"Moderate lean **{lean}** by {edge_abs:.1f} pts")
                else:
                    st.caption(f"Slight lean {lean} by {edge_abs:.1f} pts — within model uncertainty ({ou_mae:.1f} pt MAE, treat as a push)")
    
                if eff_wind >= 15:
                    st.caption(f"Wind ({eff_wind} mph) is suppressing the model's total — every 5 mph above 15 costs ~1.5 pts")
    
            if eff_wind >= 20:
                st.warning(f"High wind ({eff_wind} mph) — expect lower scoring, passing stats suppressed")
            if eff_temp < 32:
                st.warning(f"Freezing temps ({eff_temp}F) — historically reduces total scoring by ~4 pts")
            if eff_is_dome:
                st.info("Dome game — weather neutralized, passing games typically boosted")
    
    
    def _sample_week_schedule() -> dict:
        """
        Return a static sample NFL week (Week 1 2025-style) for demo / off-season use.
        Covers Thursday, Sunday (early + late + night), and Monday Night.
        """
        return {
            'Thursday': [
                {'game_id': 's_thu_1', 'home_team': 'KC',  'away_team': 'BAL',
                 'game_date_label': 'Sep 4', 'game_time_et': '8:20 PM ET',
                 'venue': 'GEHA Field at Arrowhead Stadium', 'status': 'scheduled'},
            ],
            'Sunday': [
                {'game_id': 's_sun_1',  'home_team': 'MIA', 'away_team': 'BUF',
                 'game_date_label': 'Sep 7', 'game_time_et': '1:00 PM ET',
                 'venue': 'Hard Rock Stadium', 'status': 'scheduled'},
                {'game_id': 's_sun_2',  'home_team': 'PHI', 'away_team': 'GB',
                 'game_date_label': 'Sep 7', 'game_time_et': '1:00 PM ET',
                 'venue': 'Lincoln Financial Field', 'status': 'scheduled'},
                {'game_id': 's_sun_3',  'home_team': 'DET', 'away_team': 'LA',
                 'game_date_label': 'Sep 7', 'game_time_et': '1:00 PM ET',
                 'venue': 'Ford Field', 'status': 'scheduled'},
                {'game_id': 's_sun_4',  'home_team': 'CLE', 'away_team': 'DAL',
                 'game_date_label': 'Sep 7', 'game_time_et': '1:00 PM ET',
                 'venue': 'Huntington Bank Field', 'status': 'scheduled'},
                {'game_id': 's_sun_5',  'home_team': 'NO',  'away_team': 'CAR',
                 'game_date_label': 'Sep 7', 'game_time_et': '1:00 PM ET',
                 'venue': 'Caesars Superdome', 'status': 'scheduled'},
                {'game_id': 's_sun_6',  'home_team': 'TEN', 'away_team': 'CHI',
                 'game_date_label': 'Sep 7', 'game_time_et': '1:00 PM ET',
                 'venue': 'Nissan Stadium', 'status': 'scheduled'},
                {'game_id': 's_sun_7',  'home_team': 'CIN', 'away_team': 'NE',
                 'game_date_label': 'Sep 7', 'game_time_et': '1:00 PM ET',
                 'venue': 'Paycor Stadium', 'status': 'scheduled'},
                {'game_id': 's_sun_8',  'home_team': 'NYG', 'away_team': 'MIN',
                 'game_date_label': 'Sep 7', 'game_time_et': '1:00 PM ET',
                 'venue': 'MetLife Stadium', 'status': 'scheduled'},
                {'game_id': 's_sun_9',  'home_team': 'JAX', 'away_team': 'IND',
                 'game_date_label': 'Sep 7', 'game_time_et': '1:00 PM ET',
                 'venue': 'EverBank Stadium', 'status': 'scheduled'},
                {'game_id': 's_sun_10', 'home_team': 'HOU', 'away_team': 'PIT',
                 'game_date_label': 'Sep 7', 'game_time_et': '4:25 PM ET',
                 'venue': 'NRG Stadium', 'status': 'scheduled'},
                {'game_id': 's_sun_11', 'home_team': 'ARI', 'away_team': 'LV',
                 'game_date_label': 'Sep 7', 'game_time_et': '4:25 PM ET',
                 'venue': 'State Farm Stadium', 'status': 'scheduled'},
                {'game_id': 's_sun_12', 'home_team': 'SF',  'away_team': 'NYJ',
                 'game_date_label': 'Sep 7', 'game_time_et': '4:25 PM ET',
                 'venue': "Levi's Stadium", 'status': 'scheduled'},
                {'game_id': 's_sun_13', 'home_team': 'SEA', 'away_team': 'DEN',
                 'game_date_label': 'Sep 7', 'game_time_et': '4:25 PM ET',
                 'venue': 'Lumen Field', 'status': 'scheduled'},
                {'game_id': 's_sun_14', 'home_team': 'TB',  'away_team': 'WAS',
                 'game_date_label': 'Sep 7', 'game_time_et': '8:20 PM ET',
                 'venue': 'Raymond James Stadium', 'status': 'scheduled'},
            ],
            'Monday': [
                {'game_id': 's_mon_1', 'home_team': 'LAC', 'away_team': 'ATL',
                 'game_date_label': 'Sep 8', 'game_time_et': '8:15 PM ET',
                 'venue': 'SoFi Stadium', 'status': 'scheduled'},
            ],
        }
    
    
    def _render_weekly_schedule_tab():
        """Weekly schedule view — games grouped by day, each as an expander."""
        # Bankroll setting (sidebar) — used for Kelly dollar-amount calculations
        with st.sidebar:
            st.markdown("### 💰 Bankroll Settings")
            st.number_input(
                "My bankroll ($)", min_value=100, max_value=100_000,
                value=min(st.session_state.get('bankroll', 1000), 100_000), step=100,
                key='bankroll',
                help="Set your total betting bankroll. Kelly recommendations will show bet amounts in dollars."
            )
            st.markdown("---")
            st.markdown("**🎯 Betting Settings**")
            _nfl_strat = st.selectbox(
                "Betting Strategy",
                options=["Kelly Criterion", "Fixed %", "Fixed $", "Fractional Kelly"],
                key='nfl_bet_strategy',
                help="Kelly: model edge × risk tolerance · Fixed %: set % of bankroll · Fixed $: set dollar amount · Fractional Kelly: custom fraction"
            )
            st.selectbox(
                "Risk Tolerance",
                options=["Conservative", "Moderate", "Aggressive"],
                index=1,
                key='nfl_risk_tolerance',
                help="Conservative 0.25× · Moderate 0.5× · Aggressive 1.0× Kelly multiplier"
            )
            if _nfl_strat == "Fixed %":
                st.number_input(
                    "Fixed bet (%)", min_value=0.5, max_value=10.0,
                    value=float(st.session_state.get('nfl_fixed_pct', 2.0)), step=0.5,
                    key='nfl_fixed_pct',
                    help="Bet this % of bankroll per game regardless of model edge"
                )
            elif _nfl_strat == "Fixed $":
                st.number_input(
                    "Fixed bet ($)", min_value=1, max_value=10_000,
                    value=int(st.session_state.get('nfl_fixed_dollar', 50)), step=10,
                    key='nfl_fixed_dollar',
                    help="Bet this fixed dollar amount per game regardless of model edge"
                )
            st.caption("Kelly bet amounts update automatically across all game predictions.")

            # ── Daily Summary ───────────────────────────────────────────────
            if _PH_OK:
                st.markdown("---")
                st.markdown("**📊 Today's Summary**")
                try:
                    _daily = _ph.get_daily_recs('nfl')
                    if _daily:
                        _br = int(st.session_state.get('bankroll', 1000))
                        _st = st.session_state.get('nfl_bet_strategy', 'Kelly Criterion')
                        _fp = float(st.session_state.get('nfl_fixed_pct', 2.0))
                        _fd = int(st.session_state.get('nfl_fixed_dollar', 50))
                        _rt = st.session_state.get('nfl_risk_tolerance', 'Moderate')
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

        btn_col, _, sample_col = st.columns([2, 4, 2])
        with btn_col:
            load_btn = st.button("Load / Refresh Schedule", type="secondary",
                                  use_container_width=True)
        with sample_col:
            sample_btn = st.button("Load Sample Week (Demo)", type="secondary",
                                    use_container_width=True)
    
        if load_btn:
            # Clear pre-calc cache and all stale per-game predictions
            import re as _re
            for _k in list(st.session_state.keys()):
                if _k in ('nfl_precalc_done', 'nfl_total_games') or _re.match(r'^g\d+_', _k):
                    del st.session_state[_k]
            with st.spinner("Loading this week's NFL schedule..."):
                _sched = {}
                if _WEEKLY_MODULES_OK:
                    try:
                        if pipeline:
                            _sched = fetch_weekly_schedule(
                                pipeline.espn,
                                getattr(pipeline, 'tank01', None),
                            )
                        else:
                            from apis.espn import ESPNClient
                            _sched = fetch_weekly_schedule(ESPNClient())
                    except Exception as _e:
                        st.warning(f"Could not fetch schedule: {_e}")
                else:
                    st.warning("game_week.py not found — please check your installation.")
                # If ESPN returned no games, auto-fall back to sample
                if not _sched:
                    _sched = _sample_week_schedule()
                    st.info("No live games found (off-season?). Showing sample week for demo.")
                st.session_state['weekly_schedule'] = _sched

        if sample_btn:
            import re as _re
            for _k in list(st.session_state.keys()):
                if _k in ('nfl_precalc_done', 'nfl_total_games') or _re.match(r'^g\d+_', _k):
                    del st.session_state[_k]
            st.session_state['weekly_schedule'] = _sample_week_schedule()
    
        schedule = st.session_state.get('weekly_schedule')
    
        if schedule is None:
            st.info("Click **Load / Refresh Schedule** to fetch this week's NFL games, "
                    "or **Load Sample Week** to explore the interface with demo data.")
            return
    
        if not schedule:
            st.warning("No games found. Click **Load Sample Week** to see a demo.")
            return

        _total_games_nfl = sum(len(v) for v in schedule.values())
        st.session_state['nfl_total_games'] = _total_games_nfl

        # ── Pre-calculate all predictions (once per schedule load) ────────────
        if 'nfl_precalc_done' not in st.session_state:
            _default_cond   = {'spread': 0.0, 'spread_line': 0.0, 'vegas_total': 45.0,
                               'temp': 65, 'wind': 5, 'is_dome': False, 'is_grass': False,
                               'home_rest': 7, 'away_rest': 7}
            _default_lineup = {'qb_score': 65.0, 'wr1_score': 65.0,
                               'rb1_score': 65.0, 'te1_score': 60.0, 'wr2_score': 52.0}
            with st.spinner(f"Pre-calculating predictions for {_total_games_nfl} games..."):
                _pi = 0
                for _day, _dgames in schedule.items():
                    for _game in _dgames:
                        _pfx = f"g{_pi}_"
                        if f'{_pfx}pred' not in st.session_state:
                            try:
                                _cond = st.session_state.get(f'{_pfx}cond', _default_cond)
                                _res  = run_game_prediction(
                                    _game['home_team'], _game['away_team'], _cond,
                                    _default_lineup, _default_lineup, {}, {},
                                )
                                if _res and 'error' not in _res:
                                    st.session_state[f'{_pfx}pred'] = _res
                            except Exception:
                                pass
                        _pi += 1
            st.session_state['nfl_precalc_done'] = True

        # ── Expand All / Collapse All ─────────────────────────────────────────
        _ec1, _, _ec2 = st.columns([2, 6, 2])
        with _ec1:
            if st.button("⬇ Expand All", key='nfl_expand_all', use_container_width=True):
                for _i in range(_total_games_nfl):
                    st.session_state[f'g{_i}_stay_open'] = True
                st.rerun()
        with _ec2:
            if st.button("⬆ Collapse All", key='nfl_collapse_all', use_container_width=True):
                for _i in range(_total_games_nfl):
                    st.session_state[f'g{_i}_stay_open'] = False
                st.rerun()

        game_idx = 0
        for day_name, day_games in schedule.items():
            if not day_games:
                continue
            date_label = day_games[0].get('game_date_label', '')
            st.subheader(f"{day_name}, {date_label}")
            for game_info in day_games:
                _render_game_expander(game_info, game_idx)
                game_idx += 1
    
    
    # ══════════════════════════════════════════════════
    # TAB 1: GAME PREDICTOR
    # ══════════════════════════════════════════════════
    with tab1:
        st.header("🎯 Game Predictor")
        view_mode = st.radio(
            "view_mode_tab1", ["This Week's Games", "Manual Entry"],
            horizontal=True, label_visibility="collapsed",
        )
        st.divider()
    
        if view_mode == "This Week's Games":
            _render_weekly_schedule_tab()
        else:
            _render_manual_entry_tab()
    
    # ══════════════════════════════════════════════════
    # TAB 2: PLAYER PROPS
    # ══════════════════════════════════════════════════
    with tab2:
        st.header("🏃 Player Props + Parlay Builder")
        _props_mode = st.radio(
            "props_view_mode", ["This Week's Props", "Manual Entry"],
            horizontal=True, label_visibility="collapsed",
        )
        st.divider()

        if _props_mode == "This Week's Props":
            # ── Game-card prop selection for Parlay Ladder ────────────
            schedule = st.session_state.get('weekly_schedule')
            if schedule is None:
                st.info("Load the weekly schedule from the **Game Predictor** tab first.")
            else:
                # Selection counter + Build Ladder button
                _rpl_sels = st.session_state.get('rpl_selections', {})
                _sel_ct = len(_rpl_sels)
                _sc1, _sc2 = st.columns([6, 2])
                with _sc1:
                    if _sel_ct > 0:
                        _n_games = len(set(v.get('game_label','') for v in _rpl_sels.values()))
                        st.info(f"🪜 **{_sel_ct} legs selected** from {_n_games} games")
                    else:
                        st.caption("Select legs from game cards below to build a parlay ladder")
                with _sc2:
                    if _sel_ct >= 3:
                        st.success("Switch to **Parlay Ladder** tab to build!")

                # Fetch Vegas prop lines button
                _fc1, _fc2 = st.columns([4, 4])
                with _fc1:
                    _fetch_props = st.button("📡 Fetch Vegas Prop Lines",
                        help="Calls the Odds API for player prop lines (~17 requests). Cached 4 hours.",
                        key='rpl_fetch_props')
                with _fc2:
                    _quota = None
                    if pipeline and hasattr(pipeline, 'odds'):
                        _quota = pipeline.odds.get_quota_cached()
                    if _quota:
                        st.caption(f"Odds API: {_quota.get('used','?')}/500 used this month")

                # Fetch event IDs + prop lines if requested
                if _fetch_props and pipeline and hasattr(pipeline, 'odds'):
                    with st.spinner("Fetching player prop lines from Odds API..."):
                        try:
                            _events = pipeline.odds.get_nfl_events()
                            _event_map = {}
                            for _e in _events:
                                _event_map[f"{_e['home_team']}_{_e['away_team']}"] = _e['event_id']
                                _event_map[f"{_e['away_team']}_{_e['home_team']}"] = _e['event_id']
                            _all_vegas_props = {}
                            for _ek, _eid in _event_map.items():
                                _raw_props = pipeline.odds.get_player_props(_eid)
                                for _rp in _raw_props:
                                    if _rp.get('name') == 'Over':
                                        _pk = f"{_rp['player']}_{_rp['market']}"
                                        _all_vegas_props[_pk] = _rp
                            st.session_state['rpl_vegas_props'] = _all_vegas_props
                            st.success(f"Fetched prop lines for {len(_all_vegas_props)} player markets")
                        except Exception as _ex:
                            st.warning(f"Could not fetch prop lines: {_ex}")

                st.divider()

                # Pre-calculate props for all games
                if 'rpl_props_precalc_done' not in st.session_state:
                    _vegas_props = st.session_state.get('rpl_vegas_props', {})
                    _g_idx = 0
                    with st.spinner("Pre-calculating player props for all games..."):
                        for _day, _day_games in schedule.items():
                            for _gi in _day_games:
                                _home = _gi.get('home_team', '')
                                _away = _gi.get('away_team', '')
                                _pfx = f'g{_g_idx}_'
                                _cond = st.session_state.get(f'{_pfx}cond', {})
                                _spread = _cond.get('spread', 0.0)
                                _temp = _cond.get('temp', 65)
                                _wind = _cond.get('wind', 5)
                                _is_dome = 1 if _cond.get('is_dome', False) else 0

                                _props = []
                                for _team, _is_home in [(_home, 1), (_away, 0)]:
                                    _opp = _away if _team == _home else _home
                                    _team_pl = player_lookup.get(_team, {})
                                    _sp = _spread if _is_home else -_spread

                                    # QB passing
                                    _qbs = _team_pl.get('QB', [])
                                    if _qbs:
                                        _qn = _qbs[0].get('name', '')
                                        _qa = _full_to_abbr(_qn)
                                        _qs = get_player_recent(passing, 'passer_player_name', _qa,
                                            ['pass_yards','pass_attempts','completions','pass_tds'])
                                        if _qs:
                                            _od = get_opp_pass_defense(_opp, def_pass_stats)
                                            _feat = pd.DataFrame([{
                                                'avg_pass_yards_l4': _qs.get('avg_pass_yards_l4', 220),
                                                'avg_pass_attempts_l4': _qs.get('avg_pass_attempts_l4', 32),
                                                'avg_completions_l4': _qs.get('avg_completions_l4', 21),
                                                'avg_pass_tds_l4': _qs.get('avg_pass_tds_l4', 1.5),
                                                'temp': _temp, 'wind': _wind,
                                                'is_dome': _is_dome, 'is_home': _is_home, 'spread_line': _sp,
                                                **_od,
                                            }])
                                            _pv = pass_model.predict(_feat)[0]
                                            _mae = 63.3
                                            _vk = f"{_qn}_player_pass_yds"
                                            _vp = _vegas_props.get(_vk, {})
                                            _vl = _vp.get('line')
                                            _vo = _vp.get('odds', -110)
                                            _conf, _dir, _edge, _ = _prop_confidence(_pv, _vl, _mae)
                                            _props.append({
                                                'player': _qn, 'player_abbr': _qa, 'team': _team, 'opp': _opp,
                                                'prop_type': 'Passing Yards', 'market': 'player_pass_yds',
                                                'prediction': round(_pv, 1), 'mae': _mae,
                                                'vegas_line': _vl, 'odds': _vo,
                                                'confidence': _conf, 'direction': _dir, 'edge': _edge or 0,
                                            })

                                    # RBs rushing
                                    for _rb in _team_pl.get('RB', [])[:2]:
                                        _rn = _rb.get('name', '')
                                        _ra = _full_to_abbr(_rn)
                                        _rs = get_player_recent(rushing, 'rusher_player_name', _ra,
                                            ['rush_yards','rush_attempts','rush_tds'])
                                        if _rs:
                                            _od = get_opp_rush_defense(_opp, def_rush_stats)
                                            _feat = pd.DataFrame([{
                                                'avg_rush_yards_l4': _rs.get('avg_rush_yards_l4', 55),
                                                'avg_rush_attempts_l4': _rs.get('avg_rush_attempts_l4', 14),
                                                'avg_rush_tds_l4': _rs.get('avg_rush_tds_l4', 0.4),
                                                'temp': _temp, 'wind': _wind,
                                                'is_dome': _is_dome, 'is_home': _is_home, 'spread_line': _sp,
                                                **_od,
                                            }])
                                            _pv = rush_model.predict(_feat)[0]
                                            _mae = 21.2
                                            _vk = f"{_rn}_player_rush_yds"
                                            _vp = _vegas_props.get(_vk, {})
                                            _vl = _vp.get('line')
                                            _vo = _vp.get('odds', -110)
                                            _conf, _dir, _edge, _ = _prop_confidence(_pv, _vl, _mae)
                                            _props.append({
                                                'player': _rn, 'player_abbr': _ra, 'team': _team, 'opp': _opp,
                                                'prop_type': 'Rushing Yards', 'market': 'player_rush_yds',
                                                'prediction': round(_pv, 1), 'mae': _mae,
                                                'vegas_line': _vl, 'odds': _vo,
                                                'confidence': _conf, 'direction': _dir, 'edge': _edge or 0,
                                            })

                                    # WRs + TE receiving
                                    _rec_players = _team_pl.get('WR', [])[:2] + _team_pl.get('TE', [])[:1]
                                    for _wr in _rec_players:
                                        _wn = _wr.get('name', '')
                                        _wa = _full_to_abbr(_wn)
                                        _ws = get_player_recent(receiving, 'receiver_player_name', _wa,
                                            ['rec_yards','targets','receptions','rec_tds'])
                                        if _ws:
                                            _od = get_opp_pass_defense(_opp, def_pass_stats)
                                            _feat = pd.DataFrame([{
                                                'avg_rec_yards_l4': _ws.get('avg_rec_yards_l4', 50),
                                                'avg_targets_l4': _ws.get('avg_targets_l4', 6),
                                                'avg_receptions_l4': _ws.get('avg_receptions_l4', 4),
                                                'avg_rec_tds_l4': _ws.get('avg_rec_tds_l4', 0.3),
                                                'temp': _temp, 'wind': _wind,
                                                'is_dome': _is_dome, 'is_home': _is_home, 'spread_line': _sp,
                                                **_od,
                                            }])
                                            _pv = rec_model.predict(_feat)[0]
                                            _mae = 21.3
                                            _vk = f"{_wn}_player_reception_yds"
                                            _vp = _vegas_props.get(_vk, {})
                                            _vl = _vp.get('line')
                                            _vo = _vp.get('odds', -110)
                                            _conf, _dir, _edge, _ = _prop_confidence(_pv, _vl, _mae)
                                            _props.append({
                                                'player': _wn, 'player_abbr': _wa, 'team': _team, 'opp': _opp,
                                                'prop_type': 'Receiving Yards', 'market': 'player_reception_yds',
                                                'prediction': round(_pv, 1), 'mae': _mae,
                                                'vegas_line': _vl, 'odds': _vo,
                                                'confidence': _conf, 'direction': _dir, 'edge': _edge or 0,
                                            })

                                _props.sort(key=lambda p: p.get('confidence', 0), reverse=True)
                                st.session_state[f'rpl_g{_g_idx}_props'] = _props[:10]

                                # Game-level bets (ML + O/U)
                                _game_pred = st.session_state.get(f'{_pfx}pred')
                                _game_bets = []
                                if _game_pred:
                                    _hp = _game_pred.get('home_prob', 0.5)
                                    _ap = 1.0 - _hp
                                    _pick_team = _home if _hp >= _ap else _away
                                    _pick_prob = max(_hp, _ap)
                                    _ml = _game_pred.get('moneyline') or _spread_to_ml(_spread)
                                    _game_bets.append({
                                        'leg_id': f'nfl_{_home}_{_away}_ml',
                                        'game_id': f'{_away}@{_home}',
                                        'game_label': f'{_away} @ {_home}',
                                        'home_team': _home, 'away_team': _away,
                                        'bet_type': 'ml',
                                        'description': f'{_pick_team} Moneyline Win',
                                        'confidence': _pick_prob,
                                        'direction': 'HOME' if _pick_team == _home else 'AWAY',
                                        'vegas_line': None,
                                        'odds': int(_ml) if _ml else -110,
                                        'market': 'h2h',
                                        'player': None, 'prop_type': None,
                                        'model_pred': _pick_prob, 'mae': None,
                                        'edge': _game_pred.get('edge_pct', 0),
                                    })
                                    _ou_pred = _game_pred.get('model_total')
                                    _ou_line = _game_pred.get('ou_line')
                                    if _ou_pred and _ou_line:
                                        _ou_conf, _ou_dir, _ou_edge, _ = _prop_confidence(_ou_pred, _ou_line, 10.01)
                                        _game_bets.append({
                                            'leg_id': f'nfl_{_home}_{_away}_ou',
                                            'game_id': f'{_away}@{_home}',
                                            'game_label': f'{_away} @ {_home}',
                                            'home_team': _home, 'away_team': _away,
                                            'bet_type': 'ou',
                                            'description': f'{_ou_dir} {_ou_line}',
                                            'confidence': _ou_conf,
                                            'direction': _ou_dir,
                                            'vegas_line': _ou_line,
                                            'odds': -110,
                                            'market': 'totals',
                                            'player': None, 'prop_type': None,
                                            'model_pred': _ou_pred, 'mae': 10.01,
                                            'edge': _ou_edge or 0,
                                        })
                                st.session_state[f'rpl_g{_g_idx}_game_bets'] = _game_bets
                                _g_idx += 1
                    st.session_state['rpl_props_precalc_done'] = True

                # Render game cards
                _g_idx = 0
                for _day, _day_games in schedule.items():
                    if not _day_games:
                        continue
                    st.subheader(_day)
                    for _gi in _day_games:
                        _home = _gi.get('home_team', '')
                        _away = _gi.get('away_team', '')
                        _time = _gi.get('game_time_et', 'TBD')
                        _pfx = f'rpl_g{_g_idx}_'
                        _sels = st.session_state.get('rpl_selections', {})

                        # Count legs selected from this game
                        _gid = f'{_away}@{_home}'
                        _gc = sum(1 for v in _sels.values() if v.get('game_id') == _gid)
                        _label = f"{_away} @ {_home}  |  {_time}"
                        if _gc > 0:
                            _label += f"  |  🪜 {_gc} legs"

                        with st.expander(_label, expanded=False):
                            # Game-level bets
                            _gbets = st.session_state.get(f'rpl_g{_g_idx}_game_bets', [])
                            if _gbets:
                                st.markdown("**Game-Level Bets**")
                                for _bi, _bet in enumerate(_gbets):
                                    _cbk = f'{_pfx}cb_{_bet["bet_type"]}'
                                    _in_sel = _bet['leg_id'] in _sels
                                    _c = st.columns([0.5, 3, 1.5, 1.5, 1])
                                    with _c[0]:
                                        _checked = st.checkbox("", key=_cbk, value=_in_sel)
                                    with _c[1]:
                                        st.write(_bet['description'])
                                    with _c[2]:
                                        st.write(f"Conf: **{_bet['confidence']*100:.1f}%**")
                                    with _c[3]:
                                        st.write(f"Edge: {_bet['edge']:+.1f}%")
                                    with _c[4]:
                                        _os = f"+{_bet['odds']}" if _bet['odds'] > 0 else str(_bet['odds'])
                                        st.write(_os)
                                    if _checked and not _in_sel:
                                        _rpl_add_selection(_bet)
                                        st.rerun()
                                    elif not _checked and _in_sel:
                                        _rpl_remove_selection(_bet['leg_id'])
                                        st.rerun()
                                st.divider()

                            # Player props
                            _props = st.session_state.get(f'rpl_g{_g_idx}_props', [])
                            if _props:
                                st.markdown("**Top Player Props**")
                                for _pi, _prop in enumerate(_props[:5]):
                                    _lid = f"nfl_{_home}_{_away}_prop_{_prop['player_abbr']}_{_prop['market']}"
                                    _cbk = f'{_pfx}cb_prop_{_pi}'
                                    _in_sel = _lid in _sels
                                    _c = st.columns([0.5, 2.5, 1.5, 1, 1, 1])
                                    with _c[0]:
                                        _checked = st.checkbox("", key=_cbk, value=_in_sel)
                                    with _c[1]:
                                        st.write(f"**{_prop['player']}** ({_prop['team']})")
                                        _vl_str = f"{_prop['vegas_line']:.1f}" if _prop.get('vegas_line') else 'N/A'
                                        st.caption(f"{_prop['prop_type']} {_prop['direction']} {_vl_str}")
                                    with _c[2]:
                                        st.write(f"Pred: **{_prop['prediction']:.0f}** yds")
                                    with _c[3]:
                                        st.write(f"**{_prop['confidence']*100:.1f}%**")
                                    with _c[4]:
                                        st.write(f"{_prop['edge']:+.1f}")
                                    with _c[5]:
                                        _os = f"+{_prop['odds']}" if _prop.get('odds', -110) > 0 else str(_prop.get('odds', -110))
                                        st.write(_os)

                                    if _checked and not _in_sel:
                                        _prop_leg = {
                                            'leg_id': _lid,
                                            'game_id': _gid,
                                            'game_label': f'{_away} @ {_home}',
                                            'home_team': _home, 'away_team': _away,
                                            'bet_type': 'prop',
                                            'description': f"{_prop['player']} {_prop['prop_type']} {_prop['direction']} {_vl_str}",
                                            'confidence': _prop['confidence'],
                                            'direction': _prop['direction'],
                                            'vegas_line': _prop.get('vegas_line'),
                                            'odds': _prop.get('odds', -110),
                                            'market': _prop['market'],
                                            'player': _prop['player'],
                                            'prop_type': _prop['prop_type'],
                                            'model_pred': _prop['prediction'],
                                            'mae': _prop['mae'],
                                            'edge': _prop['edge'],
                                        }
                                        _rpl_add_selection(_prop_leg)
                                        st.rerun()
                                    elif not _checked and _in_sel:
                                        _rpl_remove_selection(_lid)
                                        st.rerun()
                            else:
                                st.caption("No prop predictions available for this game")

                        _g_idx += 1

        else:
            # ── Manual Entry mode (original props UI) ──────────────────
            st.caption("Based on recent form (last 4 games) + game conditions")
            pc1, pc2 = st.columns(2)
            with pc1:
                prop_type = st.selectbox("Stat to Predict",
                    ["Passing Yards","Rushing Yards","Receiving Yards"])
                if prop_type == "Passing Yards":
                    all_players = sorted(passing['passer_player_name'].dropna().unique())
                    default = 'P.Mahomes' if 'P.Mahomes' in all_players else all_players[0]
                elif prop_type == "Rushing Yards":
                    all_players = sorted(rushing['rusher_player_name'].dropna().unique())
                    default = 'D.Henry' if 'D.Henry' in all_players else all_players[0]
                else:
                    all_players = sorted(receiving['receiver_player_name'].dropna().unique())
                    default = 'T.Hill' if 'T.Hill' in all_players else all_players[0]
                player = st.selectbox("Player", all_players,
                    index=all_players.index(default) if default in all_players else 0)

            with pc2:
                p_team   = st.selectbox("Player's Team", NFL_TEAMS, key='pt')
                opp      = st.selectbox("Opponent", NFL_TEAMS, index=1, key='opp')
                p_home   = st.checkbox("Home game?", value=True)
                p_roof   = st.selectbox("Stadium", ['outdoors','dome','closed','open'], key='pr')
                p_outdoor = p_roof in ['outdoors','open']
                p_temp   = st.slider("Temp", 20, 100, 65, key='ptemp') if p_outdoor else 72
                p_wind   = st.slider("Wind", 0, 40, 5, key='pwind')    if p_outdoor else 0
                p_spread = st.slider("Spread", -28.0, 28.0, -3.0, key='pspread')
                p_line   = st.number_input("Vegas Prop Line (yds)", min_value=0.0,
                                            max_value=600.0, value=0.0, step=0.5,
                                            help="Enter the sportsbook over/under line. Leave 0 to skip.")

            if st.button("🔮 Predict Props", type="primary", use_container_width=True):
                p_dome  = 1 if p_roof == 'dome' else 0
                is_home = 1 if p_home else 0
                pred    = None
                mae     = None

                if prop_type == "Passing Yards":
                    s = get_player_recent(passing, 'passer_player_name', player,
                        ['pass_yards','pass_attempts','completions','pass_tds'])
                    if s:
                        opp_def = get_opp_pass_defense(opp, def_pass_stats)
                        f = pd.DataFrame([{
                            'avg_pass_yards_l4':    s.get('avg_pass_yards_l4', 220),
                            'avg_pass_attempts_l4': s.get('avg_pass_attempts_l4', 32),
                            'avg_completions_l4':   s.get('avg_completions_l4', 21),
                            'avg_pass_tds_l4':      s.get('avg_pass_tds_l4', 1.5),
                            'temp': p_temp, 'wind': p_wind,
                            'is_dome': p_dome, 'is_home': is_home, 'spread_line': p_spread,
                            **opp_def,
                        }])
                        pred, mae = pass_model.predict(f)[0], 63.3

                elif prop_type == "Rushing Yards":
                    s = get_player_recent(rushing, 'rusher_player_name', player,
                        ['rush_yards','rush_attempts','rush_tds'])
                    if s:
                        opp_def = get_opp_rush_defense(opp, def_rush_stats)
                        f = pd.DataFrame([{
                            'avg_rush_yards_l4':    s.get('avg_rush_yards_l4', 55),
                            'avg_rush_attempts_l4': s.get('avg_rush_attempts_l4', 14),
                            'avg_rush_tds_l4':      s.get('avg_rush_tds_l4', 0.4),
                            'temp': p_temp, 'wind': p_wind,
                            'is_dome': p_dome, 'is_home': is_home, 'spread_line': p_spread,
                            **opp_def,
                        }])
                        pred, mae = rush_model.predict(f)[0], 21.2

                else:
                    s = get_player_recent(receiving, 'receiver_player_name', player,
                        ['rec_yards','targets','receptions','rec_tds'])
                    if s:
                        opp_def = get_opp_pass_defense(opp, def_pass_stats)
                        f = pd.DataFrame([{
                            'avg_rec_yards_l4':   s.get('avg_rec_yards_l4', 50),
                            'avg_targets_l4':     s.get('avg_targets_l4', 6),
                            'avg_receptions_l4':  s.get('avg_receptions_l4', 4),
                            'avg_rec_tds_l4':     s.get('avg_rec_tds_l4', 0.3),
                            'temp': p_temp, 'wind': p_wind,
                            'is_dome': p_dome, 'is_home': is_home, 'spread_line': p_spread,
                            **opp_def,
                        }])
                        pred, mae = rec_model.predict(f)[0], 21.3

                st.divider()
                if pred is not None and pred > 0:
                    low  = max(0, pred - mae)
                    high = pred + mae
                    m1, m2, m3 = st.columns(3)
                    m1.metric("📉 Low End",    f"{low:.0f} yds")
                    m2.metric("🎯 Projection", f"{pred:.0f} yds")
                    m3.metric("📈 High End",   f"{high:.0f} yds")
                    st.progress(min(float(pred/(high+10)), 1.0))
                    st.caption(f"MAE: ±{mae} yards")

                    if p_line > 0:
                        edge     = pred - p_line
                        lean     = "OVER" if edge > 0 else "UNDER"
                        edge_abs = abs(edge)
                        ou_c1, ou_c2, ou_c3 = st.columns(3)
                        with ou_c1:
                            st.metric("Model Projection", f"{pred:.0f} yds")
                        with ou_c2:
                            st.metric("Vegas Line", f"{p_line:.1f} yds")
                        with ou_c3:
                            st.metric("Lean", lean, delta=f"{edge_abs:.1f} yds {lean}")

                        strong_thresh  = mae * 0.30
                        moderate_thresh = mae * 0.15
                        if edge_abs >= strong_thresh:
                            st.success(
                                f"Strong lean **{lean}** — model projects {pred:.0f} vs line {p_line:.1f} "
                                f"(edge: {edge_abs:.1f} yds). Model MAE: ±{mae:.0f} yds."
                            )
                        elif edge_abs >= moderate_thresh:
                            st.info(
                                f"Moderate lean **{lean}** — {edge_abs:.1f} yd edge. "
                                f"Model MAE ±{mae:.0f} yds — treat with caution."
                            )
                        else:
                            st.caption(
                                f"Slight lean {lean} by {edge_abs:.1f} yds — within model noise. "
                                f"MAE: ±{mae:.0f} yds. No strong edge detected."
                            )

                        if prop_type == "Passing Yards" and p_wind >= 15:
                            st.caption(f"Wind ({p_wind} mph) suppresses passing — model accounts for this.")
                        if prop_type == "Rushing Yards" and p_wind >= 20:
                            st.caption(f"High wind ({p_wind} mph) may shift game to run-heavy — slight upside.")

                    if prop_type == "Passing Yards":
                        recent = passing[passing['passer_player_name'] == player]\
                            .sort_values(['season','week']).tail(5)\
                            [['season','week','posteam','defteam',
                              'pass_yards','pass_attempts','completions','pass_tds']]
                    elif prop_type == "Rushing Yards":
                        recent = rushing[rushing['rusher_player_name'] == player]\
                            .sort_values(['season','week']).tail(5)\
                            [['season','week','posteam','defteam',
                              'rush_yards','rush_attempts','rush_tds']]
                    else:
                        recent = receiving[receiving['receiver_player_name'] == player]\
                            .sort_values(['season','week']).tail(5)\
                            [['season','week','posteam','defteam',
                              'rec_yards','targets','receptions','rec_tds']]

                    st.subheader(f"📋 {player}'s Last 5 Games")
                    st.dataframe(recent.reset_index(drop=True), use_container_width=True)
                else:
                    st.warning(f"Not enough recent data for {player}.")
    
    # ══════════════════════════════════════════════════
    # TAB LADDER: PARLAY LADDER
    # ══════════════════════════════════════════════════
    with tab_ladder:
        st.header("🪜 Parlay Ladder")

        import parlay_math as _pm

        _rpl_sels = st.session_state.get('rpl_selections', {})

        if len(_rpl_sels) < 3:
            st.info(
                f"Select at least **3 legs** from the **Player Props** tab to build a ladder. "
                f"Currently selected: **{len(_rpl_sels)}** legs."
            )
            st.caption("Go to the Player Props tab → This Week's Props → expand game cards → toggle checkboxes.")
        else:
            _legs = sorted(_rpl_sels.values(), key=lambda l: l.get('confidence', 0), reverse=True)
            _corr_flags = _pm.check_correlations(_legs)

            _bankroll = int(st.session_state.get('bankroll', 1000))
            _max_exp = int(_bankroll * 0.25)
            _ladder_budget = st.slider(
                "Total Ladder Stake ($)",
                min_value=10, max_value=max(10, _max_exp),
                value=min(50, max(10, _max_exp)),
                step=5, key='rpl_ladder_budget',
                help=f"25% max daily exposure cap: ${_max_exp}"
            )

            _tiers = _pm.optimize_tiers(_legs, _ladder_budget)
            _tier_results = _pm.compute_stakes(_tiers, _ladder_budget)

            # ── Summary metrics (Gemini layout pattern) ──────────────
            _n_games = len(set(v.get('game_label', '') for v in _rpl_sels.values()))
            _max_payout = sum(t.get('payout', 0) for t in _tier_results)
            _banker_payout = _tier_results[0]['payout'] if _tier_results else 0
            _be_ok = _banker_payout >= _ladder_budget

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Legs", f"{len(_legs)}", f"{len(_tier_results)} Tiers")
            c2.metric("Total Stake", f"${_ladder_budget}", f"Bankroll: ${_bankroll:,}")
            _roi_pct = ((_max_payout - _ladder_budget) / _ladder_budget * 100) if _ladder_budget > 0 else 0
            c3.metric("Max Payout", f"${_max_payout:.0f}", f"+{_roi_pct:.0f}% ROI")
            if _be_ok:
                c4.markdown('<div class="signal-badge signal-lock">\u2705 BANKER COVERS COST</div>', unsafe_allow_html=True)
            else:
                _short = _ladder_budget - _banker_payout
                c4.markdown(f'<div class="signal-badge signal-lean">\u26a0\ufe0f BANKER SHORT ${_short:.0f}</div>', unsafe_allow_html=True)

            # Correlation flags
            if _corr_flags:
                with st.expander(f"\u26a0\ufe0f {len(_corr_flags)} Correlation Flags", expanded=False):
                    for _fl in _corr_flags:
                        st.warning(_fl['message'])

            st.caption("*The Banker keeps you in the game while waiting for the Moonshot hit.*")
            st.divider()

            # ── Tier sections (Gemini layout pattern — wired to real data) ──
            _TIER_EMOJI = ['\U0001f3e6', '\U0001f4c8', '\U0001f680', '\U0001f319']
            for i, tier in enumerate(_tier_results):
                with st.container(border=True):
                    _emoji = _TIER_EMOJI[i] if i < len(_TIER_EMOJI) else '\U0001f3af'
                    _n = tier.get('n_legs', len(tier.get('legs', [])))
                    _am = tier.get('combined_american', 0)
                    _am_str = f"+{_am}" if _am > 0 else str(_am)

                    h1, h2 = st.columns([3, 1])
                    with h1:
                        st.markdown(
                            f"**Tier {i+1}: {tier['name']}** "
                            f"<span style='color:#94a3b8'>\u2014 {tier.get('subtitle','')} \u00b7 {_n} Legs</span>",
                            unsafe_allow_html=True
                        )
                    with h2:
                        st.markdown(
                            f"<div style='text-align:right; font-family:JetBrains Mono; font-weight:bold; color:#22d3ee; font-size:1.2em'>{_am_str}</div>",
                            unsafe_allow_html=True
                        )

                    # Combined probability
                    _cp = tier.get('combined_prob', 0)
                    _cp_color = '#22c55e' if _cp > 0.3 else '#eab308' if _cp > 0.1 else '#ef4444'
                    st.markdown(
                        f"<div style='font-size:1.1em;font-weight:600;color:{_cp_color}'>Combined Probability: {_cp*100:.1f}%</div>",
                        unsafe_allow_html=True
                    )

                    # Leg details
                    for leg in tier.get('legs', []):
                        _desc = leg.get('description', '')
                        _conf = leg.get('confidence', 0)
                        _edge = leg.get('edge', 0)
                        _badge_cls = 'signal-lock' if _conf >= 0.75 else 'signal-strong' if _conf >= 0.65 else 'signal-lean' if _conf >= 0.55 else 'signal-pass'
                        l1, l2, l3 = st.columns([4, 2, 2])
                        l1.write(_desc)
                        l2.markdown(f"<span class='signal-badge {_badge_cls}'>Edge {_edge:+.1f}</span>", unsafe_allow_html=True)
                        l3.caption(f"Prob: {_conf*100:.1f}%")

                    st.divider()
                    f1, f2, f3 = st.columns([2, 2, 2])
                    f1.metric("Tier Stake", f"${tier.get('stake', 0):.2f}")
                    f2.metric("Potential Payout", f"${tier.get('payout', 0):.2f}")
                    f3.metric("Implied Prob", f"{_cp*100:.1f}%")

    # ══════════════════════════════════════════════════
    # TAB 3: HEAD TO HEAD
    # ══════════════════════════════════════════════════
    with tab3:
        st.header("📈 Historical Head-to-Head")
        h1, h2 = st.columns(2)
        with h1:
            t1 = st.selectbox("Team A", NFL_TEAMS, key='t1')
        with h2:
            t2 = st.selectbox("Team B", NFL_TEAMS, index=1, key='t2')
    
        h2h = games[
            ((games['home_team'] == t1) & (games['away_team'] == t2)) |
            ((games['home_team'] == t2) & (games['away_team'] == t1))
        ].copy()
    
        if len(h2h) > 0:
            t1_wins = (
                ((h2h['home_team'] == t1) & (h2h['home_score'] > h2h['away_score'])) |
                ((h2h['away_team'] == t1) & (h2h['away_score'] > h2h['home_score']))
            ).sum()
            t2_wins = len(h2h) - t1_wins
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"{t1} Wins", t1_wins)
            m2.metric(f"{t2} Wins", t2_wins)
            m3.metric("Total Games", len(h2h))
            m4.metric(f"{t1} Win %", f"{t1_wins/len(h2h)*100:.0f}%")
            st.dataframe(
                h2h[['season','week','home_team','home_score',
                      'away_score','away_team','temp','wind','roof']]
                .sort_values('season', ascending=False).head(15).reset_index(drop=True),
                use_container_width=True)
        else:
            st.info("No matchups found in dataset.")
    
    # ══════════════════════════════════════════════════
    # TAB 4: SUPER BOWL PREDICTOR
    # ══════════════════════════════════════════════════
    with tab4:
        st.header("🏆 Super Bowl Predictor")
        st.caption("Simulates the entire NFL playoff bracket using ELO + lineup strength")
        st.divider()
    
        st.subheader("🌱 Playoff Seeds")
        st.caption("Pre-loaded with 2025 playoff teams — adjust any seed if needed")
    
        seed_col1, seed_col2 = st.columns(2)
        with seed_col1:
            st.markdown("**🏈 AFC**")
            afc1 = st.selectbox("AFC #1 (Bye)", NFL_TEAMS, index=NFL_TEAMS.index('DEN'), key='afc1')
            afc2 = st.selectbox("AFC #2 (Bye)", NFL_TEAMS, index=NFL_TEAMS.index('NE'),  key='afc2')
            afc3 = st.selectbox("AFC #3",       NFL_TEAMS, index=NFL_TEAMS.index('JAX'), key='afc3')
            afc4 = st.selectbox("AFC #4",       NFL_TEAMS, index=NFL_TEAMS.index('PIT'), key='afc4')
            afc5 = st.selectbox("AFC #5",       NFL_TEAMS, index=NFL_TEAMS.index('HOU'), key='afc5')
            afc6 = st.selectbox("AFC #6",       NFL_TEAMS, index=NFL_TEAMS.index('BUF'), key='afc6')
            afc7 = st.selectbox("AFC #7",       NFL_TEAMS, index=NFL_TEAMS.index('LAC'), key='afc7')
    
        with seed_col2:
            st.markdown("**🏈 NFC**")
            nfc1 = st.selectbox("NFC #1 (Bye)", NFL_TEAMS, index=NFL_TEAMS.index('SEA'), key='nfc1')
            nfc2 = st.selectbox("NFC #2 (Bye)", NFL_TEAMS, index=NFL_TEAMS.index('CHI'), key='nfc2')
            nfc3 = st.selectbox("NFC #3",       NFL_TEAMS, index=NFL_TEAMS.index('PHI'), key='nfc3')
            nfc4 = st.selectbox("NFC #4",       NFL_TEAMS, index=NFL_TEAMS.index('CAR'), key='nfc4')
            nfc5 = st.selectbox("NFC #5",       NFL_TEAMS, index=NFL_TEAMS.index('LA'),  key='nfc5')
            nfc6 = st.selectbox("NFC #6",       NFL_TEAMS, index=NFL_TEAMS.index('SF'),  key='nfc6')
            nfc7 = st.selectbox("NFC #7",       NFL_TEAMS, index=NFL_TEAMS.index('GB'),  key='nfc7')
    
        st.divider()
        st.subheader("🏟️ Super Bowl Conditions")
        sb_c1, sb_c2, sb_c3 = st.columns(3)
        with sb_c1:
            sb_roof = st.selectbox("Stadium", ['dome','outdoors','closed','open'], key='sbroof')
            sb_dome = 1 if sb_roof == 'dome' else 0
        with sb_c2:
            sb_temp = st.slider("Temp (°F)", 20, 100, 72, key='sbtemp') \
                      if sb_roof in ['outdoors','open'] else 72
        with sb_c3:
            sb_wind = st.slider("Wind (mph)", 0, 40, 0, key='sbwind') \
                      if sb_roof in ['outdoors','open'] else 0
    
        st.divider()
    
        def predict_sb_game(home, away, neutral=False):
            home_elo = get_elo(home)
            away_elo = get_elo(away)
            elo_diff = 0 if neutral else (home_elo - away_elo)
            hs  = get_starters(home)
            as_ = get_starters(away)
            h_off, _, _, _, _ = calc_lineup_score(home, hs['QB'],  hs['RB'],  hs['WR'],  hs['TE'])
            a_off, _, _, _, _ = calc_lineup_score(away, as_['QB'], as_['RB'], as_['WR'], as_['TE'])
            adj   = lineup_adjustment(h_off, a_off)
            feats = pd.DataFrame([{
                'elo_diff':    elo_diff,
                'spread_line': -(elo_diff / 25),
                'home_rest':   7,
                'away_rest':   7,
                'temp':        sb_temp,
                'wind':        sb_wind,
                'is_dome':     sb_dome,
                'is_grass':    0,
                'div_game':    0,
            }])
            base = game_model.predict_proba(feats)[0][1]
            return float(np.clip(base + adj, 0.05, 0.95))
    
        def sim_conference(seeds):
            import itertools
            s1,s2,s3,s4,s5,s6,s7 = seeds
            results     = {t: 0.0 for t in seeds}
            wc_matchups = [(s2,s7),(s3,s6),(s4,s5)]
    
            for outcomes in itertools.product([0,1],[0,1],[0,1]):
                path_prob  = 1.0
                wc_winners = []
                for i,(hi,lo) in enumerate(wc_matchups):
                    p = predict_sb_game(hi, lo)
                    if outcomes[i] == 0:
                        wc_winners.append(hi); path_prob *= p
                    else:
                        wc_winners.append(lo); path_prob *= (1 - p)
    
                seed_order = {t:i for i,t in enumerate(seeds)}
                wc_winners.sort(key=lambda t: seed_order[t])
    
                div_pairs = [(s1, wc_winners[2]), (s2, wc_winners[0])]
                p_da = predict_sb_game(div_pairs[0][0], div_pairs[0][1])
                p_db = predict_sb_game(div_pairs[1][0], div_pairs[1][1])
    
                for wa, pa in [(div_pairs[0][0], p_da),(div_pairs[0][1], 1-p_da)]:
                    for wb, pb in [(div_pairs[1][0], p_db),(div_pairs[1][1], 1-p_db)]:
                        div_prob = path_prob * pa * pb
                        p_champ  = predict_sb_game(wa, wb)
                        results[wa] += div_prob * p_champ
                        results[wb] += div_prob * (1 - p_champ)
            return results
    
        if st.button("🔮 Simulate Super Bowl", type="primary", use_container_width=True):
            with st.spinner("🏈 Simulating entire playoff bracket..."):
                afc_seeds = [afc1,afc2,afc3,afc4,afc5,afc6,afc7]
                nfc_seeds = [nfc1,nfc2,nfc3,nfc4,nfc5,nfc6,nfc7]
                afc_probs = sim_conference(afc_seeds)
                nfc_probs = sim_conference(nfc_seeds)
    
            st.divider()
            r1, r2 = st.columns(2)
            with r1:
                st.subheader("🏈 AFC — Conference Win Odds")
                for team, prob in sorted(afc_probs.items(), key=lambda x: x[1], reverse=True):
                    bar = '█' * int(prob * 25)
                    st.write(f"**{team}** {bar} {prob*100:.1f}%  *(ELO: {get_elo(team):.0f})*")
            with r2:
                st.subheader("🏈 NFC — Conference Win Odds")
                for team, prob in sorted(nfc_probs.items(), key=lambda x: x[1], reverse=True):
                    bar = '█' * int(prob * 25)
                    st.write(f"**{team}** {bar} {prob*100:.1f}%  *(ELO: {get_elo(team):.0f})*")
    
            st.divider()
            st.subheader("🏆 Super Bowl Win Probabilities")
    
            sb_probs = {}
            for afc_t, afc_p in afc_probs.items():
                for nfc_t, nfc_p in nfc_probs.items():
                    match_prob = afc_p * nfc_p
                    sb_win     = predict_sb_game(afc_t, nfc_t, neutral=True)
                    sb_probs[afc_t] = sb_probs.get(afc_t, 0) + match_prob * sb_win
                    sb_probs[nfc_t] = sb_probs.get(nfc_t, 0) + match_prob * (1 - sb_win)
    
            sb_sorted  = sorted(sb_probs.items(), key=lambda x: x[1], reverse=True)
            top5       = sb_sorted[:5]
            t_cols     = st.columns(5)
            for i,(team,prob) in enumerate(top5):
                conf = "AFC" if team in afc_seeds else "NFC"
                t_cols[i].metric(f"#{i+1} {team}", f"{prob*100:.1f}%",
                                 delta=f"{conf} • ELO {get_elo(team):.0f}")
    
            st.divider()
            champion  = sb_sorted[0][0]
            runner_up = sb_sorted[1][0]
            ch_conf   = "AFC" if champion  in afc_seeds else "NFC"
            ru_conf   = "AFC" if runner_up in afc_seeds else "NFC"
    
            st.subheader("🎯 Model's Predicted Super Bowl")
            ch1, ch2 = st.columns(2)
            with ch1:
                st.metric("🏆 Predicted Champion", champion,
                          delta=f"{ch_conf} • {sb_probs[champion]*100:.1f}% SB win prob")
            with ch2:
                st.metric("🥈 Runner Up", runner_up,
                          delta=f"{ru_conf} • {sb_probs[runner_up]*100:.1f}% SB win prob")
    
            with st.expander("📊 Full odds — all 14 playoff teams"):
                sb_df = pd.DataFrame(sb_sorted, columns=['Team','Win Prob'])
                sb_df['Win Prob']   = (sb_df['Win Prob']*100).round(1).astype(str) + '%'
                sb_df['ELO']        = sb_df['Team'].apply(lambda t: f"{get_elo(t):.0f}")
                sb_df['Conference'] = sb_df['Team'].apply(
                    lambda t: 'AFC' if t in afc_seeds else 'NFC')
                st.dataframe(sb_df.reset_index(drop=True), use_container_width=True)
    
    # ══════════════════════════════════════════════════
    # TAB 5: BACKTESTING (MODEL VS ACTUAL)
    # ══════════════════════════════════════════════════
    with tab5:
        st.header("📅 Model Backtesting: Predictions vs Actuals")
        st.caption("See how the model would have performed using only pre-game data. Updates as the model improves.")
        st.divider()
    
        _ORIG_FEATURES = ['elo_diff','spread_line','home_rest','away_rest','temp','wind','is_dome','is_grass','div_game']
        _bt_features   = enhanced_features if enhanced_features else _ORIG_FEATURES
    
        if enhanced_features:
            _bt_games = load_enhanced_games()
            if _bt_games.empty:
                st.warning("Could not compute enhanced features — falling back to original 9 features.")
                _bt_features = _ORIG_FEATURES
                _bt_games    = games
        else:
            _bt_games = games
    
        _keep = [c for c in _bt_features + ['home_win','season','week','home_team','away_team','home_score','away_score']
                 if c in _bt_games.columns]
        data = _bt_games[_keep].dropna().copy().reset_index(drop=True)
    
        # Filter for last 5 years
        recent_years = sorted(data['season'].unique())[-5:]
        data5 = data[data['season'].isin(recent_years)].copy()
    
        # Predict using current model
        try:
            X     = data5[[c for c in _bt_features if c in data5.columns]]
            probs = game_model.predict_proba(X)[:,1]
        except Exception as e:
            st.error(f"Prediction error (feature mismatch): {e}")
            st.stop()
    
        preds = (probs >= 0.5).astype(int)
        data5['predicted'] = preds
        data5['prob'] = probs
        data5['predicted_winner'] = np.where(preds == 1, data5['home_team'], data5['away_team'])
        data5['actual_winner'] = np.where(data5['home_win'] == 1, data5['home_team'], data5['away_team'])
        data5['correct'] = (data5['predicted'] == data5['home_win'])
    
        # Summary stats
        acc      = data5['correct'].mean()
        baseline = max(data5['home_win'].mean(), 1 - data5['home_win'].mean())
        model_label = f"Enhanced ({len(_bt_features)}-feature)" if enhanced_features else "Original (9-feature)"
        st.metric(f"Overall Accuracy — {model_label} (last 5 seasons)",
                  f"{acc*100:.1f}%", delta=f"vs coin-flip baseline: {baseline*100:.1f}%")
    
        # Accuracy by season
        by_season = data5.groupby('season')['correct'].mean().reset_index()
        st.subheader("Accuracy by Season")
        st.dataframe(
            by_season.rename(columns={'correct':'Accuracy'})
                     .assign(Accuracy=lambda df: (df['Accuracy']*100).round(1).astype(str)+'%'),
            use_container_width=True
        )
    
        st.divider()
        st.subheader("Game-by-Game Results (last 5 seasons)")
        year = st.selectbox("Season", options=recent_years, index=len(recent_years)-1)
        show = data5[data5['season'] == year].copy()
        show['Confidence'] = (show['prob']*100).round(1).astype(str) + '%'
        show['Result'] = np.where(show['correct'], '✅', '❌')
        st.dataframe(
            show[['week','home_team','away_team','home_score','away_score',
                  'predicted_winner','actual_winner','Confidence','Result']]
                .sort_values('week'),
            use_container_width=True
        )
        st.caption("✅ = correct prediction. Confidence = model's home-win probability. Enhanced model uses 27 features.")
    
        # ── $10 Moneyline Simulation ──────────────────────────────────────────────
        st.divider()
        st.subheader("💰 $10 Moneyline Simulation")
        st.caption("What would have happened if you bet $10/game on the model's picks at actual Vegas moneylines?")
    
        _ml_cols = ['season', 'week', 'home_team', 'away_team', 'home_moneyline', 'away_moneyline']
        _ml_ok = all(c in games.columns for c in _ml_cols)
        if _ml_ok:
            _ml_raw = games[_ml_cols].dropna(subset=['home_moneyline', 'away_moneyline'])
            data5_ml = data5.merge(_ml_raw, on=['season', 'week', 'home_team', 'away_team'], how='left')
            data5_ml = data5_ml.dropna(subset=['home_moneyline', 'away_moneyline']).copy()
        else:
            data5_ml = pd.DataFrame()
    
        if len(data5_ml) < 10:
            st.info("Moneyline data not available for the selected seasons (data available from ~2009 onward).")
        else:
            def _ml_pnl(odds, won, stake=10.0):
                """Return P&L for a single $10 moneyline bet (American odds)."""
                if pd.isna(odds) or odds == 0:
                    return 0.0
                if won:
                    return (stake * odds / 100.0) if odds > 0 else (stake * 100.0 / abs(odds))
                return -stake
    
            def _implied_prob(ml):
                return abs(ml) / (abs(ml) + 100) if ml < 0 else 100 / (ml + 100)
    
            def _model_bet(row):
                """Bet on model's predicted winner at their moneyline."""
                if row['predicted'] == 1:
                    return _ml_pnl(row['home_moneyline'], row['home_win'] == 1)
                return _ml_pnl(row['away_moneyline'], row['home_win'] == 0)
    
            def _home_bet(row):
                """Always bet the home team."""
                return _ml_pnl(row['home_moneyline'], row['home_win'] == 1)
    
            def _fav_bet(row):
                """Always bet the moneyline favorite."""
                home_fav = _implied_prob(row['home_moneyline']) >= _implied_prob(row['away_moneyline'])
                if home_fav:
                    return _ml_pnl(row['home_moneyline'], row['home_win'] == 1)
                return _ml_pnl(row['away_moneyline'], row['home_win'] == 0)
    
            data5_ml['pnl_model']    = data5_ml.apply(_model_bet, axis=1)
            data5_ml['pnl_home']     = data5_ml.apply(_home_bet,  axis=1)
            data5_ml['pnl_favorite'] = data5_ml.apply(_fav_bet,   axis=1)
    
            n_games   = len(data5_ml)
            wagered   = n_games * 10.0
            model_net = data5_ml['pnl_model'].sum()
            home_net  = data5_ml['pnl_home'].sum()
            fav_net   = data5_ml['pnl_favorite'].sum()
            model_roi = (model_net / wagered) * 100
            home_roi  = (home_net  / wagered) * 100
            fav_roi   = (fav_net   / wagered) * 100
            fav_wins  = (data5_ml['pnl_favorite'] > 0).mean() * 100
    
            # Headline metrics
            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
            _mc1.metric("Model Net P&L", f"${model_net:+.0f}",
                        help="Cumulative profit/loss betting $10/game on model's picks")
            _mc2.metric("Model ROI", f"{model_roi:+.1f}%",
                        help="Return on total dollars wagered")
            _mc3.metric("Games Simulated", f"{n_games:,}",
                        help="Games with available Vegas moneyline data")
            _mc4.metric("Total Wagered", f"${wagered:,.0f}",
                        help="$10 × number of games")
    
            # Strategy comparison table
            _comp_df = pd.DataFrame({
                'Strategy':  ['🤖 Model Picks', '🏠 Always Home', '⭐ Always Favorite'],
                'Net P&L':   [f'${model_net:+.0f}', f'${home_net:+.0f}', f'${fav_net:+.0f}'],
                'ROI':       [f'{model_roi:+.1f}%', f'{home_roi:+.1f}%', f'{fav_roi:+.1f}%'],
                'Win Rate':  [
                    f"{data5_ml['correct'].mean()*100:.1f}%",
                    f"{data5_ml['home_win'].mean()*100:.1f}%",
                    f"{fav_wins:.1f}%",
                ],
            })
            st.dataframe(_comp_df, use_container_width=True, hide_index=True)
    
            # Cumulative P&L chart
            import plotly.graph_objects as _go
            _sorted = data5_ml.sort_values(['season', 'week']).reset_index(drop=True)
            _sorted['game_num']      = range(1, len(_sorted) + 1)
            _sorted['cum_model']     = _sorted['pnl_model'].cumsum()
            _sorted['cum_home']      = _sorted['pnl_home'].cumsum()
            _sorted['cum_favorite']  = _sorted['pnl_favorite'].cumsum()
    
            fig_ml = _go.Figure()
            fig_ml.add_trace(_go.Scatter(x=_sorted['game_num'], y=_sorted['cum_model'],
                                         name='Model Picks', line=dict(color='#00c4cc', width=2)))
            fig_ml.add_trace(_go.Scatter(x=_sorted['game_num'], y=_sorted['cum_home'],
                                         name='Always Home', line=dict(color='#888', width=1.5, dash='dot')))
            fig_ml.add_trace(_go.Scatter(x=_sorted['game_num'], y=_sorted['cum_favorite'],
                                         name='Always Favorite', line=dict(color='#ff7f0e', width=1.5, dash='dash')))
            fig_ml.add_hline(y=0, line_color='white', line_dash='dot', opacity=0.3)
            fig_ml.update_layout(
                title='Cumulative P&L — $10/game Moneyline Bets (last 5 seasons)',
                xaxis_title='Game # (chronological)',
                yaxis_title='Cumulative P&L ($)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                height=360,
                margin=dict(l=0, r=0, t=50, b=0),
            )
            st.plotly_chart(fig_ml, use_container_width=True)
    
            st.caption(
                "Note: Even a model with >50% accuracy typically shows negative ROI on moneyline bets "
                "because favorites pay out less than $10 profit per win (you need >52.4% win rate on -110 juice). "
                "Positive ROI requires consistently picking upsets that beat Vegas expectations."
            )
    
        # ── Kelly Strategy Backtest ───────────────────────────────────────────────
        st.divider()
        st.subheader("📈 Kelly Criterion Strategy Backtest")
        st.caption(
            "Simulates variable bet sizing over the last 5 seasons. "
            "**Your bankroll is your total account** — each game you bet a small % of it based on the model's edge vs Vegas. "
            "Wins compound into the next bet; losses reduce the next bet automatically. "
            "Max bet capped at 2% per game (conservative, real-world appropriate)."
        )
    
        _ml_cols2 = ['season', 'week', 'home_team', 'away_team', 'home_moneyline', 'away_moneyline']
        if all(c in games.columns for c in _ml_cols2):
            _ml_raw2   = games[_ml_cols2].dropna(subset=['home_moneyline', 'away_moneyline'])
            _data5_ml2 = data5.merge(_ml_raw2, on=['season','week','home_team','away_team'], how='left')
            _data5_ml2 = _data5_ml2.dropna(subset=['home_moneyline', 'away_moneyline']).copy()
        else:
            _data5_ml2 = pd.DataFrame()
    
        if len(_data5_ml2) < 10:
            st.info("Moneyline data not available for this period.")
        else:
            _bk1, _bk2 = st.columns(2)
            _start_br  = _bk1.number_input(
                "Starting bankroll ($)", min_value=100, max_value=1_000_000,
                value=1000, step=100, key='bt_bankroll'
            )
            _kfrac_opt = _bk2.selectbox(
                "Kelly fraction",
                ['Half-Kelly (0.5×) — Recommended', 'Quarter-Kelly (0.25×) — Conservative', 'Full Kelly (1.0×) — Aggressive'],
                key='bt_kfrac'
            )
            _kfrac_val = {
                'Half-Kelly (0.5×) — Recommended':    0.50,
                'Quarter-Kelly (0.25×) — Conservative': 0.25,
                'Full Kelly (1.0×) — Aggressive':      1.00,
            }[_kfrac_opt]
    
            # 2% max per game — conservative cap that prevents compounding blowup
            # while still showing whether Kelly sizing outperforms flat betting
            _BT_MAX_PCT = 2.0
    
            # Simulate Kelly bets game-by-game
            _br       = float(_start_br)
            _br_flat  = float(_start_br)
            _br_hist  = []
            _flat_hist = []
            _max_br   = _br
            _max_dd   = 0.0
            _bets_placed = 0
            _sorted_k = _data5_ml2.sort_values(['season', 'week']).reset_index(drop=True)
    
            for _, _row in _sorted_k.iterrows():
                _pick_home_k2 = int(_row['predicted']) == 1
                _prob_k2 = float(_row['prob']) if _pick_home_k2 else (1.0 - float(_row['prob']))
                _ml_k2   = float(_row['home_moneyline']) if _pick_home_k2 else float(_row['away_moneyline'])
                _won_k2  = (int(_row['home_win']) == 1) if _pick_home_k2 else (int(_row['home_win']) == 0)
    
                # Kelly bet — capped at _BT_MAX_PCT% to prevent in-sample blowup
                _kpct_k2, _, _ = _kelly_rec(_prob_k2, _ml_k2, _kfrac_val)
                _kpct_k2 = min(_kpct_k2, _BT_MAX_PCT)
                _stake_k2 = _br * (_kpct_k2 / 100.0)
                if _stake_k2 > 0 and _ml_k2 != 0:
                    _b_k2 = (100.0 / abs(_ml_k2)) if _ml_k2 < 0 else (_ml_k2 / 100.0)
                    _br  += _stake_k2 * _b_k2 if _won_k2 else -_stake_k2
                    _bets_placed += 1
    
                # Flat $10 comparison
                if _ml_k2 != 0:
                    _b_flat2 = (100.0 / abs(_ml_k2)) if _ml_k2 < 0 else (_ml_k2 / 100.0)
                    _br_flat += 10.0 * _b_flat2 if _won_k2 else -10.0
    
                _br = max(_br, 0.01)  # can't go below 1 cent
                _max_br = max(_max_br, _br)
                _max_dd = max(_max_dd, (_max_br - _br) / _max_br * 100.0)
                _br_hist.append(_br)
                _flat_hist.append(_br_flat)
    
            _skipped_k    = len(_sorted_k) - _bets_placed
            _kelly_final  = _br / _start_br * 100   # index, starts at 100
            _flat_final   = _br_flat / _start_br * 100
            _kelly_idx    = [v / _start_br * 100 for v in _br_hist]
            _flat_idx     = [v / _start_br * 100 for v in _flat_hist]
    
            # Metrics — show relative performance, not misleading absolute dollars
            _km1, _km2, _km3, _km4 = st.columns(4)
            _km1.metric("Kelly vs Starting (index)", f"{_kelly_final:.0f}",
                        delta=f"{_kelly_final - 100:+.0f} pts",
                        help="100 = break even. 120 = 20% gain. 80 = 20% loss.")
            _km2.metric("Flat $10 vs Starting (index)", f"{_flat_final:.0f}",
                        delta=f"{_flat_final - 100:+.0f} pts",
                        help="Same scale — compare Kelly vs flat betting directly.")
            _km3.metric("Max Drawdown (Kelly)", f"{_max_dd:.1f}%",
                        help="Largest peak-to-trough loss during the 5-season simulation")
            _km4.metric("Games Skipped (PASS)", f"{_skipped_k:,}",
                        help="Games where Kelly found no edge vs Vegas — no bet placed")
    
            st.caption(
                "⚠️ **Index chart only — do not use as a dollar profit forecast.** "
                "Most of these 5 seasons are in-sample (the model trained on them), "
                "so absolute returns are inflated. The chart shows whether *variable sizing* "
                "beats *flat betting* in relative terms — that comparison is valid regardless of scale."
            )
    
            # Index chart: both lines start at 100
            import plotly.graph_objects as _go3
            _gn3 = list(range(1, len(_sorted_k) + 1))
            _kfrac_short = _kfrac_opt.split('(')[0].strip()
            fig_kelly = _go3.Figure()
            fig_kelly.add_trace(_go3.Scatter(
                x=_gn3, y=_kelly_idx,
                name=f'{_kfrac_short} (variable sizing)',
                line=dict(color='#00c4cc', width=2)
            ))
            fig_kelly.add_trace(_go3.Scatter(
                x=_gn3, y=_flat_idx,
                name='Flat bet (same games)',
                line=dict(color='#ff7f0e', width=1.5, dash='dash')
            ))
            fig_kelly.add_hline(y=100, line_color='white', line_dash='dot', opacity=0.4,
                                annotation_text='Break even', annotation_position='left')
            fig_kelly.update_layout(
                title=f'Relative Performance Index: {_kfrac_short} vs Flat Betting (100 = start)',
                xaxis_title='Game # (chronological, last 5 seasons)',
                yaxis_title='Performance Index (start = 100)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                height=360,
                margin=dict(l=0, r=0, t=50, b=0),
            )
            st.plotly_chart(fig_kelly, use_container_width=True)
    
            st.caption(
                f"Kelly bets more when the model finds edge over Vegas, skips when there's none "
                f"({_skipped_k} games skipped). Max 2% of bankroll per game. "
                f"If the Kelly line stays above the flat line, variable sizing is adding value."
            )

    # ══════════════════════════════════════════════════
    # TAB 6: TRACK RECORD
    # ══════════════════════════════════════════════════
    with tab6:
        st.header("📋 Track Record")
        st.caption("Auto-logged predictions from this session · Log bets you placed · Export your data")
        st.divider()

        if not _PH_OK:
            st.error("prediction_history.py not found — Track Record unavailable.")
        else:
            _tr_h, _tr_b, _tr_e = st.tabs(["📊 Prediction History", "💰 Bet Tracker", "📤 Export / Import"])

            # ── Prediction History ────────────────────────────────────────────
            with _tr_h:
                _tr_col1, _tr_col2 = st.columns([6, 2])
                with _tr_col2:
                    if st.button("🔄 Refresh Results", key='nfl_tr_refresh', use_container_width=True,
                                 help="Check ESPN for completed games and fill in actual results"):
                        with st.spinner("Backfilling results..."):
                            _n = _ph.backfill_results()
                        if _n:
                            st.success(f"Updated {_n} game result(s)")
                        else:
                            st.info("No new results found")

                _all_preds = _ph.load_predictions()
                _nfl_preds = [p for p in _all_preds if p.get('sport') == 'nfl']

                if not _nfl_preds:
                    st.info("No NFL predictions logged yet. Load the schedule and predictions will appear here automatically.")
                else:
                    # Filters
                    _fc1, _fc2, _fc3 = st.columns(3)
                    with _fc1:
                        _sig_opts = ["All Signals", "💎 STRONG", "📈 LEAN", "👀 SMALL", "⚪ PASS"]
                        _sig_filter = st.selectbox("Signal", _sig_opts, key='nfl_tr_sig_filter')
                    with _fc2:
                        _out_opts = ["All Outcomes", "✅ Correct", "❌ Incorrect", "⏳ Pending"]
                        _out_filter = st.selectbox("Outcome", _out_opts, key='nfl_tr_out_filter')
                    with _fc3:
                        _sort_opts = ["Newest First", "Oldest First"]
                        _sort_sel = st.selectbox("Sort", _sort_opts, key='nfl_tr_sort')

                    # Apply filters
                    _sig_map = {"💎 STRONG": "STRONG", "📈 LEAN": "LEAN", "👀 SMALL": "SMALL", "⚪ PASS": "PASS"}
                    _fp = _nfl_preds[:]
                    if _sig_filter != "All Signals":
                        _fp = [p for p in _fp if p.get('kelly_signal') == _sig_map.get(_sig_filter)]
                    if _out_filter == "✅ Correct":
                        _fp = [p for p in _fp if p.get('prediction_correct') is True]
                    elif _out_filter == "❌ Incorrect":
                        _fp = [p for p in _fp if p.get('prediction_correct') is False]
                    elif _out_filter == "⏳ Pending":
                        _fp = [p for p in _fp if p.get('prediction_correct') is None]

                    _fp.sort(key=lambda x: x.get('predicted_at', ''), reverse=(_sort_sel == "Newest First"))

                    # Build display dataframe
                    _rows = []
                    for _p in _fp:
                        _winner_pred = _p['home_team'] if _p.get('model_home_prob', 0.5) >= 0.5 else _p['away_team']
                        _correct = _p.get('prediction_correct')
                        _outcome = "✅" if _correct is True else ("❌" if _correct is False else "⏳")
                        _actual = _p.get('actual_winner', '—') or '—'
                        _score = f"{_p.get('actual_score_home','')}-{_p.get('actual_score_away','')}" if _p.get('actual_score_home') is not None else '—'
                        _badge_map = {'STRONG': '💎', 'LEAN': '📈', 'SMALL': '👀', 'PASS': '⚪'}
                        _badge = _badge_map.get(_p.get('kelly_signal', 'PASS'), '⚪')
                        _rows.append({
                            'Date': _p.get('game_date', ''),
                            'Matchup': f"{_p['away_team']} @ {_p['home_team']}",
                            'Pick': _winner_pred,
                            'Win Prob': f"{_p.get('model_home_prob', 0.5)*100:.1f}%" if _winner_pred == _p['home_team'] else f"{(1-_p.get('model_home_prob',0.5))*100:.1f}%",
                            'Edge': f"{_p.get('model_edge_pct', 0):+.1f}%",
                            'Signal': f"{_badge} {_p.get('kelly_signal', 'PASS')}",
                            'Result': _actual,
                            'Score': _score,
                            'O/U': '✅' if _p.get('ou_correct') is True else ('❌' if _p.get('ou_correct') is False else '—'),
                            '': _outcome,
                        })

                    _df = pd.DataFrame(_rows)
                    st.dataframe(_df, use_container_width=True, hide_index=True)

                    # Summary stats
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

            # ── Bet Tracker ───────────────────────────────────────────────────
            with _tr_b:
                _all_preds_bt = _ph.load_predictions()
                _nfl_preds_bt = [p for p in _all_preds_bt if p.get('sport') == 'nfl']
                _all_bets = [b for b in _ph.load_bets() if b.get('sport') == 'nfl']

                # Log a new bet
                with st.expander("➕ Log a Bet", expanded=not bool(_all_bets)):
                    if not _nfl_preds_bt:
                        st.info("No NFL predictions logged yet — load the schedule first.")
                    else:
                        _pred_options = {
                            f"{p['away_team']} @ {p['home_team']} ({p.get('game_date','')})": p['id']
                            for p in sorted(_nfl_preds_bt, key=lambda x: x.get('game_date',''), reverse=True)
                        }
                        with st.form("nfl_log_bet_form"):
                            _sel_label = st.selectbox("Game", list(_pred_options.keys()), key='nfl_bt_game_sel')
                            _sel_pred_id = _pred_options[_sel_label]
                            _sel_pred = next((p for p in _nfl_preds_bt if p['id'] == _sel_pred_id), {})
                            _home_t = _sel_pred.get('home_team', '')
                            _away_t = _sel_pred.get('away_team', '')
                            _bt_c1, _bt_c2 = st.columns(2)
                            with _bt_c1:
                                _pick_team = st.selectbox("Bet On", [_home_t, _away_t] if _home_t else ["—"])
                                _bet_amount = st.number_input("Bet Amount ($)", min_value=1.0, max_value=100_000.0,
                                                              value=50.0, step=5.0)
                            with _bt_c2:
                                _odds_taken = st.number_input("Odds (American)", value=-110, format="%d",
                                                              help="e.g. -150 for favorite, +130 for underdog")
                                _sportsbook = st.text_input("Sportsbook", value="DraftKings", max_chars=30)
                            _submit_bet = st.form_submit_button("Log Bet", type="primary", use_container_width=True)
                            if _submit_bet:
                                _bet_rec = {
                                    'id': f"bet_nfl_{_sel_pred_id}_{int(datetime.now().timestamp())}",
                                    'prediction_id': _sel_pred_id,
                                    'sport': 'nfl',
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

                # Bets table + P&L
                if _all_bets:
                    _pred_lookup = {p['id']: p for p in _nfl_preds_bt}
                    _bet_rows = []
                    for _b in sorted(_all_bets, key=lambda x: x.get('logged_at', ''), reverse=True):
                        _pred = _pred_lookup.get(_b.get('prediction_id', ''), {})
                        _actual_w = _pred.get('actual_winner')
                        _result_str = '⏳'
                        _pnl = None
                        if _actual_w is not None:
                            _odds_b = _b.get('odds', -110)
                            _dec_b = (1 + 100.0 / abs(_odds_b)) if _odds_b < 0 else (1 + _odds_b / 100.0)
                            if _b.get('pick') == _actual_w:
                                _pnl = _b['amount'] * (_dec_b - 1)
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

                    # Summary metrics
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

                    # P&L chart
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

            # ── Export / Import ───────────────────────────────────────────────
            with _tr_e:
                import json as _json
                _exp_c1, _exp_c2 = st.columns(2)
                with _exp_c1:
                    st.markdown("**📥 Export Data**")
                    _all_p_exp = _ph.load_predictions()
                    _all_b_exp = _ph.load_bets()
                    _export_pkg = {'predictions': _all_p_exp, 'bets': _all_b_exp, 'exported_at': datetime.now().isoformat()}
                    st.download_button(
                        "Download All Data (JSON)",
                        data=_json.dumps(_export_pkg, indent=2, default=str),
                        file_name=f"edgeiq_track_record_{date.today().isoformat()}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download Predictions Only (JSON)",
                        data=_json.dumps(_all_p_exp, indent=2, default=str),
                        file_name=f"edgeiq_predictions_{date.today().isoformat()}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download Bets Only (JSON)",
                        data=_json.dumps(_all_b_exp, indent=2, default=str),
                        file_name=f"edgeiq_bets_{date.today().isoformat()}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                with _exp_c2:
                    st.markdown("**📤 Import Data**")
                    _up = st.file_uploader("Upload EdgeIQ JSON backup", type=["json"], key='nfl_import_upload')
                    if _up is not None:
                        try:
                            _imported = _json.load(_up)
                            _imp_preds = _imported.get('predictions', _imported) if isinstance(_imported, dict) else _imported
                            _imp_bets = _imported.get('bets', []) if isinstance(_imported, dict) else []
                            if st.button("Confirm Import", key='nfl_confirm_import', type="primary"):
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
