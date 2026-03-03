"""
build_nhl_player_model.py
=========================
Trains 3 NHL player prop models:
  1. Goals per game (regression)
  2. Assists per game (regression)
  3. Shots on goal per game (regression)

Data:
  - NHL Stats REST API skater/summary  → season aggregate features per player
  - NHL API player/{id}/game-log       → per-game outcomes (training targets)
  - nhl_team_stats_historical.csv      → opponent context (goals_against_pg, shots_against_pg, pk_pct)

Output:
  - model_nhl_player.pkl         : 3 GBR models + feature lists + MAE per model
  - nhl_skater_current_stats.csv : current 2025-26 season per-player rates (runtime)
  - .nhl_prop_log_cache.json     : API response cache (speeds up re-runs)
"""

import json
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── Constants ───────────────────────────────────────────────────────────────────

_STATS_BASE  = "https://api.nhle.com/stats/rest/en"
_API_BASE    = "https://api-web.nhle.com/v1"
_SESS        = requests.Session()
_SESS.headers['User-Agent'] = 'EdgeIQ/1.0 (educational)'

# season_id format: {start_year}{end_year}  e.g. 20242025
# season column in nhl_team_stats_historical = start_year (int)
TRAIN_SEASON_IDS   = ['20202021', '20212022', '20222023', '20232024', '20242025']
CURRENT_SEASON_ID  = '20252026'
ALL_SEASON_IDS     = TRAIN_SEASON_IDS + [CURRENT_SEASON_ID]

TOP_SKATERS        = 200   # top N per season by points
MIN_GP             = 20    # filter out injury-shortened seasons
MIN_GAME_LOG_GP    = 10    # minimum games in a player-season log to include

_CACHE_FILE = '.nhl_prop_log_cache.json'

_FWD_POSITIONS = {'C', 'L', 'R', 'LW', 'RW', 'F'}

# ── Local game-log cache ─────────────────────────────────────────────────────────

def _load_cache():
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_cache(cache):
    with open(_CACHE_FILE, 'w') as f:
        json.dump(cache, f)

# ── API helpers ───────────────────────────────────────────────────────────────────

def _get(url, retries=3, delay=0.5):
    for attempt in range(retries):
        try:
            r = _SESS.get(url, timeout=20)
            r.raise_for_status()
            time.sleep(delay)
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [warn] GET failed: {url}  {e}")
    return None


def fetch_skater_summary(season_id, max_results=300):
    """Return list of skater aggregate dicts for a season (sorted by points desc).
    Paginates automatically since the API caps each response at 100 records."""
    PAGE_SIZE = 100
    all_data = []
    start = 0
    while len(all_data) < max_results:
        url = (
            f"{_STATS_BASE}/skater/summary"
            f"?isAggregate=false&isGame=false"
            f"&sort=%5B%7B%22property%22%3A%22points%22%2C%22direction%22%3A%22DESC%22%7D%5D"
            f"&start={start}&limit={PAGE_SIZE}"
            f"&cayenneExp=seasonId%3D{season_id}%20and%20gameTypeId%3D2"
        )
        data = _get(url)
        if not data:
            break
        page = data.get('data', [])
        if not page:
            break
        all_data.extend(page)
        if len(page) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return all_data[:max_results]


def fetch_game_log(player_id, season_id, cache):
    """Fetch per-game log for a player-season; returns list of game dicts."""
    key = f"{player_id}_{season_id}"
    if key in cache:
        return cache[key]
    url = f"{_API_BASE}/player/{player_id}/game-log/{season_id}/2"
    data = _get(url, delay=0.1)
    log = data.get('gameLog', []) if data else []
    cache[key] = log
    return log


# ── Feature building ───────────────────────────────────────────────────────────────

GOALS_FEATURES   = ['goals_pg', 'shots_pg', 'shooting_pct', 'toi_pg', 'pp_goals_pg',
                     'is_forward', 'opp_goals_against_pg', 'opp_shots_against_pg',
                     'opp_pk_pct', 'is_home']

ASSISTS_FEATURES = ['assists_pg', 'shots_pg', 'toi_pg', 'pp_goals_pg', 'is_forward',
                    'opp_goals_against_pg', 'opp_shots_against_pg', 'opp_pk_pct', 'is_home']

SHOTS_FEATURES   = ['shots_pg', 'toi_pg', 'is_forward', 'goals_pg',
                     'opp_shots_against_pg', 'opp_goals_against_pg', 'is_home']


def _toi_to_seconds(toi_str):
    """Convert 'MM:SS' → float seconds. Returns 0 if unparseable."""
    try:
        m, s = str(toi_str).split(':')
        return int(m) * 60 + int(s)
    except Exception:
        return 0.0


def _season_year(season_id):
    """20242025 → 2024  (start year = historical team stats join key)"""
    return int(str(season_id)[:4])


def build_training_data(team_hist):
    """
    Fetch all player game logs for TRAIN_SEASON_IDS and build a flat
    DataFrame of feature rows, one per player-game.

    Returns: DataFrame
    """
    cache = _load_cache()
    rows = []
    total_logs = 0

    for season_id in TRAIN_SEASON_IDS:
        yr = _season_year(season_id)
        print(f"\n-- Season {season_id} (year={yr}) --")

        # Opponent context lookup: {team_abbrev → {goals_against_pg, shots_against_pg, pk_pct}}
        opp_ctx = {}
        s_hist = team_hist[team_hist['season'] == yr]
        for _, row in s_hist.iterrows():
            opp_ctx[row['team']] = {
                'opp_goals_against_pg':  row.get('goals_against_pg', 2.8),
                'opp_shots_against_pg':  row.get('shots_against_pg', 29.0),
                'opp_pk_pct':            row.get('pk_pct', 0.80),
            }

        skaters = fetch_skater_summary(season_id, max_results=TOP_SKATERS + 50)
        qualified = [s for s in skaters if s.get('gamesPlayed', 0) >= MIN_GP]
        qualified = qualified[:TOP_SKATERS]
        print(f"  {len(qualified)} qualified skaters")

        for sk in qualified:
            pid   = sk.get('playerId')
            gp    = sk.get('gamesPlayed', 1) or 1
            pos   = sk.get('positionCode', 'C').upper()

            # Season-level rates (used as features)
            goals_pg    = (sk.get('goals', 0) or 0) / gp
            assists_pg  = (sk.get('assists', 0) or 0) / gp
            shots_pg    = (sk.get('shots', 0) or 0) / gp
            shoot_pct   = sk.get('shootingPct') or 0.0
            toi_pg_s    = sk.get('timeOnIcePerGame') or 900.0  # seconds
            pp_goals_pg = (sk.get('ppGoals', 0) or 0) / gp
            is_forward  = 1 if pos in _FWD_POSITIONS else 0

            log = fetch_game_log(pid, season_id, cache)
            if len(log) < MIN_GAME_LOG_GP:
                continue
            total_logs += len(log)

            for g in log:
                opp = g.get('opponentAbbrev', '')
                ctx = opp_ctx.get(opp, {})
                is_home = 1 if g.get('homeRoadFlag', 'R') == 'H' else 0

                row = {
                    # targets
                    'target_goals':   int(g.get('goals', 0) or 0),
                    'target_assists': int(g.get('assists', 0) or 0),
                    'target_shots':   int(g.get('shots', 0) or 0),
                    # player features
                    'goals_pg':       goals_pg,
                    'assists_pg':     assists_pg,
                    'shots_pg':       shots_pg,
                    'shooting_pct':   shoot_pct,
                    'toi_pg':         toi_pg_s,
                    'pp_goals_pg':    pp_goals_pg,
                    'is_forward':     is_forward,
                    # context
                    'is_home':        is_home,
                    'opp_goals_against_pg': ctx.get('opp_goals_against_pg', 2.8),
                    'opp_shots_against_pg': ctx.get('opp_shots_against_pg', 29.0),
                    'opp_pk_pct':          ctx.get('opp_pk_pct', 0.80),
                    # metadata
                    'season':         yr,
                    'player_id':      pid,
                }
                rows.append(row)

        # Checkpoint-save cache after each season
        _save_cache(cache)
        print(f"  Total game-log rows so far: {total_logs}")

    _save_cache(cache)
    return pd.DataFrame(rows)


# ── Model training ─────────────────────────────────────────────────────────────────

_CANDIDATES = {
    'GBR': GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=42,
    ),
    'Poisson': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  PoissonRegressor(alpha=1.0, max_iter=500)),
    ]),
    'Ridge': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  Ridge(alpha=1.0)),
    ]),
}


def evaluate_and_select(df, target_col, feature_cols, label):
    """
    Temporal leave-one-season-out CV across all training seasons.
    Compares GBR, PoissonRegressor, and Ridge.
    Picks the winner by lowest MAE, checks P(1+) calibration, then
    retrains the winner on the full dataset.

    Returns: (best_model, honest_mae, winner_name)
    """
    X      = df[feature_cols].values
    y      = df[target_col].values
    groups = df['season'].values

    logo = LeaveOneGroupOut()
    res  = {name: {'maes': [], 'y_pred': [], 'y_true': []} for name in _CANDIDATES}

    for train_idx, test_idx in logo.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        for name, mdl in _CANDIDATES.items():
            m = clone(mdl)
            m.fit(X_tr, y_tr)
            preds = np.clip(m.predict(X_te), 0, None)
            res[name]['maes'].append(mean_absolute_error(y_te, preds))
            res[name]['y_pred'].extend(preds.tolist())
            res[name]['y_true'].extend(y_te.tolist())

    print(f"\n  {label} — temporal CV (leave-one-season-out):")
    print(f"  {'Model':<20}  {'MAE':>7}  {'Brier P(1+)':>11}")
    print(f"  {'-'*20}  {'-'*7}  {'-'*11}")

    best_name, best_mae = None, float('inf')
    for name, r in res.items():
        avg_mae = float(np.mean(r['maes']))
        pp      = 1.0 - np.exp(-np.clip(np.array(r['y_pred']), 0, None))
        pt      = (np.array(r['y_true']) >= 1).astype(float)
        brier   = brier_score_loss(pt, np.clip(pp, 1e-6, 1 - 1e-6))
        if avg_mae < best_mae:
            best_mae  = avg_mae
            best_name = name
        print(f"  {name:<20}  {avg_mae:>7.3f}  {brier:>11.4f}")

    print(f"  -> Best: {best_name}  (honest MAE +-{best_mae:.3f})")

    # P(1+) calibration check — 5 equal-frequency buckets
    r  = res[best_name]
    pp = 1.0 - np.exp(-np.clip(np.array(r['y_pred']), 0, None))
    pt = (np.array(r['y_true']) >= 1).astype(float)
    edges = np.percentile(pp, [0, 20, 40, 60, 80, 100])
    print(f"\n  {label} P(1+) calibration ({best_name}):")
    for i in range(5):
        hi   = edges[i + 1] if i < 4 else edges[-1] + 1e-9
        mask = (pp >= edges[i]) & (pp < hi)
        if mask.sum() > 10:
            pm  = pp[mask].mean()
            am  = pt[mask].mean()
            sym = 'ok' if abs(am - pm) < 0.05 else ('HIGH' if am > pm else 'low')
            print(f"    bucket {i+1}: pred {pm*100:.0f}% -> actual {am*100:.0f}%  {sym}")

    # Retrain winner on all data
    best_model = clone(_CANDIDATES[best_name])
    best_model.fit(X, y)

    if best_name == 'GBR':
        imp = sorted(zip(feature_cols, best_model.feature_importances_), key=lambda x: -x[1])
        print(f"  Top features: {', '.join(f'{n}({v:.3f})' for n, v in imp[:4])}")

    return best_model, best_mae, best_name


# ── Current-season stats CSV ────────────────────────────────────────────────────────

def build_current_stats_csv():
    """Fetch 2025-26 skater aggregates → nhl_skater_current_stats.csv"""
    print(f"\n-- Building nhl_skater_current_stats.csv (season {CURRENT_SEASON_ID}) --")
    skaters = fetch_skater_summary(CURRENT_SEASON_ID, max_results=900)
    rows = []
    for sk in skaters:
        gp = sk.get('gamesPlayed', 1) or 1
        pos = sk.get('positionCode', 'C').upper()
        rows.append({
            'player_id':    sk.get('playerId'),
            'name':         sk.get('skaterFullName', ''),
            'team':         sk.get('teamAbbrevs', ''),
            'position':     pos,
            'gamesPlayed':  gp,
            'goals_pg':     (sk.get('goals', 0) or 0) / gp,
            'assists_pg':   (sk.get('assists', 0) or 0) / gp,
            'shots_pg':     (sk.get('shots', 0) or 0) / gp,
            'shooting_pct': sk.get('shootingPct') or 0.0,
            'toi_pg':       sk.get('timeOnIcePerGame') or 900.0,
            'pp_goals_pg':  (sk.get('ppGoals', 0) or 0) / gp,
            'points_pg':    sk.get('pointsPerGame') or 0.0,
        })

    df = pd.DataFrame(rows).sort_values('points_pg', ascending=False)
    df.to_csv('nhl_skater_current_stats.csv', index=False)
    print(f"  Saved {len(df)} skaters -> nhl_skater_current_stats.csv")
    return df


# ── Main ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Building NHL player prop models")

    team_hist = pd.read_csv('nhl_team_stats_historical.csv')

    # ── Step 1: Build training data ──────────────────────────────────────────────
    print("\n[1/3] Fetching player game logs …")
    df = build_training_data(team_hist)
    print(f"\nTotal training rows: {len(df):,}")
    print(f"  Goals dist:    mean={df['target_goals'].mean():.3f}  max={df['target_goals'].max()}")
    print(f"  Assists dist:  mean={df['target_assists'].mean():.3f}  max={df['target_assists'].max()}")
    print(f"  Shots dist:    mean={df['target_shots'].mean():.3f}   max={df['target_shots'].max()}")

    # ── Step 2: Evaluate and select best model per stat ──────────────────────────
    print("\n[2/3] Evaluating models (temporal leave-one-season-out CV) …")
    print("      Comparing: GBR | PoissonRegressor | Ridge\n")

    goals_model,   goals_mae,   goals_winner   = evaluate_and_select(
        df, 'target_goals',   GOALS_FEATURES,   'Goals')

    assists_model, assists_mae, assists_winner = evaluate_and_select(
        df, 'target_assists', ASSISTS_FEATURES, 'Assists')

    shots_model,   shots_mae,   shots_winner   = evaluate_and_select(
        df, 'target_shots',   SHOTS_FEATURES,   'Shots')

    pkg = {
        'goals':   {'model': goals_model,   'features': GOALS_FEATURES,   'mae': goals_mae,   'winner': goals_winner},
        'assists': {'model': assists_model, 'features': ASSISTS_FEATURES, 'mae': assists_mae, 'winner': assists_winner},
        'shots':   {'model': shots_model,   'features': SHOTS_FEATURES,   'mae': shots_mae,   'winner': shots_winner},
    }

    with open('model_nhl_player.pkl', 'wb') as f:
        pickle.dump(pkg, f)
    print("\n  Saved -> model_nhl_player.pkl")

    # ── Step 3: Current-season stats ─────────────────────────────────────────────
    print("\n[3/3] Building current-season stats CSV …")
    build_current_stats_csv()

    print("\n" + "=" * 55)
    print("  FINAL RESULTS (honest temporal CV MAE)")
    print("=" * 55)
    print(f"  Goals:   ±{goals_mae:.3f} goals/game    [{goals_winner}]")
    print(f"  Assists: ±{assists_mae:.3f} assists/game  [{assists_winner}]")
    print(f"  Shots:   ±{shots_mae:.3f} shots/game   [{shots_winner}]")
    print("=" * 55)
