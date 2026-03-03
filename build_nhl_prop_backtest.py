"""
build_nhl_prop_backtest.py
==========================
Reads .nhl_prop_log_cache.json, reconstructs cumulative season-average features
(expanding mean, min 10 previous games warmup) for each player-game, runs
model_nhl_player.pkl for goals / assists / shots predictions.

Output: nhl_prop_backtest.csv
  player_id, name, team, position, season, game_date, prop_type,
  predicted_prob, predicted_value, actual_value, hit, is_forward, is_home

Hit thresholds (fixed):
  goals   : actual_goals  >= 1
  assists : actual_assists >= 1
  shots   : actual_shots   >= 4  (market line O3.5)
"""

import json
import math
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ── Constants ─────────────────────────────────────────────────────────────────

_STATS_BASE = "https://api.nhle.com/stats/rest/en"
_SESS       = requests.Session()
_SESS.headers['User-Agent'] = 'EdgeIQ/1.0 (educational)'

TRAIN_SEASON_IDS = ['20202021', '20212022', '20222023', '20232024', '20242025']
TOP_SKATERS      = 200
MIN_GP           = 20
WARMUP_GAMES     = 10   # min previous games before including a row

_CACHE_FILE  = '.nhl_prop_log_cache.json'
_MODEL_FILE  = 'model_nhl_player.pkl'
_HIST_FILE   = 'nhl_team_stats_historical.csv'
_OUTPUT_FILE = 'nhl_prop_backtest.csv'

_FWD_POSITIONS = {'C', 'L', 'R', 'LW', 'RW', 'F'}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(url, retries=3):
    for attempt in range(retries):
        try:
            r = _SESS.get(url, timeout=20)
            r.raise_for_status()
            time.sleep(0.3)
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [warn] GET failed: {url}  {e}")
    return None


def _toi_to_seconds(toi_str):
    try:
        m, s = str(toi_str).split(':')
        return int(m) * 60 + int(s)
    except Exception:
        return 0.0


def _season_year(season_id):
    return int(str(season_id)[:4])


def fetch_skater_summary(season_id, max_results=300):
    PAGE_SIZE = 100
    all_data  = []
    start     = 0
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


# ── Core backtest builder ─────────────────────────────────────────────────────

def build_backtest(model_pkg, team_hist, cache):
    rows = []

    goals_pkg   = model_pkg.get('goals',   {})
    assists_pkg = model_pkg.get('assists', {})
    shots_pkg   = model_pkg.get('shots',   {})

    goals_model   = goals_pkg.get('model')
    assists_model = assists_pkg.get('model')
    shots_model   = shots_pkg.get('model')
    shots_mae     = shots_pkg.get('mae', 1.265)

    goals_feats   = goals_pkg.get('features',   [])
    assists_feats = assists_pkg.get('features',  [])
    shots_feats   = shots_pkg.get('features',    [])

    for season_id in TRAIN_SEASON_IDS:
        yr = _season_year(season_id)
        print(f"\n-- Season {season_id} (year={yr}) --")

        # Opponent context for this season
        opp_ctx = {}
        s_hist = team_hist[team_hist['season'] == yr]
        for _, row in s_hist.iterrows():
            opp_ctx[row['team']] = {
                'opp_goals_against_pg': row.get('goals_against_pg', 2.8),
                'opp_shots_against_pg': row.get('shots_against_pg', 29.0),
                'opp_pk_pct':           row.get('pk_pct', 0.80),
            }

        # Fetch player metadata for this season (name, position)
        print(f"  Fetching skater summaries …")
        skaters   = fetch_skater_summary(season_id, max_results=TOP_SKATERS + 50)
        qualified = [s for s in skaters if s.get('gamesPlayed', 0) >= MIN_GP]
        qualified = qualified[:TOP_SKATERS]

        player_meta = {}
        for sk in qualified:
            pid = sk.get('playerId')
            player_meta[pid] = {
                'name':     sk.get('skaterFullName', str(pid)),
                'position': str(sk.get('positionCode', 'C')).upper(),
            }

        n_rows = 0
        for pid, meta in player_meta.items():
            key = f"{pid}_{season_id}"
            log = cache.get(key, [])
            if not log:
                continue

            # Sort games by date ascending
            log_sorted = sorted(log, key=lambda g: g.get('gameDate', ''))
            n_games    = len(log_sorted)

            # Running totals for expanding mean
            cum_goals     = 0.0
            cum_assists   = 0.0
            cum_shots     = 0.0
            cum_toi_s     = 0.0
            cum_pp_goals  = 0.0
            cum_gp        = 0

            pos        = meta['position']
            name       = meta['name']
            is_forward = 1 if pos in _FWD_POSITIONS else 0

            for g in log_sorted:
                if cum_gp < WARMUP_GAMES:
                    # Warmup: accumulate but don't emit rows
                    cum_goals    += int(g.get('goals',          0) or 0)
                    cum_assists  += int(g.get('assists',         0) or 0)
                    cum_shots    += int(g.get('shots',           0) or 0)
                    cum_toi_s    += _toi_to_seconds(g.get('toi', '0:00'))
                    cum_pp_goals += int(g.get('powerPlayGoals',  0) or 0)
                    cum_gp       += 1
                    continue

                # Features from previous games
                goals_pg    = cum_goals    / cum_gp
                assists_pg  = cum_assists  / cum_gp
                shots_pg    = cum_shots    / cum_gp
                toi_pg      = cum_toi_s    / cum_gp
                pp_goals_pg = cum_pp_goals / cum_gp
                shoot_pct   = goals_pg / max(shots_pg, 0.01)

                opp     = g.get('opponentAbbrev', '')
                ctx     = opp_ctx.get(opp, {})
                is_home = 1 if g.get('homeRoadFlag', 'R') == 'H' else 0
                team    = g.get('teamAbbrev', '')

                base = {
                    'goals_pg':            goals_pg,
                    'assists_pg':          assists_pg,
                    'shots_pg':            shots_pg,
                    'shooting_pct':        shoot_pct,
                    'toi_pg':              toi_pg,
                    'pp_goals_pg':         pp_goals_pg,
                    'is_forward':          is_forward,
                    'is_home':             is_home,
                    'opp_goals_against_pg': ctx.get('opp_goals_against_pg', 2.8),
                    'opp_shots_against_pg': ctx.get('opp_shots_against_pg', 29.0),
                    'opp_pk_pct':          ctx.get('opp_pk_pct', 0.80),
                }

                def _predict(model, feats):
                    if model is None or not feats:
                        return None
                    X = [[base.get(f, 0.0) for f in feats]]
                    return float(model.predict(X)[0])

                g_pred = _predict(goals_model,   goals_feats)
                a_pred = _predict(assists_model, assists_feats)
                s_pred = _predict(shots_model,   shots_feats)

                if g_pred is None:
                    # Model not available, skip
                    cum_goals    += int(g.get('goals',         0) or 0)
                    cum_assists  += int(g.get('assists',        0) or 0)
                    cum_shots    += int(g.get('shots',          0) or 0)
                    cum_toi_s    += _toi_to_seconds(g.get('toi', '0:00'))
                    cum_pp_goals += int(g.get('powerPlayGoals', 0) or 0)
                    cum_gp       += 1
                    continue

                g_pred = max(0.0, min(g_pred, 2.0))
                a_pred = max(0.0, min(a_pred, 2.5))
                s_pred = max(0.5, min(s_pred, 10.0))

                g_prob = 1.0 - math.exp(-g_pred)
                a_prob = 1.0 - math.exp(-a_pred)
                s_prob = float(norm.sf(3.5, loc=s_pred, scale=max(shots_mae, 0.1)))

                actual_goals   = int(g.get('goals',   0) or 0)
                actual_assists = int(g.get('assists',  0) or 0)
                actual_shots   = int(g.get('shots',    0) or 0)
                game_date      = g.get('gameDate', '')

                common = {
                    'player_id':  pid,
                    'name':       name,
                    'team':       team,
                    'position':   pos,
                    'season':     yr,
                    'game_date':  game_date,
                    'is_forward': is_forward,
                    'is_home':    is_home,
                }

                rows.append({**common,
                    'prop_type':       'goals',
                    'predicted_prob':  round(g_prob, 4),
                    'predicted_value': round(g_pred, 4),
                    'actual_value':    actual_goals,
                    'hit':             int(actual_goals >= 1),
                })
                rows.append({**common,
                    'prop_type':       'assists',
                    'predicted_prob':  round(a_prob, 4),
                    'predicted_value': round(a_pred, 4),
                    'actual_value':    actual_assists,
                    'hit':             int(actual_assists >= 1),
                })
                rows.append({**common,
                    'prop_type':       'shots',
                    'predicted_prob':  round(s_prob, 4),
                    'predicted_value': round(s_pred, 4),
                    'actual_value':    actual_shots,
                    'hit':             int(actual_shots >= 4),
                })
                n_rows += 3

                # Accumulate current game into expanding window
                cum_goals    += int(g.get('goals',         0) or 0)
                cum_assists  += int(g.get('assists',        0) or 0)
                cum_shots    += int(g.get('shots',          0) or 0)
                cum_toi_s    += _toi_to_seconds(g.get('toi', '0:00'))
                cum_pp_goals += int(g.get('powerPlayGoals', 0) or 0)
                cum_gp       += 1

        print(f"  {n_rows:,} rows emitted")

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Load model
    print("Loading model_nhl_player.pkl …")
    with open(_MODEL_FILE, 'rb') as f:
        model_pkg = pickle.load(f)
    for k, v in model_pkg.items():
        print(f"  {k}: winner={v.get('winner','?')}  mae={v.get('mae',0):.3f}")

    # Load cache
    print(f"\nLoading game-log cache ({_CACHE_FILE}) …")
    with open(_CACHE_FILE) as f:
        cache = json.load(f)
    print(f"  {len(cache):,} player-season keys  |  {sum(len(v) for v in cache.values()):,} game records")

    # Load team historical stats
    print(f"\nLoading {_HIST_FILE} …")
    team_hist = pd.read_csv(_HIST_FILE)

    # Build backtest
    print("\nBuilding backtest rows …")
    df = build_backtest(model_pkg, team_hist, cache)

    print(f"\nTotal rows: {len(df):,}")
    print(df.groupby('prop_type')[['hit', 'predicted_prob']].mean().round(3))

    # Save
    df.to_csv(_OUTPUT_FILE, index=False)
    print(f"\nSaved -> {_OUTPUT_FILE}")

    # Quick accuracy summary
    print("\n── Hit Rate Summary ──────────────────────────────────────────")
    for season in sorted(df['season'].unique()):
        sub = df[df['season'] == season]
        by_type = sub.groupby('prop_type')['hit'].mean()
        print(f"  {season}-{season+1}:  goals={by_type.get('goals',0)*100:.1f}%  "
              f"assists={by_type.get('assists',0)*100:.1f}%  "
              f"shots={by_type.get('shots',0)*100:.1f}%  "
              f"(n={len(sub)//3:,} games)")
    print("─────────────────────────────────────────────────────────────")
