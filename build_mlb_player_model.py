"""
build_mlb_player_model.py
=========================
Trains 4 MLB player prop models:
  1. Pitcher strikeouts (K) per start
  2. Pitcher earned runs (ER) per start
  3. Batter hits per game
  4. Batter total bases (TB) per game

Data:
  - pybaseball pitching_stats / batting_stats → season aggregate features
  - MLB Stats API /people/{id}/stats?stats=gameLog → per-game training targets
  - mlb_team_stats_historical.csv → opponent context features
  - mlb_pitcher_ratings.csv → pitcher era_minus, fip_minus

Output:
  - model_mlb_player.pkl          : 4 GBR models + feature lists + MAE
  - mlb_pitcher_season_stats.csv  : pitcher k_per_9 + ip_per_gs for runtime
  - mlb_batter_stats_current.csv  : 2025 qualified hitters for runtime
  - .mlb_prop_log_cache.json      : API response cache (speeds up re-runs)
"""

import json
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import pybaseball as pb

warnings.filterwarnings('ignore')
pb.cache.enable()

# ── Constants ──────────────────────────────────────────────────────────────────
_BASE = "https://statsapi.mlb.com/api/v1"
_SESS = requests.Session()
_SESS.headers['User-Agent'] = 'EdgeIQ/1.0'

TRAIN_SEASONS  = list(range(2020, 2026))   # 2020–2025
CURRENT_SEASON = 2025
TOP_SP         = 80      # starters per season (by GS)
TOP_BATTERS    = 150     # batters per season (by PA)

LEAGUE_K_PCT  = 0.225   # MLB batting K% average
LEAGUE_WOBA   = 0.320
LEAGUE_ERA_M  = 100.0
LEAGUE_FIP_M  = 100.0

# Full team name (from MLB Stats API) → FanGraphs abbreviation (mlb_team_stats cols)
_NAME_TO_FG = {
    'Arizona Diamondbacks': 'ARI',   'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL',      'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC',           'Chicago White Sox': 'CHW',
    'Cincinnati Reds': 'CIN',        'Cleveland Guardians': 'CLE',
    'Cleveland Indians': 'CLE',
    'Colorado Rockies': 'COL',       'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU',         'Kansas City Royals': 'KCR',
    'Los Angeles Angels': 'LAA',     'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA',          'Florida Marlins': 'MIA',
    'Milwaukee Brewers': 'MIL',      'Minnesota Twins': 'MIN',
    'New York Mets': 'NYM',          'New York Yankees': 'NYY',
    'Oakland Athletics': 'OAK',      'Athletics': 'OAK',
    'Philadelphia Phillies': 'PHI',  'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SDP',       'Seattle Mariners': 'SEA',
    'San Francisco Giants': 'SFG',   'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TBR',         'Tampa Bay Devil Rays': 'TBR',
    'Texas Rangers': 'TEX',          'Toronto Blue Jays': 'TOR',
    'Washington Nationals': 'WSN',   'Montreal Expos': 'WSN',
}

# ── API cache (local JSON) ─────────────────────────────────────────────────────
_CACHE_FILE = '.mlb_prop_log_cache.json'

def _load_cache():
    if os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE) as f:
            return json.load(f)
    return {}

def _save_cache(c):
    with open(_CACHE_FILE, 'w') as f:
        json.dump(c, f)

_cache = _load_cache()

# ── MLB Stats API helpers ──────────────────────────────────────────────────────
_id_cache = {}

def _find_player_id(name: str):
    if name in _id_cache:
        return _id_cache[name]
    try:
        r = _SESS.get(f"{_BASE}/people/search",
                      params={'names': name, 'sportId': 1}, timeout=12)
        r.raise_for_status()
        people = r.json().get('people', [])
        if people:
            pid = people[0]['id']
            _id_cache[name] = pid
            return pid
    except Exception:
        pass
    time.sleep(0.05)
    return None

def _get_pitcher_logs(player_id: int, season: int) -> list:
    key = f"p_{player_id}_{season}"
    if key in _cache:
        return _cache[key]
    try:
        r = _SESS.get(f"{_BASE}/people/{player_id}/stats",
                      params={'stats': 'gameLog', 'group': 'pitching',
                               'season': season, 'gameType': 'R'}, timeout=15)
        r.raise_for_status()
        stats_list = r.json().get('stats', [])
        if not stats_list:
            return []
        splits = stats_list[0].get('splits', [])
    except Exception:
        return []
    time.sleep(0.05)

    rows = []
    for s in splits:
        stat = s.get('stat', {})
        if stat.get('gamesStarted', 0) < 1:
            continue
        ip_str = str(stat.get('inningsPitched', '0'))
        try:
            parts = ip_str.split('.')
            ip = int(parts[0]) + (int(parts[1]) / 3 if len(parts) > 1 and parts[1] else 0)
        except Exception:
            ip = 0.0
        opp_name = s.get('opponent', {}).get('name', '')
        rows.append({
            'opp_name': opp_name,
            'opp_abbr': _NAME_TO_FG.get(opp_name, ''),
            'is_home': 1 if s.get('isHome', False) else 0,
            'strikeouts': int(stat.get('strikeOuts', 0)),
            'earned_runs': int(stat.get('earnedRuns', 0)),
            'ip': round(ip, 2),
        })
    _cache[key] = rows
    _save_cache(_cache)
    return rows

def _get_batter_logs(player_id: int, season: int) -> list:
    key = f"b_{player_id}_{season}"
    if key in _cache:
        return _cache[key]
    try:
        r = _SESS.get(f"{_BASE}/people/{player_id}/stats",
                      params={'stats': 'gameLog', 'group': 'hitting',
                               'season': season, 'gameType': 'R'}, timeout=15)
        r.raise_for_status()
        stats_list = r.json().get('stats', [])
        if not stats_list:
            return []
        splits = stats_list[0].get('splits', [])
    except Exception:
        return []
    time.sleep(0.05)

    rows = []
    for s in splits:
        stat = s.get('stat', {})
        ab = int(stat.get('atBats', 0))
        if ab < 1:
            continue
        opp_name = s.get('opponent', {}).get('name', '')
        rows.append({
            'opp_name': opp_name,
            'opp_abbr': _NAME_TO_FG.get(opp_name, ''),
            'is_home': 1 if s.get('isHome', False) else 0,
            'hits': int(stat.get('hits', 0)),
            'total_bases': int(stat.get('totalBases', 0)),
            'at_bats': ab,
        })
    _cache[key] = rows
    _save_cache(_cache)
    return rows

# ── Step 1: Load team stats for opponent features ──────────────────────────────
print("📊 Loading team stats...")
team_hist = pd.read_csv('mlb_team_stats_historical.csv')

def _opp_stat(opp_abbr, season, col, default):
    row = team_hist[(team_hist['team'] == opp_abbr) & (team_hist['season'] == season)]
    if len(row) == 0:
        return default
    v = row[col].iloc[0]
    return float(v) if not pd.isna(v) else default

# ── Step 2: Pitcher season stats from pybaseball ───────────────────────────────
print("\n⚾  Fetching pitcher season stats (pybaseball)...")
p_season_rows = []
for yr in TRAIN_SEASONS:
    print(f"   {yr}... ", end='', flush=True)
    try:
        df = pb.pitching_stats(yr, yr, qual=30)
        df['season'] = yr
        df['k_per_9']   = df['SO'] / df['IP'].clip(lower=1) * 9
        df['ip_per_gs']  = df['IP'] / df['GS'].clip(lower=1)
        starters = df[df['GS'] >= 8].nlargest(TOP_SP, 'GS')
        p_season_rows.append(starters[['Name', 'Team', 'season', 'k_per_9', 'ip_per_gs', 'ERA', 'FIP', 'SO', 'IP', 'GS']])
        print(f"{len(starters)} SP")
    except Exception as e:
        print(f"ERROR: {e}")

pitcher_season_df = pd.concat(p_season_rows, ignore_index=True)
print(f"   Total: {len(pitcher_season_df)} pitcher-seasons\n")

# Load era_minus, fip_minus from pitcher ratings
pitcher_ratings = pd.read_csv('mlb_pitcher_ratings.csv')

# Save enriched pitcher season stats (used by mlb_app.py at prediction time)
print("💾 Saving mlb_pitcher_season_stats.csv...")
pitcher_season_df.to_csv('mlb_pitcher_season_stats.csv', index=False)

# ── Step 3: Fetch pitcher game logs & build training set ───────────────────────
print("📡 Fetching pitcher game logs from MLB Stats API...")
print("   (Using local cache — only new entries hit the API)\n")

pitcher_rows = []
n_sp = len(pitcher_season_df)
n_done = 0

for _, row in pitcher_season_df.iterrows():
    name   = row['Name']
    season = int(row['season'])
    k9     = float(row['k_per_9'])   if not pd.isna(row['k_per_9'])   else 7.5
    ipgs   = float(row['ip_per_gs']) if not pd.isna(row['ip_per_gs']) else 5.5

    pr = pitcher_ratings[
        (pitcher_ratings['name'].str.lower() == name.lower()) &
        (pitcher_ratings['season'] == season)
    ]
    era_m = float(pr['era_minus'].iloc[0]) if len(pr) > 0 else LEAGUE_ERA_M
    fip_m = float(pr['fip_minus'].iloc[0]) if len(pr) > 0 else LEAGUE_FIP_M

    pid = _find_player_id(name)
    if pid is None:
        n_done += 1
        continue

    logs = _get_pitcher_logs(pid, season)

    for log in logs:
        opp = log['opp_abbr']
        opp_k  = _opp_stat(opp, season, 'k_pct',   LEAGUE_K_PCT)
        opp_wo = _opp_stat(opp, season, 'woba',     LEAGUE_WOBA)
        opp_wc = _opp_stat(opp, season, 'wrc_plus', 100.0)
        pitcher_rows.append({
            'k_per_9':     k9,
            'ip_per_gs':   ipgs,
            'era_minus':   era_m,
            'fip_minus':   fip_m,
            'is_home':     log['is_home'],
            'opp_k_pct':   opp_k,
            'opp_woba':    opp_wo,
            'opp_wrc_plus': opp_wc,
            'actual_k':    log['strikeouts'],
            'actual_er':   log['earned_runs'],
            'actual_ip':   log['ip'],
        })

    n_done += 1
    if n_done % 100 == 0:
        print(f"   {n_done}/{n_sp} pitchers → {len(pitcher_rows):,} starts")

pitcher_df = pd.DataFrame(pitcher_rows)
print(f"\n✅ Pitcher training set: {len(pitcher_df):,} starts")
print(f"   Avg K/start: {pitcher_df['actual_k'].mean():.2f} | Avg ER/start: {pitcher_df['actual_er'].mean():.2f}")

# ── Step 4: Train pitcher models ───────────────────────────────────────────────
print("\n🤖 Training pitcher prop models...")

PITCHER_K_FEATURES  = ['k_per_9', 'ip_per_gs', 'fip_minus', 'opp_k_pct', 'is_home']
PITCHER_ER_FEATURES = ['era_minus', 'fip_minus', 'ip_per_gs', 'opp_woba', 'opp_wrc_plus', 'is_home']

def _train_gbr(df, features, target, name):
    data = df[features + [target]].dropna()
    X, y = data[features], data[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    gbr.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, gbr.predict(X_te))
    print(f"   ✅ {name}  MAE ±{mae:.2f}  ({len(X_tr):,} train / {len(X_te):,} test)")
    return gbr, mae

pitcher_k_model,  mae_pk = _train_gbr(pitcher_df, PITCHER_K_FEATURES,  'actual_k',  'Pitcher K/start')
pitcher_er_model, mae_er = _train_gbr(pitcher_df, PITCHER_ER_FEATURES, 'actual_er', 'Pitcher ER/start')

# ── Step 5: Batter season stats from pybaseball ────────────────────────────────
print("\n🏏 Fetching batter season stats (pybaseball)...")
b_season_rows = []
for yr in TRAIN_SEASONS:
    print(f"   {yr}... ", end='', flush=True)
    try:
        df = pb.batting_stats(yr, yr, qual=100)
        df['season'] = yr
        top = df.nlargest(TOP_BATTERS, 'PA')
        b_season_rows.append(top[['Name', 'Team', 'season', 'PA', 'AB', 'G',
                                   'AVG', 'OBP', 'SLG', 'ISO', 'wOBA', 'wRC+',
                                   'BB%', 'K%', 'H', '2B', '3B', 'HR', 'IDfg']])
        print(f"{len(top)} batters")
    except Exception as e:
        print(f"ERROR: {e}")

batter_season_df = pd.concat(b_season_rows, ignore_index=True)
print(f"   Total: {len(batter_season_df)} batter-seasons\n")

# Save current-season (2025) batter stats for app use
print("💾 Saving mlb_batter_stats_current.csv (2025)...")
try:
    curr = pb.batting_stats(CURRENT_SEASON, CURRENT_SEASON, qual=100)
    curr['ab_per_game'] = curr['AB'] / curr['G'].clip(lower=1)
    curr[['Name', 'Team', 'PA', 'G', 'AB', 'ab_per_game', 'AVG', 'OBP',
          'SLG', 'ISO', 'wOBA', 'wRC+', 'BB%', 'K%', 'IDfg']].to_csv(
        'mlb_batter_stats_current.csv', index=False)
    print(f"   Saved {len(curr)} batters (2025)")
except Exception as e:
    print(f"   ERROR saving current batter stats: {e}")
    # Fallback to 2024
    curr = pb.batting_stats(2024, 2024, qual=100)
    curr['ab_per_game'] = curr['AB'] / curr['G'].clip(lower=1)
    curr[['Name', 'Team', 'PA', 'G', 'AB', 'ab_per_game', 'AVG', 'OBP',
          'SLG', 'ISO', 'wOBA', 'wRC+', 'BB%', 'K%', 'IDfg']].to_csv(
        'mlb_batter_stats_current.csv', index=False)
    print(f"   Saved {len(curr)} batters (2024 fallback)")

# ── Step 6: Fetch batter game logs & build training set ───────────────────────
print("\n📡 Fetching batter game logs from MLB Stats API...")

batter_rows = []
n_bat = len(batter_season_df)
n_done = 0

for _, row in batter_season_df.iterrows():
    name   = row['Name']
    season = int(row['season'])
    avg    = float(row['AVG'])  if not pd.isna(row['AVG'])  else 0.250
    obp    = float(row['OBP'])  if not pd.isna(row['OBP'])  else 0.320
    slg    = float(row['SLG'])  if not pd.isna(row['SLG'])  else 0.400
    iso    = float(row['ISO'])  if not pd.isna(row['ISO'])  else 0.150
    woba   = float(row['wOBA']) if not pd.isna(row['wOBA']) else LEAGUE_WOBA
    wrc    = float(row['wRC+']) if not pd.isna(row['wRC+']) else 100.0
    ab_g   = float(row['AB']) / max(float(row['G']), 1)

    pid = _find_player_id(name)
    if pid is None:
        n_done += 1
        continue

    logs = _get_batter_logs(pid, season)

    for log in logs:
        opp = log['opp_abbr']
        opp_era_m = _opp_stat(opp, season, 'era_minus', LEAGUE_ERA_M)
        opp_fip_m = _opp_stat(opp, season, 'fip_minus', LEAGUE_FIP_M)
        opp_whip  = _opp_stat(opp, season, 'whip',       1.30)
        batter_rows.append({
            'batter_avg':      avg,
            'batter_obp':      obp,
            'batter_slg':      slg,
            'batter_iso':      iso,
            'batter_woba':     woba,
            'batter_wrc_plus': wrc,
            'ab_per_game':     ab_g,
            'is_home':         log['is_home'],
            'opp_era_minus':   opp_era_m,
            'opp_fip_minus':   opp_fip_m,
            'opp_whip':        opp_whip,
            'actual_hits': log['hits'],
            'actual_tb':   log['total_bases'],
        })

    n_done += 1
    if n_done % 100 == 0:
        print(f"   {n_done}/{n_bat} batters → {len(batter_rows):,} game logs")

batter_df = pd.DataFrame(batter_rows)
print(f"\n✅ Batter training set: {len(batter_df):,} game-logs")
print(f"   Avg hits/game: {batter_df['actual_hits'].mean():.3f} | Avg TB/game: {batter_df['actual_tb'].mean():.3f}")

# ── Step 7: Train batter models ────────────────────────────────────────────────
print("\n🤖 Training batter prop models...")

BATTER_HITS_FEATURES = ['batter_avg', 'batter_obp', 'ab_per_game', 'is_home',
                         'opp_era_minus', 'opp_fip_minus', 'opp_whip']
BATTER_TB_FEATURES   = ['batter_iso', 'batter_slg', 'batter_woba', 'ab_per_game',
                         'is_home', 'opp_era_minus', 'opp_fip_minus']

batter_hits_model, mae_bh = _train_gbr(batter_df, BATTER_HITS_FEATURES, 'actual_hits', 'Batter hits/game')
batter_tb_model,   mae_bt = _train_gbr(batter_df, BATTER_TB_FEATURES,   'actual_tb',   'Batter TB/game')

# ── Step 8: Save everything ────────────────────────────────────────────────────
print("\n💾 Saving model_mlb_player.pkl...")

pkg = {
    'pitcher_k':  {'model': pitcher_k_model,  'features': PITCHER_K_FEATURES,  'mae': mae_pk},
    'pitcher_er': {'model': pitcher_er_model,  'features': PITCHER_ER_FEATURES, 'mae': mae_er},
    'batter_hits':{'model': batter_hits_model, 'features': BATTER_HITS_FEATURES,'mae': mae_bh},
    'batter_tb':  {'model': batter_tb_model,   'features': BATTER_TB_FEATURES,  'mae': mae_bt},
}
with open('model_mlb_player.pkl', 'wb') as f:
    pickle.dump(pkg, f)

print("\n" + "="*55)
print("  MLB PLAYER PROP MODEL SUMMARY")
print("="*55)
print(f"  Pitcher K/start       MAE ±{mae_pk:.2f} K's")
print(f"  Pitcher ER/start      MAE ±{mae_er:.2f} earned runs")
print(f"  Batter hits/game      MAE ±{mae_bh:.3f} hits")
print(f"  Batter TB/game        MAE ±{mae_bt:.3f} total bases")
print("="*55)
print("\nFiles written:")
print("  model_mlb_player.pkl")
print("  mlb_pitcher_season_stats.csv")
print("  mlb_batter_stats_current.csv")
print("\n✅ Done!")
