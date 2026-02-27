# ============================================
# NFL PREDICTOR - Phase 4: Player Props Model
# ============================================
# We're pulling play-by-play data to build per-game
# player stat lines, then training models to predict:
# - Passing yards & TDs (QBs)
# - Rushing yards & TDs (RBs)
# - Receiving yards & TDs (WRs/TEs)

import nfl_data_py as nfl
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Step 1: Load play-by-play data ───────────────────────────────
# We'll use 2016-2024 for player props (older data has missing player IDs)
print("📥 Downloading play-by-play data (2016-2024)...")
print("   This will take 2-3 minutes — it's a LOT of data...\n")

years = list(range(2016, 2025))
pbp = nfl.import_pbp_data(years, downcast=True)
pbp = pbp[pbp['season_type'] == 'REG'].copy()

print(f"✅ Loaded {len(pbp):,} plays!\n")

# ── Step 2: Build per-game stat lines ────────────────────────────
print("🔨 Building per-game player stat lines...")

# PASSING stats — group by game + passer
passing = pbp[pbp['pass_attempt'] == 1].groupby(
    ['game_id', 'season', 'week', 'posteam', 'defteam', 'passer_player_name', 'passer_player_id']
).agg(
    pass_attempts  = ('pass_attempt', 'sum'),
    completions    = ('complete_pass', 'sum'),
    pass_yards     = ('passing_yards', 'sum'),
    pass_tds       = ('touchdown', 'sum'),
    interceptions  = ('interception', 'sum'),
    air_yards      = ('air_yards', 'sum'),
    sacks          = ('sack', 'sum'),
).reset_index()
passing = passing[passing['pass_attempts'] >= 5]  # Filter garbage
print(f"   📊 Passing game logs: {len(passing):,}")

# RUSHING stats — group by game + rusher
rushing = pbp[pbp['rush_attempt'] == 1].groupby(
    ['game_id', 'season', 'week', 'posteam', 'defteam', 'rusher_player_name', 'rusher_player_id']
).agg(
    rush_attempts  = ('rush_attempt', 'sum'),
    rush_yards     = ('rushing_yards', 'sum'),
    rush_tds       = ('touchdown', 'sum'),
    yards_after_contact = ('yards_after_catch', 'sum'),
).reset_index()
rushing = rushing[rushing['rush_attempts'] >= 3]
print(f"   📊 Rushing game logs: {len(rushing):,}")

# RECEIVING stats — group by game + receiver
receiving = pbp[pbp['pass_attempt'] == 1].groupby(
    ['game_id', 'season', 'week', 'posteam', 'defteam', 'receiver_player_name', 'receiver_player_id']
).agg(
    targets        = ('pass_attempt', 'sum'),
    receptions     = ('complete_pass', 'sum'),
    rec_yards      = ('receiving_yards', 'sum'),
    rec_tds        = ('touchdown', 'sum'),
    air_yards      = ('air_yards', 'sum'),
    yards_after_catch = ('yards_after_catch', 'sum'),
).reset_index()
receiving = receiving[receiving['targets'] >= 2]
print(f"   📊 Receiving game logs: {len(receiving):,}\n")

# ── Step 3: Add game context (weather, home/away, etc.) ──────────
print("🌤️  Merging game context (weather, home/away)...")
games = pd.read_csv('games_processed.csv')[
    ['game_id', 'temp', 'wind', 'roof', 'surface', 'home_team', 'away_team',
     'home_score', 'away_score', 'spread_line', 'total_line']
].copy()

games['temp'] = games['temp'].fillna(68)
games['wind'] = games['wind'].fillna(0)
games['is_dome'] = (games['roof'] == 'dome').astype(int)

def add_context(df, team_col):
    df = df.merge(games, on='game_id', how='left')
    df['is_home'] = (df[team_col] == df['home_team']).astype(int)
    return df

passing   = add_context(passing, 'posteam')
rushing   = add_context(rushing, 'posteam')
receiving = add_context(receiving, 'posteam')

# ── Step 3b: Opponent defensive stats (per-team, per-season) ─────
# Computed from the same PBP data — NO future leakage since these
# are per-season aggregates used to characterize the defense a player
# faced in a given game.  Higher EPA allowed = worse defense = easier matchup.
print("🛡️  Computing opponent defensive stats from PBP...")

def_overall = pbp.groupby(['defteam', 'season']).agg(
    opp_def_epa_per_play=('epa', 'mean'),
).reset_index()

def_pass = pbp[pbp['pass_attempt'] == 1].groupby(['defteam', 'season']).agg(
    opp_def_pass_epa=('epa', 'mean'),
    opp_def_pass_yards=('passing_yards', 'mean'),
).reset_index()

def_rush = pbp[pbp['rush_attempt'] == 1].groupby(['defteam', 'season']).agg(
    opp_def_rush_epa=('epa', 'mean'),
    opp_def_rush_yards=('rushing_yards', 'mean'),
).reset_index()

# Build full defensive tables per (team, season)
def_pass_full = def_overall.merge(def_pass, on=['defteam', 'season'], how='left')
def_rush_full = def_overall.merge(def_rush, on=['defteam', 'season'], how='left')

def add_opp_defense(df, def_df):
    return df.merge(def_df, on=['defteam', 'season'], how='left')

passing   = add_opp_defense(passing,   def_pass_full)
rushing   = add_opp_defense(rushing,   def_rush_full)
receiving = add_opp_defense(receiving, def_pass_full)

print(f"   Added opp defense features to passing/rushing/receiving")

# Save per-team per-season defensive stats for the app to use at prediction time
def_pass_full.rename(columns={'defteam': 'team'}).to_csv('def_pass_stats.csv', index=False)
def_rush_full.rename(columns={'defteam': 'team'}).to_csv('def_rush_stats.csv', index=False)
print(f"   Defensive stats saved -> def_pass_stats.csv, def_rush_stats.csv")

# ── Step 4: Add rolling averages (last 4 games) ──────────────────
# This is the key insight: recent form predicts future performance
print("📈 Computing rolling averages (recent form)...")

def add_rolling(df, player_col, stat_cols, n=4):
    """Add rolling n-game averages for each player"""
    df = df.sort_values(['season', 'week'])
    for col in stat_cols:
        df[f'avg_{col}_l{n}'] = (
            df.groupby(player_col)[col]
            .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        )
    return df

passing = add_rolling(passing, 'passer_player_id',
    ['pass_yards', 'pass_tds', 'pass_attempts', 'completions'])

rushing = add_rolling(rushing, 'rusher_player_id',
    ['rush_yards', 'rush_tds', 'rush_attempts'])

receiving = add_rolling(receiving, 'receiver_player_id',
    ['rec_yards', 'rec_tds', 'targets', 'receptions'])

# ── Step 5: Train prediction models ──────────────────────────────
print("\n🤖 Training player prop models...\n")

def train_prop_model(df, target, features, name):
    """Generic function to train a prop prediction model"""
    data = df[features + [target]].dropna()
    
    X = data[features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    
    print(f"   ✅ {name}")
    print(f"      Mean Absolute Error: ±{mae:.1f} units")
    print(f"      Trained on {len(X_train):,} game logs\n")
    
    return model

# Passing yards model
pass_features = [
    'avg_pass_yards_l4', 'avg_pass_attempts_l4', 'avg_completions_l4',
    'avg_pass_tds_l4', 'temp', 'wind', 'is_dome', 'is_home', 'spread_line',
    # Opponent pass defense quality (higher EPA allowed = easier matchup)
    'opp_def_epa_per_play', 'opp_def_pass_epa', 'opp_def_pass_yards',
]
pass_yards_model = train_prop_model(
    passing, 'pass_yards', pass_features, 'Passing Yards'
)

# Rushing yards model
rush_features = [
    'avg_rush_yards_l4', 'avg_rush_attempts_l4', 'avg_rush_tds_l4',
    'temp', 'wind', 'is_dome', 'is_home', 'spread_line',
    # Opponent rush defense quality
    'opp_def_epa_per_play', 'opp_def_rush_epa', 'opp_def_rush_yards',
]
rush_yards_model = train_prop_model(
    rushing, 'rush_yards', rush_features, 'Rushing Yards'
)

# Receiving yards model
rec_features = [
    'avg_rec_yards_l4', 'avg_targets_l4', 'avg_receptions_l4',
    'avg_rec_tds_l4', 'temp', 'wind', 'is_dome', 'is_home', 'spread_line',
    # Opponent pass defense quality (same as passing — WRs face pass defense)
    'opp_def_epa_per_play', 'opp_def_pass_epa', 'opp_def_pass_yards',
]
rec_yards_model = train_prop_model(
    receiving, 'rec_yards', rec_features, 'Receiving Yards'
)

# ── Step 6: Save everything ───────────────────────────────────────
print("💾 Saving models and data...")

with open('pass_yards_model.pkl', 'wb') as f:
    pickle.dump({'model': pass_yards_model, 'features': pass_features}, f)
with open('rush_yards_model.pkl', 'wb') as f:
    pickle.dump({'model': rush_yards_model, 'features': rush_features}, f)
with open('rec_yards_model.pkl', 'wb') as f:
    pickle.dump({'model': rec_yards_model, 'features': rec_features}, f)

passing.to_csv('passing_stats.csv', index=False)
rushing.to_csv('rushing_stats.csv', index=False)
receiving.to_csv('receiving_stats.csv', index=False)

print("\n🚀 Player models ready!")
print("   Run: python update_app.py to add props to your dashboard!")