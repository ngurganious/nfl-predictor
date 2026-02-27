# ============================================
# NFL PREDICTOR - Phase 3: Build the Model
# ============================================
# We're building TWO things here:
# 1. An ELO rating system (like chess rankings) for every NFL team
# 2. A machine learning model that predicts game outcomes
#
# GAME THEORY NOTE: ELO captures the idea that beating a strong
# opponent tells you more than beating a weak one. It's a dynamic
# rating that updates after every game — teams are always
# responding to each other's strength.

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle  # For saving our trained model to disk

# ── Load data ────────────────────────────────────────────────────
games = pd.read_csv("games_raw.csv")
games = games[games['game_type'] == 'REG'].copy()
games = games.dropna(subset=['home_score', 'away_score'])
games = games.sort_values('gameday').reset_index(drop=True)

print(f"📊 Building model on {len(games)} games...\n")

# ── PART 1: ELO RATING SYSTEM ────────────────────────────────────
# Every team starts at 1500 (the baseline)
# Win = gain points, Loss = lose points
# The amount gained/lost depends on how surprising the result was

elo_ratings = {}  # Dictionary to store current ELO for each team
elo_history = []  # We'll store each game's ELO snapshot

K = 20  # "K-factor" — how much each game shifts ratings
        # Higher K = more reactive, Lower K = more stable

def get_elo(team):
    """Get a team's current ELO, defaulting to 1500 if new"""
    return elo_ratings.get(team, 1500)

def expected_win_prob(elo_a, elo_b):
    """
    The core ELO formula — predicts win probability based on rating gap.
    This comes from chess and is mathematically elegant:
    A 200 point gap = ~76% win probability for the stronger team
    """
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(winner, loser):
    """Update both teams' ELO after a game"""
    elo_w = get_elo(winner)
    elo_l = get_elo(loser)
    
    expected = expected_win_prob(elo_w, elo_l)
    
    # Winner gains points, loser loses the same amount
    elo_ratings[winner] = elo_w + K * (1 - expected)
    elo_ratings[loser]  = elo_l + K * (0 - (1 - expected))

# Process every game chronologically
for _, game in games.iterrows():
    home = game['home_team']
    away = game['away_team']
    
    # Store ELO BEFORE the game (this is what we'd know at prediction time)
    elo_history.append({
        'game_id': game['game_id'],
        'home_elo': get_elo(home),
        'away_elo': get_elo(away),
        'elo_diff': get_elo(home) - get_elo(away)
    })
    
    # Update ELO AFTER the game
    if game['home_score'] > game['away_score']:
        update_elo(home, away)
    elif game['away_score'] > game['home_score']:
        update_elo(away, home)
    # Ties: no update (rare in NFL)

elo_df = pd.DataFrame(elo_history).set_index('game_id')

# Print current ELO standings
print("🏆 CURRENT NFL ELO RANKINGS (end of 2024):")
current_elo = pd.Series(elo_ratings).sort_values(ascending=False)
for i, (team, elo) in enumerate(current_elo.items(), 1):
    print(f"   {i:2}. {team:4} — {elo:.0f}")

# ── PART 2: BUILD FEATURES FOR ML MODEL ─────────────────────────
# "Features" = the inputs our model uses to make predictions
# Think of these as the variables in a spreadsheet row for each game

games = games.join(elo_df, on='game_id')

# Fill missing weather data with medians (dome games have no temp/wind)
games['temp'] = games['temp'].fillna(games['temp'].median())
games['wind'] = games['wind'].fillna(0)

# Convert roof type to a number (1 = dome, 0 = outdoor)
games['is_dome'] = (games['roof'] == 'dome').astype(int)

# Convert surface to a number (1 = grass, 0 = turf)
games['is_grass'] = (games['surface'].str.contains('grass', na=False)).astype(int)

# The target variable: did the home team win? (1 = yes, 0 = no)
games['home_win'] = (games['home_score'] > games['away_score']).astype(int)

# Select our features
feature_cols = [
    'elo_diff',       # ELO rating difference (most important!)
    'spread_line',    # Vegas spread (very informative)
    'home_rest',      # Days of rest for home team
    'away_rest',      # Days of rest for away team
    'temp',           # Temperature
    'wind',           # Wind speed
    'is_dome',        # Dome game?
    'is_grass',       # Grass surface?
    'div_game',       # Divisional game?
]

# Only use rows where we have all features
model_data = games[feature_cols + ['home_win']].dropna()
print(f"\n📊 Training on {len(model_data)} games with complete data\n")

X = model_data[feature_cols]
y = model_data['home_win']

# Split: 80% training data, 20% testing data
# We test on data the model has never seen — honest evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# ── PART 3: TRAIN THE MODEL ──────────────────────────────────────
# Gradient Boosting = an ensemble of decision trees that learn
# from each other's mistakes. One of the best algorithms for
# structured data like this.

print("🤖 Training model...")
model = GradientBoostingClassifier(
    n_estimators=200,    # 200 trees
    learning_rate=0.05,  # How fast it learns
    max_depth=4,         # How complex each tree can be
    random_state=42
)
model.fit(X_train, y_train)

# ── PART 4: EVALUATE ─────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy: {accuracy*100:.1f}%")
print(f"   (Baseline = 56% just by always picking home team)")

# Feature importance — what does the model actually care about?
print(f"\n📈 WHAT MATTERS MOST (feature importance):")
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance_df.iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"   {row['feature']:15} {bar} {row['importance']*100:.1f}%")

# ── PART 5: SAVE EVERYTHING ──────────────────────────────────────
# pickle saves Python objects to disk so we can load them later
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('elo_ratings.pkl', 'wb') as f:
    pickle.dump(elo_ratings, f)

games.to_csv('games_processed.csv', index=False)

print("\n💾 Model, ELO ratings, and processed data saved!")
print("🚀 Ready to build the app!")