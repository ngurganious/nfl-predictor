# ================================================
# NFL PREDICTOR - Backtesting
# Tests our model against real historical outcomes
# ================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("🔬 BACKTESTING NFL PREDICTOR\n")

# ── Load model and data ───────────────────────────
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('elo_ratings.pkl', 'rb') as f:
    elo_ratings = pickle.load(f)

games = pd.read_csv('games_processed.csv')
games = games[games['game_type'] == 'REG'].copy()
games = games.dropna(subset=['home_score','away_score'])
games = games.sort_values('gameday').reset_index(drop=True)

# ── Build features exactly as trained ────────────
games['temp']     = games['temp'].fillna(games['temp'].median())
games['wind']     = games['wind'].fillna(0)
games['is_dome']  = (games['roof'] == 'dome').astype(int)
games['is_grass'] = (games['surface'].str.contains('grass', na=False)).astype(int)
games['home_win'] = (games['home_score'] > games['away_score']).astype(int)

# Exact feature order from training
FEATURES = ['elo_diff','spread_line','home_rest','away_rest',
            'temp','wind','is_dome','is_grass','div_game']

# Keep only rows with all features present
keep_cols = FEATURES + ['home_win','season','week',
                        'home_team','away_team',
                        'home_score','away_score']
data = games[keep_cols].dropna().copy()
data = data.reset_index(drop=True)

print(f"📊 Testing on {len(data)} games across "
      f"{data['season'].nunique()} seasons\n")

# ── Helper: build feature matrix ─────────────────
def get_X(df):
    X = df[FEATURES].copy()
    return X

# ── Test by season ────────────────────────────────
print("📅 ACCURACY BY SEASON:")
print(f"{'Season':<8} {'Games':<7} {'Correct':<9} {'Accuracy':<10} {'vs Baseline':<12} {'Result'}")
print("-" * 60)

season_results = []
for season in sorted(data['season'].unique()):
    sd = data[data['season'] == season].copy()
    X  = get_X(sd)
    y  = sd['home_win']

    preds    = model.predict(X)
    acc      = accuracy_score(y, preds)
    baseline = max(y.mean(), 1 - y.mean())
    improvement = acc - baseline

    season_results.append({
        'season':      season,
        'games':       len(y),
        'accuracy':    acc,
        'baseline':    baseline,
        'improvement': improvement
    })

    marker = "✅" if acc > baseline else "❌"
    print(f"{season:<8} {len(y):<7} {int(acc*len(y)):<9} "
          f"{acc*100:.1f}%{'':<5} {baseline*100:.1f}%{'':<6} "
          f"{marker} {improvement*100:+.1f}%")

# ── Overall summary ───────────────────────────────
rd = pd.DataFrame(season_results)
print(f"\n{'='*60}")
print(f"📊 OVERALL PERFORMANCE SUMMARY:")
print(f"   Total seasons tested: {len(rd)}")
print(f"   Average accuracy:     {rd['accuracy'].mean()*100:.1f}%")
print(f"   Average baseline:     {rd['baseline'].mean()*100:.1f}%")
print(f"   Average improvement:  {rd['improvement'].mean()*100:+.1f}%")
print(f"   Seasons beat baseline:{(rd['improvement']>0).sum()}/{len(rd)}")
print(f"   Best season:  {int(rd.loc[rd['accuracy'].idxmax(),'season'])} "
      f"— {rd['accuracy'].max()*100:.1f}%")
print(f"   Worst season: {int(rd.loc[rd['accuracy'].idxmin(),'season'])} "
      f"— {rd['accuracy'].min()*100:.1f}%")

# ── Vegas comparison ──────────────────────────────
print(f"\n{'='*60}")
print(f"🎰 OUR MODEL vs VEGAS SPREAD:")

vd = data.dropna(subset=['spread_line']).copy()
X_v = get_X(vd)

vegas_picks = (vd['spread_line'] < 0).astype(int)  # Negative = home favored
our_picks   = model.predict(X_v)
actual      = vd['home_win']

vegas_acc = accuracy_score(actual, vegas_picks)
our_acc   = accuracy_score(actual, our_picks)
agreement = (vegas_picks.values == our_picks).mean()

print(f"   Vegas accuracy:       {vegas_acc*100:.1f}%")
print(f"   Our model accuracy:   {our_acc*100:.1f}%")
print(f"   Difference:           {(our_acc-vegas_acc)*100:+.1f}%")
print(f"   Agree with Vegas:     {agreement*100:.1f}% of the time")

# ── Confidence calibration ────────────────────────
print(f"\n{'='*60}")
print(f"🎯 CONFIDENCE CALIBRATION:")
print(f"   When we predict X% confidence, how often are we right?")
print(f"\n   {'Confidence':<15} {'Games':<8} {'Predicted':<12} {'Actual Win %':<12} {'Calibrated?'}")
print(f"   {'-'*55}")

X_all  = get_X(data)
probs  = model.predict_proba(X_all)[:,1]
actual = data['home_win'].values

bins = [(0.50,0.55),(0.55,0.60),(0.60,0.65),
        (0.65,0.70),(0.70,0.80),(0.80,1.00)]

for low, high in bins:
    mask = (probs >= low) & (probs < high)
    n    = mask.sum()
    if n > 10:
        actual_pct = actual[mask].mean()
        mid        = (low + high) / 2
        calibrated = "✅" if abs(actual_pct - mid) < 0.08 else "⚠️"
        print(f"   {low:.0%}–{high:.0%}{'':<8} {n:<8} {mid:.0%}{'':<9} "
              f"{actual_pct*100:.1f}%{'':<9} {calibrated}")

# ── Playoff accuracy ──────────────────────────────
playoffs = games[games['game_type'] != 'REG'].copy()
playoffs = playoffs.dropna(subset=['home_score','away_score'])
playoffs['temp']     = playoffs['temp'].fillna(playoffs['temp'].median())
playoffs['wind']     = playoffs['wind'].fillna(0)
playoffs['is_dome']  = (playoffs['roof'] == 'dome').astype(int)
playoffs['is_grass'] = (playoffs['surface'].str.contains('grass', na=False)).astype(int)
playoffs['home_win'] = (playoffs['home_score'] > playoffs['away_score']).astype(int)

po_data = playoffs[keep_cols].dropna().copy()
if len(po_data) > 0:
    print(f"\n{'='*60}")
    print(f"🏆 PLAYOFF ACCURACY:")
    X_po    = get_X(po_data)
    po_pred = model.predict(X_po)
    po_acc  = accuracy_score(po_data['home_win'], po_pred)
    print(f"   Playoff games tested: {len(po_data)}")
    print(f"   Playoff accuracy:     {po_acc*100:.1f}%")
    print(f"   (Playoffs are harder — upsets more common)")

# ── Biggest misses ────────────────────────────────
print(f"\n{'='*60}")
print(f"😬 10 BIGGEST UPSETS WE MISSED:")
print(f"\n   {'Season':<7} {'Wk':<4} {'Home':<5} {'Away':<5} "
      f"{'Score':<10} {'Our Conf':<10} {'Actual'}")
print(f"   {'-'*55}")

data2 = data.copy()
data2['prob']      = probs
data2['predicted'] = model.predict(X_all)
data2['correct']   = (data2['predicted'] == data2['home_win'])
data2['error']     = abs(data2['prob'] - data2['home_win'])

worst = data2[data2['correct'] == False]\
    .sort_values('error', ascending=False).head(10)

for _, row in worst.iterrows():
    score  = f"{int(row['home_score'])}-{int(row['away_score'])}"
    result = "HOME WIN" if row['home_win'] else "AWAY WIN"
    conf   = f"{max(row['prob'], 1-row['prob'])*100:.0f}% {('home' if row['prob']>0.5 else 'away')}"
    print(f"   {int(row['season']):<7} {int(row['week']):<4} "
          f"{row['home_team']:<5} {row['away_team']:<5} "
          f"{score:<10} {conf:<10} {result}")

print(f"\n✅ Backtest complete!")