# ============================================
# NFL PREDICTOR - Phase 2: Data Analysis
# ============================================

import pandas as pd
import numpy as np

# Load our saved data (no need to re-download!)
games = pd.read_csv("games_raw.csv")

# Only look at completed regular season games with scores
games = games[games['game_type'] == 'REG'].copy()
games = games.dropna(subset=['home_score', 'away_score'])

print(f"📊 Analyzing {len(games)} regular season games (2000-2024)\n")

# ── INSIGHT 1: Home field advantage ──────────────────────────────
home_wins = (games['home_score'] > games['away_score']).sum()
total = len(games)
print(f"🏠 HOME FIELD ADVANTAGE:")
print(f"   Home teams win {home_wins}/{total} games = {home_wins/total*100:.1f}% of the time\n")

# ── INSIGHT 2: Does rest matter? ─────────────────────────────────
# "Short rest" = less than 7 days (Thursday games etc.)
short_rest_home = games[games['home_rest'] < 7]
normal_rest_home = games[games['home_rest'] >= 7]

short_win_pct = (short_rest_home['home_score'] > short_rest_home['away_score']).mean()
normal_win_pct = (normal_rest_home['home_score'] > normal_rest_home['away_score']).mean()

print(f"😴 REST IMPACT ON HOME TEAMS:")
print(f"   Short rest (<7 days) win rate:  {short_win_pct*100:.1f}%")
print(f"   Normal rest (7+ days) win rate: {normal_win_pct*100:.1f}%\n")

# ── INSIGHT 3: Weather effects ───────────────────────────────────
cold_games = games[games['temp'] < 32]
mild_games = games[(games['temp'] >= 32) & (games['temp'] < 75)]
hot_games = games[games['temp'] >= 75]

print(f"🌡️  WEATHER & SCORING:")
print(f"   Freezing (<32°F): avg total score = {cold_games['total'].mean():.1f} pts")
print(f"   Mild (32-75°F):   avg total score = {mild_games['total'].mean():.1f} pts")
print(f"   Hot (75°F+):      avg total score = {hot_games['total'].mean():.1f} pts\n")

# ── INSIGHT 4: Dome vs outdoor ───────────────────────────────────
dome_games = games[games['roof'] == 'dome']
outdoor_games = games[games['roof'] == 'outdoors']

print(f"🏟️  DOME vs OUTDOOR SCORING:")
print(f"   Dome games avg total:    {dome_games['total'].mean():.1f} pts")
print(f"   Outdoor games avg total: {outdoor_games['total'].mean():.1f} pts\n")

# ── INSIGHT 5: Wind effects ──────────────────────────────────────
high_wind = games[games['wind'] >= 20]
low_wind = games[games['wind'] < 20]

print(f"💨 WIND IMPACT:")
print(f"   High wind (20+ mph) avg total: {high_wind['total'].mean():.1f} pts")
print(f"   Low wind (<20 mph) avg total:  {low_wind['total'].mean():.1f} pts\n")

# ── INSIGHT 6: Divisional games ──────────────────────────────────
div = games[games['div_game'] == 1]
non_div = games[games['div_game'] == 0]

div_home_win = (div['home_score'] > div['away_score']).mean()
non_div_home_win = (non_div['home_score'] > non_div['away_score']).mean()

print(f"⚔️  DIVISIONAL RIVALRY EFFECT:")
print(f"   Home win % in division games:     {div_home_win*100:.1f}%")
print(f"   Home win % in non-division games: {non_div_home_win*100:.1f}%\n")

# ── INSIGHT 7: Top coaches all time by win % ─────────────────────
# Combine home and away coaching records
home_records = games.groupby('home_coach').agg(
    wins=('result', lambda x: (x > 0).sum()),
    games=('result', 'count')
)
away_records = games.groupby('away_coach').agg(
    wins=('result', lambda x: (x < 0).sum()),
    games=('result', 'count')
)

coach_wins = home_records['wins'].add(away_records['wins'], fill_value=0)
coach_games = home_records['games'].add(away_records['games'], fill_value=0)
coach_win_pct = (coach_wins / coach_games).round(3)

coach_df = pd.DataFrame({
    'wins': coach_wins,
    'games': coach_games,
    'win_pct': coach_win_pct
}).query('games >= 50').sort_values('win_pct', ascending=False)

print(f"🏆 TOP 10 COACHES BY WIN % (min 50 games):")
print(coach_df.head(10).to_string())

print("\n✅ Analysis complete!")