# ============================================
# NFL PREDICTOR - Phase 1: Data Exploration
# ============================================
# The '#' symbol means "comment" - Python ignores these lines.
# We use them to explain what the code is doing. Get in the habit!

import nfl_data_py as nfl  # Our NFL data library
import pandas as pd         # Data manipulation (like Excel on steroids)

print("Loading NFL game data... this may take a moment!")

# Pull every game from 2000 to 2024
# nfl.import_schedules() fetches game results, teams, scores, conditions
years = list(range(2000, 2026))  # Creates [2000, 2001, 2002, ... 2024]
games = nfl.import_schedules(years)

# Let's see what we're working with
print(f"\n✅ Loaded {len(games)} games!")        # How many total games
print(f"\n📋 Available data columns:\n")
for col in games.columns:
    print(f"  - {col}")                          # Print every data field available

# Preview the first few rows
print(f"\n🏈 Sample of the data (first 3 games):\n")
print(games.head(3).to_string())

# Save it so we don't have to re-download every time
games.to_csv("games_raw.csv", index=False)
print("\n💾 Saved to games_raw.csv")
