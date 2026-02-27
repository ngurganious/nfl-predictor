# ================================================
# NFL PREDICTOR - Lineup Model
# Build starting lineup data + depth charts
# ================================================

import nfl_data_py as nfl
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

print("📥 Downloading depth charts + player stats...")

# ── Pull depth charts (tells us who starts) ──────
depth = nfl.import_depth_charts([2024])
depth = depth[depth['season'] == 2024].copy()

print(f"✅ Loaded depth chart: {len(depth)} entries")
print(f"\n📋 Depth chart columns:")
for col in depth.columns:
    print(f"   - {col}")

# ── Pull seasonal player stats ────────────────────
print("\n📥 Downloading 2024 player season stats...")
player_stats = nfl.import_seasonal_data([2024])

print(f"✅ Loaded player stats: {len(player_stats)} players")
print(f"\n📋 Player stat columns:")
for col in player_stats.columns:
    print(f"   - {col}")

# Save raw data
depth.to_csv('depth_charts.csv', index=False)
player_stats.to_csv('player_season_stats.csv', index=False)

print("\n💾 Saved depth_charts.csv and player_season_stats.csv")