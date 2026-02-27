import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Reload and resave everything with protocol=2 (universal compatibility)
print("Reloading and resaving all models...")

files = [
    'model.pkl',
    'elo_ratings.pkl', 
    'pass_yards_model.pkl',
    'rush_yards_model.pkl',
    'rec_yards_model.pkl',
    'player_lookup.pkl'
]

for fname in files:
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, protocol=2)
    print(f"✅ Resaved {fname}")

print("\n💾 All models resaved with universal compatibility!")