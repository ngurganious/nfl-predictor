"""
build_mlb_total_model.py
========================
Trains the MLB O/U (total runs) Ridge regression model.
Mirrors build_total_model.py from the NFL predictor.

Target: actual_total - LEAGUE_AVG_TOTAL (residual from league average)
Features: team run-scoring form, wOBA, ERA-, FIP-, pitcher quality, ELO

Saves:
  model_mlb_total.pkl  {'model': ridge, 'features': list, 'mae': float,
                        'ou_accuracy': float, 'league_avg_total': float}

Usage:
    python build_mlb_total_model.py
"""

import os
import sys
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_feature_engineering import build_mlb_enhanced_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HOLDOUT_SEASONS  = 2
DATA_PATH        = "mlb_games_processed.csv"
LEAGUE_AVG_TOTAL = 9.1   # historical MLB average (2000-2025)

# O/U features — run-scoring and pitching quality
OU_FEATURES = [
    'home_l5_runs_for',
    'away_l5_runs_for',
    'home_l5_runs_against',
    'away_l5_runs_against',
    'home_l5_run_diff',
    'away_l5_run_diff',
    'home_woba',
    'away_woba',
    'home_era_minus',
    'away_era_minus',
    'home_fip_minus',
    'away_fip_minus',
    'pitcher_quality_diff',
    'mlb_elo_diff',
]


def main():
    logger.info("=== MLB O/U Model Training ===")

    if not os.path.exists(DATA_PATH):
        logger.error(f"{DATA_PATH} not found. Run build_mlb_games.py first.")
        sys.exit(1)

    logger.info(f"Loading {DATA_PATH}...")
    df_raw = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df_raw):,} games")

    logger.info("Building features...")
    df_eng = build_mlb_enhanced_features(df_raw)

    df_eng['total_runs'] = df_eng['home_score'] + df_eng['away_score']
    df_eng['target_residual'] = df_eng['total_runs'] - LEAGUE_AVG_TOTAL

    max_season    = df_eng['season'].max()
    holdout_start = max_season - HOLDOUT_SEASONS + 1

    df_train = df_eng[df_eng['season'] < holdout_start].copy()
    df_test  = df_eng[df_eng['season'] >= holdout_start].copy()

    logger.info(f"Train: {int(df_train['season'].min())}-{int(df_train['season'].max())} ({len(df_train):,} games)")
    logger.info(f"Test:  {int(df_test['season'].min())}-{int(df_test['season'].max())} ({len(df_test):,} games)")
    logger.info(f"League avg total: {df_eng['total_runs'].mean():.2f} runs | Using line: {LEAGUE_AVG_TOTAL}")

    X_train = df_train[OU_FEATURES].fillna(0.0)
    y_train = df_train['target_residual']
    X_test  = df_test[OU_FEATURES].fillna(0.0)
    y_test  = df_test['target_residual']

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    preds_test     = ridge.predict(X_test)
    mae            = mean_absolute_error(y_test, preds_test)
    predicted_total = preds_test + LEAGUE_AVG_TOTAL
    actual_total    = df_test['total_runs']

    ou_correct = np.sum(
        (predicted_total > LEAGUE_AVG_TOTAL) == (actual_total > LEAGUE_AVG_TOTAL)
    )
    ou_acc = ou_correct / len(actual_total)

    logger.info(f"\nO/U MAE: {mae:.2f} runs")
    logger.info(f"O/U Accuracy (vs {LEAGUE_AVG_TOTAL} line): {ou_acc:.3f} ({ou_acc*100:.1f}%)")

    # Per-season breakdown
    df_test = df_test.copy()
    df_test['pred_total'] = predicted_total
    logger.info("\nO/U accuracy by season:")
    for season, grp in df_test.groupby('season'):
        s_actual    = grp['total_runs']
        s_predicted = grp['pred_total']
        s_ou_acc    = np.mean(
            (s_predicted > LEAGUE_AVG_TOTAL) == (s_actual > LEAGUE_AVG_TOTAL)
        )
        logger.info(f"  {int(season)}: {s_ou_acc:.3f} ({len(grp)} games)")

    # Feature coefficients
    logger.info("\nFeature coefficients (Ridge):")
    for feat, coef in sorted(zip(OU_FEATURES, ridge.coef_), key=lambda x: abs(x[1]), reverse=True):
        logger.info(f"  {feat}: {coef:.4f}")

    pkg = {
        'model':            ridge,
        'features':         OU_FEATURES,
        'mae':              mae,
        'ou_accuracy':      ou_acc,
        'league_avg_total': LEAGUE_AVG_TOTAL,
    }
    with open("model_mlb_total.pkl", "wb") as f:
        pickle.dump(pkg, f)
    logger.info(f"\nSaved model_mlb_total.pkl (MAE={mae:.2f}, OU acc={ou_acc*100:.1f}%)")

    logger.info("\n=== Training complete ===")


if __name__ == "__main__":
    main()
