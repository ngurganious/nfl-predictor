"""
build_mlb_model.py
==================
Trains the MLB game prediction stacking ensemble.
Mirrors build_nhl_model.py from the NHL predictor.

Architecture:
  - Base models: GradientBoostingClassifier + RandomForestClassifier
  - Meta-learner: LogisticRegression (calibrates probabilities)
  - CV: TimeSeriesSplit(5) for evaluation
  - Holdout: last 2 seasons as test set

Saves:
  model_mlb_enhanced.pkl  {'model': stack, 'features': list, 'accuracy': float, 'brier': float}

Usage:
    python build_mlb_model.py
"""

import os
import sys
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_feature_engineering import build_mlb_enhanced_features, MLB_ENHANCED_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HOLDOUT_SEASONS = 2
RANDOM_STATE    = 42
DATA_PATH       = "mlb_games_processed.csv"


def build_game_model(df_train, df_test):
    features = MLB_ENHANCED_FEATURES
    target   = 'home_win'

    X_train = df_train[features].fillna(0.0)
    y_train = df_train[target].astype(int)
    X_test  = df_test[features].fillna(0.0)
    y_test  = df_test[target].astype(int)

    logger.info(f"Training: {len(X_train):,} games | Test: {len(X_test):,} games")
    logger.info(f"Features: {len(features)}")
    logger.info(f"Train home win rate: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")

    tscv = TimeSeriesSplit(n_splits=5)

    gbc_base = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.02, max_depth=3,
        min_samples_leaf=30, subsample=0.80, max_features=0.75,
        random_state=RANDOM_STATE
    )
    rf_base = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=25,
        max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1
    )

    logger.info("\nCross-validating base models...")
    gbc_cv = cross_val_score(gbc_base, X_train, y_train, cv=tscv, scoring='accuracy')
    rf_cv  = cross_val_score(rf_base,  X_train, y_train, cv=tscv, scoring='accuracy')
    logger.info(f"  GBC CV: {gbc_cv.mean():.3f} +/- {gbc_cv.std():.3f}")
    logger.info(f"  RF  CV: {rf_cv.mean():.3f} +/- {rf_cv.std():.3f}")

    stack = StackingClassifier(
        estimators=[
            ("gbc", GradientBoostingClassifier(
                n_estimators=500, learning_rate=0.02, max_depth=3,
                min_samples_leaf=30, subsample=0.80, max_features=0.75,
                random_state=RANDOM_STATE
            )),
            ("rf", RandomForestClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=25,
                max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1
            )),
        ],
        final_estimator=LogisticRegression(C=0.5, max_iter=1000),
        cv=5,
        n_jobs=-1,
    )

    logger.info("\nFitting stacking ensemble...")
    stack.fit(X_train, y_train)

    probs    = stack.predict_proba(X_test)[:, 1]
    preds    = (probs >= 0.5).astype(int)
    acc      = accuracy_score(y_test, preds)
    brier    = brier_score_loss(y_test, probs)
    baseline = y_test.mean()

    logger.info(f"\nHoldout accuracy: {acc:.3f} ({acc*100:.1f}%)")
    logger.info(f"Baseline (always home): {baseline:.3f} ({baseline*100:.1f}%)")
    logger.info(f"Improvement over baseline: +{(acc - baseline)*100:.1f}%")
    logger.info(f"Brier score: {brier:.4f}")

    df_eval = df_test.copy()
    df_eval['pred'] = preds
    df_eval['prob'] = probs
    logger.info("\nAccuracy by season (holdout):")
    for season, grp in df_eval.groupby('season'):
        sacc = accuracy_score(grp['home_win'], grp['pred'])
        logger.info(f"  {int(season)}: {sacc:.3f} ({len(grp)} games)")

    gbc_fitted  = stack.estimators_[0]
    importances = dict(zip(features, gbc_fitted.feature_importances_))
    logger.info("\nTop 10 features by importance (GBC):")
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {feat}: {imp:.4f}")

    return {
        'model':    stack,
        'features': features,
        'accuracy': acc,
        'brier':    brier,
        'baseline': baseline,
    }


def main():
    logger.info("=== MLB Model Training ===")

    if not os.path.exists(DATA_PATH):
        logger.error(f"{DATA_PATH} not found. Run build_mlb_games.py first.")
        sys.exit(1)

    logger.info(f"Loading {DATA_PATH}...")
    df_raw = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df_raw):,} games ({int(df_raw['season'].min())}-{int(df_raw['season'].max())})")

    logger.info("Building enhanced features...")
    df_eng = build_mlb_enhanced_features(df_raw)

    max_season    = df_eng['season'].max()
    holdout_start = max_season - HOLDOUT_SEASONS + 1

    df_train = df_eng[df_eng['season'] < holdout_start].copy()
    df_test  = df_eng[df_eng['season'] >= holdout_start].copy()

    logger.info(f"\nTrain: {int(df_train['season'].min())}-{int(df_train['season'].max())} ({len(df_train):,} games)")
    logger.info(f"Test:  {int(df_test['season'].min())}-{int(df_test['season'].max())} ({len(df_test):,} games)")

    game_pkg = build_game_model(df_train, df_test)

    with open("model_mlb_enhanced.pkl", "wb") as f:
        pickle.dump(game_pkg, f)
    logger.info(f"\nSaved model_mlb_enhanced.pkl ({game_pkg['accuracy']*100:.1f}% accuracy)")

    # Smoke test
    test_fv = pd.DataFrame([{feat: 0.0 for feat in MLB_ENHANCED_FEATURES}])
    prob = game_pkg['model'].predict_proba(test_fv)[0][1]
    assert 0.40 < prob < 0.65, f"Smoke test failed: neutral features → prob={prob:.3f}"
    logger.info(f"Smoke test passed: neutral features → {prob:.3f} win probability")

    logger.info("\n=== Training complete ===")
    logger.info(f"Game model: {game_pkg['accuracy']*100:.1f}% accuracy (baseline: {game_pkg['baseline']*100:.1f}%)")


if __name__ == "__main__":
    main()
