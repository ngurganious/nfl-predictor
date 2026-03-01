"""
build_nhl_model.py
==================
Trains the NHL game prediction stacking ensemble.
Mirrors retrain_model.py from the NFL predictor.

Architecture:
  - Base models: GradientBoostingClassifier + RandomForestClassifier
  - Meta-learner: LogisticRegression (calibrates probabilities)
  - CV: TimeSeriesSplit(5) for evaluation, cv=5 (StratifiedKFold) inside StackingClassifier
  - Holdout: last 2 seasons as test set

Saves:
  - model_nhl_enhanced.pkl   {'model': stack, 'features': list, 'accuracy': float, 'brier': float}
  - model_nhl_total.pkl      {'model': ridge, 'features': list, 'mae': float, 'ou_accuracy': float}

Usage:
    python build_nhl_model.py
"""

import os
import sys
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhl_feature_engineering import build_nhl_enhanced_features, NHL_ENHANCED_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HOLDOUT_SEASONS = 2   # Hold out last N seasons for testing
RANDOM_STATE    = 42
DATA_PATH       = "nhl_games_processed.csv"


def build_game_model(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """Train the stacking ensemble and evaluate on holdout."""
    features = NHL_ENHANCED_FEATURES
    target   = 'home_win'

    X_train = df_train[features].fillna(0.0)
    y_train = df_train[target].astype(int)
    X_test  = df_test[features].fillna(0.0)
    y_test  = df_test[target].astype(int)

    logger.info(f"Training: {len(X_train):,} games | Test: {len(X_test):,} games")
    logger.info(f"Features: {len(features)}")
    logger.info(f"Train home win rate: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")

    # ── Cross-validation on training set ─────────────────────────────────────
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

    # ── Stacking ensemble ─────────────────────────────────────────────────────
    # cv=5 (StratifiedKFold) inside StackingClassifier avoids the TimeSeriesSplit
    # + CalibratedClassifierCV bug seen in NFL development
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

    # ── Evaluate on holdout ───────────────────────────────────────────────────
    probs  = stack.predict_proba(X_test)[:, 1]
    preds  = (probs >= 0.5).astype(int)
    acc    = accuracy_score(y_test, preds)
    brier  = brier_score_loss(y_test, probs)
    baseline = y_test.mean()  # Always predict home wins

    logger.info(f"\nHoldout accuracy: {acc:.3f} ({acc*100:.1f}%)")
    logger.info(f"Baseline (always home): {baseline:.3f} ({baseline*100:.1f}%)")
    logger.info(f"Improvement over baseline: +{(acc - baseline)*100:.1f}%")
    logger.info(f"Brier score: {brier:.4f}")

    # Per-season breakdown
    df_eval = df_test.copy()
    df_eval['pred'] = preds
    df_eval['prob'] = probs
    logger.info("\nAccuracy by season (holdout):")
    for season, grp in df_eval.groupby('season'):
        sacc = accuracy_score(grp['home_win'], grp['pred'])
        logger.info(f"  {season}-{season+1}: {sacc:.3f} ({len(grp)} games)")

    # Feature importance (from GBC base model)
    gbc_fitted = stack.estimators_[0]
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


def build_total_model(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """
    Train the O/U (total goals) Ridge regression model.
    Target: actual_total - total_goals_line (residual)
    Features: team offensive/defensive form + xG
    """
    # O/U features (subset of NHL_ENHANCED_FEATURES)
    ou_features = [
        'home_l5_goals_for', 'away_l5_goals_for',
        'home_l5_goals_against', 'away_l5_goals_against',
        'home_off_xg_pct', 'away_off_xg_pct',
        'home_def_xg_pct', 'away_def_xg_pct',
        'xg_total_diff',
        'goalie_quality_diff',
        'nhl_elo_diff',
    ]

    # We need total_goals and puck_line for target computation
    # total_goals = actual goals in game
    # puck_line used as proxy for total_goals_line (we don't have historical lines)
    # Use league average (~6.0 goals) as the baseline line
    LEAGUE_AVG_TOTAL = 6.0

    df_train = df_train.copy()
    df_test  = df_test.copy()

    df_train['total_goals'] = df_train['home_score'] + df_train['away_score']
    df_test['total_goals']  = df_test['home_score']  + df_test['away_score']

    # Without historical Vegas totals, train on residual from league average
    df_train['target_residual'] = df_train['total_goals'] - LEAGUE_AVG_TOTAL
    df_test['target_residual']  = df_test['total_goals']  - LEAGUE_AVG_TOTAL

    X_train = df_train[ou_features].fillna(0.0)
    y_train = df_train['target_residual']
    X_test  = df_test[ou_features].fillna(0.0)
    y_test  = df_test['target_residual']

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    preds_test = ridge.predict(X_test)
    mae = mean_absolute_error(y_test, preds_test)

    # Over/under accuracy (vs league average line = 6.0)
    predicted_total = preds_test + LEAGUE_AVG_TOTAL
    actual_total    = df_test['total_goals']
    # Model says OVER if predicted > 6.0, UNDER if < 6.0
    ou_correct = np.sum(
        (predicted_total > LEAGUE_AVG_TOTAL) == (actual_total > LEAGUE_AVG_TOTAL)
    )
    ou_acc = ou_correct / len(actual_total)

    logger.info(f"\nO/U Model MAE: {mae:.2f} goals")
    logger.info(f"O/U Accuracy (vs {LEAGUE_AVG_TOTAL} line): {ou_acc:.3f} ({ou_acc*100:.1f}%)")

    return {
        'model':           ridge,
        'features':        ou_features,
        'mae':             mae,
        'ou_accuracy':     ou_acc,
        'league_avg_total': LEAGUE_AVG_TOTAL,
    }


def main():
    logger.info("=== NHL Model Training ===")

    # ── Load and engineer features ────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        logger.error(f"{DATA_PATH} not found. Run build_nhl_games.py first.")
        sys.exit(1)

    logger.info(f"Loading {DATA_PATH}...")
    df_raw = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df_raw):,} games ({df_raw['season'].min()}-{df_raw['season'].max()})")

    logger.info("Building enhanced features...")
    df_eng = build_nhl_enhanced_features(df_raw)

    # ── Train/test split ──────────────────────────────────────────────────────
    max_season = df_eng['season'].max()
    holdout_start = max_season - HOLDOUT_SEASONS + 1

    df_train = df_eng[df_eng['season'] < holdout_start].copy()
    df_test  = df_eng[df_eng['season'] >= holdout_start].copy()

    logger.info(f"\nTrain seasons: {df_train['season'].min()}-{df_train['season'].max()} ({len(df_train):,} games)")
    logger.info(f"Test seasons:  {df_test['season'].min()}-{df_test['season'].max()} ({len(df_test):,} games)")

    # ── Build game model ──────────────────────────────────────────────────────
    game_pkg = build_game_model(df_train, df_test)
    with open("model_nhl_enhanced.pkl", "wb") as f:
        pickle.dump(game_pkg, f)
    logger.info(f"\nSaved model_nhl_enhanced.pkl ({game_pkg['accuracy']*100:.1f}% accuracy)")

    # Smoke test
    test_fv = pd.DataFrame([{feat: 0.0 for feat in NHL_ENHANCED_FEATURES}])
    prob = game_pkg['model'].predict_proba(test_fv)[0][1]
    assert 0.40 < prob < 0.60, f"Smoke test failed: neutral features gave prob={prob:.3f}"
    logger.info(f"Smoke test passed: neutral features → {prob:.3f} win probability")

    # ── Build O/U model ───────────────────────────────────────────────────────
    total_pkg = build_total_model(df_train, df_test)
    with open("model_nhl_total.pkl", "wb") as f:
        pickle.dump(total_pkg, f)
    logger.info(f"Saved model_nhl_total.pkl (MAE={total_pkg['mae']:.2f}, OU acc={total_pkg['ou_accuracy']*100:.1f}%)")

    logger.info("\n=== Training complete ===")
    logger.info(f"Game model: {game_pkg['accuracy']*100:.1f}% accuracy (baseline: {game_pkg['baseline']*100:.1f}%)")


if __name__ == "__main__":
    main()
