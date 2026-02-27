"""
NFL Predictor — Over/Under Model (v2)
======================================
Trains a regressor to predict the *residual* from the Vegas total line
(actual_total - total_line) rather than the raw total score.

Why residual? Lower variance target (±10 pts vs ±47 pts) converges faster,
and the O/U decision is exactly: predicted_residual > 0 → over, < 0 → under.

v2 additions vs v1:
  - EPA features (home/away off & def EPA per play)
  - Combined scoring-environment features (sum of both teams' EPA)
  - QB quality differential
  - ELO difference (blowout risk)
  - Divisional game flag (historically ~1.5 pts lower scoring)
  - Spread implied probability (competitiveness proxy)
  - Tuned GBR hyperparameters for the residual task
  - Ridge regressor comparison (more stable on small residual target)
  - Picks the better of GBR vs Ridge by CV MAE

Run:
    python build_total_model.py

Writes:
    model_total.pkl — {model, features, mae, ou_accuracy}
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from feature_engineering import build_enhanced_features

warnings.filterwarnings("ignore")

DATA_FILE = Path("games_processed.csv")
OUT_MODEL = Path("model_total.pkl")

# All features available for predicting total scoring
# (total_line is NOT included — we predict the residual from it)
TOTAL_FEATURES = [
    # ── Team scoring trends (rolling 5-game) ──────────────────────────
    "home_l5_pts_for",    "away_l5_pts_for",
    "home_l5_pts_against","away_l5_pts_against",
    "home_l5_pts_diff",   "away_l5_pts_diff",
    "matchup_adv_home",   "matchup_adv_away",
    # ── EPA efficiency (offensive and defensive per play) ──────────────
    # These are the most reliable signals for scoring environment
    "home_off_epa",       "away_off_epa",
    "home_def_epa",       "away_def_epa",
    "scoring_env_off",    # home_off_epa + away_off_epa (both offenses)
    "scoring_env_def",    # home_def_epa + away_def_epa (both defenses)
    # ── QB and team quality ────────────────────────────────────────────
    "qb_score_diff",      # strong QB = more points; -diff = away QB advantage
    "elo_diff",           # large gap → potential blowout → late-game conservatism
    "abs_spread",         # blowout proxy: big spread → lower total (game mgmt)
    # ── Game context ───────────────────────────────────────────────────
    "div_game",           # divisional games average ~1.5 pts lower scoring
    "spread_implied_prob",# competitiveness: close games stay competitive longer
    "home_rest", "away_rest",
    # ── Weather ────────────────────────────────────────────────────────
    "wind", "temp", "is_dome",
]


def main():
    print("=" * 60)
    print("  NFL Predictor — Over/Under Model Training (v2)")
    print("=" * 60)

    # ── 1. Load ───────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    df = pd.read_csv(DATA_FILE)
    df = df[df["game_type"] == "REG"].copy()
    df = df.dropna(subset=["home_score", "away_score", "total_line"])
    df["total"] = df["home_score"] + df["away_score"]
    print(f"      {len(df):,} regular season games "
          f"({df['season'].min()}–{df['season'].max()})")
    print(f"      Total score range: {df['total'].min():.0f}–{df['total'].max():.0f} pts "
          f"(mean {df['total'].mean():.1f})")

    # ── 2. Feature engineering ────────────────────────────────────────────
    print("\n[2/5] Engineering features (~60-90 sec) ...")
    df_eng = build_enhanced_features(df)

    # Inline combined features
    df_eng["abs_spread"]      = df_eng["spread_line"].abs()
    df_eng["scoring_env_off"] = (df_eng.get("home_off_epa", 0) + df_eng.get("away_off_epa", 0))
    df_eng["scoring_env_def"] = (df_eng.get("home_def_epa", 0) + df_eng.get("away_def_epa", 0))

    # Residual target: how much did the actual total deviate from Vegas?
    df_eng["total_residual"] = df_eng["total"] - df_eng["total_line"]
    residual_std = df_eng["total_residual"].std()
    print(f"      Residual (actual - Vegas): mean={df_eng['total_residual'].mean():.2f}, "
          f"std={residual_std:.2f} pts")

    # ── 3. Prepare dataset ────────────────────────────────────────────────
    print("\n[3/5] Preparing dataset ...")
    available = [f for f in TOTAL_FEATURES if f in df_eng.columns]
    missing   = [f for f in TOTAL_FEATURES if f not in df_eng.columns]
    if missing:
        print(f"      Skipping unavailable features: {missing}")

    model_data = df_eng[available + ["total", "total_line", "total_residual", "season"]].dropna()
    model_data = model_data.sort_values("season").reset_index(drop=True)

    X = model_data[available]
    y = model_data["total_residual"]        # train on residual
    y_total  = model_data["total"]          # for MAE reporting
    y_line   = model_data["total_line"]     # Vegas baseline

    seasons       = sorted(model_data["season"].unique())
    holdout_start = seasons[-2]
    train_mask    = model_data["season"] < holdout_start
    test_mask     = model_data["season"] >= holdout_start

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    y_total_test    = y_total[test_mask]
    y_line_test     = y_line[test_mask]

    print(f"      Train: {len(X_train):,} games ({seasons[0]}–{holdout_start-1})")
    print(f"      Test:  {len(X_test):,} games ({holdout_start}–{seasons[-1]})")
    print(f"      Features: {len(available)}")

    vegas_test_mae = mean_absolute_error(y_total_test, y_line_test)
    print(f"      Vegas baseline MAE on test set: {vegas_test_mae:.2f} pts")

    # ── 4. Cross-validate GBR vs Ridge ────────────────────────────────────
    print("\n[4/5] Cross-validation: GBR vs Ridge ...")
    tscv = TimeSeriesSplit(n_splits=5)

    gbr = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.02, max_depth=3,
        min_samples_leaf=20, subsample=0.75, random_state=42,
    )
    ridge = Ridge(alpha=10.0)

    cv_results = {}
    for name, est in [("GBR  ", gbr), ("Ridge", ridge)]:
        maes = []
        for tr_idx, va_idx in tscv.split(X_train):
            Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            ytr, yva = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            est.fit(Xtr, ytr)
            pred_resid = est.predict(Xva)
            # Convert residual back to total for MAE
            total_pred = Xva["abs_spread"].values * 0  # placeholder
            # MAE on residual = MAE on total (since total_line is constant in fold)
            maes.append(mean_absolute_error(yva, pred_resid))
        cv_results[name] = np.mean(maes)
        print(f"      {name}: CV residual MAE = {np.mean(maes):.3f} +/- {np.std(maes):.3f} pts")

    best_name = min(cv_results, key=cv_results.get)
    best_est  = gbr if "GBR" in best_name else ridge
    print(f"      -> Using {best_name.strip()} for final model")

    # ── 5. Final training and evaluation ──────────────────────────────────
    print("\n[5/5] Final training on full train set, evaluating on holdout ...")
    best_est.fit(X_train, y_train)

    pred_resid_test = best_est.predict(X_test)
    pred_total_test = y_line_test.values + pred_resid_test

    test_mae    = mean_absolute_error(y_total_test, pred_total_test)
    residual_mae = mean_absolute_error(y_test, pred_resid_test)

    # O/U direction accuracy — residual sign predicts over/under
    model_over  = pred_resid_test > 0
    actual_over = y_test.values  > 0
    ou_acc      = (model_over == actual_over).mean()

    # Edge accuracy: only when model is confident (residual ≥ 3 pts)
    edge_mask = np.abs(pred_resid_test) >= 3
    if edge_mask.sum() > 0:
        edge_acc = (model_over[edge_mask] == actual_over[edge_mask]).mean()
        edge_n   = edge_mask.sum()
    else:
        edge_acc, edge_n = 0.5, 0

    # Strong edge: residual ≥ 5 pts
    strong_mask = np.abs(pred_resid_test) >= 5
    if strong_mask.sum() > 0:
        strong_acc = (model_over[strong_mask] == actual_over[strong_mask]).mean()
        strong_n   = strong_mask.sum()
    else:
        strong_acc, strong_n = 0.5, 0

    print(f"\n  {'Metric':<40} {'Vegas':>10} {'Model':>10}")
    print(f"  {'-'*62}")
    print(f"  {'Total MAE (pts)':<40} {vegas_test_mae:>10.2f} {test_mae:>10.2f}")
    print(f"  {'Residual MAE (pts)':<40} {'--':>10} {residual_mae:>10.2f}")
    print(f"  {'O/U direction accuracy (all)':<40} {'50.0%':>10} {ou_acc*100:>9.1f}%")
    lbl3 = f"O/U accuracy (model off 3+ pts, n={edge_n})"
    lbl5 = f"O/U accuracy (model off 5+ pts, n={strong_n})"
    print(f"  {lbl3:<40} {'n/a':>10} {edge_acc*100:>9.1f}%")
    print(f"  {lbl5:<40} {'n/a':>10} {strong_acc*100:>9.1f}%")

    # Feature importance
    print("\n  Top feature importances:")
    if hasattr(best_est, "feature_importances_"):
        imp = sorted(zip(available, best_est.feature_importances_),
                     key=lambda x: x[1], reverse=True)[:12]
    elif hasattr(best_est, "coef_"):
        imp = sorted(zip(available, np.abs(best_est.coef_)),
                     key=lambda x: x[1], reverse=True)[:12]
    else:
        imp = []
    for feat, score in imp:
        bar = "#" * int(score * 100)
        print(f"    {feat:<30} {bar} {score:.4f}")

    # Save
    with open(OUT_MODEL, "wb") as f:
        pickle.dump({
            "model":       best_est,
            "features":    available,
            "mae":         test_mae,
            "ou_accuracy": ou_acc,
        }, f)

    print(f"\n  Model saved -> {OUT_MODEL}")
    print(f"\n  RESULT: Total MAE {test_mae:.2f} pts | O/U accuracy {ou_acc*100:.1f}% "
          f"on {holdout_start}-{seasons[-1]} test set")
    if ou_acc > 0.506:
        print(f"  IMPROVEMENT: +{(ou_acc-0.506)*100:.1f}pp over v1 baseline (50.6%)")
    print("=" * 60)

    return best_est, available, test_mae, ou_acc


if __name__ == "__main__":
    main()
