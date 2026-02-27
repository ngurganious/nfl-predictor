"""
Feature Leakage / Redundancy Audit
====================================
Tests whether spread_line + spread_implied_prob (and elo_diff + elo_implied_prob)
are causing redundancy that hurts model generalisation.

Both pairs are monotonically related (one is a deterministic transform of the other),
so they carry identical information.  This script measures the actual correlation and
runs a fast ablation with GBC CV to see which configuration performs best — WITHOUT
the expensive 5-minute stacking retrain.

Configurations tested:
  A) Baseline — all 34 features (current)
  B) Drop spread_implied_prob  (keep raw spread_line)
  C) Drop elo_implied_prob     (keep raw elo_diff)
  D) Drop both implied probs   (keep raw inputs only)
  E) Drop both RAW inputs      (keep only implied probs)

Run:
    python check_leakage.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

from feature_engineering import ENHANCED_FEATURES, build_enhanced_features

warnings.filterwarnings("ignore")

DATA_FILE = Path("games_processed.csv")
N_CV_SPLITS = 5


def cv_eval(X_train, y_train, label: str):
    """Quick GBC + RF cross-validation — returns mean accuracy / log-loss."""
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    results = {}
    for name, est in [
        ("GBC", GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.02, max_depth=3,
            min_samples_leaf=30, subsample=0.80, max_features=0.75, random_state=42)),
        ("RF ", RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=25,
            max_features="sqrt", n_jobs=-1, random_state=42)),
    ]:
        accs, lls = [], []
        for tr_idx, va_idx in tscv.split(X_train):
            Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            ytr, yva = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            est.fit(Xtr, ytr)
            accs.append(accuracy_score(yva, est.predict(Xva)))
            lls.append(log_loss(yva, est.predict_proba(Xva)[:, 1]))
        results[name] = (np.mean(accs), np.mean(lls))

    gbc_acc, gbc_ll = results["GBC"]
    rf_acc,  rf_ll  = results["RF "]
    avg_acc = (gbc_acc + rf_acc) / 2
    avg_ll  = (gbc_ll  + rf_ll)  / 2

    print(f"  {label:<50}  GBC {gbc_acc*100:.2f}%  RF {rf_acc*100:.2f}%  "
          f"avg {avg_acc*100:.2f}%  ll {avg_ll:.4f}")

    return avg_acc, avg_ll


def main():
    print("=" * 70)
    print("  Feature Redundancy Audit — spread_line vs spread_implied_prob")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("\n[1/3] Loading and engineering features (~60-90s) ...")
    df = pd.read_csv(DATA_FILE)
    df = df[df["game_type"] == "REG"].copy()
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    df_eng = build_enhanced_features(df)

    available = [f for f in ENHANCED_FEATURES if f in df_eng.columns]
    model_data = df_eng[available + ["home_win", "season"]].dropna()
    model_data = model_data.sort_values("season").reset_index(drop=True)

    X = model_data[available]
    y = model_data["home_win"]

    # Hold out last 2 seasons (consistent with retrain_model.py)
    recent = sorted(model_data["season"].unique())
    holdout_start = recent[-2]
    train_mask = model_data["season"] < holdout_start
    X_train = X[train_mask]
    y_train = y[train_mask]

    n_feats = len(available)
    print(f"      {len(X_train):,} training games, {n_feats} features")
    print(f"      Holdout: {holdout_start}+ (not used in ablation CV)")

    # ── 2. Correlation report ─────────────────────────────────────────────
    print("\n[2/3] Correlation between redundant feature pairs:")

    corr_data = X_train.copy()

    pairs = [
        ("spread_line",      "spread_implied_prob"),
        ("elo_diff",         "elo_implied_prob"),
        ("home_l5_pts_diff", "pts_diff_advantage"),   # sanity check
        ("epa_off_diff",     "epa_total_diff"),        # partial overlap
    ]

    for a, b in pairs:
        if a in corr_data.columns and b in corr_data.columns:
            r = corr_data[[a, b]].dropna().corr().iloc[0, 1]
            flag = "  <-- HIGHLY CORRELATED" if abs(r) > 0.90 else ""
            print(f"  {a:<30} vs {b:<25}  r = {r:+.4f}{flag}")

    # ── 3. Ablation ───────────────────────────────────────────────────────
    print(f"\n[3/3] Ablation — {N_CV_SPLITS}-fold TimeSeriesSplit CV on training set:")
    print(f"  (lower log-loss = better calibration; higher acc% = better)\n")

    configs = {
        "A) Baseline (all features)":
            available,
        "B) Drop spread_implied_prob":
            [f for f in available if f != "spread_implied_prob"],
        "C) Drop elo_implied_prob":
            [f for f in available if f != "elo_implied_prob"],
        "D) Drop both implied probs":
            [f for f in available if f not in ("spread_implied_prob", "elo_implied_prob")],
        "E) Drop both RAW inputs (spread_line, elo_diff)":
            [f for f in available if f not in ("spread_line", "elo_diff")],
    }

    results = {}
    for label, feats in configs.items():
        acc, ll = cv_eval(X_train[feats], y_train, label)
        results[label] = (acc, ll, feats)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n  Summary (sorted by avg accuracy):")
    print(f"  {'Config':<50}  {'Acc':>7}  {'LogLoss':>9}  {'Features':>8}")
    print(f"  {'-'*80}")

    baseline_acc, baseline_ll, _ = results["A) Baseline (all features)"]

    for label, (acc, ll, feats) in sorted(results.items(), key=lambda x: -x[1][0]):
        delta = (acc - baseline_acc) * 100
        marker = "  <-- BEST" if label == min(results, key=lambda x: results[x][1]) else ""
        marker = "  *** BEST ACC ***" if acc == max(r[0] for r in results.values()) and label != "A) Baseline (all features)" else marker
        print(f"  {label:<50}  {acc*100:>6.2f}%  {ll:>9.4f}  {len(feats):>7}f  Δ{delta:+.2f}%{marker}")

    print("\n  Interpretation:")
    print("  - If B/C/D outperform A: the derived probability is redundant noise for tree models")
    print("  - If E beats A: the raw inputs are redundant once you have the prob encoding")
    print("  - If A wins: both forms contribute unique signal; keep all")
    print("\n  Recommendation will be printed based on results above.")

    best_label = max(results, key=lambda x: results[x][0])
    best_acc, best_ll, best_feats = results[best_label]

    print(f"\n  Best config: [{best_label}]  {best_acc*100:.2f}%  (ll {best_ll:.4f})")
    if best_label == "A) Baseline (all features)":
        print("  => Keep all features — no redundancy penalty detected.")
    elif "D)" in best_label:
        print("  => Drop BOTH implied probs — raw inputs capture all the signal.")
        print("     Update ENHANCED_FEATURES to remove spread_implied_prob + elo_implied_prob")
        print("     and retrain model_enhanced.pkl.")
    elif "B)" in best_label:
        print("  => Drop spread_implied_prob only.  spread_line alone is sufficient.")
    elif "C)" in best_label:
        print("  => Drop elo_implied_prob only.  elo_diff alone is sufficient.")
    elif "E)" in best_label:
        print("  => Drop raw inputs; use only implied probs (probability space is better signal).")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
