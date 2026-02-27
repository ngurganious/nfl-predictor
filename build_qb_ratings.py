"""
NFL QB Quality Ratings
=======================
Pulls per-QB seasonal passing stats from nfl_data_py and computes a composite
quality score (z-score across key efficiency metrics, normalised within season
so era differences don't distort comparisons).

Metrics used (all from regular-season play):
    completion_pct    — accuracy
    yards_per_att     — volume efficiency
    td_int_ratio      — ball security
    sack_rate         — pocket awareness (inverted — lower is better)
    air_yards_per_att — downfield aggressiveness / big-play ability

Run:
    python build_qb_ratings.py

Writes:
    qb_ratings.csv       — historical per-(player_id, season) scores
                           used by retrain_model.py to add QB quality as a feature
    qb_team_ratings.csv  — current season, one row per team
                           used by final_app.py and data_pipeline.py at prediction time
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

try:
    import nfl_data_py as nfl
except ImportError:
    print("ERROR: nfl_data_py not installed.  Run: pip install nfl-data-py")
    sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────────
from datetime import datetime as _dt
_cur = _dt.now().year
SEASONS = list(range(2010, _cur + 1))   # will retry without current year if 404
MIN_ATTEMPTS     = 100   # minimum pass attempts to qualify for a rating
QB_RATINGS_FILE  = Path("qb_ratings.csv")
QB_TEAM_FILE     = Path("qb_team_ratings.csv")

# Normalisation mapping: nfl_data_py column → friendly name
# (column names vary slightly by version — we try multiple)
_STAT_ALIASES: dict[str, list[str]] = {
    "completions":        ["completions"],
    "attempts":           ["attempts"],
    "passing_yards":      ["passing_yards"],
    "passing_tds":        ["passing_tds"],
    "interceptions":      ["interceptions"],
    "sacks":              ["sacks"],
    "passing_air_yards":  ["passing_air_yards", "air_yards"],
}


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map aliased column names to canonical names."""
    for canonical, aliases in _STAT_ALIASES.items():
        if canonical not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    df[canonical] = df[alias]
                    break
    return df


def compute_qb_ratings(seasons: list[int]) -> pd.DataFrame:
    """
    Pull seasonal passing stats and compute composite QB quality score.
    Returns a DataFrame with columns:
        player_id, player_name, team, season, attempts,
        completion_pct, yards_per_att, td_int_ratio, sack_rate,
        qb_score  (z-score composite), qb_pct (0-100 percentile)
    """
    # Drop seasons from the end until we get a non-empty result
    # (nfl_data_py raises 404 for seasons that haven't been published yet)
    raw = pd.DataFrame()
    attempt_seasons = list(seasons)
    while attempt_seasons:
        print(f"  Downloading seasonal data for {attempt_seasons[0]}–{attempt_seasons[-1]} ...")
        try:
            result = nfl.import_seasonal_data(attempt_seasons, s_type="REG")
            if result is not None and not result.empty:
                raw = result
                break
            print(f"  Empty result — dropping {attempt_seasons[-1]} ...")
        except Exception as e:
            print(f"  {e} — dropping {attempt_seasons[-1]} ...")
        attempt_seasons = attempt_seasons[:-1]

    if raw.empty:
        print("  ERROR: Could not download seasonal data for any year.")
        return pd.DataFrame()

    if raw is None or raw.empty:
        print("  No data returned.")
        return pd.DataFrame()

    raw = _resolve_columns(raw)

    # Filter: QBs with enough attempts
    required = ["player_id", "season", "attempts",
                "completions", "passing_yards", "passing_tds", "interceptions"]
    missing_req = [c for c in required if c not in raw.columns]
    if missing_req:
        print(f"  Missing required columns: {missing_req}")
        print(f"  Available: {list(raw.columns)}")
        return pd.DataFrame()

    # Filter QBs: anyone with enough attempts is a QB (no position column available)
    passers = raw[raw["attempts"] >= MIN_ATTEMPTS].copy()
    print(f"  {len(passers)} QB-seasons qualify ({MIN_ATTEMPTS}+ attempts)")

    # Build player_id → name / team from games_processed.csv
    id_to_name: dict[str, str] = {}
    id_to_team: dict[str, str] = {}
    games_file = Path("games_processed.csv")
    if games_file.exists():
        gdf = pd.read_csv(games_file, usecols=["home_qb_id", "home_qb_name", "home_team",
                                                "away_qb_id", "away_qb_name", "away_team",
                                                "season"])
        for _, r in gdf.iterrows():
            for side in ("home", "away"):
                pid  = r.get(f"{side}_qb_id")
                name = r.get(f"{side}_qb_name")
                team = r.get(f"{side}_team")
                if pd.notna(pid) and pd.notna(name):
                    id_to_name[str(pid)] = str(name)
                    id_to_team[str(pid)] = str(team)

    passers["player_id_str"] = passers["player_id"].astype(str)
    passers["player_name"]   = passers["player_id_str"].map(id_to_name).fillna("Unknown")
    passers["team"]          = passers["player_id_str"].map(id_to_team).fillna("UNK")

    # ── Component metrics ──────────────────────────────────────────────────────
    eps = 1e-6
    passers["completion_pct"]    = passers["completions"]     / (passers["attempts"] + eps)
    passers["yards_per_att"]     = passers["passing_yards"]   / (passers["attempts"] + eps)
    passers["td_int_ratio"]      = passers["passing_tds"]     / (passers["interceptions"].clip(lower=1))
    passers["sack_rate"]         = passers.get("sacks", 0)    / (passers["attempts"] + passers.get("sacks", 0) + eps)

    if "passing_air_yards" in passers.columns:
        passers["air_yards_per_att"] = passers["passing_air_yards"] / (passers["attempts"] + eps)
    else:
        passers["air_yards_per_att"] = np.nan

    metrics_pos = ["completion_pct", "yards_per_att", "td_int_ratio", "air_yards_per_att"]
    metrics_neg = ["sack_rate"]  # lower = better, so subtract from composite

    # ── Season-normalised z-score composite ───────────────────────────────────
    records = []
    for season, grp in passers.groupby("season"):
        grp    = grp.copy()
        scores = pd.Series(0.0, index=grp.index)
        n_used = 0

        for m in metrics_pos:
            col = grp[m].dropna()
            if len(col) > 3 and col.std() > 0:
                scores += (grp[m].fillna(grp[m].median()) - col.mean()) / col.std()
                n_used += 1

        for m in metrics_neg:
            col = grp[m].dropna()
            if len(col) > 3 and col.std() > 0:
                scores -= (grp[m].fillna(grp[m].median()) - col.mean()) / col.std()
                n_used += 1

        grp["qb_score"] = (scores / max(n_used, 1)).round(4)

        # Percentile rank within season (0 = worst, 100 = best)
        grp["qb_pct"] = (rankdata(grp["qb_score"]) / len(grp) * 100).round(1)

        # Team: prefer recent_team column
        team_col = "recent_team" if "recent_team" in grp.columns else (
                   "team"        if "team"        in grp.columns else None)

        keep_cols = ["player_id", "player_name", "season", "attempts",
                     "completions", "passing_yards", "passing_tds", "interceptions",
                     "completion_pct", "yards_per_att", "td_int_ratio",
                     "sack_rate", "qb_score", "qb_pct"]
        if team_col:
            grp = grp.rename(columns={team_col: "team"})
            keep_cols.insert(3, "team")

        records.append(grp[[c for c in keep_cols if c in grp.columns]])

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


def build_team_qb_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    From the full historical ratings, produce a current-season table
    with one row per team showing the starter (highest attempts).
    """
    if df.empty or "season" not in df.columns:
        return pd.DataFrame()

    current_season = df["season"].max()
    current = df[df["season"] == current_season].copy()

    if len(current) < 10:
        # Offseason — fall back to previous season
        current_season = sorted(df["season"].unique())[-2]
        current = df[df["season"] == current_season].copy()
        print(f"  Using {current_season} season for team ratings (current season sparse).")

    if "team" not in current.columns:
        print("  No team column — cannot build team QB ratings.")
        return pd.DataFrame()

    # Pick the QB with most attempts per team as the presumed starter
    team_qbs = (
        current
        .sort_values("attempts", ascending=False)
        .drop_duplicates("team")
        .set_index("team")
        [["player_id", "player_name", "season", "attempts",
          "completion_pct", "yards_per_att", "td_int_ratio",
          "sack_rate", "qb_score", "qb_pct"]]
    )
    return team_qbs


def main():
    print("=" * 60)
    print("  NFL QB Quality Ratings Builder")
    print("=" * 60)

    df = compute_qb_ratings(SEASONS)

    if df.empty:
        print("\nERROR: No QB data retrieved.")
        return pd.DataFrame()

    # Save historical ratings (for model training)
    df.to_csv(QB_RATINGS_FILE, index=False)
    seasons_range = f"{df['season'].min()}–{df['season'].max()}"
    print(f"\n  Historical QB ratings saved -> {QB_RATINGS_FILE}")
    print(f"  {len(df)} QB-seasons ({seasons_range})")

    # Build and save current team ratings
    team_qbs = build_team_qb_ratings(df)
    if not team_qbs.empty:
        team_qbs.to_csv(QB_TEAM_FILE)
        print(f"  Team QB ratings saved -> {QB_TEAM_FILE}  ({len(team_qbs)} teams)")

        current_season = team_qbs["season"].iloc[0] if "season" in team_qbs.columns else "?"
        print(f"\n  Top 12 QBs ({current_season} season):")
        print(f"  {'Team':<6} {'Name':<22} {'Score':>7} {'Pct':>6} "
              f"{'Cmp%':>7} {'Y/A':>6} {'TD/INT':>8}")
        print(f"  {'-'*62}")

        display_cols = ["player_name", "qb_score", "qb_pct",
                        "completion_pct", "yards_per_att", "td_int_ratio"]
        for team, row in team_qbs.nlargest(12, "qb_score").iterrows():
            print(f"  {team:<6} {str(row.get('player_name','?')):<22} "
                  f"{row.get('qb_score', 0):>7.3f} "
                  f"{row.get('qb_pct', 0):>5.0f}%  "
                  f"{row.get('completion_pct', 0):>6.1%}  "
                  f"{row.get('yards_per_att', 0):>5.1f}  "
                  f"{row.get('td_int_ratio', 0):>7.2f}")
    else:
        print("  Could not build team QB ratings.")

    print("=" * 60)
    return df


if __name__ == "__main__":
    main()
