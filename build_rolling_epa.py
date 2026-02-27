"""
Build Game-Level Rolling EPA
=============================
Computes each team's offensive and defensive EPA per play for their
LAST 5 GAMES before each matchup — much more timely than season averages.

Why this matters vs season EPA:
  - Season EPA in week 14 includes 13 games of context (slow to react)
  - Rolling 5-game EPA captures current form: injuries, scheme changes,
    hot/cold streaks in efficiency terms
  - A team 0-4 in their last 5 with poor EPA is very different from a
    team 4-0 with strong EPA, even if their season averages look similar

Output: rolling_epa.csv
  Columns: team, season, week, l5_off_epa, l5_def_epa
  Meaning: "entering this week, team X's avg EPA over their last 5 games"

Run:
    python build_rolling_epa.py
    (takes ~10-15 min first run; subsequent runs use nfl_data_py cache)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import nfl_data_py as nfl
except ImportError:
    print("ERROR: nfl_data_py not installed. Run: pip install nfl-data-py")
    sys.exit(1)

OUTPUT_FILE = Path("rolling_epa.csv")
SEASONS     = list(range(2016, 2025))   # 2016-2024 inclusive
WINDOW      = 5                          # rolling game window

# Normalise nfl_data_py team abbrevs to match games_processed.csv
NFD_TO_STD = {
    "JAC": "JAX", "LVR": "LV",  "LA": "LA",  "LAR": "LA",
    "OAK": "LV",  "SD":  "LAC", "STL": "LA",
}

def _norm(t: str) -> str:
    return NFD_TO_STD.get(str(t).upper(), str(t).upper())


def build_game_epa(season: int) -> pd.DataFrame:
    """
    For every regular-season game in `season`, compute each team's
    offensive and defensive EPA per play.

    Returns DataFrame: team, season, week, gameday, game_id, off_epa, def_epa
    """
    print(f"  [{season}] Downloading play-by-play ...")
    try:
        pbp = nfl.import_pbp_data([season])
    except Exception as e:
        print(f"  [{season}] ERROR: {e}")
        return pd.DataFrame()

    if pbp is None or pbp.empty or "season_type" not in pbp.columns:
        print(f"  [{season}] No usable data.")
        return pd.DataFrame()

    pbp = pbp[pbp["season_type"] == "REG"].copy()
    pbp["posteam"] = pbp["posteam"].apply(_norm)
    pbp["defteam"] = pbp["defteam"].apply(_norm)

    scrimmage = pbp[pbp["play_type"].isin(["pass", "run", "qb_kneel", "qb_spike"])]
    scrimmage = scrimmage.dropna(subset=["epa", "game_id", "week"])

    # Per-game offensive EPA
    off = (scrimmage.groupby(["posteam", "game_id", "week"])["epa"]
                    .mean().reset_index()
                    .rename(columns={"posteam": "team", "epa": "off_epa"}))

    # Per-game defensive EPA (from the defense's perspective)
    def_ = (scrimmage.groupby(["defteam", "game_id", "week"])["epa"]
                     .mean().reset_index()
                     .rename(columns={"defteam": "team", "epa": "def_epa"}))

    game_epa = off.merge(def_, on=["team", "game_id", "week"])

    # Add gameday from schedule for temporal sorting
    try:
        sched = nfl.import_schedules([season])
        sched = sched[sched["game_type"] == "REG"][["game_id", "gameday"]].dropna()
        game_epa = game_epa.merge(sched, on="game_id", how="left")
    except Exception:
        game_epa["gameday"] = None

    game_epa["season"] = season
    game_epa["week"]   = pd.to_numeric(game_epa["week"], errors="coerce").astype("Int64")
    return game_epa[["team", "season", "week", "gameday", "off_epa", "def_epa"]]


def build_rolling(all_game_epa: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """
    For each team at each (season, week), compute the rolling mean of
    their EPA from their previous N games (shift(1) ensures no leakage).

    Returns: team, season, week, l5_off_epa, l5_def_epa
    """
    df = all_game_epa.copy()
    df = df.sort_values(["team", "season", "week"]).reset_index(drop=True)

    # shift(1) = exclude current game; rolling(window) over previous N games
    # min_periods=2 so we don't get NaN until at least 2 games of history
    df["l5_off_epa"] = (df.groupby("team")["off_epa"]
                          .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean()))
    df["l5_def_epa"] = (df.groupby("team")["def_epa"]
                          .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean()))

    # We want one row per (team, season, week): the rolling EPA entering that week
    result = df[["team", "season", "week", "l5_off_epa", "l5_def_epa"]].copy()
    return result.dropna(subset=["l5_off_epa", "l5_def_epa"])


def main():
    print("=" * 60)
    print("  Building Game-Level Rolling EPA (2016-2024)")
    print("=" * 60)

    all_frames = []
    for season in SEASONS:
        game_epa = build_game_epa(season)
        if not game_epa.empty:
            print(f"  [{season}] {len(game_epa):,} team-game records")
            all_frames.append(game_epa)

    if not all_frames:
        print("ERROR: No data retrieved.")
        return

    print(f"\n  Combining {len(SEASONS)} seasons ...")
    combined = pd.concat(all_frames, ignore_index=True)
    print(f"  {len(combined):,} total team-game records across all seasons")

    print(f"  Computing rolling {WINDOW}-game EPA ...")
    rolling = build_rolling(combined)
    print(f"  {len(rolling):,} rows with valid rolling EPA")

    # Coverage check
    coverage = rolling.groupby("season")["team"].count()
    print(f"\n  Coverage by season (team-weeks with rolling EPA):")
    for s, n in coverage.items():
        print(f"    {s}: {n} team-weeks")

    rolling.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Saved -> {OUTPUT_FILE}")
    print(f"  Columns: {rolling.columns.tolist()}")
    print(f"  EPA range: off [{rolling['l5_off_epa'].min():.3f}, {rolling['l5_off_epa'].max():.3f}]"
          f"  def [{rolling['l5_def_epa'].min():.3f}, {rolling['l5_def_epa'].max():.3f}]")
    print("=" * 60)

    return rolling


if __name__ == "__main__":
    main()
