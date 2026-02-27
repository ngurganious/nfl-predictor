"""
Build Game-Level Rolling QB EPA
================================
For each team-game, computes the starting QB's rolling 5-game passing EPA/game
entering that week (no leakage — shift(1) before rolling).

Why this matters vs season-level QB ratings:
  - Season z-score in week 14 includes 13 games (slow to react to injury/form)
  - Rolling 5-game EPA captures: QB injury recovery, backup taking over,
    performance breakout/collapse, new OC adjustments
  - qb_score_diff is the #3 most important feature (~18% combined importance)
    so improving its quality has high upside

Output: rolling_qb_epa.csv
  Columns: team, season, week, l5_qb_epa
  Meaning: "entering this week, team X's primary QB averaged this EPA/game
            over their last 5 starts"

Run:
    python build_rolling_qb_epa.py
    (takes ~5-10 min; fetches weekly player data 2010-2024)
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

OUTPUT_FILE = Path("rolling_qb_epa.csv")
SEASONS     = list(range(2010, 2025))   # 2010-2024 (QB data available from 2010)
WINDOW      = 5
MIN_ATT     = 10  # minimum attempts to be considered the starter in a game

NFD_TO_STD = {
    "JAC": "JAX", "LVR": "LV", "LA": "LA", "LAR": "LA",
    "OAK": "LV",  "SD":  "LAC", "STL": "LA",
}

def _norm(t: str) -> str:
    return NFD_TO_STD.get(str(t).upper(), str(t).upper())


def get_season_qb_epa(season: int) -> pd.DataFrame:
    """
    For each team-game in a season, identify the primary QB (most attempts)
    and return their passing EPA.

    Returns: player_id, player_name, team, season, week, passing_epa
    """
    print(f"  [{season}] Downloading weekly player data ...")
    try:
        wp = nfl.import_weekly_data([season])
    except Exception as e:
        print(f"  [{season}] ERROR: {e}")
        return pd.DataFrame()

    if wp is None or wp.empty:
        return pd.DataFrame()

    # Regular season only
    if "season_type" in wp.columns:
        wp = wp[wp["season_type"] == "REG"].copy()

    qbs = wp[wp["position"] == "QB"].copy()
    qbs = qbs.dropna(subset=["passing_epa", "attempts"])
    qbs = qbs[qbs["attempts"] >= MIN_ATT]

    if qbs.empty:
        print(f"  [{season}] No QB data.")
        return pd.DataFrame()

    qbs["team"] = qbs["recent_team"].apply(_norm)
    qbs["week"] = pd.to_numeric(qbs["week"], errors="coerce").astype("Int64")
    qbs["season"] = season

    # One primary QB per team-game: the one with the most attempts
    qbs = (qbs.sort_values("attempts", ascending=False)
               .drop_duplicates(subset=["team", "week"], keep="first"))

    cols = ["player_id", "player_name", "team", "season", "week", "passing_epa"]
    available = [c for c in cols if c in qbs.columns]
    result = qbs[available].reset_index(drop=True)
    print(f"  [{season}] {len(result):,} team-game QB records")
    return result


def build_rolling_qb(all_qb: pd.DataFrame) -> pd.DataFrame:
    """
    For each (team, season, week), compute rolling mean of their QB's
    passing EPA over the previous WINDOW games (shift(1) = no leakage).

    Returns: team, season, week, l5_qb_epa
    """
    df = all_qb.copy()
    df = df.sort_values(["team", "season", "week"]).reset_index(drop=True)

    df["l5_qb_epa"] = (
        df.groupby("team")["passing_epa"]
          .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=2).mean())
    )

    result = df[["team", "season", "week", "l5_qb_epa"]].copy()
    return result.dropna(subset=["l5_qb_epa"])


def main():
    print("=" * 60)
    print("  Building Game-Level Rolling QB EPA (2010-2024)")
    print("=" * 60)

    all_frames = []
    for season in SEASONS:
        qb_epa = get_season_qb_epa(season)
        if not qb_epa.empty:
            all_frames.append(qb_epa)

    if not all_frames:
        print("ERROR: No data retrieved.")
        return

    print(f"\n  Combining {len(SEASONS)} seasons ...")
    combined = pd.concat(all_frames, ignore_index=True)
    print(f"  {len(combined):,} total team-game QB records")

    print(f"  Computing rolling {WINDOW}-game QB EPA ...")
    rolling = build_rolling_qb(combined)
    print(f"  {len(rolling):,} rows with valid rolling QB EPA")

    coverage = rolling.groupby("season")["team"].count()
    print(f"\n  Coverage by season:")
    for s, n in coverage.items():
        print(f"    {s}: {n} team-weeks")

    rolling.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Saved -> {OUTPUT_FILE}")
    print(f"  Columns: {rolling.columns.tolist()}")
    print(f"  QB EPA range: [{rolling['l5_qb_epa'].min():.3f}, {rolling['l5_qb_epa'].max():.3f}]")
    print("=" * 60)

    return rolling


if __name__ == "__main__":
    main()
