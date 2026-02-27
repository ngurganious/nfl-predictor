"""
NFL Team Stats Builder
=======================
Pulls current-season team-level advanced stats from nfl_data_py (free,
no API key required) and saves them to team_stats_current.csv.

Stats computed:
    off_epa_per_play     — Offensive EPA per play (higher = better offense)
    def_epa_per_play     — Defensive EPA per play allowed (lower = better defense)
    pts_for_pg           — Points scored per game
    pts_against_pg       — Points allowed per game
    scoring_margin       — pts_for_pg - pts_against_pg
    to_margin            — Turnovers forced - turnovers committed
    third_down_pct       — Third down conversion rate (offense)
    third_down_allowed   — Third down conversion rate allowed (defense)
    rz_td_pct            — Red zone TD rate (offense, inside opponent 20)
    pass_yards_per_att   — Passing yards per attempt (offense)
    rush_yards_per_carry — Rushing yards per carry (offense)
    sacks_per_game       — Sacks allowed per game (offensive line quality)
    sacks_forced_pg      — Sacks forced per game (defensive pass rush)

Run:
    python build_team_stats.py

Writes:
    team_stats_current.csv  — one row per team, used by data_pipeline.py

Schedule: re-run weekly during the NFL season to keep stats current.
Takes ~60-90 seconds (downloads play-by-play data for current season).
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

OUTPUT_FILE = Path("team_stats_current.csv")

# Team abbreviation normalisation (nfl_data_py uses some different abbrevs)
NFD_TO_STD = {
    "JAC": "JAX", "LVR": "LV", "LA": "LA", "LAR": "LA",
    "OAK": "LV",  "SD": "LAC", "STL": "LA",
}


def _norm(t: str) -> str:
    return NFD_TO_STD.get(str(t).upper(), str(t).upper())


def build_team_stats(season: int) -> pd.DataFrame:
    """
    Compute team-level stats for a given NFL season from play-by-play data.
    Returns a DataFrame indexed by team abbreviation.
    """
    print(f"  Downloading {season} play-by-play data (this takes ~30-60 seconds)...")
    try:
        pbp = nfl.import_pbp_data([season])
    except Exception as e:
        print(f"  ERROR downloading PBP: {e}")
        return pd.DataFrame()

    if pbp is None or pbp.empty or "season_type" not in pbp.columns:
        print(f"  No usable PBP data for {season} (season may not have started).")
        return pd.DataFrame()

    # Regular season only
    pbp = pbp[pbp["season_type"] == "REG"].copy()
    pbp["posteam"] = pbp["posteam"].apply(_norm)
    pbp["defteam"] = pbp["defteam"].apply(_norm)

    teams = sorted(pbp["posteam"].dropna().unique())
    print(f"  {len(pbp):,} regular season plays | {len(teams)} teams")

    stats = pd.DataFrame(index=teams)
    stats.index.name = "team"

    # ── EPA ──────────────────────────────────────────────────────────────
    scrimmage = pbp[pbp["play_type"].isin(["pass", "run", "qb_kneel", "qb_spike"])].copy()
    scrimmage = scrimmage.dropna(subset=["epa"])

    off_epa = scrimmage.groupby("posteam")["epa"].mean().rename("off_epa_per_play")
    def_epa = scrimmage.groupby("defteam")["epa"].mean().rename("def_epa_per_play")
    stats = stats.join(off_epa).join(def_epa)

    # ── Points per game ───────────────────────────────────────────────────
    try:
        sched = nfl.import_schedules([season])
        sched = sched[sched["game_type"] == "REG"].copy()
        sched["home_team"] = sched["home_team"].apply(_norm)
        sched["away_team"] = sched["away_team"].apply(_norm)
        sched = sched.dropna(subset=["home_score", "away_score"])

        home_pts_for  = sched.groupby("home_team")["home_score"].mean()
        away_pts_for  = sched.groupby("away_team")["away_score"].mean()
        home_pts_ag   = sched.groupby("home_team")["away_score"].mean()
        away_pts_ag   = sched.groupby("away_team")["home_score"].mean()

        pts_for_pg     = ((home_pts_for + away_pts_for) / 2).rename("pts_for_pg")
        pts_against_pg = ((home_pts_ag  + away_pts_ag)  / 2).rename("pts_against_pg")
        scoring_margin = (pts_for_pg - pts_against_pg).rename("scoring_margin")

        stats = stats.join(pts_for_pg).join(pts_against_pg).join(scoring_margin)
    except Exception as e:
        print(f"  Warning: could not compute points per game — {e}")

    # ── Turnover margin ───────────────────────────────────────────────────
    try:
        to_df = pbp.dropna(subset=["fumble_lost", "interception"]).copy()
        to_committed = (
            to_df.groupby("posteam")["interception"].sum() +
            to_df.groupby("posteam")["fumble_lost"].sum()
        )
        to_forced = (
            to_df.groupby("defteam")["interception"].sum() +
            to_df.groupby("defteam")["fumble_lost"].sum()
        )
        to_margin = (to_forced - to_committed).rename("to_margin")
        stats = stats.join(to_margin)
    except Exception as e:
        print(f"  Warning: could not compute turnover margin — {e}")

    # ── Third down conversion rates ───────────────────────────────────────
    try:
        td3 = pbp[pbp["down"] == 3].dropna(subset=["first_down"]).copy()
        third_conv     = td3.groupby("posteam")["first_down"].mean().rename("third_down_pct")
        third_allowed  = td3.groupby("defteam")["first_down"].mean().rename("third_down_allowed")
        stats = stats.join(third_conv).join(third_allowed)
    except Exception as e:
        print(f"  Warning: could not compute third down rates — {e}")

    # ── Red zone TD rate ─────────────────────────────────────────────────
    try:
        rz = pbp[(pbp["yardline_100"] <= 20) &
                 pbp["play_type"].isin(["pass", "run"])].dropna(subset=["touchdown"]).copy()
        rz_td = rz.groupby("posteam")["touchdown"].mean().rename("rz_td_pct")
        stats = stats.join(rz_td)
    except Exception as e:
        print(f"  Warning: could not compute red zone stats — {e}")

    # ── Passing efficiency ────────────────────────────────────────────────
    try:
        passes = pbp[(pbp["play_type"] == "pass") &
                     pbp["yards_gained"].notna() &
                     pbp["air_yards"].notna()].copy()
        pass_ypa = passes.groupby("posteam")["yards_gained"].mean().rename("pass_yards_per_att")
        stats = stats.join(pass_ypa)
    except Exception as e:
        print(f"  Warning: could not compute passing stats — {e}")

    # ── Rushing efficiency ────────────────────────────────────────────────
    try:
        runs = pbp[(pbp["play_type"] == "run") & pbp["yards_gained"].notna()].copy()
        rush_ypc = runs.groupby("posteam")["yards_gained"].mean().rename("rush_yards_per_carry")
        stats = stats.join(rush_ypc)
    except Exception as e:
        print(f"  Warning: could not compute rushing stats — {e}")

    # ── Sacks ─────────────────────────────────────────────────────────────
    try:
        sacks_df = pbp[pbp["sack"] == 1].copy()
        games_played = sched.groupby("home_team").size() + sched.groupby("away_team").size()
        games_played = (games_played / 2).rename("games")

        sacks_committed = sacks_df.groupby("posteam").size().rename("sacks_allowed")  # QB sacked
        sacks_forced    = sacks_df.groupby("defteam").size().rename("sacks_forced")

        sacks_allowed_pg = (sacks_committed / games_played).rename("sacks_allowed_pg")
        sacks_forced_pg  = (sacks_forced    / games_played).rename("sacks_forced_pg")
        stats = stats.join(sacks_allowed_pg).join(sacks_forced_pg)
    except Exception as e:
        print(f"  Warning: could not compute sack stats — {e}")

    stats["season"] = season
    return stats.round(4)


def main():
    from datetime import datetime
    current_year = datetime.now().year

    # Try current season first; fall back to previous year if too early
    seasons_to_try = [current_year, current_year - 1]

    for season in seasons_to_try:
        print(f"\nBuilding team stats for {season} season...")
        stats = build_team_stats(season)

        if stats.empty or len(stats) < 20:
            print(f"  Not enough data for {season} — trying {season - 1}...")
            continue

        stats.to_csv(OUTPUT_FILE)
        print(f"\n  Saved {len(stats)} teams to {OUTPUT_FILE}")
        print(f"\n  Key stats preview:")
        print(f"  {'Team':<6} {'OffEPA':>8} {'DefEPA':>8} {'ScoreMargin':>12} {'TO Margin':>10}")
        print(f"  {'-'*48}")

        show_cols = ["off_epa_per_play", "def_epa_per_play", "scoring_margin", "to_margin"]
        available = [c for c in show_cols if c in stats.columns]
        if available:
            display = stats[available].sort_values("off_epa_per_play", ascending=False)
            for team, row in display.head(10).iterrows():
                vals = "  ".join(f"{row.get(c, 0):>8.3f}" for c in available)
                print(f"  {team:<6} {vals}")
        return stats

    print("ERROR: Could not build team stats for any season.")
    return pd.DataFrame()


if __name__ == "__main__":
    main()
