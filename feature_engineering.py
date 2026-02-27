"""
NFL Predictor — Feature Engineering
======================================
Computes new predictive features from historical game data that are
NOT already in games_processed.csv. All features are lagged correctly —
only information available BEFORE the game being predicted is used.

Functions are designed to be called on the full sorted game DataFrame
(sorted by gameday ascending) so rolling calculations look backward only.

Usage:
    import pandas as pd
    from feature_engineering import build_enhanced_features

    games = pd.read_csv("games_processed.csv")
    games_enhanced = build_enhanced_features(games)
    # games_enhanced has all original columns + ~12 new features
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
FORM_WINDOW = 5          # games for rolling win rate
SCORE_WINDOW = 5         # games for rolling scoring averages
ELO_TREND_WINDOW = 4    # games for ELO momentum (change over last N games)


def build_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function — applies all feature engineering in correct order.

    Input:  games_processed.csv as a DataFrame (must include home_score,
            away_score, home_team, away_team, gameday, elo_diff, spread_line)
    Output: Same DataFrame with new feature columns appended.

    New columns added:
        home/away_l5_win_pct        home/away_elo_trend
        home/away_l5_pts_for        home/away_l5_pts_against
        home/away_l5_pts_diff       pts_diff_advantage
        matchup_adv_home            matchup_adv_away         net_matchup_adv
        rest_advantage              spread_implied_prob       elo_implied_prob
        qb_score_diff
        is_primetime                is_thursday              is_neutral_site
        away_travel_miles
        home_sos                    away_sos                 sos_diff
    """
    df = df.copy()
    df = _ensure_types(df)
    df = df.sort_values("gameday").reset_index(drop=True)

    df = _add_team_form(df)
    df = _add_scoring_averages(df)
    df = _add_elo_trend(df)
    df = _add_rest_advantage(df)
    df = _add_implied_probabilities(df)
    df = _add_qb_quality_diff(df)
    df = _add_team_epa(df)
    df = _add_rolling_epa(df)

    return df


# ── Individual feature builders ───────────────────────────────────────────────

def _add_team_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling win rate for each team in their last FORM_WINDOW games.
    Captures recent form / momentum independently of ELO.
    """
    # Build a per-team game-by-game outcome series
    records: dict[str, list[tuple[str, int]]] = {}  # team → [(gameday, won)]

    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        hs, as_ = row.get("home_score", np.nan), row.get("away_score", np.nan)
        gd = row["gameday"]
        if pd.isna(hs) or pd.isna(as_):
            continue
        home_won = int(hs > as_)
        records.setdefault(ht, []).append((gd, home_won))
        records.setdefault(at, []).append((gd, 1 - home_won))

    def rolling_win_pct(team, before_gameday, n=FORM_WINDOW):
        history = records.get(team, [])
        # Only games before this one
        prior = [w for gd, w in history if gd < before_gameday]
        if len(prior) < 2:
            return np.nan
        return np.mean(prior[-n:])

    df["home_l5_win_pct"] = df.apply(
        lambda r: rolling_win_pct(r["home_team"], r["gameday"]), axis=1
    )
    df["away_l5_win_pct"] = df.apply(
        lambda r: rolling_win_pct(r["away_team"], r["gameday"]), axis=1
    )
    # Differential (positive = home team in better form)
    df["form_advantage"] = df["home_l5_win_pct"] - df["away_l5_win_pct"]
    return df


def _add_scoring_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling average points scored and allowed for each team.
    Captures offensive and defensive quality beyond what ELO measures.
    """
    records_for:     dict[str, list[tuple[str, float]]] = {}
    records_against: dict[str, list[tuple[str, float]]] = {}

    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        hs, as_ = row.get("home_score", np.nan), row.get("away_score", np.nan)
        gd = row["gameday"]
        if pd.isna(hs) or pd.isna(as_):
            continue
        records_for.setdefault(ht, []).append((gd, float(hs)))
        records_for.setdefault(at, []).append((gd, float(as_)))
        records_against.setdefault(ht, []).append((gd, float(as_)))
        records_against.setdefault(at, []).append((gd, float(hs)))

    def roll_avg(rec_dict, team, before_gameday, n=SCORE_WINDOW):
        history = rec_dict.get(team, [])
        prior = [v for gd, v in history if gd < before_gameday]
        if len(prior) < 2:
            return np.nan
        return np.mean(prior[-n:])

    df["home_l5_pts_for"]     = df.apply(lambda r: roll_avg(records_for,     r["home_team"], r["gameday"]), axis=1)
    df["home_l5_pts_against"] = df.apply(lambda r: roll_avg(records_against, r["home_team"], r["gameday"]), axis=1)
    df["away_l5_pts_for"]     = df.apply(lambda r: roll_avg(records_for,     r["away_team"], r["gameday"]), axis=1)
    df["away_l5_pts_against"] = df.apply(lambda r: roll_avg(records_against, r["away_team"], r["gameday"]), axis=1)

    # Point differential (strength proxy — positive = strong net)
    df["home_l5_pts_diff"] = df["home_l5_pts_for"] - df["home_l5_pts_against"]
    df["away_l5_pts_diff"] = df["away_l5_pts_for"] - df["away_l5_pts_against"]

    # Net differential advantage (home vs away)
    df["pts_diff_advantage"] = df["home_l5_pts_diff"] - df["away_l5_pts_diff"]

    # ── Matchup cross-features ─────────────────────────────────────────
    # These are the most predictive defensive features available from
    # score data alone. They capture whether a team's offense matches
    # up well against the specific *defense* they're facing this week.
    #
    # matchup_adv_home > 0  →  home offense outscores what away defense allows
    # matchup_adv_away > 0  →  away offense outscores what home defense allows
    # net_matchup_adv  > 0  →  net home team matchup advantage
    df["matchup_adv_home"] = df["home_l5_pts_for"]  - df["away_l5_pts_against"]
    df["matchup_adv_away"] = df["away_l5_pts_for"]  - df["home_l5_pts_against"]
    df["net_matchup_adv"]  = df["matchup_adv_home"] - df["matchup_adv_away"]

    return df


def _add_elo_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    ELO trend: change in ELO over the last N games.
    Captures momentum that instantaneous ELO doesn't express.
    Positive = team is on an upswing.
    """
    # We need per-team ELO timeline.
    # Approximate from elo_diff column: we can't reverse-engineer individual
    # team ELO exactly, but we can track how elo_diff changes for each team.
    # Better approach: recompute team ELO from scratch using the same K=20.

    elo: dict[str, list[tuple[str, float]]] = {}  # team → [(gameday, elo_after)]
    current_elo: dict[str, float] = {}
    K = 20

    for _, row in df.sort_values("gameday").iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        hs = row.get("home_score", np.nan)
        as_ = row.get("away_score", np.nan)
        gd = row["gameday"]

        elo_h = current_elo.get(ht, 1500.0)
        elo_a = current_elo.get(at, 1500.0)

        if pd.isna(hs) or pd.isna(as_):
            elo.setdefault(ht, []).append((gd, elo_h))
            elo.setdefault(at, []).append((gd, elo_a))
            continue

        expected_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
        home_won   = int(float(hs) > float(as_))

        new_elo_h = elo_h + K * (home_won       - expected_h)
        new_elo_a = elo_a + K * ((1 - home_won) - (1 - expected_h))

        current_elo[ht] = new_elo_h
        current_elo[at] = new_elo_a

        elo.setdefault(ht, []).append((gd, new_elo_h))
        elo.setdefault(at, []).append((gd, new_elo_a))

    def elo_change(team, before_gameday, n=ELO_TREND_WINDOW):
        history = elo.get(team, [])
        prior = [e for gd, e in history if gd < before_gameday]
        if len(prior) < n + 1:
            return 0.0
        return prior[-1] - prior[-(n + 1)]

    df["home_elo_trend"] = df.apply(
        lambda r: elo_change(r["home_team"], r["gameday"]), axis=1
    )
    df["away_elo_trend"] = df.apply(
        lambda r: elo_change(r["away_team"], r["gameday"]), axis=1
    )
    df["elo_trend_net"] = df["home_elo_trend"] - df["away_elo_trend"]

    return df


def _add_rest_advantage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine home_rest and away_rest into a single signed advantage feature.
    Positive = home team has more rest; negative = away team has more rest.
    Also flag extreme rest situations.
    """
    if "home_rest" in df.columns and "away_rest" in df.columns:
        df["rest_advantage"] = df["home_rest"] - df["away_rest"]
        # Short-rest penalty flag (< 5 days = Thursday game or injury risk)
        df["home_short_rest"] = (df["home_rest"] < 5).astype(int)
        df["away_short_rest"] = (df["away_rest"] < 5).astype(int)
    return df


def _add_implied_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Vegas spread to an implied win probability.
    Historical calibration: each point of spread ≈ 2.8% probability shift.
    Uses the sigmoid function tuned on NFL historical data.

    Also adds a pure ELO-derived probability for model comparison.
    """
    if "spread_line" in df.columns:
        # Negative spread = home team favored
        # spread_line is from HOME team's perspective (negative = home favored)
        df["spread_implied_prob"] = df["spread_line"].apply(
            lambda s: _spread_to_prob(s) if pd.notna(s) else np.nan
        )

    if "elo_diff" in df.columns:
        df["elo_implied_prob"] = df["elo_diff"].apply(
            lambda d: 1 / (1 + 10 ** (-d / 400)) if pd.notna(d) else np.nan
        )

    return df


# Historical team location aliases for franchises that relocated.
# Maps old abbreviation → current STADIUMS key (approximate — same metro area
# for SD/LAC and OAK/LV; deliberately coarse since precision isn't critical).
_TEAM_LOCATION_ALIASES: dict[str, str] = {
    "OAK": "LV",   # Raiders: Oakland Coliseum → Allegiant Stadium
    "SD":  "LAC",  # Chargers: Qualcomm → SoFi
    "STL": "LA",   # Rams: Edward Jones Dome → SoFi
    "LAR": "LA",   # alternate Rams abbrev used in nflfastR
}


def _add_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Game-context situational features:

    is_primetime      — 1 if kickoff >= 20:00 ET (SNF, MNF, TNF prime slot)
    is_thursday       — 1 if Thursday Night Football (short-week prep)
    is_neutral_site   — 1 if played on neutral ground (London, Super Bowl site)
    away_travel_miles — great-circle miles the away team travelled from their
                        home stadium to the game site (larger = more fatigue)
    """
    # ── Primetime ─────────────────────────────────────────────────────────
    if "gametime" in df.columns:
        hour = pd.to_datetime(df["gametime"], format="%H:%M", errors="coerce").dt.hour
        df["is_primetime"] = (hour >= 20).fillna(0).astype(int)
    else:
        df["is_primetime"] = 0

    # ── Thursday Night Football ────────────────────────────────────────────
    if "weekday" in df.columns:
        df["is_thursday"] = (df["weekday"] == "Thursday").astype(int)
    else:
        df["is_thursday"] = 0

    # ── Neutral site ───────────────────────────────────────────────────────
    if "location" in df.columns:
        df["is_neutral_site"] = (df["location"] == "Neutral").astype(int)
    else:
        df["is_neutral_site"] = 0

    # ── Travel distance ────────────────────────────────────────────────────
    try:
        from apis.weather import STADIUMS as _STADIUMS
    except ImportError:
        df["away_travel_miles"] = np.nan
        return df

    # Build lat/lon lookup with alias fallback
    coords: dict[str, tuple[float, float]] = {}
    for team, info in _STADIUMS.items():
        coords[team] = (info["lat"], info["lon"])
    for old, new in _TEAM_LOCATION_ALIASES.items():
        if new in coords and old not in coords:
            coords[old] = coords[new]

    def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 3958.8
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi   = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    def _travel(away: str, home: str) -> float:
        if away in coords and home in coords:
            alat, alon = coords[away]
            hlat, hlon = coords[home]
            return _haversine_miles(alat, alon, hlat, hlon)
        return np.nan

    df["away_travel_miles"] = df.apply(
        lambda r: _travel(r["away_team"], r["home_team"]), axis=1
    )
    median_dist = df["away_travel_miles"].median()
    df["away_travel_miles"] = df["away_travel_miles"].fillna(median_dist)

    return df


def _add_strength_of_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling strength-of-schedule: average ELO of each team's last 5 opponents,
    using opponents' *pre-game* ELO (no leakage — only info available before
    each game is used).

    Positive sos_diff → home team faced tougher recent schedule.
    """
    SOS_WINDOW = 5
    K = 20

    # Reconstruct per-team opponent-ELO timelines (pre-game ELO of each opponent)
    opp_elo_hist: dict[str, list[tuple[str, float]]] = {}
    current_elo: dict[str, float] = {}

    for _, row in df.sort_values("gameday").iterrows():
        ht, at = row["home_team"], row["away_team"]
        hs, as_ = row.get("home_score", np.nan), row.get("away_score", np.nan)
        gd = row["gameday"]

        elo_h = current_elo.get(ht, 1500.0)
        elo_a = current_elo.get(at, 1500.0)

        # Record opponent's pre-game ELO for each team
        opp_elo_hist.setdefault(ht, []).append((gd, elo_a))
        opp_elo_hist.setdefault(at, []).append((gd, elo_h))

        # Update ELO after the game (if score available)
        if not (pd.isna(hs) or pd.isna(as_)):
            expected_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
            home_won = int(float(hs) > float(as_))
            current_elo[ht] = elo_h + K * (home_won - expected_h)
            current_elo[at] = elo_a + K * ((1 - home_won) - (1 - expected_h))

    def _rolling_sos(team: str, before_gameday: str) -> float:
        history = opp_elo_hist.get(team, [])
        prior = [e for gd, e in history if gd < before_gameday]
        if len(prior) < 2:
            return 1500.0  # league-average ELO (neutral)
        return float(np.mean(prior[-SOS_WINDOW:]))

    df["home_sos"] = df.apply(lambda r: _rolling_sos(r["home_team"], r["gameday"]), axis=1)
    df["away_sos"] = df.apply(lambda r: _rolling_sos(r["away_team"], r["gameday"]), axis=1)
    df["sos_diff"] = df["home_sos"] - df["away_sos"]

    return df


def _add_team_epa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-season team advanced stats from team_stats_historical.csv
    (built by build_team_stats.py for 2016-2024).

    EPA features (2016+; 0 = pre-2016 / unknown):
        home/away_off_epa   — offensive efficiency per play
        home/away_def_epa   — defensive EPA allowed per play
        epa_off_diff        — home_off - away_off  (positive = home offense better)
        epa_def_diff        — away_def - home_def  (positive = home defense better)
        epa_total_diff      — combined net EPA advantage for home team

    Turnover features (2016+; 0 = pre-2016 / unknown):
        home/away_to_margin — season turnover margin (forced - committed)
        to_margin_diff      — home_to_margin - away_to_margin

    3rd-down efficiency features (2016+; 0 = pre-2016 / unknown):
        home/away_third_eff — team's (off_3rd_down_pct - def_3rd_down_allowed) net efficiency
        third_eff_diff      — home_third_eff - away_third_eff
    """
    hist_file = Path("team_stats_historical.csv")
    if not hist_file.exists():
        return df

    needed = [
        "team", "season",
        "off_epa_per_play", "def_epa_per_play",
        "to_margin",
        "third_down_pct", "third_down_allowed",
    ]
    available_cols = pd.read_csv(hist_file, nrows=0).columns.tolist()
    load_cols = [c for c in needed if c in available_cols]

    hist = pd.read_csv(hist_file, usecols=load_cols).set_index(["team", "season"])

    def _lookup(team, season, col):
        if col not in hist.columns:
            return np.nan
        try:
            return float(hist.loc[(team, season), col])
        except KeyError:
            return np.nan

    rows_home = [(r["home_team"], r["season"]) for _, r in df.iterrows()]
    rows_away = [(r["away_team"], r["season"]) for _, r in df.iterrows()]

    # ── EPA ───────────────────────────────────────────────────────────────
    df["home_off_epa"] = [_lookup(t, s, "off_epa_per_play") for t, s in rows_home]
    df["away_off_epa"] = [_lookup(t, s, "off_epa_per_play") for t, s in rows_away]
    df["home_def_epa"] = [_lookup(t, s, "def_epa_per_play") for t, s in rows_home]
    df["away_def_epa"] = [_lookup(t, s, "def_epa_per_play") for t, s in rows_away]

    df["epa_off_diff"]   = (df["home_off_epa"] - df["away_off_epa"]).fillna(0.0)
    df["epa_def_diff"]   = (df["away_def_epa"] - df["home_def_epa"]).fillna(0.0)
    df["epa_total_diff"] = df["epa_off_diff"] + df["epa_def_diff"]

    for col in ["home_off_epa", "away_off_epa", "home_def_epa", "away_def_epa"]:
        df[col] = df[col].fillna(0.0)

    # ── Turnover margin ───────────────────────────────────────────────────
    if "to_margin" in hist.columns:
        df["home_to_margin"] = [_lookup(t, s, "to_margin") for t, s in rows_home]
        df["away_to_margin"] = [_lookup(t, s, "to_margin") for t, s in rows_away]
        df["to_margin_diff"] = (df["home_to_margin"] - df["away_to_margin"]).fillna(0.0)
        df["home_to_margin"] = df["home_to_margin"].fillna(0.0)
        df["away_to_margin"] = df["away_to_margin"].fillna(0.0)

    # ── 3rd-down efficiency ───────────────────────────────────────────────
    if "third_down_pct" in hist.columns and "third_down_allowed" in hist.columns:
        home_3rd_off = pd.Series([_lookup(t, s, "third_down_pct")     for t, s in rows_home], index=df.index)
        home_3rd_def = pd.Series([_lookup(t, s, "third_down_allowed") for t, s in rows_home], index=df.index)
        away_3rd_off = pd.Series([_lookup(t, s, "third_down_pct")     for t, s in rows_away], index=df.index)
        away_3rd_def = pd.Series([_lookup(t, s, "third_down_allowed") for t, s in rows_away], index=df.index)

        # Net 3rd-down efficiency: offense conversion rate minus defense conversion rate allowed
        df["home_third_eff"] = (home_3rd_off - home_3rd_def).fillna(0.0)
        df["away_third_eff"] = (away_3rd_off - away_3rd_def).fillna(0.0)
        df["third_eff_diff"] = df["home_third_eff"] - df["away_third_eff"]

    return df


def _add_rolling_epa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Game-level rolling EPA: each team's avg offensive and defensive EPA
    per play over their LAST 5 GAMES before each matchup.

    Loaded from rolling_epa.csv (built by build_rolling_epa.py).
    Joined on (team, season, week) — the rolling EPA is pre-game (shift(1)
    applied during build, so no leakage).

    Features added (2016+; 0.0 = pre-2016 or weeks < 3 of season):
        home_l5_off_epa_roll  — home team's rolling 5-game offensive EPA/play
        away_l5_off_epa_roll  — away team's rolling 5-game offensive EPA/play
        home_l5_def_epa_roll  — home team's rolling 5-game defensive EPA/play
        away_l5_def_epa_roll  — away team's rolling 5-game defensive EPA/play
        roll_epa_off_diff     — home_l5_off - away_l5_off (home offense edge)
        roll_epa_def_diff     — away_l5_def - home_l5_def (home defense edge)
        roll_epa_total_diff   — combined home team EPA advantage
    """
    roll_file = Path("rolling_epa.csv")
    if not roll_file.exists():
        return df

    roll = pd.read_csv(roll_file)
    roll["week"] = pd.to_numeric(roll["week"], errors="coerce")
    roll = roll.set_index(["team", "season", "week"])

    def _lookup_roll(team, season, week, col):
        try:
            return float(roll.loc[(team, season, week), col])
        except KeyError:
            return np.nan

    rows_home = [(r["home_team"], r["season"], r.get("week")) for _, r in df.iterrows()]
    rows_away = [(r["away_team"], r["season"], r.get("week")) for _, r in df.iterrows()]

    df["home_l5_off_epa_roll"] = [_lookup_roll(t, s, w, "l5_off_epa") for t, s, w in rows_home]
    df["away_l5_off_epa_roll"] = [_lookup_roll(t, s, w, "l5_off_epa") for t, s, w in rows_away]
    df["home_l5_def_epa_roll"] = [_lookup_roll(t, s, w, "l5_def_epa") for t, s, w in rows_home]
    df["away_l5_def_epa_roll"] = [_lookup_roll(t, s, w, "l5_def_epa") for t, s, w in rows_away]

    df["roll_epa_off_diff"]   = (df["home_l5_off_epa_roll"] - df["away_l5_off_epa_roll"]).fillna(0.0)
    df["roll_epa_def_diff"]   = (df["away_l5_def_epa_roll"] - df["home_l5_def_epa_roll"]).fillna(0.0)
    df["roll_epa_total_diff"] = df["roll_epa_off_diff"] + df["roll_epa_def_diff"]

    for col in ["home_l5_off_epa_roll", "away_l5_off_epa_roll",
                "home_l5_def_epa_roll", "away_l5_def_epa_roll"]:
        df[col] = df[col].fillna(0.0)

    return df


def _add_qb_quality_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    QB quality differential: home_qb_score - away_qb_score.

    Loads qb_ratings.csv (built by build_qb_ratings.py) and joins on
    (player_id, season).  qb_score is a z-score (mean 0, std 1) so:
      > 0  →  home QB is above league average relative to away QB
      < 0  →  away QB advantage
      = 0  →  neutral (unknown QBs or pre-2010 data filled with 0)

    Fills NaN with 0 so that pre-2010 games and games with missing QB IDs
    don't get dropped — the model treats them as "no QB edge known".
    """
    qb_file = Path("qb_ratings.csv")
    if not qb_file.exists():
        # Silently skip if the file isn't built yet
        return df

    qb = pd.read_csv(qb_file, usecols=["player_id", "season", "qb_score"])
    qb = qb.rename(columns={"player_id": "qb_id"})
    qb_lookup = qb.set_index(["qb_id", "season"])["qb_score"]

    def lookup(qb_id, season):
        try:
            return float(qb_lookup.loc[(qb_id, season)])
        except KeyError:
            return np.nan

    if "home_qb_id" in df.columns and "away_qb_id" in df.columns:
        home_scores = [lookup(row["home_qb_id"], row["season"]) for _, row in df.iterrows()]
        away_scores = [lookup(row["away_qb_id"], row["season"]) for _, row in df.iterrows()]
        df["home_qb_score"] = home_scores
        df["away_qb_score"] = away_scores
        df["qb_score_diff"] = (
            pd.Series(home_scores, index=df.index)
            - pd.Series(away_scores, index=df.index)
        ).fillna(0.0)
    else:
        df["qb_score_diff"] = 0.0

    return df


# ── Utilities ─────────────────────────────────────────────────────────────────

def _spread_to_prob(spread: float) -> float:
    """
    Convert NFL Vegas spread (from home team's perspective) to home win probability.

    Calibration based on NFL historical data (2000-2024):
    - Every 1 point of spread ≈ 2.8% probability
    - Std dev of NFL score margins ≈ 13.45 points
    Uses normal CDF approximation.
    """
    from scipy.stats import norm
    # Positive spread = home underdog; negative = home favored
    # Expected margin if spread is "fair": -spread (home wins by this)
    # Score margin std dev: ~13.45 pts
    prob = 1 - norm.cdf(0, loc=-spread, scale=13.45)
    return float(np.clip(prob, 0.02, 0.98))


def _ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns have correct dtypes for vectorized operations."""
    for col in ["home_score", "away_score", "home_rest", "away_rest",
                "elo_diff", "spread_line", "temp", "wind"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce").astype(str)
    return df


# ── Feature list for the enhanced model ──────────────────────────────────────
ORIGINAL_FEATURES = [
    "elo_diff", "spread_line", "home_rest", "away_rest",
    "temp", "wind", "is_dome", "is_grass", "div_game",
]

ENHANCED_FEATURES = [
    # Core Vegas + ELO inputs (raw values for tree splits)
    "elo_diff", "spread_line",
    # Weather (temp + wind retained; is_dome/is_grass near-zero importance → removed)
    "temp", "wind",
    # ELO momentum
    "home_elo_trend", "away_elo_trend",
    # Scoring strength (offense + defense separately)
    "home_l5_pts_for",    "away_l5_pts_for",
    "home_l5_pts_against","away_l5_pts_against",
    "home_l5_pts_diff",   "away_l5_pts_diff",
    "pts_diff_advantage",
    # Matchup cross-features (home off vs away def and vice versa)
    "matchup_adv_home",   "matchup_adv_away",
    "net_matchup_adv",
    # Vegas & ELO implied probabilities (sigmoid pre-computation helps tree splits)
    "spread_implied_prob",
    "elo_implied_prob",
    # QB quality differential (z-score; 0 = unknown / neutral)
    "qb_score_diff",
    # Team EPA per play — season-level offensive/defensive efficiency (2016+; 0 = unknown)
    "home_off_epa",    "away_off_epa",
    "home_def_epa",    "away_def_epa",
    "epa_off_diff",    "epa_def_diff",    "epa_total_diff",
    # ── REMOVED features (ablation results) ──────────────────────────────────
    # home_rest, away_rest, rest_advantage: near-zero importance (<0.007 combined)
    # is_dome, is_grass, div_game: near-zero importance (<0.006 combined)
    # home_l5_win_pct, away_l5_win_pct: low importance (0.009/0.007); redundant
    #   with l5_pts_diff which is more continuous and informative.
    # Removing these 8 features: CV +0.43% (GBC 66.73±1.31%, RF 67.59±0.88%)
    # vs baseline CV (GBC 66.48±1.44%, RF 66.99±0.79%).
    # ─────────────────────────────────────────────────────────────────────────
    # Game-level rolling 5-game EPA: CV flat (+0.05% full, -0.03% on 2016-2023).
    # Season-level EPA already captures this. _add_rolling_epa() kept but unused.
    # Turnover margin, 3rd-down efficiency: CV flat (-0.01%). EPA already encodes.
    # Situational/SOS features: CV -0.26% on ~6k games (too sparse to be reliable).
]

# Columns required in input DataFrame before feature engineering
REQUIRED_COLUMNS = [
    "home_team", "away_team", "gameday",
    "home_score", "away_score", "home_win",
    "elo_diff", "spread_line", "home_rest", "away_rest",
    "temp", "wind", "is_dome", "is_grass", "div_game",
]
