"""
NFL Predictor — Data Pipeline
================================
Single entry point that orchestrates all five API modules into one
enriched GameContext used by final_app.py at prediction time.

  pipeline = DataPipeline()
  ctx = pipeline.get_game_context("KC", "BUF", "2025-01-15T18:15", elo_diff=42,
                                   home_rest=7, away_rest=7, div_game=0)
  # ctx is a plain dict with weather, Vegas lines, injury scores,
  # live depth charts, and a ready-to-use feature vector.

Graceful degradation: every fetch is wrapped so the app works even
when an API is unavailable (key missing, quota exceeded, network error).
"""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from dotenv import load_dotenv

load_dotenv()  # read .env if present

from apis.tank01  import Tank01Client
from apis.espn    import ESPNClient
from apis.weather import WeatherClient
from apis.odds    import OddsClient

logger = logging.getLogger(__name__)

# ── Load team stats (EPA, turnover margin, etc.) ─────────────────────────────
_TEAM_STATS_FILE = Path(__file__).parent / "team_stats_current.csv"

def _load_team_stats() -> pd.DataFrame:
    """
    Load team_stats_current.csv produced by build_team_stats.py.
    Returns an empty DataFrame if the file doesn't exist yet.
    """
    if _TEAM_STATS_FILE.exists():
        try:
            return pd.read_csv(_TEAM_STATS_FILE, index_col="team")
        except Exception:
            pass
    return pd.DataFrame()

_TEAM_STATS: pd.DataFrame = _load_team_stats()

# ── Load QB ratings ───────────────────────────────────────────────────────────
_QB_RATINGS_FILE = Path(__file__).parent / "qb_team_ratings.csv"

def _load_qb_ratings() -> pd.DataFrame:
    """Load qb_team_ratings.csv produced by build_qb_ratings.py."""
    if _QB_RATINGS_FILE.exists():
        try:
            return pd.read_csv(_QB_RATINGS_FILE, index_col="team")
        except Exception:
            pass
    return pd.DataFrame()

_QB_RATINGS: pd.DataFrame = _load_qb_ratings()

# ── Position injury weights (match lineup engine in final_app.py) ────────────
_POS_WEIGHT = {"QB": 0.40, "WR": 0.25, "RB": 0.20, "TE": 0.15}

# Injury severity (fraction of position weight that is lost)
_STATUS_SEVERITY = {
    "OUT":         1.00,
    "IR":          1.00,
    "PUP":         1.00,
    "DNR":         1.00,
    "DOUBTFUL":    0.75,
    "QUESTIONABLE":0.25,
    "LIMITED":     0.10,
    "ACTIVE":      0.00,
}


class DataPipeline:
    """
    Orchestrates Tank01, ESPN, Open-Meteo, and The Odds API into a
    single, cached context dict for each game prediction.

    Instantiate once (via @st.cache_resource) and reuse.
    """

    def __init__(self):
        rapidapi_key = os.getenv("RAPIDAPI_KEY", "")
        odds_key     = os.getenv("ODDS_API_KEY", "")

        self.tank01      = Tank01Client(api_key=rapidapi_key)
        self.espn        = ESPNClient()
        self.weather     = WeatherClient()
        self.odds        = OddsClient(api_key=odds_key)
        self.team_stats  = _TEAM_STATS   # EPA, TO margin, etc.
        self.qb_ratings  = _QB_RATINGS  # per-team QB quality scores

        self._has_tank01      = bool(rapidapi_key)
        self._has_odds        = bool(odds_key)
        self._has_team_stats  = not self.team_stats.empty
        self._has_qb_ratings  = not self.qb_ratings.empty

    # ── Main entry point ─────────────────────────────────────────────────
    def get_game_context(
        self,
        home_team:    str,
        away_team:    str,
        game_datetime:str,
        elo_diff:     float = 0.0,
        home_rest:    int   = 7,
        away_rest:    int   = 7,
        div_game:     int   = 0,
        surface:      str   = "grass",
    ) -> dict:
        """
        Fetch and assemble all pre-game context for a matchup.

        Returns a dict with these top-level keys:
            weather        — temp, wind, is_dome, precip_prob, stadium
            odds           — spread, total, moneyline, vegas_ml_prob, formatted
            injuries       — home_impact, away_impact, home_flags, away_flags
            depth_charts   — home: {QB:[], RB:[], ...}, away: {...}
            feature_vector — ready-to-use dict for model.predict_proba()
            api_status     — which APIs succeeded (for UI diagnostics)
        """
        home = home_team.upper()
        away = away_team.upper()

        ctx: dict = {
            "weather":       {},
            "odds":          {},
            "injuries":      {},
            "depth_charts":  {"home": {}, "away": {}},
            "feature_vector":{},
            "api_status":    {},
        }

        # 1. Weather — always available (Open-Meteo, no key)
        ctx["weather"]      = self._fetch_weather(home, game_datetime)
        ctx["api_status"]["weather"] = bool(ctx["weather"])

        # 2. Vegas odds — requires ODDS_API_KEY
        ctx["odds"]         = self._fetch_odds(home, away)
        ctx["api_status"]["odds"] = bool(ctx["odds"].get("spread"))

        # 3. Injury reports — Tank01 with ESPN fallback
        home_impact, home_flags = self._fetch_injury_impact(home)
        away_impact, away_flags = self._fetch_injury_impact(away)
        ctx["injuries"] = {
            "home_impact": home_impact,
            "away_impact": away_impact,
            "home_flags":  home_flags,
            "away_flags":  away_flags,
        }
        ctx["api_status"]["injuries"] = bool(home_flags or away_flags)

        # 4. Live depth charts — Tank01 with ESPN fallback
        ctx["depth_charts"]["home"] = self._fetch_depth_chart(home)
        ctx["depth_charts"]["away"] = self._fetch_depth_chart(away)
        ctx["api_status"]["depth_charts"] = (
            bool(ctx["depth_charts"]["home"]) or bool(ctx["depth_charts"]["away"])
        )

        # 5. Team stats (EPA, turnover margin) — from team_stats_current.csv
        home_ts = self._get_team_stats(home)
        away_ts = self._get_team_stats(away)
        ctx["team_stats"]    = {"home": home_ts, "away": away_ts}
        ctx["api_status"]["team_stats"] = self._has_team_stats

        # 6. Assemble feature vector
        wx     = ctx["weather"]
        odds   = ctx["odds"]
        inj    = ctx["injuries"]

        # Determine spread: prefer live Vegas line, fall back to elo-derived estimate
        spread = odds.get("spread_home")
        if spread is None:
            # Simple ELO-derived estimate: each 25 ELO pts ≈ 1 point
            spread = round(-elo_diff / 25.0, 1)

        is_grass = 1 if surface == "grass" else 0

        # EPA matchup features: home offense EPA vs away defense EPA allowed
        # net_epa_matchup > 0 means home offense outperforms away defense
        h_off_epa  = home_ts.get("off_epa_per_play")
        h_def_epa  = home_ts.get("def_epa_per_play")
        a_off_epa  = away_ts.get("off_epa_per_play")
        a_def_epa  = away_ts.get("def_epa_per_play")

        epa_matchup_home = (h_off_epa - a_def_epa) if (h_off_epa is not None and a_def_epa is not None) else None
        epa_matchup_away = (a_off_epa - h_def_epa) if (a_off_epa is not None and h_def_epa is not None) else None
        net_epa_matchup  = ((epa_matchup_home or 0) - (epa_matchup_away or 0)) if (epa_matchup_home is not None or epa_matchup_away is not None) else None

        ctx["feature_vector"] = {
            # Original 9 features (model-compatible)
            "elo_diff":    elo_diff,
            "spread_line": spread,
            "home_rest":   home_rest,
            "away_rest":   away_rest,
            "temp":        wx.get("temp",    65),
            "wind":        wx.get("wind",    5),
            "is_dome":     wx.get("is_dome", 0),
            "is_grass":    is_grass,
            "div_game":    div_game,
            # Extended features (for enhanced model)
            "precip_prob":        wx.get("precip_prob", 0),
            "vegas_ml_prob":      odds.get("vegas_ml_prob"),
            "total_line":         odds.get("total"),
            "rest_advantage":     home_rest - away_rest,
            "home_injury_impact": inj["home_impact"],
            "away_injury_impact": inj["away_impact"],
            "net_injury_adj":     inj["away_impact"] - inj["home_impact"],
            # EPA matchup features (live, from team_stats_current.csv)
            "home_off_epa":       h_off_epa,
            "away_off_epa":       a_off_epa,
            "home_def_epa":       h_def_epa,
            "away_def_epa":       a_def_epa,
            "epa_matchup_home":   epa_matchup_home,
            "epa_matchup_away":   epa_matchup_away,
            "net_epa_matchup":    net_epa_matchup,
            # Turnover margin
            "home_to_margin":     home_ts.get("to_margin"),
            "away_to_margin":     away_ts.get("to_margin"),
            "net_to_margin":      (
                (home_ts.get("to_margin") or 0) - (away_ts.get("to_margin") or 0)
                if home_ts.get("to_margin") is not None or away_ts.get("to_margin") is not None
                else None
            ),
            # Scoring margin
            "home_scoring_margin": home_ts.get("scoring_margin"),
            "away_scoring_margin": away_ts.get("scoring_margin"),
            # QB quality differential
            "home_qb_score":  self._get_qb_score(home),
            "away_qb_score":  self._get_qb_score(away),
            "qb_score_diff":  (
                (self._get_qb_score(home) or 0) - (self._get_qb_score(away) or 0)
                if self._get_qb_score(home) is not None or self._get_qb_score(away) is not None
                else None
            ),
        }

        return ctx

    # ── Weather ──────────────────────────────────────────────────────────
    def _fetch_weather(self, home_team: str, game_datetime: str) -> dict:
        try:
            return self.weather.get_game_weather(home_team, game_datetime)
        except Exception as e:
            logger.warning("Weather fetch failed: %s", e)
            return {"temp": 65, "wind": 5, "is_dome": 0, "precip_prob": 0,
                    "stadium": "Unknown"}

    # ── Odds ─────────────────────────────────────────────────────────────
    def _fetch_odds(self, home_team: str, away_team: str) -> dict:
        result: dict = {}
        if not self._has_odds:
            return result
        try:
            game = self.odds.get_game_odds(home_team, away_team)
            if not game:
                return result

            result["spread_home"]  = (game.get("spread") or {}).get("home")
            result["spread_away"]  = (game.get("spread") or {}).get("away")
            result["total"]        = (game.get("total")  or {}).get("line")
            result["ml_home"]      = (game.get("moneyline") or {}).get("home")
            result["ml_away"]      = (game.get("moneyline") or {}).get("away")
            result["formatted"]    = self.odds.get_formatted_lines(home_team, away_team)
            result["commence"]     = game.get("commence_time", "")

            impl = self.odds.get_implied_probability(home_team, away_team)
            if impl:
                result["vegas_ml_prob"]       = impl["home"]
                result["vegas_ml_prob_away"]  = impl["away"]

        except Exception as e:
            logger.warning("Odds fetch failed: %s", e)
        return result

    # ── Injuries ─────────────────────────────────────────────────────────
    def _fetch_injury_impact(self, team: str) -> tuple[float, dict[str, str]]:
        """
        Compute a 0-1 injury impact score and a name→status flag dict.

        Impact = sum of (position_weight × severity × depth_factor)
        for all injured skill-position players.
        """
        injuries = []

        # Try Tank01 first
        if self._has_tank01:
            try:
                all_inj = self.tank01.get_injuries()
                injuries = [i for i in all_inj if i.get("team") == team]
            except Exception as e:
                logger.debug("Tank01 injury fetch failed for %s: %s", team, e)

        # Fall back to ESPN
        if not injuries:
            try:
                raw = self.espn.get_team_injuries(team)
                injuries = [
                    {
                        "name":        i["name"],
                        "position":    i["position"],
                        "game_status": i["status"].upper(),
                        "depth":       1,
                    }
                    for i in raw
                ]
            except Exception as e:
                logger.debug("ESPN injury fetch failed for %s: %s", team, e)

        impact = 0.0
        flags: dict[str, str] = {}

        for inj in injuries:
            pos       = (inj.get("position") or "").upper()
            name      = inj.get("name", "Unknown")
            raw_status = (inj.get("game_status") or inj.get("status") or "ACTIVE").upper()

            # Normalize status strings
            status = _normalize_status(raw_status)
            severity = _STATUS_SEVERITY.get(status, 0.0)
            if severity == 0.0:
                continue

            weight       = _POS_WEIGHT.get(pos, 0.0)
            depth        = inj.get("depth", 2)
            depth_factor = 1.0 if depth == 1 else 0.35

            impact += weight * severity * depth_factor
            flags[name] = status

        return round(min(impact, 1.0), 4), flags

    # ── Depth charts ─────────────────────────────────────────────────────
    def _fetch_depth_chart(self, team: str) -> dict[str, list[dict]]:
        """Tank01 primary, ESPN fallback."""
        if self._has_tank01:
            try:
                chart = self.tank01.get_depth_chart(team)
                if chart:
                    return chart
            except Exception as e:
                logger.debug("Tank01 depth chart failed for %s: %s", team, e)

        try:
            return self.espn.get_depth_chart(team)
        except Exception as e:
            logger.debug("ESPN depth chart failed for %s: %s", team, e)
            return {}

    # ── Team stats (EPA, turnover margin, etc.) ──────────────────────────
    def _get_team_stats(self, team: str) -> dict:
        """
        Return current-season stats dict for a team from team_stats_current.csv.
        Keys: off_epa_per_play, def_epa_per_play, scoring_margin, to_margin, etc.
        Returns an empty dict if the file wasn't loaded or the team isn't found.
        """
        if self.team_stats.empty or team not in self.team_stats.index:
            return {}
        row = self.team_stats.loc[team]
        return {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}

    # ── QB quality ───────────────────────────────────────────────────────
    def _get_qb_score(self, team: str) -> float | None:
        """
        Return the QB quality z-score for a team's current starter.
        None if qb_team_ratings.csv was not loaded or team not found.
        """
        if self.qb_ratings.empty or team not in self.qb_ratings.index:
            return None
        val = self.qb_ratings.loc[team, "qb_score"]
        return None if pd.isna(val) else float(val)

    # ── Live lineup helpers ───────────────────────────────────────────────
    def get_live_starters(self, team: str) -> dict[str, dict]:
        """
        Return the depth-1 starter for QB, RB, WR, TE with injury flags.

        Returns:
            {
              "QB": {"name": "P.Mahomes", "player_id": "...", "is_out": False},
              "RB": {...}, "WR": {...}, "TE": {...}
            }
        """
        chart   = self._fetch_depth_chart(team)
        _, flags = self._fetch_injury_impact(team)
        starters = {}

        for pos in ("QB", "RB", "WR", "TE"):
            entries = chart.get(pos, [])
            if entries:
                starter = entries[0]
                name    = starter.get("name", "Unknown")
                starters[pos] = {
                    "name":      name,
                    "player_id": starter.get("player_id", ""),
                    "depth":     1,
                    "is_out":    flags.get(name, "ACTIVE") in ("OUT", "IR", "PUP", "DNR"),
                    "status":    flags.get(name, "ACTIVE"),
                }
        return starters

    def get_live_roster_list(self, team: str, pos: str) -> list[dict]:
        """
        Return ordered roster for a position with injury status.
        Used to populate the lineup dropdowns in the app.

        Returns: [{"name": ..., "depth": 1, "is_out": False, "status": "ACTIVE"}, ...]
        """
        chart    = self._fetch_depth_chart(team)
        _, flags = self._fetch_injury_impact(team)
        entries  = chart.get(pos.upper(), [])

        result = []
        for e in entries:
            name = e.get("name", "Unknown")
            result.append({
                "name":    name,
                "depth":   e.get("depth", 1),
                "is_out":  flags.get(name, "ACTIVE") in ("OUT", "IR", "PUP", "DNR"),
                "status":  flags.get(name, "ACTIVE"),
            })
        return result

    # ── Convenience ──────────────────────────────────────────────────────
    def get_api_status(self) -> dict[str, str]:
        """
        Quick check of which APIs are configured.
        Used in the app sidebar for diagnostics.
        """
        return {
            "Tank01 (live stats)": "✅ Key set" if self._has_tank01 else "⚠️ No key — add RAPIDAPI_KEY to .env",
            "ESPN (injuries)":     "✅ Live (no key needed)",
            "Open-Meteo (weather)":"✅ Live (no key needed)",
            "Odds API (Vegas)":    "✅ Key set" if self._has_odds else "⚠️ No key — add ODDS_API_KEY to .env",
            "PFR (historical)":    "✅ Scraping enabled",
            "Team Stats (EPA)":    "✅ Loaded" if self._has_team_stats else "⚠️ Run build_team_stats.py first",
            "QB Ratings":          "✅ Loaded" if self._has_qb_ratings  else "⚠️ Run build_qb_ratings.py first",
        }


# ── Helpers ──────────────────────────────────────────────────────────────────
def _normalize_status(raw: str) -> str:
    """Map various injury status strings to our standard keys."""
    raw = raw.upper().strip()
    if any(k in raw for k in ("OUT", "INACTIVE")):
        return "OUT"
    if any(k in raw for k in ("IR ", "INJURED RESERVE", "SEASON")):
        return "IR"
    if "DOUBTFUL" in raw:
        return "DOUBTFUL"
    if "QUESTIONABLE" in raw:
        return "QUESTIONABLE"
    if "LIMITED" in raw:
        return "LIMITED"
    if "PUP" in raw or "PHYSICALLY UNABLE" in raw:
        return "PUP"
    return "ACTIVE"
