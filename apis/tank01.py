"""
Tank01 NFL API Client (via RapidAPI)
=====================================
Primary live data feed. Provides:
  - Live / recent game scores
  - NFL schedule
  - Team rosters
  - Depth charts
  - Player stats (season + game-level)
  - Injury reports

Usage:
    from apis.tank01 import Tank01Client
    client = Tank01Client(api_key="YOUR_RAPIDAPI_KEY")

    injuries  = client.get_injuries()
    depth     = client.get_depth_chart("KC")
    scores    = client.get_scores(week=1, season=2024)

Rate limits (free tier): ~100 requests / day
All responses are cached — see cache TTLs below.

Set RAPIDAPI_KEY in your .env file or pass it to the constructor.
"""

import os
import time
import logging
from typing import Optional

import requests

from . import cache

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
BASE_URL = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
HOST     = "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"

# Cache TTLs (seconds)
TTL_SCORES     = 60 * 5        # 5 min  — live scores
TTL_INJURIES   = 60 * 60       # 1 hr   — injury updates
TTL_DEPTH      = 60 * 60 * 6   # 6 hrs  — depth charts
TTL_ROSTER     = 60 * 60 * 24  # 24 hrs — rosters
TTL_SCHEDULE   = 60 * 60 * 24  # 24 hrs — schedule
TTL_STATS      = 60 * 60 * 12  # 12 hrs — player stats

# Tank01 → our 3-letter abbreviation mapping
# Tank01 uses full abbreviations; some differ from nfl_data_py style
TANK01_TO_STD = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
    "JAC": "JAX", "LVR": "LV",  "RAM": "LA",  "SFO": "SF",
    "TBB": "TB",  "WAS": "WAS",
}


def _normalize_team(abbr: str) -> str:
    """Convert Tank01 abbreviation to our standard 3-letter code."""
    if not abbr:
        return abbr
    abbr = abbr.upper()
    return TANK01_TO_STD.get(abbr, abbr)


# ── Client ──────────────────────────────────────────────────────────────────
class Tank01Client:
    """
    Wrapper around the Tank01 NFL Live Statistics API on RapidAPI.

    All methods return plain Python dicts/lists — no pandas dependency here.
    The data_pipeline module normalises the output into DataFrames.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("RAPIDAPI_KEY", "")
        if not self.api_key:
            logger.warning(
                "RAPIDAPI_KEY not set. Tank01 calls will fail. "
                "Add it to your .env file."
            )
        self.session = requests.Session()
        self.session.headers.update({
            "x-rapidapi-host": HOST,
            "x-rapidapi-key":  self.api_key,
        })

    # ── Internal ─────────────────────────────────────────────────────────
    def _get(self, endpoint: str, params: dict, ttl: int) -> Optional[dict]:
        """Execute a GET request with caching. Returns parsed JSON or None."""
        cache_key = f"tank01:{endpoint}:{sorted(params.items())}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        if not self.api_key:
            logger.error("Cannot call Tank01 — RAPIDAPI_KEY missing.")
            return None

        url = f"{BASE_URL}/{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            cache.set(cache_key, data, ttl=ttl)
            return data
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                logger.warning("Tank01 rate limit hit (429). Try again later.")
            else:
                logger.error("Tank01 HTTP error on %s: %s", endpoint, e)
        except Exception as e:
            logger.error("Tank01 request failed on %s: %s", endpoint, e)
        return None

    # ── Public API methods ───────────────────────────────────────────────

    def get_scores(self, week: int, season: int) -> list[dict]:
        """
        Fetch game scores for a given week and season.

        Returns a list of game dicts, each containing:
            game_id, home_team, away_team, home_score, away_score,
            game_status, game_date, season, week
        """
        raw = self._get(
            "getNFLScores",
            {"gameWeek": str(week), "season": str(season)},
            ttl=TTL_SCORES,
        )
        if raw is None:
            return []

        games = raw.get("body", {})
        if isinstance(games, dict):
            games = list(games.values())

        results = []
        for g in games:
            try:
                results.append({
                    "game_id":     g.get("gameID", ""),
                    "home_team":   _normalize_team(g.get("homeTeam", "")),
                    "away_team":   _normalize_team(g.get("awayTeam", "")),
                    "home_score":  _safe_int(g.get("homePts")),
                    "away_score":  _safe_int(g.get("awayPts")),
                    "game_status": g.get("gameStatus", ""),
                    "game_date":   g.get("gameDate", ""),
                    "season":      season,
                    "week":        week,
                })
            except Exception:
                continue
        return results

    def get_schedule(self, season: int, week: Optional[int] = None) -> list[dict]:
        """
        Fetch the NFL schedule for a season (optionally filtered by week).

        Returns a list of game dicts with scheduling info.
        """
        params = {"season": str(season)}
        if week is not None:
            params["gameWeek"] = str(week)

        raw = self._get("getNFLGames", params, ttl=TTL_SCHEDULE)
        if raw is None:
            return []

        games = raw.get("body", {})
        if isinstance(games, dict):
            games = list(games.values())

        results = []
        for g in games:
            try:
                results.append({
                    "game_id":   g.get("gameID", ""),
                    "home_team": _normalize_team(g.get("homeTeam", "")),
                    "away_team": _normalize_team(g.get("awayTeam", "")),
                    "game_date": g.get("gameDate", ""),
                    "game_time": g.get("gameTime", ""),
                    "season":    season,
                    "week":      _safe_int(g.get("gameWeek")),
                    "stadium":   g.get("stadium", ""),
                })
            except Exception:
                continue
        return results

    def get_roster(self, team: str) -> list[dict]:
        """
        Fetch the full roster for a team.

        Args:
            team: 3-letter team abbreviation (e.g. "KC", "BUF")

        Returns a list of player dicts:
            player_id, name, position, jersey, status
        """
        raw = self._get(
            "getNFLTeamRoster",
            {"teamAbv": team, "getStats": "true"},
            ttl=TTL_ROSTER,
        )
        if raw is None:
            return []

        players_raw = raw.get("body", {}).get("roster", [])
        results = []
        for p in players_raw:
            try:
                results.append({
                    "player_id": p.get("playerID", ""),
                    "name":      p.get("longName", p.get("shortName", "")),
                    "position":  p.get("pos", ""),
                    "jersey":    p.get("jerseyNum", ""),
                    "status":    p.get("injury", {}).get("injStatus", "Active"),
                    "team":      team,
                })
            except Exception:
                continue
        return results

    def get_depth_chart(self, team: str) -> dict[str, list[dict]]:
        """
        Fetch the depth chart for a team.

        Args:
            team: 3-letter team abbreviation

        Returns a dict keyed by position, each containing an ordered list:
            { "QB": [{"name": ..., "player_id": ..., "depth": 1}, ...], ... }
        """
        raw = self._get(
            "getNFLTeamDepthChart",
            {"teamAbv": team},
            ttl=TTL_DEPTH,
        )
        if raw is None:
            return {}

        chart_raw = raw.get("body", {}).get("depthChart", {})
        chart = {}
        for pos, entries in chart_raw.items():
            pos_upper = pos.upper()
            chart[pos_upper] = []
            if isinstance(entries, list):
                for i, entry in enumerate(entries):
                    chart[pos_upper].append({
                        "name":      entry.get("longName", entry.get("playerName", "")),
                        "player_id": entry.get("playerID", ""),
                        "depth":     i + 1,
                        "team":      team,
                        "position":  pos_upper,
                    })
            elif isinstance(entries, dict):
                # Some responses use depth number as key
                for depth_key in sorted(entries.keys()):
                    entry = entries[depth_key]
                    chart[pos_upper].append({
                        "name":      entry.get("longName", entry.get("playerName", "")),
                        "player_id": entry.get("playerID", ""),
                        "depth":     _safe_int(depth_key) or 1,
                        "team":      team,
                        "position":  pos_upper,
                    })
        return chart

    def get_injuries(self, practice_status: bool = True) -> list[dict]:
        """
        Fetch the current NFL-wide injury report.

        Returns a list of injury dicts:
            player_id, name, team, position, injury_desc,
            practice_status, game_status, report_date
        """
        raw = self._get(
            "getNFLInjuries",
            {"fantasyPoints": "true"},
            ttl=TTL_INJURIES,
        )
        if raw is None:
            return []

        injuries_raw = raw.get("body", {})
        results = []

        # Response format varies: sometimes dict of teams → list of players
        if isinstance(injuries_raw, dict):
            for team_key, players in injuries_raw.items():
                if not isinstance(players, list):
                    continue
                for p in players:
                    results.append(_parse_injury(p, team_key))
        elif isinstance(injuries_raw, list):
            for p in injuries_raw:
                results.append(_parse_injury(p, p.get("team", "")))

        return [r for r in results if r]  # drop None entries

    def get_player_stats(
        self,
        player_id: str,
        season: int,
        game_id: Optional[str] = None,
    ) -> dict:
        """
        Fetch stats for a specific player.

        Args:
            player_id: Tank01 player ID
            season:    NFL season year
            game_id:   Optional — filter to a specific game

        Returns a dict with season or game-level stats.
        """
        params = {"playerID": player_id, "season": str(season)}
        if game_id:
            params["gameIDFilter"] = game_id

        raw = self._get("getNFLPlayerStats", params, ttl=TTL_STATS)
        if raw is None:
            return {}

        return raw.get("body", {})

    def get_box_score(self, game_id: str) -> dict:
        """
        Fetch detailed box score for a completed game.

        Args:
            game_id: Tank01 game ID (e.g. "20241201_KC@LV")

        Returns full box score dict including player stats.
        """
        raw = self._get(
            "getNFLBoxScore",
            {"gameID": game_id, "playByPlay": "false", "fantasyPoints": "true"},
            ttl=TTL_SCORES,
        )
        if raw is None:
            return {}
        return raw.get("body", {})

    def get_team_stats(self, team: str, season: int) -> dict:
        """
        Fetch team-level stats for a season.

        Returns a flat dict with passing, rushing, defense stats.
        """
        raw = self._get(
            "getNFLTeamStats",
            {"teamAbv": team, "season": str(season)},
            ttl=TTL_STATS,
        )
        if raw is None:
            return {}
        return raw.get("body", {})

    # ── Convenience helpers ──────────────────────────────────────────────

    def get_starters(self, team: str) -> dict[str, dict]:
        """
        Return the depth-1 starter for each key offensive position.

        Returns:
            { "QB": {"name": ..., "player_id": ...}, "RB": {...}, ... }
        """
        chart = self.get_depth_chart(team)
        starters = {}
        for pos in ("QB", "RB", "WR", "TE", "K"):
            entries = chart.get(pos, [])
            if entries:
                starters[pos] = entries[0]
        return starters

    def get_injury_status(self, team: str) -> dict[str, str]:
        """
        Return a name → game_status mapping for all injured players on a team.

        Useful for quickly checking whether a key player is OUT/DOUBTFUL.
        """
        all_injuries = self.get_injuries()
        return {
            inj["name"]: inj["game_status"]
            for inj in all_injuries
            if _normalize_team(inj.get("team", "")) == team and inj.get("game_status")
        }

    def is_player_out(self, name: str, team: str) -> bool:
        """Return True if a player is listed as OUT or IR."""
        statuses = self.get_injury_status(team)
        status = statuses.get(name, "Active").upper()
        return status in ("OUT", "IR", "PUP", "DNR")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _safe_int(val) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _parse_injury(p: dict, team_key: str) -> Optional[dict]:
    try:
        return {
            "player_id":       p.get("playerID", ""),
            "name":            p.get("longName", p.get("playerName", "")),
            "team":            _normalize_team(team_key),
            "position":        p.get("pos", ""),
            "injury_desc":     p.get("injury", {}).get("description", ""),
            "practice_status": p.get("injury", {}).get("practiceStatus", ""),
            "game_status":     p.get("injury", {}).get("injStatus", ""),
            "report_date":     p.get("injury", {}).get("injDate", ""),
        }
    except Exception:
        return None
