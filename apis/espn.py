"""
ESPN Unofficial API Client
===========================
No API key required. Uses ESPN's public (undocumented) JSON endpoints.

Provides:
  - Real-time injury reports (league-wide and per-team)
  - Live scoreboard
  - Team rosters with depth chart ordering
  - Team metadata (stadium, location, ESPN team ID)

Usage:
    from apis.espn import ESPNClient
    client = ESPNClient()

    injuries  = client.get_all_injuries()
    scoreboard = client.get_scoreboard()
    roster    = client.get_roster("KC")

Note: ESPN's unofficial API has no documented rate limits, but be
respectful — the cache TTLs here are intentionally conservative.
"""

import logging
from typing import Optional

import requests

from . import cache

logger = logging.getLogger(__name__)

# ── ESPN base URLs ───────────────────────────────────────────────────────────
_SITE_API  = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
_CORE_API  = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

# Cache TTLs
TTL_SCOREBOARD = 60 * 3        # 3 min — live scores
TTL_INJURIES   = 60 * 30       # 30 min
TTL_ROSTER     = 60 * 60 * 6   # 6 hrs
TTL_TEAMS      = 60 * 60 * 24  # 24 hrs

# ESPN team ID ↔ our abbreviation mapping
# ESPN IDs are fixed; abbreviations match our NFL_TEAMS list in final_app.py
ESPN_ID_TO_ABV = {
    "1":  "ATL", "2":  "BUF", "3":  "CHI", "4":  "CIN",
    "5":  "CLE", "6":  "DAL", "7":  "DEN", "8":  "DET",
    "9":  "GB",  "10": "TEN", "11": "IND", "12": "KC",
    "13": "LV",  "14": "LA",  "15": "MIA", "16": "MIN",
    "17": "NE",  "18": "NO",  "19": "NYG", "20": "NYJ",
    "21": "PHI", "22": "ARI", "23": "PIT", "24": "LAC",
    "25": "SF",  "26": "SEA", "27": "TB",  "28": "WAS",
    "29": "CAR", "30": "JAX", "33": "BAL", "34": "HOU",
}
ABV_TO_ESPN_ID = {v: k for k, v in ESPN_ID_TO_ABV.items()}


class ESPNClient:
    """
    Client for ESPN's public (no-auth) JSON API endpoints.

    All methods return plain Python dicts/lists. Heavy normalization happens
    in data_pipeline.py, not here.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        })

    # ── Internal ─────────────────────────────────────────────────────────
    def _get(self, url: str, ttl: int, params: Optional[dict] = None) -> Optional[dict]:
        cache_key = f"espn:{url}:{sorted((params or {}).items())}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            cache.set(cache_key, data, ttl=ttl)
            return data
        except requests.HTTPError as e:
            logger.error("ESPN HTTP error: %s — %s", url, e)
        except Exception as e:
            logger.error("ESPN request failed: %s — %s", url, e)
        return None

    # ── Scoreboard ───────────────────────────────────────────────────────
    def get_scoreboard(self, week: Optional[int] = None) -> list[dict]:
        """
        Fetch the current (or specified week's) NFL scoreboard.

        Returns a list of game dicts:
            game_id, home_team, away_team, home_score, away_score,
            status, game_date, venue
        """
        url    = f"{_SITE_API}/scoreboard"
        params = {}
        if week:
            params["week"] = week
        raw = self._get(url, ttl=TTL_SCOREBOARD, params=params)
        if raw is None:
            return []

        results = []
        for event in raw.get("events", []):
            try:
                comp  = event["competitions"][0]
                teams = {c["homeAway"]: c for c in comp["competitors"]}
                home  = teams.get("home", {})
                away  = teams.get("away", {})

                results.append({
                    "game_id":    event.get("id", ""),
                    "home_team":  _espn_abv(home.get("team", {})),
                    "away_team":  _espn_abv(away.get("team", {})),
                    "home_score": _safe_int(home.get("score")),
                    "away_score": _safe_int(away.get("score")),
                    "status":     event.get("status", {}).get("type", {}).get("name", ""),
                    "game_date":  event.get("date", ""),
                    "venue":      comp.get("venue", {}).get("fullName", ""),
                    "neutral":    comp.get("neutralSite", False),
                })
            except Exception:
                continue
        return results

    # ── Injuries ─────────────────────────────────────────────────────────
    def get_all_injuries(self) -> list[dict]:
        """
        Fetch the league-wide injury report.

        Returns a list of injury dicts:
            player_id, name, team, position, status, details
        """
        url = f"{_SITE_API}/injuries"
        raw = self._get(url, ttl=TTL_INJURIES)
        if raw is None:
            return []

        results = []
        for item in raw.get("injuries", []):
            try:
                athlete = item.get("athlete", {})
                team    = item.get("team", {})
                results.append({
                    "player_id": athlete.get("id", ""),
                    "name":      athlete.get("fullName", ""),
                    "team":      _espn_abv(team),
                    "position":  athlete.get("position", {}).get("abbreviation", ""),
                    "status":    item.get("status", ""),
                    "details":   item.get("details", {}).get("type", ""),
                    "side":      item.get("details", {}).get("location", ""),
                    "return_date": item.get("details", {}).get("returnDate", ""),
                })
            except Exception:
                continue
        return results

    def get_team_injuries(self, team: str) -> list[dict]:
        """
        Fetch injuries for a specific team by abbreviation (e.g. "KC").

        Returns the same structure as get_all_injuries() but filtered.
        """
        espn_id = ABV_TO_ESPN_ID.get(team.upper())
        if not espn_id:
            logger.warning("Unknown team abbreviation for ESPN: %s", team)
            return []

        url = f"{_SITE_API}/teams/{espn_id}/injuries"
        raw = self._get(url, ttl=TTL_INJURIES)
        if raw is None:
            return []

        results = []
        for item in raw.get("injuries", []):
            try:
                athlete = item.get("athlete", {})
                results.append({
                    "player_id": athlete.get("id", ""),
                    "name":      athlete.get("fullName", ""),
                    "team":      team,
                    "position":  athlete.get("position", {}).get("abbreviation", ""),
                    "status":    item.get("status", ""),
                    "details":   item.get("details", {}).get("type", ""),
                    "side":      item.get("details", {}).get("location", ""),
                    "return_date": item.get("details", {}).get("returnDate", ""),
                })
            except Exception:
                continue
        return results

    # ── Rosters ──────────────────────────────────────────────────────────
    def get_roster(self, team: str) -> list[dict]:
        """
        Fetch the full roster for a team, ordered by depth within each position.

        Args:
            team: 3-letter abbreviation (e.g. "KC")

        Returns a list of player dicts:
            player_id, name, team, position, jersey, depth, status
        """
        espn_id = ABV_TO_ESPN_ID.get(team.upper())
        if not espn_id:
            logger.warning("Unknown team abbreviation for ESPN: %s", team)
            return []

        url = f"{_SITE_API}/teams/{espn_id}/roster"
        raw = self._get(url, ttl=TTL_ROSTER)
        if raw is None:
            return []

        results = []
        for group in raw.get("athletes", []):
            pos_label = group.get("position", "")
            for i, athlete in enumerate(group.get("items", [])):
                try:
                    injuries = athlete.get("injuries", [])
                    status   = injuries[0].get("status", "Active") if injuries else "Active"
                    results.append({
                        "player_id": athlete.get("id", ""),
                        "name":      athlete.get("fullName", ""),
                        "team":      team,
                        "position":  athlete.get("position", {}).get("abbreviation", pos_label),
                        "jersey":    athlete.get("jersey", ""),
                        "depth":     i + 1,
                        "status":    status,
                        "weight":    athlete.get("weight"),
                        "height":    athlete.get("displayHeight", ""),
                        "age":       athlete.get("age"),
                        "experience": athlete.get("experience", {}).get("years", 0),
                    })
                except Exception:
                    continue
        return results

    def get_depth_chart(self, team: str) -> dict[str, list[dict]]:
        """
        Build a position-keyed depth chart from the ESPN roster endpoint.

        Returns the same structure as Tank01Client.get_depth_chart():
            { "QB": [{"name": ..., "depth": 1, ...}, ...], "RB": [...], ... }
        """
        roster = self.get_roster(team)
        chart: dict[str, list[dict]] = {}
        for player in roster:
            pos = player["position"]
            chart.setdefault(pos, [])
            chart[pos].append(player)
        # Sort each group by depth
        for pos in chart:
            chart[pos].sort(key=lambda p: p["depth"])
        return chart

    # ── Team metadata ────────────────────────────────────────────────────
    def get_all_teams(self) -> list[dict]:
        """
        Fetch basic metadata for all 32 NFL teams.

        Returns a list of dicts:
            espn_id, abbreviation, name, city, stadium, stadium_indoor
        """
        url = f"{_SITE_API}/teams"
        raw = self._get(url, ttl=TTL_TEAMS)
        if raw is None:
            return []

        results = []
        for entry in raw.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
            team = entry.get("team", {})
            try:
                venue = team.get("venue", {})
                results.append({
                    "espn_id":      team.get("id", ""),
                    "abbreviation": team.get("abbreviation", ""),
                    "name":         team.get("displayName", ""),
                    "city":         team.get("location", ""),
                    "stadium":      venue.get("fullName", ""),
                    "stadium_id":   venue.get("id", ""),
                    "indoor":       venue.get("indoor", False),
                })
            except Exception:
                continue
        return results

    # ── Convenience helpers ──────────────────────────────────────────────
    def get_key_injury_flags(self, team: str) -> dict[str, bool]:
        """
        Return a dict of {player_name: is_out} for all injured skill players
        on a team. Positions checked: QB, RB, WR, TE.

        Used in the game predictor to flag meaningful lineup changes.
        """
        key_positions = {"QB", "RB", "WR", "TE"}
        out_statuses  = {"Out", "Injured Reserve", "Physically Unable to Perform",
                         "Did Not Report"}

        injuries = self.get_team_injuries(team)
        flags = {}
        for inj in injuries:
            if inj["position"] in key_positions:
                flags[inj["name"]] = inj["status"] in out_statuses
        return flags


# ── Helpers ──────────────────────────────────────────────────────────────────
def _espn_abv(team_obj: dict) -> str:
    """Extract our standard 3-letter abbreviation from an ESPN team object."""
    abv = team_obj.get("abbreviation", "")
    # Some ESPN abbreviations differ from our standard
    _overrides = {"JAC": "JAX", "LV": "LV", "LAR": "LA"}
    return _overrides.get(abv.upper(), abv.upper())


def _safe_int(val) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None
