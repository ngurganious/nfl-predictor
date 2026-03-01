"""
NHL Official API Client
========================
No API key required. Uses the official NHL API at api-web.nhle.com/v1/.

Provides:
  - Season schedules (game IDs and metadata)
  - Game boxscores (scores, goalie stats, skater stats)
  - Current standings
  - Team rosters
  - Goalie season stats
  - Player game logs

Usage:
    from apis.nhl import NHLClient
    client = NHLClient()

    schedule = client.get_season_schedule("20242025")
    boxscore = client.get_boxscore(2024020001)
    standings = client.get_standings()

Note: The NHL API has no documented rate limits but be respectful.
Cache TTLs here are conservative. Add 0.1s sleep between bulk fetches.
"""

import logging
import time
from typing import Optional, List, Dict, Any

import requests

from . import cache

logger = logging.getLogger(__name__)

# ── Base URL ─────────────────────────────────────────────────────────────────
_BASE = "https://api-web.nhle.com/v1"

# Cache TTLs
TTL_SCHEDULE   = 60 * 60 * 3    # 3 hrs
TTL_BOXSCORE   = 60 * 60 * 24   # 24 hrs (historical games never change)
TTL_STANDINGS  = 60 * 60 * 1    # 1 hr
TTL_ROSTER     = 60 * 60 * 12   # 12 hrs
TTL_PLAYER     = 60 * 60 * 6    # 6 hrs
TTL_GOALIE     = 60 * 60 * 6    # 6 hrs

# ── Team abbreviation mapping ─────────────────────────────────────────────────
# Active NHL teams (2024-25 season)
NHL_TEAMS = sorted([
    'ANA', 'BOS', 'BUF', 'CGY', 'CAR', 'CHI', 'COL', 'CBJ',
    'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NSH',
    'NJD', 'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'SEA', 'SJS',
    'STL', 'TBL', 'TOR', 'UTA', 'VAN', 'VGK', 'WSH', 'WPG',
])

# Historical abbreviations → current abbreviations (for ELO continuity)
NHL_HISTORICAL_ALIASES: Dict[str, str] = {
    'PHX': 'ARI',  # Phoenix Coyotes → Arizona Coyotes (pre-2014)
    'ARI': 'UTA',  # Arizona Coyotes → Utah Hockey Club (2024+)
    # Note: we apply ARI→UTA only for seasons >= 2025
    'ATL': 'WPG',  # Atlanta Thrashers → Winnipeg Jets (2011)
    'MNS': None,   # Minnesota North Stars (dissolved 1993, before dataset)
    'QUE': None,   # Quebec Nordiques (dissolved 1995, before dataset)
    'HFD': None,   # Hartford Whalers (dissolved 1997, before dataset)
    'WST': 'WSH',  # Washington Capitals alternate
    'S.J': 'SJS',  # San Jose Sharks API variant
    'N.J': 'NJD',  # New Jersey Devils API variant
    'T.B': 'TBL',  # Tampa Bay Lightning API variant
    'L.A': 'LAK',  # Los Angeles Kings API variant
}

# Full team names → abbreviations (for Odds API matching)
NHL_NAME_TO_ABV: Dict[str, str] = {
    "Anaheim Ducks":         "ANA",
    "Arizona Coyotes":       "ARI",
    "Boston Bruins":         "BOS",
    "Buffalo Sabres":        "BUF",
    "Calgary Flames":        "CGY",
    "Carolina Hurricanes":   "CAR",
    "Chicago Blackhawks":    "CHI",
    "Colorado Avalanche":    "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars":          "DAL",
    "Detroit Red Wings":     "DET",
    "Edmonton Oilers":       "EDM",
    "Florida Panthers":      "FLA",
    "Los Angeles Kings":     "LAK",
    "Minnesota Wild":        "MIN",
    "Montreal Canadiens":    "MTL",
    "Montréal Canadiens":    "MTL",
    "Nashville Predators":   "NSH",
    "New Jersey Devils":     "NJD",
    "New York Islanders":    "NYI",
    "New York Rangers":      "NYR",
    "Ottawa Senators":       "OTT",
    "Philadelphia Flyers":   "PHI",
    "Pittsburgh Penguins":   "PIT",
    "San Jose Sharks":       "SJS",
    "Seattle Kraken":        "SEA",
    "St. Louis Blues":       "STL",
    "Tampa Bay Lightning":   "TBL",
    "Toronto Maple Leafs":   "TOR",
    "Utah Hockey Club":      "UTA",
    "Utah Mammoth":          "UTA",
    "Vancouver Canucks":     "VAN",
    "Vegas Golden Knights":  "VGK",
    "Washington Capitals":   "WSH",
    "Winnipeg Jets":         "WPG",
    # Historical
    "Atlanta Thrashers":     "WPG",
    "Phoenix Coyotes":       "ARI",
}

# Abbreviation → full team name
NHL_ABV_TO_NAME: Dict[str, str] = {v: k for k, v in NHL_NAME_TO_ABV.items()
                                    if k not in ("Atlanta Thrashers", "Phoenix Coyotes",
                                                 "Arizona Coyotes")}
NHL_ABV_TO_NAME['UTA'] = "Utah Hockey Club"
NHL_ABV_TO_NAME['ARI'] = "Arizona Coyotes"


def normalize_team(abbv: str, season_year: int = 2025) -> Optional[str]:
    """Normalize a raw API abbreviation to our canonical form."""
    if abbv is None:
        return None
    abbv = abbv.upper().strip()
    # Dot variants first
    if abbv in NHL_HISTORICAL_ALIASES:
        result = NHL_HISTORICAL_ALIASES[abbv]
        # ARI → UTA only for 2025+ seasons
        if abbv == 'ARI' and season_year < 2025:
            return 'ARI'
        return result
    return abbv if abbv in NHL_TEAMS or abbv == 'ARI' else abbv


class NHLClient:
    """Client for the official NHL API (api-web.nhle.com/v1/, free, no auth)."""

    def __init__(self, sleep_between: float = 0.1):
        self._sleep = sleep_between
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "SportsPredictor/1.0 (educational; contact via GitHub)",
            "Accept": "application/json",
        })

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get(self, path: str, ttl: int = TTL_BOXSCORE) -> Optional[Dict]:
        url = f"{_BASE}{path}"
        cached = cache.get(url)
        if cached is not None:
            return cached
        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            cache.set(url, data, ttl=ttl)
            time.sleep(self._sleep)
            return data
        except Exception as e:
            logger.warning(f"NHL API error {url}: {e}")
            return None

    # ── Schedule ──────────────────────────────────────────────────────────────

    def get_season_schedule(
        self,
        season_start_date: str,
        season_end_date: str,
        game_type_filter: str = "2",
        season_year: int = 2024,
    ) -> List[Dict]:
        """
        Walk week-by-week from season_start_date to season_end_date,
        collecting all games of the given type.

        Uses /schedule/{date} endpoint (returns one week per call,
        includes nextStartDate for easy walking).

        Args:
            season_start_date: "YYYY-MM-DD" of first week to fetch
            season_end_date:   "YYYY-MM-DD" — stop walking after this date
            game_type_filter:  "2" = regular season, "3" = playoffs, "" = all
            season_year:       integer year (e.g. 2024 for 2024-25 season)

        Returns list of game dicts.
        """
        from datetime import datetime, timedelta

        games = []
        current_date = season_start_date
        end_dt = datetime.strptime(season_end_date, "%Y-%m-%d")

        visited = set()
        while True:
            data = self._get(f"/schedule/{current_date}", ttl=TTL_SCHEDULE)
            if not data:
                break

            for week_block in data.get("gameWeek", []):
                for g in week_block.get("games", []):
                    gtype = str(g.get("gameType", ""))
                    if game_type_filter and gtype != game_type_filter:
                        continue
                    gid = g.get("id")
                    if gid in visited:
                        continue
                    visited.add(gid)
                    home = g.get("homeTeam", {})
                    away = g.get("awayTeam", {})
                    # Extract date: gameDate is often null; use startTimeUTC or week date
                    gameday = (
                        g.get("gameDate")
                        or (g.get("startTimeUTC") or "")[:10]
                        or week_block.get("date", "")
                    )
                    games.append({
                        "game_id":    gid,
                        "gameday":    gameday,
                        "home_team":  home.get("abbrev", ""),
                        "away_team":  away.get("abbrev", ""),
                        "home_score": home.get("score"),
                        "away_score": away.get("score"),
                        "game_type":  gtype,
                        "season":     season_year,
                        "status":     g.get("gameState", ""),
                    })

            next_date = data.get("nextStartDate")
            if not next_date:
                break
            try:
                next_dt = datetime.strptime(next_date, "%Y-%m-%d")
            except Exception:
                break
            if next_dt > end_dt:
                break
            current_date = next_date

        return games

    def get_current_week_schedule(self) -> List[Dict]:
        """Return today's + next 7 days of NHL games (all game types).

        Uses today's date as the anchor so we always see the current week's
        remaining games, then merges next week's block for a full 7-day view.
        Filters out games before today.
        """
        from datetime import datetime, timedelta
        # Use Eastern Time so the cutoff matches the user's timezone, not UTC
        # (UTC can be a day ahead of US time zones in the evening)
        try:
            import pytz
            et = pytz.timezone("America/New_York")
            today = datetime.now(et).date()
        except Exception:
            from datetime import timezone
            # UTC-5 fallback (EST)
            today = (datetime.now(timezone.utc) - timedelta(hours=5)).date()
        next_week = today + timedelta(days=7)
        today_str = today.strftime("%Y-%m-%d")
        next_str  = next_week.strftime("%Y-%m-%d")

        seen = set()
        games = []

        def _extract(data):
            if not data:
                return
            for week_block in data.get("gameWeek", []):
                for g in week_block.get("games", []):
                    gid = g.get("id")
                    if gid in seen:
                        continue
                    seen.add(gid)
                    home = g.get("homeTeam", {})
                    away = g.get("awayTeam", {})
                    # Prefer ET-based dates: gameDate first, then week_block date.
                    # startTimeUTC[:10] is UTC and can be a day ahead of ET — use last.
                    gameday = (
                        g.get("gameDate")
                        or week_block.get("date", "")
                        or (g.get("startTimeUTC") or "")[:10]
                    )
                    # Skip games that already happened
                    if gameday and gameday < today_str:
                        continue
                    games.append({
                        "game_id":        gid,
                        "gameday":        gameday,
                        "home_team":      home.get("abbrev", ""),
                        "away_team":      away.get("abbrev", ""),
                        "home_score":     home.get("score"),
                        "away_score":     away.get("score"),
                        "game_type":      str(g.get("gameType", "")),
                        "status":         g.get("gameState", ""),
                        "venue":          g.get("venue", {}).get("default", ""),
                        "start_time_utc": g.get("startTimeUTC", ""),
                    })

        # Fetch the week containing today (may include past days — filtered above)
        _extract(self._get(f"/schedule/{today_str}", ttl=TTL_SCHEDULE))
        # Also fetch next week's block for a full 7-day view
        _extract(self._get(f"/schedule/{next_str}", ttl=TTL_SCHEDULE))

        return games

    # ── Boxscore ──────────────────────────────────────────────────────────────

    def get_boxscore(self, game_id: int) -> Optional[Dict]:
        """
        Return full boxscore for a single game.

        Returns dict with keys:
          home_team, away_team, home_score, away_score,
          period_end (REG/OT/SO), home_goalie, away_goalie,
          home_shots, away_shots, home_pp_pct, away_pp_pct
        """
        data = self._get(f"/gamecenter/{game_id}/boxscore", ttl=TTL_BOXSCORE)
        if not data:
            return None

        home = data.get("homeTeam", {})
        away = data.get("awayTeam", {})

        # Determine how game ended (REG / OT / SO)
        # gameOutcome.lastPeriodType is the most reliable field across all eras
        outcome = data.get("gameOutcome", {})
        game_end = outcome.get("lastPeriodType", "REG")  # REG, OT, SO
        if not game_end:
            # Fallback: check periodDescriptor
            pd = data.get("periodDescriptor", {})
            ptype = pd.get("periodType", "REG")
            game_end = ptype if ptype in ("REG", "OT", "SO") else "REG"

        # Goalies (first listed = starter who played most)
        def _lead_goalie(team_dict: dict) -> dict:
            goalies = team_dict.get("goalies", [])
            if not goalies:
                return {}
            # Sort by toi (time on ice) descending
            goalies_sorted = sorted(goalies, key=lambda g: g.get("toi", "00:00"), reverse=True)
            g = goalies_sorted[0]
            return {
                "player_id":   g.get("playerId"),
                "name":        g.get("name", {}).get("default", ""),
                "sv_pct":      g.get("savePctg"),
                "saves":       g.get("saves"),
                "shots_against": g.get("shotsAgainst"),
                "goals_against": g.get("goalsAgainst"),
            }

        # Power play
        def _pp_pct(team_dict: dict) -> Optional[float]:
            pp = team_dict.get("powerPlayConversions", "")
            # Format: "1/3" → 0.333
            if pp and "/" in str(pp):
                parts = str(pp).split("/")
                try:
                    n, d = int(parts[0]), int(parts[1])
                    return n / d if d > 0 else 0.0
                except Exception:
                    return None
            return None

        return {
            "game_id":      game_id,
            "home_team":    home.get("abbrev", ""),
            "away_team":    away.get("abbrev", ""),
            "home_score":   home.get("score", 0),
            "away_score":   away.get("score", 0),
            "game_end":     game_end,     # REG / OT / SO
            "home_goalie":  _lead_goalie(home),
            "away_goalie":  _lead_goalie(away),
            "home_shots":   home.get("sog"),
            "away_shots":   away.get("sog"),
            "home_pp_pct":  _pp_pct(home),
            "away_pp_pct":  _pp_pct(away),
        }

    # ── Standings ─────────────────────────────────────────────────────────────

    def get_standings(self) -> List[Dict]:
        """Return current standings for all teams."""
        data = self._get("/standings/now", ttl=TTL_STANDINGS)
        if not data:
            return []
        standings = []
        for rec in data.get("standings", []):
            standings.append({
                "team":      rec.get("teamAbbrev", {}).get("default", ""),
                "gp":        rec.get("gamesPlayed", 0),
                "wins":      rec.get("wins", 0),
                "losses":    rec.get("losses", 0),
                "ot_losses": rec.get("otLosses", 0),
                "points":    rec.get("points", 0),
                "gf":        rec.get("goalFor", 0),
                "ga":        rec.get("goalAgainst", 0),
                "win_pct":   rec.get("winPctg", 0.0),
            })
        return standings

    # ── Roster ────────────────────────────────────────────────────────────────

    def get_team_roster(self, team_abbrev: str, season_id: str = "20242025") -> Dict:
        """
        Return current roster for a team, organized by position group.

        Returns: {forwards: [...], defensemen: [...], goalies: [...]}
        Each player dict has: player_id, name, position, jersey_number
        """
        data = self._get(f"/roster/{team_abbrev}/{season_id}", ttl=TTL_ROSTER)
        if not data:
            return {"forwards": [], "defensemen": [], "goalies": []}

        def _parse_players(lst):
            result = []
            for p in (lst or []):
                name_dict = p.get("firstName", {})
                last_dict = p.get("lastName", {})
                result.append({
                    "player_id":     p.get("id"),
                    "name":          f"{name_dict.get('default', '')} {last_dict.get('default', '')}".strip(),
                    "position":      p.get("positionCode", ""),
                    "jersey_number": p.get("sweaterNumber"),
                })
            return result

        return {
            "forwards":   _parse_players(data.get("forwards", [])),
            "defensemen": _parse_players(data.get("defensemen", [])),
            "goalies":    _parse_players(data.get("goalies", [])),
        }

    # ── Goalie stats ──────────────────────────────────────────────────────────

    def get_goalie_season_stats(self, season_id: str) -> List[Dict]:
        """
        Return aggregate season stats for all NHL goalies in a season.

        Fetches from the NHL stats API (different base path).
        Returns list of dicts: {player_id, team, name, gp, sv_pct, gaa, wins, losses}
        """
        # NHL stats API endpoint for goalie summary
        url = (
            f"https://api.nhle.com/stats/rest/en/goalie/summary"
            f"?isAggregate=false&isGame=false&sort=%5B%7B%22property%22:%22wins%22"
            f",%22direction%22:%22DESC%22%7D%5D&start=0&limit=200"
            f"&cayenneExp=seasonId%3D{season_id}%20and%20gameTypeId%3D2"
        )
        cached = cache.get(url)
        if cached is not None:
            return cached

        try:
            resp = self._session.get(url, timeout=20)
            resp.raise_for_status()
            raw = resp.json()
            time.sleep(self._sleep)
        except Exception as e:
            logger.warning(f"NHL goalie stats error {season_id}: {e}")
            return []

        goalies = []
        for g in raw.get("data", []):
            goalies.append({
                "player_id": g.get("playerId"),
                "name":      g.get("goalieFullName", ""),
                "team":      g.get("teamAbbrevs", ""),
                "gp":        g.get("gamesPlayed", 0),
                "wins":      g.get("wins", 0),
                "losses":    g.get("losses", 0),
                "ot_losses": g.get("otLosses", 0),
                "sv_pct":    g.get("savePct"),
                "gaa":       g.get("goalsAgainstAverage"),
                "shots_against": g.get("shotsAgainst", 0),
                "saves":     g.get("saves", 0),
            })
        cache.set(url, goalies, ttl=TTL_GOALIE)
        return goalies

    # ── Team stats (shots/xG proxy) ───────────────────────────────────────────

    def get_team_season_stats(self, season_id: str) -> List[Dict]:
        """
        Return aggregate team stats for a season (shots for/against, PP/PK).

        Returns list of dicts per team.
        """
        url = (
            f"https://api.nhle.com/stats/rest/en/team/summary"
            f"?isAggregate=false&isGame=false&sort=%5B%7B%22property%22:%22wins%22"
            f",%22direction%22:%22DESC%22%7D%5D&start=0&limit=50"
            f"&cayenneExp=seasonId%3D{season_id}%20and%20gameTypeId%3D2"
        )
        cached = cache.get(url)
        if cached is not None:
            return cached

        try:
            resp = self._session.get(url, timeout=20)
            resp.raise_for_status()
            raw = resp.json()
            time.sleep(self._sleep)
        except Exception as e:
            logger.warning(f"NHL team stats error {season_id}: {e}")
            return []

        teams = []
        for t in raw.get("data", []):
            gp = t.get("gamesPlayed", 1) or 1
            teams.append({
                "team":            NHL_NAME_TO_ABV.get(t.get("teamFullName", ""), t.get("teamAbbrevs", "")),
                "gp":              gp,
                "wins":            t.get("wins", 0),
                "losses":          t.get("losses", 0),
                "goals_for":       t.get("goalsFor", 0),
                "goals_against":   t.get("goalsAgainst", 0),
                "goals_for_pg":    (t.get("goalsFor", 0) or 0) / gp,
                "goals_against_pg": (t.get("goalsAgainst", 0) or 0) / gp,
                "shots_for_pg":    t.get("shotsForPerGame"),
                "shots_against_pg": t.get("shotsAgainstPerGame"),
                "pp_pct":          t.get("powerPlayPct"),
                "pk_pct":          t.get("penaltyKillPct"),
                "shot_diff_pg":    (
                    (t.get("shotsForPerGame") or 0) - (t.get("shotsAgainstPerGame") or 0)
                ),
            })
        cache.set(url, teams, ttl=TTL_GOALIE)
        return teams

    # ── Player game log ───────────────────────────────────────────────────────

    def get_player_game_log(
        self, player_id: int, season_id: str, game_type: str = "2"
    ) -> List[Dict]:
        """Return per-game stats for a player in a season."""
        data = self._get(
            f"/player/{player_id}/game-log/{season_id}/{game_type}",
            ttl=TTL_PLAYER,
        )
        if not data:
            return []
        return data.get("gameLog", [])
