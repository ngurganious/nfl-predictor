"""
MLB Official Stats API Client
==============================
No API key required. Uses the official MLB Stats API at statsapi.mlb.com/api/v1/.

Provides:
  - Season schedules with final scores (historical + current)
  - Current week's schedule
  - Probable starting pitchers
  - Team rosters / lineup cards
  - Team standings

Usage:
    from apis.mlb import MLBClient
    client = MLBClient()

    games = client.get_season_schedule(2024)
    week  = client.get_current_week_schedule()
    sp    = client.get_probable_pitchers(gamePk)

Note: The MLB Stats API has no documented rate limits but be respectful.
Cache TTLs are conservative. Historical game data is permanent (never changes).
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

import requests

from . import cache

logger = logging.getLogger(__name__)

_BASE = "https://statsapi.mlb.com/api/v1"

TTL_SCHEDULE_HIST = 60 * 60 * 24 * 7   # 7 days (historical games never change)
TTL_SCHEDULE_CURR = 60 * 60 * 3         # 3 hrs (current week, scores update)
TTL_ROSTER        = 60 * 60 * 12        # 12 hrs
TTL_STANDINGS     = 60 * 60 * 1         # 1 hr
TTL_TEAMS         = 60 * 60 * 24 * 30   # 30 days (team list rarely changes)

# ── MLB team constants ─────────────────────────────────────────────────────────
# Standard abbreviations as returned by statsapi.mlb.com (2024 season)
MLB_TEAMS = [
    'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CIN', 'CLE', 'COL',
    'CWS', 'DET', 'HOU', 'KC',  'LAA', 'LAD', 'MIA', 'MIL',
    'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SD',  'SEA',
    'SF',  'STL', 'TB',  'TEX', 'TOR', 'WSH',
]

# Historical franchise aliases for ELO continuity
# Key = old abbreviation, value = current abbreviation used by EdgeIQ
MLB_HISTORICAL_ALIASES: Dict[str, str] = {
    'FLA':  'MIA',   # Florida Marlins → Miami Marlins (2012)
    'MON':  'WSH',   # Montreal Expos → Washington Nationals (2005)
    'TBA':  'TB',    # Tampa Bay Devil Rays old code
    'TBD':  'TB',    # Tampa Bay Devil Rays another variant
}

def normalize_team(abbr: str) -> str:
    if not abbr:
        return abbr
    return MLB_HISTORICAL_ALIASES.get(abbr.upper(), abbr.upper())


class MLBClient:
    def __init__(self, timeout: int = 20):
        self.timeout  = timeout
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': 'EdgeIQ/1.0 (edgeiq-app)'})

    def _get(self, endpoint: str, params: dict = None, ttl: int = 3600) -> Optional[dict]:
        url = f"{_BASE}{endpoint}"
        cache_key = url + str(sorted((params or {}).items()))
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            r = self._session.get(url, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            cache.set(cache_key, data, ttl=ttl)
            return data
        except Exception as e:
            logger.warning(f"MLBClient GET {url} failed: {e}")
            return None

    # ── Team ID → abbreviation lookup ─────────────────────────────────────────
    def _get_team_id_map(self, season: int = 2024) -> Dict[int, str]:
        """Return {team_id: abbreviation} dict, cached."""
        data = self._get("/teams", params={'sportId': 1, 'season': season}, ttl=TTL_TEAMS)
        if not data:
            return {}
        return {t['id']: normalize_team(t.get('abbreviation', ''))
                for t in data.get('teams', [])}

    # ── Season schedule ────────────────────────────────────────────────────────
    def get_season_schedule(self, year: int) -> List[Dict]:
        """Return all regular-season games for a given year with final scores."""
        from calendar import month_abbr
        start = f"{year}-03-01"
        end   = f"{year}-11-30"
        ttl   = TTL_SCHEDULE_HIST if year < datetime.now().year else TTL_SCHEDULE_CURR
        data  = self._get("/schedule", params={
            'sportId': 1,
            'season':  year,
            'gameType': 'R',
            'startDate': start,
            'endDate':   end,
        }, ttl=ttl)
        if not data:
            return []
        id_map = self._get_team_id_map(season=year)
        games = []
        for date_block in data.get('dates', []):
            for g in date_block.get('games', []):
                status = g.get('status', {}).get('abstractGameState', '')
                if status != 'Final':
                    continue
                home = g['teams']['home']
                away = g['teams']['away']
                home_score = home.get('score')
                away_score = away.get('score')
                if home_score is None or away_score is None:
                    continue
                home_id = home['team']['id']
                away_id = away['team']['id']
                games.append({
                    'game_pk':     g['gamePk'],
                    'game_date':   g['officialDate'],
                    'season':      year,
                    'home_team':   id_map.get(home_id, str(home_id)),
                    'away_team':   id_map.get(away_id, str(away_id)),
                    'home_score':  int(home_score),
                    'away_score':  int(away_score),
                    'home_win':    1 if int(home_score) > int(away_score) else 0,
                    'day_night':   g.get('dayNight', 'night'),
                    'is_day_game': 1 if g.get('dayNight', 'night').lower() == 'day' else 0,
                    'venue':       g.get('venue', {}).get('name', ''),
                })
        return games

    # ── Current week schedule ──────────────────────────────────────────────────
    def get_current_week_schedule(self, days_ahead: int = 7) -> List[Dict]:
        """Return this week's upcoming games (scheduled + in-progress + final)."""
        today = datetime.now(timezone.utc).date()
        end   = today + timedelta(days=days_ahead)
        data  = self._get("/schedule", params={
            'sportId':   1,
            'gameType':  'R',
            'startDate': today.isoformat(),
            'endDate':   end.isoformat(),
            'hydrate':   'probablePitcher(note),lineups',
        }, ttl=TTL_SCHEDULE_CURR)
        if not data:
            return []
        id_map = self._get_team_id_map()
        games = []
        for date_block in data.get('dates', []):
            for g in date_block.get('games', []):
                home = g['teams']['home']
                away = g['teams']['away']
                home_id = home['team']['id']
                away_id = away['team']['id']
                status = g.get('status', {}).get('detailedState', '')
                home_sp = self._extract_probable_pitcher(home)
                away_sp = self._extract_probable_pitcher(away)
                games.append({
                    'game_pk':        g['gamePk'],
                    'game_date':      g['officialDate'],
                    'game_time_utc':  g.get('gameDate', ''),
                    'status':         status,
                    'home_team':      id_map.get(home_id, str(home_id)),
                    'away_team':      id_map.get(away_id, str(away_id)),
                    'home_team_name': home['team'].get('name', ''),
                    'away_team_name': away['team'].get('name', ''),
                    'home_score':     home.get('score'),
                    'away_score':     away.get('score'),
                    'home_sp':        home_sp,
                    'away_sp':        away_sp,
                    'venue':          g.get('venue', {}).get('name', ''),
                    'day_night':      g.get('dayNight', 'night'),
                })
        return games

    def _extract_probable_pitcher(self, team_data: dict) -> Optional[Dict]:
        pp = team_data.get('probablePitcher')
        if not pp:
            return None
        return {
            'id':         pp.get('id'),
            'name':       pp.get('fullName', pp.get('name', 'TBD')),
            'note':       pp.get('note', ''),
        }

    # ── Probable pitchers for a specific game ─────────────────────────────────
    def get_probable_pitchers(self, game_pk: int) -> Dict:
        """Return probable starters for a specific game."""
        data = self._get(f"/game/{game_pk}/feed/live", ttl=TTL_SCHEDULE_CURR)
        if not data:
            return {'home': None, 'away': None}
        try:
            boxscore = data['liveData']['boxscore']['teams']
            home_id  = data['gameData']['teams']['home']['id']
            away_id  = data['gameData']['teams']['away']['id']
            home_sp  = self._get_sp_from_boxscore(boxscore.get('home', {}))
            away_sp  = self._get_sp_from_boxscore(boxscore.get('away', {}))
            return {'home': home_sp, 'away': away_sp}
        except Exception:
            return {'home': None, 'away': None}

    def _get_sp_from_boxscore(self, team_box: dict) -> Optional[Dict]:
        pitchers = team_box.get('pitchers', [])
        players  = team_box.get('players', {})
        if not pitchers:
            return None
        sp_id   = f"ID{pitchers[0]}"
        sp_data = players.get(sp_id, {}).get('person', {})
        stats   = players.get(sp_id, {}).get('stats', {}).get('pitching', {})
        return {
            'id':   pitchers[0],
            'name': sp_data.get('fullName', 'Unknown'),
            'ip':   stats.get('inningsPitched', '0.0'),
            'k':    stats.get('strikeOuts', 0),
            'er':   stats.get('earnedRuns', 0),
        }

    # ── Team roster ───────────────────────────────────────────────────────────
    def get_team_roster(self, team_id: int) -> List[Dict]:
        data = self._get(f"/teams/{team_id}/roster", params={'rosterType': 'active'}, ttl=TTL_ROSTER)
        if not data:
            return []
        return [
            {
                'id':       p['person']['id'],
                'name':     p['person']['fullName'],
                'position': p.get('position', {}).get('abbreviation', ''),
                'jersey':   p.get('jerseyNumber', ''),
            }
            for p in data.get('roster', [])
        ]

    # ── Standings ─────────────────────────────────────────────────────────────
    def get_standings(self, season: Optional[int] = None) -> Dict[str, Any]:
        params = {'leagueId': '103,104', 'standingsTypes': 'regularSeason'}
        if season:
            params['season'] = season
        return self._get("/standings", params=params, ttl=TTL_STANDINGS) or {}

    # ── All current MLB teams ─────────────────────────────────────────────────
    def get_teams(self, season: Optional[int] = None) -> List[Dict]:
        params = {'sportId': 1}
        if season:
            params['season'] = season
        data = self._get("/teams", params=params, ttl=TTL_TEAMS)
        if not data:
            return []
        return [
            {
                'id':           t['id'],
                'abbreviation': normalize_team(t.get('abbreviation', '')),
                'name':         t.get('name', ''),
                'short_name':   t.get('shortName', ''),
                'division':     t.get('division', {}).get('name', ''),
                'league':       t.get('league', {}).get('name', ''),
                'venue':        t.get('venue', {}).get('name', ''),
            }
            for t in data.get('teams', [])
        ]
