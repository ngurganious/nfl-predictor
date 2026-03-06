"""
mlb_game_week.py
================
Fetches the current week's MLB schedule and provides SP / lineup helpers.
Mirrors nhl_game_week.py from the NHL predictor.

Usage:
    from mlb_game_week import (
        fetch_mlb_weekly_schedule, get_mlb_team_roster_by_position,
        get_mlb_starter_pitcher, MLB_POSITION_GROUPS,
    )
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Position groupings for MLB lineup display
MLB_POSITION_GROUPS = {
    'pitcher':  ['SP', 'P'],
    'catcher':  ['C'],
    'infield':  ['1B', '2B', '3B', 'SS'],
    'outfield': ['LF', 'CF', 'RF'],
    'dh':       ['DH'],
}

# ── Datetime helpers ──────────────────────────────────────────────────────────

def _parse_game_datetime(game_time_utc: str) -> tuple:
    """
    Parse game_time_utc from MLB API into (day_of_week, date_label, time_et_str, datetime_et).
    Falls back gracefully if parsing fails.
    """
    if not game_time_utc:
        return ("Unknown", "TBD", "TBD", None)

    try:
        import pytz
        et_zone = pytz.timezone('America/New_York')
        dt_utc  = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
        dt_et   = dt_utc.astimezone(et_zone)
        return (
            dt_et.strftime("%A"),
            dt_et.strftime("%b %d"),
            dt_et.strftime("%I:%M %p ET").lstrip("0"),
            dt_et,
        )
    except Exception:
        pass

    # Fallback: manual UTC→ET offset
    try:
        dt_utc  = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
        month   = dt_utc.month
        offset  = timedelta(hours=-4) if 3 <= month <= 11 else timedelta(hours=-5)
        dt_et   = dt_utc + offset
        return (
            dt_et.strftime("%A"),
            dt_et.strftime("%b %d"),
            dt_et.strftime("%I:%M %p ET").lstrip("0"),
            dt_et,
        )
    except Exception:
        pass

    # Date-only fallback so day-of-week headers still work
    try:
        dt = datetime.strptime(game_time_utc[:10], "%Y-%m-%d")
        return (dt.strftime("%A"), dt.strftime("%b %d"), "TBD", dt.replace(tzinfo=timezone.utc))
    except Exception:
        return ("Unknown", "TBD", "TBD", None)


# ── Schedule ──────────────────────────────────────────────────────────────────

def fetch_mlb_weekly_schedule(mlb_client) -> Dict[str, List[Dict]]:
    """
    Fetch this week's MLB games and organize by day of week.

    Returns: {'Tuesday': [game_dict, ...], 'Wednesday': [...], ...}
    Each game_dict has:
      game_pk, home_team, away_team, home_team_name, away_team_name,
      game_date_label, game_time_et, venue, status, datetime_et,
      home_sp, away_sp, day_night
    """
    try:
        raw_games = mlb_client.get_current_week_schedule()
    except Exception as e:
        logger.error(f"Failed to fetch MLB schedule: {e}")
        return {}

    by_day: Dict[str, List[Dict]] = {}

    for g in raw_games:
        game_time_utc = g.get('game_time_utc', g.get('game_date', ''))
        day_of_week, date_label, time_et_str, dt_et = _parse_game_datetime(game_time_utc)

        game_dict = {
            'game_pk':        g.get('game_pk'),
            'home_team':      g.get('home_team', ''),
            'away_team':      g.get('away_team', ''),
            'home_team_name': g.get('home_team_name', g.get('home_team', '')),
            'away_team_name': g.get('away_team_name', g.get('away_team', '')),
            'game_date_label': date_label,
            'game_time_et':   time_et_str,
            'venue':          g.get('venue', ''),
            'status':         g.get('status', 'Scheduled'),
            'datetime_et':    dt_et,
            'home_sp':        g.get('home_sp'),   # dict with id/name or None
            'away_sp':        g.get('away_sp'),
            'day_night':      g.get('day_night', 'night'),
        }

        day_key = f"{day_of_week} {date_label}" if date_label != "TBD" else day_of_week
        by_day.setdefault(day_key, []).append(game_dict)

    # Sort day buckets by the actual game date of the first entry in each bucket
    from datetime import date as _date
    def _day_sort_key(kv):
        for g in kv[1]:
            dt = g.get('datetime_et')
            if dt is not None:
                try:
                    return dt.date()
                except Exception:
                    pass
        return _date.max

    by_day_sorted = dict(sorted(by_day.items(), key=_day_sort_key))

    total = sum(len(v) for v in by_day_sorted.values())
    logger.info(f"MLB weekly schedule: {total} games across {len(by_day_sorted)} days")
    return by_day_sorted


# ── Roster / lineup helpers ───────────────────────────────────────────────────

def get_mlb_team_roster_by_position(team_abbrev: str, mlb_client, team_id: int = None) -> Dict[str, List[Dict]]:
    """
    Return active roster grouped by position category.

    If team_id is not provided, it is looked up via the client's team list.

    Returns: {
        'pitcher':  [{'id': ..., 'name': ..., 'position': 'SP'}, ...],
        'catcher':  [...],
        'infield':  [...],
        'outfield': [...],
        'dh':       [...],
        'other':    [...],
    }
    """
    if team_id is None:
        team_id = _lookup_team_id(team_abbrev, mlb_client)
    if team_id is None:
        logger.warning(f"Could not resolve team_id for {team_abbrev}")
        return {k: [] for k in MLB_POSITION_GROUPS}

    try:
        roster = mlb_client.get_team_roster(team_id)
    except Exception as e:
        logger.error(f"Failed to fetch {team_abbrev} roster: {e}")
        return {k: [] for k in MLB_POSITION_GROUPS}

    grouped: Dict[str, List[Dict]] = {k: [] for k in list(MLB_POSITION_GROUPS.keys()) + ['other']}

    for player in roster:
        pos = player.get('position', '').upper()
        placed = False
        for group, codes in MLB_POSITION_GROUPS.items():
            if pos in codes or (group == 'pitcher' and pos in ('P', 'RP', 'SP', 'CL')):
                grouped[group].append(player)
                placed = True
                break
        if not placed:
            grouped['other'].append(player)

    return grouped


def _lookup_team_id(team_abbrev: str, mlb_client) -> Optional[int]:
    """Find team_id for an abbreviation using the client's get_teams() endpoint."""
    try:
        teams = mlb_client.get_teams()
        for t in teams:
            if t.get('abbreviation', '').upper() == team_abbrev.upper():
                return t['id']
    except Exception as e:
        logger.warning(f"Team ID lookup failed for {team_abbrev}: {e}")
    return None


# Lineup slots: (display_label, position_group, slot_index_within_group)
MLB_LINEUP_SLOTS = [
    ('C',   'catcher',  0),
    ('1B',  'infield',  0),
    ('2B',  'infield',  1),
    ('3B',  'infield',  2),
    ('SS',  'infield',  3),
    ('LF',  'outfield', 0),
    ('CF',  'outfield', 1),
    ('RF',  'outfield', 2),
    ('DH',  'dh',       0),
]


def get_mlb_players_for_slot(roster_by_pos, pos_group, slot_idx):
    players = roster_by_pos.get(pos_group, [])
    if not players:
        return ['Unknown']
    names = [p.get('name', 'Unknown') for p in players]
    if slot_idx < len(names):
        top = names[slot_idx]
        rest = [n for i, n in enumerate(names) if i != slot_idx]
        return [top] + rest
    return names if names else ['Unknown']


def get_mlb_starter_pitcher(game: Dict, side: str = 'home') -> Optional[Dict]:
    """
    Return the probable starting pitcher for a side ('home' or 'away').
    Returns dict with 'id' and 'name', or None if not yet announced.
    """
    sp = game.get(f'{side}_sp')
    return sp if sp and sp.get('name') else None


def get_mlb_sp_display_name(sp: Optional[Dict]) -> str:
    """Human-readable SP name for display — 'TBD' if not confirmed."""
    if not sp:
        return "TBD"
    return sp.get('name', 'TBD')
