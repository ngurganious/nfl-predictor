"""
nhl_game_week.py
================
Fetches the current week's NHL schedule and provides team lineup/roster helpers.
Mirrors game_week.py from the NFL predictor.

Usage:
    from nhl_game_week import (
        fetch_nhl_weekly_schedule, get_nhl_team_depth_chart,
        get_nhl_starter_goalie, NHL_POSITIONS,
    )
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# NHL hockey position groups (simplified vs NFL's 24 positions)
NHL_POSITIONS = {
    'forwards':   ['C', 'LW', 'RW', 'F'],
    'defensemen': ['D'],
    'goalies':    ['G'],
}

# NHL arena locations (for weather context — mostly N/A, but needed for outdoor games)
# Outdoor NHL games: Winter Classic, Stadium Series
OUTDOOR_VENUES = {
    "outdoor",
    "winter classic",
    "stadium series",
    "heritage classic",
    "navy-marine corps memorial stadium",
    "notre dame stadium",
    "fenway park",
    "lambeau field",
    "wrigley field",
    "dodger stadium",
    "levi's stadium",
    "heinz field",
    "soldier field",
    "metlife stadium",
    "big house",
    "cotton bowl",
    "cotton bowl stadium",
    "neyland stadium",
}

# ── Day-of-week ordering ────────────────────────────────────────────────────
_DAY_ORDER = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
    'Friday': 4, 'Saturday': 5, 'Sunday': 6,
}

# ── Lineup position slots ────────────────────────────────────────────────────
# Each slot: (display_label, depth_chart_key, depth_slot)
NHL_FWD_SLOTS = [
    ('LW1', 'LW', 1), ('C1', 'C', 1), ('RW1', 'RW', 1),
    ('LW2', 'LW', 2), ('C2', 'C', 2), ('RW2', 'RW', 2),
]
NHL_DEF_SLOTS = [
    ('D1', 'D', 1), ('D2', 'D', 2),
    ('D3', 'D', 3), ('D4', 'D', 4),
]
NHL_GOALIE_SLOTS = [
    ('G',  'G', 1),  # starter
    ('G2', 'G', 2),  # backup
]


def _parse_game_datetime(start_time_utc: str) -> tuple:
    """
    Parse startTimeUTC from NHL API into usable components.
    Returns: (day_of_week, date_label, time_et_str, datetime_et)
    """
    if not start_time_utc:
        return ("Unknown", "TBD", "TBD", None)

    # Try pytz for accurate DST-aware conversion (same as NFL game_week.py)
    try:
        import pytz
        et_zone = pytz.timezone('America/New_York')
        dt_utc  = datetime.fromisoformat(start_time_utc.replace("Z", "+00:00"))
        dt_et   = dt_utc.astimezone(et_zone)
        day_of_week = dt_et.strftime("%A")
        date_label  = dt_et.strftime("%b %d")
        time_et_str = dt_et.strftime("%I:%M %p ET").lstrip("0")
        return (day_of_week, date_label, time_et_str, dt_et)
    except Exception:
        pass

    # Fallback: manual UTC→ET offset (no pytz)
    try:
        dt_utc = datetime.fromisoformat(start_time_utc.replace("Z", "+00:00"))
        month = dt_utc.month
        et_offset = timedelta(hours=-4) if 3 <= month <= 11 else timedelta(hours=-5)
        dt_et = dt_utc + et_offset
        day_of_week = dt_et.strftime("%A")
        date_label  = dt_et.strftime("%b %d")
        time_et_str = dt_et.strftime("%I:%M %p ET").lstrip("0")
        return (day_of_week, date_label, time_et_str, dt_et)
    except Exception:
        pass

    # Last fallback: parse just the date portion so day-of-week headers still work
    try:
        clean = start_time_utc[:10]
        dt = datetime.strptime(clean, "%Y-%m-%d")
        return (dt.strftime("%A"), dt.strftime("%b %d"), "TBD", dt.replace(tzinfo=timezone.utc))
    except Exception:
        return ("Unknown", "TBD", "TBD", None)


def fetch_nhl_weekly_schedule(nhl_client) -> Dict[str, List[Dict]]:
    """
    Fetch this week's NHL games and organize by day of week.

    Returns: {'Tuesday': [game_dict, ...], 'Wednesday': [...], ...}
    Each game_dict has:
      game_id, home_team, away_team, game_date_label, game_time_et,
      venue, status, datetime_et, is_outdoor
    """
    try:
        raw_games = nhl_client.get_current_week_schedule()
    except Exception as e:
        logger.error(f"Failed to fetch NHL schedule: {e}")
        return {}

    by_day: Dict[str, List[Dict]] = {}

    for g in raw_games:
        # Only include regular season games (game_type "2") or playoffs ("3")
        gtype = str(g.get('game_type', '2'))
        if gtype not in ('2', '3'):
            continue

        start_time_utc = g.get('start_time_utc', '')
        day_of_week, date_label, time_et_str, dt_et = _parse_game_datetime(start_time_utc)

        venue = g.get('venue', '')
        is_outdoor = any(kw in venue.lower() for kw in OUTDOOR_VENUES)

        game_dict = {
            'game_id':        g.get('game_id'),
            'home_team':      g.get('home_team', ''),
            'away_team':      g.get('away_team', ''),
            'game_date_label': date_label,
            'game_time_et':   time_et_str,
            'venue':          venue,
            'status':         g.get('status', 'FUT'),
            'datetime_et':    dt_et,
            'is_outdoor':     is_outdoor,
            'game_type':      gtype,
        }

        # Key by "Day Mon DD" (e.g. "Saturday Mar 01") so two different Saturdays
        # across a week boundary don't get merged into the same section.
        day_key = f"{day_of_week} {date_label}" if date_label != "TBD" else day_of_week
        if day_key not in by_day:
            by_day[day_key] = []
        by_day[day_key].append(game_dict)

    # Sort days by actual date of first game (not by day-of-week name), so that
    # e.g. "Friday Feb 27" appears before "Monday Mar 02" across a week boundary.
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
    logger.info(f"NHL weekly schedule: {total} games across {len(by_day_sorted)} days")
    return by_day_sorted


def get_nhl_team_depth_chart(team: str, nhl_client) -> Dict[str, List[Dict]]:
    """
    Return a simplified depth chart for an NHL team.

    Returns: {
        'C':  [{'name': '...', 'player_id': ...}, ...],  # Centers (line 1, 2, 3, 4)
        'LW': [...],   # Left wings
        'RW': [...],   # Right wings
        'D':  [...],   # Defensemen (pairs 1, 2, 3)
        'G':  [...],   # Goalies (starter, backup)
    }
    """
    try:
        roster = nhl_client.get_team_roster(team)
    except Exception as e:
        logger.error(f"Failed to fetch {team} roster: {e}")
        return {'C': [], 'LW': [], 'RW': [], 'D': [], 'G': []}

    chart = {'C': [], 'LW': [], 'RW': [], 'D': [], 'G': []}

    for player in roster.get('forwards', []):
        pos = player.get('position', 'F').upper()
        if pos == 'C':
            chart['C'].append(player)
        elif pos in ('LW', 'L'):   # API returns 'L' for left wing
            chart['LW'].append(player)
        elif pos in ('RW', 'R'):   # API returns 'R' for right wing
            chart['RW'].append(player)
        else:
            # Unknown forward position — put in C
            chart['C'].append(player)

    for player in roster.get('defensemen', []):
        chart['D'].append(player)

    for player in roster.get('goalies', []):
        chart['G'].append(player)

    return chart


def get_nhl_starter_goalie(depth_chart: Dict[str, List[Dict]]) -> Optional[Dict]:
    """Return the first (starter) goalie from the depth chart."""
    goalies = depth_chart.get('G', [])
    return goalies[0] if goalies else None


def get_nhl_top_forwards(depth_chart: Dict[str, List[Dict]], n: int = 3) -> List[Dict]:
    """Return top N centers as a proxy for the top forward line."""
    return depth_chart.get('C', [])[:n]


def get_nhl_top_defense(depth_chart: Dict[str, List[Dict]], n: int = 2) -> List[Dict]:
    """Return top N defensemen (first pair)."""
    return depth_chart.get('D', [])[:n]


def is_outdoor_game(game: dict) -> bool:
    """Check if a game is an outdoor event (Winter Classic, Stadium Series, etc.)."""
    return game.get('is_outdoor', False)


def get_nhl_players_for_slot(
    depth_chart: Dict[str, List[Dict]],
    pos_key: str,
    slot_depth: int,
) -> List[str]:
    """
    Return an ordered list of player names for a given position slot.

    The player at `slot_depth` is put first (the expected starter for that line/pair),
    followed by the rest of the position group as swap-in options.

    Args:
        depth_chart: from get_nhl_team_depth_chart()
        pos_key:     'C', 'LW', 'RW', 'D', or 'G'
        slot_depth:  1=first line/pair, 2=second line/pair, etc.

    Returns:
        Ordered list of player names.
    """
    # Use roster order as-is — NHL API returns players in approximate line order
    # (line 1 first). Sorting by a 'depth' key that doesn't exist in the API
    # would scramble the order, so we skip it.
    players = depth_chart.get(pos_key, [])
    names = [p['name'] for p in players if p.get('name')]

    if not names:
        return [f'{pos_key} (N/A)']

    # Put the slot-depth player first, then remaining as alternatives
    idx = slot_depth - 1
    if idx < len(names):
        ordered = [names[idx]] + [n for i, n in enumerate(names) if i != idx]
    else:
        ordered = names

    return ordered
