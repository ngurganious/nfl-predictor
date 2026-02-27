"""
NFL Weekly Schedule Helpers
============================
Fetches the current week's schedule from ESPN (no API key required)
and organises games by day-of-week for the weekly schedule view in Tab 1.

Also provides an extended depth-chart builder that merges live API data
with the existing player_lookup.pkl scores.
"""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

DAY_ORDER = ['Thursday', 'Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Other']

# Offensive positions to display in order
OFF_POSITIONS = [
    ('QB',  'QB'),
    ('WR1', 'WR'),   # depth 1 WR
    ('WR2', 'WR'),   # depth 2 WR
    ('WR3', 'WR'),   # depth 3 WR
    ('WR4', 'WR'),   # depth 4 WR
    ('RB1', 'RB'),   # depth 1 RB
    ('RB2', 'RB'),   # depth 2 RB
    ('TE1', 'TE'),   # depth 1 TE
    ('TE2', 'TE'),   # depth 2 TE
    ('LT',  'LT'),
    ('LG',  'LG'),
    ('C',   'C'),
    ('RG',  'RG'),
    ('RT',  'RT'),
]

# Defensive positions to display in order
DEF_POSITIONS = [
    ('DE1', ['DE', 'EDGE', 'OLB', 'RDE', 'LDE'], 1),   # (label, pos aliases, depth slot)
    ('DE2', ['DE', 'EDGE', 'OLB', 'RDE', 'LDE'], 2),
    ('DT',  ['DT', 'NT', 'IDL'],                  1),
    ('LB1', ['LB', 'MLB', 'ILB', 'WLB', 'SLB'],   1),
    ('LB2', ['LB', 'MLB', 'ILB', 'WLB', 'SLB'],   2),
    ('LB3', ['LB', 'MLB', 'ILB', 'WLB', 'SLB'],   3),
    ('CB1', ['CB'],                                1),
    ('CB2', ['CB'],                                2),
    ('FS',  ['FS', 'S'],                           1),
    ('SS',  ['SS', 'S'],                           2),
]

# ── Schedule helpers ──────────────────────────────────────────────────────────

def fetch_weekly_schedule(espn_client, tank01_client=None) -> dict:
    """
    Fetch this week's NFL schedule from ESPN (no key required).
    Falls back to Tank01 if ESPN fails.

    Returns an ordered dict:
        { 'Thursday': [game_dict, ...], 'Sunday': [...], ... }

    Each game_dict contains:
        game_id, home_team, away_team, game_date_label,
        game_time_et, venue, status, datetime_et
    """
    raw_games = []

    # ESPN — primary source (no key required)
    try:
        raw_games = espn_client.get_scoreboard()
    except Exception:
        pass

    if not raw_games and tank01_client:
        try:
            from datetime import date
            year = date.today().year
            raw_games = tank01_client.get_schedule(season=year)
        except Exception:
            pass

    grouped = defaultdict(list)

    for g in raw_games:
        date_str = g.get('game_date', '')
        home     = g.get('home_team', '')
        away     = g.get('away_team', '')
        if not home or not away:
            continue

        parsed = _parse_game_datetime(date_str)
        day_name   = parsed['day_name']
        date_label = parsed['date_label']
        time_et    = parsed['time_et']
        dt_et      = parsed['dt_et']

        grouped[day_name].append({
            'game_id':        g.get('game_id', f'{away}@{home}'),
            'home_team':      home,
            'away_team':      away,
            'game_date_label':date_label,
            'game_time_et':   time_et,
            'venue':          g.get('venue', ''),
            'status':         g.get('status', ''),
            'datetime_et':    dt_et,
        })

    # Sort each day's games by kickoff
    for day in grouped:
        grouped[day].sort(key=lambda g: g['datetime_et'] or datetime.min.replace(tzinfo=timezone.utc))

    # Return in canonical NFL day order
    result = {}
    for day in DAY_ORDER:
        if day in grouped:
            result[day] = grouped[day]

    return result


def _parse_game_datetime(date_str: str) -> dict:
    """Parse an ISO-8601 game date string and return day/time info in ET."""
    fallback = {
        'day_name': 'Other',
        'date_label': date_str[:10] if date_str else 'TBD',
        'time_et': 'TBD',
        'dt_et': None,
    }
    if not date_str:
        return fallback

    try:
        import pytz
        et_zone = pytz.timezone('America/New_York')
        dt_utc  = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        dt_et   = dt_utc.astimezone(et_zone)
        return {
            'day_name':   dt_et.strftime('%A'),
            'date_label': dt_et.strftime('%B %d'),
            'time_et':    dt_et.strftime('%I:%M %p ET').lstrip('0'),
            'dt_et':      dt_et,
        }
    except Exception:
        pass

    # Fallback: parse as plain date string (YYYYMMDD or YYYY-MM-DD)
    try:
        clean = date_str[:10].replace('/', '-')
        dt    = datetime.strptime(clean, '%Y-%m-%d')
        return {
            'day_name':   dt.strftime('%A'),
            'date_label': dt.strftime('%B %d'),
            'time_et':    'TBD',
            'dt_et':      dt.replace(tzinfo=timezone.utc),
        }
    except Exception:
        return fallback


# ── Extreme weather flag ──────────────────────────────────────────────────────

def get_weather_flag(temp, wind, is_dome) -> Optional[str]:
    """
    Return a short extreme-weather string for collapsed game cards, or None.

    Thresholds: temp < 32 °F (freezing) or wind >= 20 mph.
    """
    if is_dome:
        return None
    flags = []
    if temp is not None and isinstance(temp, (int, float)) and temp < 32:
        flags.append(f"FREEZING {temp}°F")
    if wind is not None and isinstance(wind, (int, float)) and wind >= 20:
        flags.append(f"WIND {wind} mph")
    return " | ".join(flags) if flags else None


# ── Depth chart + player scoring ──────────────────────────────────────────────

def get_team_full_depth_chart(team: str, espn_client, tank01_client=None,
                               player_lookup: dict = None) -> dict:
    """
    Build a complete position-keyed depth chart for a team.

    Sources (tried in order):
    1. ESPN roster depth chart (no key required)
    2. Tank01 depth chart (requires RAPIDAPI_KEY)
    3. Fall back to player_lookup.pkl starters

    Returns:
        dict[str, list[dict]]  e.g. {'QB': [{'name':..., 'score':..., 'depth':1}], ...}
    """
    chart: dict[str, list[dict]] = {}

    # 1 — ESPN
    try:
        espn_chart = espn_client.get_depth_chart(team)
        for pos, players in espn_chart.items():
            pos_upper = pos.upper()
            chart[pos_upper] = [
                {
                    'name':     p.get('name', 'Unknown'),
                    'depth':    p.get('depth', i + 1),
                    'position': pos_upper,
                    'source':   'espn',
                }
                for i, p in enumerate(players)
            ]
    except Exception:
        pass

    # 2 — Tank01 supplement for missing positions
    if tank01_client and not chart:
        try:
            t01 = tank01_client.get_depth_chart(team)
            for pos, players in t01.items():
                pos_upper = pos.upper()
                if pos_upper not in chart:
                    chart[pos_upper] = [
                        {
                            'name':     p.get('name', 'Unknown'),
                            'depth':    p.get('depth', i + 1),
                            'position': pos_upper,
                            'source':   'tank01',
                        }
                        for i, p in enumerate(players)
                    ]
        except Exception:
            pass

    # 3 — Supplement from player_lookup.pkl for QB/RB/WR/TE
    if player_lookup:
        for pos in ['QB', 'RB', 'WR', 'TE']:
            lookup_players = player_lookup.get(team, {}).get(pos, [])
            if lookup_players:
                if pos not in chart:
                    chart[pos] = [
                        {'name': p['name'], 'depth': i + 1, 'position': pos, 'source': 'lookup'}
                        for i, p in enumerate(lookup_players)
                    ]
                # Merge scores into existing chart entries
                score_map = {p['name']: p.get('score', 50.0) for p in lookup_players}
                for entry in chart.get(pos, []):
                    if entry['name'] in score_map:
                        entry['score'] = score_map[entry['name']]

    # Fill any missing scores with depth-based defaults
    for pos, players in chart.items():
        for p in players:
            if 'score' not in p:
                p['score'] = _depth_to_score(p.get('depth', 1))

    return chart


def build_lineup_dict(team: str, depth_chart: dict, selections: dict) -> dict:
    """
    Build a full lineup dict from user widget selections.

    Args:
        team:        team abbreviation
        depth_chart: from get_team_full_depth_chart()
        selections:  dict of {label: player_name} e.g. {'QB': 'P.Mahomes', 'WR1': 'T.Hill', ...}

    Returns:
        dict with keys:  qb, wr1, wr2, wr3, wr4, rb1, rb2, te1, te2, lt, lg, c, rg, rt
                    and: qb_score, wr1_score, ..., rb1_score, ..., te1_score, ..., lt_score, ...
    """
    result = {}
    score_cache: dict[str, float] = {}

    def _score_of(pos_alias: str, name: str) -> float:
        key = f'{pos_alias}::{name}'
        if key in score_cache:
            return score_cache[key]
        for p in depth_chart.get(pos_alias, []):
            if p.get('name') == name:
                score_cache[key] = p.get('score', 55.0)
                return score_cache[key]
        score_cache[key] = 55.0
        return 55.0

    label_to_base_pos = {
        'QB':  'QB',  'WR1': 'WR',  'WR2': 'WR',  'WR3': 'WR',  'WR4': 'WR',
        'RB1': 'RB',  'RB2': 'RB',  'TE1': 'TE',  'TE2': 'TE',
        'LT':  'LT',  'LG':  'LG',  'C':   'C',   'RG':  'RG',  'RT':  'RT',
        # defense
        'DE1': 'DE',  'DE2': 'DE',  'DT':  'DT',
        'LB1': 'LB',  'LB2': 'LB',  'LB3': 'LB',
        'CB1': 'CB',  'CB2': 'CB',
        'FS':  'FS',  'SS':  'SS',
    }

    for label, name in selections.items():
        result[label.lower()] = name
        base_pos = label_to_base_pos.get(label, label)
        result[f'{label.lower()}_score'] = _score_of(base_pos, name)

    return result


def calc_offense_score(lineup: dict) -> float:
    """
    Weighted offensive quality score (0–100) from lineup dict.

    Weights: QB 35%, WR1 20%, WR2 8%, RB1 15%, TE1 10%, OL (avg) 12%
    """
    qb  = lineup.get('qb_score',  65.0)
    wr1 = lineup.get('wr1_score', 65.0)
    wr2 = lineup.get('wr2_score', 52.0)
    rb1 = lineup.get('rb1_score', 65.0)
    te1 = lineup.get('te1_score', 60.0)
    # OL average (if available)
    ol_scores = [lineup.get(f'{p}_score', 65.0) for p in ['lt', 'lg', 'c', 'rg', 'rt']]
    ol_avg = sum(ol_scores) / len(ol_scores)

    return (qb * 0.35 + wr1 * 0.20 + wr2 * 0.08 + rb1 * 0.15 +
            te1 * 0.10 + ol_avg * 0.12)


# ── Depth chart player list helpers ──────────────────────────────────────────

def get_players_for_position(depth_chart: dict, pos_label: str) -> list[str]:
    """
    Return an ordered list of player names for a display position label.

    Handles multi-depth labels like WR1/WR2/WR3 by slicing from the WR list.
    """
    pos_alias, depth_map = _resolve_position(pos_label)
    players = depth_chart.get(pos_alias, [])
    if not players:
        return [f'{pos_label} (Not Available)']

    players_sorted = sorted(players, key=lambda p: p.get('depth', 99))
    idx = depth_map - 1  # 0-based

    # The starter is the expected "slot" index; return all remaining as alternatives
    if idx < len(players_sorted):
        ordered = [players_sorted[idx]] + [p for i, p in enumerate(players_sorted) if i != idx]
    else:
        ordered = players_sorted

    names = [p['name'] for p in ordered if p.get('name')]
    return names if names else [f'{pos_label} (Not Available)']


def get_starter_for_position(depth_chart: dict, pos_label: str) -> str:
    """Return the projected starter's name for a position label."""
    players = get_players_for_position(depth_chart, pos_label)
    return players[0] if players else 'Unknown'


def _resolve_position(pos_label: str) -> tuple[str, int]:
    """
    Map a display position label to (depth_chart_key, depth_slot).

    WR1 → ('WR', 1),  WR2 → ('WR', 2),  DE1 → ('DE', 1), etc.
    OL positions map to themselves with depth 1.
    """
    label_map = {
        'QB':  ('QB',   1), 'WR1': ('WR', 1), 'WR2': ('WR', 2),
        'WR3': ('WR',   3), 'WR4': ('WR', 4),
        'RB1': ('RB',   1), 'RB2': ('RB', 2),
        'TE1': ('TE',   1), 'TE2': ('TE', 2),
        'LT':  ('LT',   1), 'LG':  ('LG',  1), 'C':  ('C',   1),
        'RG':  ('RG',   1), 'RT':  ('RT',  1),
        'DE1': ('DE',   1), 'DE2': ('DE',  2),
        'DT':  ('DT',   1),
        'LB1': ('LB',   1), 'LB2': ('LB',  2), 'LB3': ('LB', 3),
        'CB1': ('CB',   1), 'CB2': ('CB',  2),
        'FS':  ('FS',   1), 'SS':  ('SS',  1),
    }
    if pos_label in label_map:
        return label_map[pos_label]

    # Try to extract numeric suffix
    import re
    m = re.match(r'([A-Z]+)(\d+)$', pos_label)
    if m:
        return m.group(1), int(m.group(2))

    return pos_label, 1


def _depth_to_score(depth: int) -> float:
    """Convert depth-chart slot to a rough player quality score (0–100)."""
    return {1: 65.0, 2: 52.0, 3: 42.0, 4: 35.0}.get(min(depth, 4), 30.0)
