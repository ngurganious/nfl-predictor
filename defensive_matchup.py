"""
Defensive Matchup Engine
========================
Computes a position-by-position matchup quality score between an offense
and the defense they face. Produces a small win-probability adjustment
(≈ ±4%) that sits on top of the team-level EPA signal already in the model.

No retraining required — this is a post-model calibration step, similar in
spirit to the existing lineup_adjustment() in final_app.py.

Usage:
    from defensive_matchup import calc_matchup_adj
    adj, breakdown = calc_matchup_adj(home_off_lineup, away_def_chart,
                                       away_team, team_stats_df)
    # adj  ≈ -0.04 … +0.04
    # breakdown  = list of (label, off_score, def_score, edge_str)
"""

import numpy as np
import pandas as pd

# ── Matchup weight table ──────────────────────────────────────────────────────
# (def_group, off_group, weight)
# def_group  — key into _DEF_POS_MAP below
# off_group  — key into off_lineup dict (using lowercase score keys)
# weight     — must sum to 1.0

_MATCHUP_WEIGHTS = [
    ('pass_rush', 'qb_score',   0.28),  # DE/EDGE pressure vs QB
    ('cb_1',      'wr1_score',  0.22),  # CB1 coverage vs WR1
    ('cb_2',      'wr2_score',  0.14),  # CB2 coverage vs WR2
    ('lb_run',    'rb1_score',  0.14),  # LB+DT run-stop vs RB
    ('lb_cov',    'te1_score',  0.12),  # LB zone coverage vs TE
    ('safety',    'wr3_score',  0.10),  # Safety vs slot / WR3
]

# Depth-chart position aliases for each defensive group
_DEF_POS_MAP = {
    'pass_rush': ['DE', 'EDGE', 'OLB', 'LOLB', 'ROLB', 'RDE', 'LDE'],
    'cb_1':      ['CB'],
    'cb_2':      ['CB'],
    'lb_run':    ['LB', 'MLB', 'ILB', 'WLB', 'SLB', 'DT', 'NT', 'IDL'],
    'lb_cov':    ['LB', 'MLB', 'ILB', 'WLB', 'SLB'],
    'safety':    ['S', 'FS', 'SS'],
}

# Human-readable labels for the UI breakdown table
_MATCHUP_LABELS = {
    'pass_rush': 'Pass Rush vs QB',
    'cb_1':      'CB1 Coverage vs WR1',
    'cb_2':      'CB2 Coverage vs WR2',
    'lb_run':    'DL/LB Run-Stop vs RB',
    'lb_cov':    'LB Zone Coverage vs TE',
    'safety':    'Safety vs Slot/WR3',
}


# ── Public API ────────────────────────────────────────────────────────────────

def calc_matchup_adj(
    off_lineup: dict,
    def_chart: dict,
    def_team: str,
    team_stats: pd.DataFrame,
) -> tuple:
    """
    Compute a win-probability adjustment from individual position matchups.

    Args:
        off_lineup:  offense lineup scores dict from build_lineup_dict()
                     Keys like 'qb_score', 'wr1_score', 'rb1_score', etc.
        def_chart:   depth chart dict for the defending team,
                     from get_team_full_depth_chart()
        def_team:    defending team abbreviation (for EPA fallback)
        team_stats:  team_stats_current.csv loaded as DataFrame (index=team)

    Returns:
        (adj, breakdown) where:
          adj       — float ≈ -0.04 to +0.04 added to win probability
                      positive = offense has matchup advantage
          breakdown — list of (label, off_score, def_score, edge_pts, pct_adv)
                      for display in the UI
    """
    total_weighted_edge = 0.0
    breakdown = []

    for def_group, off_key, weight in _MATCHUP_WEIGHTS:
        off_s = float(off_lineup.get(off_key, 55.0))
        def_s = _get_def_group_score(def_chart, def_group, def_team, team_stats)

        # Positive edge → offense advantage (more points = QB beats pass rush, etc.)
        edge = off_s - def_s  # range roughly -40 to +40 pts
        # Normalise to [-1, +1] using a scale of ±30 pts as "significant"
        norm_edge = float(np.clip(edge / 30.0, -1.0, 1.0))
        total_weighted_edge += norm_edge * weight

        # Percentage advantage label for display
        if abs(edge) < 3:
            adv = "even"
        elif edge > 0:
            adv = f"+{edge:.0f} OFF"
        else:
            adv = f"{edge:.0f} DEF"

        breakdown.append((
            _MATCHUP_LABELS.get(def_group, def_group),
            round(off_s),
            round(def_s),
            round(edge),
            adv,
        ))

    # Scale to win-probability space: ±4% maximum
    adj = float(total_weighted_edge * 0.04)
    return adj, breakdown


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_def_group_score(
    def_chart: dict,
    group: str,
    team: str,
    team_stats: pd.DataFrame,
) -> float:
    """
    Compute an aggregate quality score (0–100) for a defensive position group.

    Blends individual depth-chart player scores (50%) with the team's
    overall defensive EPA quality (50%) to avoid over-fitting on sparse
    individual-player data.
    """
    position_aliases = _DEF_POS_MAP.get(group, [])
    players = _collect_players(def_chart, position_aliases)
    team_score = _team_def_epa_to_score(team, team_stats)

    if not players:
        return team_score

    players_sorted = sorted(players, key=lambda p: p.get('depth', 99))

    # Choose the relevant depth slot(s) per group
    if group == 'cb_2':
        # Second corner — take index 1 if available
        relevant = players_sorted[1:2] if len(players_sorted) > 1 else players_sorted[:1]
    elif group in ('pass_rush', 'lb_run'):
        # Average of top-2 (or top-1 if only one)
        relevant = players_sorted[:2]
    elif group == 'lb_cov':
        relevant = players_sorted[:2]
    elif group == 'safety':
        relevant = players_sorted[:2]
    else:
        relevant = players_sorted[:1]

    if not relevant:
        return team_score

    individual_score = float(np.mean([p.get('score', 65.0) for p in relevant]))
    # 50 / 50 blend: individual data is noisy, team EPA is more reliable
    return 0.50 * individual_score + 0.50 * team_score


def _collect_players(def_chart: dict, position_aliases: list) -> list:
    """
    Return all players whose position key matches any of the given aliases.
    Case-insensitive prefix match so 'LB' matches 'LB', 'MLB', 'ILB', etc.
    """
    collected = []
    aliases_upper = [a.upper() for a in position_aliases]
    for pos_key, players in def_chart.items():
        k = pos_key.upper()
        if k in aliases_upper or any(k.startswith(a) for a in aliases_upper):
            collected.extend(players)
    return collected


def _team_def_epa_to_score(team: str, team_stats: pd.DataFrame) -> float:
    """
    Convert a team's defensive EPA per play into a 0–100 quality score.

    Convention in our data:
        negative def_epa  → fewer points allowed per play → GOOD defense
        positive def_epa  → more points allowed per play → BAD defense

    Mapping (approximate):
        -0.15 EPA/play  → 82  (elite)
         0.00 EPA/play  → 65  (average)
        +0.10 EPA/play  → 55  (poor)
    """
    if team_stats is None or (hasattr(team_stats, 'empty') and team_stats.empty):
        return 65.0
    if team not in team_stats.index:
        return 65.0

    try:
        def_epa = float(team_stats.loc[team].get('def_epa_per_play', 0.0))
    except Exception:
        return 65.0

    # Linear mapping: 65 - (def_epa * 113) clipped to [30, 90]
    score = 65.0 - def_epa * 113.0
    return float(np.clip(score, 30.0, 90.0))


def format_breakdown_table(breakdown: list) -> pd.DataFrame:
    """
    Convert calc_matchup_adj() breakdown list into a display DataFrame.

    Columns: Matchup | Off Score | Def Score | Edge | Advantage
    """
    rows = []
    for label, off_s, def_s, edge, adv in breakdown:
        rows.append({
            'Matchup':   label,
            'Offense':   off_s,
            'Defense':   def_s,
            'Edge (pts)': f'{edge:+d}',
            'Advantage': adv,
        })
    return pd.DataFrame(rows)
