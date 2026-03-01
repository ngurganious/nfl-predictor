"""
Parlay Math Engine for EdgeIQ Recursive Parlay Ladder (RPL).

Standalone module — no Streamlit dependency. All pure math for:
  - American/decimal odds conversion
  - Combined parlay odds & probability
  - Dynamic tier optimization (Banker/Accelerator/Moonshot)
  - Break-even stake sizing
  - Correlation filter for same-game conflicts
"""

from collections import defaultdict


# ── Odds conversion ─────────────────────────────────────────────────

def american_to_decimal(american):
    if american >= 100:
        return 1.0 + american / 100.0
    elif american <= -100:
        return 1.0 + 100.0 / abs(american)
    return 2.0  # fallback for invalid odds


def decimal_to_american(decimal_odds):
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1.0) * 100)
    elif decimal_odds > 1.0:
        return round(-100.0 / (decimal_odds - 1.0))
    return -110  # fallback


def implied_probability(american):
    if american <= -100:
        return abs(american) / (abs(american) + 100.0)
    elif american >= 100:
        return 100.0 / (american + 100.0)
    return 0.5


# ── Parlay combinatorics ────────────────────────────────────────────

def combined_parlay_decimal(legs):
    product = 1.0
    for leg in legs:
        product *= american_to_decimal(leg.get('odds', -110))
    return product


def combined_parlay_american(legs):
    return decimal_to_american(combined_parlay_decimal(legs))


def combined_probability(legs):
    prob = 1.0
    for leg in legs:
        prob *= leg.get('confidence', 0.5)
    return prob


def parlay_payout(stake, legs):
    return stake * combined_parlay_decimal(legs)


def parlay_ev(stake, legs):
    payout = parlay_payout(stake, legs)
    prob = combined_probability(legs)
    return payout * prob - stake


# ── Tier optimization ────────────────────────────────────────────────

def optimize_tiers(legs, total_budget):
    """
    Dynamically size 4 parlay tiers from a ranked list of legs.

    Args:
        legs: list of leg dicts sorted by descending confidence.
              Each must have 'odds' (American) and 'confidence' (0-1).
        total_budget: total dollar amount allocated to the ladder.

    Returns:
        list of tier dicts, each with:
          name, emoji, subtitle, legs, combined_decimal, combined_prob
    """
    if len(legs) < 3:
        return []

    banker_n = _find_banker_size(legs, total_budget)
    remaining = legs[banker_n:]

    tiers = [{
        'name': 'The Safety',
        'emoji': '\U0001f3e6',
        'subtitle': 'Banker',
        'legs': legs[:banker_n],
    }]

    if len(remaining) == 0:
        pass  # only Banker
    elif len(remaining) <= 3:
        # Not enough for 3 more tiers — just one Moonshot
        tiers.append({
            'name': 'The Jackpot',
            'emoji': '\U0001f319',
            'subtitle': 'Moonshot',
            'legs': legs[:banker_n + len(remaining)],
        })
    elif len(remaining) <= 6:
        # Split into Accelerator + Moonshot
        mid = len(remaining) // 2
        tiers.append({
            'name': 'The Growth',
            'emoji': '\U0001f4c8',
            'subtitle': 'Accelerator',
            'legs': legs[:banker_n + mid],
        })
        tiers.append({
            'name': 'The Jackpot',
            'emoji': '\U0001f319',
            'subtitle': 'Moonshot',
            'legs': legs[:banker_n + len(remaining)],
        })
    else:
        # Full 4-tier ladder: split remaining into 3 roughly equal groups
        third = len(remaining) // 3
        cut1 = third
        cut2 = third * 2
        tiers.append({
            'name': 'The Growth',
            'emoji': '\U0001f4c8',
            'subtitle': 'Accelerator 1',
            'legs': legs[:banker_n + cut1],
        })
        tiers.append({
            'name': 'The Growth',
            'emoji': '\U0001f680',
            'subtitle': 'Accelerator 2',
            'legs': legs[:banker_n + cut2],
        })
        tiers.append({
            'name': 'The Jackpot',
            'emoji': '\U0001f319',
            'subtitle': 'Moonshot',
            'legs': legs[:banker_n + len(remaining)],
        })

    # Compute combined odds and probability for each tier
    for tier in tiers:
        tier['combined_decimal'] = combined_parlay_decimal(tier['legs'])
        tier['combined_prob'] = combined_probability(tier['legs'])
        tier['combined_american'] = decimal_to_american(tier['combined_decimal'])
        tier['n_legs'] = len(tier['legs'])

    return tiers


def _find_banker_size(legs, total_budget):
    """
    Find smallest N such that the Banker parlay (legs[0:N]) can pay out
    >= total_budget while keeping the Banker stake <= 75% of total_budget.

    Banker_stake = total_budget / decimal_odds.
    We need Banker_stake <= 0.75 * total_budget, i.e. decimal_odds >= 1.333.
    """
    for n in range(2, min(len(legs) + 1, 8)):
        tier_legs = legs[:n]
        dec = combined_parlay_decimal(tier_legs)
        banker_stake = total_budget / dec
        if banker_stake <= total_budget * 0.75:
            return n
    return min(3, len(legs))


# ── Stake sizing ─────────────────────────────────────────────────────

def compute_stakes(tiers, total_budget):
    """
    Compute stake per tier with the break-even constraint:
      Banker payout >= total_budget.

    Banker stake = total_budget / banker_decimal_odds.
    Remaining budget split evenly across other tiers.
    """
    if not tiers:
        return []

    banker_dec = tiers[0]['combined_decimal']
    banker_stake = total_budget / banker_dec
    remaining = total_budget - banker_stake
    other_count = max(len(tiers) - 1, 1)
    other_stake = remaining / other_count if len(tiers) > 1 else 0.0

    results = []
    for i, tier in enumerate(tiers):
        stake = banker_stake if i == 0 else other_stake
        payout = stake * tier['combined_decimal']
        results.append({
            **tier,
            'stake': round(stake, 2),
            'payout': round(payout, 2),
        })
    return results


# ── Correlation filter ───────────────────────────────────────────────

def check_correlations(legs):
    """
    Scan legs for same-game conflicts.

    Returns list of flag dicts: {leg_indices, conflict_type, message, severity}.
    severity: 'block' or 'warn'.
    """
    flags = []
    by_game = defaultdict(list)
    for i, leg in enumerate(legs):
        gid = leg.get('game_id') or leg.get('game_label', '')
        by_game[gid].append((i, leg))

    for game_id, game_legs in by_game.items():
        if len(game_legs) < 2:
            continue

        # Rule 1: Two "under" props from same game
        unders = [(i, l) for i, l in game_legs
                  if l.get('direction', '').upper() == 'UNDER'
                  and l.get('bet_type') == 'prop']
        if len(unders) >= 2:
            flags.append({
                'leg_indices': [i for i, _ in unders],
                'conflict_type': 'same_game_double_under',
                'message': f"Two UNDER props in same game ({game_id}) — correlated risk",
                'severity': 'warn',
            })

        # Rule 2: Both QBs passing Over in same game
        pass_overs = [(i, l) for i, l in game_legs
                      if l.get('market') == 'player_pass_yds'
                      and l.get('direction', '').upper() == 'OVER']
        if len(pass_overs) >= 2:
            flags.append({
                'leg_indices': [i for i, _ in pass_overs],
                'conflict_type': 'opposing_qb_over',
                'message': f"Both QBs passing Over in same game ({game_id}) — positively correlated",
                'severity': 'warn',
            })

        # Rule 3: General same-game warning (2+ legs from one game)
        if len(game_legs) >= 2:
            flags.append({
                'leg_indices': [i for i, _ in game_legs],
                'conflict_type': 'same_game_general',
                'message': f"{len(game_legs)} legs from same game ({game_id}) — correlation risk",
                'severity': 'warn',
            })

    return flags
