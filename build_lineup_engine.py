# ================================================
# NFL PREDICTOR - Lineup Engine v3 (fixed)
# ================================================

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🔨 Building lineup engine...\n")

# ── Load data ─────────────────────────────────────
depth  = pd.read_csv('depth_charts.csv')
stats  = pd.read_csv('player_season_stats.csv')

# ── Step 1: Filter to offense only, real positions
offense_pos = ['QB', 'WR', 'RB', 'HB', 'TE', 'FB']
starters = depth[
    (depth['formation'] == 'Offense') &
    (depth['depth_team'] <= 3) &
    (depth['depth_position'].isin(offense_pos))
].copy()

# Get most recent week per team + position
starters = starters.sort_values(['week', 'depth_team'], ascending=[False, True])
starters = starters.drop_duplicates(subset=['club_code', 'depth_position', 'full_name'])

# Normalize HB -> RB
starters['depth_position'] = starters['depth_position'].replace('HB', 'RB')

# Clean full name
starters['full_name'] = starters['first_name'] + ' ' + starters['last_name']

print(f"✅ Found {len(starters)} starters across skill positions\n")

# ── Step 2: Merge with season stats ──────────────
merged = starters.merge(
    stats,
    left_on='gsis_id',
    right_on='player_id',
    how='left'
)

# Fill missing numeric stats with 0
stat_cols = [
    'passing_yards','passing_tds','interceptions','passing_epa',
    'rushing_yards','rushing_tds','rushing_epa',
    'receiving_yards','receiving_tds','receiving_epa',
    'target_share','games'
]
for col in stat_cols:
    if col in merged.columns:
        merged[col] = merged[col].fillna(0)

merged['games'] = merged['games'].replace(0, 1)

# ── Step 3: Player strength scores ───────────────
def qb_score(row):
    yards  = row.get('passing_yards', 0)
    tds    = row.get('passing_tds', 0)
    ints   = row.get('interceptions', 0)
    epa    = row.get('passing_epa', 0)
    g      = max(row.get('games', 1), 1)
    # Each component already produces 0-40, 0-30 etc. No extra *100
    score = (yards/g/300*40) + (tds/g/2*30) + (epa/g/8*20) - (ints/g*10)
    return round(max(0, min(100, score)), 1)

def rb_score(row):
    yards = row.get('rushing_yards', 0)
    tds   = row.get('rushing_tds', 0)
    rec   = row.get('receiving_yards', 0)
    epa   = row.get('rushing_epa', 0)
    g     = max(row.get('games', 1), 1)
    score = (yards/g/80*40) + (tds/g/0.8*30) + (rec/g/30*20) + (epa/g/3*10)
    return round(max(0, min(100, score)), 1)

def wr_te_score(row):
    yards  = row.get('receiving_yards', 0)
    tds    = row.get('receiving_tds', 0)
    tgt_sh = row.get('target_share', 0)
    epa    = row.get('receiving_epa', 0)
    g      = max(row.get('games', 1), 1)
    score  = (yards/g/60*40) + (tds/g/0.5*25) + (tgt_sh*25) + (epa/g/3*10)
    return round(max(0, min(100, score)), 1)

def calc_score(row):
    pos = row['depth_position']
    if pos == 'QB': return qb_score(row)
    if pos == 'RB': return rb_score(row)
    return wr_te_score(row)

merged['player_score'] = merged.apply(calc_score, axis=1)

# ── Step 4: Print offensive rankings ─────────────
NFL_TEAMS = sorted([
    'ARI','ATL','BAL','BUF','CAR','CHI','CIN','CLE',
    'DAL','DEN','DET','GB', 'HOU','IND','JAX','KC',
    'LA', 'LAC','LV', 'MIA','MIN','NE', 'NO', 'NYG',
    'NYJ','PHI','PIT','SEA','SF', 'TB', 'TEN','WAS'
])

lineup_rows = []
for team in NFL_TEAMS:
    tm = merged[merged['club_code'] == team]
    qb  = tm[tm['depth_position'] == 'QB']
    rb  = tm[tm['depth_position'] == 'RB']
    wrs = tm[tm['depth_position'] == 'WR']
    te  = tm[tm['depth_position'] == 'TE']

    qb_s  = qb['player_score'].mean()  if len(qb)  > 0 else 50
    rb_s  = rb['player_score'].mean()  if len(rb)  > 0 else 50
    wr_s  = wrs['player_score'].mean() if len(wrs) > 0 else 50
    te_s  = te['player_score'].mean()  if len(te)  > 0 else 50

    off_score = qb_s*0.40 + wr_s*0.25 + rb_s*0.20 + te_s*0.15

    lineup_rows.append({
        'team':         team,
        'offense_score': round(off_score, 1),
        'qb_score':     qb_s,
        'rb_score':     rb_s,
        'wr_score':     wr_s,
        'te_score':     te_s,
        'qb_name':      qb['full_name'].iloc[0]  if len(qb)  > 0 else 'Unknown',
        'rb_name':      rb['full_name'].iloc[0]  if len(rb)  > 0 else 'Unknown',
        'wr1_name':     wrs['full_name'].iloc[0] if len(wrs) > 0 else 'Unknown',
        'wr2_name':     wrs['full_name'].iloc[1] if len(wrs) > 1 else 'Unknown',
        'wr3_name':     wrs['full_name'].iloc[2] if len(wrs) > 2 else 'Unknown',
        'te_name':      te['full_name'].iloc[0]  if len(te)  > 0 else 'Unknown',
    })

lineup_df = pd.DataFrame(lineup_rows).sort_values('offense_score', ascending=False)

print(f"🏆 NFL OFFENSIVE STRENGTH RANKINGS (2024):")
print(f"{'Rank':<5}{'Team':<5}{'Score':<7}{'QB':<22}{'RB':<22}{'WR1':<22}{'TE':<20}")
print("-"*100)
for i, row in enumerate(lineup_df.itertuples(), 1):
    print(f"{i:<5}{row.team:<5}{row.offense_score:<7}"
          f"{row.qb_name:<22}{row.rb_name:<22}{row.wr1_name:<22}{row.te_name:<20}")

# ── Step 5: Build player lookup for app dropdowns
print("\n🔨 Building player lookup for dropdowns...")

player_lookup = {}
for team in NFL_TEAMS:
    tm = merged[merged['club_code'] == team].copy()
    player_lookup[team] = {}

    for pos in ['QB', 'RB', 'WR', 'TE']:
        pos_players = tm[tm['depth_position'] == pos]\
            .sort_values('player_score', ascending=False)

        player_lookup[team][pos] = [
            {
                'name':  row['full_name'],
                'score': row['player_score'],
                'stats': {
                    'pass_yds': int(row.get('passing_yards',  0) or 0),
                    'rush_yds': int(row.get('rushing_yards',  0) or 0),
                    'rec_yds':  int(row.get('receiving_yards',0) or 0),
                    'tds':      int((row.get('passing_tds',   0) or 0) +
                                    (row.get('rushing_tds',   0) or 0) +
                                    (row.get('receiving_tds', 0) or 0)),
                    'games':    int(row.get('games', 1) or 1),
                }
            }
            for _, row in pos_players.iterrows()
        ]

# ── Step 6: Save everything ───────────────────────
lineup_df.to_csv('lineup_summary.csv', index=False)
merged.to_csv('starters_with_stats.csv', index=False)

with open('player_lookup.pkl', 'wb') as f:
    pickle.dump(player_lookup, f)

print(f"\n💾 All files saved!")
print(f"✅ Lineup engine ready! Now run: python final_app.py")