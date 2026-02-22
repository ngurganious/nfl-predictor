# ================================================
# NFL PREDICTOR - Final App with Lineup Builder
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="NFL Predictor Pro",
    page_icon="🏈",
    layout="wide"
)

# ── Load everything ───────────────────────────────
@st.cache_resource
def load_all():
    with open('model.pkl', 'rb') as f:
        game_model = pickle.load(f)
    with open('elo_ratings.pkl', 'rb') as f:
        elo = pickle.load(f)
    with open('pass_yards_model.pkl', 'rb') as f:
        pass_model = pickle.load(f)
    with open('rush_yards_model.pkl', 'rb') as f:
        rush_model = pickle.load(f)
    with open('rec_yards_model.pkl', 'rb') as f:
        rec_model = pickle.load(f)
    with open('player_lookup.pkl', 'rb') as f:
        players = pickle.load(f)
    return game_model, elo, pass_model, rush_model, rec_model, players

@st.cache_data
def load_data():
    games     = pd.read_csv('games_processed.csv')
    passing   = pd.read_csv('passing_stats.csv')
    rushing   = pd.read_csv('rushing_stats.csv')
    receiving = pd.read_csv('receiving_stats.csv')
    lineup_df = pd.read_csv('lineup_summary.csv')
    return games, passing, rushing, receiving, lineup_df

game_model, elo_ratings, pass_model, rush_model, \
    rec_model, player_lookup = load_all()
games, passing, rushing, receiving, lineup_df = load_data()

NFL_TEAMS = sorted([
    'ARI','ATL','BAL','BUF','CAR','CHI','CIN','CLE',
    'DAL','DEN','DET','GB', 'HOU','IND','JAX','KC',
    'LA', 'LAC','LV', 'MIA','MIN','NE', 'NO', 'NYG',
    'NYJ','PHI','PIT','SEA','SF', 'TB', 'TEN','WAS'
])

# ── Helpers ───────────────────────────────────────
def get_elo(team):
    return elo_ratings.get(team, 1500)

def elo_win_prob(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def get_starters(team):
    """Get default starters for a team from player lookup"""
    result = {}
    for pos in ['QB','RB','WR','TE']:
        players = player_lookup.get(team, {}).get(pos, [])
        result[pos] = players[0]['name'] if players else 'Unknown'
        result[f'{pos}_list'] = [p['name'] for p in players] if players else ['Unknown']
        result[f'{pos}_score'] = players[0]['score'] if players else 50.0
    return result

def calc_lineup_score(team, qb, rb, wr, te):
    """
    Calculate offensive strength score for a given lineup.
    Looks up each player's individual score and weights by position.
    """
    def find_score(pos, name):
        for p in player_lookup.get(team, {}).get(pos, []):
            if p['name'] == name:
                return p['score']
        return 50.0  # Default if player not found

    qb_s = find_score('QB', qb)
    rb_s = find_score('RB', rb)
    wr_s = find_score('WR', wr)
    te_s = find_score('TE', te)

    # Weighted offensive score
    score = qb_s*0.40 + wr_s*0.25 + rb_s*0.20 + te_s*0.15
    return score, qb_s, rb_s, wr_s, te_s

def lineup_adjustment(home_score, away_score):
    """
    Convert lineup scores into a win probability adjustment.
    A stronger lineup shifts the prediction in their favor.
    Difference of 10 points ~ 5% swing in win probability.
    """
    diff = home_score - away_score
    adjustment = diff * 0.005  # Scale factor
    return adjustment

def get_player_recent(df, name_col, name, stat_cols, n=4):
    """Get player's rolling average from last n games"""
    rows = df[df[name_col] == name].sort_values(['season','week']).tail(n)
    if len(rows) == 0:
        return None
    result = {}
    for col in stat_cols:
        avg_col = f'avg_{col}_l4'
        if avg_col in rows.columns:
            result[avg_col] = rows[avg_col].iloc[-1]
    return result

# ── Header ────────────────────────────────────────
st.title("🏈 NFL Predictor Pro")
st.markdown("*25 years of data • ELO ratings • Lineup-adjusted ML predictions • Player props*")
st.divider()

# ── Sidebar ───────────────────────────────────────
with st.sidebar:
    st.header("📊 ELO Power Rankings")
    elo_df = pd.Series(elo_ratings).reset_index()
    elo_df.columns = ['Team','ELO']
    elo_df = elo_df[elo_df['Team'].isin(NFL_TEAMS)]\
        .sort_values('ELO', ascending=False).reset_index(drop=True)
    elo_df.index += 1
    elo_df['ELO'] = elo_df['ELO'].round(0).astype(int)
    st.dataframe(elo_df, use_container_width=True)

# ── Tabs ──────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Game Predictor + Lineups",
    "🏃 Player Props",
    "📈 Head-to-Head",
    "🏆 Super Bowl Predictor"
])

# ══════════════════════════════════════════════════
# TAB 1: GAME PREDICTOR WITH LINEUPS
# ══════════════════════════════════════════════════
with tab1:
    st.header("🎯 Game Predictor")
    st.caption("Pre-populated with likely starters — swap anyone out to see how it affects the prediction")
    st.divider()

    # ── Team + conditions row ─────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("🏠 Home Team")
        home_team  = st.selectbox("Home Team", NFL_TEAMS, index=NFL_TEAMS.index('KC'))
        home_rest  = st.slider("Days of Rest", 3, 14, 7, key='hr')
        spread     = st.slider("Vegas Spread (neg = home favored)", -28.0, 28.0, -3.0, 0.5)

    with c2:
        st.subheader("✈️ Away Team")
        away_team  = st.selectbox("Away Team", NFL_TEAMS, index=NFL_TEAMS.index('BUF'))
        away_rest  = st.slider("Days of Rest", 3, 14, 7, key='ar')
        is_div     = st.checkbox("Divisional Game?")

    with c3:
        st.subheader("🌤️ Conditions")
        roof       = st.selectbox("Stadium", ['outdoors','dome','closed','open'])
        is_dome    = 1 if roof == 'dome' else 0
        outdoor_weather = roof in ['outdoors', 'open']
        temp       = st.slider("Temp (°F)", 20, 100, 65) if outdoor_weather else 72
        wind       = st.slider("Wind (mph)", 0, 40, 5)   if outdoor_weather else 0
        surface    = st.selectbox("Surface", ['grass','turf'])
        is_grass   = 1 if surface == 'grass' else 0

    st.divider()

    # ── Lineup Builder ────────────────────────────
    st.subheader("👥 Starting Lineups")
    st.caption("Auto-populated with projected starters — change any player to see the impact")

    home_starters = get_starters(home_team)
    away_starters = get_starters(away_team)

    lu_col1, lu_spacer, lu_col2 = st.columns([5, 1, 5])

    with lu_col1:
        st.markdown(f"**🏠 {home_team} Offense**")

        home_qb = st.selectbox(
            "QB", home_starters['QB_list'],
            index=0, key='hqb'
        )
        home_rb = st.selectbox(
            "RB", home_starters['RB_list'],
            index=0, key='hrb'
        )
        home_wr = st.selectbox(
            "WR1", home_starters['WR_list'],
            index=0, key='hwr'
        )
        home_te = st.selectbox(
            "TE", home_starters['TE_list'],
            index=0, key='hte'
        )

    with lu_spacer:
        st.markdown("<br><br><br><br><br><br>**VS**", unsafe_allow_html=True)

    with lu_col2:
        st.markdown(f"**✈️ {away_team} Offense**")
        away_starters = get_starters(away_team)
        away_qb = st.selectbox("QB", away_starters['QB_list'], index=0, key='aqb')
        away_rb = st.selectbox("RB", away_starters['RB_list'], index=0, key='arb')
        away_wr = st.selectbox("WR1", away_starters['WR_list'], index=0, key='awr')
        away_te = st.selectbox("TE", away_starters['TE_list'], index=0, key='ate')

    st.divider()

    # ── Predict Button ────────────────────────────
    if st.button("🔮 Predict Game", type="primary", use_container_width=True):

        # Base model prediction
        home_elo  = get_elo(home_team)
        away_elo  = get_elo(away_team)
        elo_diff  = home_elo - away_elo

        features = pd.DataFrame([{
            'elo_diff':    elo_diff,
            'spread_line': spread,
            'home_rest':   home_rest,
            'away_rest':   away_rest,
            'temp':        temp,
            'wind':        wind,
            'is_dome':     is_dome,
            'is_grass':    is_grass,
            'div_game':    int(is_div),
        }])

        base_prob_home = game_model.predict_proba(features)[0][1]

        # Lineup adjustment
        home_off, home_qb_s, home_rb_s, home_wr_s, home_te_s = \
            calc_lineup_score(home_team, home_qb, home_rb, home_wr, home_te)
        away_off, away_qb_s, away_rb_s, away_wr_s, away_te_s = \
            calc_lineup_score(away_team, away_qb, away_rb, away_wr, away_te)

        adj = lineup_adjustment(home_off, away_off)

        # Final probability (clipped to 5%-95%)
        final_prob_home = float(np.clip(base_prob_home + adj, 0.05, 0.95))
        final_prob_away = 1 - final_prob_home
        elo_prob        = elo_win_prob(home_elo, away_elo)

        # ── Results ───────────────────────────────
        st.subheader("📊 Prediction Results")

        r1, r2 = st.columns(2)
        with r1:
            st.metric(
                f"🏠 {home_team} Win Probability",
                f"{final_prob_home*100:.1f}%",
                delta=f"{(final_prob_home - base_prob_home)*100:+.1f}% lineup adj"
            )
            st.progress(final_prob_home)
        with r2:
            st.metric(
                f"✈️ {away_team} Win Probability",
                f"{final_prob_away*100:.1f}%",
                delta=f"ELO baseline: {(1-elo_prob)*100:.1f}%"
            )
            st.progress(final_prob_away)

        winner     = home_team if final_prob_home > 0.5 else away_team
        confidence = max(final_prob_home, final_prob_away)
        label = ("🔥 HIGH CONFIDENCE" if confidence > 0.70
                 else "✅ MODERATE CONFIDENCE" if confidence > 0.60
                 else "⚠️ TOSS-UP")

        st.subheader(f"{label}: **{winner}** wins")

        # ── Lineup Breakdown ──────────────────────
        with st.expander("👥 Lineup Strength Breakdown"):
            lc1, lc2 = st.columns(2)

            with lc1:
                st.markdown(f"**🏠 {home_team} Offense: {home_off:.1f}/100**")
                st.write(f"QB  {home_qb} — {home_qb_s:.0f}/100")
                st.write(f"RB  {home_rb} — {home_rb_s:.0f}/100")
                st.write(f"WR  {home_wr} — {home_wr_s:.0f}/100")
                st.write(f"TE  {home_te} — {home_te_s:.0f}/100")

            with lc2:
                st.markdown(f"**✈️ {away_team} Offense: {away_off:.1f}/100**")
                st.write(f"QB  {away_qb} — {away_qb_s:.0f}/100")
                st.write(f"RB  {away_rb} — {away_rb_s:.0f}/100")
                st.write(f"WR  {away_wr} — {away_wr_s:.0f}/100")
                st.write(f"TE  {away_te} — {away_te_s:.0f}/100")

            st.caption(f"Lineup adjustment: {adj*100:+.1f}% → "
                       f"Base: {base_prob_home*100:.1f}% → "
                       f"Final: {final_prob_home*100:.1f}%")

        # ── Weather impact note ───────────────────
        if wind >= 20:
            st.warning(f"💨 High wind ({wind} mph) — expect lower scoring, "
                       f"passing stats suppressed")
        if temp < 32:
            st.warning(f"❄️ Freezing temps ({temp}°F) — historically reduces "
                       f"total scoring by ~4 pts")
        if is_dome:
            st.info("🏟️ Dome game — weather neutralized, "
                    "passing games typically boosted")

# ══════════════════════════════════════════════════
# TAB 2: PLAYER PROPS
# ══════════════════════════════════════════════════
with tab2:
    st.header("🏃 Player Prop Predictions")
    st.caption("Based on recent form (last 4 games) + game conditions")

    pc1, pc2 = st.columns(2)

    with pc1:
        prop_type = st.selectbox("Stat to Predict",
            ["Passing Yards","Rushing Yards","Receiving Yards"])

        if prop_type == "Passing Yards":
            all_players = sorted(passing['passer_player_name'].dropna().unique())
            default = 'P.Mahomes' if 'P.Mahomes' in all_players else all_players[0]
        elif prop_type == "Rushing Yards":
            all_players = sorted(rushing['rusher_player_name'].dropna().unique())
            default = 'D.Henry' if 'D.Henry' in all_players else all_players[0]
        else:
            all_players = sorted(receiving['receiver_player_name'].dropna().unique())
            default = 'T.Hill' if 'T.Hill' in all_players else all_players[0]

        player = st.selectbox("Player",
            all_players,
            index=all_players.index(default) if default in all_players else 0
        )

    with pc2:
        p_team   = st.selectbox("Player's Team", NFL_TEAMS, key='pt')
        opp      = st.selectbox("Opponent", NFL_TEAMS, index=1, key='opp')
        p_home   = st.checkbox("Home game?", value=True)
        p_roof   = st.selectbox("Stadium", ['outdoors','dome','closed','open'], key='pr')
        p_temp   = st.slider("Temp", 20, 100, 65, key='ptemp') if p_roof == 'outdoors' else 72
        p_wind   = st.slider("Wind", 0, 40, 5, key='pwind')   if p_roof == 'outdoors' else 0
        p_spread = st.slider("Spread", -28.0, 28.0, -3.0, key='pspread')

    if st.button("🔮 Predict Props", type="primary", use_container_width=True):
        p_dome  = 1 if p_roof == 'dome' else 0
        is_home = 1 if p_home else 0
        pred = None

        if prop_type == "Passing Yards":
            s = get_player_recent(passing, 'passer_player_name', player,
                ['pass_yards','pass_attempts','completions','pass_tds'])
            if s:
                f = pd.DataFrame([{
                    'avg_pass_yards_l4':    s.get('avg_pass_yards_l4', 220),
                    'avg_pass_attempts_l4': s.get('avg_pass_attempts_l4', 32),
                    'avg_completions_l4':   s.get('avg_completions_l4', 21),
                    'avg_pass_tds_l4':      s.get('avg_pass_tds_l4', 1.5),
                    'temp': p_temp, 'wind': p_wind,
                    'is_dome': p_dome, 'is_home': is_home,
                    'spread_line': p_spread
                }])
                pred, mae = pass_model.predict(f)[0], 63.6

        elif prop_type == "Rushing Yards":
            s = get_player_recent(rushing, 'rusher_player_name', player,
                ['rush_yards','rush_attempts','rush_tds'])
            if s:
                f = pd.DataFrame([{
                    'avg_rush_yards_l4':    s.get('avg_rush_yards_l4', 55),
                    'avg_rush_attempts_l4': s.get('avg_rush_attempts_l4', 14),
                    'avg_rush_tds_l4':      s.get('avg_rush_tds_l4', 0.4),
                    'temp': p_temp, 'wind': p_wind,
                    'is_dome': p_dome, 'is_home': is_home,
                    'spread_line': p_spread
                }])
                pred, mae = rush_model.predict(f)[0], 21.6

        else:
            s = get_player_recent(receiving, 'receiver_player_name', player,
                ['rec_yards','targets','receptions','rec_tds'])
            if s:
                f = pd.DataFrame([{
                    'avg_rec_yards_l4':   s.get('avg_rec_yards_l4', 50),
                    'avg_targets_l4':     s.get('avg_targets_l4', 6),
                    'avg_receptions_l4':  s.get('avg_receptions_l4', 4),
                    'avg_rec_tds_l4':     s.get('avg_rec_tds_l4', 0.3),
                    'temp': p_temp, 'wind': p_wind,
                    'is_dome': p_dome, 'is_home': is_home,
                    'spread_line': p_spread
                }])
                pred, mae = rec_model.predict(f)[0], 21.4

        st.divider()
        if pred and pred > 0:
            low  = max(0, pred - mae)
            high = pred + mae

            m1, m2, m3 = st.columns(3)
            m1.metric("📉 Low End",    f"{low:.0f} yds")
            m2.metric("🎯 Projection", f"{pred:.0f} yds")
            m3.metric("📈 High End",   f"{high:.0f} yds")
            st.progress(min(float(pred/(high+10)), 1.0))
            st.caption(f"MAE: ±{mae} yards")

            # Recent game log
            if prop_type == "Passing Yards":
                recent = passing[passing['passer_player_name'] == player]\
                    .sort_values(['season','week']).tail(5)\
                    [['season','week','posteam','defteam',
                      'pass_yards','pass_attempts','completions','pass_tds']]
            elif prop_type == "Rushing Yards":
                recent = rushing[rushing['rusher_player_name'] == player]\
                    .sort_values(['season','week']).tail(5)\
                    [['season','week','posteam','defteam',
                      'rush_yards','rush_attempts','rush_tds']]
            else:
                recent = receiving[receiving['receiver_player_name'] == player]\
                    .sort_values(['season','week']).tail(5)\
                    [['season','week','posteam','defteam',
                      'rec_yards','targets','receptions','rec_tds']]

            st.subheader(f"📋 {player}'s Last 5 Games")
            st.dataframe(recent.reset_index(drop=True), use_container_width=True)
        else:
            st.warning(f"Not enough recent data for {player}.")

# ══════════════════════════════════════════════════
# TAB 3: HEAD TO HEAD
# ══════════════════════════════════════════════════
with tab3:
    st.header("📈 Historical Head-to-Head")

    h1, h2 = st.columns(2)
    with h1:
        t1 = st.selectbox("Team A", NFL_TEAMS, key='t1')
    with h2:
        t2 = st.selectbox("Team B", NFL_TEAMS, index=1, key='t2')

    h2h = games[
        ((games['home_team'] == t1) & (games['away_team'] == t2)) |
        ((games['home_team'] == t2) & (games['away_team'] == t1))
    ].copy()

    if len(h2h) > 0:
        t1_wins = (
            ((h2h['home_team'] == t1) & (h2h['home_score'] > h2h['away_score'])) |
            ((h2h['away_team'] == t1) & (h2h['away_score'] > h2h['home_score']))
        ).sum()
        t2_wins = len(h2h) - t1_wins

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{t1} Wins", t1_wins)
        m2.metric(f"{t2} Wins", t2_wins)
        m3.metric("Total Games", len(h2h))
        m4.metric(f"{t1} Win %", f"{t1_wins/len(h2h)*100:.0f}%")

        st.dataframe(
            h2h[['season','week','home_team','home_score',
                  'away_score','away_team','temp','wind','roof']]
            .sort_values('season', ascending=False)
            .head(15)
            .reset_index(drop=True),
            use_container_width=True
        )
    else:
        st.info("No matchups found in dataset.")
# ══════════════════════════════════════════════════
# TAB 4: SUPER BOWL PREDICTOR
# ══════════════════════════════════════════════════
with tab4:
    st.header("🏆 Super Bowl Predictor")
    st.caption("Simulates the entire NFL playoff bracket using ELO + lineup strength")
    st.divider()

    st.subheader("🌱 Playoff Seeds")
    st.caption("Pre-loaded with 2024 playoff teams — adjust any seed if needed")

    seed_col1, seed_col2 = st.columns(2)

    with seed_col1:
        st.markdown("**🏈 AFC**")
        afc1 = st.selectbox("AFC #1 (Bye)", NFL_TEAMS, index=NFL_TEAMS.index('DEN'), key='afc1')
        afc2 = st.selectbox("AFC #2 (Bye)", NFL_TEAMS, index=NFL_TEAMS.index('NE'),  key='afc2')
        afc3 = st.selectbox("AFC #3",       NFL_TEAMS, index=NFL_TEAMS.index('JAX'), key='afc3')
        afc4 = st.selectbox("AFC #4",       NFL_TEAMS, index=NFL_TEAMS.index('PIT'), key='afc4')
        afc5 = st.selectbox("AFC #5",       NFL_TEAMS, index=NFL_TEAMS.index('HOU'), key='afc5')
        afc6 = st.selectbox("AFC #6",       NFL_TEAMS, index=NFL_TEAMS.index('BUF'), key='afc6')
        afc7 = st.selectbox("AFC #7",       NFL_TEAMS, index=NFL_TEAMS.index('LAC'), key='afc7')

    with seed_col2:
        st.markdown("**🏈 NFC**")
        nfc1 = st.selectbox("NFC #1 (Bye)", NFL_TEAMS, index=NFL_TEAMS.index('SEA'), key='nfc1')
        nfc2 = st.selectbox("NFC #2 (Bye)", NFL_TEAMS, index=NFL_TEAMS.index('CHI'), key='nfc2')
        nfc3 = st.selectbox("NFC #3",       NFL_TEAMS, index=NFL_TEAMS.index('PHI'), key='nfc3')
        nfc4 = st.selectbox("NFC #4",       NFL_TEAMS, index=NFL_TEAMS.index('CAR'), key='nfc4')
        nfc5 = st.selectbox("NFC #5",       NFL_TEAMS, index=NFL_TEAMS.index('LA'),  key='nfc5')
        nfc6 = st.selectbox("NFC #6",       NFL_TEAMS, index=NFL_TEAMS.index('SF'),  key='nfc6')
        nfc7 = st.selectbox("NFC #7",       NFL_TEAMS, index=NFL_TEAMS.index('GB'),  key='nfc7')

    st.divider()
    st.subheader("🏟️ Super Bowl Conditions")
    sb_c1, sb_c2, sb_c3 = st.columns(3)
    with sb_c1:
        sb_roof   = st.selectbox("Stadium", ['dome','outdoors','closed','open'], key='sbroof')
        sb_dome   = 1 if sb_roof == 'dome' else 0
    with sb_c2:
        sb_temp   = st.slider("Temp (°F)", 20, 100, 72, key='sbtemp') \
                    if sb_roof in ['outdoors','open'] else 72
    with sb_c3:
        sb_wind   = st.slider("Wind (mph)", 0, 40, 0, key='sbwind') \
                    if sb_roof in ['outdoors','open'] else 0

    st.divider()

    def predict_sb_game(home, away, neutral=False):
        home_elo  = get_elo(home)
        away_elo  = get_elo(away)
        elo_diff  = 0 if neutral else (home_elo - away_elo)

        hs = get_starters(home)
        as_ = get_starters(away)
        h_off, _, _, _, _ = calc_lineup_score(home, hs['QB'], hs['RB'], hs['WR'], hs['TE'])
        a_off, _, _, _, _ = calc_lineup_score(away, as_['QB'], as_['RB'], as_['WR'], as_['TE'])
        adj = lineup_adjustment(h_off, a_off)

        feats = pd.DataFrame([{
            'elo_diff':    elo_diff,
            'spread_line': -(elo_diff / 25),
            'home_rest':   7,
            'away_rest':   7,
            'temp':        sb_temp,
            'wind':        sb_wind,
            'is_dome':     sb_dome,
            'is_grass':    0,
            'div_game':    0,
        }])

        base = game_model.predict_proba(feats)[0][1]
        return float(np.clip(base + adj, 0.05, 0.95))

    def sim_conference(seeds):
        import itertools
        s1,s2,s3,s4,s5,s6,s7 = seeds
        results = {t: 0.0 for t in seeds}

        wc_matchups = [(s2,s7),(s3,s6),(s4,s5)]

        for outcomes in itertools.product([0,1],[0,1],[0,1]):
            path_prob = 1.0
            wc_winners = []

            for i,(hi,lo) in enumerate(wc_matchups):
                p = predict_sb_game(hi, lo)
                if outcomes[i] == 0:
                    wc_winners.append(hi)
                    path_prob *= p
                else:
                    wc_winners.append(lo)
                    path_prob *= (1 - p)

            # Sort by original seeding
            seed_order = {t:i for i,t in enumerate(seeds)}
            wc_winners.sort(key=lambda t: seed_order[t])

            # Divisional: 1 hosts worst WC winner, 2 hosts best
            div_pairs = [
                (s1, wc_winners[2]),
                (s2, wc_winners[0]),
            ]

            p_da = predict_sb_game(div_pairs[0][0], div_pairs[0][1])
            p_db = predict_sb_game(div_pairs[1][0], div_pairs[1][1])

            for wa, pa in [(div_pairs[0][0], p_da),(div_pairs[0][1], 1-p_da)]:
                for wb, pb in [(div_pairs[1][0], p_db),(div_pairs[1][1], 1-p_db)]:
                    div_prob = path_prob * pa * pb
                    p_champ  = predict_sb_game(wa, wb)
                    results[wa] += div_prob * p_champ
                    results[wb] += div_prob * (1 - p_champ)

        return results

    if st.button("🔮 Simulate Super Bowl", type="primary", use_container_width=True):
        with st.spinner("🏈 Simulating entire playoff bracket..."):
            afc_seeds = [afc1,afc2,afc3,afc4,afc5,afc6,afc7]
            nfc_seeds = [nfc1,nfc2,nfc3,nfc4,nfc5,nfc6,nfc7]
            afc_probs = sim_conference(afc_seeds)
            nfc_probs = sim_conference(nfc_seeds)

        st.divider()
        r1, r2 = st.columns(2)

        with r1:
            st.subheader("🏈 AFC — Conference Win Odds")
            for team, prob in sorted(afc_probs.items(), key=lambda x: x[1], reverse=True):
                bar = '█' * int(prob * 25)
                st.write(f"**{team}** {bar} {prob*100:.1f}%  "
                         f"*(ELO: {get_elo(team):.0f})*")

        with r2:
            st.subheader("🏈 NFC — Conference Win Odds")
            for team, prob in sorted(nfc_probs.items(), key=lambda x: x[1], reverse=True):
                bar = '█' * int(prob * 25)
                st.write(f"**{team}** {bar} {prob*100:.1f}%  "
                         f"*(ELO: {get_elo(team):.0f})*")

        st.divider()
        st.subheader("🏆 Super Bowl Win Probabilities")

        sb_probs = {}
        for afc_t, afc_p in afc_probs.items():
            for nfc_t, nfc_p in nfc_probs.items():
                match_prob  = afc_p * nfc_p
                sb_win      = predict_sb_game(afc_t, nfc_t, neutral=True)
                sb_probs[afc_t] = sb_probs.get(afc_t, 0) + match_prob * sb_win
                sb_probs[nfc_t] = sb_probs.get(nfc_t, 0) + match_prob * (1 - sb_win)

        sb_sorted = sorted(sb_probs.items(), key=lambda x: x[1], reverse=True)

        # Top 5 trophy display
        top5 = sb_sorted[:5]
        t_cols = st.columns(5)
        for i,(team,prob) in enumerate(top5):
            conf = "AFC" if team in afc_seeds else "NFC"
            t_cols[i].metric(
                f"#{i+1} {team}",
                f"{prob*100:.1f}%",
                delta=f"{conf} • ELO {get_elo(team):.0f}"
            )

        st.divider()

        # Predicted champion
        champion  = sb_sorted[0][0]
        runner_up = sb_sorted[1][0]
        ch_conf   = "AFC" if champion  in afc_seeds else "NFC"
        ru_conf   = "AFC" if runner_up in afc_seeds else "NFC"

        st.subheader("🎯 Model's Predicted Super Bowl")
        ch1, ch2 = st.columns(2)
        with ch1:
            st.metric("🏆 Predicted Champion", champion,
                      delta=f"{ch_conf} • {sb_probs[champion]*100:.1f}% SB win prob")
        with ch2:
            st.metric("🥈 Runner Up", runner_up,
                      delta=f"{ru_conf} • {sb_probs[runner_up]*100:.1f}% SB win prob")

        # Full table
        with st.expander("📊 Full odds — all 14 playoff teams"):
            sb_df = pd.DataFrame(sb_sorted, columns=['Team','Win Prob'])
            sb_df['Win Prob']    = (sb_df['Win Prob']*100).round(1).astype(str) + '%'
            sb_df['ELO']         = sb_df['Team'].apply(lambda t: f"{get_elo(t):.0f}")
            sb_df['Conference']  = sb_df['Team'].apply(
                lambda t: 'AFC' if t in afc_seeds else 'NFC'
            )
            st.dataframe(sb_df.reset_index(drop=True), use_container_width=True)