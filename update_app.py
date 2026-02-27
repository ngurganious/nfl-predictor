# ================================================
# NFL PREDICTOR - App v2 with Player Props
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ──────────────────────────────────
st.set_page_config(
    page_title="NFL Predictor",
    page_icon="🏈",
    layout="wide"
)

# ── Load everything ───────────────────────────────
@st.cache_resource
def load_models():
    with open('model.pkl', 'rb') as f:
        game_model = pickle.load(f)
    with open('elo_ratings.pkl', 'rb') as f:
        elo_ratings = pickle.load(f)
    with open('pass_yards_model.pkl', 'rb') as f:
        pass_model = pickle.load(f)
    with open('rush_yards_model.pkl', 'rb') as f:
        rush_model = pickle.load(f)
    with open('rec_yards_model.pkl', 'rb') as f:
        rec_model = pickle.load(f)
    return game_model, elo_ratings, pass_model, rush_model, rec_model

@st.cache_data
def load_data():
    games    = pd.read_csv('games_processed.csv')
    passing  = pd.read_csv('passing_stats.csv')
    rushing  = pd.read_csv('rushing_stats.csv')
    receiving = pd.read_csv('receiving_stats.csv')
    return games, passing, rushing, receiving

game_model, elo_ratings, pass_model, rush_model, rec_model = load_models()
games, passing, rushing, receiving = load_data()

# ── Helpers ───────────────────────────────────────
def get_elo(team):
    return elo_ratings.get(team, 1500)

def expected_win_prob(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

NFL_TEAMS = sorted([
    'ARI','ATL','BAL','BUF','CAR','CHI','CIN','CLE',
    'DAL','DEN','DET','GB', 'HOU','IND','JAX','KC',
    'LA', 'LAC','LV', 'MIA','MIN','NE', 'NO', 'NYG',
    'NYJ','PHI','PIT','SEA','SF', 'TB', 'TEN','WAS'
])

def get_player_recent_stats(df, player_col, id_col, player_name, stat_cols):
    """Get a player's last 4 game rolling averages"""
    player_games = df[df[player_col] == player_name].sort_values(
        ['season','week']
    ).tail(4)
    if len(player_games) == 0:
        return None
    avg_cols = [f'avg_{c}_l4' for c in stat_cols]
    available = [c for c in avg_cols if c in player_games.columns]
    return player_games[available].iloc[-1] if available else None

# ── Header ────────────────────────────────────────
st.title("🏈 NFL Predictor Pro")
st.markdown("*25 years of data • ELO ratings • ML predictions • Player props*")
st.divider()

# ── Sidebar: ELO Rankings ─────────────────────────
with st.sidebar:
    st.header("📊 ELO Power Rankings")
    elo_df = pd.Series(elo_ratings).reset_index()
    elo_df.columns = ['Team', 'ELO']
    elo_df = elo_df[elo_df['Team'].isin(NFL_TEAMS)]
    elo_df = elo_df.sort_values('ELO', ascending=False).reset_index(drop=True)
    elo_df.index += 1
    elo_df['ELO'] = elo_df['ELO'].round(0).astype(int)
    st.dataframe(elo_df, use_container_width=True)

# ══════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🎯 Game Predictor", "🏃 Player Props", "📈 Head-to-Head"])

# ──────────────────────────────────────────────────
# TAB 1: GAME PREDICTOR
# ──────────────────────────────────────────────────
with tab1:
    st.header("Game Outcome Predictor")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏠 Home Team")
        home_team = st.selectbox("Home Team", NFL_TEAMS, index=NFL_TEAMS.index('KC'))
        home_rest = st.slider("Days of Rest", 3, 14, 7, key='hr')
        spread    = st.slider("Vegas Spread (neg = home favored)", -28.0, 28.0, -3.0, 0.5)

    with col2:
        st.subheader("✈️ Away Team")
        away_team = st.selectbox("Away Team", NFL_TEAMS, index=NFL_TEAMS.index('BUF'))
        away_rest = st.slider("Days of Rest", 3, 14, 7, key='ar')
        is_div    = st.checkbox("Divisional Game?")

    with col3:
        st.subheader("🌤️ Conditions")
        roof      = st.selectbox("Stadium", ['outdoors','dome','closed','open'])
        is_dome   = 1 if roof == 'dome' else 0
        temp      = st.slider("Temperature (°F)", 20, 100, 65) if roof == 'outdoors' else 72
        wind      = st.slider("Wind (mph)", 0, 40, 5) if roof == 'outdoors' else 0
        surface   = st.selectbox("Surface", ['grass','turf'])
        is_grass  = 1 if surface == 'grass' else 0

    if st.button("🔮 Predict Game", type="primary", use_container_width=True):
        home_elo = get_elo(home_team)
        away_elo = get_elo(away_team)
        elo_diff = home_elo - away_elo

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

        prob_home = game_model.predict_proba(features)[0][1]
        prob_away = 1 - prob_home
        elo_prob  = expected_win_prob(home_elo, away_elo)

        st.divider()
        r1, r2 = st.columns(2)
        with r1:
            st.metric(f"🏠 {home_team}", f"{prob_home*100:.1f}%",
                      delta=f"ELO: {elo_prob*100:.1f}%")
            st.progress(float(prob_home))
        with r2:
            st.metric(f"✈️ {away_team}", f"{prob_away*100:.1f}%",
                      delta=f"ELO: {(1-elo_prob)*100:.1f}%")
            st.progress(float(prob_away))

        winner     = home_team if prob_home > 0.5 else away_team
        confidence = max(prob_home, prob_away)
        label      = ("🔥 HIGH CONFIDENCE" if confidence > 0.70
                      else "✅ MODERATE CONFIDENCE" if confidence > 0.60
                      else "⚠️ TOSS-UP")
        st.subheader(f"{label}: {winner} wins")

        with st.expander("📊 Prediction breakdown"):
            st.write(f"**{home_team} ELO:** {home_elo:.0f} | "
                     f"**{away_team} ELO:** {away_elo:.0f} | "
                     f"**Diff:** {elo_diff:.0f}")
            st.write(f"**Spread:** {spread} | **Temp:** {temp}°F | "
                     f"**Wind:** {wind}mph | **Dome:** {'Yes' if is_dome else 'No'}")

# ──────────────────────────────────────────────────
# TAB 2: PLAYER PROPS
# ──────────────────────────────────────────────────
with tab2:
    st.header("Player Prop Predictions")
    st.caption("Based on recent form (last 4 games) + game conditions")

    prop_col1, prop_col2 = st.columns(2)

    with prop_col1:
        prop_type = st.selectbox("Stat to Predict",
            ["Passing Yards", "Rushing Yards", "Receiving Yards"])
        
        # Dynamic player list based on prop type
        if prop_type == "Passing Yards":
            players = sorted(passing['passer_player_name'].dropna().unique())
            default = 'P.Mahomes' if 'P.Mahomes' in players else players[0]
        elif prop_type == "Rushing Yards":
            players = sorted(rushing['rusher_player_name'].dropna().unique())
            default = 'D.Henry' if 'D.Henry' in players else players[0]
        else:
            players = sorted(receiving['receiver_player_name'].dropna().unique())
            default = 'T.Hill' if 'T.Hill' in players else players[0]

        player = st.selectbox("Select Player", players,
                              index=players.index(default) if default in players else 0)

    with prop_col2:
        p_team    = st.selectbox("Player's Team", NFL_TEAMS, key='pt')
        opp_team  = st.selectbox("Opponent", NFL_TEAMS, index=1, key='opp')
        p_home    = st.checkbox("Home game?", value=True)
        p_roof    = st.selectbox("Stadium", ['outdoors','dome','closed','open'], key='pr')
        p_temp    = st.slider("Temp (°F)", 20, 100, 65, key='pt2') if p_roof == 'outdoors' else 72
        p_wind    = st.slider("Wind (mph)", 0, 40, 5, key='pw')    if p_roof == 'outdoors' else 0
        p_spread  = st.slider("Game Spread", -28.0, 28.0, -3.0, key='ps')

    if st.button("🔮 Predict Player Props", type="primary", use_container_width=True):

        p_dome  = 1 if p_roof == 'dome' else 0
        is_home = 1 if p_home else 0

        if prop_type == "Passing Yards":
            stats = get_player_recent_stats(
                passing, 'passer_player_name', 'passer_player_id',
                player, ['pass_yards','pass_attempts','completions','pass_tds']
            )
            if stats is not None:
                feats = pd.DataFrame([{
                    'avg_pass_yards_l4':    stats.get('avg_pass_yards_l4', 220),
                    'avg_pass_attempts_l4': stats.get('avg_pass_attempts_l4', 32),
                    'avg_completions_l4':   stats.get('avg_completions_l4', 21),
                    'avg_pass_tds_l4':      stats.get('avg_pass_tds_l4', 1.5),
                    'temp': p_temp, 'wind': p_wind,
                    'is_dome': p_dome, 'is_home': is_home, 'spread_line': p_spread
                }])
                pred = pass_model.predict(feats)[0]
                margin = 63.6
                stat_label = "Passing Yards"
            else:
                pred, margin, stat_label = None, None, None

        elif prop_type == "Rushing Yards":
            stats = get_player_recent_stats(
                rushing, 'rusher_player_name', 'rusher_player_id',
                player, ['rush_yards','rush_attempts','rush_tds']
            )
            if stats is not None:
                feats = pd.DataFrame([{
                    'avg_rush_yards_l4':    stats.get('avg_rush_yards_l4', 55),
                    'avg_rush_attempts_l4': stats.get('avg_rush_attempts_l4', 14),
                    'avg_rush_tds_l4':      stats.get('avg_rush_tds_l4', 0.4),
                    'temp': p_temp, 'wind': p_wind,
                    'is_dome': p_dome, 'is_home': is_home, 'spread_line': p_spread
                }])
                pred = rush_model.predict(feats)[0]
                margin = 21.6
                stat_label = "Rushing Yards"
            else:
                pred, margin, stat_label = None, None, None

        else:  # Receiving
            stats = get_player_recent_stats(
                receiving, 'receiver_player_name', 'receiver_player_id',
                player, ['rec_yards','targets','receptions','rec_tds']
            )
            if stats is not None:
                feats = pd.DataFrame([{
                    'avg_rec_yards_l4':   stats.get('avg_rec_yards_l4', 50),
                    'avg_targets_l4':     stats.get('avg_targets_l4', 6),
                    'avg_receptions_l4':  stats.get('avg_receptions_l4', 4),
                    'avg_rec_tds_l4':     stats.get('avg_rec_tds_l4', 0.3),
                    'temp': p_temp, 'wind': p_wind,
                    'is_dome': p_dome, 'is_home': is_home, 'spread_line': p_spread
                }])
                pred = rec_model.predict(feats)[0]
                margin = 21.4
                stat_label = "Receiving Yards"
            else:
                pred, margin, stat_label = None, None, None

        st.divider()
        if pred is not None and pred > 0:
            low  = max(0, pred - margin)
            high = pred + margin

            m1, m2, m3 = st.columns(3)
            m1.metric("📉 Low End",  f"{low:.0f} yds")
            m2.metric("🎯 Projection", f"{pred:.0f} yds")
            m3.metric("📈 High End",  f"{high:.0f} yds")

            st.progress(min(float(pred / (high + 10)), 1.0))
            st.caption(f"Model MAE: ±{margin} yards | "
                       f"Range based on last 4 games + conditions")

            # Show recent games
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
            st.warning(f"Not enough data for {player}. Try another player!")

# ──────────────────────────────────────────────────
# TAB 3: HEAD TO HEAD
# ──────────────────────────────────────────────────
with tab3:
    st.header("Historical Head-to-Head")

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
        st.info("No matchups found.")