# ============================================
# NFL PREDICTOR - The Streamlit App
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="NFL Game Predictor",
    page_icon="🏈",
    layout="wide"
)

# ── Load our trained model and data ──────────────────────────────
@st.cache_resource  # Cache so it only loads once
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('elo_ratings.pkl', 'rb') as f:
        elo_ratings = pickle.load(f)
    return model, elo_ratings

@st.cache_data
def load_data():
    return pd.read_csv('games_processed.csv')

model, elo_ratings = load_model()
games = load_data()

# ── ELO helper ───────────────────────────────────────────────────
def get_elo(team):
    return elo_ratings.get(team, 1500)

def expected_win_prob(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

# ── NFL Teams ─────────────────────────────────────────────────────
NFL_TEAMS = sorted([
    'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
    'DAL', 'DEN', 'DET', 'GB',  'HOU', 'IND', 'JAX', 'KC',
    'LA',  'LAC', 'LV',  'MIA', 'MIN', 'NE',  'NO',  'NYG',
    'NYJ', 'PHI', 'PIT', 'SEA', 'SF',  'TB',  'TEN', 'WAS'
])

# ── Header ───────────────────────────────────────────────────────
st.title("🏈 NFL Game Predictor")
st.markdown("*Powered by 25 years of NFL data, ELO ratings, and machine learning*")
st.divider()

# ── ELO Rankings Sidebar ─────────────────────────────────────────
with st.sidebar:
    st.header("📊 Current ELO Rankings")
    st.caption("Based on all games 2000–2024")
    
    elo_df = pd.Series(elo_ratings).reset_index()
    elo_df.columns = ['Team', 'ELO']
    elo_df = elo_df[elo_df['Team'].isin(NFL_TEAMS)]
    elo_df = elo_df.sort_values('ELO', ascending=False).reset_index(drop=True)
    elo_df.index += 1
    elo_df['ELO'] = elo_df['ELO'].round(0).astype(int)
    st.dataframe(elo_df, use_container_width=True)

# ── Main Prediction UI ───────────────────────────────────────────
st.header("🎯 Game Predictor")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🏠 Home Team")
    home_team = st.selectbox("Select Home Team", NFL_TEAMS, index=NFL_TEAMS.index('KC'))
    home_rest = st.slider("Days of Rest", 3, 14, 7, key='home_rest')
    spread = st.slider("Vegas Spread (negative = home favored)", -28.0, 28.0, -3.0, 0.5)

with col2:
    st.subheader("✈️ Away Team")
    away_team = st.selectbox("Select Away Team", NFL_TEAMS, index=NFL_TEAMS.index('BUF'))
    away_rest = st.slider("Days of Rest", 3, 14, 7, key='away_rest')
    is_div = st.checkbox("Divisional Game?")

with col3:
    st.subheader("🌤️ Conditions")
    roof = st.selectbox("Stadium Type", ['outdoors', 'dome', 'closed', 'open'])
    is_dome = 1 if roof == 'dome' else 0
    temp = st.slider("Temperature (°F)", 20, 100, 65) if roof == 'outdoors' else 72
    wind = st.slider("Wind Speed (mph)", 0, 40, 5) if roof == 'outdoors' else 0
    surface = st.selectbox("Surface", ['grass', 'turf'])
    is_grass = 1 if surface == 'grass' else 0

st.divider()

# ── Prediction ───────────────────────────────────────────────────
if st.button("🔮 PREDICT", type="primary", use_container_width=True):
    
    home_elo = get_elo(home_team)
    away_elo = get_elo(away_team)
    elo_diff = home_elo - away_elo
    
    # Build the feature vector (same order as training!)
    features = pd.DataFrame([{
        'elo_diff':   elo_diff,
        'spread_line': spread,
        'home_rest':  home_rest,
        'away_rest':  away_rest,
        'temp':       temp,
        'wind':       wind,
        'is_dome':    is_dome,
        'is_grass':   is_grass,
        'div_game':   int(is_div),
    }])
    
    # Get win probability
    prob_home_win = model.predict_proba(features)[0][1]
    prob_away_win = 1 - prob_home_win
    
    # ELO-based probability (for comparison)
    elo_prob = expected_win_prob(home_elo, away_elo)
    
    # Display results
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(
            f"🏠 {home_team} Win Probability",
            f"{prob_home_win*100:.1f}%",
            delta=f"ELO says: {elo_prob*100:.1f}%"
        )
        st.progress(prob_home_win)
    
    with res_col2:
        st.metric(
            f"✈️ {away_team} Win Probability",
            f"{prob_away_win*100:.1f}%",
            delta=f"ELO says: {(1-elo_prob)*100:.1f}%"
        )
        st.progress(prob_away_win)

    # Verdict
    st.divider()
    winner = home_team if prob_home_win > 0.5 else away_team
    confidence = max(prob_home_win, prob_away_win)
    
    if confidence > 0.70:
        conf_label = "🔥 HIGH CONFIDENCE"
    elif confidence > 0.60:
        conf_label = "✅ MODERATE CONFIDENCE"
    else:
        conf_label = "⚠️ TOSS-UP"

    st.subheader(f"{conf_label}: {winner} wins")
    
    # Breakdown
    with st.expander("📊 See prediction breakdown"):
        st.write(f"**{home_team} ELO:** {home_elo:.0f}")
        st.write(f"**{away_team} ELO:** {away_elo:.0f}")
        st.write(f"**ELO Difference:** {elo_diff:.0f} (home advantage)")
        st.write(f"**Vegas Spread:** {spread}")
        st.write(f"**Temperature:** {temp}°F")
        st.write(f"**Wind:** {wind} mph")
        st.write(f"**Dome:** {'Yes' if is_dome else 'No'}")
        st.write(f"**Grass:** {'Yes' if is_grass else 'No'}")
        st.write(f"**Divisional:** {'Yes' if is_div else 'No'}")

# ── Historical Stats ─────────────────────────────────────────────
st.divider()
st.header("📈 Historical Head-to-Head")

h2h_col1, h2h_col2 = st.columns(2)
with h2h_col1:
    h2h_home = st.selectbox("Team A", NFL_TEAMS, key='h2h_home')
with h2h_col2:
    h2h_away = st.selectbox("Team B", NFL_TEAMS, index=1, key='h2h_away')

h2h = games[
    ((games['home_team'] == h2h_home) & (games['away_team'] == h2h_away)) |
    ((games['home_team'] == h2h_away) & (games['away_team'] == h2h_home))
].copy()

if len(h2h) > 0:
    team_a_wins = (
        ((h2h['home_team'] == h2h_home) & (h2h['home_score'] > h2h['away_score'])) |
        ((h2h['away_team'] == h2h_home) & (h2h['away_score'] > h2h['home_score']))
    ).sum()
    
    team_b_wins = len(h2h) - team_a_wins
    
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{h2h_home} Wins", team_a_wins)
    m2.metric(f"{h2h_away} Wins", team_b_wins)
    m3.metric("Total Games", len(h2h))
    
    st.dataframe(
        h2h[['season', 'week', 'home_team', 'home_score', 'away_score', 'away_team', 'temp', 'wind', 'roof']]
        .sort_values('season', ascending=False)
        .head(10)
        .reset_index(drop=True),
        use_container_width=True
    )
else:
    st.info("No head-to-head matchups found in the dataset.")