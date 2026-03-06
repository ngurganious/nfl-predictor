# ============================================================
# EdgeIQ — Multi-Sport Home Page Router
# Entry point: streamlit run app.py
# ============================================================

import streamlit as st
from pathlib import Path

# Must be first Streamlit call
st.set_page_config(
    page_title="EdgeIQ",
    page_icon="⚡",
    layout="wide"
)

# Load .env keys (RAPIDAPI_KEY, ODDS_API_KEY) if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Streamlit Cloud: inject st.secrets into os.environ
try:
    import os
    for _sk, _sv in st.secrets.items():
        os.environ.setdefault(str(_sk), str(_sv))
except Exception:
    pass

# ── CSS Loader ────────────────────────────────────────────────────────────────
@st.cache_data
def _read_css():
    search_paths = [Path(__file__).parent / "assets" / "style.css", Path(__file__).parent / "style.css"]
    for css_file in search_paths:
        if css_file.exists():
            with open(css_file) as f:
                return f.read()
    return ""

st.markdown(f"<style>{_read_css()}</style>", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if 'sport' not in st.session_state:
    st.session_state['sport'] = None
if 'account_name' not in st.session_state:
    st.session_state['account_name'] = 'Guest'
if 'show_acct_modal' not in st.session_state:
    st.session_state['show_acct_modal'] = False

# ── Account placeholder (top right) ──────────────────────────────────────────
def _render_account_header():
    _, _, acct_col = st.columns([6, 2, 2])
    with acct_col:
        name = st.session_state.get('account_name', 'Guest')
        st.markdown(f"**{name}** &nbsp; 👤", unsafe_allow_html=True)
        if st.button("⚙️ Settings", key="acct_settings_btn", use_container_width=True):
            st.session_state['show_acct_modal'] = not st.session_state.get('show_acct_modal', False)
            st.rerun()

    if st.session_state.get('show_acct_modal'):
        with st.container(border=True):
            st.markdown("**Account Settings**")
            new_name = st.text_input(
                "Display name",
                value=st.session_state.get('account_name', 'Guest'),
                key='acct_name_input'
            )
            save_col, cancel_col = st.columns(2)
            with save_col:
                if st.button("Save", key='acct_save', type='primary'):
                    st.session_state['account_name'] = new_name
                    st.session_state['show_acct_modal'] = False
                    st.rerun()
            with cancel_col:
                if st.button("Cancel", key='acct_cancel'):
                    st.session_state['show_acct_modal'] = False
                    st.rerun()

# ── Sport definitions ─────────────────────────────────────────────────────────
_SPORTS = [
    {"key": "nfl",   "icon": "\U0001f3c8", "label": "NFL",    "active": True},
    {"key": "nba",   "icon": "\U0001f3c0", "label": "NBA",    "active": False},
    {"key": "mlb",   "icon": "\u26be",     "label": "MLB",    "active": True},
    {"key": "nhl",   "icon": "\U0001f3d2", "label": "NHL",    "active": True},
    {"key": "ncaaf", "icon": "\U0001f3c8", "label": "NCAAF",  "active": False},
    {"key": "ncaab", "icon": "\U0001f3c0", "label": "NCAAB",  "active": False},
    {"key": "soccer","icon": "\u26bd",     "label": "Soccer", "active": False},
    {"key": "ufc",   "icon": "\U0001f94a", "label": "UFC",    "active": False},
]

# ── Top picks collector ──────────────────────────────────────────────────────
def _collect_top_picks(limit=10):
    picks = []
    # NFL game predictions
    for k, v in st.session_state.items():
        if k.startswith('g') and k.endswith('_pred') and not k.startswith(('nhl_', 'mlb_')):
            if isinstance(v, dict) and 'prob_home' in v:
                prob = max(v.get('prob_home', 0.5), 1 - v.get('prob_home', 0.5))
                winner = v.get('home_team', '?') if v.get('prob_home', 0.5) >= 0.5 else v.get('away_team', '?')
                picks.append({
                    'sport': 'NFL', 'sport_css': 'nfl',
                    'matchup': f"{v.get('away_team', '?')} @ {v.get('home_team', '?')}",
                    'bet': f"{winner} ML",
                    'prob': prob,
                    'type': 'Game',
                })
    # NHL game predictions
    for k, v in st.session_state.items():
        if k.startswith('nhl_g') and k.endswith('_pred'):
            if isinstance(v, dict) and 'prob_home' in v:
                prob = max(v.get('prob_home', 0.5), 1 - v.get('prob_home', 0.5))
                winner = v.get('home_team', '?') if v.get('prob_home', 0.5) >= 0.5 else v.get('away_team', '?')
                picks.append({
                    'sport': 'NHL', 'sport_css': 'nhl',
                    'matchup': f"{v.get('away_team', '?')} @ {v.get('home_team', '?')}",
                    'bet': f"{winner} ML",
                    'prob': prob,
                    'type': 'Game',
                })
    # MLB game predictions
    for k, v in st.session_state.items():
        if k.startswith('mlb_g') and k.endswith('_pred'):
            if isinstance(v, dict) and 'prob_home' in v:
                prob = max(v.get('prob_home', 0.5), 1 - v.get('prob_home', 0.5))
                winner = v.get('home_team', '?') if v.get('prob_home', 0.5) >= 0.5 else v.get('away_team', '?')
                picks.append({
                    'sport': 'MLB', 'sport_css': 'mlb',
                    'matchup': f"{v.get('away_team', '?')} @ {v.get('home_team', '?')}",
                    'bet': f"{winner} ML",
                    'prob': prob,
                    'type': 'Game',
                })
    # NHL player props
    for k, v in st.session_state.items():
        if k.startswith('nhl_props_g') and isinstance(v, dict):
            for player, pdata in v.items():
                if isinstance(pdata, dict) and 'goals' in pdata:
                    for prop_type in ('goals', 'assists', 'shots'):
                        pred = pdata.get(prop_type)
                        conf = pdata.get(f'{prop_type}_conf')
                        if pred is not None and conf is not None and conf > 0.5:
                            picks.append({
                                'sport': 'NHL', 'sport_css': 'nhl',
                                'matchup': player,
                                'bet': f"Over 0.5 {prop_type}",
                                'prob': conf,
                                'type': 'Prop',
                            })
    # MLB player props
    for k, v in st.session_state.items():
        if k.startswith('mlb_props_g') and isinstance(v, dict):
            for player, pdata in v.items():
                if isinstance(pdata, dict):
                    for prop_type in ('strikeouts', 'hits', 'total_bases'):
                        pred = pdata.get(prop_type)
                        conf = pdata.get(f'{prop_type}_conf')
                        if pred is not None and conf is not None and conf > 0.5:
                            picks.append({
                                'sport': 'MLB', 'sport_css': 'mlb',
                                'matchup': player,
                                'bet': f"Over 0.5 {prop_type.replace('_', ' ')}",
                                'prob': conf,
                                'type': 'Prop',
                            })
    picks.sort(key=lambda x: x['prob'], reverse=True)
    return picks[:limit]

def _signal_badge(prob):
    if prob >= 0.75:
        return '<span class="signal-badge signal-lock">LOCK</span>'
    elif prob >= 0.65:
        return '<span class="signal-badge signal-strong">STRONG</span>'
    elif prob >= 0.58:
        return '<span class="signal-badge signal-lean">LEAN</span>'
    else:
        return '<span class="signal-badge signal-pass">PASS</span>'

# ── Home page ─────────────────────────────────────────────────────────────────
def _render_home():
    # Hero
    logo_path = Path(__file__).parent / "assets" / "logo.svg"
    if not logo_path.exists():
        logo_path = Path(__file__).parent / "logo.svg"

    if logo_path.exists():
        with open(logo_path, "r") as f:
            logo_html = f'<div style="width: 380px; margin: 0 auto 1rem auto;">{f.read()}</div>'
    else:
        logo_html = '<div class="edgeiq-logo"><span class="edgeiq-icon">\u26a1</span> EdgeIQ</div>'

    st.markdown(f"""
        <div style="text-align: center; padding: 32px 0 16px 0;">
            {logo_html}
            <p style="font-size: 1.15rem; color: #94a3b8; max-width: 680px; margin: 0 auto;">
                Vegas sets the line. We find the gaps.<br>
                Three sports. One systematic edge.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── Sport Icon Bar ────────────────────────────────────────────────
    icons_html = '<div class="sport-bar">'
    for s in _SPORTS:
        cls = "active" if s["active"] else "inactive"
        icons_html += (
            f'<div class="sport-icon {cls}" data-sport="{s["key"]}">'
            f'  <div class="icon-circle">{s["icon"]}</div>'
            f'  <span class="icon-label">{s["label"]}</span>'
            f'</div>'
        )
    icons_html += '</div>'
    st.markdown(icons_html, unsafe_allow_html=True)

    # Streamlit buttons (hidden visually but functional) — one row of 8 columns
    cols = st.columns(len(_SPORTS))
    for i, s in enumerate(_SPORTS):
        with cols[i]:
            if s["active"]:
                if st.button(s["label"], key=f"home_sport_{s['key']}", use_container_width=True):
                    st.session_state['sport'] = s['key']
                    st.rerun()
            else:
                if st.button(s["label"], key=f"home_sport_{s['key']}", use_container_width=True):
                    st.toast(f"{s['label']} coming soon!")

    st.divider()

    # ── Sport Cards (details below icons) ─────────────────────────────
    nfl_col, nhl_col, mlb_col = st.columns(3, gap="large")

    with nfl_col:
        with st.container(border=True):
            st.markdown("### \U0001f3c8 NFL Football")
            st.markdown("""
            <div style="margin-bottom: 16px;">
                <span class="signal-badge signal-strong">69.3% Accuracy</span>
                <span class="signal-badge signal-pass">26 Features</span>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Game predictions \u00b7 Player props \u00b7 Kelly bet sizing \u00b7 Backtesting")

    with nhl_col:
        with st.container(border=True):
            st.markdown("### \U0001f3d2 NHL Hockey")
            st.markdown("""
            <div style="margin-bottom: 16px;">
                <span class="signal-badge signal-lean">58.0% Accuracy</span>
                <span class="signal-badge signal-pass">Goalie Quality</span>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Moneyline model \u00b7 Total goals \u00b7 Player props \u00b7 Parlay builder")

    with mlb_col:
        with st.container(border=True):
            st.markdown("### \u26be MLB Baseball")
            st.markdown("""
            <div style="margin-bottom: 16px;">
                <span class="signal-badge signal-lean">58.0% Accuracy</span>
                <span class="signal-badge signal-pass">29 Features</span>
            </div>
            """, unsafe_allow_html=True)
            st.caption("SP quality model \u00b7 Run totals \u00b7 Player props \u00b7 Kelly sizing")

    st.divider()

    # ── Top Picks Table ───────────────────────────────────────────────
    picks = _collect_top_picks()
    if picks:
        st.markdown("### \U0001f4ca Highest Probability Picks")
        rows_html = ""
        for p in picks:
            badge = _signal_badge(p['prob'])
            rows_html += (
                f"<tr>"
                f"<td><span class='pick-sport {p['sport_css']}'>{p['sport']}</span></td>"
                f"<td>{p['matchup']}</td>"
                f"<td><strong>{p['bet']}</strong></td>"
                f"<td>{p['prob']*100:.1f}%</td>"
                f"<td>{badge}</td>"
                f"<td>{p['type']}</td>"
                f"</tr>"
            )
        st.markdown(f"""
        <table class="top-picks-table">
            <thead>
                <tr>
                    <th>Sport</th><th>Matchup</th><th>Bet</th>
                    <th>Prob</th><th>Signal</th><th>Type</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### \U0001f4ca Top Picks")
        st.info("Load a sport schedule to see today's highest probability picks across all sports.")

    st.divider()

    # Value Props Grid
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### \U0001f4d0 The Math")
        st.caption("Gradient Boosting + Random Forest ensembles stacked with Logistic Regression meta-learners. 25 years of training data.")
    with c2:
        st.markdown("#### \U0001f6e1\ufe0f The Discipline")
        st.caption("Kelly Criterion bankroll management. We size every bet based on your edge and risk tolerance.")
    with c3:
        st.markdown("#### \U0001f50d The Transparency")
        st.caption("Every prediction shows confidence scores, feature breakdowns, and historical backtesting results.")

# ── Router ────────────────────────────────────────────────────────────────────
_render_account_header()

sport = st.session_state.get('sport')

if sport is None:
    _render_home()

elif sport == 'nfl':
    try:
        from final_app import render_nfl_app
        render_nfl_app()
    except Exception as e:
        st.error(f"Failed to load NFL section: {e}")
        if st.button("Return to Home"):
            st.session_state['sport'] = None
            st.rerun()

elif sport == 'nhl':
    try:
        from nhl_app import render_nhl_app
        render_nhl_app()
    except Exception as e:
        st.error(f"NHL section not yet available: {e}")
        st.info("The NHL model is still being trained. Check back soon!")
        if st.button("Return to Home", key="nhl_err_home"):
            st.session_state['sport'] = None
            st.rerun()

elif sport == 'mlb':
    try:
        from mlb_app import render_mlb_app
        render_mlb_app()
    except Exception as e:
        st.error(f"Failed to load MLB section: {e}")
        if st.button("Return to Home", key="mlb_err_home"):
            st.session_state['sport'] = None
            st.rerun()
