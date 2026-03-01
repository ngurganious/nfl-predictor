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
def load_css():
    # Check both locations: assets/style.css (preferred) or style.css (root)
    search_paths = [Path(__file__).parent / "assets" / "style.css", Path(__file__).parent / "style.css"]
    for css_file in search_paths:
        if css_file.exists():
            with open(css_file) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            return
load_css()

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

# ── Home page ─────────────────────────────────────────────────────────────────
def _render_home():
    # Hero Section
    logo_path = Path(__file__).parent / "assets" / "logo.svg"
    if not logo_path.exists():
        logo_path = Path(__file__).parent / "logo.svg"

    if logo_path.exists():
        with open(logo_path, "r") as f:
            logo_html = f'<div style="width: 380px; margin: 0 auto 1rem auto;">{f.read()}</div>'
    else:
        logo_html = '<div class="edgeiq-logo"><span class="edgeiq-icon">⚡</span> EdgeIQ</div>'

    st.markdown(f"""
        <div style="text-align: center; padding: 40px 0;">
            {logo_html}
            <h1 style="font-size: 3rem; margin-bottom: 10px;">Institutional-Grade Sports Analytics.</h1>
            <p style="font-size: 1.2rem; color: #94a3b8; max-width: 700px; margin: 0 auto;">
                Stop betting on gut feelings. EdgeIQ uses stacking ensemble models trained on 25 years of data 
                to find the edge between true probability and the Vegas line.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    nfl_col, nhl_col = st.columns(2, gap="large")

    with nfl_col:
        with st.container(border=True):
            st.markdown("### 🏈 NFL Football")
            st.markdown("""
            <div style="margin-bottom: 20px;">
                <span class="signal-badge signal-strong">69.3% Accuracy</span>
                <span class="signal-badge signal-pass">26 Features</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            *   **Game Predictor:** Live odds, weather, and injury adjustments.
            *   **Player Props:** Passing, rushing, and receiving models.
            *   **Strategy:** Kelly Criterion bet sizing & backtesting.
            """)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Launch NFL Terminal", type="primary",
                         use_container_width=True, key="go_nfl"):
                st.session_state['sport'] = 'nfl'
                st.rerun()

    with nhl_col:
        with st.container(border=True):
            st.markdown("### 🏒 NHL Hockey")
            st.markdown("""
            <div style="margin-bottom: 20px;">
                <span class="signal-badge signal-lean">58.0% Accuracy</span>
                <span class="signal-badge signal-pass">High Volatility</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            *   **Moneyline Model:** ELO + Goalie Quality adjustments.
            *   **Total Goals:** Over/Under prediction engine.
            *   **Goalie Ratings:** Advanced save % above expected.
            """)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Launch NHL Terminal", type="primary",
                         use_container_width=True, key="go_nhl"):
                st.session_state['sport'] = 'nhl'
                st.rerun()

    st.divider()
    
    # Value Props Grid
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 📐 The Math")
        st.caption("We don't guess. We use Gradient Boosting and Random Forest ensembles stacked with Logistic Regression meta-learners.")
    with c2:
        st.markdown("#### 🛡️ The Discipline")
        st.caption("The Kelly Criterion manages your bankroll. We tell you exactly how much to bet based on your edge and risk tolerance.")
    with c3:
        st.markdown("#### 🔍 The Transparency")
        st.caption("We show our work. Every prediction comes with a confidence score, feature breakdown, and historical backtesting.")

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
