# ============================================================
# EdgeIQ — Multi-Sport Home Page Router
# Entry point: streamlit run app.py
# ============================================================

import streamlit as st

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
    st.title("⚡ EdgeIQ")
    st.markdown("*Machine learning game predictions — powered by decades of historical data*")
    st.divider()

    nfl_col, nhl_col = st.columns(2, gap="large")

    with nfl_col:
        with st.container(border=True):
            st.markdown("## 🏈 NFL Football")
            st.markdown("**69.3% accuracy** · 26-feature stacking ensemble · 25 seasons of data")
            st.markdown("")
            st.markdown("✅ Game Predictor with live Vegas lines")
            st.markdown("✅ Player Props (passing, rushing, receiving)")
            st.markdown("✅ Head-to-Head history")
            st.markdown("✅ Super Bowl simulator")
            st.markdown("✅ Backtesting with Kelly criterion")
            st.markdown("")
            if st.button("Enter NFL Section →", type="primary",
                         use_container_width=True, key="go_nfl"):
                st.session_state['sport'] = 'nfl'
                st.rerun()

    with nhl_col:
        with st.container(border=True):
            st.markdown("## 🏒 NHL Hockey")
            st.markdown("**60-63% accuracy** · 22-feature stacking ensemble · 25 seasons of data")
            st.markdown("")
            st.markdown("✅ Game Predictor with live schedule")
            st.markdown("✅ Goalie quality ratings")
            st.markdown("✅ Over/Under total goals model")
            st.markdown("✅ Backtesting with Kelly criterion")
            st.markdown("🔜 Player props, H2H, Stanley Cup — coming soon")
            st.markdown("")
            if st.button("Enter NHL Section →", type="primary",
                         use_container_width=True, key="go_nhl"):
                st.session_state['sport'] = 'nhl'
                st.rerun()

    st.divider()
    st.caption("More sports coming soon: NBA, MLB, NFL International, College Football")

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
