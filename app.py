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
            with open(css_file, encoding="utf-8") as f:
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
if 'edgeiq_sportsbook' not in st.session_state:
    st.session_state['edgeiq_sportsbook'] = 'DraftKings'
if 'parlay_tray' not in st.session_state:
    st.session_state['parlay_tray'] = []

# ── Parlay tray query-param actions (JS → Python bridge) ─────────────────────
_tray_clear = st.query_params.get('tray_clear')
if _tray_clear:
    st.session_state['parlay_tray'] = []
    st.query_params.clear()
    st.rerun()

_tray_remove = st.query_params.get('tray_remove')
if _tray_remove:
    st.session_state['parlay_tray'] = [
        p for p in st.session_state.get('parlay_tray', [])
        if str(p.get('leg_id', '')) != _tray_remove
    ]
    st.query_params.clear()
    st.rerun()

_tray_build = st.query_params.get('tray_build')
if _tray_build:
    tray = st.session_state.get('parlay_tray', [])
    if tray:
        from collections import Counter
        _sport_counts = Counter(l.get('sport_css', 'nhl') for l in tray)
        _majority_sport = _sport_counts.most_common(1)[0][0]
        st.session_state['sport'] = _majority_sport
        st.session_state['ladder_from_tray'] = True
    st.query_params.clear()
    st.rerun()

# ── Sportsbook options (display label → API key) ─────────────────────────────
_SPORTSBOOK_LABELS = ["DraftKings", "FanDuel", "BetMGM", "Caesars", "PointsBet", "Bovada"]

# ── Header (sportsbook top-left, account top-right) ──────────────────────────
def _render_header():
    sb_col, _, acct_col = st.columns([2, 5, 3])
    with sb_col:
        _sb_idx = _SPORTSBOOK_LABELS.index(st.session_state['edgeiq_sportsbook']) if st.session_state['edgeiq_sportsbook'] in _SPORTSBOOK_LABELS else 0
        st.selectbox(
            "Sportsbook", _SPORTSBOOK_LABELS, index=_sb_idx,
            key="edgeiq_sportsbook",
            label_visibility="collapsed",
        )
    with acct_col:
        name = st.session_state.get('account_name', 'Guest')
        st.markdown(f"**{name}** &nbsp; 👤", unsafe_allow_html=True)
        if st.button("Settings", key="acct_settings_btn", use_container_width=True):
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
    from datetime import datetime, timedelta, timezone

    picks = []

    # Build flat game lists from schedules for date/time lookup
    def _flat_games(sched_key):
        sched = st.session_state.get(sched_key, {})
        return [g for gs in sched.values() for g in gs]

    nfl_games = _flat_games('weekly_schedule')
    nhl_games = _flat_games('nhl_weekly_schedule')
    mlb_games = _flat_games('mlb_weekly_schedule')

    # 48-hour cutoff for filtering props to near-term games only
    _now = datetime.now(timezone.utc)
    _cutoff = _now + timedelta(hours=48)

    def _within_48h(games, idx):
        if 0 <= idx < len(games):
            dt = games[idx].get('datetime_et')
            if dt is not None:
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=timezone.utc) <= _cutoff
                return dt <= _cutoff
        return True  # no datetime → include (NFL ELO-only, etc.)

    def _game_time(games, idx):
        if 0 <= idx < len(games):
            g = games[idx]
            dl = g.get('game_date_label', '')
            tl = g.get('game_time_et', '')
            return f"{dl} · {tl}" if dl and tl else dl or tl or ''
        return ''

    # NFL game predictions
    for k, v in st.session_state.items():
        if k.startswith('g') and k.endswith('_pred') and not k.startswith(('nhl_', 'mlb_')):
            if isinstance(v, dict) and 'final_prob_home' in v:
                idx = int(k[1:].replace('_pred', '')) if k[1:].replace('_pred', '').isdigit() else -1
                if not _within_48h(nfl_games, idx):
                    continue
                prob = max(v.get('final_prob_home', 0.5), 1 - v.get('final_prob_home', 0.5))
                winner = v.get('home_team', '?') if v.get('final_prob_home', 0.5) >= 0.5 else v.get('away_team', '?')
                loser = v.get('away_team', '?') if v.get('final_prob_home', 0.5) >= 0.5 else v.get('home_team', '?')
                picks.append({
                    'sport': 'NFL', 'sport_css': 'nfl',
                    'player': f"{winner} vs {loser}",
                    'team': winner, 'position': '', 'is_forward': 0,
                    'bet': 'Moneyline',
                    'prob': prob, 'pred': prob,
                    'edge_pct': 0, 'line': None,
                    'odds': -110, 'has_line': False,
                    'type': 'Game',
                    'game_time': _game_time(nfl_games, idx),
                })
    # NHL game predictions
    for k, v in st.session_state.items():
        if k.startswith('nhl_g') and k.endswith('_pred'):
            if isinstance(v, dict) and 'home_win_prob' in v:
                idx_str = k.replace('nhl_g', '').replace('_pred', '')
                idx = int(idx_str) if idx_str.isdigit() else -1
                if not _within_48h(nhl_games, idx):
                    continue
                prob = max(v.get('home_win_prob', 0.5), 1 - v.get('home_win_prob', 0.5))
                winner = v.get('home_team', '?') if v.get('home_win_prob', 0.5) >= 0.5 else v.get('away_team', '?')
                loser = v.get('away_team', '?') if v.get('home_win_prob', 0.5) >= 0.5 else v.get('home_team', '?')
                picks.append({
                    'sport': 'NHL', 'sport_css': 'nhl',
                    'player': f"{winner} vs {loser}",
                    'team': winner, 'position': '', 'is_forward': 0,
                    'bet': 'Moneyline',
                    'prob': prob, 'pred': prob,
                    'edge_pct': 0, 'line': None,
                    'odds': -110, 'has_line': False,
                    'type': 'Game',
                    'game_time': _game_time(nhl_games, idx),
                })
    # MLB game predictions
    for k, v in st.session_state.items():
        if k.startswith('mlb_g') and k.endswith('_pred'):
            if isinstance(v, dict) and 'home_win_prob' in v:
                prob = max(v.get('home_win_prob', 0.5), 1 - v.get('home_win_prob', 0.5))
                winner = v.get('home_team', '?') if v.get('home_win_prob', 0.5) >= 0.5 else v.get('away_team', '?')
                loser = v.get('away_team', '?') if v.get('home_win_prob', 0.5) >= 0.5 else v.get('home_team', '?')
                picks.append({
                    'sport': 'MLB', 'sport_css': 'mlb',
                    'player': f"{winner} vs {loser}",
                    'team': winner, 'position': '', 'is_forward': 0,
                    'bet': 'Moneyline',
                    'prob': prob, 'pred': prob,
                    'edge_pct': 0, 'line': None,
                    'odds': -110, 'has_line': False,
                    'type': 'Game',
                    'game_time': '',
                })
    # NHL player props (list of dicts from _compute_game_props)
    for k, v in st.session_state.items():
        if k.startswith('nhl_props_g') and isinstance(v, list):
            idx_str = k.replace('nhl_props_g', '')
            idx = int(idx_str) if idx_str.isdigit() else -1
            if not _within_48h(nhl_games, idx):
                continue
            gt = _game_time(nhl_games, idx)
            _g = nhl_games[idx] if 0 <= idx < len(nhl_games) else {}
            _h = _g.get('home_team', '')
            _a = _g.get('away_team', '')
            for prop in v:
                if isinstance(prop, dict) and 'best_prob' in prop:
                    picks.append({
                        'sport': 'NHL', 'sport_css': 'nhl',
                        'player': prop.get('name', '?'),
                        'team': prop.get('team', ''),
                        'position': prop.get('position', ''),
                        'is_forward': prop.get('is_forward', 1),
                        'bet': f"{prop.get('best_type', 'Shots')} {prop.get('best_direction', 'OVER')}",
                        'prob': prop.get('best_prob', 0),
                        'pred': prop.get('best_pred', 0),
                        'edge_pct': prop.get('best_edge_pct', 0),
                        'line': prop.get('best_line'),
                        'odds': prop.get('best_odds', -110),
                        'has_line': prop.get('best_has_line', False),
                        'type': 'Prop',
                        'game_time': gt,
                        'home_team': _h,
                        'away_team': _a,
                        'best_market': prop.get('best_market', ''),
                        'best_type': prop.get('best_type', ''),
                        'best_direction': prop.get('best_direction', 'OVER'),
                        'best_pred': prop.get('best_pred', 0),
                        'best_desc': prop.get('best_desc', ''),
                        'best_prob': prop.get('best_prob', 0),
                        'best_mae': prop.get('best_mae', 0),
                        'best_edge_pct': prop.get('best_edge_pct', 0),
                        'best_book': prop.get('best_book'),
                    })
    # MLB player props (list of dicts from _compute_game_props)
    for k, v in st.session_state.items():
        if k.startswith('mlb_props_g') and isinstance(v, list):
            for prop in v:
                if isinstance(prop, dict) and 'best_prob' in prop:
                    picks.append({
                        'sport': 'MLB', 'sport_css': 'mlb',
                        'player': prop.get('name', '?'),
                        'team': prop.get('team', ''),
                        'position': prop.get('position', ''),
                        'is_forward': 0,
                        'bet': f"{prop.get('best_type', 'K')} {prop.get('best_direction', 'OVER')}",
                        'prob': prop.get('best_prob', 0),
                        'pred': prop.get('best_pred', 0),
                        'edge_pct': prop.get('best_edge_pct', 0),
                        'line': prop.get('best_line'),
                        'odds': prop.get('best_odds', -110),
                        'has_line': prop.get('best_has_line', False),
                        'type': 'Prop',
                        'game_time': '',
                    })

    # Sort by probability descending
    picks.sort(key=lambda x: x['prob'], reverse=True)

    # Deduplicate: max 1 prop per player, max 3 of any prop type
    seen_players = set()
    type_counts = {}
    filtered = []
    for p in picks:
        player = p['player']
        # Extract prop type (e.g. "Shots", "Goals") — first word of bet for props
        prop_type = p['bet'].split(' ', 1)[0] if p['type'] == 'Prop' else p['bet']

        # Max 1 entry per player
        if player in seen_players:
            continue
        # Max 3 of any single prop type
        if type_counts.get(prop_type, 0) >= 3:
            continue

        seen_players.add(player)
        type_counts[prop_type] = type_counts.get(prop_type, 0) + 1
        filtered.append(p)
        if len(filtered) >= limit:
            break

    return filtered

def _signal_badge(edge_pct):
    if edge_pct >= 0.04:
        return "<span class='signal-badge signal-strong'>STRONG</span>"
    elif edge_pct >= 0.02:
        return "<span class='signal-badge signal-lean'>LEAN</span>"
    elif edge_pct >= 0.01:
        return "<span class='signal-badge signal-small'>SMALL</span>"
    else:
        return "<span class='signal-badge signal-pass'>PASS</span>"

# ── Pre-load picks for active sports ──────────────────────────────────────────
def _preload_active_sports():
    if 'homepage_preload_done' in st.session_state:
        return

    import pickle

    with st.spinner("Loading today's picks across all sports..."):
        # ── NHL ──────────────────────────────────────────────────────
        try:
            from apis.nhl import NHLClient
            from nhl_game_week import fetch_nhl_weekly_schedule
            from nhl_app import (load_nhl_model, run_nhl_prediction,
                                 load_nhl_total_model, load_nhl_games,
                                 load_nhl_goalie_ratings, load_nhl_team_stats,
                                 load_nhl_full_goalie_ratings,
                                 load_nhl_player_models, load_nhl_skater_stats,
                                 _compute_game_props)

            nhl_sched = fetch_nhl_weekly_schedule(NHLClient())
            if nhl_sched:
                st.session_state['nhl_weekly_schedule'] = nhl_sched
                _m, _f, _, _elo = load_nhl_model()
                if _m is not None:
                    _tp = load_nhl_total_model()
                    _gr = load_nhl_goalie_ratings()
                    _ts = load_nhl_team_stats()
                    _ng = load_nhl_games()
                    _fg = load_nhl_full_goalie_ratings()
                    _idx = 0
                    for _day, _games in nhl_sched.items():
                        for _g in _games:
                            _pk = f"nhl_g{_idx}_pred"
                            if _pk not in st.session_state:
                                try:
                                    _r = run_nhl_prediction(
                                        _g['home_team'], _g['away_team'],
                                        _m, _f, _elo, _gr, _ts, _tp,
                                        nhl_games=_ng, full_goalie_ratings=_fg,
                                    )
                                    if _r and 'error' not in _r:
                                        st.session_state[_pk] = _r
                                except Exception:
                                    pass
                            _idx += 1
                    st.session_state['nhl_precalc_done'] = True
                # NHL player props
                _pm = load_nhl_player_models()
                _ss = load_nhl_skater_stats()
                if _pm and not _ss.empty:
                    _all = [g for gs in nhl_sched.values() for g in gs]
                    for _pi, _pg in enumerate(_all):
                        _ppk = f'nhl_props_g{_pi}'
                        if _ppk not in st.session_state:
                            try:
                                st.session_state[_ppk] = _compute_game_props(
                                    _pg['home_team'], _pg['away_team'],
                                    _pm, _ss, _ts,
                                )
                            except Exception:
                                pass
                    st.session_state['nhl_props_precalc_done'] = True
        except Exception:
            pass

        # ── MLB ──────────────────────────────────────────────────────
        try:
            from apis.mlb import MLBClient
            from mlb_game_week import fetch_mlb_weekly_schedule
            from mlb_app import (load_mlb_model, run_mlb_prediction,
                                 load_mlb_total_model, load_mlb_games,
                                 load_mlb_pitcher_ratings, load_mlb_team_stats)

            mlb_sched = fetch_mlb_weekly_schedule(MLBClient())
            if mlb_sched:
                st.session_state['mlb_weekly_schedule'] = mlb_sched
                _m, _f, _, _elo = load_mlb_model()
                if _m is not None:
                    _tp = load_mlb_total_model()
                    _pr = load_mlb_pitcher_ratings()
                    _ts = load_mlb_team_stats()
                    _mg = load_mlb_games()
                    for _day, _games in mlb_sched.items():
                        for _i, _g in enumerate(_games):
                            _pk = f"mlb_g{_day}{_i}_pred"
                            if _pk not in st.session_state:
                                try:
                                    _r = run_mlb_prediction(
                                        _g['home_team'], _g['away_team'],
                                        _m, _f, _elo, _pr, _ts, _tp,
                                        mlb_games=_mg,
                                    )
                                    if _r and 'error' not in _r:
                                        st.session_state[_pk] = _r
                                except Exception:
                                    pass
                    st.session_state['mlb_precalc_done'] = True
        except Exception:
            pass

        # ── NFL (ELO-only — full model runs when user enters NFL) ─
        try:
            from apis.espn import ESPNClient
            from game_week import fetch_weekly_schedule

            nfl_sched = fetch_weekly_schedule(ESPNClient())
            if nfl_sched:
                st.session_state['weekly_schedule'] = nfl_sched
                try:
                    with open("elo_ratings.pkl", "rb") as f:
                        _nfl_elo = pickle.load(f)
                except Exception:
                    _nfl_elo = {}
                if _nfl_elo:
                    _idx = 0
                    for _day, _games in nfl_sched.items():
                        for _g in _games:
                            _pk = f"g{_idx}_pred"
                            if _pk not in st.session_state:
                                _he = _nfl_elo.get(_g['home_team'], 1500)
                                _ae = _nfl_elo.get(_g['away_team'], 1500)
                                _ph = 1.0 / (1.0 + 10 ** ((_ae - _he) / 400))
                                st.session_state[_pk] = {
                                    'home_team': _g['home_team'],
                                    'away_team': _g['away_team'],
                                    'final_prob_home': _ph,
                                    'base_prob_home': _ph,
                                }
                            _idx += 1
        except Exception:
            pass

    st.session_state['homepage_preload_done'] = True


# ── Home page ─────────────────────────────────────────────────────────────────
def _render_home():
    _preload_active_sports()

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

    # ── Sport Icon Buttons ────────────────────────────────────────────
    cols = st.columns(len(_SPORTS))
    for i, s in enumerate(_SPORTS):
        with cols[i]:
            cls = "active" if s["active"] else "inactive"
            st.markdown(
                f'<div class="sport-icon {cls}">'
                f'  <div class="icon-circle">{s["icon"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if s["active"]:
                if st.button(s["label"], key=f"home_sport_{s['key']}", use_container_width=True):
                    st.session_state['sport'] = s['key']
                    st.rerun()
            else:
                if st.button(s["label"], key=f"home_sport_{s['key']}", use_container_width=True, disabled=True):
                    pass

    # ── Top Picks Table ───────────────────────────────────────────────
    picks = _collect_top_picks()
    if picks:
        st.markdown("### \U0001f3c6 Top Picks")
        st.caption("Top 10 props across today's slate — sorted by model probability")
        hc = st.columns([0.4, 2.2, 1.2, 0.8, 0.8, 0.9, 0.8, 0.9])
        hc[0].caption("Pick")
        hc[1].caption("Player")
        hc[2].caption("Prop")
        hc[3].caption("Line")
        hc[4].caption("Pred")
        hc[5].caption("Edge%")
        hc[6].caption("Odds")
        hc[7].caption("Signal")
        st.markdown("<hr style='margin:2px 0 6px'>", unsafe_allow_html=True)
        from utils.parlay_utils import make_leg_id, toggle_pick

        for ti, p in enumerate(picks):
            _sport = p['sport_css']
            if p['type'] == 'Prop' and p.get('home_team'):
                lid = make_leg_id(_sport, p['player'], p.get('best_market', p['bet']), p['home_team'], p['away_team'])
            else:
                lid = make_leg_id(_sport, p['player'], p['bet'], p.get('home_team', ''), p.get('away_team', ''))
            cb_key = f"parlay_pick_{lid}"
            in_tray = any(x.get('leg_id') == lid for x in st.session_state.get('parlay_tray', []))
            _dir = p.get('best_direction', 'OVER')
            _leg = {
                'leg_id': lid, 'sport': p['sport'].upper(), 'sport_css': _sport,
                'player': p['player'], 'bet': p['bet'], 'odds': int(p.get('odds', -110)),
                'game': f"{p.get('away_team', '')} @ {p.get('home_team', '')}",
                'game_time': p.get('game_time', ''),
                'prop_type': p.get('best_type', p['bet']),
                'pred': p.get('best_pred', p.get('pred', 0)),
                'edge': round(p.get('best_edge_pct', p.get('edge_pct', 0)) * 100, 1),
                'confidence': p.get('best_prob', p.get('prob', 0)),
            }
            cols = st.columns([0.4, 2.2, 1.2, 0.8, 0.8, 0.9, 0.8, 0.9])
            cols[0].checkbox("Select", value=in_tray, key=cb_key,
                             on_change=toggle_pick, args=(cb_key, _leg), label_visibility='collapsed')
            # Player + sport badge + position badge + team
            sport_colors = {'nfl': '#22c55e', 'nhl': '#38bdf8', 'mlb': '#f87171'}
            sc = sport_colors.get(p['sport_css'], '#94a3b8')
            pos = p.get('position', '')
            pos_html = ''
            if pos:
                pc = '#22d3ee' if p.get('is_forward') else '#a78bfa'
                pos_html = (f"&nbsp;<span style='background:{pc};color:#0f172a;border-radius:4px;"
                            f"padding:1px 5px;font-size:0.75em;font-weight:700'>{pos}</span>")
            team_label = f"&nbsp;{p['team']}" if p.get('team') else ''
            gt_html = ''
            if p.get('game_time'):
                gt_html = f"<br><span style='color:#64748b;font-size:0.78em'>{p['game_time']}</span>"
            cols[1].markdown(
                f"**{p['player']}** "
                f"<span style='background:{sc};color:#0f172a;border-radius:4px;"
                f"padding:1px 5px;font-size:0.7em;font-weight:700'>{p['sport'].upper()}</span>"
                f"{pos_html}{team_label}{gt_html}",
                unsafe_allow_html=True,
            )
            # Prop type + direction
            if p['type'] == 'Prop':
                parts = p['bet'].split(' ', 1)
                prop_type = parts[0] if parts else p['bet']
                direction = parts[1] if len(parts) > 1 else 'OVER'
                dir_color = '#22c55e' if direction == 'OVER' else '#ef4444'
                cols[2].markdown(
                    f"<span style='color:#f1f5f9'>{prop_type}</span>"
                    f"<br><span style='color:{dir_color};font-size:0.82em'>{direction}</span>",
                    unsafe_allow_html=True,
                )
            else:
                cols[2].markdown(f"<span style='color:#f1f5f9'>{p['bet']}</span>", unsafe_allow_html=True)
            # Line
            if p.get('has_line') and p.get('line') is not None:
                cols[3].markdown(f"<span style='color:#f1f5f9;font-weight:600'>{p['line']}</span>", unsafe_allow_html=True)
            else:
                cols[3].markdown("<span style='color:#64748b;font-size:0.78em'>No line</span>", unsafe_allow_html=True)
            # Pred
            pred_val = p.get('pred', p.get('prob', 0))
            cols[4].markdown(f"<span style='color:#f1f5f9'>{pred_val:.2f}</span>", unsafe_allow_html=True)
            # Edge%
            ep = p.get('edge_pct', 0)
            if p.get('has_line'):
                ep_color = '#22c55e' if ep >= 0.02 else '#eab308' if ep >= 0.01 else '#94a3b8'
                cols[5].markdown(f"<span style='color:{ep_color};font-weight:600'>{ep*100:.1f}%</span>", unsafe_allow_html=True)
            else:
                # No sportsbook line — show model prob as context
                mp = p.get('prob', 0)
                mp_color = '#22c55e' if mp >= 0.65 else '#eab308' if mp >= 0.55 else '#94a3b8'
                cols[5].markdown(f"<span style='color:{mp_color};font-size:0.82em'>{mp*100:.0f}%</span>", unsafe_allow_html=True)
            # Odds
            _odds = p.get('odds', -110)
            cols[6].markdown(f"<span style='color:#cbd5e1'>{_odds:+d}</span>", unsafe_allow_html=True)
            # Signal
            if p.get('has_line'):
                cols[7].markdown(_signal_badge(ep), unsafe_allow_html=True)
            else:
                cols[7].markdown("<span style='color:#64748b;font-size:0.78em'>Model</span>", unsafe_allow_html=True)
    else:
        st.markdown("### \U0001f3c6 Top Picks")
        st.info("Load a sport schedule to see today's highest probability picks across all sports.")


# ── Parlay Tray (viewport-fixed bottom bar + expanded sheet via st.markdown) ──
def _render_parlay_tray():
    tray = st.session_state.get('parlay_tray', [])
    count = len(tray)

    def _combined_decimal(picks):
        dec = 1.0
        for p in picks:
            odds = p.get('odds', -110)
            if odds > 0:
                dec *= 1 + odds / 100
            else:
                dec *= 1 + 100 / abs(odds)
        return dec

    if tray:
        combo_dec = _combined_decimal(tray)
        if combo_dec >= 2.0:
            combo_american = f"+{int(round((combo_dec - 1) * 100))}"
        else:
            combo_american = f"-{int(round(100 / (combo_dec - 1)))}"
    else:
        combo_dec = 1.0
        combo_american = "—"

    # Build pick rows HTML
    pick_rows_html = ""
    for p in tray:
        sport_css = p.get('sport_css', 'nfl')
        odds = p.get('odds', -110)
        leg_id = p.get('leg_id', '')
        pick_rows_html += f"""
        <div class="pt-pick">
            <div class="pt-pick-left">
                <span class="pt-sport-badge pt-{sport_css}">{p.get('sport', 'NFL')}</span>
                <span class="pt-pick-name">{p.get('player', '?')}</span>
                <span class="pt-pick-bet">{p.get('bet', '')}</span>
            </div>
            <div class="pt-pick-right">
                <span class="pt-pick-odds">{odds:+d}</span>
                <button class="pt-remove-btn" data-leg="{leg_id}" title="Remove">&#10005;</button>
            </div>
        </div>"""

    combo_payout_10 = combo_dec * 10

    # CSS + HTML via st.markdown (renders in main DOM — position:fixed works)
    tray_css_html = f"""
<style>
.pt-bar {{
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 1000000;
    background: linear-gradient(180deg, #131c2e 0%, #0d1424 100%);
    border-top: 1px solid #0e7490;
    padding: 10px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 -4px 20px rgba(0,0,0,0.5);
    backdrop-filter: blur(12px);
    cursor: pointer;
    user-select: none;
    -webkit-user-select: none;
    font-family: 'Inter', sans-serif;
}}
.pt-bar-left {{
    display: flex;
    align-items: center;
    gap: 12px;
}}
.pt-bar-count {{
    background: #22d3ee;
    color: #0f172a;
    font-weight: 800;
    font-size: 0.85rem;
    padding: 2px 10px;
    border-radius: 9999px;
    font-family: 'JetBrains Mono', monospace;
}}
.pt-bar-label {{
    color: #e2e8f0;
    font-weight: 600;
    font-size: 0.9rem;
}}
.pt-bar-right {{
    display: flex;
    align-items: center;
    gap: 16px;
}}
.pt-bar-odds {{
    color: #22d3ee;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.95rem;
}}
.pt-bar-toggle {{
    color: #94a3b8;
    font-size: 0.9rem;
    transition: transform 0.2s;
}}
.pt-bar-toggle.open {{
    transform: rotate(180deg);
}}
.pt-sheet {{
    position: fixed;
    bottom: 44px;
    left: 0;
    right: 0;
    z-index: 999999;
    background: #0f172a;
    border-top: 2px solid #0e7490;
    box-shadow: 0 -8px 32px rgba(0,0,0,0.6);
    max-height: 50vh;
    overflow-y: auto;
    padding: 16px 24px 16px;
    display: none;
    font-family: 'Inter', sans-serif;
}}
.pt-sheet.open {{
    display: block;
}}
.pt-sheet-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2d45;
}}
.pt-sheet-title {{
    color: #fff;
    font-weight: 800;
    font-size: 1.1rem;
}}
.pt-sheet-subtitle {{
    color: #94a3b8;
    font-size: 0.82rem;
}}
.pt-pick {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid rgba(30, 45, 69, 0.5);
}}
.pt-pick-left {{
    display: flex;
    align-items: center;
    gap: 10px;
}}
.pt-pick-right {{
    display: flex;
    align-items: center;
    gap: 12px;
}}
.pt-sport-badge {{
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    padding: 1px 6px;
    border-radius: 4px;
    background: rgba(34, 211, 238, 0.15);
    color: #22d3ee;
}}
.pt-nfl {{ background: rgba(34, 197, 94, 0.15); color: #22c55e; }}
.pt-nhl {{ background: rgba(56, 189, 248, 0.15); color: #38bdf8; }}
.pt-mlb {{ background: rgba(248, 113, 113, 0.15); color: #f87171; }}
.pt-pick-name {{
    color: #e2e8f0;
    font-weight: 600;
    font-size: 0.88rem;
}}
.pt-pick-bet {{
    color: #94a3b8;
    font-size: 0.82rem;
}}
.pt-pick-odds {{
    color: #22d3ee;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.85rem;
}}
.pt-remove-btn {{
    background: none;
    border: 1px solid #334155;
    color: #94a3b8;
    font-size: 0.75rem;
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 4px;
    transition: all 0.15s;
}}
.pt-remove-btn:hover {{
    color: #ef4444;
    border-color: #ef4444;
}}
.pt-actions {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px solid #1e2d45;
}}
.pt-clear-btn {{
    background: none;
    border: 1px solid #334155;
    color: #94a3b8;
    font-size: 0.82rem;
    cursor: pointer;
    padding: 6px 14px;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    transition: all 0.15s;
}}
.pt-clear-btn:hover {{
    color: #ef4444;
    border-color: #ef4444;
}}
.pt-build-btn {{
    background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
    border: none;
    color: #fff;
    font-weight: 600;
    font-size: 0.88rem;
    cursor: pointer;
    padding: 8px 20px;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    box-shadow: 0 4px 12px rgba(6, 182, 212, 0.25);
    transition: all 0.15s;
}}
.pt-build-btn:hover {{
    box-shadow: 0 6px 16px rgba(6, 182, 212, 0.4);
    transform: translateY(-1px);
}}
.pt-spacer {{
    height: 52px;
}}
.pt-toggle-disabled {{
    opacity: 0.35;
    pointer-events: none;
}}
.pt-build-disabled {{
    opacity: 0.4;
    cursor: not-allowed;
    pointer-events: none;
}}
</style>

<div class="pt-sheet" id="ptSheet">
    <div class="pt-sheet-header">
        <div>
            <div class="pt-sheet-title">Parlay Builder</div>
            <div class="pt-sheet-subtitle">{count} leg{'s' if count != 1 else ''} &middot; {combo_american} combo &middot; ${combo_payout_10:.2f} on $10</div>
        </div>
    </div>
    <div class="pt-picks">{pick_rows_html}</div>
    <div class="pt-actions">
        <button class="pt-clear-btn" id="ptClearBtn">Clear All</button>
        <button class="pt-build-btn{' pt-build-disabled' if count < 3 else ''}" id="ptBuildBtn" {'disabled' if count < 3 else ''}>Build Ladder &rarr;</button>
    </div>
</div>

<div class="pt-bar" id="ptBar">
    <div class="pt-bar-left">
        <span class="pt-bar-count">{'🎯 ' + str(count)}</span>
        <span class="pt-bar-label">{'Add picks to build a parlay' if count == 0 else 'Parlay Tray'}</span>
    </div>
    <div class="pt-bar-right">
        <span class="pt-bar-odds">{combo_american if count > 0 else ''}</span>
        <span class="pt-bar-toggle{' pt-toggle-disabled' if count == 0 else ''}" id="ptToggleIcon">&#9650;</span>
    </div>
</div>

<div class="pt-spacer"></div>
"""
    st.markdown(tray_css_html, unsafe_allow_html=True)

    # JS via components.html — runs in iframe, targets parent DOM
    # Uses retrying interval to handle Streamlit DOM timing
    import streamlit.components.v1 as components
    tray_js = f"""
<script>
(function() {{
    function bind() {{
        var doc = window.parent.document;
        var bar = doc.getElementById('ptBar');
        var sheet = doc.getElementById('ptSheet');
        var icon = doc.getElementById('ptToggleIcon');
        if (!bar || !sheet) return false;

        var clearBtn = doc.getElementById('ptClearBtn');
        var buildBtn = doc.getElementById('ptBuildBtn');

        var open = false;
        bar.onclick = function() {{
            if ({count} === 0) return;
            open = !open;
            sheet.classList.toggle('open', open);
            icon.classList.toggle('open', open);
        }};

        var rmBtns = doc.querySelectorAll('.pt-remove-btn[data-leg]');
        rmBtns.forEach(function(btn) {{
            btn.onclick = function(e) {{
                e.stopPropagation();
                var url = new URL(window.parent.location.href);
                url.searchParams.set('tray_remove', btn.getAttribute('data-leg'));
                window.parent.location.href = url.toString();
            }};
        }});

        if (clearBtn) {{
            clearBtn.onclick = function(e) {{
                e.stopPropagation();
                var url = new URL(window.parent.location.href);
                url.searchParams.set('tray_clear', '1');
                window.parent.location.href = url.toString();
            }};
        }}

        if (buildBtn) {{
            buildBtn.onclick = function(e) {{
                e.stopPropagation();
                var url = new URL(window.parent.location.href);
                url.searchParams.set('tray_build', '1');
                window.parent.location.href = url.toString();
            }};
        }}
        return true;
    }}

    if (!bind()) {{
        var attempts = 0;
        var iv = setInterval(function() {{
            if (bind() || ++attempts > 20) clearInterval(iv);
        }}, 100);
    }}
}})();
</script>
"""
    components.html(tray_js, height=0, scrolling=False)


# ── Router ────────────────────────────────────────────────────────────────────
_render_header()

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

# ── Parlay Tray (render last so it floats at bottom) ─────────────────────────
_render_parlay_tray()
