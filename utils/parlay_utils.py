import streamlit as st


def make_leg_id(sport, player, prop_type, home, away):
    safe_name = player.replace(' ', '_')
    safe_prop = prop_type.replace(' ', '_')
    return f"{sport}_{home}_{away}_{safe_name}_{safe_prop}"


def toggle_pick(key, leg_dict):
    tray = st.session_state.get('parlay_tray', [])
    lid = leg_dict.get('leg_id', '')
    if st.session_state.get(key, False):
        if not any(l.get('leg_id') == lid for l in tray):
            tray.append(leg_dict)
            st.session_state['parlay_tray'] = tray
    else:
        st.session_state['parlay_tray'] = [l for l in tray if l.get('leg_id') != lid]
