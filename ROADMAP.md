# EdgeIQ — Build Roadmap

**Last updated:** 2026-03-01 (Phase 2 complete, Phase 5 RPL scoped)
**Source of truth:** `PRD.md` §4.9 for requirements detail. `EdgeIQ.md` for product standards.

> Claude: after completing any item, update status, fill in Completed date, and add a one-line note. See the Roadmap Rule in `CLAUDE.md`.

---

## Status Key
| Symbol | Meaning |
|--------|---------|
| 🔲 | Not Started |
| 🚧 | In Progress |
| ✅ | Done |

---

## Phase 1 — Core Standards & Quick Wins
*Low-effort items that bring both sports to a consistent baseline.*

| # | Item | Effort | Status | Completed | Notes |
|---|------|--------|--------|-----------|-------|
| 1 | Standardize Kelly cap to 10%, enforce 1% min and 10% max single bet (both sports) | Low | ✅ | 2026-03-01 | `final_app.py` `_kelly_rec()` cap 20%→10% · `nhl_app.py` `_nhl_kelly()` cap same · bankroll max_value 1M→100k |
| 2 | Risk tolerance slider → Kelly multiplier in sidebar (both sports) | Low | ✅ | 2026-03-01 | `selectbox` Conservative/Moderate/Aggressive in both sidebars · maps to 0.25×/0.5×/1.0× Kelly fraction |
| 3 | Betting strategy selector — 4 modes: Kelly / Fixed % / Fixed $ / Fractional Kelly | Medium | ✅ | 2026-03-01 | Sidebar `selectbox` + conditional Fixed %/$ inputs in both files · bet amount in game cards updates per strategy |
| 4 | "Lock" badge at >75% win probability confidence tier | Low | ✅ | 2026-03-01 | Added 🔒 LOCK tier in both confidence display sections in `final_app.py` (×2) and `nhl_app.py` |
| 5 | Bet signal color palette — apply `#22c55e` / `#eab308` / `#94a3b8` / `#ef4444` via custom CSS | Low | ✅ | 2026-03-01 | Confidence badge + Kelly Signal badge use colored HTML divs via `unsafe_allow_html=True` in both files |

---

## Phase 2 — Track Record Layer
*The features that make EdgeIQ something you return to. High user value.*

| # | Item | Effort | Status | Completed | Notes |
|---|------|--------|--------|-----------|-------|
| 6 | **Prediction History** — auto-log predictions at render time + results review tab (both sports) | Medium | ✅ | 2026-03-01 | `prediction_history.py` new module · `log_prediction()` hooked into `render_prediction_result()` (NFL) and `render_nhl_prediction_result()` (NHL) · "📋 Track Record" tab added to both sports · backfill via ESPN/NHL API |
| 7 | Daily bet summary panel in sidebar — total bets today, potential win, max loss, EV | Medium | ✅ | 2026-03-01 | Added "📊 Today's Summary" section in both sidebars · reads from `prediction_history.json` · shows count, stake, pot. win, EV |
| 8 | **Bet Tracker** — log placed bets, P&L chart, ROI (both sports) | Medium | ✅ | 2026-03-01 | "💰 Bet Tracker" sub-tab in Track Record · form to log bets · `user_bets.json` · Plotly cumulative P&L chart · ROI metric |
| 9 | JSON export/import — bankroll settings + bet history + prediction history | Low | ✅ | 2026-03-01 | "📤 Export/Import" sub-tab in Track Record · `st.download_button` (all data / preds only / bets only) · `st.file_uploader` for import |

---

## Phase 3 — NHL Parity
*Bring NHL to feature parity with NFL.*

| # | Item | Effort | Status | Completed | Notes |
|---|------|--------|--------|-----------|-------|
| 10 | NHL: wire Odds API for live moneylines and O/U (already built in `apis/odds.py`) | Medium | 🔲 | — | Add `OddsClient` call in `nhl_app.py` data pipeline · PRD §4.6 |
| 11 | NHL: Stanley Cup Predictor tab — 16-seed bracket, ELO-based, conference + Cup odds | Medium | 🔲 | — | Mirror `final_app.py` Super Bowl Predictor tab · uses `nhl_elo_ratings.pkl` · PRD §3.2.2 |
| 12 | NHL: Head-to-Head tab — historical matchup comparison | Medium | 🔲 | — | Mirror NFL H2H tab · `nhl_games_processed.csv` as data source · PRD §3.2.1 / §4.6 |

---

## Phase 4 — Advanced NHL Features
*Requires new model training and data pipelines.*

| # | Item | Effort | Status | Completed | Notes |
|---|------|--------|--------|-----------|-------|
| 13 | NHL: Player props tab — goals, assists, shots on goal prediction models | High | 🔲 | — | New `build_nhl_player_model.py` · GBR models · new tab in `nhl_app.py` · PRD §4.6 |
| 14 | NHL: Positional line matchup engine — top line vs top pairing scoring | High | 🔲 | — | Mirror `defensive_matchup.py` for hockey lines · PRD §4.6 |

---

## Completed Items
*Moved here when done. Keeps the active table clean.*

| # | Item | Completed | Notes |
|---|------|-----------|-------|
| — | Kelly game card UI — align NHL caption format to match NFL (period separators, "to limit volatility") | 2026-03-01 | `nhl_app.py` lines 538–549 · help text + caption standardized |
| — | PRD created — cross-sport constants, user requirements, gap analysis | 2026-03-01 | `PRD.md` created |
| — | EdgeIQ.md created — product definition, brand standards, feature standards | 2026-03-01 | `EdgeIQ.md` created |
| — | CLAUDE.md updated — lean project overview, roadmap rule added | 2026-03-01 | `CLAUDE.md` updated |

---

## Phase 5 — Recursive Parlay Ladder (RPL)
*The feature that moves EdgeIQ from "data tool" to "wealth management tool." Volatility dampening via anchored parlay tiers.*

| # | Item | Effort | Status | Completed | Notes |
|---|------|--------|--------|-----------|-------|
| 15 | NFL: Prop selection toggles + "🪜 Build Ladder" button on Player Props tab | Medium | 🔲 | — | Checkbox per prop card · auto-select top 3 by confidence · PRD §3.6.1 |
| 16 | NFL: Parlay Ladder tab — 4-tier ladder (Banker/Accelerator×2/Moonshot), odds calc, break-even stake sizing | Medium | 🔲 | — | New tab in `final_app.py` · anchor break-even rule · PRD §3.6.2–3.6.6 |
| 17 | NFL: Correlation filter — same-game conflict detection for ladder legs | Medium | 🔲 | — | Under/under + opposing-side filters · PRD §3.6.5 |
| 18 | NFL: Backtested ladder ROI from historical prop data | Medium | 🔲 | — | Simulate ladder performance on past prop outcomes · PRD §3.6.7 |
| 19 | NHL: Parlay Ladder tab (mirrors NFL) | Medium | 🔲 | — | Blocked by NHL Player Props (item 13) · PRD §3.6.8 |

---

## On Deck (Not Yet Scheduled)
*Captured in PRD but not prioritized for active development.*

| Item | PRD Ref | Notes |
|------|---------|-------|
| ~~Bankroll min/max validation on `number_input` ($100–$100k)~~ | §4.1 | ✅ Done — bundled with Phase 1 item #1 (2026-03-01) |
| NHL: live weather fetch for outdoor games | §4.6 | Open-Meteo already used for NFL — extend for NHL stadiums |
| NHL: injury feed from NHL API | §4.6 | NHL API has roster/injury data — needs `nhl_data_pipeline.py` |
| Line movement tracking (opening vs current line) | Appendix | Requires Odds API historical polling — not yet scoped |
| ~~Parlay builder~~ | Appendix | Superseded by Phase 5 — Recursive Parlay Ladder |
| Push notifications | Appendix | Not feasible in Streamlit — requires external service |
