# EdgeIQ — Product Requirements Document
**Status:** Draft v1.1 — 2026-03-01
**Purpose:** Define cross-sport standards, document what has been built, capture new requirements, and drive gap analysis.

---

## 1. Cross-Sport Constants (Standards Applied to Both NFL & NHL)

These are the patterns that are currently implemented consistently across **both** sports.

### 1.1 Model Architecture
| Constant | Value |
|----------|-------|
| Model type | Stacking ensemble: GBC + RF → LogReg meta-learner |
| Validation | TimeSeriesSplit (no data leakage) |
| O/U model | Ridge regression on residual (actual − Vegas line) |
| Output | Calibrated probability [0, 1] |

### 1.2 Kelly Criterion Bet Sizing
Both sports implement identical Kelly logic on game cards and in backtesting:

| Constant | Value |
|----------|-------|
| Formula | `kelly_pct = (b*p - q) / b` where b = decimal odds − 1, p = P(win), q = 1 − p |
| Default fraction | Half-Kelly (0.5×) |
| Fractions offered | Quarter (0.25×), Half (0.5×), Full (1.0×) |
| Max Kelly bet (safety cap) | 10% of bankroll |
| Min single bet | 1% of bankroll |
| Max single bet | 10% of bankroll |
| Max daily exposure | 25% of bankroll |
| Signal badges | 💎 STRONG ≥4% \| 📈 LEAN 2-4% \| 👀 SMALL 1-2% \| ⚪ PASS <1% |
| Edge definition | Model win prob − Vegas ML implied prob |
| Vig assumption | 4.55% (moneyline-derived when available; -110 fallback) |
| Risk tolerance multiplier | Conservative: 0.25× \| Moderate: 0.5× \| Aggressive: 1.0× |

#### Game Card Kelly UI Spec (both sports — standard layout)
```
#### 📐 Kelly Bet Sizing
─────────────────────────────────────────────────────────
  Model Edge ⓘ   │   Half-Kelly ⓘ   │  Bet Amount ⓘ  │  Signal ⓘ
    +21.3%        │      20.0%        │     $200        │  💎 STRONG
─────────────────────────────────────────────────────────
Betting on [TEAM] at XX.X% model confidence. [Moneyline: ±XXX. (NHL only)]
Vegas implied: XX.X%. Half-Kelly caps at 20% of bankroll to limit volatility.
```
- Rendered via `st.columns(4)` + `st.metric()` (one per column)
- Caption via `st.caption()` — sentence-separated with periods (not pipes)
- NFL: no moneyline in caption (derived from spread)
- NHL: includes `Moneyline: {ml}.` in caption since puck-line odds are explicit

### 1.11 Confidence Thresholds (Cross-Sport Standard)
| Threshold | Behavior |
|-----------|---------|
| < 55% confidence | Do not show as a bet recommendation |
| ≥ 55% | Show bet |
| > 65% | Flag as "strong bet" |
| > 75% | "Lock" badge |

### 1.12 Bankroll Constants (Cross-Sport Standard)
| Constant | Value |
|----------|-------|
| Default bankroll | $1,000 |
| Min bankroll | $100 |
| Max bankroll | $100,000 |
| Persistence | Streamlit session state (aligned to localStorage spec in Section 3.4) |

### 1.13 Bet Signal Color Palette (Cross-Sport Standard)
| Signal | Color |
|--------|-------|
| Strong bet / Win | `#22c55e` (green) |
| Decent bet | `#eab308` (yellow) |
| Skip bet | `#94a3b8` (gray) |
| Loss | `#ef4444` (red) |

### 1.3 Win Probability Display
| Constant | Value |
|----------|-------|
| Visual | Progress bar, Home % vs Away % |
| Confidence tiers | 🔥 HIGH >65% \| ✅ MODERATE 58–65% \| ⚠️ TOSS-UP <58% |
| Vegas comparison | Vegas ML implied prob shown alongside model prob |

### 1.4 Over/Under Prediction
| Constant | Value |
|----------|-------|
| Output | Model total vs Vegas line + OVER/UNDER lean + edge magnitude |
| Confidence tiers | Strong (≥4 pts/goals edge) \| Moderate (2–4) \| Slight (<2) |
| MAE shown | NFL: ±10.01 pts \| NHL: ±1.1 goals |

### 1.5 Weekly Schedule UI
| Constant | Value |
|----------|-------|
| View modes | "This Week's Games" (API-fed) vs "Manual Entry" |
| Game card layout | Expandable card per game, grouped by day |
| Pre-calculation | All game predictions computed on schedule load (precalc guard) |
| Expander controls | Expand All / Collapse All |
| Predicted winner | Shown on collapsed card label with win % badge |

### 1.6 Backtesting Tab
| Constant | Value |
|----------|-------|
| History depth | Last 5 seasons |
| Tables | By-season accuracy + game-by-game results (season-selectable) |
| Simulations | $10 flat moneyline bet + Kelly Criterion strategy |
| Charts | Cumulative P&L (Plotly) |
| Bankroll input | Sidebar widget per sport |

### 1.7 ELO Rating System
| Constant | Value |
|----------|-------|
| ELO update rule | Standard K=20 win/loss (NFL) \| K=6 (NHL, lower volatility) |
| Home advantage | 48 pts (NFL) \| 28 pts (NHL) |
| Trend window | 4 games rolling (optimal from ablation) |
| Season regression | Standard mean-regression at season start |

### 1.8 Implied Probability Features
| Constant | Value |
|----------|-------|
| Vegas spread → prob | Sigmoid conversion (scipy) |
| ELO → prob | Logistic sigmoid |
| Moneyline → prob | Standard American odds formula |
| Both pairs kept | Despite high correlation (r=−0.997); helps tree splits |

### 1.9 "Specialty Rating" (QB / Goalie)
| Constant | Value |
|----------|-------|
| Method | Per-(player, season) z-score from historical stats |
| Training file | `qb_ratings.csv` (NFL) \| `nhl_goalie_ratings.csv` (NHL) |
| Current season file | `qb_team_ratings.csv` \| `nhl_goalie_team_ratings.csv` |
| Unknown players | Default 0 (league average) |
| Feature name | `qb_score_diff` (NFL) \| `goalie_quality_diff` (NHL) |

### 1.10 Shared Infrastructure
| Component | Details |
|-----------|---------|
| Cache | `apis/cache.py` — JSON file cache with configurable TTL; used by ALL API modules |
| Entry router | `app.py` — sport selector → calls `render_nfl_app()` or `render_nhl_app()` |
| Session state | `sport`, `bankroll` / `nhl_bankroll`, `*_precalc_done`, `*_total_games` |
| Widget key namespacing | NFL: `g{idx}_*`; NHL: `nhl_g{idx}_*`; avoid collisions |
| Deployment | Streamlit Community Cloud, auto-redeploy on `git push master` |
| API keys | `RAPIDAPI_KEY` (NFL only), `ODDS_API_KEY` (NFL only); NHL has no key requirements |

---

## 2. What Has Been Built (Inventory by Sport)

### 2.1 NFL — Built Features

**Tabs:**
1. **Game Predictor + Lineups** — Weekly schedule or manual entry, live data fetch (weather/odds/injuries), 14-position offense + 10-position defense depth charts, defensive matchup engine, QB quality panel, O/U prediction, Kelly sizing
2. **Player Props** — Passing/Rushing/Receiving yards prediction vs Vegas prop line (MAE ±63/21/21 yds), opponent defensive EPA wired in
3. **Head-to-Head** — Historical matchup comparison (exists; detail TBD)
4. **Super Bowl Predictor** — Full 14-seed bracket simulation (wild card → SB), ELO-based per-matchup, outputs conference odds + SB odds
5. **Backtesting** — 5-season history, flat + Kelly simulation, Plotly P&L charts

**Unique to NFL:**
- 6-position defensive matchup engine (`defensive_matchup.py`) → ±4% win prob adjustment
- Player prop models (3 GBR models), opponent defensive EPA features
- Full bracket playoff simulator
- 5 live data APIs (ESPN, Tank01, Weather, Odds API, PFR)
- 26-feature stacking ensemble (69.3% holdout accuracy)

### 2.2 NHL — Built Features

**Tabs:**
1. **Game Predictor** — Weekly schedule (NHL API) or manual entry, goalie quality differential, O/U prediction (goals), Kelly sizing, team depth chart (fwd/def/goalie slots)
2. **Backtesting** — 5-season history, flat + Kelly simulation, Plotly P&L charts

**Unique to NHL:**
- Outdoor game detection (Winter Classic, Stadium Series)
- PP% / PK% power play features
- xG% (expected goals) shot metrics
- Day-keyed schedule ("Mon Mar 01") to prevent cross-week duplicates
- 29-feature stacking ensemble (58% holdout accuracy)
- NHL API integration (`apis/nhl.py`, api-web.nhle.com/v1)

---

## 3. User Requirements

### 3.1 Bankroll Management (Left Sidebar — Always Visible)

The sidebar is persistent across all sport views and contains four sub-sections:

#### 3.1.1 Bankroll Input
- Current bankroll amount as an editable field
- Validates: $100 – $100,000
- Persists across sessions (localStorage / session state)
- Default: $1,000

#### 3.1.2 Betting Strategy Selector
- Dropdown with four options:
  1. **Kelly Criterion** (default) — full formula, risk tolerance multiplier applied
  2. **Fixed %** — user sets a fixed percentage of bankroll per bet
  3. **Fixed $** — user sets a fixed dollar amount per bet
  4. **Fractional Kelly** — explicit 0.5× or 0.25× selector
- Tooltip on each option explaining the strategy

#### 3.1.3 Risk Tolerance Slider
- Three positions: **Conservative / Moderate / Aggressive**
- Maps to Kelly multiplier:
  - Conservative → 0.25× Kelly
  - Moderate → 0.5× Kelly
  - Aggressive → 1.0× Kelly

#### 3.1.4 Daily Summary (sidebar, below settings)
- Total recommended bets today: $XXX
- Total potential win: $XXX
- Total max loss: $XXX
- Expected value: +$XXX

---

### 3.2 Prediction Tab Enhancements (Both Sports)

#### 3.2.1 Head-to-Head Tab
- Historical matchup comparison (NFL: tab exists, detail TBD)
- Should be added to NHL section to match NFL parity

#### 3.2.2 Championship Simulator
- NFL: Super Bowl Predictor already exists (full 14-seed bracket)
- NHL: **Stanley Cup Predictor** — equivalent bracket simulator
  - 16 playoff seeds (8 per conference)
  - ELO-based per-matchup simulation
  - Outputs: conference odds + Stanley Cup odds per team

---

### 3.3 Prediction History

> **Distinct from Backtesting.** Backtesting re-runs the model on all historical data retroactively to measure model accuracy. Prediction History logs what EdgeIQ *actually predicted at the time you used it*, game by game, and fills in actual outcomes afterward so you can audit your edge in practice.

A cross-sport feature that auto-logs every prediction EdgeIQ generates for a real upcoming game and, once the game is played, shows the result alongside the prediction.

#### What gets logged (auto, no user action)
Every time a game card prediction is rendered or a manual entry prediction is run, save:

| Field | Description |
|-------|-------------|
| `sport` | "nfl" / "nhl" |
| `game_date` | Date of the game |
| `home_team` / `away_team` | Team names |
| `predicted_at` | Timestamp when EdgeIQ generated the prediction |
| `model_home_prob` | Model win probability for home team |
| `vegas_ml_home` | Vegas moneyline at prediction time (if available) |
| `vegas_implied_prob` | Vegas implied win prob at prediction time |
| `model_edge_pct` | model prob − Vegas implied (the "edge") |
| `kelly_signal` | STRONG / LEAN / SMALL / PASS |
| `kelly_pct` | Half-Kelly % recommended |
| `ou_line` | Vegas total at prediction time |
| `model_total` | EdgeIQ projected total |
| `ou_lean` | OVER / UNDER |

#### What gets filled in after the game
| Field | Description |
|-------|-------------|
| `actual_winner` | Home or Away |
| `actual_score_home` / `actual_score_away` | Final score |
| `actual_total` | Final combined score/goals |
| `prediction_correct` | Boolean |
| `ou_correct` | Boolean |

#### Results Review UI (cross-sport standard)
- Filterable table: by sport, date range, signal tier, correct/incorrect
- Columns: Date, Matchup, EdgeIQ Pick, Win Prob, Kelly Signal, Actual Result, ✅/❌
- Summary row: overall hit rate %, STRONG-only hit rate %, average edge when correct
- Link to Bet Tracker: if user logged a bet on this game, show bet amount + P&L inline
- Persisted as `prediction_history` JSON file on disk (same pattern as `user_bets`)

#### How this differs from Backtesting
| | Backtesting | Prediction History |
|--|-------------|-------------------|
| Data source | All historical games (re-run retroactively) | Only games you actually predicted while using EdgeIQ |
| Vegas line | Historical lines from stored data | Live line at the moment of prediction |
| Purpose | Validate model accuracy | Audit your real-world edge |
| User bets | Simulated | Actual bets logged in Bet Tracker |

---

### 3.4 Bet Tracker
A new tab (or sub-section) available in both sports:

- User can **log bets they actually placed** (team, amount, odds, bet type)
- Pulls from Prediction History as the base record — user adds: bet amount, actual odds taken, sportsbook
- Compare placed bets vs EdgeIQ recommendations
- Track P&L over time (cumulative chart)
- ROI calculation: `(total_won − total_staked) / total_staked`
- Persists in session state / localStorage as `user_bets` array (linked to `prediction_history` by game ID)

---

### 3.4 UI / UX Specifications

#### Layout (target)
```
|----------------------------------|
|  SIDEBAR   |   MAIN CONTENT      |
|  (280px)   |                     |
|            |  - Game predictions |
|  Bankroll  |  - Bet recommendations |
|  Settings  |  - Track record     |
|  Strategy  |  - etc.             |
|  Summary   |                     |
|            |                     |
|----------------------------------|
```

#### Responsive Behavior
- **Desktop:** sidebar always visible (280px fixed)
- **Tablet:** sidebar can be toggled
- **Mobile:** sidebar collapses to hamburger menu

---

### 3.5 Data Persistence

#### Session State / LocalStorage Keys (canonical names)
| Key | Type | Description |
|-----|------|-------------|
| `bankroll_amount` | number | Current bankroll |
| `betting_strategy` | string | Selected strategy name |
| `risk_tolerance` | string | "conservative" / "moderate" / "aggressive" |
| `user_bets` | array | Logged bet objects |
| `settings_version` | string | Schema version for migration |

#### Export / Import
- Allow user to export settings + bet history as JSON
- Import settings from file
- Useful for backup or switching devices

---

## 4. Gap Analysis

Key: ✅ Done | ⚠️ Partial | ❌ Missing | 🔲 New requirement

### 4.1 Bankroll Management Sidebar

| Requirement | NFL | NHL | Gap | Notes |
|-------------|-----|-----|-----|-------|
| Bankroll input widget | ✅ | ✅ | None — exists in both sidebars | Values not persisted across page reloads |
| Betting strategy selector (4 options) | ⚠️ Kelly only | ⚠️ Kelly only | ❌ Fixed %, Fixed $, Fractional Kelly modes not implemented | New logic needed in both sports |
| Risk tolerance slider (3 positions) | ❌ | ❌ | Full build required | Maps to Kelly multiplier; replaces current quarter/half/full dropdown in backtesting |
| Daily summary panel | ❌ | ❌ | Full build required | Requires aggregating Kelly recommendations across all game cards loaded that day |
| Sidebar persistence (session state) | ✅ Streamlit state | ✅ Streamlit state | localStorage persistence not available in Streamlit; session state resets on reload | Workaround: store in URL params or user-keyed file |
| Bankroll min/max validation ($100–$100k) | ❌ | ❌ | Minor — add to both `number_input` widgets | 1-line fix per sport |

**Build priority:** Medium. Betting strategy selector + risk tolerance are the highest-value items. Daily summary requires all game cards to be pre-calculated first (already done via precalc).

---

### 4.2 Kelly Criterion Logic Updates

| Requirement | Current State | Gap |
|-------------|--------------|-----|
| Formula: `(b*p - q) / b` | ✅ Correct in both sports | None |
| 10% max Kelly cap | ❌ NFL: 20% soft cap; NHL: 0.5% hard cap | Inconsistent — standardize to 10% |
| 1% min bet | ❌ Not enforced | Add floor to both |
| 10% max single bet | ❌ Not enforced | Add ceiling to both |
| 25% max daily exposure | ❌ Not tracked | Requires daily summary aggregator |
| Risk tolerance multiplier | ❌ | New — replaces current static 0.5× default |

---

### 4.3 Confidence Thresholds

| Requirement | Current State | Gap |
|-------------|--------------|-----|
| Only show bets > 55% confidence | ❌ All probabilities shown | Filter needed on game cards + bet recommendations |
| Flag "strong bet" > 65% | ⚠️ 🔥 HIGH tier exists at >65% | ✅ Tier is correct; badge text can be aligned to "Strong Bet" language |
| "Lock" badge > 75% | ❌ Not implemented | New tier added to win prob display in both sports |

---

### 4.4 Bet Signal Color Palette

| Requirement | Current State | Gap |
|-------------|--------------|-----|
| Strong/Win: `#22c55e` | ⚠️ Streamlit default green used | Standardize with `st.markdown` + custom CSS |
| Decent: `#eab308` | ⚠️ Streamlit default | Standardize |
| Skip: `#94a3b8` | ⚠️ Streamlit default | Standardize |
| Loss: `#ef4444` | ⚠️ Streamlit default red | Standardize |

**Note:** Streamlit's theming is limited; custom HTML/CSS via `st.markdown(unsafe_allow_html=True)` is the path forward.

---

### 4.5 Prediction History

| Requirement | Current State | Gap |
|-------------|--------------|-----|
| Auto-log predictions at render time | ❌ | Hook into `render_prediction_result()` (NFL) and `render_nhl_prediction_result()` (NHL) — append to `prediction_history.json` |
| Store Vegas line at prediction time | ⚠️ Live odds fetched but not persisted | Capture snapshot into the log record at prediction time |
| Fill in actual results after game | ❌ | Needs a results-fetch job (ESPN scoreboard already in `apis/espn.py`) or manual entry |
| Results review UI (filterable table) | ❌ | New tab in both sports — `st.dataframe` with filter controls |
| Summary stats (hit rate, edge quality) | ❌ | Derived from `prediction_history.json` — straightforward aggregation |
| Link to Bet Tracker entries | ❌ | Join on game_id between `prediction_history` and `user_bets` |
| Persist to disk | ❌ | JSON file `prediction_history.json` (same pattern as `user_bets`) |
| Export as JSON/CSV | ❌ | `st.download_button` — trivial |

**Key design note:** The logging hook goes in the two `render_*_prediction_result()` functions. Each prediction generates one record immediately. Actual results get back-filled either via the ESPN scoreboard API (already built) or via a nightly update pattern. This is the most architecturally significant new feature — it requires careful deduplication (same game loaded multiple times should update, not append).

**Build priority:** High. This is the feature that makes EdgeIQ a tool you come *back* to, not just a pre-game lookup. Prediction History + Bet Tracker together form the "track record" layer of the product.

---

### 4.6 Bet Tracker

| Requirement | Current State | Gap |
|-------------|--------------|-----|
| Log placed bets | ❌ | Full build — new tab or sidebar section; pulls base record from Prediction History |
| Link to Prediction History | ❌ | Join on game_id |
| P&L over time chart | ❌ | Plotly chart (same pattern as backtesting tab) |
| ROI calculation | ❌ | `(total_won − total_staked) / total_staked` |
| Persist `user_bets` | ❌ | JSON file on disk; session state only until then |
| Export / Import as JSON | ❌ | `st.download_button` / `st.file_uploader` — straightforward |

**Build priority:** Medium-high. Depends on Prediction History (§4.5) for the base game record.

---

### 4.6 NHL Parity Gaps (vs NFL)

| Feature | NFL | NHL | Gap |
|---------|-----|-----|-----|
| Head-to-Head tab | ✅ | ❌ | Build NHL H2H tab matching NFL |
| Championship simulator | ✅ Super Bowl | ❌ | Build Stanley Cup Predictor (16-seed bracket) |
| Vegas odds integration (Odds API) | ✅ | ❌ | Wire `apis/odds.py` into `nhl_app.py` |
| Live weather fetch | ✅ | ❌ | Add for outdoor games (Winter Classic uses stadium) |
| Injury feed | ✅ ESPN+Tank01 | ❌ | NHL injuries available via NHL API — add to pipeline |
| Player props tab | ✅ 3 models | ❌ | NHL props: goals, assists, shots on goal |
| Positional matchup engine | ✅ 6-position | ❌ | NHL: line matchups (top line vs top pairing) |

---

### 4.7 UI / UX — Responsive Layout

| Requirement | Current State | Gap |
|-------------|--------------|-----|
| 280px fixed sidebar (desktop) | ⚠️ Streamlit default sidebar, not fixed-width | CSS override possible but limited in Streamlit |
| Mobile hamburger collapse | ⚠️ Streamlit has native mobile collapse | Functional but not styled per spec |
| Tablet toggle | ⚠️ Same as mobile collapse | No custom toggle UI |

**Note:** Streamlit's layout control is constrained. Full sidebar width control and custom responsive behavior would require migrating to a custom frontend (React/Next.js) or using Streamlit's `st.columns` layout instead of the native sidebar.

---

### 4.8 Data Persistence

| Requirement | Current State | Gap |
|-------------|--------------|-----|
| `bankroll_amount` persists across reloads | ❌ Session state resets | Use `st.query_params` or JSON file in project dir |
| `betting_strategy` persists | ❌ | Same |
| `risk_tolerance` persists | ❌ | Same |
| `user_bets` array persists | ❌ | JSON file on disk (acceptable since single-user) |
| Export settings as JSON | ❌ | `st.download_button(data=json.dumps(...))` — trivial |
| Import settings from JSON | ❌ | `st.file_uploader` → `json.load()` — trivial |

---

### 4.9 Recommended Build Order

| Priority | Item | Effort | Dependency |
|----------|------|--------|------------|
| 1 | Standardize Kelly cap (10%), min/max bet limits | Low | None |
| 2 | Risk tolerance slider → Kelly multiplier (both sports) | Low | None |
| 3 | Betting strategy selector (4 options) | Medium | Risk tolerance |
| 4 | "Lock" badge at >75% confidence | Low | None |
| 5 | Bet signal color palette (custom CSS) | Low | None |
| 6 | **Prediction History** — auto-log predictions + results review tab (both sports) | Medium | Both render functions; ESPN scoreboard for results backfill |
| 7 | Daily summary panel in sidebar | Medium | Pre-calc (already done) |
| 8 | Bet Tracker tab (log + P&L + ROI) | Medium | Prediction History (§4.5) |
| 9 | JSON export/import (bankroll + bets + history) | Low | Bet Tracker |
| 10 | NHL: wire Odds API for live moneylines | Medium | `apis/odds.py` (already built) |
| 11 | NHL: Stanley Cup Predictor tab | Medium | Mirrors NFL Super Bowl code |
| 12 | NHL: Head-to-Head tab | Medium | Mirrors NFL H2H code |
| 13 | NHL: Player props (goals/assists/shots) | High | New training data + models |
| 14 | NHL: Positional line matchup engine | High | New feature engineering |

---

## Appendix: Feature Comparison Matrix

| Feature | NFL | NHL | Standard? |
|---------|-----|-----|-----------|
| Weekly schedule view | ✅ | ✅ | ✅ Yes |
| Manual entry mode | ✅ | ✅ | ✅ Yes |
| Win probability (bar + tiers) | ✅ | ✅ | ✅ Yes |
| Vegas ML implied prob | ✅ | ✅ | ✅ Yes |
| O/U prediction + lean | ✅ | ✅ | ✅ Yes |
| Kelly bet sizing (game cards) | ✅ | ✅ | ✅ Yes |
| Kelly backtesting simulation | ✅ | ✅ | ✅ Yes |
| Predicted winner on card label | ✅ | ✅ | ✅ Yes |
| Expand All / Collapse All | ✅ | ✅ | ✅ Yes |
| 5-season backtesting | ✅ | ✅ | ✅ Yes |
| Specialty rating (QB/Goalie) | ✅ QB | ✅ Goalie | ✅ Yes |
| ELO rating system | ✅ | ✅ | ✅ Yes |
| Stacking ensemble model | ✅ | ✅ | ✅ Yes |
| Depth chart / lineup view | ✅ 14+10 pos | ✅ 6fwd+4def+2G | ✅ Yes |
| Live data APIs (weather/odds) | ✅ (5 APIs) | ❌ (NHL API only) | ⚠️ Gap |
| Vegas odds integration | ✅ Odds API | ❌ Manual input only | ⚠️ Gap |
| Player props tab | ✅ 3 models | ❌ None | ⚠️ Gap |
| Positional matchup engine | ✅ 6 positions | ❌ None | ⚠️ Gap |
| Championship simulator | ✅ Super Bowl | ❌ None | ⚠️ Gap |
| Head-to-head tab | ✅ | ❌ | ⚠️ Gap |
| Live weather fetch | ✅ Open-Meteo | ❌ | ⚠️ Gap |
| Injury feed | ✅ ESPN+Tank01 | ❌ | ⚠️ Gap |
| Line movement tracking | ❌ | ❌ | 🔲 Planned |
| Parlay builder | ❌ | ❌ | 🔲 Planned |
| Prediction History (auto-log + results review) | ❌ | ❌ | 🔲 §3.3 — medium build, cross-sport |
| Bet tracker / P&L log | ❌ | ❌ | 🔲 §3.4 — medium build (depends on §3.3) |
| JSON export/import (settings + bets) | ❌ | ❌ | 🔲 §3.5 — trivial |
| Betting strategy selector (4 modes) | ✅ | ✅ | ✅ Done — Phase 1 |
| Risk tolerance slider | ✅ | ✅ | ✅ Done — Phase 1 |
| Daily bet summary (sidebar) | ❌ | ❌ | 🔲 §3.1.4 — medium build |
| "Lock" badge >75% confidence | ✅ | ✅ | ✅ Done — Phase 1 |
| Standardized bet signal colors | ✅ | ✅ | ✅ Done — Phase 1 |
| Bankroll min/max validation | ✅ | ✅ | ✅ Done — Phase 1 (bundled) |
| Standardized Kelly cap (10%) | ✅ | ✅ | ✅ Done — Phase 1 |
| Stanley Cup Predictor tab | ❌ | ❌ | 🔲 §3.2.2 — medium build |
| NHL Head-to-Head tab | ✅ | ❌ | ⚠️ §4.6 — medium build |
| NHL Vegas odds integration | ✅ | ❌ | ⚠️ §4.6 — medium build |
| NHL player props | ✅ | ❌ | ⚠️ §4.6 — high build |
| NHL positional matchup engine | ✅ | ❌ | ⚠️ §4.6 — high build |
| Push notifications | ❌ | ❌ | 🔲 Future |
