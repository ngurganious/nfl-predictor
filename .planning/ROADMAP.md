# EdgeIQ Rework — Roadmap

## Milestone 1: Architecture + UI Rework

### Phase 0: Caching Fixes
> Fix 5 Streamlit caching issues causing unnecessary work on every rerun.

**Requirements:** CACHE-01, CACHE-02, CACHE-03, CACHE-04, CACHE-05

**Acceptance Criteria:**
- NFL module-level loads moved inside `render_nfl_app()` (no re-execution on rerun)
- CSS file read cached with `@st.cache_data`
- MLBClient wrapped in `@st.cache_resource`
- NHLClient wrapped in `@st.cache_resource` (replaces session state pattern)
- NHL schedule requires button click (no auto-fetch on first visit)

**Status:** ✅ Done — 2026-03-05

---

### Phase 0.5: Homepage Redesign
> DraftKings-inspired homepage with sport icon bar, top picks table, and new tagline.

**Requirements:** HOME-01, HOME-02, HOME-03

**Acceptance Criteria:**
- 8 sport icons in horizontal bar (3 active, 5 grayed "Coming Soon")
- Active icon click navigates to sport, inactive shows toast
- Top 10 picks table populates from session state (game + prop predictions)
- New tagline: "Vegas sets the line. We find the gaps."
- Sport detail cards retained below icon bar

**Status:** ✅ Done — 2026-03-06

---

### Phase 0.75: Game Card Standardization
> Standardize collapsed labels, conditions sections, lineups, and inline props across all sports.

**Requirements:** CARD-01, CARD-02, CARD-03, CARD-04, CARD-05, CARD-06, CARD-07

**Acceptance Criteria:**
- NHL + MLB collapsed labels match NFL format (pipe-separated: ML | O/U | Time | Pred + Kelly%)
- NHL conditions: 3-col (Venue/Indoor, Lines, Goalie preview)
- MLB conditions: 3-col (Venue/Day-Night, SP panels, Vegas lines)
- MLB batting lineups: 9 position selectboxes (C/1B/2B/3B/SS/LF/CF/RF/DH)
- NHL + MLB inline props: "Show Player Props" toggle with compact table per game card
- NFL game cards: caption pointing to Props tab

**Status:** ✅ Done — 2026-03-06

---

### Phase 1: Core Extraction
> Extract shared pure-Python logic into `core/` package. Lowest risk, highest deduplication value.

**Requirements:** ARCH-01, ARCH-02, ARCH-03, ARCH-04, ARCH-05

**Acceptance Criteria:**
- `core/kelly.py` replaces 3 inline Kelly functions across NFL/NHL/MLB
- `core/elo.py` provides shared ELO computation
- `core/odds.py` handles all American↔implied probability conversions
- `core/backtest.py` runs backtesting simulation (computation only, no rendering)
- All existing app functionality unchanged (regression-free)
- All sport app files import from `core/` instead of inline implementations

**Status:** 🔲 Not Started

---

### Phase 2: Sports Package Structure
> Reorganize sport-specific code into `sports/{sport}/` packages with clean separation.

**Requirements:** ARCH-06, ARCH-07, ARCH-08, ARCH-09, ARCH-17

**Acceptance Criteria:**
- `sports/nfl/`, `sports/nhl/`, `sports/mlb/` packages created with `__init__.py`
- Feature engineering moved to `sports/{sport}/features.py`
- Schedule/game week moved to `sports/{sport}/schedule.py`
- Sport-specific config (feature lists, model paths, constants) in `sports/{sport}/config.py`
- All imports updated — no broken references
- App still runs identically

**Status:** 🔲 Not Started

**Depends on:** Phase 1

---

### Phase 3: Monolith Splitting
> Break the 3 monolithic app files into per-tab modules (~200-400 lines each).

**Requirements:** ARCH-10, ARCH-11, ARCH-12, ARCH-13, ARCH-14, ARCH-15, ARCH-16

**Acceptance Criteria:**
- `final_app.py` (3,627 lines) → `sports/nfl/tabs/` (predictor, backtest, props, parlay, track_record)
- `nhl_app.py` (2,731 lines) → `sports/nhl/tabs/` (predictor, backtest, props, parlay, track_record)
- `mlb_app.py` (1,767 lines) → `sports/mlb/tabs/` (predictor, backtest, props, parlay, track_record)
- `training/` package created for `build_*.py` and `retrain_*.py` scripts
- `app.py` entry point updated to route to new tab modules
- All `.pkl` models load from correct paths
- All Streamlit widget keys remain unique (no collisions)

**Status:** 🔲 Not Started

**Depends on:** Phase 2

---

### Phase 4: UX/UI Redesign
> Premium dark-mode Bloomberg-terminal redesign. No ML model, prediction logic, or API changes — purely UI/UX.
> Full spec: EdgestackIQ UX/UI Redesign Spec (user-provided, 13 implementation steps).

**Requirements:** UI-01 through UI-10

**Acceptance Criteria (13 steps, in priority order):**

| # | Step | Files | Status |
|---|------|-------|--------|
| 1 | CSS design tokens — `assets/style.css` + `.streamlit/config.toml` | `assets/style.css`, `.streamlit/config.toml` | ✅ Done — 2026-03-09 |
| 2 | Signal badge CSS classes — replace all inline badge styles | `final_app.py`, `nhl_app.py`, `mlb_app.py` | 🔲 Not Started |
| 3 | Homepage cleanup — remove sport cards + value props grid | `app.py` | 🔲 Not Started |
| 4 | Sportsbook selector — add to header, wire to session state | `app.py` | 🔲 Not Started |
| 5 | Sport icon availability logic — dim/coming-soon states | `app.py` | 🔲 Not Started |
| 6 | Top 10 pre-load — background fetch on app load for ACTIVE sports | `app.py` | 🔲 Not Started |
| 7 | Parlay tray session state — add `parlay_tray` + `parlay_tray_open` | `app.py` | 🔲 Not Started |
| 8 | Floating tray UI — collapsed bar + expanded bottom sheet | `app.py`, `assets/style.css` | 🔲 Not Started |
| 9 | Tab restructure — remove Track Record + Super Bowl, reorder tabs | `final_app.py`, `nhl_app.py`, `mlb_app.py` | 🔲 Not Started |
| 10 | Back navigation — "← Sports" button on all sport views | `final_app.py`, `nhl_app.py`, `mlb_app.py` | 🔲 Not Started |
| 11 | "Add to Parlay Tray" buttons — wire Game Prediction + Props to tray | `final_app.py`, `nhl_app.py`, `mlb_app.py` | 🔲 Not Started |
| 12 | Cross-sport ladder header — multi-sport parlay detection | `final_app.py`, `nhl_app.py`, `mlb_app.py` | 🔲 Not Started |
| 13 | Head-to-Head tab — add to NHL and MLB (already exists in NFL) | `nhl_app.py`, `mlb_app.py` | 🔲 Not Started |

**Status:** 🚧 In Progress (1/13 steps done)

**Depends on:** None (UI-only, no architecture refactoring required)

---

### Phase 6: Test Suite
> Build pragmatic test coverage for core logic and contracts.

**Requirements:** TEST-01 through TEST-10

**Acceptance Criteria:**
- pytest infrastructure: `tests/conftest.py`, `pytest.ini`, fixture files
- `core/` fully tested: Kelly, ELO, odds, backtest, parlay math
- Feature engineering contract tests: column presence, NaN detection, row preservation
- Abbreviation mapping tests (`_stats_key()`, `_NAME_TO_FG`)
- Mock model pipeline tests: prediction shape, probability range
- API fixture tests: mock responses for all external APIs
- All tests pass with `pytest` from project root

**Status:** 🔲 Not Started

**Depends on:** Phase 1 (core must be extracted before testing it)

---

## Milestone 2: Player Props Overhaul

### Phase 7: Props API + Edge Engine
> Extend Odds API client for all 3 sports. Standardize edge calculation. Add sportsbook selector + quota display.

**Requirements:** PROPS-01 to PROPS-10

**Acceptance Criteria:**
- `apis/odds.py` supports MLB sport key, events, game odds, and player props
- `get_player_props()` accepts sport_key + bookmakers parameter for all 3 sports
- `get_alternate_props()` method for on-demand alternate line scanning
- Standardized `_calc_prop_edge()` function replaces 3 ad-hoc confidence calculations
- Sportsbook dropdown (DraftKings, FanDuel, BetMGM, Caesars, PointsBet, Bovada) persists in session state
- Quota header displays "API Credits: X / Y used" after every API call
- Season gating disables prop fetch for sports with no scheduled games

**Status:** ✅ Done — 2026-03-06

---

### Phase 8: Props Tab Redesign
> Rewrite all 3 sport Props tabs to fetch real sportsbook lines, match against model predictions, show edge table, and feed real odds to parlay ladder.

**Requirements:** PROPS-11 to PROPS-19

**Acceptance Criteria:**
- Each sport's Props tab: sportsbook selector → "Fetch Prop Lines" button → edge table
- Edge table columns: Player | Prop | Line | Pred | Edge% | Odds | Value | Signal
- Top Picks ranked by value_score (edge × model_prob), not raw confidence
- "Scan Alternates" button per game for on-demand alternate line scanning
- "Model Only — No Live Line" indicator where sportsbook data is missing
- Signal badges (STRONG/LEAN/SMALL/PASS) based on edge_pct
- Parlay legs carry real sportsbook odds and book name
- NFL, NHL, MLB all use identical edge calculation + display patterns

**Status:** 🔲 Not Started

**Depends on:** Phase 7

---

### Phase 9: Model Enhancement
> Add matchup-specific features to prop models across all sports, retrain, validate improvement.

**Requirements:** MODEL-01 to MODEL-07

**Acceptance Criteria:**
- MLB batter models include opposing SP quality features → hits MAE < 0.68, TB MAE < 1.40
- MLB park factors built and integrated into prop + game models
- NHL prop models include opposing goalie quality features → goals MAE < 0.41
- NHL prop models include `is_outdoor` venue feature
- All modified models retrained with validated MAE improvement over baseline
- Old .pkl files kept as backup until validated

**Status:** 🔲 Not Started

**Depends on:** Phase 8

---

## Phase Summary

| Phase | Name | Requirements | Effort | Risk | Status |
|-------|------|-------------|--------|------|--------|
| 0 | Caching Fixes | CACHE-01 to CACHE-05 | Low | Low | ✅ |
| 0.5 | Homepage Redesign | HOME-01 to HOME-03 | Low | Low | ✅ |
| 0.75 | Card Standardization | CARD-01 to CARD-07 | Medium | Low | ✅ |
| 1 | Core Extraction | ARCH-01 to ARCH-05 | Medium | Low | 🔲 |
| 2 | Sports Packages | ARCH-06 to ARCH-09, ARCH-17 | Medium | Low | 🔲 |
| 3 | Monolith Splitting | ARCH-10 to ARCH-16 | High | Medium | 🔲 |
| 4 | UX/UI Redesign (13 steps) | UI-01 to UI-10 | High | Low | 🚧 1/13 |
| 6 | Test Suite | TEST-01 to TEST-10 | Medium | Low | 🔲 |
| 7 | Props API + Edge Engine | PROPS-01 to PROPS-10 | Medium | Low | ✅ |
| 8 | Props Tab Redesign | PROPS-11 to PROPS-19 | High | Medium | 🔲 |
| 9 | Model Enhancement | MODEL-01 to MODEL-07 | High | Medium | 🔲 |
