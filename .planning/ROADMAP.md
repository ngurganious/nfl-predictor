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

### Phase 4: Design System Foundation
> Create the CSS design system and Streamlit theming for a professional SaaS look.

**Requirements:** UI-01, UI-02, UI-03, UI-04, UI-05

**Acceptance Criteria:**
- `.streamlit/config.toml` created with EdgeIQ palette
- CSS design tokens expanded to ~30 variables (colors, spacing, borders, shadows, typography)
- Streamlit chrome hidden (hamburger, footer, toolbar)
- All inline `st.markdown(style=...)` consolidated into CSS classes
- Tab navigation has professional styling (hover, active states, transitions)
- App looks noticeably more professional without any layout changes

**Status:** 🔲 Not Started

**Depends on:** Phase 3 (so CSS classes reference final file structure)

---

### Phase 5: UI Component Polish
> Build shared UI components and apply consistent design across all sports.

**Requirements:** UI-06, UI-07, UI-08, UI-09, UI-10

**Acceptance Criteria:**
- Shared game card component with sport-specific content slots
- Consistent sidebar branding and controls across all sports
- Loading states / skeleton screens during data fetches
- Signal badges (STRONG/LEAN/SMALL/PASS) rendered via CSS classes, not inline styles
- Confidence tier badges (LOCK/HIGH/MOD/TOSS-UP) rendered via CSS classes
- Visual consistency verified across NFL, NHL, MLB tabs

**Status:** 🔲 Not Started

**Depends on:** Phase 4

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

## Phase Summary

| Phase | Name | Requirements | Effort | Risk | Status |
|-------|------|-------------|--------|------|--------|
| 0 | Caching Fixes | CACHE-01 to CACHE-05 | Low | Low | ✅ |
| 0.5 | Homepage Redesign | HOME-01 to HOME-03 | Low | Low | ✅ |
| 1 | Core Extraction | ARCH-01 to ARCH-05 | Medium | Low | 🔲 |
| 2 | Sports Packages | ARCH-06 to ARCH-09, ARCH-17 | Medium | Low | 🔲 |
| 3 | Monolith Splitting | ARCH-10 to ARCH-16 | High | Medium | 🔲 |
| 4 | Design System | UI-01 to UI-05 | Medium | Low | 🔲 |
| 5 | UI Components | UI-06 to UI-10 | Medium | Low | 🔲 |
| 6 | Test Suite | TEST-01 to TEST-10 | Medium | Low | 🔲 |
