# EdgeIQ Rework — Requirements

## v1 Requirements (Committed Scope)

### Caching / Performance (CACHE)

- [x] **CACHE-01** — Move NFL module-level data loads inside `render_nfl_app()` to avoid re-execution on every Streamlit rerun
- [x] **CACHE-02** — Add `@st.cache_data` to CSS file read in `app.py` so the file isn't re-read on every rerun
- [x] **CACHE-03** — Wrap `MLBClient()` in `@st.cache_resource` instead of creating a new instance every rerun
- [x] **CACHE-04** — Replace `NHLClient` session-state pattern with `@st.cache_resource` singleton
- [x] **CACHE-05** — Add "Load / Refresh Schedule" button to NHL (stop auto-fetching schedule + odds + precalc on first visit)

### Architecture (ARCH)

- [ ] **ARCH-01** — Create `core/` package with shared pure-Python modules: `kelly.py`, `elo.py`, `odds.py`, `backtest.py`
- [ ] **ARCH-02** — Extract Kelly criterion into single `core/kelly.py` replacing `_kelly_rec()`, `_nhl_kelly()`, `_mlb_kelly()` (3→1)
- [ ] **ARCH-03** — Extract ELO computation into `core/elo.py` (shared across all 3 sports)
- [ ] **ARCH-04** — Extract odds conversion (American↔implied prob, vig removal) into `core/odds.py`
- [ ] **ARCH-05** — Extract backtesting simulation engine into `core/backtest.py` (computation only, no Plotly)
- [ ] **ARCH-06** — Create `sports/` package structure: `sports/nfl/`, `sports/nhl/`, `sports/mlb/` with `config.py`, `features.py`, `schedule.py`
- [ ] **ARCH-07** — Move NFL feature engineering from `feature_engineering.py` → `sports/nfl/features.py`
- [ ] **ARCH-08** — Move NHL feature engineering from `nhl_feature_engineering.py` → `sports/nhl/features.py`
- [ ] **ARCH-09** — Move MLB feature engineering from `mlb_feature_engineering.py` → `sports/mlb/features.py`
- [ ] **ARCH-10** — Split `final_app.py` (3,627 lines) into per-tab modules under `sports/nfl/tabs/`
- [ ] **ARCH-11** — Split `nhl_app.py` (2,731 lines) into per-tab modules under `sports/nhl/tabs/`
- [ ] **ARCH-12** — Split `mlb_app.py` (1,767 lines) into per-tab modules under `sports/mlb/tabs/`
- [ ] **ARCH-13** — Move API modules from `apis/` to remain as-is (already well-structured)
- [ ] **ARCH-14** — Create `training/` package for offline model training scripts with shared harness
- [ ] **ARCH-15** — Update `app.py` entry point to import from new package structure
- [ ] **ARCH-16** — Ensure all `.pkl` model files load correctly from new paths (no retraining)
- [ ] **ARCH-17** — Create sport-specific `config.py` files (feature lists, model paths, display constants)

### UI/Design System (UI)

- [ ] **UI-01** — Create `.streamlit/config.toml` with EdgeIQ color palette (primary=#22d3ee, bg=#0f172a)
- [ ] **UI-02** — Expand CSS design tokens from 7 to ~30 (semantic colors, spacing scale, border radius, shadows, typography)
- [ ] **UI-03** — Hide Streamlit chrome (hamburger menu, footer, toolbar) via CSS
- [ ] **UI-04** — Consolidate all inline `st.markdown(style=...)` calls into CSS classes
- [ ] **UI-05** — Professional tab navigation styling (hover, active, transitions)
- [ ] **UI-06** — Consistent game card component layout across all sports (shared structure, sport-specific content slots)
- [ ] **UI-07** — Consistent sidebar design: branding, sport-specific controls, bankroll inputs
- [ ] **UI-08** — Loading states and skeleton screens for data fetching
- [ ] **UI-09** — Standardized signal badge rendering (STRONG/LEAN/SMALL/PASS) as CSS classes
- [ ] **UI-10** — Standardized confidence tier badges (LOCK/HIGH/MOD/TOSS-UP) as CSS classes

### Testing (TEST)

- [ ] **TEST-01** — Set up pytest infrastructure: `tests/conftest.py`, `pytest.ini`, sample fixtures
- [ ] **TEST-02** — Unit tests for `core/kelly.py` (parametrized: edge cases, caps, all strategies)
- [ ] **TEST-03** — Unit tests for `core/odds.py` (American↔implied, vig removal, edge cases)
- [ ] **TEST-04** — Unit tests for `core/elo.py` (rating updates, expected score, initial ratings)
- [ ] **TEST-05** — Unit tests for `core/backtest.py` (simulation accuracy, Kelly sizing, P&L)
- [ ] **TEST-06** — Unit tests for `parlay_math.py` (accumulation, leg removal, payout)
- [ ] **TEST-07** — Contract tests for feature engineering (column presence, NaN detection, row preservation)
- [ ] **TEST-08** — Unit tests for abbreviation mapping (`_stats_key()`, `_NAME_TO_FG`)
- [ ] **TEST-09** — Mock model tests for prediction pipelines (shape, probability range, feature count)
- [ ] **TEST-10** — API response fixture tests (mock ESPN, NHL, MLB, Odds API responses)

## v2 Requirements (Deferred)

### Enhanced Features
- **V2-FEAT-01** — Live score integration and auto-refresh during games
- **V2-FEAT-02** — Cross-sport parlay builder (combine NFL + NHL + MLB legs)
- **V2-FEAT-03** — Automated bet tracking with results verification
- **V2-FEAT-04** — Historical performance dashboard with date range filters

### Model Improvements
- **V2-MODEL-01** — Explore neural network models (if accuracy ceiling can be raised)
- **V2-MODEL-02** — Add more player prop categories (NFL receiving TDs, NHL power play points)
- **V2-MODEL-03** — Real-time odds movement tracking and line value detection

### Platform
- **V2-PLAT-01** — User accounts and saved preferences
- **V2-PLAT-02** — Mobile-optimized responsive layout
- **V2-PLAT-03** — Email/push notifications for high-value bets

## Out of Scope (with reasoning)

| Exclusion | Reasoning |
|-----------|-----------|
| Framework migration (React/Next.js) | Decided to stay on Streamlit — sufficient for SaaS look with custom CSS |
| New sports (NBA, soccer) | Get architecture right for 3 sports first |
| Model retraining during rework | Models at practical ceiling; rework is about code quality + UI, not accuracy |
| Paid data sources | Free data constraint is a hard project constraint |
| Backend API server | Streamlit handles everything; no need for separate Flask/FastAPI |
| CI/CD pipeline | Solo developer; manual testing + Streamlit Cloud auto-deploy is sufficient |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CACHE-01 to CACHE-05 | Phase 0: Caching Fixes | ✅ Done |
| ARCH-01 to ARCH-05 | Phase 1: Core Extraction | 🔲 Not Started |
| ARCH-06 to ARCH-09 | Phase 2: Sports Packages | 🔲 Not Started |
| ARCH-10 to ARCH-12 | Phase 3: Tab Splitting | 🔲 Not Started |
| ARCH-13 to ARCH-17 | Phase 3: Tab Splitting | 🔲 Not Started |
| UI-01 to UI-05 | Phase 4: Design System | 🔲 Not Started |
| UI-06 to UI-10 | Phase 5: UI Components | 🔲 Not Started |
| TEST-01 to TEST-06 | Phase 6: Testing | 🔲 Not Started |
| TEST-07 to TEST-10 | Phase 6: Testing | 🔲 Not Started |

**Coverage:** 42 total requirements · 42 mapped · 0 unmapped
