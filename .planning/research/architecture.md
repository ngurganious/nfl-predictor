# Architecture Research: Multi-Domain ML Prediction Platform

> Research on best practices for structuring EdgeIQ — a multi-sport ML prediction platform
> with NFL, NHL, and MLB domains sharing common patterns.
> Date: 2026-03-05

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Modular Directory Structures](#2-modular-directory-structures)
3. [Abstract Base Classes / Shared Interfaces](#3-abstract-base-classes--shared-interfaces)
4. [DRY Patterns for Pipelines](#4-dry-patterns-for-pipelines)
5. [Configuration-Driven vs Inheritance-Driven](#5-configuration-driven-vs-inheritance-driven)
6. [Clean Layer Separation](#6-clean-layer-separation)
7. [File Organization: Flat vs Nested](#7-file-organization-flat-vs-nested)
8. [Shared Utility Patterns](#8-shared-utility-patterns)
9. [Recommended Architecture for EdgeIQ](#9-recommended-architecture-for-edgeiq)
10. [Migration Strategy](#10-migration-strategy)

---

## 1. Current State Analysis

### What exists today

The project is a flat directory with 80+ files at root level. Each sport has:

| Component | NFL | NHL | MLB |
|-----------|-----|-----|-----|
| App (UI) | `final_app.py` (3,627 lines) | `nhl_app.py` (2,731 lines) | `mlb_app.py` (1,767 lines) |
| Feature Engineering | `feature_engineering.py` (638 lines) | `nhl_feature_engineering.py` (463 lines) | `mlb_feature_engineering.py` (404 lines) |
| Game Week / Schedule | `game_week.py` (386 lines) | `nhl_game_week.py` (287 lines) | `mlb_game_week.py` (208 lines) |
| Build Model | `retrain_model.py` | `build_nhl_model.py` | `build_mlb_model.py` |
| Build Games | `build_model.py` | `build_nhl_games.py` | `build_mlb_games.py` |
| Build Ratings | `build_qb_ratings.py` | `build_nhl_goalie_ratings.py` | `build_mlb_pitcher_ratings.py` |
| Build Team Stats | `build_team_stats.py` | `build_nhl_team_stats.py` | `build_mlb_team_stats.py` |
| Build Player Model | `build_player_model.py` | `build_nhl_player_model.py` | `build_mlb_player_model.py` |
| Build Total Model | `build_total_model.py` | (in build_nhl_model) | `build_mlb_total_model.py` |

Plus shared modules: `app.py` (router), `parlay_math.py`, `prediction_history.py`, `data_pipeline.py`, `defensive_matchup.py`, `backtest.py`, `style.css`.

**Total app code: ~8,100 lines across 3 monolithic sport files.**

### Identified duplication

The following logic is duplicated 3 times (once per sport):

1. **Kelly criterion** — `_kelly_rec()`, `_nhl_kelly()`, `_mlb_kelly()` — identical formula with cosmetic differences
2. **ELO computation** — `_elo_win_prob()`, `_elo_expected()`, `expected_win()` — same formula copy-pasted
3. **Moneyline to probability** — `_moneyline_to_prob()` — identical in all 3 feature engineering files
4. **ELO trend computation** — `_add_elo_trend()` — same rolling-window logic with different column prefixes
5. **Model training pipeline** — GBC + RF stacking ensemble with LogReg meta-learner, identical hyperparameters
6. **Backtesting simulation** — Kelly simulation loop, flat bet comparison, Plotly chart rendering
7. **Prediction result rendering** — confidence badges, signal tiers, metric cards, progress bars
8. **Track Record tab** — prediction history display, bet tracker, signal filtering
9. **Parlay Ladder tab** — leg selection, tier optimization, stake sizing (all use `parlay_math.py`)
10. **Season regression** — regress ELOs toward mean between seasons

---

## 2. Modular Directory Structures

### Best practice: Domain-centric packages

The dominant pattern in multi-domain ML projects is **domain-centric packaging** — group by sport first, then by layer within each sport. This is preferred over layer-centric packaging (grouping all models together, all UIs together) because:

- Each sport is a self-contained vertical slice that can be developed independently
- Adding a new sport means creating one new package, not touching N existing packages
- Developers work within one sport context at a time — co-location reduces context switching

### Recommended structure for EdgeIQ

```
edgeiq/
    __init__.py
    app.py                          # Streamlit entry point + router

    # ── Shared core (cross-sport) ──────────────────────────────
    core/
        __init__.py
        elo.py                      # ELO rating system (compute, update, regress, trend)
        kelly.py                    # Kelly criterion sizing + signal badges
        odds.py                     # American/decimal conversion, implied prob
        backtest.py                 # Backtesting engine (Kelly sim, flat bet, P&L series)
        parlay.py                   # Parlay math (tiers, stakes, correlation checks)
        history.py                  # Prediction logging + bet tracker persistence
        constants.py                # Cross-sport constants (colors, badge thresholds, bankroll limits)

    # ── UI components (shared Streamlit widgets) ───────────────
    ui/
        __init__.py
        prediction_card.py          # Render prediction result (badges, metrics, progress bars)
        backtest_charts.py          # Plotly backtesting charts (Kelly vs flat, cumulative P&L)
        parlay_ladder.py            # Parlay ladder tab UI (shared across sports)
        track_record.py             # Track record tab UI (history, bet log, export)
        sidebar.py                  # Shared sidebar components (bankroll, risk tolerance)
        style.css                   # Global styles

    # ── Per-sport domains ──────────────────────────────────────
    sports/
        __init__.py

        nfl/
            __init__.py
            app.py                  # render_nfl_app() — tabs + routing
            config.py               # NFL-specific constants (teams, features list, model paths)
            features.py             # NFL feature engineering (26 features)
            schedule.py             # NFL weekly schedule (ESPN)
            pipeline.py             # NFL data pipeline (5 API orchestration)
            matchup.py              # Defensive matchup adjustments
            tabs/
                __init__.py
                predictor.py        # Game Predictor tab
                backtesting.py      # Backtesting tab
                props.py            # Player Props tab
                parlay.py           # Parlay Ladder tab
                record.py           # Track Record tab

        nhl/
            __init__.py
            app.py                  # render_nhl_app()
            config.py               # NHL-specific constants
            features.py             # NHL feature engineering (29 features)
            schedule.py             # NHL weekly schedule
            tabs/
                __init__.py
                predictor.py
                backtesting.py
                props.py
                parlay.py
                record.py

        mlb/
            __init__.py
            app.py                  # render_mlb_app()
            config.py               # MLB-specific constants
            features.py             # MLB feature engineering (29 features)
            schedule.py             # MLB weekly schedule
            tabs/
                __init__.py
                predictor.py
                backtesting.py
                props.py
                parlay.py
                record.py

    # ── Data API clients (unchanged) ──────────────────────────
    apis/
        __init__.py
        cache.py
        espn.py
        mlb.py
        nhl.py
        odds.py
        pfr.py
        tank01.py
        weather.py

    # ── Training scripts (offline, not imported by app) ────────
    training/
        __init__.py
        base_trainer.py             # Shared stacking ensemble trainer
        nfl/
            train_model.py
            train_total.py
            train_props.py
            build_games.py
            build_team_stats.py
            build_ratings.py
        nhl/
            train_model.py
            train_total.py
            train_props.py
            build_games.py
            build_team_stats.py
            build_ratings.py
        mlb/
            train_model.py
            train_total.py
            train_props.py
            build_games.py
            build_team_stats.py
            build_ratings.py

    # ── Data artifacts (generated, gitignored except .csv/.pkl) ──
    data/
        models/                     # .pkl files
        csv/                        # .csv data files
        cache/                      # API response cache
```

### Key principles

1. **`core/` is pure Python** — no Streamlit, no pandas dependency where avoidable, easily testable
2. **`ui/` is pure Streamlit** — reusable widget functions that accept data, not sport-specific logic
3. **`sports/{sport}/` is the glue** — combines core logic with sport-specific config and domain knowledge
4. **`training/` is offline** — never imported by the running app; produces artifacts consumed by `sports/`
5. **`apis/` stays as-is** — already well-structured and shared

---

## 3. Abstract Base Classes / Shared Interfaces

### When to use ABCs vs not

For this project, **ABCs are NOT recommended** as the primary abstraction. Here is why:

- Python ABCs add ceremony (metaclass, `@abstractmethod`) for little benefit when there are only 3 domains
- The sports share *structure* (same tabs, same pipeline stages) but not *behavior* (NFL has weather, MLB has day/night, NHL has OT loss)
- ABCs enforce a contract at class definition time, but Streamlit apps are procedural — they are sequences of `st.` calls, not method overrides

### What IS recommended: Protocol classes + composition

Use Python `Protocol` (structural subtyping) for the few places where you genuinely need pluggable behavior:

```python
# core/protocols.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class SportConfig(Protocol):
    """Every sport module must expose these attributes."""
    SPORT_KEY: str                    # 'nfl', 'nhl', 'mlb'
    TEAMS: list[str]                  # team abbreviation list
    TEAM_NAMES: dict[str, str]        # abbrev -> full name
    FEATURES: list[str]              # model feature column names
    MODEL_PATH: str                   # path to .pkl
    ELO_PATH: str                     # path to ELO .pkl
    GAMES_CSV: str                    # path to games_processed.csv
    HOME_ADV_ELO: int                # ELO home advantage points

@runtime_checkable
class FeatureBuilder(Protocol):
    """Feature engineering contract."""
    def build_features(self, df) -> 'pd.DataFrame': ...
    def build_live_features(self, home, away, context) -> 'pd.DataFrame': ...
```

But for most of the app, **plain functions with sport-specific config dicts** are simpler and more Pythonic.

### Recommended pattern: Config dicts, not class hierarchies

```python
# core/kelly.py
def kelly_criterion(model_prob, moneyline_odds, fraction=0.5, cap=0.10):
    """Universal Kelly criterion. Used by all sports."""
    try:
        b = (100.0 / abs(moneyline_odds)) if moneyline_odds < 0 else (moneyline_odds / 100.0)
        full = (b * model_prob - (1.0 - model_prob)) / b
        pct = max(0.0, min(full * fraction, cap)) * 100
    except Exception:
        return 0.0, 'signal-pass', 'PASS'
    if pct >= 4.0: return pct, 'signal-strong', 'STRONG EDGE'
    if pct >= 2.0: return pct, 'signal-lean',   'LEAN'
    if pct >= 1.0: return pct, 'signal-lean',   'SMALL EDGE'
    return pct, 'signal-pass', 'PASS'
```

This single function replaces `_kelly_rec()`, `_nhl_kelly()`, and `_mlb_kelly()`.

---

## 4. DRY Patterns for Pipelines

### 4.1 Feature Engineering

**Current problem**: Three `_elo_win_prob()` functions, three `_moneyline_to_prob()` functions, three `_add_elo_trend()` functions — all doing the same math with different column name prefixes.

**Solution: Parameterized shared functions + sport-specific wrappers**

```python
# core/elo.py
def elo_win_prob(elo_diff, home_adv=0):
    """Home win probability from ELO difference."""
    return 1.0 / (1.0 + 10.0 ** (-(elo_diff + home_adv) / 400.0))

def update_elo(elo_dict, home, away, home_win, k=20, home_adv=0):
    """Update ELO ratings. Returns (new_home, new_away, pre_diff)."""
    h = elo_dict.get(home, 1500)
    a = elo_dict.get(away, 1500)
    diff = h - a
    expected = elo_win_prob(diff, home_adv)
    elo_dict[home] = h + k * (home_win - expected)
    elo_dict[away] = a + k * ((1 - home_win) - (1 - expected))
    return elo_dict[home], elo_dict[away], diff

def regress_elos(elo_dict, rate=0.75, mean=1500):
    """Season regression toward the mean."""
    for team in list(elo_dict):
        elo_dict[team] = elo_dict[team] * rate + mean * (1 - rate)

def add_elo_trend(df, elo_col, prefix, window=4):
    """Add rolling ELO trend feature. Sport-agnostic."""
    # ... parameterized by column names
```

Each sport's `features.py` then calls these shared functions with sport-specific column names:

```python
# sports/nhl/features.py
from core.elo import elo_win_prob, add_elo_trend

def build_nhl_enhanced_features(df):
    df['nhl_elo_implied_prob'] = df['nhl_elo_diff'].apply(
        lambda d: elo_win_prob(d, home_adv=28)
    )
    df = add_elo_trend(df, elo_col='nhl_elo_diff', prefix='nhl', window=4)
    # ... NHL-specific features
```

### 4.2 Model Training

**Current problem**: `build_nhl_model.py` and `build_mlb_model.py` are 90%+ identical — same GBC hyperparameters, same RF hyperparameters, same stacking architecture, same evaluation code.

**Solution: A shared training harness**

```python
# training/base_trainer.py
def train_stacking_ensemble(
    X_train, y_train, X_test, y_test,
    features,
    gbc_params=None,
    rf_params=None,
    random_state=42,
    n_cv_splits=5,
):
    """Train GBC+RF stacking ensemble with LogReg meta-learner.

    Returns dict: {'model': stack, 'features': features, 'accuracy': float, 'brier': float}
    """
    gbc_params = gbc_params or {
        'n_estimators': 500, 'learning_rate': 0.02, 'max_depth': 3,
        'min_samples_leaf': 30, 'subsample': 0.80, 'max_features': 0.75,
    }
    rf_params = rf_params or {
        'n_estimators': 300, 'max_depth': 6, 'min_samples_leaf': 25,
        'max_features': 'sqrt',
    }
    # ... identical training logic, parameterized
```

Each sport's training script becomes a thin wrapper:

```python
# training/nhl/train_model.py
from training.base_trainer import train_stacking_ensemble
from sports.nhl.config import NHL_FEATURES, DATA_PATH

def main():
    df = pd.read_csv(DATA_PATH)
    df = build_nhl_enhanced_features(df)
    # ... sport-specific data prep
    result = train_stacking_ensemble(X_train, y_train, X_test, y_test, NHL_FEATURES)
    pickle.dump(result, open('model_nhl_enhanced.pkl', 'wb'))
```

### 4.3 Backtesting

**Current problem**: Each sport has a 100-200 line Kelly simulation + Plotly chart block that is nearly identical.

**Solution: A shared backtesting engine**

```python
# core/backtest.py
def kelly_simulation(games_df, prob_col, correct_col, bankroll_start=1000,
                     kelly_frac=0.5, kelly_cap=0.10, odds=-110):
    """Run Kelly criterion backtest over historical games.

    Returns dict with kelly_history, flat_history, final_bankroll, etc.
    """
    # ... identical simulation logic

# ui/backtest_charts.py
def render_kelly_chart(sim_result, sport_prefix=''):
    """Render Plotly Kelly vs Flat bankroll chart."""
    # ... identical chart logic
```

---

## 5. Configuration-Driven vs Inheritance-Driven

### Recommendation: Configuration-driven (strongly)

For EdgeIQ, **configuration-driven architecture** is the clear winner over inheritance-driven:

| Factor | Config-Driven | Inheritance-Driven |
|--------|--------------|-------------------|
| Complexity | Low — dicts and functions | High — class hierarchies, method resolution order |
| Adding a sport | Add a config dict + thin wrapper functions | Add a subclass, override N methods, risk breaking base |
| Testing | Test config values + shared functions independently | Must test each subclass separately, mocking base class |
| Streamlit compatibility | Natural — Streamlit is procedural | Awkward — Streamlit has no OOP rendering model |
| Python idiom | "We're all adults here" — simple is better | Enterprise Java pattern, not Pythonic for small teams |
| Readability | Easy to see what differs between sports | Must trace through class hierarchy to understand behavior |

### How config-driven works in practice

Each sport has a `config.py` that defines its domain-specific values:

```python
# sports/nhl/config.py
SPORT_KEY       = 'nhl'
SPORT_LABEL     = 'NHL Hockey'
SPORT_EMOJI     = '\U0001f3d2'
HOME_ADV_ELO    = 28
ELO_K           = 6
ELO_K_OTL       = 3
REGRESSION_RATE = 0.75
FEATURES        = [...]       # 29 NHL features
MODEL_PATH      = 'model_nhl_enhanced.pkl'
ELO_PATH        = 'nhl_elo_ratings.pkl'
GAMES_CSV       = 'nhl_games_processed.csv'
TEAMS           = [...]
TEAM_NAMES      = {...}
BANKROLL_KEY    = 'nhl_bankroll'
DEFAULT_ODDS    = -110
KELLY_CAP       = 0.10         # 10% max per bet

# Backtesting config
BACKTEST_SEASONS = range(2018, 2026)
HOLDOUT_SEASONS  = 2
```

Shared code reads from this config:

```python
# core/backtest.py
def run_backtest(config, games_df):
    """config is the sport's config module or a dict."""
    for season in config.BACKTEST_SEASONS:
        # ... parameterized by config values
```

### When inheritance IS useful

The one place where a base class makes sense is the **training harness** — where the pipeline stages (load data -> engineer features -> split -> train -> evaluate -> save) are identical in *sequence* but differ in *content*:

```python
# training/base_trainer.py
class BaseTrainer:
    """Optional base class for training scripts. Not required."""
    def load_data(self): raise NotImplementedError
    def engineer_features(self, df): raise NotImplementedError
    def get_features_list(self): raise NotImplementedError
    def get_model_path(self): raise NotImplementedError

    def run(self):
        """Template method — shared pipeline."""
        df = self.load_data()
        df = self.engineer_features(df)
        features = self.get_features_list()
        # ... shared train/eval/save logic
```

But even this could be a plain function with callbacks. The Template Method pattern is the one OOP pattern that genuinely fits here.

---

## 6. Clean Layer Separation

### The 4-layer architecture

```
Layer 1: Data Fetching (apis/)
    - HTTP clients, JSON parsing, caching
    - No business logic, no ML, no UI
    - Returns raw dicts/DataFrames

Layer 2: Feature Engineering (sports/{sport}/features.py + core/elo.py)
    - Transforms raw data into model-ready features
    - Uses shared math (ELO, odds conversion) from core/
    - Returns feature DataFrames with named columns
    - No UI, no model loading

Layer 3: Model Prediction (sports/{sport}/predict.py + core/kelly.py)
    - Loads trained model from .pkl
    - Accepts feature DataFrame, returns prediction dict
    - Computes Kelly sizing, confidence badges
    - No UI

Layer 4: UI Rendering (sports/{sport}/tabs/*.py + ui/*.py)
    - Streamlit widgets, layout, charts
    - Calls Layer 3 for predictions, Layer 2 for features
    - Never computes math directly — delegates to core/
```

### Why this matters

The current codebase violates layer separation in several ways:

1. **Kelly criterion is computed inside the UI rendering function** — the `_kelly_rec()` function is defined inside `final_app.py` alongside `st.metric()` calls. If you want to use Kelly in a backtesting script or API endpoint, you would have to import from a Streamlit app file.

2. **ELO is computed both in feature engineering AND in the app files** — `get_elo()` and `elo_win_prob()` are defined in `final_app.py`, `nhl_app.py`, and `mlb_app.py` as local helpers, duplicating what exists in the feature engineering files.

3. **Backtesting simulation is embedded in the UI tab** — the Kelly simulation loop is interleaved with `st.metric()` and `st.plotly_chart()` calls, making it impossible to run backtests from the command line.

### Clean separation principle

**Rule: Every function should live at the lowest layer that needs it.**

- Math (ELO, Kelly, odds) -> `core/`
- Feature computation -> `sports/{sport}/features.py`
- Prediction assembly -> `sports/{sport}/predict.py`
- Rendering -> `sports/{sport}/tabs/` or `ui/`

**Test: Can you run a backtest from a Python script without importing Streamlit?** If yes, the layers are clean. If no, you have coupling.

---

## 7. File Organization: Flat vs Nested

### When flat works

Flat structure works when:
- The project has fewer than ~20 Python files
- There is only one developer
- There is only one "domain"
- You never import between files in the same directory

EdgeIQ has outgrown this — at 80+ files, flat structure causes:
- Cognitive overhead scanning the file list
- Name collision risk (multiple `build_*` files blur together)
- No namespace isolation between sports
- Difficulty finding "which file handles X for sport Y"

### When to switch to packages

Switch to packages when:
- You have 3+ domains with parallel structure (EdgeIQ has 3 sports)
- Files follow a naming convention with prefixes (`nhl_`, `mlb_`, `build_nhl_`, etc.) — this is a signal that the prefix should be a directory
- You have shared utilities used across domains
- A new developer would need a map to navigate the codebase

**EdgeIQ has all four signals.** The `nhl_` and `mlb_` prefixes on 20+ files are literally encoding a directory structure in filenames.

### Recommended approach: 2-level nesting max

Avoid deep nesting (3+ levels). Two levels is the sweet spot:

```
sports/nhl/features.py      # 2 levels — clear, navigable
sports/nhl/tabs/props.py    # 3 levels — acceptable for tab isolation
sports/nhl/tabs/ui/cards.py # 4 levels — too deep, avoid
```

### Package vs module decision

| Use a **module** (single .py file) when: | Use a **package** (directory with __init__.py) when: |
|------------------------------------------|------------------------------------------------------|
| The code is < 500 lines | The code exceeds 500 lines |
| It has a single responsibility | It has sub-responsibilities that benefit from splitting |
| It is not expected to grow | It will grow as features are added |
| Example: `core/kelly.py` (50 lines) | Example: `sports/nfl/` (multiple tabs, pipeline, config) |

---

## 8. Shared Utility Patterns

### 8.1 Kelly Criterion

**Current state**: 3 copies with cosmetic differences.

**Recommended**: Single function in `core/kelly.py`.

```python
# core/kelly.py

# Signal badge thresholds (from EdgeIQ.md)
SIGNAL_THRESHOLDS = [
    (4.0, 'signal-strong', 'STRONG EDGE'),
    (2.0, 'signal-lean',   'LEAN'),
    (1.0, 'signal-lean',   'SMALL EDGE'),
]
DEFAULT_SIGNAL = (0.0, 'signal-pass', 'PASS')

def kelly_criterion(model_prob, moneyline_odds, fraction=0.5, cap=0.10):
    """Universal Kelly criterion bet sizing.

    Args:
        model_prob: Model's predicted win probability (0-1)
        moneyline_odds: American odds (e.g., -110, +150)
        fraction: Kelly fraction (0.25=quarter, 0.5=half, 1.0=full)
        cap: Maximum bet as fraction of bankroll (default 10%)

    Returns:
        (kelly_pct, css_class, signal_label)
    """
    try:
        b = (100.0 / abs(moneyline_odds)) if moneyline_odds < 0 else (moneyline_odds / 100.0)
        full = (b * model_prob - (1.0 - model_prob)) / b
        pct = max(0.0, min(full * fraction, cap)) * 100
    except Exception:
        return DEFAULT_SIGNAL

    for threshold, css, label in SIGNAL_THRESHOLDS:
        if pct >= threshold:
            return pct, css, label
    return pct, DEFAULT_SIGNAL[1], DEFAULT_SIGNAL[2]

# Risk tolerance mapping (used in all sport sidebars)
RISK_FRACTIONS = {
    'Conservative': 0.25,
    'Moderate': 0.50,
    'Aggressive': 1.00,
}
```

### 8.2 ELO System

**Current state**: ELO logic duplicated in `build_model.py`, `build_nhl_games.py`, `build_mlb_games.py`, plus feature engineering files.

**Recommended**: `core/elo.py` with parameterized constants.

```python
# core/elo.py

STARTING_ELO = 1500

def expected_win_prob(elo_home, elo_away, home_adv=0):
    return 1.0 / (1.0 + 10.0 ** ((elo_away - (elo_home + home_adv)) / 400.0))

def update_elo(elo_dict, home, away, home_won, k=20, home_adv=0,
               partial_loss_k=None, is_partial_loss=False):
    """Update ELO ratings after a game.

    Args:
        partial_loss_k: K-factor for partial losses (NHL OTL, etc.)
        is_partial_loss: Whether the losing team gets partial credit
    """
    h = elo_dict.get(home, STARTING_ELO)
    a = elo_dict.get(away, STARTING_ELO)
    diff = h - a
    exp = expected_win_prob(h, a, home_adv)

    effective_k = (partial_loss_k or k) if is_partial_loss else k

    elo_dict[home] = h + effective_k * (home_won - exp)
    elo_dict[away] = a + effective_k * ((1 - home_won) - (1 - exp))
    return elo_dict[home], elo_dict[away], diff

def regress_elos(elo_dict, rate=0.75, mean=STARTING_ELO):
    """Between-season regression toward the mean."""
    for team in list(elo_dict):
        elo_dict[team] = elo_dict[team] * rate + mean * (1 - rate)

def compute_elo_trend(elo_history, current_elo, window=4):
    """Compute ELO momentum (change over last N games)."""
    if len(elo_history) < 2:
        return 0.0
    recent = elo_history[-window:] if len(elo_history) >= window else elo_history
    return current_elo - recent[0]
```

Each sport configures its ELO params:

```python
# sports/nhl/config.py
ELO_CONFIG = {
    'k': 6,
    'k_otl': 3,          # OT loss partial credit
    'home_adv': 28,
    'regression_rate': 0.75,
    'regression_mean': 1500,
    'trend_window': 4,
}
```

### 8.3 Backtesting

**Current state**: Each sport app has 100+ lines of inline backtesting code mixed with Streamlit calls.

**Recommended**: Separate computation from rendering.

```python
# core/backtest.py

def backtest_moneyline(games_df, prob_col='prob', result_col='correct',
                       odds_col=None, default_odds=-110):
    """Compute backtesting metrics from historical predictions.

    Returns dict with: accuracy, record, by_season, roi, etc.
    """
    # ... pure computation, no Streamlit

def kelly_bankroll_simulation(games_df, prob_col='prob', result_col='correct',
                              bankroll_start=1000, kelly_frac=0.5, kelly_cap=0.10,
                              odds=-110, flat_pct=0.005):
    """Simulate Kelly and flat bet bankroll trajectories.

    Returns dict with: kelly_history, flat_history, kelly_final, flat_final
    """
    # ... pure computation, returns data only

# ui/backtest_charts.py

def render_backtest_metrics(results, sport_prefix=''):
    """Display accuracy, ROI, record as st.metric()."""
    # ... Streamlit rendering only

def render_kelly_chart(sim, sport_prefix=''):
    """Render Kelly vs Flat bankroll Plotly chart."""
    # ... Streamlit rendering only
```

### 8.4 Odds Conversion

**Current state**: `_moneyline_to_prob()` appears in every feature engineering file. `parlay_math.py` has its own `american_to_decimal()` and `implied_probability()`.

**Recommended**: `core/odds.py` as the single source.

```python
# core/odds.py

def american_to_decimal(american):
    if american >= 100:
        return 1.0 + american / 100.0
    elif american <= -100:
        return 1.0 + 100.0 / abs(american)
    return 2.0

def decimal_to_american(decimal_odds):
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1.0) * 100)
    elif decimal_odds > 1.0:
        return round(-100.0 / (decimal_odds - 1.0))
    return -110

def moneyline_to_implied_prob(ml):
    if ml is None:
        return 0.5
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return abs(ml) / (abs(ml) + 100.0)
```

### 8.5 Prediction Result Rendering

**Current state**: Each sport has a `render_*_prediction_result()` function that is 80% identical (confidence badges, metric cards, progress bars, Kelly display) with 20% sport-specific content (spread vs puck line vs run line, weather, pitchers, goalies).

**Recommended**: A shared base renderer with sport-specific "slots."

```python
# ui/prediction_card.py

def render_prediction_header(home, away, prob_h, prob_a, prefix=''):
    """Shared: two-column win probability display with progress bars."""
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"Home {home}", f"{prob_h*100:.1f}%")
        st.progress(float(prob_h))
    with col2:
        st.metric(f"Away {away}", f"{prob_a*100:.1f}%")
        st.progress(float(prob_a))

def render_confidence_badge(conf, winner):
    """Shared: LOCK / HIGH / MOD / TOSS-UP badge."""
    if conf > 0.75:   label, css = "LOCK", "signal-lock"
    elif conf > 0.65: label, css = "HIGH CONFIDENCE", "signal-strong"
    elif conf > 0.58: label, css = "MODERATE", "signal-lean"
    else:             label, css = "TOSS-UP", "signal-pass"
    st.markdown(f'<div class="signal-badge {css}">{label}: {winner}</div>',
                unsafe_allow_html=True)

def render_kelly_sizing(kelly_pct, tier, badge, bankroll, risk_label, fraction):
    """Shared: 4-column Kelly metric display."""
    # ... identical across all sports

def render_edge_analysis(model_prob, implied_prob, spread_value=None, spread_label='Spread'):
    """Shared: edge vs Vegas implied display."""
    # ... identical across all sports
```

Each sport composes these:

```python
# sports/nhl/tabs/predictor.py
from ui.prediction_card import (
    render_prediction_header, render_confidence_badge,
    render_kelly_sizing, render_edge_analysis
)

def render_nhl_prediction(result, bankroll, risk_tol):
    render_prediction_header(result['home'], result['away'], result['prob_h'], result['prob_a'])
    render_confidence_badge(max(result['prob_h'], result['prob_a']), result['pick'])

    # NHL-specific: goalie quality display
    if result.get('goalie_diff'):
        st.caption(f"Goalie edge: {result['goalie_diff']:+.2f}")

    render_kelly_sizing(...)
    render_edge_analysis(...)
```

---

## 9. Recommended Architecture for EdgeIQ

### Design principles (prioritized)

1. **Minimize risk** — refactor incrementally, not all at once
2. **Shared math in `core/`** — Kelly, ELO, odds, backtesting computation
3. **Sport-specific domain in `sports/{sport}/`** — features, config, tabs
4. **Shared UI in `ui/`** — reusable Streamlit components
5. **Training scripts separate from app** — `training/` is offline
6. **Configuration over inheritance** — dicts and functions, not class hierarchies
7. **2-level nesting max** — `sports/nhl/features.py`, not `sports/nhl/features/elo/trend.py`

### What NOT to do

1. **Do not create abstract base classes for each sport** — they add complexity without benefit for 3 domains
2. **Do not use a plugin/registry pattern** — overkill for a known, fixed set of sports
3. **Do not use dependency injection frameworks** — Python's import system is sufficient
4. **Do not refactor everything at once** — extract shared code incrementally
5. **Do not move data files into a package** — Streamlit Cloud expects them at predictable paths

### Lines of code impact estimate

| Component | Current Lines | After Refactor | Savings |
|-----------|--------------|----------------|---------|
| Kelly (3 copies) | ~90 | ~35 (1 copy) | ~55 |
| ELO (5+ copies) | ~200 | ~60 (1 copy) | ~140 |
| Odds conversion (4 copies) | ~80 | ~25 (1 copy) | ~55 |
| Backtest simulation (3 copies) | ~300 | ~80 (1 copy + render) | ~220 |
| Prediction rendering (3 copies) | ~300 | ~120 (shared + sport slots) | ~180 |
| Track Record tab (3 copies) | ~300 | ~100 (shared + sport filter) | ~200 |
| Model training (3 copies) | ~300 | ~120 (1 harness + 3 configs) | ~180 |
| **Total** | **~1,570** | **~540** | **~1,030** |

Approximately **1,000 lines** of duplication can be eliminated.

---

## 10. Migration Strategy

### Phase 1: Extract `core/` (lowest risk, highest impact)

**Do first.** These are pure functions with no UI dependency. Each extraction is a safe refactor.

1. Create `core/__init__.py`
2. Extract `core/kelly.py` from the three `_*_kelly()` functions
3. Extract `core/elo.py` from the five `_elo_*()` functions
4. Extract `core/odds.py` from `parlay_math.py` + feature engineering files
5. Update all imports — find-and-replace `_kelly_rec` -> `from core.kelly import kelly_criterion`
6. Run the app, verify all 3 sports work identically
7. Commit

**Risk: Low.** Pure function extraction. No behavior change. Easy to verify.

### Phase 2: Extract `core/backtest.py` (medium risk)

1. Move the Kelly simulation loop from each sport's backtesting tab into `core/backtest.py`
2. Return data (dicts/lists), not Streamlit widgets
3. Create `ui/backtest_charts.py` for the Plotly rendering
4. Update each sport's backtesting tab to call shared functions
5. Verify backtesting numbers match exactly

**Risk: Medium.** Separating computation from rendering requires careful testing.

### Phase 3: Create `sports/` packages (medium risk, high effort)

1. Create `sports/nfl/`, `sports/nhl/`, `sports/mlb/` packages
2. Move `nhl_feature_engineering.py` -> `sports/nhl/features.py`
3. Move `nhl_game_week.py` -> `sports/nhl/schedule.py`
4. Move `nhl_app.py` -> `sports/nhl/app.py`
5. Update `app.py` router imports
6. Repeat for NFL and MLB

**Risk: Medium.** Import paths change. Must update all cross-references. Streamlit Cloud deployment path may need adjustment.

### Phase 4: Split monolithic app files into tabs (low risk per tab)

1. For each sport, split the 2000+ line app file into `tabs/predictor.py`, `tabs/backtesting.py`, `tabs/props.py`, `tabs/parlay.py`, `tabs/record.py`
2. The sport's `app.py` becomes a thin router that imports and calls each tab
3. Each tab is 200-400 lines instead of one 2000+ line file

**Risk: Low per tab, medium in aggregate.** Session state keys must be carefully preserved.

### Phase 5: Extract shared UI components (low risk)

1. Extract `ui/prediction_card.py` from the three `render_*_prediction_result()` functions
2. Extract `ui/track_record.py` from the three Track Record tab implementations
3. Each sport's tab imports shared UI and adds sport-specific additions

**Risk: Low.** UI changes are visually verifiable.

### Phase 6: Restructure training scripts (low risk, isolated)

1. Create `training/base_trainer.py` with shared stacking ensemble logic
2. Move each `build_*_model.py` into `training/{sport}/`
3. Update any CLI references

**Risk: Low.** Training scripts are offline and don't affect the running app.

### What to defer

- Moving data files (`.csv`, `.pkl`) into a `data/` directory — requires updating every hardcoded path and Streamlit Cloud deployment config. Do this last, if at all.
- Moving `apis/` — already well-structured, no benefit to moving.
- Adding automated tests — valuable but orthogonal to architecture refactoring.

---

## Appendix A: Patterns from Real-World Multi-Domain ML Projects

### Pattern 1: Scikit-learn's estimator interface

Scikit-learn uses a consistent interface (`fit`, `predict`, `score`) across all algorithms. This works because all algorithms share the same contract: accept X/y, produce predictions. EdgeIQ's sports share a similar contract: accept features, produce win probabilities.

**Takeaway**: A shared `predict(features) -> dict` interface is natural. But don't force it into a class hierarchy — a function that takes a model + features is simpler.

### Pattern 2: Django's app structure

Django organizes by "apps" (domains), each with `models.py`, `views.py`, `urls.py`, `admin.py`. This maps well to EdgeIQ's structure where each sport has features, predictions, tabs, config.

**Takeaway**: The `sports/{sport}/` package with known filenames (`features.py`, `config.py`, `app.py`) is the Django-style pattern applied to ML.

### Pattern 3: Airflow's operator pattern

Airflow uses a base `BaseOperator` with task-specific subclasses. But operators are thin wrappers around hooks (data connectors) and shared utilities.

**Takeaway**: Keep the "sport operator" (app.py) thin. Push logic into shared utilities (core/) and reusable hooks (apis/).

### Pattern 4: Kedro's pipeline structure

Kedro (a Python ML framework) separates pipelines into nodes (pure functions), pipelines (DAGs), and IO (data loading). This enforces the rule that business logic is always in pure functions.

**Takeaway**: The `core/` layer should contain **pure functions** — no side effects, no Streamlit, no file I/O. This makes them testable and reusable.

---

## Appendix B: Decision Record

### Why not microservices / separate repos?

- EdgeIQ is a single Streamlit app deployed to Streamlit Community Cloud
- All sports share the same deployment, the same CSS, the same session state
- Separate repos would add deployment complexity for no benefit at this scale
- If EdgeIQ ever needs a REST API backend, that is the time to split

### Why not a database (SQLite/Postgres)?

- CSV files are sufficient for the current data volume (~60K games max per sport)
- Pickle files are the standard for scikit-learn model persistence
- A database would add deployment complexity on Streamlit Cloud
- If EdgeIQ ever needs multi-user state or real-time data, that is the time to add a DB

### Why not Pydantic models for config?

- Pydantic is great for API validation but overkill for internal config
- The config values are static constants, not user input
- Plain dicts and module-level constants are simpler and have zero dependencies
- If EdgeIQ ever exposes a REST API, Pydantic models would be appropriate then

### Why configuration over inheritance?

- The sports differ in *data and constants*, not in *behavior*
- When behavior differs (NFL has weather, NHL has OTL, MLB has day/night), it is handled by conditional logic within shared functions, not by method overrides
- Three sport files that each call `kelly_criterion(prob, odds)` is simpler than three subclasses that override `compute_kelly()`
- The codebase has zero classes today — introducing a class hierarchy would be a paradigm shift for no gain
