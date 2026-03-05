# DRY Cross-Domain Prediction Systems: Patterns, Pitfalls, and Recommendations

> Research for EdgeIQ refactoring — eliminating copy-paste duplication across NFL/NHL/MLB
> while avoiding premature abstraction.

---

## 1. Current State: Measuring the Duplication

### 1.1 Lines of Code by Sport App

| File | Lines | Purpose |
|------|-------|---------|
| `final_app.py` | 3,627 | NFL — 7 tabs |
| `nhl_app.py` | 2,731 | NHL — 5 tabs |
| `mlb_app.py` | 1,767 | MLB — 5 tabs |
| **Total** | **8,125** | |

### 1.2 Duplicated Code Blocks (Nearly Identical Across 3 Sports)

| Block | NFL Location | NHL Location | MLB Location | ~Lines Each | Variation |
|-------|-------------|-------------|-------------|------------|-----------|
| **Kelly criterion calc** | `_kelly_rec()` L499-530 | `_nhl_kelly()` L669-679 | `_mlb_kelly()` L431-441 | 15-30 | Identical formula, different var names |
| **Kelly UI rendering** | `render_prediction_result()` L939-994 | `render_nhl_prediction_result()` L653-721 | `render_mlb_prediction_result()` L498-557 | 55-60 | Identical layout: 4 cols, metrics, caption. Only session key prefixes differ |
| **Confidence tier badges** | L919-929 | L633-643 | L465-475 | 10 | Identical thresholds: >75% LOCK, >65% HIGH, >58% MOD, else TOSS-UP |
| **Win prob display** | L907-915 | L620-630 | L454-463 | 10 | Identical: two columns, st.metric + st.progress |
| **Bet strategy routing** | L961-973 | L688-699 | L522-533 | 12 | Identical if/elif/else: Fixed % / Fixed $ / Kelly |
| **Sidebar betting controls** | L1872-1881 | L2626-2635 | L1659-1668 | 10 | Identical widgets, different session key prefixes |
| **Prediction logging** | L996-1012 | L723-728 | L559-562 | 15-20 | Identical pattern, different sport string |
| **Backtesting: accuracy metrics** | L3066-3071 | L1431-1441 | L989-994 | 6-10 | Same st.metric pattern, different label text |
| **Backtesting: season accuracy table** | L3073-3080 | L1443-1460 | L996-1001 | 8-15 | Same groupby/rename/display |
| **Backtesting: game-by-game** | L3083-3094 | L1462-1479 | (not present) | 12-15 | Same select + display pattern |
| **Backtesting: Kelly bankroll sim** | (inline) | (inline) | L930-953 | 20 | Same loop: bet = bankroll * kelly_pct, win/loss update |
| **Track Record: prediction history** | L3351+ | L1775-1834 | L1048-1064 | 40-60 | Same filter/sort/display; NHL most complete, MLB simplified |
| **Track Record: bet tracker form** | (in final_app) | (in nhl_app) | L1066-1095 | 25-30 | Nearly identical form fields |
| **Track Record: export/import** | (in final_app) | (in nhl_app) | L1097-1105 | 20-30 | Same download_button pattern |
| **Parlay Ladder tab** | L2651+ | L1703-1772 | L1537-1540 | 60-100 | Uses shared `parlay_math.py` but UI rendering duplicated |
| **Prop card selection UI** | L2073-2143 | L2290-2293 | L1259-1264 | 30-50 | Same select-legs-from-game-cards pattern |
| **Vegas implied prob calc** | L958-959 | L682-685 | L516-519 | 3-4 | Identical formula, repeated inline 6+ times across files |

**Estimated duplicated code: ~400-500 lines repeated 3 times = ~800-1000 lines that could be collapsed.**

### 1.3 What Is Already Shared (Good)

- `parlay_math.py` — odds conversion, tier optimization, stake sizing, correlation filter (289 lines, fully sport-agnostic)
- `prediction_history.py` — log/load/backfill predictions (sport-agnostic storage)
- `apis/cache.py` — shared JSON file cache with TTL
- Confidence tier thresholds are consistent across all three sports (>75/65/58)

### 1.4 What Varies By Sport (Genuinely Different)

| Concern | NFL | NHL | MLB |
|---------|-----|-----|-----|
| Feature engineering | 26 features | 29 features | 29 features |
| Feature modules | `feature_engineering.py` | `nhl_feature_engineering.py` | `mlb_feature_engineering.py` |
| Model pickle | `model_enhanced.pkl` | `model_nhl_enhanced.pkl` | `model_mlb_enhanced.pkl` |
| Schedule fetcher | `game_week.py` | `nhl_game_week.py` | `mlb_game_week.py` |
| O/U model | Ridge, points | Ridge, goals | Ridge, runs |
| O/U MAE | 10.01 pts | 1.1 goals | 3.44 runs |
| Prop stat types | pass/rush/rec yards | goals/assists/shots | pitcher K/ER, batter hits/TB |
| Prop confidence | Poisson / Normal | Poisson / Normal | GBR direct prediction |
| Data pipeline | 5 APIs (ESPN, Tank01, odds, weather, PFR) | NHL API + odds | MLB Stats API + odds |
| Spread-to-ML conversion | Yes (NFL has point spreads) | No (moneyline sport) | No (moneyline sport) |
| Defensive matchup adj | Yes (6-position) | No | No |
| Weather impact | Yes (outdoor stadiums) | No (indoor) | Partial (open-air parks) |
| Session key prefix | `nfl_`, `g{idx}_` | `nhl_`, `nhl_g{idx}_` | `mlb_`, `mlb_g{idx}_` |

---

## 2. Pattern Analysis

### 2.1 Strategy Pattern (Recommended for: Kelly + Bet Sizing)

**What it is:** Define a family of algorithms, encapsulate each one, and make them interchangeable. The algorithm varies independently from the clients that use it.

**How it applies:**
```python
# betting.py — shared module

def kelly_criterion(model_prob, moneyline_odds, fraction=0.5):
    """Pure Kelly calculation — no Streamlit, no session state."""
    if moneyline_odds is None or moneyline_odds == 0 or model_prob is None:
        return 0.0, 'signal-pass', 'PASS'
    try:
        b = (100.0 / abs(moneyline_odds)) if moneyline_odds < 0 else (moneyline_odds / 100.0)
        q = 1.0 - float(model_prob)
        full_kelly = (b * float(model_prob) - q) / b
        kelly = max(0.0, min(full_kelly * fraction, 0.10))
        pct = kelly * 100.0
    except Exception:
        return 0.0, 'signal-pass', 'PASS'
    if pct >= 4.0:
        return pct, 'signal-strong', 'STRONG EDGE'
    if pct >= 2.0:
        return pct, 'signal-lean', 'LEAN'
    if pct >= 1.0:
        return pct, 'signal-lean', 'SMALL EDGE'
    return pct, 'signal-pass', 'PASS'

def compute_bet_amount(strategy, bankroll, kelly_pct, fixed_pct, fixed_dollar):
    """Route to the right sizing strategy."""
    if strategy == 'Fixed %':
        return bankroll * fixed_pct / 100, "Fixed %", f"{fixed_pct:.1f}%"
    elif strategy == 'Fixed $':
        return float(fixed_dollar), "Fixed $", f"${fixed_dollar}"
    else:
        return bankroll * kelly_pct / 100, "Kelly %", f"{kelly_pct:.1f}%"

def vegas_implied_prob(moneyline):
    """American ML to implied probability."""
    if moneyline < 0:
        return abs(moneyline) / (abs(moneyline) + 100)
    return 100 / (moneyline + 100)

def confidence_tier(prob):
    """Map win probability to badge label + CSS class."""
    if prob > 0.75: return "LOCK", "signal-lock"
    if prob > 0.65: return "HIGH CONFIDENCE", "signal-strong"
    if prob > 0.58: return "MODERATE", "signal-lean"
    return "TOSS-UP", "signal-pass"
```

**Why this works:** The Kelly formula, bet sizing logic, and confidence tiers are 100% identical across all three sports. Zero sport-specific variation. This is the lowest-risk, highest-reward refactoring target.

**Estimated savings:** ~90 lines removed from each sport file = **~180 lines eliminated**.

### 2.2 Configuration Objects (Recommended for: Session State + Sidebar)

**What it is:** Instead of passing sport-specific prefixes everywhere, define a config dict/dataclass per sport that carries all the sport-specific keys, defaults, and labels.

**How it applies:**
```python
# sport_config.py

SPORT_CONFIGS = {
    'nfl': {
        'prefix': 'nfl',
        'bankroll_key': 'bankroll',          # historical: NFL uses 'bankroll' not 'nfl_bankroll'
        'strategy_key': 'nfl_bet_strategy',
        'risk_key': 'nfl_risk_tolerance',
        'fixed_pct_key': 'nfl_fixed_pct',
        'fixed_dollar_key': 'nfl_fixed_dollar',
        'emoji': '🏈',
        'ou_unit': 'pts',
        'ou_mae': 10.01,
        'baseline_label': 'Home Win Rate',
    },
    'nhl': {
        'prefix': 'nhl',
        'bankroll_key': 'nhl_bankroll',
        'strategy_key': 'nhl_bet_strategy',
        'risk_key': 'nhl_risk_tolerance',
        'fixed_pct_key': 'nhl_fixed_pct',
        'fixed_dollar_key': 'nhl_fixed_dollar',
        'emoji': '🏒',
        'ou_unit': 'goals',
        'ou_mae': 1.1,
        'baseline_label': 'Home Win Rate',
    },
    'mlb': {
        'prefix': 'mlb',
        'bankroll_key': 'mlb_bankroll',
        'strategy_key': 'mlb_bet_strategy',
        'risk_key': 'mlb_risk_tolerance',
        'fixed_pct_key': 'mlb_fixed_pct',
        'fixed_dollar_key': 'mlb_fixed_dollar',
        'emoji': '⚾',
        'ou_unit': 'runs',
        'ou_mae': 3.44,
        'baseline_label': 'Home Win Rate',
    },
}
```

**Why config objects over class hierarchies:** Streamlit is function-oriented. Session state is a flat dict of strings. Introducing class hierarchies (SportPredictor -> NFLPredictor, NHLPredictor, MLBPredictor) would fight against Streamlit's execution model where the entire script re-runs on every interaction. Config dicts are the right abstraction for "same code, different keys."

**Anti-pattern to avoid:** Don't create `class NFLConfig(BaseSportConfig)` — this adds indirection without benefit when the differences are just string values. A plain dict (or dataclass for type safety) is sufficient.

### 2.3 Template Method (Recommended for: Backtesting Tab)

**What it is:** Define the skeleton of an algorithm in a base function, deferring some steps to sport-specific callbacks.

**How it applies:**
```python
# shared_ui.py

def render_backtest_tab(
    sport_config,
    load_historical_fn,      # () -> (df_engineered, feature_list)
    model_pkg,               # {'model': ..., 'features': [...], 'accuracy': float}
    season_column='season',
    game_date_column='gameday',
):
    """Shared backtesting skeleton — works for any sport."""
    df_eng, feat_list = load_historical_fn()
    if df_eng.empty:
        st.error("No historical data found.")
        return

    # Season selector (identical across sports)
    all_seasons = sorted(df_eng[season_column].unique(), reverse=True)
    ...
    # Predict (identical)
    X = df[features].fillna(0.0)
    df['prob_home'] = model.predict_proba(X)[:, 1]
    ...
    # Accuracy metrics (identical layout)
    # Season accuracy table (identical)
    # Kelly bankroll sim (identical loop)
    # Plotly charts (identical)
```

**Why this works well for backtesting:** The backtest loop (load data -> filter seasons -> predict -> measure accuracy -> simulate bankroll -> plot) is structurally identical. The only variations are:
- Data loading function
- Feature list
- Season format (NFL: 2024, NHL: 2024-25)
- Column names for date/teams
- O/U MAE and units

These are all easily parameterized.

**Estimated savings:** ~80-120 lines removed from each sport file = **~160-240 lines eliminated**.

### 2.4 Adapter Pattern (Good for: Data Pipeline Normalization)

**What it is:** Convert the interface of a class into another interface that clients expect. Allows classes with incompatible interfaces to work together.

**Where it applies (future, not immediate):** If you add NBA, you'd want each sport's game_week module to return a normalized schedule format:
```python
# Normalized game dict
{
    'game_id': str,
    'home_team': str,
    'away_team': str,
    'game_date': str,       # ISO format
    'game_time': str,       # ET
    'venue': str,
    'moneyline_home': float,
    'moneyline_away': float,
    'total_line': float,
    'sport': str,
}
```

**Current state:** Each sport's game_week module returns slightly different dict structures. This is fine at 3 sports — not worth abstracting now, but worth noting for NBA.

**Recommendation:** Don't build adapters now. Instead, document the expected dict shape in `EdgeIQ.md` so new sports follow the pattern naturally.

### 2.5 Composite UI Components (Recommended for: Game Card Rendering)

**What it is:** Extract repeated Streamlit widget patterns into shared rendering functions.

```python
# shared_ui.py

def render_win_probability(home_team, away_team, prob_home, prob_away):
    """Two-column win probability display with progress bars."""
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"🏠 {home_team} Win Prob", f"{prob_home*100:.1f}%")
        st.progress(float(prob_home))
    with col2:
        st.metric(f"✈️ {away_team} Win Prob", f"{prob_away*100:.1f}%")
        st.progress(float(prob_away))

def render_confidence_badge(winner, prob):
    label, css = confidence_tier(prob)
    st.markdown(
        f'<div class="signal-badge {css}">{label}: {winner}</div>',
        unsafe_allow_html=True)

def render_kelly_sizing(config, model_prob, moneyline, prob_home, prob_away,
                        home_team, away_team):
    """Shared Kelly bet sizing UI block — 4 metric columns + caption."""
    # ... 40 lines of identical rendering logic
    # Only input: sport config (for session state keys) + game data
```

**Estimated savings:** ~55 lines removed from each sport file = **~110 lines eliminated**.

---

## 3. Pitfalls and Anti-Patterns

### 3.1 PITFALL: Premature Deep Abstraction

**The temptation:** Create `BaseSportApp`, `NFLApp(BaseSportApp)`, `NHLApp(BaseSportApp)` with abstract methods for every tab.

**Why it fails in Streamlit:**
- Streamlit re-executes the entire script on every widget interaction
- Class instances don't persist across reruns (must go into session_state)
- `st.cache_data` doesn't work with methods on mutable objects
- Tab rendering involves deeply nested widget trees that don't decompose cleanly into method overrides
- You end up with `super().render_game_card()` calls that are harder to debug than the copy-paste they replaced

**The EdgeIQ-specific risk:** NFL has 7 tabs including Head-to-Head and Super Bowl Predictor (unique to football). NHL and MLB have 5 tabs each but different ones. A base class would need 10+ abstract methods, most of which are only implemented by 1 sport. That's not polymorphism — that's an interface compliance tax.

**Rule of thumb:** If more than 30% of the methods in your abstract base class are only implemented by one subclass, you don't have a hierarchy — you have forced conformity.

### 3.2 PITFALL: Over-Parameterizing Shared Functions

**The temptation:** Make `render_prediction_result()` generic by adding 15 parameters for every sport-specific variation.

```python
# DON'T DO THIS
def render_prediction_result(
    result, home, away, prefix, game_date,
    show_ou=True, ou_unit='pts', ou_mae=10.01,
    show_lineup_adj=False, show_weather=False,
    show_defensive_matchup=False, show_elo_base=True,
    show_pitcher_info=False, pitcher_field='starting_pitcher',
    show_goalie_info=False, goalie_field='starting_goalie',
    spread_available=True, moneyline_primary=False,
    ...
):
```

**Why it fails:** Boolean parameter explosion creates functions that are harder to understand than the original duplicated code. Each sport ends up calling the function with a unique combination of 15 flags, and debugging requires tracing through conditionals that are harder to follow than reading the sport-specific version.

**Better approach:** Extract only the parts that are truly identical (Kelly sizing, confidence badges, win prob display) as small, focused shared functions. Keep sport-specific assembly in each sport file.

### 3.3 PITFALL: Abstracting UI Layout Too Aggressively

**The temptation:** Create a generic `render_game_card(game, sport_config)` that works for all three sports.

**Why it's risky:** Game cards have genuinely different content:
- NFL: spread, weather, injury adjustments, lineup depth charts, defensive matchup grades
- NHL: goalie matchup, ice advantage, rest days
- MLB: starting pitcher matchup, park factors, bullpen data

The layout chrome (expander, columns, predict button) is similar, but the content inside is 60-70% different. Abstracting the layout forces you to either:
1. Pass content as callbacks (awkward in Streamlit)
2. Use slot-based rendering (not idiomatic in Streamlit)
3. Conditionalize everything (back to boolean flags)

**Recommendation:** Don't abstract game card content. Do abstract the prediction *result* display that appears after the predict button is clicked — that part is genuinely identical.

### 3.4 PITFALL: Shared State Manager

**The temptation:** Create a `BettingState` class that wraps `st.session_state` and handles all bankroll/strategy/risk reads.

**Why it's fragile:**
- Session state in Streamlit is global and must be set before widgets render
- Widget `key=` parameters directly bind to session state keys
- A state manager that reads/writes session state can cause ordering bugs (reading before widget sets the value)
- It adds a layer of indirection that makes debugging "why is my slider value wrong" much harder

**Better approach:** Use the config dict to centralize the *key names*, but let each sport file read session state directly. The config dict eliminates typos/drift without hiding the Streamlit mechanics.

### 3.5 PITFALL: Generic Backtesting Engine With Plugin Architecture

**The temptation:** Build a backtesting framework with hooks: `on_load`, `on_predict`, `on_evaluate`, `on_render`.

**Why it's overkill for 3 sports:** Plugin architectures pay off at 10+ plugins. With 3 sports, the "framework" is harder to modify than the individual implementations. When you want to add a sport-specific backtesting feature (e.g., NFL's "bet the spread" simulation), you have to either extend the plugin interface or bypass the framework — both are worse than just editing the sport file.

**Better approach:** Extract the shared *data processing* (predict, measure accuracy, simulate bankroll) into pure functions. Keep the *UI rendering* in each sport file. The data functions are easily testable and truly identical; the UI rendering has enough variation to justify per-sport code.

### 3.6 PITFALL: DRY-ing Things That Only Look Similar

Some code looks duplicated but serves different evolutionary purposes:
- **NFL backtesting** has spread-based betting, moneyline simulation, AND Kelly simulation (3 strategies)
- **NHL backtesting** has game-specific moneyline derivation from model probability with 4.55% vig
- **MLB backtesting** uses flat -110 league average ML for Kelly sim

These will continue to diverge as each sport's backtesting matures. Forcing them into a shared function now means you'll be fighting the abstraction later when NHL needs game-specific moneyline bucketing or MLB adds run-line betting.

**Rule of thumb:** Only DRY code that you're confident will **stay identical**. The Kelly formula will never vary by sport. The backtesting simulation logic is already diverging.

---

## 4. Recommended Refactoring Plan

### Phase 1: Extract Pure Functions (Lowest Risk, Highest Value)

Create `betting.py` — a pure-function module with zero Streamlit imports:

| Function | Current Locations | Lines Saved |
|----------|-------------------|-------------|
| `kelly_criterion(prob, ml, frac)` | `_kelly_rec()`, `_nhl_kelly()`, `_mlb_kelly()` | ~45 |
| `compute_bet_amount(strategy, bankroll, kelly_pct, fixed_pct, fixed_dol)` | 3x inline if/elif blocks | ~30 |
| `vegas_implied_prob(ml)` | 6+ inline calculations | ~15 |
| `confidence_tier(prob)` | 3x inline if/elif blocks | ~24 |
| `ml_pnl(odds, won, stake)` | `_ml_pnl()` in nhl_app + final_app + mlb_app | ~15 |
| `implied_prob_from_ml(ml)` | `_implied_prob()` in nhl_app + final_app | ~10 |

**Total: ~140 lines eliminated, zero risk of breaking sport-specific behavior.**

Note: `parlay_math.py` already follows this pattern. `betting.py` is the same idea for non-parlay betting math.

### Phase 2: Extract Shared UI Components (Medium Risk, Medium Value)

Create `shared_ui.py` — Streamlit rendering helpers:

| Function | Current Locations | Lines Saved |
|----------|-------------------|-------------|
| `render_win_probability(home, away, prob_h, prob_a)` | 3x inline | ~24 |
| `render_confidence_badge(winner, prob)` | 3x inline | ~12 |
| `render_kelly_sizing(config, prob, ml, home, away)` | 3x inline | ~120 |
| `render_sidebar_betting_controls(config)` | 3x inline | ~30 |
| `render_prediction_log(config, result, game_date)` | 3x inline | ~45 |

**Total: ~230 lines eliminated. Moderate risk — requires careful testing of session state key routing.**

### Phase 3: Shared Backtesting Data Functions (Medium Risk, Medium Value)

Add to `betting.py` or create `backtest.py`:

| Function | Lines Saved |
|----------|-------------|
| `run_backtest(df, model, features)` — predict + accuracy + bankroll sim | ~60 |
| `season_accuracy_table(df)` — groupby season, compute accuracy | ~20 |
| `kelly_bankroll_simulation(df, initial=1000)` — vectorized sim | ~30 |

**Total: ~110 lines eliminated. Keep per-sport rendering in each app file.**

### Phase 4: Shared Track Record Tab (Lower Priority)

The Track Record tab is the most duplicated UI section (prediction history + bet tracker + export). But MLB's version is simpler than NHL's. Recommend waiting until MLB Track Record is feature-complete before extracting.

### What NOT to Refactor

| Leave As-Is | Reason |
|-------------|--------|
| Game card content (inside expanders) | 60-70% different per sport |
| Feature engineering modules | Completely sport-specific |
| Data pipeline orchestration | Different APIs per sport |
| Schedule fetching | Different APIs per sport |
| Prop prediction UI content | Different stat types per sport |
| Model training scripts | Completely sport-specific |

---

## 5. Configuration Objects vs Class Hierarchies: Decision Framework

### When to Use Config Objects (Dicts/Dataclasses)

Use when:
- Differences are in **data** (string keys, numeric thresholds, labels)
- Functions are **stateless** (Streamlit re-execution model)
- You need `st.cache_data` compatibility (no mutable objects)
- The "inheritance" is only 1 level deep (sport -> config)
- You're working in a script-oriented framework (Streamlit)

EdgeIQ verdict: **Config objects are the right choice.** All sport variations in the duplicated code reduce to different session state key prefixes, different labels, and different numeric thresholds.

### When to Use Class Hierarchies

Use when:
- Behavior genuinely differs (different algorithms, not just parameters)
- Objects maintain state across calls
- You have 3+ levels of specialization
- The framework supports OOP patterns natively

EdgeIQ verdict: **Not appropriate.** Streamlit's execution model (full re-run per interaction) means class instances are ephemeral. `st.cache_resource` could hold them, but then you're fighting the framework.

---

## 6. Real-World Multi-Domain ML Platform Patterns

### 6.1 scikit-learn's Own Pattern

scikit-learn itself is the best example of multi-domain ML with shared infrastructure:
- **Shared:** `Pipeline`, `cross_val_score`, `GridSearchCV`, `train_test_split`
- **Pluggable:** Estimators with `fit()/predict()/predict_proba()` interface
- **Config-driven:** Hyperparameters are plain dicts passed to constructors

EdgeIQ already follows this: all three sport models use `GradientBoostingClassifier + RandomForest -> LogisticRegression` stacking. The training scripts are sport-specific (different features), but the model interface is identical.

### 6.2 Betting Industry Pattern: "Pricer" Architecture

Professional sportsbooks use a common architecture:
- **Pricer module** — pure math: Kelly, EV, implied prob, edge calculation (sport-agnostic)
- **Market module** — sport-specific: how to interpret lines, derive fair odds
- **Risk module** — portfolio: daily exposure, bankroll management (sport-agnostic)
- **Display module** — UI: bet slips, game cards (mostly shared, sport-specific content)

EdgeIQ's natural mapping:
- `betting.py` = Pricer module (Phase 1 above)
- `feature_engineering.py` / `nhl_feature_engineering.py` / `mlb_feature_engineering.py` = Market module
- `parlay_math.py` = already exists as a shared Risk/Strategy module
- `shared_ui.py` = Display module (Phase 2 above)

### 6.3 FiveThirtyEight / The Athletic Models

These multi-sport prediction platforms use:
- **Shared ELO engine** — same K-factor tuning framework, sport-specific parameters
- **Shared evaluation framework** — Brier score, calibration plots, accuracy over time
- **Sport-specific feature pipelines** — completely separate
- **Shared visualization templates** — same chart types, different data

EdgeIQ already has this with `elo_ratings.pkl`, `nhl_elo_ratings.pkl`, `mlb_elo_ratings.pkl` — all built by sport-specific scripts but consumed the same way.

### 6.4 The "Hexagonal Architecture" for ML Apps

The pattern that best fits EdgeIQ's structure:
```
[Core Domain]        betting.py, parlay_math.py     (pure logic, no framework deps)
[Application Layer]  backtest.py, shared_ui.py       (Streamlit rendering helpers)
[Adapters]           final_app.py, nhl_app.py, etc.  (sport-specific orchestration)
[Ports]              apis/*                           (external data sources)
```

The key insight: the **core domain** (betting math) should have zero Streamlit imports. The **application layer** can import Streamlit. The **adapters** (sport app files) orchestrate everything. This matches the existing `parlay_math.py` pattern.

---

## 7. Migration Strategy: How to Do This Safely

### Step 1: Create `betting.py` with tests

Write the pure functions. Import them in one sport file (say MLB, smallest file). Run the app, verify nothing changes. Then swap NFL and NHL.

### Step 2: Create `shared_ui.py` with render helpers

Start with `render_kelly_sizing()` — the single largest block of identical code. Wire it into MLB first, verify, then NHL and NFL.

### Step 3: Use config dicts to parameterize session state keys

Define `SPORT_CONFIGS` in `sport_config.py`. Import in each sport file. Replace hardcoded `'nhl_bet_strategy'` with `config['strategy_key']`.

### Step 4: Shared backtesting data functions

Extract `run_backtest()` into `backtest.py`. Wire into MLB first (simplest backtesting tab), then NHL, then NFL.

### Step 5: Do NOT attempt to merge the three app files

`final_app.py`, `nhl_app.py`, and `mlb_app.py` should remain separate files. They are the "adapters" in the hexagonal architecture — they orchestrate sport-specific behavior. The goal is to make them thinner by extracting shared code, not to eliminate them.

---

## 8. Summary Decision Matrix

| Component | Abstract? | Pattern | Priority | Risk |
|-----------|----------|---------|----------|------|
| Kelly formula | YES | Pure function in `betting.py` | P1 | Very Low |
| Vegas implied prob | YES | Pure function in `betting.py` | P1 | Very Low |
| Confidence tiers | YES | Pure function in `betting.py` | P1 | Very Low |
| Bet strategy routing | YES | Pure function in `betting.py` | P1 | Very Low |
| ML P&L calculation | YES | Pure function in `betting.py` | P1 | Very Low |
| Kelly UI rendering | YES | Shared function in `shared_ui.py` | P2 | Low |
| Win prob display | YES | Shared function in `shared_ui.py` | P2 | Low |
| Sidebar betting controls | YES | Shared function in `shared_ui.py` | P2 | Low |
| Session state keys | YES | Config dicts in `sport_config.py` | P2 | Low |
| Backtesting data processing | YES | Functions in `backtest.py` | P3 | Medium |
| Track Record tab | MAYBE | Wait for MLB parity, then extract | P4 | Medium |
| Game card content | NO | Sport-specific | - | - |
| Feature engineering | NO | Sport-specific | - | - |
| Prop prediction content | NO | Sport-specific | - | - |
| Data pipelines | NO | Sport-specific | - | - |
| Model training | NO | Sport-specific | - | - |

**Total estimated lines eliminated: ~480-580 from 8,125 (6-7%).**
**Total estimated complexity reduction: Moderate — main benefit is single source of truth for betting math.**
**Main benefit: Adding NBA (or any 4th sport) requires ~200 fewer lines of copy-paste.**
