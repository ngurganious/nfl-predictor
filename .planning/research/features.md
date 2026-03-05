# Testing ML Prediction Pipelines with pytest — Research & Recommendations

> Practical guide for adding a pytest test suite to EdgeIQ (~50 Python files,
> Streamlit + scikit-learn, three sports, no existing tests).

---

## 1. Project Structure & Organization

### Recommended directory layout

```
tests/
    conftest.py                 # Shared fixtures: sample DataFrames, mock models, tmp paths
    test_parlay_math.py         # Pure math — odds, parlays, tiers, correlations
    test_kelly.py               # Kelly criterion functions (all three sports)
    test_feature_engineering.py  # NFL feature_engineering.py
    test_nhl_features.py        # nhl_feature_engineering.py
    test_mlb_features.py        # mlb_feature_engineering.py
    test_prediction_pipeline.py # Model loading → feature matrix → predict_proba shape/range
    test_data_pipeline.py       # DataPipeline with mocked APIs
    test_cache.py               # apis/cache.py — TTL, expiry, clear
    test_prediction_history.py  # JSON persistence — log, load, upsert, dedup
    test_abbreviations.py       # _stats_key(), NHL_HISTORICAL_ALIASES, NAME_TO_ABV dicts
    test_game_week.py           # Schedule parsing (all 3 sports)
    fixtures/
        sample_nfl_games.csv    # ~50 rows of games_processed.csv for feature engineering
        sample_nhl_games.csv    # ~50 rows of nhl_games_processed.csv
        sample_mlb_games.csv    # ~50 rows of mlb_games_processed.csv
        mock_odds_response.json # Typical Odds API JSON (one game)
        mock_nhl_schedule.json  # Typical NHL API schedule response
        mock_mlb_schedule.json  # Typical MLB Stats API schedule response
```

### Key structural decisions

- **One `conftest.py` at `tests/`** with shared fixtures. Do NOT scatter conftest
  files into subdirectories unless the test suite grows past ~30 files.
- **Name test files by module**, not by "unit" vs "integration". `test_parlay_math.py`
  clearly maps to `parlay_math.py`.
- **`fixtures/` subdirectory** holds static CSV/JSON files that tests load. These are
  small subsets of production data (50 rows, not 60,000). Commit them to git — they
  are deterministic test inputs, not caches.
- **`pytest.ini` or `pyproject.toml`** at the repo root:
  ```ini
  # pytest.ini
  [pytest]
  testpaths = tests
  python_files = test_*.py
  python_functions = test_*
  markers =
      slow: marks tests that load real .pkl models (deselect with -m "not slow")
  ```

### What goes in `conftest.py`

```python
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ── Sample DataFrames ────────────────────────────────────────────────

@pytest.fixture
def sample_nfl_games():
    """Minimal games_processed.csv subset (50 rows) with all required columns."""
    return pd.read_csv(FIXTURES_DIR / "sample_nfl_games.csv")

@pytest.fixture
def sample_nhl_games():
    return pd.read_csv(FIXTURES_DIR / "sample_nhl_games.csv")

@pytest.fixture
def sample_mlb_games():
    return pd.read_csv(FIXTURES_DIR / "sample_mlb_games.csv")

# ── Mock model ───────────────────────────────────────────────────────

@pytest.fixture
def mock_sklearn_model():
    """Fake model that returns deterministic probabilities."""
    model = MagicMock()
    model.predict_proba = MagicMock(
        return_value=np.array([[0.35, 0.65]])  # away 35%, home 65%
    )
    model.predict = MagicMock(return_value=np.array([1]))
    return model

# ── Temp directory for cache tests ───────────────────────────────────

@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Override cache.CACHE_DIR to use pytest's tmp_path."""
    import apis.cache as cache_mod
    original = cache_mod.CACHE_DIR
    cache_mod.CACHE_DIR = tmp_path
    yield tmp_path
    cache_mod.CACHE_DIR = original
```

---

## 2. Testing Feature Engineering Functions

Feature engineering is the **highest-value testing target** in an ML project. These
are pure-ish functions (DataFrame in, DataFrame out) with clear contracts.

### What to test

| Test | Why |
|------|-----|
| Output has all expected columns | Catches renames, typos, missing features |
| Output row count == input row count | Feature engineering should never drop rows |
| No NaN in feature columns (after fillna) | NaN in model input = silent prediction garbage |
| Feature values in expected ranges | ELO diff should be roughly [-600, 600], probabilities [0, 1] |
| Symmetric inputs produce symmetric outputs | If home and away are identical, diff features should be 0 |
| Column order matches `ENHANCED_FEATURES` list | Model expects features in training order |
| Edge case: single-row input | Rolling windows should degrade gracefully, not crash |
| Edge case: team with < FORM_WINDOW games | Should get NaN or 0, not KeyError |

### Example: `test_feature_engineering.py`

```python
import pandas as pd
import numpy as np
import pytest
from feature_engineering import build_enhanced_features, ENHANCED_FEATURES

# -- or if no ENHANCED_FEATURES constant, define the expected list inline --

def test_output_columns(sample_nfl_games):
    result = build_enhanced_features(sample_nfl_games)
    for col in ENHANCED_FEATURES:
        assert col in result.columns, f"Missing feature column: {col}"

def test_row_count_preserved(sample_nfl_games):
    result = build_enhanced_features(sample_nfl_games)
    assert len(result) == len(sample_nfl_games)

def test_no_nan_in_final_features(sample_nfl_games):
    result = build_enhanced_features(sample_nfl_games)
    # Only check the last 20 rows (earlier rows may have NaN from rolling windows)
    tail = result.tail(20)
    for col in ENHANCED_FEATURES:
        if col in tail.columns:
            nan_count = tail[col].isna().sum()
            assert nan_count == 0, f"{col} has {nan_count} NaN in tail 20 rows"

def test_probabilities_in_range(sample_nfl_games):
    result = build_enhanced_features(sample_nfl_games)
    for col in ['spread_implied_prob', 'elo_implied_prob']:
        if col in result.columns:
            valid = result[col].dropna()
            assert valid.between(0, 1).all(), f"{col} has values outside [0, 1]"

def test_symmetric_matchup(sample_nfl_games):
    """When a team plays itself, all diff features should be ~0."""
    row = sample_nfl_games.iloc[0].copy()
    row['away_team'] = row['home_team']
    row['away_score'] = row['home_score']
    df = pd.DataFrame([row] * 10)
    result = build_enhanced_features(df)
    last = result.iloc[-1]
    # ELO diff should be close to 0 for same-team matchups
    # (may not be exactly 0 due to home advantage constant)
```

### MLB `_stats_key()` — a critical pure function

```python
from mlb_feature_engineering import _stats_key

@pytest.mark.parametrize("team,season,expected", [
    ("AZ",  2024, "ARI"),     # Arizona Diamondbacks
    ("CWS", 2024, "CHW"),     # Chicago White Sox
    ("TB",  2024, "TBR"),     # Tampa Bay Rays (post-2007)
    ("TB",  2005, "TBD"),     # Tampa Bay Devil Rays (pre-2008)
    ("MIA", 2024, "MIA"),     # Miami Marlins (post-2011)
    ("MIA", 2010, "FLA"),     # Florida Marlins (pre-2012)
    ("NYY", 2024, "NYY"),     # Passthrough — no mapping needed
    ("KC",  2024, "KCR"),
    ("SD",  2024, "SDP"),
    ("SF",  2024, "SFG"),
    ("WSH", 2024, "WSN"),
    ("LA",  2024, "LAD"),
])
def test_stats_key(team, season, expected):
    assert _stats_key(team, season) == expected
```

This is a **must-have** test. The abbreviation bridge is a frequent source of bugs
when joining games data with FanGraphs stats.

### NHL features — same pattern

```python
from nhl_feature_engineering import build_nhl_enhanced_features, NHL_ENHANCED_FEATURES

def test_nhl_feature_count(sample_nhl_games):
    result = build_nhl_enhanced_features(sample_nhl_games)
    missing = [f for f in NHL_ENHANCED_FEATURES if f not in result.columns]
    assert missing == [], f"Missing NHL features: {missing}"
```

---

## 3. Testing Model Prediction Pipelines

### What to test

The question "does the model predict correctly?" is a **training concern**, not a
unit test concern. The test suite should verify the **plumbing**, not the accuracy.

| Test | What it verifies |
|------|-----------------|
| Model loads from .pkl without error | File not corrupted, pickle-compatible |
| `predict_proba()` returns shape (n, 2) | Binary classification contract |
| Probabilities sum to ~1.0 per row | Model output is well-formed |
| Feature matrix has correct column count | 26 for NFL, 29 for NHL, 29 for MLB |
| Prediction with all-zero features doesn't crash | Degenerate input resilience |
| Prediction with NaN features raises or returns NaN | Not silently wrong |

### Mocking vs loading real models

**Two strategies** — use both:

1. **Mock model (fast, no .pkl needed)**: For testing pipeline plumbing — that
   the code correctly assembles features, calls `predict_proba()`, and interprets
   the output. Use `unittest.mock.MagicMock` with a canned return value.

2. **Real model (slow, marked `@pytest.mark.slow`)**: For smoke tests that verify
   the .pkl files load and produce valid output. Run these in CI or before deploy,
   not on every save.

```python
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

def test_prediction_shape_mock(mock_sklearn_model):
    """Pipeline produces correct output structure with a mock model."""
    features = np.zeros((1, 26))  # 26 NFL features
    proba = mock_sklearn_model.predict_proba(features)
    assert proba.shape == (1, 2)
    assert abs(proba[0].sum() - 1.0) < 0.01

@pytest.mark.slow
def test_real_nfl_model_loads():
    """Smoke test: real model loads and accepts the right feature count."""
    import pickle
    from pathlib import Path
    model_path = Path(__file__).parent.parent / "model_enhanced.pkl"
    if not model_path.exists():
        pytest.skip("model_enhanced.pkl not found")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    dummy = np.zeros((1, 26))
    proba = model.predict_proba(dummy)
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0

@pytest.mark.slow
def test_real_nhl_model_loads():
    import pickle
    from pathlib import Path
    model_path = Path(__file__).parent.parent / "model_nhl_enhanced.pkl"
    if not model_path.exists():
        pytest.skip("model_nhl_enhanced.pkl not found")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    dummy = np.zeros((1, 29))
    proba = model.predict_proba(dummy)
    assert proba.shape == (1, 2)
```

### Do NOT test accuracy in unit tests

Accuracy is a property of the data + algorithm + hyperparameters. It fluctuates
with retraining and is validated by the `build_*_model.py` scripts (which print
CV scores and holdout accuracy). Unit tests that assert `accuracy > 0.69` are
**brittle** and **meaningless** — they pass when the model is overtrained and
fail when you change training data.

The right place for accuracy validation:
- `build_mlb_model.py` prints `accuracy: 58.0%` on the holdout set
- `retrain_model.py` prints CV and holdout scores
- Backtesting tab in the Streamlit app

---

## 4. Testing API Data Pipelines

### Strategy: mock at the HTTP layer, not the client layer

The `apis/` modules (`odds.py`, `nhl.py`, `mlb.py`, etc.) all use `requests.get()`
internally plus `apis/cache.py` for caching. The best testing pattern mocks
`requests.get` and injects deterministic JSON responses.

### Using `responses` library (recommended over `unittest.mock`)

```python
# pip install responses
import responses
import json

@responses.activate
def test_odds_api_returns_spread():
    """OddsClient returns parsed spread when API responds normally."""
    mock_body = {
        "id": "abc123",
        "home_team": "Kansas City Chiefs",
        "away_team": "Buffalo Bills",
        "bookmakers": [{
            "key": "draftkings",
            "markets": [{
                "key": "spreads",
                "outcomes": [
                    {"name": "Kansas City Chiefs", "point": -3.5, "price": -110},
                    {"name": "Buffalo Bills", "point": 3.5, "price": -110},
                ]
            }]
        }]
    }
    responses.add(
        responses.GET,
        "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds",
        json=[mock_body],
        status=200,
        headers={"x-requests-remaining": "499"},
    )
    from apis.odds import OddsClient
    client = OddsClient(api_key="test-key")
    odds = client.get_nfl_odds()
    assert len(odds) >= 1
```

### Alternative: `unittest.mock.patch` (no extra dependency)

```python
from unittest.mock import patch, MagicMock

def test_nhl_schedule_parsing():
    """NHLClient parses schedule JSON into game dicts."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "games": [
            {
                "id": 2024020001,
                "startTimeUTC": "2024-10-04T23:00:00Z",
                "homeTeam": {"abbrev": "TOR"},
                "awayTeam": {"abbrev": "MTL"},
                "gameState": "FUT",
            }
        ]
    }
    with patch("apis.nhl.requests.get", return_value=mock_response):
        from apis.nhl import NHLClient
        client = NHLClient()
        # ... test schedule parsing logic
```

### What to test in API modules

| Test | Why |
|------|-----|
| Normal response parses correctly | Happy path |
| API returns 404/500 → graceful fallback | App should not crash on API downtime |
| API returns empty data → empty result | No KeyError on missing keys |
| Cache hit skips HTTP call | Verify `apis/cache.get()` short-circuits |
| Team name mapping is correct | `NAME_TO_ABV` dicts cover all teams |

### What NOT to test in API modules

- **Do not test the external API itself.** If the NHL changes their JSON schema,
  your test suite should not suddenly fail — that is a monitoring/alerting concern.
- **Do not make real HTTP calls in tests.** Every test must be offline-capable.
  Use `@responses.activate`, `@patch`, or fixture JSON files.

### Testing the cache module (`apis/cache.py`)

The cache is simple and self-contained — test it directly:

```python
import time
from apis import cache

def test_cache_set_and_get(tmp_cache_dir):
    cache.set("test_key", {"data": 42}, ttl=60)
    result = cache.get("test_key")
    assert result == {"data": 42}

def test_cache_expiry(tmp_cache_dir):
    cache.set("expire_key", "old", ttl=1)
    time.sleep(1.1)
    assert cache.get("expire_key") is None

def test_cache_clear(tmp_cache_dir):
    cache.set("k1", "v1", ttl=60)
    cache.set("k2", "v2", ttl=60)
    cache.clear()
    assert cache.get("k1") is None
    assert cache.get("k2") is None
```

---

## 5. Testing Kelly Criterion & Bankroll Calculations

These are **pure math functions** — no side effects, no state, no I/O. They are
the **easiest and most valuable** tests to write.

### Extracting Kelly for testability

Currently, `_kelly_rec()` is defined inside `render_nfl_app()` (a local function),
`_nhl_kelly()` is inside `render_nhl_prediction_result()`, and `_mlb_kelly()` is
at module level in `mlb_app.py`. For testability, consider:

**Option A (no refactor):** Test `_mlb_kelly` directly (it is already importable).
For NFL/NHL, duplicate the logic in test code and verify known inputs/outputs.

**Option B (recommended refactor):** Extract all Kelly functions into a shared
module (e.g., `kelly.py` or add to `parlay_math.py`). Then all three sports call
the same function. This is also a Phase 1 roadmap item (standardize Kelly).

### Test cases for Kelly

```python
from mlb_app import _mlb_kelly  # or from kelly import kelly_rec

@pytest.mark.parametrize("prob,ml,frac,expected_tier", [
    # Strong edge: 70% model prob, -150 favorite
    (0.70, -150, 0.5, "STRONG EDGE"),
    # Lean: 60% prob, -110 line
    (0.60, -110, 0.5, "LEAN"),
    # Pass: 52% prob, -110 (no edge over vig)
    (0.52, -110, 0.5, "PASS"),
    # Underdog: 45% prob, +200 (Kelly positive because b=2.0, 2*0.45-0.55 > 0)
    (0.45, 200, 0.5, "LEAN"),
    # Edge case: 50/50 at even odds
    (0.50, 100, 0.5, "PASS"),
    # Edge case: 100% probability
    (1.0, -200, 0.5, "STRONG EDGE"),
])
def test_kelly_tier(prob, ml, frac, expected_tier):
    pct, css_class, badge = _mlb_kelly(prob, ml, frac)
    assert badge == expected_tier

def test_kelly_pct_capped_at_10():
    """Kelly should never exceed 10% of bankroll (hard cap)."""
    pct, _, _ = _mlb_kelly(0.95, 100, 1.0)
    assert pct <= 10.0

def test_kelly_never_negative():
    """Kelly should never recommend a negative bet."""
    pct, _, _ = _mlb_kelly(0.30, -200, 0.5)
    assert pct >= 0.0

def test_kelly_zero_odds_returns_pass():
    """Zero moneyline should not crash."""
    pct, _, badge = _mlb_kelly(0.60, 0, 0.5)
    # The function catches this with try/except
    assert badge == "PASS"
```

### Testing `parlay_math.py`

This is the **most testable module in the codebase** — zero dependencies, pure math.

```python
from parlay_math import (
    american_to_decimal, decimal_to_american, implied_probability,
    combined_parlay_decimal, combined_probability, parlay_ev,
    optimize_tiers, compute_stakes, check_correlations,
)

class TestOddsConversion:
    def test_favorite_to_decimal(self):
        assert american_to_decimal(-200) == 1.5

    def test_underdog_to_decimal(self):
        assert american_to_decimal(200) == 3.0

    def test_even_to_decimal(self):
        assert american_to_decimal(100) == 2.0

    def test_roundtrip(self):
        """decimal → american → decimal should be identity (within rounding)."""
        for ml in [-300, -150, -110, 100, 150, 300]:
            dec = american_to_decimal(ml)
            back = decimal_to_american(dec)
            assert abs(back - ml) <= 1, f"Roundtrip failed for {ml}"

class TestImpliedProbability:
    def test_heavy_favorite(self):
        assert abs(implied_probability(-300) - 0.75) < 0.01

    def test_even(self):
        assert implied_probability(100) == 0.5

    def test_underdog(self):
        assert abs(implied_probability(200) - 0.333) < 0.01

class TestParlayMath:
    @pytest.fixture
    def three_legs(self):
        return [
            {'odds': -110, 'confidence': 0.60},
            {'odds': -110, 'confidence': 0.58},
            {'odds': 150,  'confidence': 0.45},
        ]

    def test_combined_probability(self, three_legs):
        prob = combined_probability(three_legs)
        assert 0 < prob < 1
        # Should be product: 0.60 * 0.58 * 0.45 = 0.1566
        assert abs(prob - 0.1566) < 0.001

    def test_parlay_ev_positive_means_profitable(self, three_legs):
        ev = parlay_ev(10, three_legs)
        # EV can be positive or negative — just ensure it is a number
        assert isinstance(ev, float)

class TestCorrelationFilter:
    def test_same_game_two_unders_flagged(self):
        legs = [
            {'game_id': 'G1', 'direction': 'UNDER', 'bet_type': 'prop'},
            {'game_id': 'G1', 'direction': 'UNDER', 'bet_type': 'prop'},
        ]
        flags = check_correlations(legs)
        types = [f['conflict_type'] for f in flags]
        assert 'same_game_double_under' in types

    def test_different_games_no_correlation(self):
        legs = [
            {'game_id': 'G1', 'direction': 'OVER', 'bet_type': 'prop'},
            {'game_id': 'G2', 'direction': 'OVER', 'bet_type': 'prop'},
        ]
        flags = check_correlations(legs)
        assert len(flags) == 0

class TestTierOptimization:
    def test_fewer_than_3_legs_returns_empty(self):
        assert optimize_tiers([{'odds': -110, 'confidence': 0.6}] * 2, 100) == []

    def test_3_legs_returns_at_least_1_tier(self):
        legs = [{'odds': -110, 'confidence': 0.6}] * 4
        tiers = optimize_tiers(legs, 100)
        assert len(tiers) >= 1
        assert tiers[0]['name'] == 'The Safety'

    def test_stakes_sum_to_budget(self):
        legs = [{'odds': -110, 'confidence': 0.6}] * 6
        tiers = optimize_tiers(legs, 100)
        sized = compute_stakes(tiers, 100)
        total = sum(t['stake'] for t in sized)
        assert abs(total - 100) < 0.02  # rounding tolerance
```

---

## 6. Testing Streamlit UI Components

### The reality: Streamlit is hard to unit test

Streamlit's execution model (rerun the entire script top-to-bottom on every
interaction) makes traditional unit testing awkward. The available options:

### Option A: `streamlit.testing.v1` (Streamlit >= 1.28, experimental)

```python
from streamlit.testing.v1 import AppTest

def test_home_page_loads():
    at = AppTest.from_file("app.py")
    at.run()
    assert not at.exception
    # Check that sport selection buttons exist
    assert len(at.button) >= 3  # NFL, NHL, MLB

def test_nfl_tab_renders():
    at = AppTest.from_file("app.py")
    at.run()
    # Simulate clicking NFL button
    at.button[0].click()
    at.run()
    assert not at.exception
```

**Limitations of `AppTest`:**
- Experimental API — may change between Streamlit versions
- Cannot test complex session state interactions well
- Cannot test components that depend on live API data (need mocking)
- Slow — each `at.run()` executes the full script
- Does not render CSS or capture visual output

### Option B: Extract logic out of Streamlit functions (recommended)

The most effective strategy is to **separate business logic from rendering**. If
`render_nfl_prediction_result()` does both computation AND `st.metric()` calls,
extract the computation into a pure function and test that:

```python
# Instead of testing this (requires Streamlit context):
def render_nfl_prediction_result(result):
    prob_h = result['home_win_prob']
    winner = result['home_team'] if prob_h > 0.5 else result['away_team']
    conf = max(prob_h, 1 - prob_h)
    if conf > 0.75: tier = "LOCK"
    elif conf > 0.65: tier = "HIGH"
    # ... st.metric(), st.progress(), etc.

# Extract and test the pure logic:
def classify_confidence(prob_h):
    conf = max(prob_h, 1 - prob_h)
    if conf > 0.75: return "LOCK"
    if conf > 0.65: return "HIGH"
    if conf > 0.58: return "MODERATE"
    return "TOSS-UP"

# Now testable:
def test_confidence_tiers():
    assert classify_confidence(0.80) == "LOCK"
    assert classify_confidence(0.70) == "HIGH"
    assert classify_confidence(0.60) == "MODERATE"
    assert classify_confidence(0.52) == "TOSS-UP"
```

### Option C: Snapshot testing (not recommended for this project)

Some teams use `pytest-snapshot` to capture rendered HTML and diff it. This is
brittle for Streamlit apps because Streamlit generates dynamic IDs and the
output format changes between versions. Skip this.

### Recommendation for EdgeIQ

Do NOT invest in Streamlit UI tests initially. Instead:

1. **Extract pure logic** from `final_app.py`, `nhl_app.py`, `mlb_app.py` into
   testable helper functions (Kelly, confidence classification, bet sizing, etc.)
2. **Test the helpers** thoroughly
3. **Manual testing** for UI rendering (this is the current approach and it works)
4. Consider `AppTest` smoke tests later (one per sport: "does it load without error?")

---

## 7. What to Test vs What NOT to Test

### High-value test targets (DO test these)

| Module | Why | Effort |
|--------|-----|--------|
| `parlay_math.py` | Pure math, zero deps, 8 public functions, lots of edge cases | Low |
| `_stats_key()` / abbreviation maps | Frequent source of join bugs, parametrize easily | Low |
| `_kelly_rec()` / `_mlb_kelly()` / `_nhl_kelly()` | Pure math, critical for bet sizing | Low |
| `apis/cache.py` | 3 functions, simple I/O, use `tmp_path` | Low |
| `prediction_history.py` | JSON persistence, upsert logic, dedup by ID | Low |
| Feature engineering (output contracts) | Column presence, shape, NaN, ranges | Medium |
| `data_pipeline.py` (with mocked APIs) | Integration plumbing — features assembled correctly | Medium |
| Model loading smoke tests | .pkl files load and produce valid output | Medium |

### Low-value / skip (do NOT test these)

| Module/Concern | Why |
|----------------|-----|
| Model accuracy (e.g., `assert accuracy > 0.69`) | Belongs in training scripts, not unit tests. Accuracy varies with data and is validated during retraining. |
| Exact model predictions for specific inputs | Model is a black box; testing exact outputs is brittle and couples tests to training data. |
| Streamlit rendering (CSS, layout, widget placement) | Requires Streamlit runtime, changes frequently, better tested manually. |
| External API response schemas | Not your code to test. Mock the response, but do not test that NHL's API returns a certain JSON shape. |
| `build_*.py` training scripts | These are one-shot scripts, not library code. They print their own validation metrics. |
| CSV/pickle file contents | These are build artifacts. Test the code that produces them, not the files themselves. |
| ELO rating calculations in `build_*_games.py` | Complex iterative computation — test the feature engineering that consumes ELO, not the ELO builder. |

### The "testing pyramid" for ML projects

```
                    /\
                   /  \           Manual / exploratory
                  / UI \          (Streamlit: run app, click around)
                 /------\
                / Integ. \        Integration tests
               / Pipeline \       (feature eng → model → prediction shape)
              /------------\
             /  Unit Tests  \     Pure functions: Kelly, parlay math,
            / (Pure Logic)   \    abbreviation maps, cache, odds conversion
           /------------------\
```

Invest most effort at the **bottom** (pure logic). These tests are fast, stable,
and catch real bugs. The integration layer (feature engineering contracts) catches
a different class of bugs (schema drift, column renames). UI testing adds the
least value relative to effort.

---

## 8. Fixture Patterns for Sports Data

### Creating sample DataFrames

Do NOT use full production CSVs (60,000+ rows) in tests. Instead, create minimal
fixtures with only the columns that matter.

**Method 1: CSV fixture files (recommended for feature engineering tests)**

```bash
# Extract 50 rows from production data for test fixtures
python -c "
import pandas as pd
df = pd.read_csv('games_processed.csv')
df.tail(50).to_csv('tests/fixtures/sample_nfl_games.csv', index=False)
"
```

Commit these to git. They are small, deterministic, and do not change unless
the schema changes (which is exactly when you want tests to fail).

**Method 2: Inline DataFrame factories (for simple tests)**

```python
@pytest.fixture
def minimal_game_row():
    """Single game row with minimum required columns."""
    return pd.DataFrame([{
        'home_team': 'KC', 'away_team': 'BUF',
        'home_score': 27, 'away_score': 24,
        'gameday': '2024-01-15',
        'elo_diff': 42.0,
        'spread_line': -3.5,
        'home_moneyline': -175,
        'away_moneyline': 150,
        'total_line': 47.5,
        'season': 2024,
    }])
```

**Method 3: Factory functions with parametrize**

```python
def make_game(home='KC', away='BUF', home_score=27, away_score=24, **kwargs):
    """Factory for single game rows — override any field via kwargs."""
    base = {
        'home_team': home, 'away_team': away,
        'home_score': home_score, 'away_score': away_score,
        'gameday': '2024-01-15', 'elo_diff': 0.0,
        'spread_line': 0.0, 'season': 2024,
    }
    base.update(kwargs)
    return pd.DataFrame([base])

def test_home_favorite_elo():
    game = make_game(elo_diff=100)
    # ...
```

### Mock API responses

Store realistic API responses as JSON fixtures:

```
tests/fixtures/
    mock_odds_response.json      # One game from the-odds-api.com
    mock_nhl_schedule.json       # One day from api-web.nhle.com
    mock_mlb_schedule.json       # One day from statsapi.mlb.com
    mock_espn_injuries.json      # Injury report for one team
```

Load them in fixtures:

```python
@pytest.fixture
def mock_odds_json():
    with open(FIXTURES_DIR / "mock_odds_response.json") as f:
        return json.load(f)
```

### Key rule: fixtures must be stable

A test fixture that loads from a CSV file should **never change** unless the
schema of the production data changes. If you regenerate `games_processed.csv`
with new data, the fixture should remain the same 50 rows. This is why you
commit the fixture files and do not regenerate them automatically.

---

## 9. Implementation Roadmap

### Phase 1 — Quick wins (1-2 hours)

1. Create `tests/` directory, `conftest.py`, `pytest.ini`
2. `test_parlay_math.py` — test all 8 public functions in `parlay_math.py`
3. `test_kelly.py` — test `_mlb_kelly()` (already importable) with parametrize
4. `test_abbreviations.py` — test `_stats_key()` and team name dicts
5. `test_cache.py` — test set/get/expiry/clear with `tmp_path`

### Phase 2 — Feature engineering contracts (2-3 hours)

6. Create `tests/fixtures/sample_*.csv` (50-row subsets)
7. `test_feature_engineering.py` — NFL output contracts
8. `test_nhl_features.py` — NHL output contracts
9. `test_mlb_features.py` — MLB output contracts

### Phase 3 — Pipeline & model smoke tests (1-2 hours)

10. `test_prediction_pipeline.py` — mock model + feature matrix shape
11. `test_prediction_history.py` — JSON CRUD operations
12. Model smoke tests (marked `@pytest.mark.slow`)

### Phase 4 — API mocking (2-3 hours, optional)

13. `test_data_pipeline.py` — DataPipeline with all APIs mocked
14. `test_game_week.py` — schedule parsing with fixture JSON

### Running the tests

```bash
# Run all fast tests
pytest

# Run everything including slow model tests
pytest -m ""

# Run a specific test file
pytest tests/test_parlay_math.py -v

# Run with coverage report
pytest --cov=. --cov-report=term-missing --cov-omit="venv/*,tests/*,build_*.py"
```

### Dependencies to add

```
# requirements.txt (test section)
pytest>=7.0
pytest-cov>=4.0
responses>=0.23      # optional, for API mocking
```

---

## 10. Common Pitfalls to Avoid

1. **Testing model accuracy in unit tests.** This is the #1 mistake in ML test suites.
   Model accuracy is a property of training, not code correctness. If you retrain
   on different data, accuracy changes. That does not mean your code is broken.

2. **Making real HTTP calls in tests.** Every API test must work offline. Use
   `@responses.activate`, `@patch("requests.get")`, or fixture JSON. Tests that
   hit real APIs are slow, flaky, and will fail in CI.

3. **Using full production CSVs as fixtures.** 60,000 rows makes tests slow and
   hides the actual test logic. Use 50-row subsets.

4. **Testing Streamlit rendering in pytest.** The ROI is terrible. Extract the
   logic, test the logic, eyeball the rendering.

5. **Asserting exact floating-point values.** Use `pytest.approx()` or
   `abs(actual - expected) < epsilon`. Kelly percentages, probabilities, and
   ELO scores are all floats.

6. **Forgetting `tmp_path` for I/O tests.** The cache and prediction_history
   modules write files. Always redirect to pytest's `tmp_path` fixture so tests
   do not pollute the real project directory.

7. **Importing Streamlit at module level in test files.** If a module under test
   imports `streamlit`, the test will fail unless Streamlit is installed. Use
   `@patch("streamlit")` or only test modules that do not import Streamlit
   (`parlay_math.py`, `feature_engineering.py`, `apis/cache.py`, etc.).

8. **Over-testing trivial code.** Do not write tests for `pd.read_csv()` wrappers
   or simple dict lookups. Test code that has **logic** — conditionals, math,
   loops, state transitions.
