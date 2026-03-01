# Sports Predictor Pro — CLAUDE.md

## Project Overview
A multi-sport ML prediction platform with NFL and NHL sections. Users can:
- Predict game outcomes with win probability, spread, and O/U
- View this week's schedule with pre-calculated predictions on every game card
- Analyze player props (NFL: passing/rushing/receiving yards)
- Run backtesting with Kelly Criterion and $10 flat bet simulations

Deployed at Streamlit Community Cloud. Entry point is `app.py`.

---

## Tech Stack
- **Python 3.11**
- **Streamlit** — UI framework (all tabs, widgets, session state)
- **scikit-learn** — GradientBoostingClassifier + RandomForest stacking ensemble
- **pandas / numpy** — data wrangling
- **plotly** — charts in backtesting tab
- **scipy** — spread_to_prob conversion
- **pytz** — ET timezone handling for NHL schedule dates
- **nfl-data-py** — historical NFL play-by-play and seasonal stats
- **python-dotenv** — local API key loading from `.env`
- **beautifulsoup4 / lxml** — Pro Football Reference scraping

---

## Common Commands

```bash
# Activate virtual environment (always do this first)
venv/Scripts/activate          # Windows
source venv/bin/activate       # Mac/Linux

# Run the app locally
streamlit run app.py

# Retrain NFL model (run seasonally, ~3 min)
python retrain_model.py

# Rebuild NHL model from scratch
python build_nhl_games.py          # fetch + process historical games
python build_nhl_team_stats.py     # team ELO + stats
python build_nhl_goalie_ratings.py # goalie quality ratings
python build_nhl_model.py          # train model (outputs model_nhl_enhanced.pkl)

# Rebuild NFL supporting data
python build_team_stats.py         # current season EPA stats
python build_qb_ratings.py         # QB quality z-scores
python build_player_model.py       # prop prediction models
python build_total_model.py        # O/U model

# Clear API cache (if stale data)
python -c "from apis.cache import clear; clear()"
```

---

## File Structure

```
app.py                      # Entry point — multi-sport home page router
final_app.py                # NFL section — render_nfl_app() — 5 tabs
nhl_app.py                  # NHL section — render_nhl_app() — 2 tabs

apis/
  cache.py                  # Shared JSON file cache with TTL (all APIs use this)
  espn.py                   # ESPN injuries + scoreboard (no key needed)
  nhl.py                    # NHLClient — api-web.nhle.com/v1/ (no key needed)
  odds.py                   # Vegas lines — ODDS_API_KEY
  pfr.py                    # Pro Football Reference scraper
  tank01.py                 # Tank01 live NFL data — RAPIDAPI_KEY
  weather.py                # Open-Meteo stadium weather (no key needed)

# NFL model training scripts
build_model.py              # Original 9-feature GBC model
retrain_model.py            # Full 26-feature stacking ensemble retrainer
build_team_stats.py         # NFL EPA stats (current + historical)
build_qb_ratings.py         # QB quality z-scores from nfl_data_py
build_player_model.py       # Prop models (pass/rush/rec yards)
build_total_model.py        # O/U Ridge regression model
build_rolling_epa.py        # Rolling 5-game EPA (not used in model — flat)
build_rolling_qb_epa.py     # Rolling QB EPA (tested, not used — season z-score is better)

# NHL model training scripts
build_nhl_games.py          # Fetch + process NHL historical games
build_nhl_team_stats.py     # NHL team ELO + stats
build_nhl_goalie_ratings.py # Goalie quality ratings
build_nhl_model.py          # Train NHL stacking ensemble

# NFL support modules
feature_engineering.py      # 26-feature computation (ELO, EPA, form, QB, matchup)
data_pipeline.py            # Orchestrates all 5 APIs for live game context
game_week.py                # ESPN weekly NFL schedule + depth charts
defensive_matchup.py        # 6-position defensive matchup → ±4% win prob adjustment

# NHL support modules
nhl_feature_engineering.py  # NHL feature computation
nhl_game_week.py            # NHL weekly schedule + roster depth charts

# Trained models (pickle files — do not delete)
model.pkl                   # Original NFL GBC (9 features, ~65.6% acc)
model_enhanced.pkl          # NFL stacking ensemble (26 features, 69.3% acc)
model_total.pkl             # NFL O/U Ridge model (54.4% accuracy)
model_nhl_enhanced.pkl      # NHL stacking ensemble (58.0% acc)
model_nhl_total.pkl         # NHL O/U model
elo_ratings.pkl             # NFL ELO ratings dict (all 32 teams)
nhl_elo_ratings.pkl         # NHL ELO ratings dict (all teams)
player_lookup.pkl           # NFL player ID → name/position lookup

# Data files (CSV)
games_processed.csv         # NFL feature-engineered game data 2000–2025
games_raw.csv               # Raw NFL game data
nhl_games_processed.csv     # NHL historical games
team_rolling_stats.csv      # NFL per-team l5 form + ELO trend
team_stats_current.csv      # NFL 2025 EPA / turnover / red zone stats
team_stats_historical.csv   # NFL per-team EPA 2016–2024
qb_ratings.csv              # Historical per-(player, season) QB z-scores
qb_team_ratings.csv         # Current season QB score per team
nhl_goalie_ratings.csv      # Historical per-(goalie, season) ratings
nhl_goalie_team_ratings.csv # Current season goalie per team
nhl_team_stats_current.csv  # NHL current season team stats
nhl_team_stats_historical.csv # NHL historical team stats
def_pass_stats.csv          # NFL defensive pass EPA per team/season
def_rush_stats.csv          # NFL defensive rush EPA per team/season
passing_stats.csv           # NFL player passing stats
rushing_stats.csv           # NFL player rushing stats
receiving_stats.csv         # NFL player receiving stats

# Config
requirements.txt            # pip dependencies
runtime.txt                 # Python 3.11 (for Streamlit Cloud)
.env.example                # API key template — copy to .env and fill in
.gitignore                  # .env, venv/, cache/ excluded
```

---

## API Keys

Copy `.env.example` to `.env` and fill in:
```
RAPIDAPI_KEY=...   # Tank01 NFL live data — rapidapi.com
ODDS_API_KEY=...   # Vegas lines — the-odds-api.com (500 req/month free)
```
ESPN, Open-Meteo, NHL API, and PFR need no keys.

For Streamlit Cloud deployment, add both keys in the app's **Secrets** section (TOML format).

---

## Git Workflow (Required)

1. **Before making changes** — run `git diff` and `git status` to understand current state, then commit any uncommitted work so there's a clean rollback point
2. **After making changes** — run `git diff` to review everything before committing; summarize what changed
3. **Always commit when done** — or explicitly offer to rollback with `git restore` if the changes aren't right
4. Never leave the repo in a dirty state at the end of a task

---

## Code Style Rules

- **No type annotations** unless already present in the file
- **No docstrings** on functions unless already present
- **No comments** unless logic is non-obvious
- Prefer editing existing files over creating new ones
- Keep solutions minimal — don't add features beyond what's asked
- All Streamlit widget keys must be unique — use prefixes (`nfl_`, `nhl_`, `g{idx}_`, etc.)
- Session state keys follow established patterns:
  - NFL game expanders: `g{idx}_stay_open`
  - NHL game expanders: `nhl_g{idx}_expanded`
  - Pre-calc guards: `nfl_precalc_done`, `nhl_precalc_done`
  - Depth chart cache: `dc_{team}`

---

## NFL Model — Key Facts

- **26 features**, GBC + RF stacking ensemble → LogReg meta
- **69.3% accuracy** on 2024–25 holdout (practical ceiling for free public data)
- Top predictors in order: spread_line, qb_score_diff, epa_total_diff, elo_diff
- Features **intentionally excluded** (tested, hurt CV): rest days, is_dome, is_grass,
  div_game, l5_win_pct, rolling game-level EPA, interaction features, situational flags
- O/U model: Ridge on residual (actual_total − vegas_line), 54.4% accuracy

## NHL Model — Key Facts

- **58.0% accuracy** on holdout
- Features: ELO diff, team EPA, goalie quality diff, recent form
- Backtesting Kelly uses **game-specific moneylines** derived from model probability
  + 4.55% vig (not flat -110) — heavy favorites (-200+) require >66% confidence to bet

---

## Testing Approach

No automated test suite. Validation is done by:
1. Running `streamlit run app.py` and manually testing each tab
2. Checking backtesting accuracy numbers in the Backtesting tab
3. Running the relevant `build_*.py` script and checking printed output/logs
4. For model changes: compare CV score in retrain output vs baseline 69.3%

---

## Deployment

Hosted on **Streamlit Community Cloud** from `github.com/ngurganious/nfl-predictor`.
- Branch: `master`
- Main file: `app.py`
- Secrets: `RAPIDAPI_KEY`, `ODDS_API_KEY`

To redeploy after changes: `git add ... && git commit && git push` — Streamlit auto-redeploys.
Cold start takes ~2–3 minutes on first load.
