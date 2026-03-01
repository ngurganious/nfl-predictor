# EdgeIQ — Product Definition

> **This document defines what EdgeIQ is.** For how to build it, see `CLAUDE.md`. For what to build next, see `PRD.md`.

---

## 1. Product Vision

**EdgeIQ** is a multi-sport ML prediction platform that turns model-driven probability estimates into actionable betting intelligence. It is not a tipster service — it is a *decision-support tool* that shows users where their model has an edge over the Vegas line and how to size bets responsibly.

**The user promise:**
> "I know what the market thinks. EdgeIQ tells me if the model disagrees — and by how much."

---

## 2. Brand Principles

| Principle | What It Means in Practice |
|-----------|--------------------------|
| **Edge-first** | Every prediction is framed relative to Vegas, not in isolation. Model prob vs implied prob is always shown. |
| **Transparent confidence** | Accuracy, MAE, and model limitations are always surfaced. We don't hide that NFL tops out at ~70%. |
| **Responsible sizing** | Kelly Criterion is the default. Hard caps and daily exposure limits are enforced. We never encourage chasing. |
| **Parity across sports** | Every sport gets the same prediction surfaces: win prob, O/U, Kelly sizing, schedule view, backtesting. NHL ≠ second-class. |
| **Minimal input, maximum signal** | Users should get a full prediction from just two team names. Manual overrides (spread, weather, lineups) improve it, not require it. |

---

## 3. What EdgeIQ Is NOT

- Not a live odds aggregator (we use The Odds API as a data source, not as the product)
- Not a generic parlay calculator (EdgeIQ offers structured parlay ladders with model-backed legs, not arbitrary parlays)
- Not a lock service (no guaranteed picks — confidence tiers make uncertainty explicit)
- Not real-time (predictions update when the schedule is loaded, not tick-by-tick)

---

## 4. Sports Coverage

| Sport | Status | Model Accuracy | Tabs |
|-------|--------|---------------|------|
| **NFL** | ✅ Full | 69.3% (26-feature stacking ensemble) | Game Predictor, Player Props, Head-to-Head, Super Bowl Predictor, Backtesting |
| **NHL** | ✅ Core | 58.0% (29-feature stacking ensemble) | Game Predictor, Backtesting |
| NBA | 🔲 Planned | — | — |
| MLB | 🔲 Planned | — | — |

---

## 5. Feature Standards (All Sports)

These are non-negotiable across every sport EdgeIQ supports.

### 5.1 Game Card — Required Surfaces
Every game card in the weekly schedule view must show:
1. **Win probability** — home % vs away %, progress bar, confidence tier badge
2. **Vegas ML implied probability** — shown alongside model prob for direct comparison
3. **O/U prediction** — model total vs Vegas line, OVER/UNDER lean, edge in points/goals
4. **Kelly Bet Sizing** — 4-column layout (Model Edge, Half-Kelly %, Bet Amount $, Signal badge)
5. **Predicted winner** — visible on the collapsed card label with win % (no need to expand)

### 5.2 Win Probability Tiers
| Tier | Threshold | Badge |
|------|-----------|-------|
| High confidence | > 65% | 🔥 HIGH |
| Moderate confidence | 58–65% | ✅ MODERATE |
| Toss-up | < 58% | ⚠️ TOSS-UP |
| Lock (planned) | > 75% | 🔒 LOCK |

*Bet recommendations are only surfaced when model confidence ≥ 55%.*

### 5.3 Kelly Bet Sizing — Standard
```
#### 📐 Kelly Bet Sizing
──────────────────────────────────────────────────────
  Model Edge ⓘ  │  Half-Kelly ⓘ  │  Bet Amount ⓘ  │  Signal ⓘ
    +21.3%       │     20.0%       │      $200       │  💎 STRONG
──────────────────────────────────────────────────────
Betting on [TEAM] at XX.X% model confidence. Vegas implied: XX.X%.
Half-Kelly caps at 20% of bankroll to limit volatility.
```
- Rendered: `st.columns(4)` + `st.metric()`, caption via `st.caption()`
- Formula: `kelly_pct = (b*p - q) / b` — b = decimal odds − 1, p = P(win), q = 1 − p
- Default fraction: Half-Kelly (0.5×)
- Max single bet: 10% of bankroll
- Min single bet: 1% of bankroll
- Max daily exposure: 25% of bankroll (tracked via daily summary)

### 5.4 Signal Badges
| Badge | Threshold | Color |
|-------|-----------|-------|
| 💎 STRONG | Model edge ≥ 4% | `#22c55e` green |
| 📈 LEAN | Model edge 2–4% | `#eab308` yellow |
| 👀 SMALL | Model edge 1–2% | neutral |
| ⚪ PASS | Model edge < 1% | `#94a3b8` gray |

### 5.5 O/U Prediction — Standard
| Field | Value |
|-------|-------|
| Output | Model total, Vegas line, lean (OVER/UNDER), edge magnitude |
| Confidence | Strong ≥ 4pts/goals \| Moderate 2–4 \| Slight < 2 |
| MAE shown | NFL: ±10.01 pts \| NHL: ±1.1 goals |
| Model | Ridge regression on residual (actual − Vegas line) |

### 5.6 Backtesting Tab — Standard
Every sport must have a Backtesting tab with:
- Last 5 seasons accuracy breakdown (by-season table)
- Game-by-game results (season-selectable)
- $10 flat moneyline simulation → cumulative P&L chart
- Kelly Criterion strategy simulation (quarter / half / full Kelly selector)
- Bankroll input in sidebar

### 5.7 Schedule View — Standard
- "This Week's Games" mode: API-fed, grouped by day, all predictions pre-calculated on load
- "Manual Entry" mode: two-team entry with optional condition overrides
- Expand All / Collapse All controls
- Predicted winner + win % visible on collapsed card label (no expand required)

### 5.8 Recursive Parlay Ladder (RPL) — "The EdgeIQ Ladder"

> A structured parlay strategy that turns high-confidence props into tiered ladders. The Banker anchor subsidizes longer-shot parlays — break-even by design.

#### Player Props Tab — Game Card Selection
- **One expandable card per game** (same layout as Game Predictor tab)
- **Expand a card** → shows **game-level bets** (moneyline win, O/U lean) + **top 5 player props**, all ranked by model confidence
- Each selectable leg shows: bet description, model prediction, **prediction accuracy %**, Vegas line, edge, **selection toggle**
- **All bet types are valid ladder legs** — outright wins, over/unders, and player props are treated equally and ranked together by confidence
- **Cross-game selection:** toggle legs from multiple expanded game cards; selections persist across expand/collapse
- **Auto-selection:** top 3 legs by confidence (across all games, all bet types) pre-selected as Banker base
- **"🪜 Build Ladder" button** (visible when ≥3 legs selected) → navigates to Parlay Ladder tab

#### Parlay Ladder Tab — 4 Sections
- **"🪜 Parlay Ladder" tab** in both `final_app.py` and `nhl_app.py`
- 4 sections (one per tier), each showing: leg details, combined odds, **overarching prediction %**, stake, payout

#### 5.8.1 Leg Eligibility
- **Leg types:** moneyline (outright win), Over/Under, player props — all treated equally
- Pool: any bet where model confidence **P ≥ 75%** (Lock-tier)
- Ranked by descending confidence across all games and bet types
- Per game: up to 2 game-level bets (ML + O/U) + top 5 player props shown
- Minimum pool: **3 legs** for Banker only

#### 5.8.2 Tier Structure — Dynamic Sizing
Tier leg counts are **computed dynamically**, not hardcoded. The optimization algorithm:
1. Banker tier: smallest N-leg parlay where payout ≥ total ladder cost (might be 2, 3, or 4)
2. Accelerator tiers: partition remaining legs to maximize EV separation between tiers
3. Moonshot tier: all remaining selected props

| Tier | Name | Legs | Sizing Rule |
|------|------|------|-------------|
| 1 | **The Safety** ("Banker") | N₁ | Smallest parlay where payout ≥ total ladder cost |
| 2 | **The Growth** ("Accelerator 1") | N₂ | Tier 1 + next best; combined P still viable |
| 3 | **The Growth** ("Accelerator 2") | N₃ | Tier 2 + next best; crosses into long-shot |
| 4 | **The Jackpot** ("Moonshot") | N₄ | All remaining selected props |

Each tier prominently displays its **overarching prediction %** (product of individual leg probabilities, adjusted for correlation).

#### 5.8.3 Anchor Break-Even Rule
The Banker tier's payout at combined parlay odds **must equal or exceed** the total wager across all four tiers. The tier optimization solves for this constraint dynamically.

#### 5.8.4 Correlation Filter
- Do not combine two "under" props from the same high-scoring projected game
- Do not combine opposing-side props that conflict (e.g., QB1 over passing + QB2 over passing in a low-total game)
- Flag (warn) when multiple props from the same game are selected — inform user of correlation risk
- Filtered/flagged props shown with reason in the ladder view

#### 5.8.5 Stake Sizing
- Total ladder wager respects **25% max daily exposure** cap (§6)
- Tier 1 (Banker): largest stake — sized so payout ≥ total ladder cost
- Tiers 2–4: equal smaller stakes (remaining budget split evenly)
- Kelly Criterion applies to total ladder allocation, not individual legs

#### 5.8.6 Ladder Tab UI
Each tier section shows:
| Element | Description |
|---------|-------------|
| Tier header | Name + leg count (e.g., "🏦 The Safety — 3 Legs") |
| Overarching prediction % | Combined probability of all legs hitting |
| Leg detail table | Player, game, prop type, model prediction, accuracy %, Vegas line |
| Combined parlay odds | American odds equivalent |
| Suggested stake | Dollar amount from break-even math |
| Potential payout | Stake × combined odds |

Ladder-wide summary above the tier sections:
| Element | Description |
|---------|-------------|
| Break-even status | Banker payout vs total ladder cost |
| Tier size rationale | "Optimized: N₁ / N₂ / N₃ / N₄ legs" |
| Correlation flags | Filtered/flagged props with reasons |
| Historical ladder ROI | Backtested from prop data |
| Volatility note | "The Banker keeps you in the game while waiting for the Moonshot hit" |

---

### 5.9 Prediction History — Cross-Sport Standard

> Prediction History is **not** Backtesting. Backtesting re-runs the model on historical data to validate accuracy. Prediction History logs what EdgeIQ predicted *in real use*, so you can audit your actual edge over time.

Every sport that renders game predictions must:

1. **Auto-log** each prediction at render time to `prediction_history.json` (one record per game, upsert on game ID to avoid duplicates)
2. **Back-fill results** once the game is played (via ESPN/NHL API scoreboard, already in codebase)
3. **Provide a Results Review tab** with:
   - Filterable table: date range, sport, signal tier, outcome
   - Columns: Date | Matchup | EdgeIQ Pick | Win Prob | Kelly Signal | Result | ✅/❌
   - Summary: overall hit rate %, signal-tier hit rate breakdown, average edge when correct
   - Linked to Bet Tracker: show placed bet + P&L inline per row (if applicable)

**Canonical prediction record schema (sport-agnostic):**
```json
{
  "id": "{sport}_{home}_{away}_{game_date}",
  "sport": "nfl | nhl",
  "game_date": "YYYY-MM-DD",
  "home_team": "KC",
  "away_team": "BUF",
  "predicted_at": "ISO-8601 timestamp",
  "model_home_prob": 0.623,
  "vegas_ml_home": -175,
  "vegas_implied_prob": 0.637,
  "model_edge_pct": 4.2,
  "kelly_signal": "STRONG",
  "kelly_pct": 4.2,
  "ou_line": 47.5,
  "model_total": 51.2,
  "ou_lean": "OVER",
  "actual_winner": null,
  "actual_score_home": null,
  "actual_score_away": null,
  "actual_total": null,
  "prediction_correct": null,
  "ou_correct": null
}
```
Fields starting with `actual_` are `null` until the game is complete.

---

## 6. Bankroll Management Standards

These constants are enforced identically in all sports and all bet sizing contexts.

| Constant | Value |
|----------|-------|
| Default bankroll | $1,000 |
| Min bankroll | $100 |
| Max bankroll | $100,000 |
| Max single bet | 10% of bankroll |
| Min single bet | 1% of bankroll |
| Max daily exposure | 25% of bankroll |
| Kelly cap | 10% (hard) |

### Betting Strategy Options (sidebar)
1. **Kelly Criterion** (default) — formula-driven, risk tolerance multiplier applied
2. **Fixed %** — user sets a flat percentage per bet
3. **Fixed $** — user sets a flat dollar amount per bet
4. **Fractional Kelly** — explicit 0.5× or 0.25× selector

### Risk Tolerance → Kelly Multiplier
| Tolerance | Kelly Multiplier |
|-----------|-----------------|
| Conservative | 0.25× |
| Moderate (default) | 0.5× |
| Aggressive | 1.0× |

---

## 7. ELO Rating System

Both sports use ELO as a core predictive feature. Parameters differ to account for sport pace.

| Parameter | NFL | NHL |
|-----------|-----|-----|
| K-factor | 20 | 6 |
| Home advantage | 48 pts | 28 pts |
| ELO trend window | 4 games | 4 games |
| Season regression | Yes (standard) | Yes (standard) |
| ELO → prob | Logistic sigmoid | Logistic sigmoid |

---

## 8. Specialty Rating (QB / Goalie Quality)

Every sport with a "star player" impact surface must implement a seasonal quality z-score:

| Field | NFL | NHL |
|-------|-----|-----|
| Rating method | Per-(player, season) z-score | Per-(goalie, season) z-score |
| Stats used | Completion %, Y/A, TD/INT | Save %, GAA, Quality Starts |
| Feature name | `qb_score_diff` | `goalie_quality_diff` |
| Unknown player | 0 (league average) | 0 (league average) |
| Coverage | 2010+ | Full history |

---

## 9. Color Palette

| Use | Hex | Context |
|-----|-----|---------|
| Strong / Win / Positive | `#22c55e` | Strong bet badge, win result |
| Decent / Lean | `#eab308` | Lean badge, moderate signal |
| Skip / Neutral | `#94a3b8` | Pass badge, toss-up |
| Loss / Negative | `#ef4444` | Loss result, warning |

*Applied via `st.markdown(unsafe_allow_html=True)` — Streamlit's default theme colors are overridden where these standards apply.*

---

## 10. Data Philosophy

- **No paid data sources required to run the app.** Free APIs (ESPN, NHL API, Open-Meteo, PFR) cover all live data. The Odds API (500 req/month free tier) is optional but strongly recommended.
- **Models are trained offline** — pickle files are committed to the repo. The app never trains at runtime.
- **Cache-first API calls** — all external calls go through `apis/cache.py` with TTL. Cold data never blocks UI.
- **Practical accuracy ceiling acknowledged** — NFL at ~70% and NHL at ~58% represent the ceiling for free tabular public data. EdgeIQ is honest about this rather than overstating model power.
- **Feature leakage is actively guarded** — all training uses TimeSeriesSplit. No future data leaks into past predictions.

---

## 11. Document Map

| File | Purpose | Audience |
|------|---------|---------|
| `EdgeIQ.md` | **This file** — product definition, brand, standards | Product / design |
| `CLAUDE.md` | Developer instructions for Claude Code | Claude AI assistant |
| `PRD.md` | Living requirements — what to build next, gap analysis | Planning / build sessions |
