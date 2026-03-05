# EdgeIQ — Project Context

## What This Is

EdgeIQ is an ML-powered sports betting prediction platform covering NFL, NHL, and MLB. It provides game outcome predictions, player prop projections, Kelly-criterion bankroll sizing, backtesting, and parlay ladder simulations — deployed as a Streamlit web app on Streamlit Community Cloud.

## Core Value

Give sports bettors a data-driven edge by combining ensemble ML models with live odds, producing actionable bet recommendations with proper bankroll sizing.

## Requirements

### Validated (Shipped — working in production)

- NFL game prediction (26-feature GBC+RF stacking ensemble, 69.3% accuracy)
- NHL game prediction (29-feature stacking ensemble, 58.0% accuracy)
- MLB game prediction (29-feature stacking ensemble, 58.0% accuracy)
- NFL/NHL/MLB over/under models
- NFL/NHL/MLB player prop predictions (passing, rushing, receiving / goals, assists, shots / K, ER, hits, TB)
- Kelly criterion bankroll recommendations (all 3 sports)
- Backtesting with historical accuracy validation (all 3 sports)
- Parlay ladder simulator (all 3 sports)
- Track record tab (all 3 sports)
- Multi-sport home page with sport selection
- Live Vegas odds integration (Odds API)
- Live injury data (ESPN)
- Weather impact for NFL
- ELO rating systems (NFL, NHL, MLB)
- Defensive matchup adjustments (NFL)
- QB/goalie/pitcher quality ratings

### Active (Current rework scope)

- **ARCH-01** — Reorganize codebase into modular directory structure (core/, sports/nfl/, sports/nhl/, sports/mlb/, ui/)
- **ARCH-02** — Build shared cross-sport prediction engine (one base class, sport-specific adapters)
- **ARCH-03** — Build shared Kelly/bankroll system (DRY across all sports)
- **ARCH-04** — Build shared backtesting framework (DRY across all sports)
- **ARCH-05** — Build shared player prop framework (DRY across all sports)
- **ARCH-06** — Build shared parlay ladder framework (DRY across all sports)
- **UI-01** — Design system: color palette, typography, spacing, card components
- **UI-02** — Custom CSS framework for professional SaaS appearance
- **UI-03** — Consistent game card layout across all sports
- **UI-04** — Consistent sidebar controls across all sports
- **UI-05** — Polished tab navigation and page structure
- **TEST-01** — pytest infrastructure with fixtures and test utilities
- **TEST-02** — Unit tests for ML model prediction pipelines
- **TEST-03** — Unit tests for feature engineering (NFL, NHL, MLB)
- **TEST-04** — Unit tests for Kelly/bankroll calculations
- **TEST-05** — Integration tests for API data pipelines

### Out of Scope (with reasoning)

- Live score updates / real-time streaming — adds complexity without improving prediction quality
- Push notifications / alerts — no backend infrastructure for this on Streamlit
- User accounts / authentication — single-user tool, no need for auth
- Mobile app — Streamlit handles responsive layout adequately
- New sports (NBA, soccer, etc.) — get the architecture right for 3 sports first
- Model accuracy improvements — models are at practical ceiling for free public data; rework focuses on code quality and UI
- Moving off Streamlit — staying on Streamlit per project decision

## Context

- **Brownfield project** — ~15,000 lines of Python across 50+ files, deployed and functional
- **Solo developer** — one human + Claude Code as engineering partner
- **Free data only** — no paid data providers; ESPN, NHL API, MLB Stats API, pybaseball, nfl-data-py
- **Model accuracy ceilings** — NFL 69.3%, NHL 58.0%, MLB 58.0% represent practical limits with public data
- **Existing users** — app is live on Streamlit Cloud, changes must not break the deployed version
- **Get Shit Done workflow** — using GSD for structured discuss → plan → execute → verify cycle
- **Oh My Claude Code** — hooks for context monitoring, commit quality, TDD enforcement, delegation

## Constraints

| Constraint | Details |
|-----------|---------|
| **Framework** | Streamlit (staying — no framework migration) |
| **Language** | Python 3.11 |
| **Hosting** | Streamlit Community Cloud (free tier) |
| **API keys** | RAPIDAPI_KEY (Tank01), ODDS_API_KEY (the-odds-api.com) — both in .env / Streamlit Secrets |
| **No paid data** | All data from free APIs and open-source libraries |
| **Backward compat** | Deployed app must continue working throughout rework (incremental, not big-bang) |
| **No new deps** | Prefer using existing dependencies; new packages only if clearly necessary |
| **Model files** | .pkl files must remain compatible — do not retrain models during the architecture rework |

## Key Decisions

| Decision | Choice | Rationale | Date |
|----------|--------|-----------|------|
| Framework | Stay on Streamlit | Faster iteration, Python-native, existing deployment works | 2026-03-05 |
| Sport scope | All 3 (NFL, NHL, MLB) | Already built — rework is about quality, not cutting scope | 2026-03-05 |
| Feature scope | Same features, done better | Polish and engineer existing features before adding new ones | 2026-03-05 |
| Priority #1 | Architecture / file structure | Clean foundation enables everything else (UI, tests, new features) | 2026-03-05 |
| Approach | Incremental refactor | Don't break production — migrate module by module | 2026-03-05 |
| Workflow | GSD (get-shit-done) | Structured phases with research, planning, execution, verification | 2026-03-05 |
| Timeline | Quality over speed | No rush — get the architecture and UI right | 2026-03-05 |

## Last Updated

2026-03-05 — Initial GSD onboarding, full rethink of project structure and priorities.
