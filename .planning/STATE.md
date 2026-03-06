# EdgeIQ — Project State

## Current Position

- **Milestone:** 1 — Architecture + UI Rework
- **Phase:** 0.75 — Game Card Standardization (DONE)
- **Status:** Phases 0, 0.5, 0.75 complete. Caching + Homepage + Cards all done. Ready for Phase 1 (Core Extraction).
- **Next action:** `/gsd:plan-phase 1` for Core Extraction or continue with approved plan Parts

## Decisions

| Decision | Choice | Date |
|----------|--------|------|
| Full GSD onboarding | Yes — rethink project structure from scratch | 2026-03-05 |
| Keep all 3 sports | NFL + NHL + MLB stay | 2026-03-05 |
| Stay on Streamlit | Custom CSS for SaaS look, no framework migration | 2026-03-05 |
| Same features, done better | No new features — polish existing ones | 2026-03-05 |
| Architecture first | Reorganize codebase before UI or tests | 2026-03-05 |
| Config over inheritance | Plain dicts + pure functions, not class hierarchies | 2026-03-05 |
| Incremental migration | Module by module, never break production | 2026-03-05 |
| Caching first | Fix performance issues before architecture rework | 2026-03-05 |

## Blockers

None.

## Research Completed

- `.planning/research/stack.md` — Streamlit SaaS design patterns
- `.planning/research/architecture.md` — Python ML project structure
- `.planning/research/features.md` — pytest for ML pipelines
- `.planning/research/pitfalls.md` — Cross-sport DRY abstraction patterns

## Last Updated

2026-03-06 — Phase 0.75 (Card Standardization) complete. All 3 sports standardized.
