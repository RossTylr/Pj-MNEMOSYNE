# Pj-MNEMOSYNE — CLAUDE.md

## Project
38-notebook ML learning repo + 13-notebook research track.
Surrogate model for military MEDEVAC: P(survival | patient, system, t+12h)
Venues: JDMS / MORS

## State
Active phase: BUILD_PHASE_C.md
Last completed notebook: 03_training_science/06_learning_rate_scheduling.ipynb
Last updated: 2026-03-08

## Phases
- [x] A — Scaffold
- [x] B — Spine (00, 01, 03 — 14 notebooks)
- [ ] C — Architecture Zoo (02 — 7 notebooks)
- [ ] D0 — Generator validation
- [ ] Da — Experiment 1: Baseline + H4 (4 notebooks)
- [ ] Db — Experiment 2: Surrogate H1 H2 H3 (5 notebooks, PRIMARY)
- [ ] Dc — Experiment 3: Sequence H5 (4 notebooks, deferrable)
- [ ] E — Papers + diagnostics toolkit

## Layout
```
00_foundations/  01_from_scratch/  02_architecture_zoo/
03_training_science/  05_paper_implementations/
06_research_track/
├── data/mnemosyne_synthetic/     single data source
├── 01_experiment_baseline/       H4
├── 02_experiment_surrogate/      H1 H2 H3
└── 03_experiment_sequence/       H5
07_diagnostics_toolkit/
```

## Rules
1. Launch: `claude --system-prompt "$(cat .claude/AGENTS.md)"`
2. Check `.claude/sessions/` for last session's next action
3. Read active BUILD_PHASE_X.md in full before writing code
4. 50–100 LOC per iteration; execute every cell before writing the next
5. Completion Gate passed item-by-item; human confirms at phase boundary
6. Update State section above when a notebook or phase completes

## Conventions
- seed=42; splits saved and reloaded within experiment phases
- Clinical data in mnemosyne_synthetic/generator.py — single source
- EXPERIMENT_LOG.md is append-only; findings include validity notes
- Templates: NOTEBOOK_TEMPLATE.ipynb / EXPERIMENT_TEMPLATE.ipynb

## Inviolable
- No building out of sequence
- No scope additions — Section 04 deferred, no new notebooks or features
- No fabricated results, no [x.xx] placeholders in committed notebooks
- No generator constant changes without D0 re-validation
- No silent reinterpretation of clinical thresholds — flag and wait
- No design work during implementation — execute the spec, don't redesign it

## References
BUILD_PHASE_A→E.md | AGENTS.md | RESEARCH_NOTES.md | EXPERIMENT_LOG.md

## Phase Notes
### Phase A — 2026-03-08
Completed: Full scaffold — 38 notebooks, 13 research notebooks, templates, hooks, tests
Key metrics: pytest -m smoke PASSED (1/1), both session hooks run clean
Deviation from spec: none
Lesson learned: Bulk notebook creation via script is efficient for scaffold phases
Verified baseline for next phase: directory tree matches spec, templates match full cell skeletons, generator raises NotImplementedError

### Phase B — 2026-03-08
Completed: 13 spine notebooks (00/02, 00/03, 01/01–05, 03/01–06) + 5 new smoke tests (8 total)
Key metrics: pytest -m smoke PASSED (8/8), all 13 notebooks execute cleanly
Enhancement: Added 40 three-tier explainers (plain/intuition/formal) + 20 new visualizations across all 13 notebooks
Smoke tests added: linear algebra shapes, Beta-Bernoulli posteriors, batchnorm backward, training loop convergence, Kaiming activation stability
Deviation from spec: none
Lesson learned: Shapiro-Wilk test too sensitive for CLT verification with large samples — use moment matching instead
Verified baseline for next phase: all pedagogical spine content complete, ready for Architecture Zoo (Phase C)
