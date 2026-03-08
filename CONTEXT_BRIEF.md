# Pj-MNEMOSYNE — Context Brief

## Project
38-notebook ML learning repo + 13-notebook research track producing a
publishable surrogate model for military MEDEVAC simulation (FAER(MIL)).
P(survival | patient_state, system_state, t+12h) → JDMS / MORS

## Current state (7 March 2026)
Fully reviewed build plan (8 phase files, 3 review passes). CLAUDE.md and
AGENTS.md written to RAIE v3 standard. Phase A is entry point — nothing
built in code yet.

## Phases
A (scaffold) → B (spine, 14 nb) → C (arch zoo, 7 nb) → D0 (generator) →
Da (baseline+H4, 4 nb) → Db (surrogate H1-H3, 5 nb, PRIMARY) →
Dc (sequence H5, 4 nb, deferrable) → E (papers+toolkit)

## Hypotheses
- H1: System-state features reduce RMSE ≥30% vs patient-only
- H2: PFC duration is dominant patient-side predictor
- H3: MC Dropout CI widens under MASCAL
- H4: Peacekeeping model degrades on drone-warfare data, ΔF1 >0.15
- H5: LSTM beats MLP on MASCAL stress forecasting, ΔF1 ≥0.10

## Operating constraints
- 50–100 LOC per iteration; execute every cell before writing the next
- No design work during implementation — spec is complete, execute it
- No scope additions — nothing beyond what the phase files specify
- Generator constants frozen after D0; clinical thresholds inviolable
- Silent correctness: every notebook asserts on output domain range
- seed=42; splits saved and reloaded; EXPERIMENT_LOG is append-only
- CLAUDE.md under 80 lines; phase detail in BUILD_PHASE files only
- Human confirms at every phase boundary before next phase starts

## Failure modes to watch
- Spec without verification — execute cells as you write [ARCH-01]
- Silent reinterpretation — clinical invariants are not suggestions [BUG-01]
- Placeholder debt — no [x.xx], no TODO in committed code [BUG-02]
- Handoff theatre — assert on output ranges, not just no-exception [PROC-04]
- Generator drift — record rejections honestly [BUG-08]
- Design drift — if you're designing, you've left your lane [NEW]
- Scope creep — no new notebooks, features, or "while I'm here" [NEW]

## Synthetic data
All results reflect MnemosyneGenerator encodings. Findings describe the
modelling framework, not clinical discoveries. Validity notes in every
finding cell. Full statement in RESEARCH_NOTES.md Threats to Validity.

## Start
Launch: `claude --system-prompt "$(cat .claude/AGENTS.md)"`
Session-start hook shows last sessions' next actions.
Read CLAUDE.md → active phase + last completed notebook.
Read that BUILD_PHASE_X.md in full. Check Prerequisites.
Load only the active phase file.
