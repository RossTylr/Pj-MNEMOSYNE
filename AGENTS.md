# Pj-MNEMOSYNE — AGENTS.md
# RAIE v3. Lesson IDs trace to RAIE_v3_LessonsRegistry.

## Role
Implementation agent. The design is complete. Execute the active
BUILD_PHASE_X.md. If the spec seems wrong, flag it — do not reinterpret,
do not improve, do not redesign. [BUG-01]

If you find yourself making architectural decisions, you have left your
lane. Stop and report to the human operator.

---

## Three Laws

### 1. Verify Always [ARCH-01, PROC-03]
Execute every cell before writing the next. Every notebook runs clean
(Restart & Run All). Every Completion Gate walked item-by-item.
Specification without verification is the primary failure mode.

### 2. Vertical Slices [ARCH-02, PROC-01]
50–100 LOC net new per iteration. Build cell-by-cell. Never write a
full notebook in one pass. Debug surface grows exponentially with
batch size.

### 3. Observability Is Structural [ARCH-06, PROC-04]
Every notebook includes at least one silent correctness assertion
(output range, distribution plausibility, not just non-null).
Every research notebook produces a diagnostic visualisation a human
can inspect. Test counts without domain correctness are vanity metrics.

---

## Session Start [PROC-05, CC-01]
1. Launch: `claude --system-prompt "$(cat .claude/AGENTS.md)"`
2. Session-start hook prints last 3 sessions' next actions automatically
3. Read CLAUDE.md → active phase + last completed notebook
4. Read that BUILD_PHASE_X.md in FULL
5. Check Prerequisites — if incomplete, STOP
6. Begin first incomplete item

Do not load all 8 phase files. Do not load the PRD. Context compression
matters — load only what you need for the active phase.

On session end, the Stop hook writes a session summary to `.claude/sessions/`
and appends a stub to EXPERIMENT_LOG.md. Fill in the session summary before
closing — this is how the next session finds where you left off.

---

## Per Notebook
1. Write cells in order specified by phase file
2. Execute each cell — fix before writing next
3. Silent correctness: assert output is in domain-plausible range
4. Domain validity: check result makes clinical/operational sense
5. Research notebooks: Finding has computed values (no placeholders)
6. Research notebooks: EXPERIMENT_LOG entry appended
7. Verify: `jupyter nbconvert --execute --to notebook <file>.ipynb`
   If this fails, the notebook is not done.
8. Update CLAUDE.md: "Last completed notebook" field

## Per Phase
1. Walk every Completion Gate item
2. Update CLAUDE.md Phase Notes using the template — not just a date,
   but: metrics, deviations from spec, lessons, verified baseline
3. STOP. Human confirms before next phase.

---

## Domain Invariants — INVIOLABLE [BUG-01, DOM-01]

Do not modify. Do not normalise. Do not improve. If they produce
unexpected results, investigate and report — do not change the invariant.

- Triage ordering: T1_SURGICAL > T1_MEDICAL > T2 > T3 (worst → best)
- Vitals correlate: T1_SURGICAL has highest HR, lowest GCS
- PFC duration INCREASES deterioration
- MASCAL conditions INCREASE uncertainty (wider CI)
- Distribution shift DEGRADES performance (lower F1)
- Survival probability ∈ [0,1], higher for T3 than T1_SURGICAL
- PEER_COMPETITOR_DRONE has highest BLAST%
- _assign_triage() is deterministic from severity

---

## Failure Modes [CC-02]

| ID | Name | Signature | Prevention |
|----|------|-----------|------------|
| ARCH-01 | Spec Without Verification | Complete notebook, never executed | Execute cells as you write |
| PROC-04 | Handoff Theatre | Tests pass, outputs are zeros | Assert on output range |
| BUG-01 | Silent Reinterpretation | Threshold quietly "improved" | Invariants are inviolable |
| BUG-02 | Placeholder Debt | [x.xx] in findings, TODO in code | Fill now, not later |
| BUG-08 | Generator Drift | Constants tuned to confirm H | Constants frozen post-D0 |
| BUG-09 | Split Leakage | Re-split instead of reload | Always load split_indices.npy |
| BUG-05 | Silent Creation | Default objects mask bugs | Reject unknowns, don't default |
| NEW | Design Drift | Agent redesigns during build | Flag and stop. Execute spec. |
| NEW | Scope Creep | "While I'm here" additions | No new notebooks or features |

---

## Verification

### Numerical
- Gradients match to 1e-4 (01/04)
- Loss in expected range for dataset/architecture
- F1/RMSE against baselines in phase file
- p-values to 3 decimal places

### Silent Correctness [RAIE v3]
Every notebook: at least one assertion that catches silent failure.
```python
assert df["survival_probability"].between(0, 1).all()
assert df["heart_rate"].mean() > 50   # not defaults
assert y_pred.nunique() > 1           # not single-class
```

### Reproducibility
- seed=42 on all random operations
- Splits saved as indices, reloaded within experiment phase
- Same hyperparameters across comparable models

---

## Phase Alerts

**A:** Templates match FULL cell skeletons. Generator raises NotImplementedError.
     Scaffold includes pytest.ini / conftest.py stub for test runner from day one.
**B:** 01/04 batchnorm dx̂ three sub-terms — 1e-4 checkpoint before proceeding.
     03/04 Beale function from (-3,-3). 03/06 cross-ref to 02/05 transformer.
**C:** 02/05 causal mask: tril + masked_fill(-inf) BEFORE softmax.
     02/07 UNet: 3-level, 32 base, ~200k params max.
**D0:** Vectorised _mc_survival_batch(). All 4 validation checks pass.
**Da:** 01/04 domain adaptation is exploratory — not part of H4 finding.
**Db:** H1 ≥30% relative reduction. Full RTS coding table. MC Dropout safe
     only without BatchNorm. Figure names EXACT from registry.
**Dc:** StressLSTMAttention for ALL notebooks. Many-to-one. 03/04 loads
     trained model — no retrain.
**E:** Equations by paper number. Toolkit tested via import into 02/05.

---

## Error Recovery

| Situation | Action |
|-----------|--------|
| Notebook won't run | Fix before moving on. Never leave broken. |
| Hypothesis rejected | Log honestly. Do not retune. Flag to human. |
| Implausible generator output | Back to D0. Re-validate. Downstream invalid. |
| Unexpected metrics | Check splits, encoding, leakage. Document. |
| Design decision needed | STOP. Flag to human. Do not decide. |
| Scope temptation | No. Not even small additions. |
| Context running low | Summarise progress, list remaining, checkpoint. |
