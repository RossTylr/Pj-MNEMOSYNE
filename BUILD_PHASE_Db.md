# BUILD_PHASE_Db.md — Experiment 2: System-Aware Survival Surrogate

## Phase Goal

Build the primary research contribution.
Five notebooks testing H1, H2, H3.
Every notebook produces at least one paper figure.

This is the paper. Work carefully and iteratively.
If a result is unexpected, understand it before moving on.

## Prerequisites

- [ ] Phase Da complete — H4 logged, baseline F1 established
- [ ] Phase B 03/04_optimisers.ipynb complete (Adam, weight decay)
- [ ] Phase B 03/05_regularisation.ipynb complete (dropout — needed for 04)
- [ ] Phase C 02/04_attention_mechanism.ipynb complete (conceptual readiness — partial dependence
  uses a similar sweep-and-observe pattern, but does not reuse attention code)

## Completion Gate

- [ ] H1 confirmed or rejected with RMSE values logged
- [ ] H2 confirmed or rejected with PFC ranking logged
- [ ] H3 confirmed or rejected with CI width comparison logged
- [ ] RESEARCH_NOTES.md structured abstract Methods and Results sections filled
- [ ] All five paper figures produced and saved to `06_research_track/PAPER_DRAFT/figures/`
- [ ] `pytest -m smoke` passes with new Db tests added

## Smoke Test Additions

Add to `tests/test_smoke.py` when Phase Db completes:
```python
@pytest.mark.smoke
def test_survival_probabilities_in_range():
    """Phase Db: surrogate model outputs valid probabilities."""
    # Load saved test predictions, assert all in [0, 1]
    # Assert mean prediction != 0 and != 1 (not collapsed)

@pytest.mark.smoke
def test_system_features_improve_rmse():
    """Phase Db: H1 — full-feature RMSE < patient-only RMSE."""
    # Load saved RMSE values from ablation, assert D < C
```

---

## Before Starting: Paper Figure Registry

Create `06_research_track/PAPER_DRAFT/figures/` directory.
Save every figure from this phase here with consistent naming:

```
fig1_pathway_diagram.png        ← hand-drawn or diagram tool, not a notebook
fig2_ablation_rmse.png          ← from 02/03_ablation_study.ipynb
fig3_pfc_sensitivity.png        ← from 02/05_pfc_sensitivity_curve.ipynb
fig4_uncertainty_intervals.png  ← from 02/04_uncertainty_quantification.ipynb
fig5_distribution_shift.png     ← from 01/04 (Phase Da) — already exists
```

---

## 02/01_patient_only_baseline.ipynb

**Hypothesis:** A patient-only MLP achieves RMSE > 0.14 on survival
probability prediction, establishing a baseline that system-state
features must improve upon.

### Cell sequence

**Cell 1 — Hypothesis**

**Cell 2 — Generate survival dataset**
```python
from data.mnemosyne_synthetic.generator import MnemosyneGenerator

gen = MnemosyneGenerator("HIGH_INTENSITY", seed=42)
df  = gen.generate_survival_dataset(n=10_000, n_mc_runs=500)

PATIENT_FEATURES = [
    "severity", "triage_category", "mechanism", "primary_region",
    "is_polytrauma", "heart_rate", "systolic_bp", "respiratory_rate",
    "gcs", "spo2", "pfc_duration_mins"
]
TARGET = "survival_probability"
```

Note: `triage_category` needs label encoding before use as a feature.

**Cell 3 — Train/val/test split**
70/15/15. No stratification needed (continuous target).
Fix random_state=42. Use this same split for all Db notebooks.
Save indices: `np.save(ROOT / "06_research_track" / "02_experiment_surrogate" / "split_indices.npy", {"train":tr_idx,"val":va_idx,"test":te_idx})`

**Cell 4 — Majority/mean baseline**
```python
# Predict mean survival probability for all test examples
mean_pred = np.full(len(y_te), y_tr.mean())
rmse_mean = np.sqrt(np.mean((mean_pred - y_te)**2))
print(f"Mean prediction baseline RMSE: {rmse_mean:.4f}")
```

**Cell 5 — TRISS benchmark**
Implement simplified TRISS:
```python
def rts_score(gcs: float, sbp: float, rr: float) -> float:
    """
    Revised Trauma Score — Champion et al. coding table.

    GCS  coded: 13-15→4, 9-12→3, 6-8→2, 4-5→1, 3→0
    SBP  coded: >89→4, 76-89→3, 50-75→2, 1-49→1, 0→0
    RR   coded: 10-29→4, >29→3, 6-9→2, 1-5→1, 0→0

    RTS = 0.9368×GCS_c + 0.7326×SBP_c + 0.2908×RR_c
    """
    # GCS coding
    if gcs >= 13:   gcs_c = 4
    elif gcs >= 9:  gcs_c = 3
    elif gcs >= 6:  gcs_c = 2
    elif gcs >= 4:  gcs_c = 1
    else:           gcs_c = 0

    # SBP coding
    if sbp > 89:    sbp_c = 4
    elif sbp >= 76: sbp_c = 3
    elif sbp >= 50: sbp_c = 2
    elif sbp >= 1:  sbp_c = 1
    else:           sbp_c = 0

    # RR coding
    if 10 <= rr <= 29: rr_c = 4
    elif rr > 29:      rr_c = 3
    elif rr >= 6:      rr_c = 2
    elif rr >= 1:      rr_c = 1
    else:              rr_c = 0

    return 0.9368*gcs_c + 0.7326*sbp_c + 0.2908*rr_c


def approximate_iss(severity: float, is_polytrauma: int) -> float:
    """
    Approximate ISS from severity.
    ISS range: 1-75. Map severity [0,1] to [1,75].
    Polytrauma adds a multiplicative factor (multiple body regions).
    This is an approximation — real ISS requires AIS per body region.
    """
    base_iss = 1 + severity * 74
    if is_polytrauma:
        base_iss = min(75, base_iss * 1.3)
    return base_iss


def triss_survival(rts: float, iss: float,
                   age_factor: float = 0.0) -> float:
    """
    TRISS probability of survival.
    b = b0 + b1*RTS + b2*ISS + b3*age_factor
    Ps = 1 / (1 + exp(-b))
    Blunt coefficients (Copes et al.):
        b0=-1.2470, b1=0.9544, b2=-0.0768, b3=-1.9052
    """
    b = -1.2470 + 0.9544*rts + (-0.0768)*iss + (-1.9052)*age_factor
    return 1.0 / (1.0 + np.exp(-b))

# Compute RTS and ISS from our features, then TRISS survival
```

Compute TRISS RMSE on test set. This is the clinical benchmark to beat.

**Cell 6 — Patient-only MLP**
```python
class SurvivalMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32),         nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1),          nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze(-1)
```

Train: 100 epochs, Adam lr=1e-3, MSE loss.
Log: train RMSE, val RMSE per epoch.

**Cell 7 — Loss function comparison (brief)**
Show train curves for MSE, MAE, Huber on same model.
Justify MSE selection (smooth gradients near zero, penalises large errors).

**Cell 8 — Results**
Table: Mean-baseline RMSE / TRISS RMSE / Patient-only MLP RMSE.

**Cell 9 — Finding**
```
Finding: Patient-only MLP RMSE = [x.xx] (mean baseline=[x.xx], TRISS=[x.xx]).
Baseline established. Hypothesis [confirmed/rejected].
MLP [beats/does not beat] TRISS — [interpretation].
Note: Results reflect MnemosyneGenerator synthetic distributions, not empirical data.
```

**Cell 10 — EXPERIMENT_LOG append**

---

## 02/02_system_state_features.ipynb

**Hypothesis:** Adding system-state features reduces RMSE by ≥ 0.04
in absolute terms vs the patient-only baseline.

### Cell sequence

**Cell 1 — Hypothesis**

**Cell 2 — Load existing split and add system features**
```python
SYSTEM_FEATURES = [
    "r1_capacity_pct", "r2_surgical_queue", "transport_eta_mins",
    "route_threat_level", "r2_blood_stock_units", "concurrent_casualties"
]
FULL_FEATURES = PATIENT_FEATURES + SYSTEM_FEATURES
```
Load split indices from 02/01 — same train/val/test split.

**Cell 3 — Feature correlation heatmap**
Correlate all features with `survival_probability`.
Which system features have highest correlation?
Expected: `transport_eta_mins`, `r1_capacity_pct`, `concurrent_casualties`.

**Cell 4 — Train full-feature model**
Same architecture as 02/01 SurvivalMLP, extended input dimension.
100 epochs, same hyperparameters.

**Cell 5 — RMSE comparison**
Patient-only RMSE vs full-feature RMSE. Compute absolute reduction.

**Cell 6 — Finding**
```
Finding: Full-feature RMSE = [x.xx]. Reduction from patient-only = [x.xx].
Hypothesis [confirmed/rejected] (threshold was 0.04).
Top system-state correlates: [list top 3 by |correlation|].
Note: Reduction expected by construction — system-state features are direct
inputs to _mc_survival(). Value is in quantifying the magnitude.
```

**Cell 7 — EXPERIMENT_LOG append**

---

## 02/03_ablation_study.ipynb  ← H1 PRIMARY TEST  ← Paper Figure 2

**Hypothesis H1:** System-state features reduce RMSE by ≥ 30% relative
to the patient-only baseline (group C). Ablation confirms the source of
the gain. The absolute RMSE threshold is set adaptively: if patient-only
RMSE = X, then full-feature RMSE must be ≤ 0.7×X.

> **Note:** Because MnemosyneGenerator encodes system-state penalties directly
> in `_mc_survival()`, a large RMSE reduction is expected by construction.
> The value of H1 is confirming the *magnitude* of system-state contribution
> and establishing the ablation curve shape, not "discovering" that system
> state matters. Frame findings accordingly.

This notebook produces the central evidence for H1 and Paper Figure 2.

### Cell sequence

**Cell 1 — Hypothesis H1** (quote from RESEARCH_NOTES.md)

**Cell 2 — Define feature groups**
```python
FEATURE_GROUPS = {
    "A: Vitals only":          ["heart_rate","systolic_bp","respiratory_rate","gcs","spo2"],
    "B: + Injury profile":     ["A features"] + ["mechanism","primary_region",
                                "severity","is_polytrauma","is_surgical_region"],
    "C: + PFC (patient full)": ["B features"] + ["pfc_duration_mins","triage_category"],
    "D: + System state (full)":["C features"] + SYSTEM_FEATURES,
}
```

**Cell 3 — Train and evaluate each group**
Same model architecture (resize input layer), same hyperparameters, same split.
Record val RMSE and test RMSE for each group.

**Cell 4 — Figure 2: Ablation bar chart**
```python
# Four bars: A, B, C, D
# Annotate: patient-only boundary (between C and D)
# Annotate: TRISS benchmark as horizontal dashed line
# Save: FIGURE_DIR / "fig2_ablation_rmse.png"  (FIGURE_DIR defined via ROOT convention)
```

Clean, publication-ready. No title (caption in paper). Axis labels in full.

**Cell 5 — Statistical significance**
Welch's t-test: is D vs C improvement statistically significant?
```python
from scipy import stats
t_stat, p_val = stats.ttest_ind(residuals_C, residuals_D, equal_var=False)
print(f"D vs C: t={t_stat:.3f}, p={p_val:.4f}")
```

**Cell 6 — Finding**
```
Finding: RMSE — A=[x.xx], B=[x.xx], C=[x.xx], D=[x.xx].
System-state contribution: [x.xx] RMSE reduction (C→D), p=[x.xxx].
H1 [confirmed/rejected].
[Optional: note any unexpected result — e.g. B→C jump larger than expected]
Note: System-state contribution reflects encoded penalties in _mc_survival().
The ablation demonstrates the modelling framework, not a clinical discovery.
Validation against empirical data required before operational interpretation.
```

**Cell 7 — EXPERIMENT_LOG append**
```
---
Experiment: 2 — Ablation (H1)
Notebook: 02_experiment_surrogate/03_ablation_study.ipynb
Hypothesis: H1 — system-state features reduce RMSE by ≥ 30% vs patient-only
Date: [date]
Result: A=[x.xx] B=[x.xx] C=[x.xx] D=[x.xx], p(D vs C)=[x.xxx]
Finding: H1 [confirmed/rejected]. System contribution = [x.xx] RMSE.
Next: Proceed to 02/04 uncertainty.
---
```

---

## 02/04_uncertainty_quantification.ipynb  ← H3  ← Paper Figure 4

**Hypothesis H3:** MC Dropout uncertainty intervals are statistically wider
under MASCAL conditions (concurrent_casualties ≥ 3) than under normal
conditions (concurrent_casualties < 3).

### Cell sequence

**Cell 1 — Hypothesis H3**

**Cell 2 — Reload full model with dropout retained at inference**
```python
# Modify SurvivalMLP: ensure Dropout layers are active at inference
# Enable with model.train() OR implement explicit mc_dropout flag
#
# WARNING: model.train() also sets BatchNorm to training mode.
# SurvivalMLP uses Dropout but not BatchNorm, so model.train() is safe here.
# If the architecture is later changed to include BatchNorm, use a custom
# flag instead:
#   for m in model.modules():
#       if isinstance(m, nn.Dropout): m.train()

def mc_predict(model, x, n_samples=100):
    model.train()   # keep dropout active (safe: no BatchNorm in SurvivalMLP)
    with torch.no_grad():
        preds = torch.stack([model(x) for _ in range(n_samples)])
    return preds.mean(0), preds.std(0)   # mean and std per sample
```

**Cell 3 — Compute intervals on test set**
For each test example: mean prediction + 95% CI (mean ± 1.96×std).

**Cell 4 — Segment by MASCAL**
```python
mascal_mask  = df_te["concurrent_casualties"] >= 3
normal_mask  = df_te["concurrent_casualties"] <  3
ci_mascal    = ci_widths[mascal_mask]
ci_normal    = ci_widths[normal_mask]
```

**Cell 5 — Statistical test**
Welch's t-test: `ci_mascal` vs `ci_normal` interval widths.

**Cell 6 — Figure 4: Scatter plot**
```python
# x-axis: predicted survival probability
# y-axis: actual survival probability
# Colour: CI width (viridis — blue=narrow, yellow=wide)
# Annotate: "MASCAL cases" cluster
# Save: FIGURE_DIR / "fig4_uncertainty_intervals.png"
```

**Cell 7 — Clinical interpretation cell**
Narrative: "A prediction of 0.62 ± 0.18 means..."
vs "0.62 ± 0.03 means..." — what this difference means
for a medical planner at a Role 2 facility.

**Cell 8 — Finding**
```
Finding: MASCAL CI width = [x.xx ± x.xx] vs normal [x.xx ± x.xx].
Welch's t: t=[x.xx], p=[x.xxx].
H3 [confirmed/rejected].
Note: CI widths reflect noise variance in _mc_survival() interacting with
dropout stochasticity. Wider intervals under MASCAL are expected given
the concurrent_casualties penalty term, but the magnitude is informative.
```

**Cell 9 — EXPERIMENT_LOG append**

---

## 02/05_pfc_sensitivity_curve.ipynb  ← H2  ← Paper Figure 3

**Hypothesis H2:** PFC duration has the highest partial dependence magnitude
among all patient-side features.

### Cell sequence

**Cell 1 — Hypothesis H2**

**Cell 2 — Partial dependence function**
```python
def partial_dependence(model, X, feature_idx, grid_points=50):
    """
    Fix all features at median. Sweep feature_idx across its range.
    Return (grid_values, mean_predictions).
    """
    X_pd   = X.copy()
    grid   = np.linspace(X[:, feature_idx].min(),
                         X[:, feature_idx].max(), grid_points)
    preds  = []
    for val in grid:
        X_pd[:, feature_idx] = val
        with torch.no_grad():
            pred = model(torch.tensor(X_pd, dtype=torch.float32))
        preds.append(pred.mean().item())
    return grid, np.array(preds)
```

**Cell 3 — Compute PD for all patient features**
Run `partial_dependence` for each feature in group C (patient-side full).

**Cell 4 — Rank by magnitude**
`magnitude = max(preds) - min(preds)` per feature.
Table: feature name + magnitude, sorted descending.

**Cell 5 — Figure 3: PFC sensitivity curve**
```python
# x-axis: pfc_duration_mins (0 to 480)
# y-axis: P(survival) — predicted
# Add vertical lines:
#   x=60  — "Golden hour" (annotated)
#   x=240 — "PFC threshold" (annotated)
# Add shading: 0–60 "standard MEDEVAC window", 60–240 "PFC range", 240+ "extended PFC"
# Save: FIGURE_DIR / "fig3_pfc_sensitivity.png"
```

**Cell 6 — Finding**
```
Finding: PFC duration ranks [1st/2nd/...] by partial dependence magnitude ([x.xx]).
H2 [confirmed/rejected].
[x.xx]% survival probability reduction per hour of PFC delay (linear approximation).
Golden hour threshold visible at [survival probability at 60 min].
Note: PFC importance reflects DETERIORATION_RATE_PER_HOUR encoded in generator.
The ranking validates that the model recovers the intended clinical relationship.
```

**Cell 7 — EXPERIMENT_LOG append**
```
---
Experiment: 2 — PFC Sensitivity (H2)
Notebook: 02_experiment_surrogate/05_pfc_sensitivity_curve.ipynb
Hypothesis: H2 — PFC duration is dominant patient-side predictor
Date: [date]
Result: PFC ranked [x], magnitude=[x.xx]. Top feature: [name] ([x.xx])
Finding: H2 [confirmed/rejected].
Next: Update RESEARCH_NOTES.md abstract. Proceed to Dc or E.
---
```

---

## RESEARCH_NOTES.md Update

After 02/05, update the structured abstract in RESEARCH_NOTES.md:

```markdown
## Structured Abstract (complete after Db)

**Background:** Military survivability models score patients at a snapshot,
ignoring operational system state.

**Objective:** Develop a neural surrogate of FAER(MIL) that conditions
survival probability on the full evacuation system state over a 12-hour
horizon.

**Methods:** MnemosyneGenerator produces synthetic datasets from FAER(MIL)
clinical distributions across three operational contexts (HIGH_INTENSITY,
PEACEKEEPING, PEER_COMPETITOR_DRONE). A multilayer perceptron is trained
on [n=10,000] casualties with patient-state and system-state features.
Ablation study (H1), partial dependence analysis (H2), MC Dropout
uncertainty quantification (H3), and distribution shift experiment (H4)
test pre-registered hypotheses.

**Results:** System-state features reduced RMSE from [x.xx] to [x.xx]
(H1 [confirmed/rejected], p=[x.xxx]). PFC duration was the [xth] ranked
patient-side predictor by partial dependence (H2 [confirmed/rejected]).
Uncertainty intervals were [x.x]× wider under MASCAL conditions
(H3 [confirmed/rejected]). Distribution shift from PEACEKEEPING to
PEER_COMPETITOR_DRONE reduced classifier F1 by [x.xx] (H4 confirmed).

**Conclusions:** [complete after reviewing all findings]
```

---

## CLAUDE.md Update on Completion

Update CLAUDE.md State section and append Phase Notes:

```markdown
Active phase: BUILD_PHASE_Dc.md (or BUILD_PHASE_E.md if skipping Dc)
Last completed notebook: 02/05_pfc_sensitivity_curve.ipynb

### Phase Db — [date]
Completed: 5 notebooks (02/01–05)
Key metrics: H1 [confirmed/rejected] RMSE A=[x.xx] B=[x.xx] C=[x.xx] D=[x.xx] p=[x.xxx].
  H2 [confirmed/rejected] PFC ranked [x]. H3 [confirmed/rejected] MASCAL CI=[x.xx±x.xx].
Deviation from spec: [none / description]
Lesson learned: [one sentence]
Verified baseline for next phase: surrogate model trained, paper figures saved
```

Smoke tests: `pytest -m smoke` passes with Db tests added.

---

## Next Phase

On completion, proceed to:
`BUILD_PHASE_Dc.md` — Experiment 3: LSTM System State Forecaster

Note: Dc is deferrable. If time is constrained, Dc can be skipped and
Phase E started. Dc provides the Future Work section of the paper —
it strengthens the submission but does not gate it.
