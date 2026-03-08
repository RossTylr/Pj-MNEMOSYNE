# BUILD_PHASE_Da.md — Experiment 1: Baseline Establishment

## Phase Goal

Validate the synthetic data substrate and establish baselines.
Four notebooks. Primary output: H4 finding + confirmed data quality.

This phase answers: "Is the MnemosyneGenerator producing clinically
plausible data that a model can learn?" If the answer is no, stop and
fix D0 before proceeding to Db.

## Prerequisites

- [ ] Phase D0 complete — validation script passes all checks
- [ ] Phase B 01/02_perceptron.ipynb complete (MLP classification pattern)
- [ ] Phase B 03/05_regularisation.ipynb complete (class imbalance handling)

## Completion Gate

- [ ] H4 finding recorded in EXPERIMENT_LOG.md
- [ ] ΔF1 (peacekeeping → drone) measured and logged
- [ ] Data validation notebook confirms plausible distributions
- [ ] Baseline F1 established for Db to beat
- [ ] `pytest -m smoke` passes with new Da tests added

## Smoke Test Additions

Add to `tests/test_smoke.py` when Phase Da completes:
```python
@pytest.mark.smoke
def test_baseline_f1_above_majority():
    """Phase Da: trained MLP beats majority class baseline."""
    # Load saved baseline results, assert macro F1 > majority F1

@pytest.mark.smoke
def test_distribution_shift_degrades():
    """Phase Da: H4 — shifted F1 < in-distribution F1."""
    # Load saved H4 results, assert shifted_f1 < source_f1
```

---

## 01/01_data_validation.ipynb

**Hypothesis:** MnemosyneGenerator produces a clinically plausible injury
distribution — triage ratios, vitals separation, and mechanism shift are
consistent with FAER(MIL) assumptions.

Use EXPERIMENT_TEMPLATE, not NOTEBOOK_TEMPLATE.

### Cell sequence

**Cell 1 — Hypothesis**
```
Hypothesis: MnemosyneGenerator produces clinically plausible distributions
across all three operational contexts. Specifically:
- T1:T3 ratio ≈ 1:6 in HIGH_INTENSITY
- Vitals are separated by triage category (T1 HR > T3 HR)
- PEER_COMPETITOR_DRONE has ≥2× the BLAST proportion of PEACEKEEPING
```

**Cell 2 — Generate data**
```python
from data.mnemosyne_synthetic.generator import MnemosyneGenerator
gen_hi = MnemosyneGenerator("HIGH_INTENSITY",        seed=42)
gen_pk = MnemosyneGenerator("PEACEKEEPING",           seed=42)
gen_dr = MnemosyneGenerator("PEER_COMPETITOR_DRONE",  seed=42)

df_hi = gen_hi.generate_triage_dataset(n=10_000)
df_pk = gen_pk.generate_triage_dataset(n=10_000)
df_dr = gen_dr.generate_triage_dataset(n=10_000)
```

**Cell 3 — Triage distribution plot**
Three-panel bar chart: triage proportions across all three contexts.
Annotate T1:T3 ratio for HIGH_INTENSITY.

**Cell 4 — Vitals separation**
Box plots: HR, SBP, GCS by triage category (HIGH_INTENSITY only).
Should show clear ordering: T1_SURGICAL worst vitals, T3 best.

**Cell 5 — Mechanism shift**
Grouped bar chart: mechanism proportions for PEACEKEEPING vs PEER_COMPETITOR_DRONE.
BLAST bar should be visibly higher for DRONE context.

**Cell 6 — Severity distribution**
Histogram: severity by context. PEER_COMPETITOR_DRONE should skew higher.

**Cell 7 — Clinical plausibility checks**
```python
# Programmatic checks — all must pass
assert df_hi["triage_category"].value_counts(normalize=True)["T3"] > 0.40
assert df_hi["triage_category"].value_counts(normalize=True)["T1_SURGICAL"] < 0.15
assert df_hi[df_hi.triage_category=="T1_SURGICAL"]["heart_rate"].mean() > \
       df_hi[df_hi.triage_category=="T3"]["heart_rate"].mean()
blast_hi = df_hi["mechanism"].eq(0).mean()
blast_dr = df_dr["mechanism"].eq(0).mean()
assert blast_dr > blast_hi * 1.5, f"BLAST shift insufficient: {blast_hi:.2f}→{blast_dr:.2f}"
print("All plausibility checks passed.")
```

**Cell 8 — Finding**
```
Finding: [confirm/reject hypothesis].
T1:T3 ratio = [x:x]. Vitals separation confirmed/not confirmed.
BLAST shift: PK=[x.xx] → DRONE=[x.xx] (ratio=[x.x]×).
Data substrate [is/is not] suitable for research track.
Note: This notebook validates generator outputs against its own design
parameters. Plausibility here confirms correct implementation, not
clinical accuracy — see RESEARCH_NOTES.md Threats to Validity.
```

**Cell 9 — EXPERIMENT_LOG append**
Manually append to `EXPERIMENT_LOG.md`:
```
---
Experiment: 1 — Data Validation
Notebook: 01_experiment_baseline/01_data_validation.ipynb
Hypothesis: Generator produces plausible clinical distributions
Date: [date]
Result: T1:T3=[x:x], BLAST shift=[x.x]×, vitals separation=[confirmed/not]
Finding: [one sentence]
Next: Proceed to 01/02 if confirmed. Fix D0 generator if not.
---
```

---

## 01/02_baseline_classifier.ipynb

**Hypothesis:** A 2-layer MLP achieves macro F1 > 0.80 on balanced
HIGH_INTENSITY triage classification, confirming the data is learnable.

### Cell sequence

**Cell 1 — Hypothesis** (state H before code)

**Cell 2 — Data prep**
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = gen_hi.generate_triage_dataset(n=10_000)

FEATURES = ["heart_rate","systolic_bp","respiratory_rate","gcs","spo2",
            "mechanism","primary_region","severity","is_polytrauma",
            "is_surgical_region","pfc_hours_elapsed"]
TARGET   = "triage_category"

le = LabelEncoder().fit(df[TARGET])
X  = StandardScaler().fit_transform(df[FEATURES])
y  = le.transform(df[TARGET])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

**Cell 3 — Majority class baseline**
```python
from sklearn.metrics import f1_score
majority = np.bincount(y_tr).argmax()
y_pred_majority = np.full_like(y_te, majority)
f1_majority = f1_score(y_te, y_pred_majority, average="macro")
print(f"Majority class baseline F1: {f1_majority:.3f}")
```

**Cell 4 — Model definition (PyTorch)**
```python
import torch, torch.nn as nn

class TriageMLP(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )
    def forward(self, x): return self.net(x)
```

**Cell 5 — Training loop**
50 epochs, Adam lr=1e-3, CrossEntropyLoss.
Log: train loss, val macro F1 per epoch.

**Cell 6 — Evaluation**
Confusion matrix heatmap + per-class precision/recall/F1 table.

**Cell 7 — Finding**
```
Finding: Baseline MLP macro F1 = [x.xx] (majority baseline = [x.xx]).
Hypothesis [confirmed/rejected].
Weakest class: [class] F1=[x.xx] — [clinical interpretation].
Note: Class separability reflects vitals distributions encoded in VITALS_BY_TRIAGE.
```

**Cell 8 — EXPERIMENT_LOG append**

---

## 01/03_class_imbalance.ipynb

**Hypothesis:** Using real T1:T3 ratios reduces macro F1 by > 0.10 vs
balanced data. Class weighting recovers > 0.07 of that loss.

### Cell sequence

**Cell 1 — Hypothesis**

**Cell 2 — Generate imbalanced dataset**
```python
# Real prevalence: T3 ~45%, T2 ~30%, T1_M ~10%, T1_S ~8%, T4 ~7%
# Sample with these proportions rather than uniform
REAL_PROBS = {"T3":0.45, "T2":0.30, "T1_MEDICAL":0.10,
              "T1_SURGICAL":0.08, "T4":0.07}
```

**Cell 3 — Train 4 variants and compare**

| Variant | Setup |
|---------|-------|
| A | Balanced sampling (01/02 setup) |
| B | Real prevalence, no correction |
| C | Real prevalence + class weights |
| D | Real prevalence + SMOTE |

Record macro F1 for each. Same model architecture throughout.

**Cell 4 — F1 comparison bar chart**
Four bars, one per variant. Annotate ΔF1 from A to B (imbalance penalty)
and ΔF1 from B to C (class weighting recovery).

**Cell 5 — Finding**
```
Finding: Imbalance penalty ΔF1 = [x.xx] (A→B).
Class weighting recovers ΔF1 = [x.xx] (B→C).
Hypothesis [confirmed/rejected].
Recommendation for Db: use [class weighting / SMOTE / both].
Note: Imbalance effects are a function of the triage threshold boundaries
in _assign_triage() and the Beta severity parameters.
```

**Cell 6 — EXPERIMENT_LOG append**

---

## 01/04_distribution_shift.ipynb  ← H4

**Hypothesis H4:** A classifier trained on PEACEKEEPING data degrades
measurably on PEER_COMPETITOR_DRONE data, with ΔF1 > 0.15, due to
mechanism distribution shift.

This notebook directly tests H4 and feeds paper Section 3 (Data & Methods).

### Cell sequence

**Cell 1 — Hypothesis H4** (quote H4 from RESEARCH_NOTES.md exactly)

**Cell 2 — Generate source and target data**
```python
from data.mnemosyne_synthetic.distribution_shift import generate_shifted_pair

df_source, df_target = generate_shifted_pair(
    n=10_000,
    source_context="PEACEKEEPING",
    target_context="PEER_COMPETITOR_DRONE",
    seed=42,
)
# Hold out 20% of target as test set (no training on target)
```

**Cell 3 — Train on source only**
Train TriageMLP on PEACEKEEPING train set only.

**Cell 4 — Evaluate on source test set (in-distribution)**
Record macro F1 — this is the upper bound.

**Cell 5 — Evaluate on full target set (shifted distribution)**
Record macro F1 — this is the shifted performance.

**Cell 6 — Per-class degradation**
Table: F1 per triage class, source vs target.
Expect T1_SURGICAL to degrade most (blast injuries dominate in drone context).

**Cell 7 — Mechanism shift visualisation**
Side-by-side: PEACEKEEPING vs DRONE mechanism distribution.
Annotate: "Model trained here" / "Evaluated here".

**Cell 8 — Domain adaptation (partial)**
Fine-tune on 500 DRONE examples (5% of target).
Record recovered F1 — shows adaptation is possible with small target sample.

**Cell 9 — Finding**
```
Finding: In-distribution F1 = [x.xx]. Shifted F1 = [x.xx]. ΔF1 = [x.xx].
H4 [confirmed/rejected] (threshold was 0.15).
Most degraded class: [class] — [interpretation re: mechanism shift].
Domain adaptation on 500 examples recovers ΔF1 = [x.xx].
Note: Distribution shift reflects CONTEXT_MECHANISM_PROBS differences encoded
in MnemosyneGenerator. ΔF1 validates the modelling concern but the magnitude
depends on assumed mechanism distributions, not empirical operational data.
```

**Cell 10 — EXPERIMENT_LOG append**
```
---
Experiment: 1 — Distribution Shift (H4)
Notebook: 01_experiment_baseline/04_distribution_shift.ipynb
Hypothesis: H4 — peacekeeping model degrades on drone-warfare data
Date: [date]
Result: in-dist F1=[x.xx], shifted F1=[x.xx], ΔF1=[x.xx]
Finding: H4 [confirmed/rejected]. ΔF1=[x.xx]. Driver: [mechanism].
Next: Use ΔF1 finding in paper Section 3. Proceed to Db.
---
```

---

## CLAUDE.md Update on Completion

Update CLAUDE.md State section and append Phase Notes:

```markdown
Active phase: BUILD_PHASE_Db.md
Last completed notebook: 01/04_distribution_shift.ipynb

### Phase Da — [date]
Completed: 4 notebooks (01/01–04)
Key metrics: Baseline F1=[x.xx]. H4 ΔF1=[x.xx] ([confirmed/rejected]).
Deviation from spec: [none / description]
Lesson learned: [one sentence]
Verified baseline for next phase: data substrate valid, baseline F1 established
```

Smoke tests: `pytest -m smoke` passes with Da tests added.

---

## Next Phase

On completion, proceed to:
`BUILD_PHASE_Db.md` — Experiment 2: System-Aware Survival Surrogate
