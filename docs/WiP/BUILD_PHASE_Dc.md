# BUILD_PHASE_Dc.md — Experiment 3: LSTM System State Forecaster

## Phase Goal

Build the sequence prediction extension.
Four notebooks testing H5.
Primary output: H5 finding + MASCAL detection demonstration.

This phase provides the paper's Future Work / Discussion section.
It is deferrable — the paper can be submitted without Dc.
If proceeding, complete Db first.

## Prerequisites

- [ ] Phase Db complete — H1, H2, H3 logged
- [ ] Phase C 02/03_rnn_lstm.ipynb complete (LSTM architecture)
- [ ] Phase C 02/04_attention_mechanism.ipynb complete (attention weights)
- [ ] Phase D0 `generate_sequence_dataset()` validated

## Completion Gate

- [ ] H5 confirmed or rejected with F1 values logged
- [ ] MASCAL detection lead time measured and logged
- [ ] Attention weights notebook shows interpretable peak pattern
- [ ] All four EXPERIMENT_LOG entries appended
- [ ] `pytest -m smoke` passes with new Dc tests added

## Smoke Test Additions

Add to `tests/test_smoke.py` when Phase Dc completes:
```python
@pytest.mark.smoke
def test_lstm_predicts_four_classes():
    """Phase Dc: LSTM stress forecaster predicts all 4 stress levels."""
    # Load saved predictions, assert nunique >= 3 (GREEN/AMBER/RED at minimum)
```

---

## Architecture Note

All Dc notebooks use `StressLSTMAttention` (LSTM with a learned attention
layer) as the primary model. This avoids the complexity of retrofitting
attention onto a trained LSTM mid-experiment. The attention weights are
available from the start for interpretability analysis in 03/04.

```python
class StressLSTMAttention(nn.Module):
    def __init__(self, n_features=8, hidden_size=128, n_layers=2, n_classes=4):
        super().__init__()
        self.lstm    = nn.LSTM(n_features, hidden_size, n_layers,
                               batch_first=True, dropout=0.2)
        self.attn    = nn.Linear(hidden_size, 1)   # score each timestep
        self.head    = nn.Linear(hidden_size, n_classes)

    def forward(self, x, return_weights=False):
        lstm_out, _ = self.lstm(x)                # (B, T, H)
        scores      = self.attn(lstm_out)          # (B, T, 1)
        weights     = torch.softmax(scores, dim=1) # (B, T, 1)
        context     = (weights * lstm_out).sum(1)  # (B, H)
        out         = self.head(context)
        if return_weights:
            return out, weights.squeeze(-1)
        return out
```

---

## 03/01_lstm_stress_forecast.ipynb

**Hypothesis:** A 2-layer LSTM achieves RED state F1 > 0.75 on
60-minute-ahead system stress forecasting.

### Cell sequence

**Cell 1 — Hypothesis**

**Cell 2 — Load sequence data**
```python
from data.mnemosyne_synthetic.generator import MnemosyneGenerator

gen = MnemosyneGenerator("HIGH_INTENSITY", seed=42)
arr = gen.generate_sequence_dataset(n_runs=1_000, duration_hours=12)
# arr shape: (1000, 144, 8)

# Many-to-one: predict stress level at final tick from preceding sequence
# Target: system_stress_level (feature index 7) at the last timestep
X = arr[:, :-12, :]                    # first 132 timesteps as input
y = arr[:, -1, 7].astype(int)          # stress level at t=144 (final tick)

# Train/val/test: 700/150/150
```

**Cell 3 — Model (StressLSTMAttention from shared definition above)**
```python
# Use StressLSTMAttention defined in Architecture Note above.
# This is the same model used across all Dc notebooks.
model = StressLSTMAttention(n_features=8, hidden_size=128, n_layers=2, n_classes=4)
```

**Cell 4 — Training loop**
50 epochs, Adam lr=1e-3, CrossEntropyLoss.
Log: train loss, val F1 per RED class per epoch.

**Cell 5 — Evaluation**
Confusion matrix (4×4: GREEN/AMBER/RED/BLACK).
Per-class F1 table. Focus: RED class F1.

**Cell 6 — Sample sequence visualisation**
Pick 3 test sequences containing MASCAL injects.
Plot: true stress level vs predicted stress level over 144 timesteps.

**Cell 7 — Finding**
```
Finding: RED class F1 = [x.xx].
Hypothesis [confirmed/rejected] (threshold was 0.75).
Weakest class: [class] — [interpretation].
```

**Cell 8 — Save model for reuse in 03/04**
```python
MODEL_DIR = ROOT / "06_research_track" / "03_experiment_sequence" / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)
torch.save(model.state_dict(), MODEL_DIR / "stress_lstm_attention.pt")
```

**Cell 9 — EXPERIMENT_LOG append**

---

## 03/02_mlp_vs_lstm_comparison.ipynb  ← H5

**Hypothesis H5:** LSTM outperforms an MLP trained on the flattened
sequence on RED state F1, specifically during MASCAL inject scenarios
(ΔF1 ≥ 0.10 on MASCAL subset).

### Cell sequence

**Cell 1 — Hypothesis H5**

**Cell 2 — MLP baseline on flattened sequence**
```python
class StressMLP(nn.Module):
    def __init__(self, seq_len=143, n_features=8, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len * n_features, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),                  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        return self.net(x.reshape(x.size(0), -1))
```

Train same epochs, same hyperparameters as LSTM.

**Cell 3 — Compare: full test set**
Table: LSTM vs MLP — per-class F1, macro F1, accuracy.

**Cell 4 — Segment by MASCAL**
```python
# Identify which test sequences contain a MASCAL inject
# A sequence is "MASCAL" if max(new_casualties_this_tick) > 5
mascal_mask = arr_te[:, :, 6].max(axis=1) > 5
normal_mask = ~mascal_mask

# Compute RED F1 separately for each segment
```

**Cell 5 — Key comparison table**
```
| Metric              | LSTM  | MLP   | Δ     |
|---------------------|-------|-------|-------|
| Overall RED F1      | x.xx  | x.xx  | x.xx  |
| MASCAL RED F1       | x.xx  | x.xx  | x.xx  |
| Normal RED F1       | x.xx  | x.xx  | x.xx  |
```

**Cell 6 — Why flattening loses information**
Narrative cell explaining the temporal structure that MLP cannot see:
the inject-to-RED pattern spans 3–5 ticks and is lost when the
sequence is treated as an unordered bag of features.

**Cell 7 — Finding**
```
Finding: LSTM vs MLP — overall RED ΔF1 = [x.xx], MASCAL RED ΔF1 = [x.xx].
H5 [confirmed/rejected] (threshold was MASCAL ΔF1 ≥ 0.10).
[Interpretation: LSTM advantage is [concentrated in MASCAL scenarios / general]].
```

**Cell 8 — EXPERIMENT_LOG append**
```
---
Experiment: 3 — LSTM vs MLP (H5)
Notebook: 03_experiment_sequence/02_mlp_vs_lstm_comparison.ipynb
Hypothesis: H5 — LSTM outperforms MLP on MASCAL stress forecasting
Date: [date]
Result: Overall RED ΔF1=[x.xx], MASCAL RED ΔF1=[x.xx]
Finding: H5 [confirmed/rejected].
Next: Proceed to 03/03 MASCAL detection.
---
```

---

## 03/03_mascal_detection.ipynb

**Hypothesis:** The LSTM can detect an incoming MASCAL event at least
2 ticks (10 minutes) before it manifests as RED system stress.

This is an operational planning notebook — the finding directly answers
"what warning time does the model give a medical planner?"

### Cell sequence

**Cell 1 — Hypothesis**

**Cell 2 — Reframe as detection problem**
```python
# For each MASCAL test sequence:
# - Find the tick where true stress first becomes RED (t_red)
# - Find the earliest tick where model predicts RED (t_pred_red)
# - Lead time = t_red - t_pred_red (positive = early warning)

lead_times = []
false_positives = 0

for seq, true_stress in zip(X_te_mascal, y_te_mascal):
    t_red      = np.argmax(true_stress >= 2)   # first RED tick
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(seq).unsqueeze(0).float())
    pred_stress = preds.argmax(-1).numpy()
    pred_red_ticks = np.where(pred_stress >= 2)[0]

    if len(pred_red_ticks) > 0:
        t_pred_red = pred_red_ticks[0]
        lead_times.append(t_red - t_pred_red)
    else:
        lead_times.append(0)   # no early warning
```

**Cell 3 — Lead time distribution**
Histogram: lead times across all MASCAL test sequences.
Annotate: mean, median, proportion with ≥ 2 tick lead time.

**Cell 4 — False positive rate**
How often does the model predict RED on normal (non-MASCAL) sequences?
```python
fpr = (pred_stress_normal >= 2).mean()
print(f"False positive rate: {fpr:.3f}")
```

**Cell 5 — Operational value narrative**
Markdown cell: "With a mean lead time of [x] ticks (= [x] minutes), a
medical planner at Role 2 could [pre-position surgical team / request
additional transport / alert Role 3] before the MASCAL manifests."

**Cell 6 — Finding**
```
Finding: Mean lead time = [x.x] ticks ([x] minutes).
[x]% of MASCAL events detected ≥ 2 ticks early.
False positive rate = [x.x]%.
Hypothesis [confirmed/rejected].
```

**Cell 7 — EXPERIMENT_LOG append**

---

## 03/04_attention_weights_viz.ipynb

**Hypothesis:** Attention weights peak at the 2–3 ticks following a
MASCAL inject, confirming the model has learned the causal temporal
pattern and not a spurious correlation.

This notebook requires attention weights from the model. Since all Dc
notebooks use StressLSTMAttention from the start, no architecture change
is needed — simply call `model(x, return_weights=True)`.

### Cell sequence

**Cell 1 — Hypothesis**

**Cell 2 — Reload trained model from 03/01**
The StressLSTMAttention model already has attention. No retraining needed.
```python
# Load saved model from 03/01
MODEL_DIR = ROOT / "06_research_track" / "03_experiment_sequence" / "saved_models"
model.load_state_dict(torch.load(MODEL_DIR / "stress_lstm_attention.pt"))
```

**Cell 3 — Confirm F1 matches 03/01**
Quick sanity check — F1 should be identical to 03/01 evaluation.

**Cell 4 — Collect attention weights on MASCAL test set**
```python
model.eval()
all_weights = []
all_inject_ticks = []

for seq, inject_tick in zip(X_te_mascal, inject_ticks_mascal):
    with torch.no_grad():
        _, weights = model(torch.tensor(seq).unsqueeze(0).float(),
                           return_weights=True)
    all_weights.append(weights.squeeze().numpy())
    all_inject_ticks.append(inject_tick)
```

**Cell 5 — Align to inject tick and average**
```python
# Set inject_tick = t=0 for each sequence
# Compute mean attention weight at each relative tick (-10 to +20)
relative_weights = align_to_inject(all_weights, all_inject_ticks)
mean_weights     = relative_weights.mean(axis=0)
```

**Cell 6 — Plot: attention weight vs relative time**
```python
# x-axis: relative tick (−10 to +20 around inject)
# y-axis: mean attention weight
# Annotate: vertical line at t=0 "MASCAL inject"
# Shade: expected peak window t+2 to t+3
```

**Cell 7 — Qualitative: non-MASCAL sequences**
What do attention weights look like without a MASCAL inject?
Expect: distributed attention, no clear spike.

**Cell 8 — Finding**
```
Finding: Attention weight peaks at t+[x] relative to MASCAL inject.
Pattern [is/is not] consistent across [x]% of test sequences.
Non-MASCAL sequences show [distributed/uniform] attention.
Hypothesis [confirmed/rejected].
```

**Cell 9 — EXPERIMENT_LOG append**
```
---
Experiment: 3 — Attention Weights (final Dc notebook)
Notebook: 03_experiment_sequence/04_attention_weights_viz.ipynb
Hypothesis: Attention peaks at t+2 to t+3 post-inject
Date: [date]
Result: Peak at t+[x]. Consistent in [x]% of sequences.
Finding: [confirmed/rejected]. Model [has/has not] learned causal pattern.
Next: Phase E. Update paper Discussion section with Dc findings.
---
```

---

## Paper Discussion Update

After Dc, add to `PAPER_DRAFT/STRUCTURE.md` Section 5 (Discussion):

```markdown
## Section 5: Discussion — additions after Dc

### Future Work paragraph
The LSTM forecaster (Experiment 3) demonstrates that the system state
sequence contains predictable structure prior to MASCAL events.
H5 [was/was not] confirmed (MASCAL RED ΔF1 = [x.xx]).
The attention weight analysis confirms the model attends to the
[x] ticks following the casualty inject, which corresponds to the
physical propagation time of casualties through the Role 1 triage queue.
Future work should: (1) integrate the surrogate model and LSTM forecaster
into a unified planning tool within FAER(MIL), (2) validate against
real JTTR data, (3) extend to multi-echelon forecasting.
```

---

## CLAUDE.md Update on Completion

Update CLAUDE.md State section and append Phase Notes:

```markdown
Active phase: BUILD_PHASE_E.md
Last completed notebook: 03/04_attention_weights_viz.ipynb

### Phase Dc — [date]
Completed: 4 notebooks (03/01–04)
Key metrics: H5 [confirmed/rejected] MASCAL RED ΔF1=[x.xx].
  Lead time=[x.x] ticks ([x] min). FPR=[x.x]%. Attention peak at t+[x].
Deviation from spec: [none / description]
Lesson learned: [one sentence]
Verified baseline for next phase: all 5 hypotheses tested, research track complete
```

Smoke tests: `pytest -m smoke` passes with Dc tests added.

---

## Next Phase

On completion, proceed to:
`BUILD_PHASE_E.md` — Paper implementations + diagnostics toolkit

All five hypotheses now tested. Research track complete.
RESEARCH_NOTES.md Conclusions section can now be written.
