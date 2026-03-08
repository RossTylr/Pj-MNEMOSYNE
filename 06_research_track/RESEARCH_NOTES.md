# Research Notes — Pj-MNEMOSYNE Research Track

## Contribution Claim

A system-aware survivability probability model for military medical
evacuation, conditioned on patient state, operational system state,
and a 12-hour projection window. Implemented as a neural surrogate
of FAER(MIL) Monte Carlo simulation outputs.

## Novelty Statement

Existing military survivability models (TRISS, NISS, KTS) score patients
at a snapshot — injuries and vitals at point of injury. This model treats
survivability as an emergent property of the patient-system interaction
over time. The same patient has different survival odds depending on R2
surgical saturation, transport ETA, route threat level, and PFC duration.

## Core Model

P(survival | patient_state, system_state, t+12h)

## Falsifiable Hypotheses

| ID | Hypothesis | Test | Notebook |
|----|------------|------|----------|
| H1 | System-state features reduce RMSE vs patient-only baseline | Ablation study | 02/03_ablation_study.ipynb |
| H2 | PFC duration is the dominant patient-side predictor | Partial dependence | 02/05_pfc_sensitivity_curve.ipynb |
| H3 | Uncertainty intervals widen under MASCAL conditions | MC Dropout vs load | 02/04_uncertainty_quantification.ipynb |
| H4 | Peacekeeping-trained model degrades on drone-warfare data | Distribution shift | 01/04_distribution_shift.ipynb |
| H5 | LSTM outperforms MLP on MASCAL stress forecasting | Architecture comparison | 03/02_mlp_vs_lstm_comparison.ipynb |

## Baseline Comparison

Static clinical baselines to benchmark against:
- TRISS (Trauma and Injury Severity Score)
- NISS (New Injury Severity Score)

## Target Venues (priority order)

1. Journal of Defence Modelling and Simulation (JDMS)
2. Military Operations Research Society (MORS)
3. NATO RTO technical report
4. PhD chapter / fellowship application appendix

## Structured Abstract (draft — update as experiments complete)

**Background:** Military survivability models score patients at a snapshot,
ignoring operational system state. **Objective:** Develop a neural surrogate
of FAER(MIL) that conditions survival probability on the full evacuation
system state over a 12-hour horizon. **Methods:** [complete after Exp 2]
**Results:** [complete after Exp 2] **Conclusions:** [complete after Exp 2]

## Post-Publication Next Step

Import trained surrogate back into FAER(MIL) as a fast-path probability
estimator. Replace expensive MC runs in real-time planning contexts.

## Threats to Validity

1. **Self-referential synthetic data.** MnemosyneGenerator encodes known
   clinical relationships (PFC deterioration rates, system-state penalties)
   in its `_mc_survival()` function. Models trained on this data will
   "discover" relationships that were explicitly programmed. All findings
   from the research track describe the modelling *framework* (conditioning
   survival on system state), not clinical *discoveries*. Validation against
   empirical data (e.g. JTTR, DMRTI registries) is a necessary next step.

2. **Single simulation substrate.** All synthetic data derives from one
   generator with one set of clinical constants. Generalisability to other
   simulation models or real-world data is not demonstrated.

3. **Mechanism distribution assumptions.** Context-specific mechanism
   probabilities (e.g. 55% BLAST in PEER_COMPETITOR_DRONE) are informed
   estimates, not empirical measurements. Distribution shift findings (H4)
   are valid within the assumed distributions but may not reflect real
   operational shifts.

4. **Simplified survival model.** The `_mc_survival()` function uses
   additive penalty terms, not a physiologically-based model. The survival
   probability is a useful proxy for modelling purposes but should not be
   interpreted as a clinical prediction.

5. **No explicit treatment intervention.** The model conditions survival
   on system capacity (R2 surgical queue, transport ETA) but does not
   represent whether a specific intervention (e.g. surgery, blood
   transfusion) actually occurs. A patient at a facility with available
   capacity receives the same capacity bonus regardless of whether
   treatment is administered. Future work should model the
   treatment-decision pathway explicitly.
