# Paper Structure — System-Aware Survivability Probability for Military MEDEVAC

## Target: JDMS / MORS

## Section Map

1. **Introduction**
   - Gap: static scoring vs system-aware probability
   - Contribution claim
   - Paper structure

2. **Background**
   - Military survivability models (TRISS, NISS, KTS) — limitations
   - Neural surrogates in simulation — prior work
   - FAER(MIL) simulation architecture — brief

3. **Data and Methods**
   - MnemosyneGenerator — synthetic data substrate
   - Feature schema: patient state + system state
   - H4 result: distribution shift as validity concern (from Experiment 1)
   - Model architecture and training protocol

4. **Results** ← PRIMARY
   - H1: ablation — system-state features vs patient-only
   - H2: PFC duration as dominant predictor
   - H3: uncertainty under MASCAL
   - Benchmark vs TRISS / NISS

5. **Discussion**
   - Clinical and operational interpretation
   - H5 and sequence forecasting as future work (Experiment 3)
   - Limitations: synthetic data, single simulation substrate

6. **Conclusion**

## Figures (planned)
- Fig 1: FAER(MIL) pathway diagram — system state features illustrated
- Fig 2: Ablation bar chart — RMSE by feature group
- Fig 3: PFC sensitivity curve — P(survival) vs PFC duration
- Fig 4: Uncertainty intervals — baseline vs MASCAL conditions
- Fig 5: Distribution shift degradation — peacekeeping vs drone-warfare F1
