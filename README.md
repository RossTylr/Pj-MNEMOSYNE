# Pj-MNEMOSYNE

A notebook-first machine learning repository that teaches neural networks from
first principles — then uses that knowledge to build a publishable surrogate
model for military medical evacuation simulation.

## What this is

**The learning track** goes from scalar calculus to diffusion models across
38 notebooks. Every architecture is built from scratch in NumPy before moving
to PyTorch. Every notebook ends with diagnostic probing — activation statistics,
gradient health, weight distributions — because the goal is internals fluency,
not library familiarity.

**The research track** produces a system-aware survivability model:

> P(survival | patient_state, system_state, t+12h)

Existing military trauma scores (TRISS, NISS) treat each patient in isolation.
This model treats survivability as an emergent property of the patient-system
interaction over time — the same patient has different survival odds depending
on surgical saturation, transport ETA, route threat level, and pre-hospital
care duration. Implemented as a neural surrogate of FAER(MIL) Monte Carlo
simulation outputs.

Target venues: JDMS / MORS.

## Repository layout

```
00_foundations/              calculus, linear algebra, probability
01_from_scratch/             scalar autograd, perceptron, MLP, backprop, train loops
02_architecture_zoo/         MLP, CNN, RNN/LSTM, attention, transformer, VAE, diffusion
03_training_science/         activations, gradient flow, normalisation, optimisers, LR scheduling
05_paper_implementations/    Attention Is All You Need, BatchNorm, Dropout, ResNet
06_research_track/
├── data/mnemosyne_synthetic/   synthetic data generator (single source of truth)
├── 01_experiment_baseline/     data validation + distribution shift (H4)
├── 02_experiment_surrogate/    system-aware model — primary contribution (H1, H2, H3)
└── 03_experiment_sequence/     LSTM stress forecasting extension (H5)
07_diagnostics_toolkit/      reusable diagnostic utilities
assets/                      visualisations, templates, learning map
tests/                       pytest smoke tests
```

> Section 04 (Applied Domains) is deferred.

## Research hypotheses

| ID | Hypothesis | Test |
|----|------------|------|
| H1 | System-state features reduce RMSE vs patient-only baseline | Ablation study |
| H2 | PFC duration is the dominant patient-side predictor | Partial dependence sweep |
| H3 | Uncertainty intervals widen under MASCAL conditions | MC Dropout comparison |
| H4 | Peacekeeping-trained model degrades on drone-warfare data | Distribution shift |
| H5 | LSTM outperforms MLP on MASCAL stress forecasting | Architecture comparison |

Full research framing, novelty statement, and threats to validity are in
[`06_research_track/RESEARCH_NOTES.md`](06_research_track/RESEARCH_NOTES.md).

## Getting started

```bash
git clone https://github.com/RossTylr/Pj-MNEMOSYNE.git
cd Pj-MNEMOSYNE
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Datasets (MNIST, CIFAR-10, Tiny Shakespeare) are downloaded automatically by
the notebooks that use them. The research track uses a self-contained synthetic
generator — no external data dependencies.

Run the smoke tests:

```bash
pytest -m smoke
```

## Navigation

Notebooks are numbered and intended to be worked through in order within each
section. See [`assets/LEARNING_MAP.md`](assets/LEARNING_MAP.md) for the full
prerequisite graph.

The research track can run in parallel with the learning track once Section 01
is complete.

## Conventions

- `seed=42` throughout; train/val/test splits are saved and reloaded
- Single synthetic data source in `06_research_track/data/mnemosyne_synthetic/`
- Experiment results are append-only in [`06_research_track/EXPERIMENT_LOG.md`](06_research_track/EXPERIMENT_LOG.md)
- No fabricated results — every notebook executes cleanly end to end

## Requirements

Python 3.10+ with PyTorch, NumPy, pandas, scikit-learn, matplotlib, seaborn,
and Jupyter. Full list in [`requirements.txt`](requirements.txt).

## Licence

This repository is for academic and research purposes.
