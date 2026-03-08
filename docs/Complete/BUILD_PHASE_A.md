# BUILD_PHASE_A.md — Scaffold

## Phase Goal

Create the complete directory and file skeleton for Pj-MNEMOSYNE.
No notebook content yet — structure, templates, and config only.

## What You Are Building

A neural network learning repository with a first-class research track.
The repository teaches neural networks from scalar autograd to transformer,
using a structured experiment programme (Section 06) that produces a
publishable surrogate model for the FAER(MIL) military medical evacuation
simulation.

Full specification: `Pj-MNEMOSYNE-prd-v2.docx`
Plain language summary: see `PLAIN_GOAL.md`

---

## Step 0 — Claude Code Workflow Setup

Complete these three tasks before creating the scaffold. They configure
session persistence, system-prompt injection, and automated logging.

### 0.1 — Session hooks

Create `.claude/hooks/session-end.js`:

```javascript
#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

// 1. Write session summary file
const sessionsDir = path.join('.claude', 'sessions');
if (!fs.existsSync(sessionsDir)) fs.mkdirSync(sessionsDir, { recursive: true });

const today = new Date().toISOString().split('T')[0];
const sessionFile = path.join(sessionsDir, `${today}.md`);
const timestamp = new Date().toISOString();

const sessionTemplate = `## Session: ${timestamp}

### Last completed notebook
<!-- Fill: e.g. 01/03_mlp_numpy.ipynb Cell 7 — loss curve rendering -->

### What was verified working
<!-- Fill: what E2E smoke test / nbconvert confirmed -->

### Decisions made
<!-- Fill: any threshold or implementation decisions -->

### Next action
<!-- Fill: exact first step for next session -->

### Blockers / open questions
<!-- Fill: anything unresolved -->

---
`;
fs.appendFileSync(sessionFile, sessionTemplate);

// 2. Append stub entry to EXPERIMENT_LOG.md
const logFile = path.join('06_research_track', 'EXPERIMENT_LOG.md');
const logEntry = `\n<!-- SESSION END ${timestamp} — fill in completed notebook and findings above -->\n`;
if (fs.existsSync(logFile)) {
  fs.appendFileSync(logFile, logEntry);
}

console.log(`[session-end] Session summary: ${sessionFile}`);
console.log(`[session-end] EXPERIMENT_LOG stub appended.`);
```

Create `.claude/hooks/session-start.js`:

```javascript
#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const sessionsDir = path.join('.claude', 'sessions');
if (!fs.existsSync(sessionsDir)) {
  console.log('[session-start] No previous sessions found. Starting fresh.');
  process.exit(0);
}

const files = fs.readdirSync(sessionsDir)
  .filter(f => f.endsWith('.md'))
  .sort()
  .reverse()
  .slice(0, 3);

if (files.length === 0) {
  console.log('[session-start] No session history found.');
  process.exit(0);
}

console.log(`[session-start] Found ${files.length} recent session(s):\n`);
files.forEach(f => {
  const content = fs.readFileSync(path.join(sessionsDir, f), 'utf8');
  const nextAction = content.match(/### Next action\n([\s\S]*?)\n---/)?.[1]?.trim();
  console.log(`  ${f}: ${nextAction || '(no next action recorded)'}`);
});
console.log('\nLoad most recent session with: cat .claude/sessions/' + files[0]);
```

### 0.2 — Settings and AGENTS.md placement

Create `.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [
      {
        "type": "command",
        "command": "node .claude/hooks/session-end.js"
      }
    ],
    "SessionStart": [
      {
        "type": "command",
        "command": "node .claude/hooks/session-start.js"
      }
    ]
  }
}
```

Copy `AGENTS.md` to `.claude/AGENTS.md` (the root copy remains for
other tools; the `.claude/` copy is for system-prompt injection).

### 0.3 — Verification

Before proceeding to the scaffold:

- [ ] `node .claude/hooks/session-end.js` runs without error
- [ ] `.claude/sessions/YYYY-MM-DD.md` created with template
- [ ] `node .claude/hooks/session-start.js` runs without error
- [ ] `.claude/settings.json` exists with both hooks registered
- [ ] `.claude/AGENTS.md` present with domain invariants

---

## Exact Output Required

```
Pj-MNEMOSYNE/
│
├── .claude/
│   ├── settings.json                  ← hooks configuration
│   ├── AGENTS.md                      ← copy for system-prompt injection
│   ├── hooks/
│   │   ├── session-end.js             ← session persistence + log append
│   │   └── session-start.js           ← session history on startup
│   └── sessions/                      ← auto-generated session summaries
│       └── .gitkeep
│
├── CLAUDE.md                          ← living build doc (see template below)
├── AGENTS.md                          ← agent operating instructions (root copy)
├── AGENTS.md                          ← agent operating instructions
├── README.md                          ← one-paragraph project description
├── PLAIN_GOAL.md                      ← plain English goal statement
├── requirements.txt                   ← Python dependencies
├── .gitignore
├── pytest.ini                         ← test runner config
│
├── tests/
│   ├── conftest.py                    ← shared fixtures
│   └── test_smoke.py                 ← E2E smoke test (grows with project)
│
├── assets/
│   ├── LEARNING_MAP.md                ← navigation guide (see template below)
│   ├── NOTEBOOK_TEMPLATE.ipynb        ← standard learning template
│   ├── EXPERIMENT_TEMPLATE.ipynb      ← research track template
│   └── COMPANION_NOTES/
│       └── .gitkeep
│
├── 00_foundations/
│   ├── README.md
│   ├── 01_calculus_refresher.ipynb    ← empty, titled only
│   ├── 02_linear_algebra.ipynb
│   └── 03_probability_distributions.ipynb
│
├── 01_from_scratch/
│   ├── README.md
│   ├── 01_scalar_autograd.ipynb
│   ├── 02_perceptron.ipynb
│   ├── 03_mlp_numpy.ipynb
│   ├── 04_backprop_by_hand.ipynb
│   └── 05_train_loop_anatomy.ipynb
│
├── 02_architecture_zoo/
│   ├── README.md
│   ├── 01_mlp.ipynb
│   ├── 02_cnn.ipynb
│   ├── 03_rnn_lstm.ipynb
│   ├── 04_attention_mechanism.ipynb
│   ├── 05_transformer.ipynb
│   ├── 06_vae.ipynb
│   └── 07_diffusion_basics.ipynb
│
├── 03_training_science/
│   ├── README.md
│   ├── 01_activation_statistics.ipynb
│   ├── 02_gradient_flow.ipynb
│   ├── 03_batch_norm_layernorm.ipynb
│   ├── 04_optimisers.ipynb
│   ├── 05_regularisation.ipynb
│   └── 06_learning_rate_scheduling.ipynb
│
├── 05_paper_implementations/
│   ├── README.md
│   ├── attention_is_all_you_need.ipynb
│   ├── batch_norm_ioffe_szegedy.ipynb
│   ├── resnet.ipynb
│   └── dropout_srivastava.ipynb
│
├── 06_research_track/
│   ├── README.md
│   ├── RESEARCH_NOTES.md              ← structured abstract + hypotheses (see template below)
│   ├── EXPERIMENT_LOG.md              ← hypothesis → result → next step log
│   ├── PAPER_DRAFT/
│   │   ├── .gitkeep
│   │   └── STRUCTURE.md               ← paper section map
│   │
│   ├── data/
│   │   └── mnemosyne_synthetic/
│   │       ├── __init__.py
│   │       ├── generator.py           ← MnemosyneGenerator (constants hardcoded from FAER(MIL))
│   │       ├── noise_injection.py     ← skeleton
│   │       ├── distribution_shift.py  ← skeleton
│   │       └── README.md
│   │
│   ├── 01_experiment_baseline/
│   │   ├── README.md
│   │   ├── 01_data_validation.ipynb
│   │   ├── 02_baseline_classifier.ipynb
│   │   ├── 03_class_imbalance.ipynb
│   │   └── 04_distribution_shift.ipynb
│   │
│   ├── 02_experiment_surrogate/
│   │   ├── README.md
│   │   ├── 01_patient_only_baseline.ipynb
│   │   ├── 02_system_state_features.ipynb
│   │   ├── 03_ablation_study.ipynb
│   │   ├── 04_uncertainty_quantification.ipynb
│   │   └── 05_pfc_sensitivity_curve.ipynb
│   │
│   └── 03_experiment_sequence/
│       ├── README.md
│       ├── 01_lstm_stress_forecast.ipynb
│       ├── 02_mlp_vs_lstm_comparison.ipynb
│       ├── 03_mascal_detection.ipynb
│       └── 04_attention_weights_viz.ipynb
│
└── 07_diagnostics_toolkit/
    ├── README.md
    ├── __init__.py
    ├── loss_landscape_viz.py          ← skeleton
    ├── activation_histogram.py        ← skeleton
    └── gradient_norm_tracker.py       ← skeleton
```

---

## Import Path Convention

All notebooks and tests use paths relative to the repo root. To enable this,
`tests/conftest.py` (defined in the Test Scaffold section below) adds the repo
root and key package directories to `sys.path`.

All notebooks should include an equivalent cell at the top:
```python
import sys
from pathlib import Path
ROOT = Path(".").resolve().parent  # adjust depth to match notebook location
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "06_research_track" / "data"))
    sys.path.insert(0, str(ROOT / "07_diagnostics_toolkit"))
```

**Canonical import forms (use these everywhere):**
```python
from mnemosyne_synthetic.generator import MnemosyneGenerator
from mnemosyne_synthetic.distribution_shift import generate_shifted_pair
from mnemosyne_synthetic.noise_injection import inject_noise
import loss_landscape_viz
import activation_histogram
import gradient_norm_tracker
```

Do not use `from data.mnemosyne_synthetic...` or `from generator import...`
or `from diagnostics_toolkit import...`. The conftest.py sys.path additions
make the canonical forms work from any location.

---

## File Path Convention

All file I/O in notebooks uses paths relative to the repo root via a `ROOT`
variable. Artifacts that are shared between notebooks go in a named location:

```python
ROOT = Path(".").resolve().parent.parent  # from 06_research_track/02_experiment_surrogate/
SPLIT_PATH = ROOT / "06_research_track" / "02_experiment_surrogate" / "split_indices.npy"
FIGURE_DIR = ROOT / "06_research_track" / "PAPER_DRAFT" / "figures"
MODEL_DIR  = ROOT / "06_research_track" / "03_experiment_sequence" / "saved_models"
```

---

## Section README Files

Each section directory contains a `README.md` with a one-paragraph description
and a table of its notebooks. Use this template:

```markdown
# [Section Number] — [Section Name]

[One paragraph: what this section covers and what the learner can do after.]

| Notebook | Title | Status |
|----------|-------|--------|
| 01 | [Title] | [ ] |
| 02 | [Title] | [ ] |
...
```

---

## File Content Specifications

### CLAUDE.md and AGENTS.md

The CLAUDE.md and AGENTS.md delivered with the build plan (in repo root
alongside this file) are the canonical versions. Copy both verbatim into
the scaffold. Do not modify their content during Phase A.

After copying, tick the Phase A checkbox in CLAUDE.md and set Active phase
to BUILD_PHASE_A.md.

### README.md

```markdown
# Pj-MNEMOSYNE

Neural network learning repository — scalar autograd to transformer,
with a research track producing a publishable surrogate model for
military medical evacuation simulation.

**Learning arc:** 6 sections, ~38 notebooks, building from first principles.
**Research output:** System-aware survivability probability model — 
P(survival | patient_state, system_state, t+12h) — targeting JDMS / MORS.

See `assets/LEARNING_MAP.md` for navigation.
See `06_research_track/RESEARCH_NOTES.md` for research framing.
```

### PLAIN_GOAL.md

```markdown
# What Is Pj-MNEMOSYNE?

A notebook-first learning repository that teaches neural networks from
first principles — and uses that knowledge to build something real.

**The learning track** goes from scalar calculus to transformers. Every
architecture is built from scratch in NumPy before using PyTorch. Every
notebook ends with diagnostic probing: activation stats, gradient health,
weight distributions. The goal is internals fluency — being able to debug
a failing training run from first principles, not just call library functions.

**The research track** produces a publishable model:
P(survival | patient_state, system_state, t+12h) — a survival probability
that depends not just on how badly someone is injured, but on whether the
surgical team is overwhelmed, how far the helicopter has to fly, and whether
the supply route is contested. This is a departure from existing military
trauma scores (TRISS, NISS) which treat each patient in isolation.

The research output targets the Journal of Defence Modelling and Simulation
(JDMS) or Military Operations Research Society (MORS).
```

### .gitignore

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
dist/
build/
*.egg

# Jupyter
.ipynb_checkpoints/

# Environment
.env
.venv/
venv/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Claude Code sessions (local state, not committed)
.claude/sessions/

# Data / outputs
*.pt
*.pth
split_indices.npy
```

### assets/LEARNING_MAP.md

```markdown
# Pj-MNEMOSYNE — Learning Map

## You Are Here
Update this section as you progress.
Current position: [ ]

## Learning Sections (sequential)

| Section | Name | Notebooks | Status |
|---------|------|-----------|--------|
| 00 | Foundations | 3 | [ ] |
| 01 | From Scratch | 5 | [ ] |
| 02 | Architecture Zoo | 7 | [ ] |
| 03 | Training Science | 6 | [ ] |
| 05 | Paper Implementations | 4 | [ ] |
| 07 | Diagnostics Toolkit | tools | [ ] |

> **Note:** Section 04 (Applied Domains) is deferred to v1.1.

## Research Track (parallel, not sequential)

| Experiment | Role | Hypotheses | Status |
|------------|------|------------|--------|
| 01 Baseline | Data validation + H4 | H4 | [ ] |
| 02 Surrogate | PRIMARY CONTRIBUTION | H1, H2, H3 | [ ] |
| 03 Sequence | Extension / future work | H5 | [ ] |

## Prerequisites by Section

- **00** → basic Python, high school calculus
- **01** → 00 complete
- **02** → 01 complete
- **03** → 02 complete (can run parallel with 02 from notebook 3 onward)
- **06** → 01 complete, 03 in progress
```

### 06_research_track/RESEARCH_NOTES.md

```markdown
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
```

### 06_research_track/EXPERIMENT_LOG.md

```markdown
# Experiment Log — Pj-MNEMOSYNE Research Track

Format per entry:
---
**Experiment:** [number and name]
**Notebook:** [filename]
**Hypothesis:** [H-number and statement]
**Date:** [date]
**Result:** [metric / finding]
**Finding:** [one sentence answer to hypothesis]
**Next:** [what this opens or requires]
---

<!-- Entries appended as experiments complete -->
```

### 06_research_track/PAPER_DRAFT/STRUCTURE.md

```markdown
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
```

---

## Template Specifications

### assets/NOTEBOOK_TEMPLATE.ipynb — Full Cell Skeleton

This template defines the canonical structure for all learning notebooks
(Sections 00–03, 05). Every notebook follows this structure exactly.

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["# [Section] — [Notebook Title]\n", "\n", "**Status:** Not started\n", "\n", "**Learning outcome:** [One sentence — what the learner can do after this notebook]\n"]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## Why This Matters\n", "\n", "[1–2 paragraphs: what problem this solves, why you need to understand it]\n"]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## Theory Sketch\n", "\n", "[Mathematical motivation — equations, diagrams, intuition. No code.]\n"]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## From-Scratch Implementation (NumPy)\n"]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": ["# NumPy implementation goes here\n"],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## PyTorch Rewrite\n"]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": ["# PyTorch implementation goes here\n"],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## Training Run\n"]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": ["# Training with logged metrics goes here\n"],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## Internal Probing\n", "\n", "[Activation stats, gradient norms, weight distributions — at least one diagnostic visualisation]\n"]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": ["# Diagnostic visualisation goes here\n"],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## Structured Interpretation\n", "\n", "### Results\n", "[What happened — numbers, observations]\n", "\n", "### Findings\n", "[What this means — interpretation of results]\n", "\n", "### Implications\n", "[What this enables — how it connects to later notebooks]\n", "\n", "### Considerations\n", "[Caveats, failure modes, what to watch for]\n"]
    }
  ],
  "metadata": {
    "kernelspec": { "display_name": "Python 3", "language": "python", "name": "python3" },
    "language_info": { "name": "python", "version": "3.11.0" }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

### assets/EXPERIMENT_TEMPLATE.ipynb — Full Cell Skeleton

This template defines the canonical structure for all research track notebooks
(Section 06). Hypothesis is stated before any code. Finding is recorded after.

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["# [Experiment] — [Notebook Title]\n", "\n", "**Status:** Not started\n"]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## Hypothesis\n", "\n", "> **[H-number]:** [State hypothesis before any code]\n", "\n", "**Threshold:** [Quantitative threshold for confirmation/rejection]\n"]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## Setup & Data\n"]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": ["# Data generation / loading goes here\n"],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## Method\n"]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": ["# Model / analysis implementation goes here\n"],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## Results\n"]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": ["# Evaluation, metrics, visualisation goes here\n"],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## Finding\n", "\n", "**Finding:** [One-sentence answer to hypothesis]\n", "\n", "[Detailed interpretation — what this means for the research track]\n", "\n", "> **Validity note:** Results reflect relationships encoded in MnemosyneGenerator\n", "> synthetic distributions. Validation against empirical data (e.g. JTTR) is a\n", "> necessary next step before clinical interpretation.\n"]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## EXPERIMENT_LOG Entry\n", "\n", "Append to `EXPERIMENT_LOG.md`:\n", "\n", "```\n", "---\n", "Experiment: [number and name]\n", "Notebook: [filename]\n", "Hypothesis: [H-number and statement]\n", "Date: [date]\n", "Result: [metric / finding]\n", "Finding: [one sentence]\n", "Next: [what this opens or requires]\n", "---\n", "```\n"]
    }
  ],
  "metadata": {
    "kernelspec": { "display_name": "Python 3", "language": "python", "name": "python3" },
    "language_info": { "name": "python", "version": "3.11.0" }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

---

## Empty Notebook Specification

All notebooks created in this scaffold phase should be **titled but empty**,
following the structure of NOTEBOOK_TEMPLATE (learning notebooks) or
EXPERIMENT_TEMPLATE (research track notebooks) above, but with only
the title cell and (for research track) the hypothesis cell populated.

Minimal learning notebook:

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "source": ["# [Section] — [Notebook Title]\n", "\n", "**Status:** Not started\n"]
    }
  ],
  "metadata": {
    "kernelspec": { "display_name": "Python 3", "language": "python", "name": "python3" },
    "language_info": { "name": "python", "version": "3.11.0" }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

Minimal research track notebook:

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "source": ["# [Experiment] — [Notebook Title]\n", "\n", "**Status:** Not started\n"]
    },
    {
      "cell_type": "markdown",
      "source": ["## Hypothesis\n", "\n", "> [State hypothesis before any code — H-number]\n"]
    }
  ],
  "metadata": {
    "kernelspec": { "display_name": "Python 3", "language": "python", "name": "python3" },
    "language_info": { "name": "python", "version": "3.11.0" }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

---

## Generator Skeleton Specification

`06_research_track/data/mnemosyne_synthetic/generator.py` should contain:

```python
"""
MnemosyneGenerator — Synthetic data generator for Pj-MNEMOSYNE research track.

Produces ML-ready datasets from hardcoded clinical distributions derived
from FAER(MIL) sampling parameters. No live simulation required.

Clinical constants are hardcoded in this module. If FAER(MIL) distributions
change, update the constants here and re-run Phase D0 validation.

Usage:
    gen = MnemosyneGenerator(context="HIGH_INTENSITY", seed=42)
    df  = gen.generate_triage_dataset(n=10_000)
    df  = gen.generate_survival_dataset(n=10_000, n_mc_runs=500)
    arr = gen.generate_sequence_dataset(n_runs=1_000, duration_hours=12)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal

CONTEXTS = Literal["HIGH_INTENSITY", "PEACEKEEPING", "PEER_COMPETITOR_DRONE"]


class MnemosyneGenerator:
    """Synthetic data generator backed by FAER(MIL) clinical distributions."""

    def __init__(
        self,
        context: CONTEXTS = "HIGH_INTENSITY",
        seed: int = 42,
    ) -> None:
        self.context = context
        self.rng = np.random.default_rng(seed)
        # TODO Phase D0: initialise from hardcoded clinical constants
        raise NotImplementedError("Implement in Phase D0")

    def generate_triage_dataset(self, n: int = 10_000) -> pd.DataFrame:
        """
        Generate n casualties with features and triage label.
        Returns DataFrame with columns:
            heart_rate, systolic_bp, respiratory_rate, gcs, spo2,
            mechanism, primary_region, severity, is_polytrauma,
            is_surgical_region, pfc_hours_elapsed, context,
            triage_category (label)
        """
        raise NotImplementedError("Implement in Phase D0")

    def generate_survival_dataset(
        self, n: int = 10_000, n_mc_runs: int = 500
    ) -> pd.DataFrame:
        """
        Generate n casualties with patient + system state features.
        survival_probability computed as mean across n_mc_runs.
        Returns DataFrame — see PRD Section 4.3 Experiment 2 for schema.
        """
        raise NotImplementedError("Implement in Phase D0")

    def generate_sequence_dataset(
        self, n_runs: int = 1_000, duration_hours: int = 12
    ) -> np.ndarray:
        """
        Generate n_runs simulation sequences.
        Shape: (n_runs, timesteps, n_features)
        timesteps = duration_hours * 12  (5-min ticks)
        """
        raise NotImplementedError("Implement in Phase D0")
```

---

## Diagnostics Toolkit Skeleton

`07_diagnostics_toolkit/__init__.py` — empty file (modules imported directly).

All three .py files follow the same pattern: docstring + function/class
signature + `raise NotImplementedError`. Full implementations are in Phase E.

`07_diagnostics_toolkit/loss_landscape_viz.py`:
```python
"""Loss landscape visualisation. Full implementation in Phase E."""
raise NotImplementedError("Implement in Phase E")
```

`07_diagnostics_toolkit/activation_histogram.py`:
```python
"""Activation distribution histograms. Full implementation in Phase E."""
raise NotImplementedError("Implement in Phase E")
```

`07_diagnostics_toolkit/gradient_norm_tracker.py`:
```python
"""Gradient norm tracking during training. Full implementation in Phase E."""
raise NotImplementedError("Implement in Phase E")
```

`06_research_track/data/mnemosyne_synthetic/__init__.py` — empty file.

---

## requirements.txt

```
numpy>=1.26
pandas>=2.1
matplotlib>=3.8
seaborn>=0.13
torch>=2.2
torchvision>=0.17
jupyter>=1.0
ipykernel>=6.0
nbconvert>=7.0
scikit-learn>=1.4
scipy>=1.12
sympy>=1.12
graphviz>=0.20
tqdm>=4.66
pytest>=8.0
```

---

## Test Scaffold

Testing travels with the project from Phase A. The test runner is configured
at scaffold time so that every subsequent phase can add smoke tests
incrementally.

### pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
markers =
    smoke: E2E smoke tests (run after every iteration)
    slow: tests that take >10s (skip in quick runs)
```

### tests/conftest.py

```python
"""Shared fixtures for Pj-MNEMOSYNE tests."""
import sys
import pytest
from pathlib import Path

# Add repo root and key package directories to sys.path
# so canonical imports work from any test or notebook location.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "06_research_track" / "data"))
sys.path.insert(0, str(ROOT / "07_diagnostics_toolkit"))

# Fixtures added as phases complete — this file grows.
# Phase D0 will add: generator fixtures
# Phase Da will add: dataset fixtures with saved splits
```

### tests/test_smoke.py

```python
"""
E2E smoke test — grows with the project.
Run after every iteration: pytest -m smoke
Must pass before any notebook is considered done.
"""
import pytest

@pytest.mark.smoke
def test_project_structure():
    """Phase A: verify scaffold exists."""
    from pathlib import Path
    root = Path(__file__).parent.parent
    assert (root / "CLAUDE.md").exists()
    assert (root / "AGENTS.md").exists()
    assert (root / "assets" / "NOTEBOOK_TEMPLATE.ipynb").exists()
    assert (root / "assets" / "EXPERIMENT_TEMPLATE.ipynb").exists()
    assert (root / "06_research_track" / "RESEARCH_NOTES.md").exists()
    assert (root / "06_research_track" / "EXPERIMENT_LOG.md").exists()

# Phase D0 will add: test_generator_runs
# Phase Da will add: test_baseline_f1_above_majority
# Phase Db will add: test_survival_probabilities_in_range
```

---

## Validation Checklist

Before marking Phase A complete, verify:

Step 0 (Claude Code workflow):
- [ ] `.claude/hooks/session-end.js` runs without error
- [ ] `.claude/hooks/session-start.js` runs without error
- [ ] `.claude/settings.json` exists with both hooks registered
- [ ] `.claude/AGENTS.md` present (copy of root AGENTS.md)

Scaffold:
- [ ] All directories exist (including `.claude/sessions/`)
- [ ] All notebooks exist with title cell
- [ ] Research track notebooks have Hypothesis cell
- [ ] NOTEBOOK_TEMPLATE.ipynb matches full cell skeleton from this file
- [ ] EXPERIMENT_TEMPLATE.ipynb matches full cell skeleton from this file
- [ ] CLAUDE.md and AGENTS.md present at repo root
- [ ] CLAUDE.md Phase A checkbox ticked
- [ ] MnemosyneGenerator skeleton raises NotImplementedError (not silent pass)
- [ ] RESEARCH_NOTES.md has all 5 hypotheses with notebook paths
- [ ] EXPERIMENT_LOG.md exists (session-start instruction at top, no fabricated entries)
- [ ] PAPER_DRAFT/STRUCTURE.md section map is complete
- [ ] requirements.txt installs cleanly in fresh venv
- [ ] `pytest -m smoke` passes (test_project_structure green)

---

## Next Phase

On completion of scaffold, proceed to:
`BUILD_PHASE_B.md` — Spine notebooks (00_foundations → 03_training_science)

---

## Phase File Index

| File | Phase | Content |
|------|-------|---------|
| BUILD_PHASE_A.md | A | Scaffold (this file) |
| BUILD_PHASE_B.md | B | Spine — Sections 00, 01, 03 |
| BUILD_PHASE_C.md | C | Architecture Zoo — Section 02 |
| BUILD_PHASE_D0.md | D0 | MnemosyneGenerator data layer |
| BUILD_PHASE_Da.md | Da | Experiment 1: Baseline + H4 |
| BUILD_PHASE_Db.md | Db | Experiment 2: Surrogate (PRIMARY) — H1, H2, H3 |
| BUILD_PHASE_Dc.md | Dc | Experiment 3: Sequence — H5 (deferrable) |
| BUILD_PHASE_E.md | E | Paper implementations + diagnostics toolkit |
