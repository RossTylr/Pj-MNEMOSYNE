# BUILD_PHASE_E.md — Paper Implementations + Diagnostics Toolkit

## Phase Goal

Two parallel tracks:
1. Section 05 — annotated paper implementations (4 notebooks)
2. Section 07 — shared diagnostics toolkit (3 modules)

Phase E is the capstone. By this point the learner has internals fluency
(Phase B), architectural literacy (Phase C), and a research output (Phase D).
Phase E builds the habit of reading primary literature and the tooling that
supports all other sections.

## Prerequisites

- [ ] Phase Db complete — research track primary experiments done (Dc optional)
- [ ] Phases B and C complete — all architecture notebooks functional
- [ ] 07_diagnostics_toolkit/ skeleton exists from Phase A

## Completion Criteria

Paper implementations must:
- Open with a "Paper Summary" cell: 3 sentences — problem, contribution, impact
- Implement the core idea from scratch before using any library version
- Annotate every non-obvious code block with its equation number from the paper
- End with: "What this paper enabled" — what architectures or techniques it unlocked
- Contain at least one silent correctness assertion [RAIE v3]

Diagnostics modules must:
- Work as standalone imports across all other notebooks
- Accept a PyTorch model and dataloader as inputs (no notebook-specific coupling)
- Produce publication-ready figures (clean axes, labels, no debugging artifacts)
- Include module-level doctests or unit tests in tests/ [RAIE v3]

---

## Section 05 — Paper Implementations

### 05/attention_is_all_you_need.ipynb

**Paper:** Vaswani et al., 2017 — "Attention Is All You Need"

**Paper summary cell:**
> Transformer architecture using self-attention eliminates recurrence and
> convolution entirely. Multi-head attention + positional encoding + residuals
> achieves state-of-the-art on translation. Enabled GPT, BERT, and essentially
> all modern large language models.

Content:
1. Paper summary cell (3 sentences: problem / contribution / impact)
2. Why this paper: the architecture that 02/05 is built on — close the loop
3. Annotated multi-head attention implementation:
   - Eq 1: `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V` — annotated
   - Eq 2: `MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O` — annotated
4. Positional encoding: Eq 3/4 — sinusoidal, annotated
5. Full encoder block: annotate each sublayer against paper Figure 1
6. Full decoder block: annotate masked attention
7. Compare to 02/05 implementation — where did we diverge and why?
8. "What this paper enabled" cell

### 05/batch_norm_ioffe_szegedy.ipynb

**Paper:** Ioffe & Szegedy, 2015 — "Batch Normalization: Accelerating Deep
Network Training by Reducing Internal Covariate Shift"

**Paper summary cell:**
> Internal covariate shift (changing layer input distributions during training)
> slows convergence. Normalising each mini-batch per layer stabilises training,
> allows higher learning rates, and acts as regularisation. Made training
> networks 10× faster and enabled much deeper architectures.

Content:
1. Paper summary cell
2. The internal covariate shift problem — why it matters
3. Algorithm 1 from paper — annotated line by line:
   - Mini-batch mean and variance
   - Normalise: x̂_i = (x_i - μ_B) / sqrt(σ²_B + ε)
   - Scale and shift: y_i = γ x̂_i + β
4. Training vs inference behaviour — running statistics
5. Verify against `nn.BatchNorm1d` — gradients match
6. From 03/03: confirm our scratch implementation matches this paper
7. Why layernorm instead for transformers — brief
8. "What this paper enabled" cell

### 05/resnet.ipynb

**Paper:** He et al., 2016 — "Deep Residual Learning for Image Recognition"

**Paper summary cell:**
> Deep networks degrade in training accuracy (not just overfitting) due to
> optimisation difficulty. Residual connections (skip connections) allow the
> network to learn residual functions, making 100+ layer networks trainable.
> Won ImageNet 2015, enabled modern deep vision architectures.

Content:
1. Paper summary cell
2. The degradation problem — show empirically: 20-layer vs 56-layer plain net,
   56-layer has higher training error (reproduce Fig 1 from paper on CIFAR-10)
3. Residual block: `F(x) + x` — why addition works (gradient highway)
4. Implement ResBlock from scratch — BasicBlock and Bottleneck variants
5. Build ResNet-20 for CIFAR-10 (small enough to train)
6. Train: compare to plain-20 (same depth, no skip connections)
7. Gradient flow: plot gradient norms — show skip connections maintain flow
8. "What this paper enabled" cell

### 05/dropout_srivastava.ipynb

**Paper:** Srivastava et al., 2014 — "Dropout: A Simple Way to Prevent
Neural Networks from Overfitting"

**Paper summary cell:**
> Neural networks overfit by co-adapting features — neurons rely on each other.
> Dropout randomly zeros activations during training, forcing independent feature
> learning. Acts as an ensemble of exponentially many networks at negligible cost.

Content:
1. Paper summary cell
2. Co-adaptation problem — demonstrate: overfit a small network, inspect weights
3. Implement dropout from scratch — train mode vs eval mode
4. Verify: at inference, weights scaled by (1-p) — test exact scaling
5. Ensemble interpretation: dropout as geometric mean of 2^n sub-networks
6. Practical guide from paper: p=0.5 for hidden layers, p=0.2 for input
7. MC Dropout connection: forward with dropout at inference = uncertainty estimate
   (this closes the loop to 02/04_uncertainty_quantification.ipynb)
8. "What this paper enabled" cell

---

## Section 07 — Diagnostics Toolkit

Three standalone modules. Each is imported by other notebooks — they are
infrastructure, not lessons. Build them to be robust.

### 07/loss_landscape_viz.py

**Purpose:** Visualise the loss landscape around a trained model's weights
in 2D by perturbing along two random directions.

**Interface:**

```python
def plot_loss_landscape(
    model: nn.Module,
    loss_fn: Callable,
    dataloader: DataLoader,
    resolution: int = 40,
    range_: float = 1.0,
    title: str = "Loss Landscape",
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plot 2D loss landscape by perturbing model weights along two
    random normalised directions (Gaussian, filter-normalised).

    Args:
        model: trained PyTorch model
        loss_fn: callable(outputs, targets) → scalar loss
        dataloader: evaluation data
        resolution: grid resolution (resolution × resolution evaluations)
        range_: perturbation magnitude
        title: plot title
        save_path: if provided, save figure here

    Returns:
        matplotlib Figure
    """
```

Implementation notes:
- Use Li et al. (2018) filter normalisation: scale perturbation directions
  by the Frobenius norm of each filter/layer to account for scale invariance
- Contour plot with 20 levels + surface plot as subplot
- Mark the trained model position (0,0) with a star

### 07/activation_histogram.py

**Purpose:** Plot activation distributions across all layers after a
forward pass. Identifies saturation, dead neurons, improper initialisation.

**Interface:**

```python
def plot_activation_histograms(
    model: nn.Module,
    dataloader: DataLoader,
    n_batches: int = 5,
    layers: list[str] | None = None,
    title: str = "Activation Distributions",
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Register forward hooks on all Linear/Conv layers, run n_batches,
    collect activations, plot histogram grid.

    Args:
        model: PyTorch model
        dataloader: data to run forward pass on
        n_batches: number of batches to collect
        layers: if None, auto-detect all Linear and Conv2d layers
        title: plot title
        save_path: if provided, save figure here

    Returns:
        matplotlib Figure with one subplot per layer
    """

def summarise_activations(
    model: nn.Module,
    dataloader: DataLoader,
    n_batches: int = 5,
) -> pd.DataFrame:
    """
    Return DataFrame: layer_name, mean, std, pct_dead (< 0.01), pct_saturated (> 0.99)
    """
```

### 07/gradient_norm_tracker.py

**Purpose:** Track and plot gradient norms across layers during training.
Identifies vanishing/exploding gradients in real time.

**Interface:**

```python
class GradientNormTracker:
    """
    Register backward hooks during training to track gradient norms.
    Call .plot() at any time to see the current gradient health.

    Usage:
        tracker = GradientNormTracker(model)
        for epoch in range(n_epochs):
            for x, y in dataloader:
                loss = ...
                loss.backward()
                tracker.step()   # call after backward, before optimizer.step()
                optimizer.step()
        tracker.plot()
    """

    def __init__(self, model: nn.Module) -> None: ...

    def step(self) -> None:
        """Record current gradient norms for all parameters."""

    def plot(
        self,
        last_n_steps: int | None = None,
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Plot gradient norm over training steps.
        One line per layer. Log scale on y-axis.
        Shade regions where mean norm < 1e-4 (vanishing) or > 10 (exploding).
        """

    def summary(self) -> pd.DataFrame:
        """
        Return DataFrame: layer_name, mean_norm, min_norm, max_norm,
        pct_vanishing (< 1e-4), pct_exploding (> 10)
        """
```

---

## Integration Test

After all Phase E work is complete, run an integration test:

1. Open `02/05_transformer.ipynb`
2. Import all three toolkit modules
3. Run `plot_activation_histograms` on the trained transformer
4. Run `GradientNormTracker` through one additional training epoch
5. Run `plot_loss_landscape` — this will be slow, use resolution=20
6. Confirm: all three produce clean figures with no errors

This integration test confirms the toolkit works on the most complex model
in the repository.

---

## CLAUDE.md Update on Completion

Update CLAUDE.md State section and append Phase Notes:

```markdown
Active phase: COMPLETE
Last completed notebook: 05/dropout_srivastava.ipynb

### Phase E — [date]
Completed: 4 paper notebooks + 3 diagnostics modules. Integration test passed.
Key metrics: [x] notebooks total. [x] hypotheses confirmed, [x] rejected.
Deviation from spec: [none / description]
Lesson learned: [one sentence]
Verified baseline: all notebooks run clean, diagnostics toolkit importable,
  integration test on transformer passes, pytest -m smoke all green.
```

## Smoke Test Additions

Add to `tests/test_smoke.py` when Phase E completes:
```python
@pytest.mark.smoke
def test_diagnostics_import():
    """Phase E: all three toolkit modules importable."""
    from diagnostics_toolkit import loss_landscape_viz
    from diagnostics_toolkit import activation_histogram
    from diagnostics_toolkit import gradient_norm_tracker

@pytest.mark.smoke
def test_full_smoke():
    """Phase E: final project-wide smoke test."""
    # Confirm: generator works, at least one model trains,
    # survival predictions in [0,1], diagnostics produce figures
```

## Final Gate

`pytest -m smoke` must pass with ALL accumulated tests from Phases A→E.
This is the project-level E2E check.

---

## Repository Complete

Pj-MNEMOSYNE is done when Phase E passes the integration test and
CLAUDE.md shows all five phases checked.

The trained surrogate from `06/02_experiment_surrogate/` is the deliverable
that feeds back into FAER(MIL). Everything else was preparation.
