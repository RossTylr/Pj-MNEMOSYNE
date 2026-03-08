# BUILD_PHASE_B.md — Spine Notebooks

## Phase Goal

Build the pedagogical spine: sections 00, 01, and 03.
These establish the internals fluency that everything else depends on.
Section 02 (Architecture Zoo) is started here but completed in Phase C.

## Prerequisites

- [ ] Phase A (Scaffold / BUILD_PHASE_A.md) complete and validated
- [ ] CLAUDE.md Phase A checkbox ticked
- [ ] requirements.txt installed in active Python environment

## Completion Criteria

Every notebook in this phase must:
- Follow `assets/NOTEBOOK_TEMPLATE.ipynb` structure exactly
- End with: Results → Findings → Implications → Considerations
- Run to completion without errors in a clean kernel (Restart & Run All)
- Contain at least one diagnostic visualisation (plot, histogram, or printed tensor stats)
- Contain at least one silent correctness assertion (e.g. loss < initial, accuracy > random) [RAIE v3]

## Smoke Test Additions

Add to `tests/test_smoke.py` when Phase B completes:
```python
@pytest.mark.smoke
def test_value_class_backward():
    """Phase B: autograd produces correct gradients."""
    # Import Value from 01_from_scratch, verify backward on simple graph
    # Gradient must match PyTorch to 1e-4

@pytest.mark.smoke
def test_numpy_mlp_trains():
    """Phase B: NumPy MLP loss decreases over 10 epochs."""
    # Train for 10 epochs, assert final_loss < initial_loss
```

---

## Section 00 — Foundations

**Goal:** Learner can differentiate a scalar function by hand and translate
that to code. Understands why matrix shapes are the primary source of bugs.

### 00/01_calculus_refresher.ipynb

**Learning outcome:** Implement forward and backward pass of a scalar
function manually. Verify against sympy symbolic derivative.

Content:
1. What a derivative means computationally — rate of change as a limit
2. Chain rule — walk through d/dx[f(g(x))] with a concrete example
3. Implement: `def numerical_gradient(f, x, h=1e-5)` — finite difference
4. Implement: `def analytic_gradient(x)` — by hand for f(x) = x³ + 2x²
5. Compare: show they agree to 4 decimal places
6. Extend to multivariate: partial derivatives, gradient vector
7. Visualisation: plot f(x), overlay tangent line at 3 points
8. Structured interpretation block

### 00/02_linear_algebra.ipynb

**Learning outcome:** Confident with matrix multiply, transpose, dot product
as NumPy operations. Can predict output shapes before running code.

Content:
1. Vectors and matrices as arrays — shape as first-class concern
2. Matrix multiply: implement from scratch with nested loops, then np.matmul
3. Shape rules: (m,n) @ (n,p) → (m,p) — drill with 5 examples
4. Transpose, inverse, determinant — geometric intuition
5. Eigenvalues — one worked example (2×2), geometric meaning
6. **Shape prediction exercise:** 10 matrix operations, predict shape before running
7. Visualisation: 2D linear transformation as arrow rotation/stretch
8. Structured interpretation block

### 00/03_probability_distributions.ipynb

**Learning outcome:** Sample from, visualise, and characterise the
distributions that appear throughout neural network training.

Content:
1. Why probability matters in ML — stochastic gradient descent is stochastic
2. Uniform, Gaussian, Bernoulli — sample and plot each
3. Beta distribution — key for FAER(MIL) injury severity sampling
4. Law of large numbers — empirical mean converges to expectation
5. Central Limit Theorem — demonstrate with repeated sampling
6. KL divergence — intuition and computation
7. Visualisation: distribution gallery — 6 distributions, 2×3 grid
8. Structured interpretation block

---

## Section 01 — From Scratch

**Goal:** Learner can implement scalar autograd and backpropagate manually
through any computation graph. Could write autograd from scratch independently.

This section follows the Karpathy micrograd arc closely — it is the most
important section in the repository. Do not rush it.

### 01/01_scalar_autograd.ipynb

**Learning outcome:** Build a scalar Value class with forward and backward
pass. Verify gradients match PyTorch on identical computation.

Content:
1. What autograd is: a computation graph where every node knows its gradient
2. Build `Value` class:
   - `__init__(self, data, _children=(), _op='')` — store data, grad=0
   - `__add__`, `__mul__`, `__pow__` — create child nodes, define `_backward`
   - `__neg__`, `__sub__`, `__truediv__`, `__radd__`, `__rmul__` — derived ops
   - `tanh` — implement as a method
   - `exp`, `relu` — implement as methods
3. `backward()` — topological sort, call `_backward` in reverse order
4. Build a two-neuron computation: `L = (a*b + c).tanh()`
5. Call `L.backward()`, print all gradients
6. Verify: replicate exact computation in PyTorch, assert gradients match
7. Visualisation: draw the computation graph (graphviz or matplotlib)
8. Structured interpretation block

### 01/02_perceptron.ipynb

**Learning outcome:** Implement a single neuron with weights, bias, activation.
Train it on linearly separable data. Understand the weight update rule.

Content:
1. A neuron: weighted sum + bias + activation — the three components
2. Implement `Neuron` class using `Value` objects from previous notebook
3. Implement `Layer` — list of neurons
4. Forward pass: `output = neuron(x)` where x is a list of Values
5. Loss: mean squared error — implement using Value arithmetic
6. Manual training loop: zero grad → forward → backward → update
7. Train on XOR — show it fails on non-linearly separable data
8. Visualisation: decision boundary at epoch 0, 10, 100
9. Structured interpretation block

### 01/03_mlp_numpy.ipynb

**Learning outcome:** Build a multi-layer perceptron using only NumPy.
Implement forward pass, loss, and gradient descent without autograd.

Content:
1. Extend from single neuron to layers — weight matrix W, bias vector b
2. Forward pass: `Z = X @ W + b`, `A = relu(Z)` — numpy only
3. Loss: cross-entropy — implement from scratch, verify numerically
4. Backward pass: derive gradients for each layer manually
   - dL/dW₂, dL/db₂ — output layer
   - dL/dW₁, dL/db₁ — hidden layer (chain rule through relu)
5. Training loop: 100 epochs on MNIST subset (first 1000 samples)
6. Compare to notebook 01 Value-based MLP — same results, different mechanism
7. Visualisation: loss curve, accuracy curve
8. Structured interpretation block

### 01/04_backprop_by_hand.ipynb

**Learning outcome:** Backpropagate manually through cross-entropy loss,
linear layer, tanh, batchnorm, embedding table. No autograd used.

This is the hardest notebook in the section. Take time on batchnorm.

Content:
1. Setup: 2-layer MLP, character-level language model (Karpathy makemore arc)
2. Forward pass — log every intermediate tensor by name and shape
3. Manual backward:
   - Through cross-entropy loss: `dlogits = probs.copy(); dlogits[range(n), Yb] -= 1; dlogits /= n`
   - Through second linear layer: `dh = dlogits @ W2.T`, `dW2 = h.T @ dlogits`
   - Through tanh: `dhpreact = (1 - h**2) * dh`
   - Through batchnorm: derive all three terms (dγ, dβ, dx̂)
     **Checkpoint:** dx̂ expands into three sub-terms via chain rule through
     mean and variance. If gradients don't match to 1e-4 after batchnorm,
     stop and debug — the three-term expansion of dx̂ is the most common
     failure point. Do not proceed to the embedding backward until
     batchnorm gradients are verified.
   - Through first linear layer and embedding lookup
4. Compare every gradient to `loss.backward()` — must match to 1e-4
5. Visualisation: gradient magnitude heatmap across layers
6. Structured interpretation block

### 01/05_train_loop_anatomy.ipynb

**Learning outcome:** Understand every line of a PyTorch training loop.
Implement training, evaluation, and checkpointing from scratch.

Content:
1. The six steps: zero_grad → forward → loss → backward → clip → step
2. Why zero_grad first — accumulated gradients as a bug
3. Gradient clipping — why and when, implement `nn.utils.clip_grad_norm_`
4. Evaluation mode — batch norm and dropout behaviour difference
5. Train/val/test split — implement proper split, no data leakage
6. Overfitting deliberately — train tiny model on tiny data, watch val loss diverge
7. Early stopping — implement with patience
8. Checkpointing — save and reload model state
9. Visualisation: training dashboard — loss, lr, grad norm in one figure
10. Structured interpretation block

---

## Section 03 — Training Science

**Goal:** Learner can read a training run's activation and gradient statistics
and diagnose instability. This section runs in parallel with Section 02 from
notebook 3 onward — both should be open simultaneously.

### 03/01_activation_statistics.ipynb

**Learning outcome:** Plot activation distributions across layers. Identify
saturation, dead neurons, and improper initialisation from histograms alone.

Content:
1. Why activations matter — a saturated tanh has near-zero gradient
2. Build a 6-layer MLP, initialise with large random weights
3. Forward pass on a batch — collect activations at every layer
4. Plot: histogram grid (6 subplots) showing activation distributions
5. Show: most activations are ±1 (saturated tanh) → dead gradients
6. Fix: scale weights by 1/sqrt(fan_in) — Kaiming / Xavier init
7. Show: activations now roughly Gaussian — training can proceed
8. Visualisation: before/after init comparison — 2×6 histogram grid
9. Structured interpretation block

### 03/02_gradient_flow.ipynb

**Learning outcome:** Plot gradient norms across layers. Identify vanishing
and exploding gradients. Understand why deep networks are fragile.

Content:
1. The problem: gradient signal decays (vanishes) or grows (explodes) with depth
2. Build a 10-layer network — run one backward pass
3. Plot gradient norms layer by layer — show the exponential decay
4. Demonstrate exploding: large weights, no clipping — loss goes NaN
5. Fix vanishing: residual connections — add skip connections, replot
6. Fix exploding: gradient clipping — implement, show stable training
7. Numerical stability: log-sum-exp trick — why it matters
8. Visualisation: gradient norm plot across layers, with/without residuals
9. Structured interpretation block

### 03/03_batch_norm_layernorm.ipynb

**Learning outcome:** Implement batchnorm and layernorm from scratch.
Explain why each normalisation approach works and when to prefer each.

Content:
1. The problem batchnorm solves — internal covariate shift
2. Implement batchnorm from scratch: mean, variance, normalise, scale, shift
3. Training vs inference difference — running mean/variance
4. Implement layernorm from scratch — normalise over features, not batch
5. When to use which: CNNs → batchnorm, Transformers → layernorm
6. Verify: both implementations match PyTorch nn.BatchNorm1d / LayerNorm
7. Visualisation: activation distributions before/after each norm type
8. Structured interpretation block

### 03/04_optimisers.ipynb

**Learning outcome:** Implement SGD, SGD+Momentum, RMSProp, Adam from scratch.
Understand what each adds and why Adam is the default starting point.

Content:
1. SGD — implement, show it oscillates in ravines
2. SGD + Momentum — exponential moving average of gradient
3. RMSProp — adaptive learning rate per parameter
4. Adam — momentum + adaptive LR, bias correction
5. AdamW — decoupled weight decay (current best practice)
6. Implement all five as classes with `step()` method
7. Race them on a 2D loss landscape — visualise trajectories
   Use the Beale function: f(x,y) = (1.5-x+xy)² + (2.25-x+xy²)² + (2.625-x+xy³)²
   Minimum at (3, 0.5). Starting point: (-3, -3). This surface has a narrow
   curved valley that exposes oscillation in vanilla SGD and shows momentum's
   advantage clearly.
8. Structured interpretation block

### 03/05_regularisation.ipynb

**Learning outcome:** Implement and understand L1, L2, dropout, and early
stopping. Know which to reach for first and why.

Content:
1. Overfit deliberately — tiny dataset, large model
2. L2 regularisation (weight decay) — implement, show effect on weights
3. L1 regularisation — sparse weights, implement, compare
4. Dropout — implement from scratch including train/eval mode toggle
5. Early stopping — implement with patience parameter
6. Batch size as regularisation — small batch = noisy gradients = regularisation
7. Visualisation: loss curves with/without each regulariser
8. Structured interpretation block

### 03/06_learning_rate_scheduling.ipynb

**Learning outcome:** Implement step decay, cosine annealing, warmup, and
OneCycleLR. Understand why LR scheduling is often the highest-leverage knob.

Content:
1. Why LR matters — too high diverges, too low stalls
2. LR range test — implement, find the right order of magnitude
3. Step decay — implement, show stepwise loss improvement
4. Cosine annealing — smooth decay, implement from scratch
5. Linear warmup — why it helps with Adam, implement
   **Cross-reference:** Revisit this notebook after completing 02/05_transformer.ipynb —
   the warmup schedule directly determines whether the transformer converges.
   The transformer notebook uses warmup+cosine; this notebook teaches why.
6. OneCycleLR — warmup then decay, PyTorch implementation
7. Visualisation: LR schedule curves + corresponding loss curves
8. Structured interpretation block

---

## CLAUDE.md Update on Completion

Update CLAUDE.md State section and append Phase Notes:

```markdown
Active phase: BUILD_PHASE_C.md
Last completed notebook: 03/06_learning_rate_scheduling.ipynb

### Phase B — [date]
Completed: 14 notebooks (00/01–03, 01/01–05, 03/01–06)
Key metrics: Gradient match in 01/04 verified to 1e-4. All 14 run clean.
Deviation from spec: [none / description]
Lesson learned: [one sentence]
Verified baseline for next phase: autograd, backprop, training loop all working
```

Smoke tests: `pytest -m smoke` passes with new Phase B tests added.

---

## Next Phase

On completion, proceed to:
`BUILD_PHASE_C.md` — Architecture Zoo (02_architecture_zoo)
