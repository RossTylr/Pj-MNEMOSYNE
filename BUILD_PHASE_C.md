# BUILD_PHASE_C.md — Architecture Zoo

## Phase Goal

Build Section 02 — seven notebooks covering the major neural network
architectures from MLP to diffusion. Each notebook answers the same
question: what problem was this architecture solving that the previous one
couldn't?

## Prerequisites

- [ ] Phase B complete and validated
- [ ] 01/01_scalar_autograd.ipynb working (used in 02/01)
- [ ] 03/02_gradient_flow.ipynb complete (needed for transformer stability discussion)

## Completion Criteria

Every notebook must:
- Follow `assets/NOTEBOOK_TEMPLATE.ipynb` — theory sketch → NumPy → PyTorch → train → probe
- Include a "Why this architecture?" section before any code
- Train on a real (small) dataset — not toy synthetic data
- End with internal probing: weight distributions, activation stats, or attention maps
- End with structured interpretation block
- Contain at least one silent correctness assertion (e.g. val loss < 2.0, accuracy > chance) [RAIE v3]

## Smoke Test Additions

Add to `tests/test_smoke.py` when Phase C completes:
```python
@pytest.mark.smoke
def test_transformer_generates_text():
    """Phase C: trained GPT produces non-empty text samples."""
    # Load saved model, generate 100 chars, assert len > 0 and not all same char
```

---

## 02/01_mlp.ipynb

**Problem solved:** Approximate any function — universal approximation.
**Dataset:** MNIST (60k train, 10k test — use torchvision)

Content:
1. "Why this architecture?" — the universal approximation theorem, intuitively
2. Theory sketch: layers as function composition, depth vs width
3. NumPy implementation — 2-layer MLP, forward + backward
4. PyTorch rewrite — nn.Linear, nn.ReLU, nn.CrossEntropyLoss
5. Train: 10 epochs, log loss and accuracy
6. Internal probing: weight distribution histograms for each layer
7. Failure mode: show what happens without any hidden layers (linear classifier)
8. Structured interpretation block

---

## 02/02_cnn.ipynb

**Problem solved:** Spatial invariance — MLPs don't know pixels are neighbours.
**Dataset:** CIFAR-10

Content:
1. "Why this architecture?" — parameter sharing, local receptive fields
2. Theory sketch: convolution as sliding dot product, feature maps, pooling
3. Implement convolution from scratch in NumPy (2D, single channel)
4. PyTorch rewrite — nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d
5. Build LeNet-style architecture — 2 conv layers + 2 fc layers
6. Train: 20 epochs on CIFAR-10, plot learning curves
7. Internal probing: visualise first-layer filters (should show edge detectors)
8. Compare: MLP on same data — show CNN wins with fewer parameters
9. Structured interpretation block

---

## 02/03_rnn_lstm.ipynb

**Problem solved:** Sequential dependencies — CNNs and MLPs have no memory.
**Dataset:** Shakespeare character-level text (Karpathy's tiny-shakespeare)

Content:
1. "Why this architecture?" — hidden state as compressed history
2. Theory sketch: RNN unrolled, vanishing gradient through time
3. Implement vanilla RNN from scratch in NumPy — forward pass only
4. Show vanishing gradient: plot gradient norms across timesteps
5. LSTM: forget gate, input gate, output gate — implement each
6. PyTorch rewrite — nn.LSTM, train character-level language model
7. Train: 5000 steps, sample generated text at 1000/3000/5000
8. Internal probing: cell state and hidden state distributions
9. Structured interpretation block

---

## 02/04_attention_mechanism.ipynb

**Problem solved:** Long-range dependencies — RNNs forget, attention doesn't.
**Dataset:** Synthetic sequence-to-sequence (copy task)

Content:
1. "Why this architecture?" — RNN bottleneck: entire history in one vector
2. Theory sketch: queries, keys, values — scaled dot-product attention
3. Implement attention from scratch in NumPy:
   `scores = (Q @ K.T) / sqrt(d_k)`, `weights = softmax(scores)`, `out = weights @ V`
4. Implement multi-head attention — h parallel attention heads
5. PyTorch rewrite — verify against nn.MultiheadAttention
6. Train on copy task: sequence in → same sequence out
7. Internal probing: visualise attention weights as heatmap
8. Show: attention weight on correct position for each output token
9. Structured interpretation block

---

## 02/05_transformer.ipynb

**Problem solved:** Attention alone isn't enough — need positional encoding,
residual connections, layer norm, feedforward sublayers.

**Dataset:** tiny-shakespeare character-level GPT (Karpathy GPT arc)

This is the centrepiece notebook. Do not rush it.

Content:
1. "Why this architecture?" — parallelism over sequences that RNNs can't do
2. Theory sketch: full transformer block — attention → add&norm → FFN → add&norm
3. Positional encoding — implement sinusoidal and learned variants
4. Build decoder-only transformer from scratch:
   - `CausalSelfAttention` — masked attention, no peeking at future tokens
     Implementation: `mask = torch.tril(torch.ones(T, T))`, apply as
     `scores.masked_fill(mask == 0, float('-inf'))` *before* softmax.
     This is the single most common implementation bug — get it right.
   - `MLP` — two linear layers with GELU
   - `Block` — attention + MLP with residuals and layernorm
   - `GPT` — embedding + n blocks + final linear head
5. Train: character-level GPT on tiny-shakespeare
   - Target: val loss < 1.5 in reasonable time
   - Log: loss, tokens/sec, sample text every 500 steps
6. Internal probing:
   - Attention pattern visualisation — all heads, sample sequence
   - Residual stream norms across layers
   - Weight distributions (should be roughly Gaussian if init is right)
7. Structured interpretation block

---

## 02/06_vae.ipynb

**Problem solved:** Generative modelling with a structured latent space —
autoencoders memorise, VAEs generalise.
**Dataset:** MNIST

Content:
1. "Why this architecture?" — deterministic autoencoder vs probabilistic encoder
2. Theory sketch: encoder → (μ, σ) → reparameterisation trick → decoder → ELBO loss
3. Implement VAE from scratch — encoder, reparameterisation, decoder
4. ELBO loss: reconstruction term + KL divergence term
5. PyTorch implementation — train on MNIST
6. Internal probing:
   - 2D latent space visualisation (colour by digit class)
   - Interpolation: decode a straight line through latent space
7. KL collapse: show what happens with too-high β, discuss β-VAE
8. Structured interpretation block

---

## 02/07_diffusion_basics.ipynb

**Problem solved:** High-quality generation without adversarial training —
GANs are unstable, diffusion is not.
**Dataset:** MNIST (simplified — full diffusion on CIFAR-10 is slow)

Content:
1. "Why this architecture?" — score matching, denoising as generation
2. Theory sketch: forward process (add noise), reverse process (denoise)
   - q(x_t | x_{t-1}) — Gaussian noise schedule
   - p_θ(x_{t-1} | x_t) — learned denoising
3. Implement noise schedule: linear beta schedule, compute α̅_t
4. Implement forward diffusion: `q_sample(x0, t, noise)`
5. Build a simple UNet-style denoiser (small, for MNIST)
   Architecture: 3-level UNet, 32 base channels (32→64→128), sinusoidal
   time embedding (dim=64), single ResBlock per level. Total ~200k params.
   Do not over-engineer — MNIST does not need a full Stable Diffusion UNet.
6. Training objective: predict the noise ε that was added
7. Sampling: DDPM reverse loop — iteratively denoise from Gaussian noise
8. Internal probing: visualise the denoising trajectory (T → 0)
9. Comparison to VAE: what diffusion gains (quality) and what it costs (speed)
10. Structured interpretation block

---

## CLAUDE.md Update on Completion

Update CLAUDE.md State section and append Phase Notes:

```markdown
Active phase: BUILD_PHASE_D0.md
Last completed notebook: 02/07_diffusion_basics.ipynb

### Phase C — [date]
Completed: 7 notebooks (02/01–07)
Key metrics: Transformer val loss = [x.xx]. All 7 run clean.
Deviation from spec: [none / description]
Lesson learned: [one sentence]
Verified baseline for next phase: all architectures MLP→diffusion working
```

Smoke tests: `pytest -m smoke` passes with new Phase C tests added.

---

## Next Phase

On completion, proceed to:
`BUILD_PHASE_D0.md` — Research Track (Section 06) — MnemosyneGenerator data layer
