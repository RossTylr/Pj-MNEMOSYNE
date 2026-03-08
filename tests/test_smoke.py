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


@pytest.mark.smoke
def test_value_class_backward():
    """Phase B: Value autograd engine matches PyTorch."""
    import math
    import torch

    class Value:
        def __init__(self, data, _children=(), _op=''):
            self.data = float(data)
            self.grad = 0.0
            self._backward = lambda: None
            self._prev = set(_children)
            self._op = _op

        def __add__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data + other.data, (self, other), '+')
            def _backward():
                self.grad += out.grad
                other.grad += out.grad
            out._backward = _backward
            return out

        def __mul__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data * other.data, (self, other), '*')
            def _backward():
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            out._backward = _backward
            return out

        def tanh(self):
            t = math.tanh(self.data)
            out = Value(t, (self,), 'tanh')
            def _backward():
                self.grad += (1 - t ** 2) * out.grad
            out._backward = _backward
            return out

        def backward(self):
            topo, visited = [], set()
            def build(v):
                if v not in visited:
                    visited.add(v)
                    for c in v._prev:
                        build(c)
                    topo.append(v)
            build(self)
            self.grad = 1.0
            for v in reversed(topo):
                v._backward()

        def __radd__(self, other): return self + other
        def __rmul__(self, other): return self * other

    # Test: (a * b + c).tanh()
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    L = (a * b + c).tanh()
    L.backward()

    a_pt = torch.tensor(2.0, requires_grad=True)
    b_pt = torch.tensor(-3.0, requires_grad=True)
    c_pt = torch.tensor(10.0, requires_grad=True)
    L_pt = (a_pt * b_pt + c_pt).tanh()
    L_pt.backward()

    assert abs(a.grad - a_pt.grad.item()) < 1e-4
    assert abs(b.grad - b_pt.grad.item()) < 1e-4
    assert abs(c.grad - c_pt.grad.item()) < 1e-4


@pytest.mark.smoke
def test_numpy_mlp_trains():
    """Phase B: NumPy MLP loss decreases over 10 epochs."""
    import numpy as np
    np.random.seed(42)

    # Synthetic 2-class data
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y_oh = np.zeros((100, 2))
    y_oh[np.arange(100), y] = 1

    # 2-layer MLP
    W1 = np.random.randn(4, 16) * np.sqrt(2.0 / 4)
    b1 = np.zeros(16)
    W2 = np.random.randn(16, 2) * np.sqrt(2.0 / 16)
    b2 = np.zeros(2)

    def softmax(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    initial_loss = None
    for epoch in range(10):
        Z1 = X @ W1 + b1
        A1 = np.maximum(Z1, 0)
        Z2 = A1 @ W2 + b2
        probs = softmax(Z2)
        loss = -np.sum(y_oh * np.log(probs + 1e-12)) / len(X)

        if initial_loss is None:
            initial_loss = loss

        N = len(X)
        dZ2 = (probs - y_oh) / N
        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0)

        lr = 0.5
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

    assert loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} → {loss:.4f}"


@pytest.mark.smoke
def test_foundations_linear_algebra_shapes():
    """Phase B: matrix shape rules and eigendecomposition are correct."""
    import numpy as np
    np.random.seed(42)
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 2)
    C = A @ B
    assert C.shape == (3, 2)
    assert A.T.shape == (4, 3)
    # Eigendecomposition verification
    E = np.array([[2.0, 1.0], [1.0, 3.0]])
    vals, vecs = np.linalg.eig(E)
    for i in range(2):
        assert np.allclose(E @ vecs[:, i], vals[i] * vecs[:, i])


@pytest.mark.smoke
def test_foundations_distributions():
    """Phase B: Beta-Bernoulli posterior concentrates correctly."""
    alpha_prior, beta_prior = 2, 2
    s, n = 70, 100
    alpha_post = alpha_prior + s
    beta_post = beta_prior + (n - s)
    post_mean = alpha_post / (alpha_post + beta_post)
    assert abs(post_mean - 0.70) < 0.02


@pytest.mark.smoke
def test_batchnorm_backward_matches_pytorch():
    """Phase B: hand-derived batchnorm gradient matches PyTorch."""
    import numpy as np
    import torch
    np.random.seed(42)
    N, D, eps = 4, 3, 1e-5
    x = np.random.randn(N, D)
    gamma = np.random.randn(D)
    # NumPy forward
    mu = x.mean(axis=0)
    xmu = x - mu
    var = (xmu ** 2).mean(axis=0)
    std_inv = 1.0 / np.sqrt(var + eps)
    xhat = xmu * std_inv
    # Compact backward
    np.random.seed(99)
    dy = np.random.randn(N, D)
    dxhat = dy * gamma
    dx = (1.0 / N) * std_inv * (
        N * dxhat - dxhat.sum(axis=0)
        - xhat * (dxhat * xhat).sum(axis=0)
    )
    # PyTorch reference
    x_pt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    gamma_pt = torch.tensor(gamma, dtype=torch.float64, requires_grad=True)
    beta_pt = torch.tensor(np.random.randn(D), dtype=torch.float64, requires_grad=True)
    mu_pt = x_pt.mean(dim=0)
    xhat_pt = (x_pt - mu_pt) / torch.sqrt((x_pt - mu_pt).pow(2).mean(dim=0) + eps)
    y_pt = gamma_pt * xhat_pt + beta_pt
    y_pt.backward(torch.tensor(dy, dtype=torch.float64))
    assert np.max(np.abs(dx - x_pt.grad.numpy())) < 1e-4


@pytest.mark.smoke
def test_training_loop_converges():
    """Phase B: small model loss decreases over 50 training steps."""
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    X = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    initial_loss = criterion(model(X), y).item()
    for _ in range(50):
        opt.zero_grad()
        criterion(model(X), y).backward()
        opt.step()
    final_loss = criterion(model(X), y).item()
    assert final_loss < initial_loss


@pytest.mark.smoke
def test_kaiming_preserves_activation_scale():
    """Phase B: Kaiming init keeps activation std stable across 10 layers."""
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    width, n_layers = 256, 10
    x = torch.randn(64, width)
    h = x
    for _ in range(n_layers):
        lin = nn.Linear(width, width, bias=False)
        nn.init.kaiming_normal_(lin.weight, mode='fan_in', nonlinearity='relu')
        h = torch.relu(lin(h))
    ratio = h.std().item() / x.std().item()
    assert 0.05 < ratio < 20, f"Activation std ratio {ratio:.2f} out of range"


# Phase D0 will add: test_generator_runs
# Phase Da will add: test_baseline_f1_above_majority
# Phase Db will add: test_survival_probabilities_in_range
