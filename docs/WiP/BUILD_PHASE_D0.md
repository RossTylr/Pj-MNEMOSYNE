# BUILD_PHASE_D0.md — MnemosyneGenerator

## Phase Goal

Build and validate the shared data layer for the entire research track.
Nothing in Da, Db, or Dc runs without this.

`06_research_track/data/mnemosyne_synthetic/generator.py`

The skeleton from Phase A raises `NotImplementedError` throughout.
This phase replaces it with a complete, tested implementation.

## Prerequisites

- [ ] Phase A complete — skeleton files exist
- [ ] Phase B complete — familiar with NumPy sampling patterns

## Completion Gate

Before marking D0 complete:

- [ ] `MnemosyneGenerator("HIGH_INTENSITY").generate_triage_dataset(n=100)` runs without error
- [ ] `MnemosyneGenerator("HIGH_INTENSITY").generate_survival_dataset(n=100, n_mc_runs=10)` runs without error
- [ ] `MnemosyneGenerator("HIGH_INTENSITY").generate_sequence_dataset(n_runs=10, duration_hours=12)` returns shape `(10, 144, 8)`
- [ ] Validation notebook `01_experiment_baseline/01_data_validation.ipynb` (stub run) produces plausible distributions
- [ ] All three context strings produce different mechanism distributions
- [ ] `NotImplementedError` no longer raised anywhere in the module
- [ ] `pytest -m smoke` passes with new D0 tests added

## Smoke Test Additions

Add to `tests/test_smoke.py` when Phase D0 completes:
```python
@pytest.mark.smoke
def test_generator_triage():
    """Phase D0: generator produces plausible triage dataset."""
    from mnemosyne_synthetic.generator import MnemosyneGenerator
    gen = MnemosyneGenerator("HIGH_INTENSITY", seed=42)
    df = gen.generate_triage_dataset(n=100)
    assert df.shape[0] == 100
    assert df["severity"].between(0, 1).all()
    assert df["triage_category"].nunique() >= 3

@pytest.mark.smoke
def test_generator_survival():
    """Phase D0: survival probabilities in valid range."""
    from mnemosyne_synthetic.generator import MnemosyneGenerator
    gen = MnemosyneGenerator("HIGH_INTENSITY", seed=42)
    sdf = gen.generate_survival_dataset(n=50, n_mc_runs=10)
    assert sdf["survival_probability"].between(0, 1).all()
    assert sdf["survival_probability"].std() > 0.01  # not all same value

@pytest.mark.smoke
def test_generator_context_shift():
    """Phase D0: different contexts produce different distributions."""
    from mnemosyne_synthetic.generator import MnemosyneGenerator
    pk = MnemosyneGenerator("PEACEKEEPING", seed=42).generate_triage_dataset(500)
    dr = MnemosyneGenerator("PEER_COMPETITOR_DRONE", seed=42).generate_triage_dataset(500)
    assert dr["mechanism"].eq(0).mean() > pk["mechanism"].eq(0).mean()  # more BLAST
```

---

## Clinical Constants

Replace the placeholder `raise NotImplementedError` in `generator.py` with
the following constants and implementation.

### Mechanism distributions by context

```python
MECHANISMS = ["BLAST", "GSW", "BURN", "FRAGMENTATION", "BLUNT"]

CONTEXT_MECHANISM_PROBS = {
    "HIGH_INTENSITY":        [0.40, 0.25, 0.08, 0.20, 0.07],
    "PEACEKEEPING":          [0.15, 0.50, 0.05, 0.20, 0.10],
    "PEER_COMPETITOR_DRONE": [0.55, 0.15, 0.05, 0.22, 0.03],
}
```

### Anatomical regions

```python
REGIONS = [
    "HEAD", "THORAX", "ABDOMEN", "UPPER_EXTREMITY",
    "LOWER_EXTREMITY", "SPINE", "PELVIS", "FACE", "NECK", "MULTIPLE"
]

SURGICAL_REGIONS = {"HEAD", "THORAX", "ABDOMEN", "SPINE", "PELVIS"}

MECHANISM_REGION_PROBS = {
    "BLAST":         [0.15, 0.20, 0.15, 0.15, 0.20, 0.05, 0.05, 0.03, 0.02, 0.00],
    "GSW":           [0.12, 0.18, 0.20, 0.18, 0.18, 0.05, 0.04, 0.02, 0.03, 0.00],
    "BURN":          [0.10, 0.15, 0.10, 0.25, 0.25, 0.02, 0.03, 0.05, 0.03, 0.02],
    "FRAGMENTATION": [0.12, 0.18, 0.12, 0.20, 0.22, 0.04, 0.04, 0.04, 0.02, 0.02],
    "BLUNT":         [0.15, 0.15, 0.20, 0.15, 0.15, 0.10, 0.05, 0.02, 0.02, 0.01],
}
```

### Severity sampling (Beta distribution parameters by context)

```python
# Beta(alpha, beta) — HIGH_INTENSITY skews toward higher severity
SEVERITY_BETA_PARAMS = {
    "HIGH_INTENSITY":        (3.0, 2.0),
    "PEACEKEEPING":          (2.0, 3.0),
    "PEER_COMPETITOR_DRONE": (3.5, 1.8),
}

MECHANISM_SEVERITY_MODIFIER = {
    "BLAST":         1.2,
    "GSW":           1.0,
    "BURN":          1.3,
    "FRAGMENTATION": 1.1,
    "BLUNT":         0.8,
}
```

### Triage assignment from severity

```python
# Severity thresholds — assign triage from severity float
# Applied after mechanism modifier and clipping to [0, 1]
def _assign_triage(severity: float) -> str:
    if severity >= 0.85:
        return "T4"           # unsurvivable
    elif severity >= 0.72:
        return "T1_SURGICAL"  # immediate surgical
    elif severity >= 0.58:
        return "T1_MEDICAL"   # immediate medical
    elif severity >= 0.35:
        return "T2"           # delayed
    else:
        return "T3"           # minimal
```

### Vitals by triage (mean, std) — (HR, SBP, RR, GCS, SpO2)

```python
VITALS_BY_TRIAGE = {
    "T1_SURGICAL": [(130,20), (70,15),  (28,5), (8,3),  (88,5)],
    "T1_MEDICAL":  [(125,18), (75,12),  (26,4), (10,3), (90,5)],
    "T2":          [(105,15), (95,10),  (22,3), (13,2), (94,3)],
    "T3":          [(85,10),  (115,8),  (16,2), (15,1), (97,2)],
    "T4":          [(45,20),  (50,20),  (6,3),  (4,2),  (78,8)],
}
VITALS_NAMES = ["heart_rate", "systolic_bp", "respiratory_rate", "gcs", "spo2"]
VITALS_CLIP  = [(30,200),     (0,200),       (0,60),             (3,15),(70,100)]
```

### Deterioration and PFC

```python
DETERIORATION_RATE_PER_HOUR = {
    "T1_SURGICAL": 0.12,
    "T1_MEDICAL":  0.08,
    "T2":          0.04,
    "T3":          0.01,
    "T4":          0.20,
}
PFC_MITIGATION_FACTOR = 0.70   # PFC reduces deterioration rate by 70%
```

---

## generate_triage_dataset() — full implementation

```python
def generate_triage_dataset(self, n: int = 10_000) -> pd.DataFrame:
    rows = []
    mechanism_probs = CONTEXT_MECHANISM_PROBS[self.context]
    beta_a, beta_b  = SEVERITY_BETA_PARAMS[self.context]

    for _ in range(n):
        # Sample mechanism and region
        mechanism = self.rng.choice(MECHANISMS, p=mechanism_probs)
        region    = self.rng.choice(REGIONS, p=MECHANISM_REGION_PROBS[mechanism])

        # Sample severity
        raw_sev  = self.rng.beta(beta_a, beta_b)
        severity = float(np.clip(raw_sev * MECHANISM_SEVERITY_MODIFIER[mechanism], 0, 1))

        # Polytrauma
        polytrauma_prob = {"BLAST":0.45, "GSW":0.20, "BURN":0.15,
                           "FRAGMENTATION":0.35, "BLUNT":0.25}[mechanism]
        is_polytrauma   = bool(self.rng.random() < polytrauma_prob)

        # Assign triage
        triage = _assign_triage(severity)

        # Sample vitals
        vitals = {}
        for (mean, std), name, (lo, hi) in zip(
            VITALS_BY_TRIAGE[triage], VITALS_NAMES, VITALS_CLIP
        ):
            val = self.rng.normal(mean, std)
            vitals[name] = float(np.clip(val, lo, hi))

        # PFC hours (time before any care reaches casualty)
        pfc_hours = float(self.rng.exponential(
            scale={"HIGH_INTENSITY":2.5, "PEACEKEEPING":0.5,
                   "PEER_COMPETITOR_DRONE":4.0}[self.context]
        ))

        rows.append({
            **vitals,
            "mechanism":         MECHANISMS.index(mechanism),
            "primary_region":    REGIONS.index(region),
            "severity":          severity,
            "is_polytrauma":     int(is_polytrauma),
            "is_surgical_region":int(region in SURGICAL_REGIONS),
            "pfc_hours_elapsed": pfc_hours,
            "context":           ["HIGH_INTENSITY","PEACEKEEPING",
                                  "PEER_COMPETITOR_DRONE"].index(self.context),
            "triage_category":   triage,
        })

    return pd.DataFrame(rows)
```

---

## generate_survival_dataset() — full implementation

> **Runtime note:** For n=10,000 and n_mc_runs=500, the row-by-row loop
> below takes approximately 5–15 minutes depending on hardware. The
> vectorised `_mc_survival_batch()` method below the loop version reduces
> this to ~30 seconds. Use the vectorised version for production; the
> loop version is retained for readability.

```python
def generate_survival_dataset(
    self, n: int = 10_000, n_mc_runs: int = 500
) -> pd.DataFrame:
    triage_df = self.generate_triage_dataset(n=n)

    # Sample system state for all casualties at once
    r1_caps     = np.clip(self.rng.beta(2, 3, size=n), 0, 1)
    r2_queues   = self.rng.poisson(3, size=n)
    eta_mins    = np.clip(self.rng.exponential(60, size=n), 5, 480)
    threats     = self.rng.choice([0,1,2], size=n, p=[0.5, 0.35, 0.15])
    blood_stocks= np.clip(self.rng.normal(10, 4, size=n), 0, 20)
    concurrents = self.rng.poisson(2, size=n)

    # Vectorised MC survival computation
    survival_probs = self._mc_survival_batch(
        triage_cats=triage_df["triage_category"].values,
        severities=triage_df["severity"].values,
        pfc_hours=triage_df["pfc_hours_elapsed"].values,
        r1_caps=r1_caps, eta_mins=eta_mins, threats=threats,
        blood_stocks=blood_stocks, concurrents=concurrents,
        n_mc_runs=n_mc_runs,
    )

    result = pd.DataFrame({
        # Patient features
        "severity":              triage_df["severity"].values,
        "triage_category":       triage_df["triage_category"].values,
        "mechanism":             triage_df["mechanism"].values,
        "primary_region":        triage_df["primary_region"].values,
        "is_polytrauma":         triage_df["is_polytrauma"].values,
        "heart_rate":            triage_df["heart_rate"].values,
        "systolic_bp":           triage_df["systolic_bp"].values,
        "respiratory_rate":      triage_df["respiratory_rate"].values,
        "gcs":                   triage_df["gcs"].values,
        "spo2":                  triage_df["spo2"].values,
        "pfc_duration_mins":     triage_df["pfc_hours_elapsed"].values * 60,
        # System state features
        "r1_capacity_pct":       r1_caps,
        "r2_surgical_queue":     r2_queues,
        "transport_eta_mins":    eta_mins,
        "route_threat_level":    threats,
        "r2_blood_stock_units":  blood_stocks,
        "concurrent_casualties": concurrents,
        # Targets
        "survival_probability":  survival_probs,
        "survived":              (survival_probs > 0.5).astype(int),
    })
    return result


def _mc_survival_batch(
    self, triage_cats, severities, pfc_hours,
    r1_caps, eta_mins, threats, blood_stocks, concurrents,
    n_mc_runs: int,
) -> np.ndarray:
    """Vectorised MC survival — operates on arrays, not scalars."""
    n = len(severities)

    # Map triage categories to deterioration rates
    det_rate_map = {k: v for k, v in DETERIORATION_RATE_PER_HOUR.items()}
    det_rates = np.array([det_rate_map[t] for t in triage_cats])

    # Base survival (vectorised)
    base = 1.0 - severities
    det  = det_rates * pfc_hours * PFC_MITIGATION_FACTOR
    base = np.maximum(0.0, base - det)

    # System-state penalties (vectorised)
    penalties = (
        np.minimum(0.25, eta_mins / 480.0 * 0.25)
        + np.maximum(0.0, r1_caps - 0.75) * 0.30
        + threats * 0.08
        + np.minimum(0.15, concurrents / 10.0 * 0.15)
    )
    bonus = np.minimum(0.05, blood_stocks / 20.0 * 0.05)

    # MC noise (n_mc_runs samples per casualty)
    noise = self.rng.normal(0, 0.03, size=(n_mc_runs, n))
    raw   = base - penalties + bonus + noise   # shape: (n_mc_runs, n)
    return np.clip(raw, 0.0, 1.0).mean(axis=0)  # shape: (n,)
```

---

## generate_sequence_dataset() — full implementation

```python
SEQUENCE_FEATURES = [
    "casualties_at_r1", "casualties_at_r2", "casualties_in_transit",
    "r1_utilisation", "r2_surgical_utilisation",
    "transport_assets_available", "new_casualties_this_tick",
    "system_stress_level",   # 0=GREEN 1=AMBER 2=RED 3=BLACK
]

def generate_sequence_dataset(
    self, n_runs: int = 1_000, duration_hours: int = 12
) -> np.ndarray:
    """
    Returns array of shape (n_runs, timesteps, 8).
    timesteps = duration_hours * 12  (5-min ticks).
    Each run includes a randomly timed MASCAL inject.
    """
    timesteps = duration_hours * 12
    arr = np.zeros((n_runs, timesteps, len(SEQUENCE_FEATURES)))

    for run in range(n_runs):
        # MASCAL inject: random tick between t=30 and t=100
        mascal_tick   = int(self.rng.integers(30, 100))
        mascal_size   = int(self.rng.integers(5, 16))

        r1 = 2.0; r2 = 1.0; transit = 0.0
        r1_cap = 10; r2_cap = 4; transport_cap = 3

        for t in range(timesteps):
            # New arrivals — Poisson baseline + MASCAL inject
            new_cas = int(self.rng.poisson(0.5))
            if t == mascal_tick:
                new_cas += mascal_size

            # Simple flow model
            transit  += new_cas
            arriving  = min(transit, 2.0)
            transit   = max(0.0, transit - arriving)
            r1       += arriving
            moving_r2 = min(r1 * 0.3, float(r2_cap - r2))
            r1        = max(0.0, r1 - moving_r2)
            r2       += moving_r2
            discharged = min(r2 * 0.2, r2)
            r2        = max(0.0, r2 - discharged)

            r1_util = min(1.0, r1 / r1_cap)
            r2_util = min(1.0, r2 / r2_cap)

            # Stress level
            if r1_util > 0.9 and r2_util > 0.9:
                stress = 3  # BLACK
            elif r1_util > 0.9 or r2_util > 0.9:
                stress = 2  # RED
            elif r1_util > 0.7 or r2_util > 0.7:
                stress = 1  # AMBER
            else:
                stress = 0  # GREEN

            arr[run, t] = [
                r1, r2, transit, r1_util, r2_util,
                max(0, transport_cap - int(transit / 2)),
                new_cas, stress
            ]

    return arr
```

---

## noise_injection.py — full implementation

```python
"""Noise injection for distribution robustness testing."""
import numpy as np
import pandas as pd


def inject_noise(
    df: pd.DataFrame,
    corruption_rate: float = 0.15,
    vitals_cols: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Corrupt vitals columns at corruption_rate by replacing with
    out-of-range or missing values. Simulates sensor failure / field conditions.

    Args:
        df: casualty DataFrame
        corruption_rate: fraction of rows to corrupt per column
        vitals_cols: columns to corrupt (defaults to 5 vital signs)
        seed: random seed

    Returns:
        Corrupted copy of df
    """
    rng = np.random.default_rng(seed)
    df  = df.copy()
    if vitals_cols is None:
        vitals_cols = ["heart_rate","systolic_bp","respiratory_rate","gcs","spo2"]

    for col in vitals_cols:
        mask = rng.random(len(df)) < corruption_rate
        # Replace with column mean (simulates default/missing value fill)
        df.loc[mask, col] = df[col].mean()

    return df
```

---

## distribution_shift.py — full implementation

```python
"""Distribution shift utilities for H4 testing."""
import pandas as pd
from .generator import MnemosyneGenerator


def generate_shifted_pair(
    n: int = 10_000,
    source_context: str = "PEACEKEEPING",
    target_context: str = "PEER_COMPETITOR_DRONE",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate matched source and target datasets for distribution shift
    experiments (H4).

    Returns:
        (source_df, target_df) — same n, different context distributions
    """
    source = MnemosyneGenerator(context=source_context, seed=seed)
    target = MnemosyneGenerator(context=target_context, seed=seed+1)
    return source.generate_triage_dataset(n), target.generate_triage_dataset(n)
```

---

## Validation Script

After implementation, run this in a terminal to confirm D0 complete:

```bash
cd 06_research_track/data/mnemosyne_synthetic
python - <<'EOF'
from generator import MnemosyneGenerator
import numpy as np

gen = MnemosyneGenerator("HIGH_INTENSITY", seed=42)

# Test 1: triage dataset
df = gen.generate_triage_dataset(n=1000)
assert df.shape == (1000, 13), f"Shape: {df.shape}"
assert df["triage_category"].nunique() == 5, "Should have 5 triage classes"
assert df["severity"].between(0, 1).all(), "Severity out of range"
print(f"✓ Triage dataset: {df.shape}, classes: {df['triage_category'].value_counts().to_dict()}")

# Test 2: survival dataset
sdf = gen.generate_survival_dataset(n=200, n_mc_runs=50)
assert sdf.shape == (200, 19), f"Shape: {sdf.shape}"
assert sdf["survival_probability"].between(0, 1).all()
print(f"✓ Survival dataset: {sdf.shape}, mean P(survival): {sdf['survival_probability'].mean():.3f}")

# Test 3: sequence dataset
arr = gen.generate_sequence_dataset(n_runs=20, duration_hours=12)
assert arr.shape == (20, 144, 8), f"Shape: {arr.shape}"
print(f"✓ Sequence dataset: {arr.shape}")

# Test 4: context differences
gen_pk = MnemosyneGenerator("PEACEKEEPING", seed=42)
gen_dr = MnemosyneGenerator("PEER_COMPETITOR_DRONE", seed=42)
df_pk  = gen_pk.generate_triage_dataset(n=1000)
df_dr  = gen_dr.generate_triage_dataset(n=1000)
# Drone context should have more BLAST (mechanism=0)
assert df_dr["mechanism"].eq(0).mean() > df_pk["mechanism"].eq(0).mean(), \
    "Drone context should have higher BLAST proportion"
print(f"✓ Context shift: BLAST% PK={df_pk['mechanism'].eq(0).mean():.2f}, "
      f"DRONE={df_dr['mechanism'].eq(0).mean():.2f}")

print("\nAll D0 validation checks passed.")
EOF
```

---

## CLAUDE.md Update on Completion

Update CLAUDE.md State section and append Phase Notes:

```markdown
Active phase: BUILD_PHASE_Da.md
Last completed notebook: (D0 is module work, not notebooks)

### Phase D0 — [date]
Completed: MnemosyneGenerator — triage, survival, sequence datasets
Key metrics: Validation all 4 checks pass. mean P(survival) HI=[x.xxx].
  BLAST% PK=[x.xx] → DRONE=[x.xx]. Sequence shape (n, 144, 8) confirmed.
Deviation from spec: [none / description]
Lesson learned: [one sentence]
Verified baseline for next phase: generator produces plausible data across 3 contexts
```

Smoke tests: `pytest -m smoke` passes with D0 generator tests added.

---

## Next Phase

On completion, proceed to:
`BUILD_PHASE_Da.md` — Experiment 1: Baseline Establishment
