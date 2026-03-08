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
