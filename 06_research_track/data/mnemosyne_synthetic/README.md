# mnemosyne_synthetic — Synthetic Data Generator

Single source of truth for all synthetic data in the research track.
All clinical constants are hardcoded from FAER(MIL) sampling parameters.

| Module | Purpose | Phase |
|--------|---------|-------|
| generator.py | MnemosyneGenerator class | D0 |
| noise_injection.py | Noise injection utilities | D0 |
| distribution_shift.py | Cross-context evaluation | D0 |
