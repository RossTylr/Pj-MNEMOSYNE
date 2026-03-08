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
