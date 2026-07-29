"""Shared fixtures for the analysis test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def calc():
    """CHGNetCalculator instance on CPU, shared across the analysis test session."""
    pytest.importorskip("chgnet")
    from materialsframework.calculators.chgnet import CHGNetCalculator

    return CHGNetCalculator(device="cpu")
