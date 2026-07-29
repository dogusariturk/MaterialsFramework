"""Tests for USFETransformation."""

from __future__ import annotations

import numpy as np
import pytest

from materialsframework.transformations.usfe import USFETransformation

_DEFAULT_NUM_STEPS = 11
_TEST_STEPS = 5
_DISPLACEMENT_TOL = 1e-8


def test_default_params() -> None:
    """Default values are stored correctly."""
    t = USFETransformation()
    assert t.slip_plane == "110"
    assert len(t.displacement_fractions) == _DEFAULT_NUM_STEPS
    assert t.displacement_fractions[0] == pytest.approx(0.0)
    assert t.displacement_fractions[-1] == pytest.approx(1.0)


def test_invalid_slip_plane_raises() -> None:
    """Unsupported slip planes raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported slip_plane"):
        USFETransformation(slip_plane="111")


@pytest.mark.parametrize("slip_plane", ["110", "112"])
def test_apply_transformation_returns_structures(bcc_fe, slip_plane: str) -> None:
    """apply_transformation returns one displaced structure per displacement fraction."""
    t = USFETransformation(slip_plane=slip_plane, start=0.0, stop=1.0, num_steps=_TEST_STEPS)
    result = t.apply_transformation(bcc_fe)
    displaced_structures = result["displaced_structures"]

    assert len(displaced_structures) == _TEST_STEPS
    assert result["fault_area"] is not None
    assert result["fault_area"] > 0

    for frac, struct in displaced_structures.items():
        assert isinstance(frac, float)
        assert struct.__class__.__name__ == "Structure"
        assert len(struct) == len(bcc_fe)


def test_displacement_is_zero_at_fraction_zero(bcc_fe) -> None:
    """At zero displacement fraction, generated structure equals the input."""
    t = USFETransformation(slip_plane="110", start=0.0, stop=1.0, num_steps=3)
    result = t.apply_transformation(bcc_fe)

    zero_struct = result["displaced_structures"][0.0]
    assert np.allclose(zero_struct.cart_coords, bcc_fe.cart_coords)


def test_upper_half_is_shifted_for_nonzero_fraction(bcc_fe) -> None:
    """A non-zero displacement shifts only part of the sites across the fault plane."""
    t = USFETransformation(slip_plane="110", start=0.0, stop=1.0, num_steps=2)
    result = t.apply_transformation(bcc_fe)

    displaced = result["displaced_structures"][1.0]
    delta = displaced.cart_coords - bcc_fe.cart_coords
    moved = np.linalg.norm(delta, axis=1) > _DISPLACEMENT_TOL

    assert moved.any()
    assert (~moved).any()
