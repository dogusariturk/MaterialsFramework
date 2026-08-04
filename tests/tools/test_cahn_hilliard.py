"""Tests for Cahn-Hilliard phase field classes.

SimulationGrid and PhaseFieldModel.laplacian are tested here without a
pycalphad Database (no external I/O required).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pycalphad")

pytestmark = pytest.mark.integration

from materialsframework.tools.cahn_hilliard import SimulationGrid


def test_simulation_grid_default_shape() -> None:
    """Default SimulationGrid has 128x128 phi field."""
    grid = SimulationGrid()
    assert grid.phi.shape == (128, 128)


def test_simulation_grid_custom_shape() -> None:
    """Custom nx/ny produce a correspondingly shaped phi field."""
    grid = SimulationGrid(nx=16, ny=32)
    assert grid.phi.shape == (16, 32)
    assert grid.nx == 16
    assert grid.ny == 32


def test_simulation_grid_dx() -> None:
    """Dx is computed as lx / (nx - 1)."""
    grid = SimulationGrid(nx=11, lx=1.0)
    assert grid.dx == pytest.approx(0.1)


def test_simulation_grid_phi_initialised_to_zero() -> None:
    """Phi array is initialised to all zeros."""
    grid = SimulationGrid(nx=8, ny=8)
    assert np.all(grid.phi == 0.0)


def test_simulation_grid_laplace_factor() -> None:
    """laplace_factor equals 1 / (4 * dx^2)."""
    grid = SimulationGrid(nx=11, lx=1.0)
    expected = 1.0 / (4.0 * grid.dx**2)
    assert grid.laplace_factor == pytest.approx(expected)


def test_simulation_grid_auxiliary_arrays_shape() -> None:
    """All auxiliary arrays (lap_phi, chem_pot, lap_chem_pot) match phi shape."""
    grid = SimulationGrid(nx=16, ny=16)
    for arr in (grid.lap_phi, grid.chem_pot, grid.lap_chem_pot):
        assert arr.shape == grid.phi.shape


def test_phase_field_model_laplacian_of_flat_field() -> None:
    """laplacian() of a uniform field is zero at interior points."""
    import numpy as np

    from materialsframework.tools.cahn_hilliard import PhaseFieldModel

    class _StubMaterial:
        composition = 0.5
        kappa = np.longdouble(1e-11)
        mobility = np.longdouble(1e-10)
        free_energy_poly_deriv = np.poly1d([0.0])

    grid = SimulationGrid(nx=16, ny=16)
    model = PhaseFieldModel(_StubMaterial(), simulation_grid=grid)  # ty: ignore[invalid-argument-type]
    flat = np.ones((16, 16))
    lap = model.laplacian(flat)
    assert np.allclose(lap[1:-1, 1:-1], 0.0, atol=1e-10)
