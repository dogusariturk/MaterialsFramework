"""Tests for TrajectoryObserver using ASE's built-in EMT calculator."""

from __future__ import annotations

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import FIRE

from materialsframework.tools.trajectory import TrajectoryObserver


@pytest.fixture
def cu_atoms():
    """FCC Cu Atoms object with EMT calculator attached."""
    atoms = bulk("Cu", "fcc", a=3.6) * (1, 1, 1)
    atoms.calc = EMT()
    return atoms


def test_observer_records_on_call(cu_atoms) -> None:
    """Calling the observer once records one step."""
    obs = TrajectoryObserver(cu_atoms)
    obs()
    assert len(obs) == 1


def test_observer_records_multiple_steps(cu_atoms) -> None:
    """Each call to the observer appends one step."""
    obs = TrajectoryObserver(cu_atoms)
    for _ in range(5):
        obs()
    assert len(obs) == 5


def test_observer_potential_energy(cu_atoms) -> None:
    """Recorded potential energy is a finite float."""
    obs = TrajectoryObserver(cu_atoms)
    obs()
    energy = obs.potential_energies[0]
    assert isinstance(energy, float)


def test_observer_forces_shape(cu_atoms) -> None:
    """Recorded forces have shape (n_atoms, 3)."""
    import numpy as np

    obs = TrajectoryObserver(cu_atoms)
    obs()
    forces = np.array(obs.forces[0])
    assert forces.shape == (len(cu_atoms), 3)


def test_observer_as_pandas(cu_atoms) -> None:
    """as_pandas() returns a DataFrame with expected columns."""
    obs = TrajectoryObserver(cu_atoms)
    obs()
    df = obs.as_pandas()
    assert "potential_energies" in df.columns
    assert "forces" in df.columns
    assert len(df) == 1


def test_observer_attached_during_optimization(cu_atoms) -> None:
    """Observer accumulates steps when attached to an ASE optimizer."""
    obs = TrajectoryObserver(cu_atoms)
    optimizer = FIRE(cu_atoms, logfile=None)
    optimizer.attach(obs, interval=1)
    optimizer.run(fmax=1.0, steps=3)
    assert len(obs) >= 1
