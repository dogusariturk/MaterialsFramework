"""Tests for BaseCalculator using RandomCalculator (no ML extras required)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from ase.calculators.emt import EMT
from ase.optimize import BFGS

import materialsframework.tools.calculator as calculator_module
from materialsframework.tools.calculator import OPTIMIZERS, BaseCalculator

if TYPE_CHECKING:
    from pymatgen.core import Structure

from materialsframework.calculators.random import RandomCalculator


@pytest.fixture
def calc() -> RandomCalculator:
    """RandomCalculator instance for testing BaseCalculator behavior."""
    return RandomCalculator()


class _EMTCalculator(BaseCalculator):
    """Minimal concrete `BaseCalculator` backed by ASE's dependency-free EMT calculator.

    Unlike `RandomCalculator`, this does NOT override `relax()`/`calculate()`, so it exercises
    `BaseCalculator`'s own shared logic (constraints, cell-filter wrapping, optimizer dispatch)
    directly, the same way every real MLIP calculator does.
    """

    AVAILABLE_PROPERTIES = ["energy", "forces"]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._emt = None

    @property
    def calculator(self):
        if self._emt is None:
            self._emt = EMT()
        return self._emt


@pytest.fixture
def perturbed_fcc_ni(fcc_ni: Structure) -> Structure:
    """FCC Ni with one atom displaced off its lattice site, so unconstrained relaxation moves it."""
    perturbed = fcc_ni.copy()
    perturbed.translate_sites(indices=[0], vector=[0.2, 0.0, 0.0], frac_coords=False, to_unit_cell=True)
    return perturbed


@pytest.fixture
def strained_fcc_ni(fcc_ni: Structure) -> Structure:
    """FCC Ni with its cell volume inflated well past equilibrium, so cell relaxation has visible work to do."""
    strained = fcc_ni.copy()
    strained.scale_lattice(strained.volume * 1.2)
    return strained


def test_available_properties_defined() -> None:
    """AVAILABLE_PROPERTIES must be a non-empty list."""
    assert isinstance(RandomCalculator.AVAILABLE_PROPERTIES, list)
    assert len(RandomCalculator.AVAILABLE_PROPERTIES) > 0


def test_calculate_returns_required_keys(calc: RandomCalculator, bcc_fe: Structure) -> None:
    """calculate() returns a dict containing all AVAILABLE_PROPERTIES."""
    result = calc.calculate(bcc_fe)
    for key in RandomCalculator.AVAILABLE_PROPERTIES:
        assert key in result


def test_calculate_energy_is_float(calc: RandomCalculator, bcc_fe: Structure) -> None:
    """Energy returned by calculate() is a float."""
    result = calc.calculate(bcc_fe)
    assert isinstance(result["energy"], float)


def test_relax_returns_final_structure(calc: RandomCalculator, bcc_fe: Structure) -> None:
    """relax() returns a dict with 'final_structure' key."""
    result = calc.relax(bcc_fe)
    assert "final_structure" in result


def test_default_fmax() -> None:
    """Default fmax is 0.1 eV/A."""
    calc = RandomCalculator()
    assert calc.fmax == pytest.approx(0.1)


def test_custom_fmax() -> None:
    """Custom fmax is stored on the instance."""
    calc = RandomCalculator(fmax=0.05)
    assert calc.fmax == pytest.approx(0.05)


def test_calculate_accepts_ase_atoms(ase_bcc_fe) -> None:
    """calculate() accepts ase.Atoms as well as pymatgen Structure."""
    result = RandomCalculator().calculate(ase_bcc_fe)
    for key in RandomCalculator.AVAILABLE_PROPERTIES:
        assert key in result


def test_relax_accepts_ase_atoms(ase_bcc_fe) -> None:
    """relax() accepts ase.Atoms and returns 'final_structure'."""
    result = RandomCalculator().relax(ase_bcc_fe)
    assert "final_structure" in result


def test_calculate_returns_exact_key_set(perturbed_fcc_ni: Structure) -> None:
    """calculate() returns exactly {'final_structure', *AVAILABLE_PROPERTIES}, nothing more or less."""
    result = _EMTCalculator().calculate(perturbed_fcc_ni)
    assert set(result) == {"final_structure", "energy", "forces"}


def test_relax_returns_trajectory_observer_and_converged_flag(perturbed_fcc_ni: Structure) -> None:
    """relax() returns a TrajectoryObserver and sets a boolean `converged` flag on the instance."""
    from materialsframework.tools.trajectory import TrajectoryObserver

    calc = _EMTCalculator(relax_cell=False, fmax=1.0, steps=5)
    result = calc.relax(perturbed_fcc_ni)

    assert isinstance(result["trajectory"], TrajectoryObserver)
    assert isinstance(calc.converged, bool)


def test_relax_with_fix_atoms_freezes_positions(perturbed_fcc_ni: Structure) -> None:
    """fix_atoms=True prevents the optimizer from moving any atom off its input position."""
    calc = _EMTCalculator(fix_atoms=True, relax_cell=False, fmax=1.0, steps=5)
    result = calc.relax(perturbed_fcc_ni)

    assert result["final_structure"].cart_coords == pytest.approx(perturbed_fcc_ni.cart_coords)


def test_relax_without_fix_atoms_moves_positions(perturbed_fcc_ni: Structure) -> None:
    """Sanity check for the test above: without fix_atoms, the displaced atom actually relaxes back."""
    calc = _EMTCalculator(relax_cell=False, fmax=1.0, steps=5)
    result = calc.relax(perturbed_fcc_ni)

    assert result["final_structure"].cart_coords != pytest.approx(perturbed_fcc_ni.cart_coords)


def test_relax_applies_fix_symmetry_constraint_when_enabled(monkeypatch, perturbed_fcc_ni: Structure) -> None:
    """relax() constructs a FixSymmetry constraint, with the configured symprec, when fix_symmetry=True."""
    calls = []
    real_fix_symmetry = calculator_module.FixSymmetry

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_fix_symmetry(*args, **kwargs)

    monkeypatch.setattr(calculator_module, "FixSymmetry", spy)

    _EMTCalculator(fix_symmetry=True, symprec=0.05, relax_cell=False, fmax=1.0, steps=2).relax(perturbed_fcc_ni)

    assert len(calls) == 1
    assert calls[0]["symprec"] == pytest.approx(0.05)


def test_relax_skips_fix_symmetry_constraint_by_default(monkeypatch, perturbed_fcc_ni: Structure) -> None:
    """relax() never constructs a FixSymmetry constraint when fix_symmetry=False (the default)."""
    calls = []
    monkeypatch.setattr(calculator_module, "FixSymmetry", lambda *a, **k: calls.append((a, k)))

    _EMTCalculator(relax_cell=False, fmax=1.0, steps=2).relax(perturbed_fcc_ni)

    assert calls == []


def test_relax_applies_fix_atoms_constraint_with_full_mask(monkeypatch, perturbed_fcc_ni: Structure) -> None:
    """relax() constructs a FixAtoms constraint masking every atom when fix_atoms=True."""
    calls = []
    real_fix_atoms = calculator_module.FixAtoms

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_fix_atoms(*args, **kwargs)

    monkeypatch.setattr(calculator_module, "FixAtoms", spy)

    _EMTCalculator(fix_atoms=True, relax_cell=False, fmax=1.0, steps=2).relax(perturbed_fcc_ni)

    assert len(calls) == 1
    assert calls[0]["mask"] == [True] * len(perturbed_fcc_ni)


def _spy_frechetcellfilter_subclass(calls: list) -> type:
    """Builds a FrechetCellFilter subclass that records its constructor kwargs.

    A plain function/lambda can't stand in for `FrechetCellFilter` here: `relax()` also does
    `isinstance(atoms, FrechetCellFilter)` after the optimizer runs, which requires the patched
    name to still be a real type (and, for that check to behave correctly, a type real
    `FrechetCellFilter` instances actually are).
    """
    real_filter = calculator_module.FrechetCellFilter

    class _Spy(real_filter):
        def __init__(self, *args, **kwargs) -> None:
            calls.append(kwargs)
            super().__init__(*args, **kwargs)

    return _Spy


def test_relax_wraps_atoms_in_frechetcellfilter_forwarding_params(monkeypatch, perturbed_fcc_ni: Structure) -> None:
    """relax() wraps atoms in FrechetCellFilter, forwarding hydrostatic_strain and params_asecellfilter, when relax_cell=True."""
    calls = []
    monkeypatch.setattr(calculator_module, "FrechetCellFilter", _spy_frechetcellfilter_subclass(calls))

    _EMTCalculator(
        relax_cell=True,
        hydrostatic_strain=True,
        params_asecellfilter={"constant_volume": True},
        fmax=1.0,
        steps=2,
    ).relax(perturbed_fcc_ni)

    assert len(calls) == 1
    assert calls[0]["hydrostatic_strain"] is True
    assert calls[0]["constant_volume"] is True


def test_relax_skips_frechetcellfilter_when_relax_cell_false(monkeypatch, perturbed_fcc_ni: Structure) -> None:
    """relax() never wraps atoms in FrechetCellFilter when relax_cell=False."""
    calls = []
    monkeypatch.setattr(calculator_module, "FrechetCellFilter", _spy_frechetcellfilter_subclass(calls))

    _EMTCalculator(relax_cell=False, fmax=1.0, steps=2).relax(perturbed_fcc_ni)

    assert calls == []


def test_relax_cell_changes_when_relax_cell_true(strained_fcc_ni: Structure) -> None:
    """relax_cell=True lets the optimizer shrink a deliberately over-inflated cell back down."""
    calc = _EMTCalculator(relax_cell=True, fmax=1.0, steps=15)
    result = calc.relax(strained_fcc_ni)

    assert result["final_structure"].volume < strained_fcc_ni.volume * 0.99


def test_relax_cell_frozen_when_relax_cell_false(strained_fcc_ni: Structure) -> None:
    """relax_cell=False leaves the cell exactly as given, even when it's far from equilibrium."""
    calc = _EMTCalculator(relax_cell=False, fmax=1.0, steps=15)
    result = calc.relax(strained_fcc_ni)

    assert result["final_structure"].volume == pytest.approx(strained_fcc_ni.volume)


@pytest.mark.parametrize("name", [member.name for member in OPTIMIZERS])
def test_optimizer_string_dispatches_to_expected_class(name: str) -> None:
    """Every OPTIMIZERS member name, passed as a string, resolves `.optimizer` to the matching ASE class."""
    calc = _EMTCalculator(optimizer=name)
    assert calc.optimizer is OPTIMIZERS[name].value


def test_optimizer_accepts_class_directly() -> None:
    """Passing an Optimizer subclass directly (not a string) is stored as-is."""
    calc = _EMTCalculator(optimizer=BFGS)
    assert calc.optimizer is BFGS


@pytest.mark.parametrize("name", [member.name for member in OPTIMIZERS])
def test_relax_runs_end_to_end_for_every_optimizer(name: str, perturbed_fcc_ni: Structure) -> None:
    """relax() completes and returns valid results for every supported optimizer, not just the FIRE default."""
    calc = _EMTCalculator(optimizer=name, relax_cell=False, fmax=1.0, steps=3)
    result = calc.relax(perturbed_fcc_ni)

    assert "final_structure" in result
    assert isinstance(calc.converged, bool)
