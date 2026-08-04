"""Tests for FormationEnergyAnalyzer."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from materialsframework.analysis.formation_energy import FormationEnergyAnalyzer
from materialsframework.transformations.formation_energy import FormationEnergyTransformation


@pytest.fixture(scope="module")
def analyzer(calc):
    """FormationEnergyAnalyzer with CHGNet calculator."""
    return FormationEnergyAnalyzer(calculator=calc)


@pytest.fixture(scope="module")
def result(analyzer, l10_feni):
    """Single formation energy calculation shared by all result-checking tests."""
    return analyzer.calculate(l10_feni, is_relaxed=True)


def test_default_params() -> None:
    """Analyzer initialises with no calculator and no transformation."""
    analyzer = FormationEnergyAnalyzer()
    assert analyzer._calculator is None
    assert analyzer._formation_energy_transformation is None


def test_formation_energy_transformation_lazy_property() -> None:
    """Accessing .formation_energy_transformation creates a FormationEnergyTransformation."""
    analyzer = FormationEnergyAnalyzer()
    assert isinstance(analyzer.formation_energy_transformation, FormationEnergyTransformation)


def test_calculate_raises_without_energy_property(l10_feni) -> None:
    """calculate() raises if the calculator lacks the 'energy' property, before doing any real work."""
    from materialsframework.calculators.random import RandomCalculator

    class _NoEnergyCalculator(RandomCalculator):
        AVAILABLE_PROPERTIES = ["forces"]

    analyzer = FormationEnergyAnalyzer(calculator=_NoEnergyCalculator())
    with pytest.raises(ValueError, match="'energy'"):
        analyzer.calculate(l10_feni, is_relaxed=True)


def test_calculate_returns_key_with_random_calc(l10_feni) -> None:
    """calculate() returns formation_energy key without any ML dependency."""
    from materialsframework.calculators.random import RandomCalculator

    analyzer = FormationEnergyAnalyzer(calculator=RandomCalculator())
    result = analyzer.calculate(l10_feni, is_relaxed=True)
    assert "formation_energy" in result
    assert isinstance(result["formation_energy"], (float, np.floating))


def test_calculate_accepts_ase_atoms(ase_l10_feni) -> None:
    """calculate() accepts ase.Atoms in addition to pymatgen Structure."""
    from materialsframework.calculators.random import RandomCalculator

    analyzer = FormationEnergyAnalyzer(calculator=RandomCalculator())
    result = analyzer.calculate(ase_l10_feni, is_relaxed=True)
    assert "formation_energy" in result


def test_calculate_relaxes_compound_when_not_already_relaxed(l10_feni) -> None:
    """calculate() relaxes the compound structure itself when is_relaxed is False (the default)."""
    from materialsframework.calculators.random import RandomCalculator

    analyzer = FormationEnergyAnalyzer(calculator=RandomCalculator())
    result = analyzer.calculate(l10_feni)
    assert "formation_energy" in result
    assert isinstance(result["formation_energy"], (float, np.floating))


@pytest.mark.integration
def test_single_element_raises(analyzer, bcc_fe) -> None:
    """calculate() raises ValueError when the structure has only one element type."""
    with pytest.raises(ValueError, match="at least two different elements"):
        analyzer.calculate(bcc_fe, is_relaxed=True)


def test_single_element_with_mixed_oxidation_states_raises() -> None:
    """calculate() still raises ValueError when oxidation-state-decorated species mask a single element."""
    from materialsframework.calculators.random import RandomCalculator

    structure = Structure(Lattice.cubic(3.6), ["Fe2+", "Fe3+"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    analyzer = FormationEnergyAnalyzer(calculator=RandomCalculator())
    with pytest.raises(ValueError, match="at least two different elements"):
        analyzer.calculate(structure, is_relaxed=True)


@pytest.mark.integration
def test_calculate_returns_formation_energy(result) -> None:
    """calculate() returns a dict with a float 'formation_energy' key."""
    assert "formation_energy" in result
    assert isinstance(result["formation_energy"], (float, np.floating))


@pytest.mark.integration
def test_formation_energy_is_negative_for_feni(result) -> None:
    """FeNi L1_0 should have a negative formation energy (stable intermetallic)."""
    assert result["formation_energy"] < 0


def test_elemental_references_reports_known_ground_states(l10_feni) -> None:
    """Fe and Ni both have a tabulated ground state, so neither reference is flagged as guessed."""
    from materialsframework.calculators.random import RandomCalculator

    analyzer = FormationEnergyAnalyzer(calculator=RandomCalculator())
    result = analyzer.calculate(l10_feni, is_relaxed=True)
    assert set(result["elemental_references"]) == {"Fe", "Ni"}
    for element, reference in result["elemental_references"].items():
        assert isinstance(reference["energy_per_atom"], (float, np.floating))
        assert reference["is_guessed"] is False
        assert isinstance(reference["structure"], Structure)
        assert reference["structure"].elements[0].symbol == element


def test_elemental_references_flags_guessed_fallback() -> None:
    """An element with no tabulated ground state (Mn) is flagged as guessed."""
    from materialsframework.calculators.random import RandomCalculator

    structure = Structure(Lattice.cubic(3.6), ["Mn", "Ni"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    analyzer = FormationEnergyAnalyzer(calculator=RandomCalculator())
    result = analyzer.calculate(structure, is_relaxed=True)
    assert result["elemental_references"]["Mn"]["is_guessed"] is True
    assert result["elemental_references"]["Ni"]["is_guessed"] is False
    for element, reference in result["elemental_references"].items():
        assert isinstance(reference["structure"], Structure)
        assert reference["structure"].elements[0].symbol == element


def test_calculate_handles_ternary_compound() -> None:
    """calculate() works end-to-end for a compound with three distinct elements."""
    from materialsframework.calculators.random import RandomCalculator

    structure = Structure(
        Lattice.cubic(3.6),
        ["Fe", "Ni", "Cr", "Fe"],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )
    analyzer = FormationEnergyAnalyzer(calculator=RandomCalculator())
    result = analyzer.calculate(structure, is_relaxed=True)
    assert set(result["elemental_references"]) == {"Fe", "Ni", "Cr"}
    assert isinstance(result["formation_energy"], (float, np.floating))


def test_calculate_caches_pure_element_references_across_calls(l10_feni) -> None:
    """A second calculate() call reuses the same cached pure-element reference objects."""
    from materialsframework.calculators.random import RandomCalculator

    analyzer = FormationEnergyAnalyzer(calculator=RandomCalculator())
    first = analyzer.calculate(l10_feni, is_relaxed=True)
    second = analyzer.calculate(l10_feni, is_relaxed=True)

    for element, reference in first["elemental_references"].items():
        assert second["elemental_references"][element] is reference


def test_calculate_caches_pure_element_references_across_different_structures(l10_feni) -> None:
    """A shared element's cached reference is reused across calls on different compounds."""
    from materialsframework.calculators.random import RandomCalculator

    other_structure = Structure(Lattice.cubic(2.87), ["Fe", "Cr"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    analyzer = FormationEnergyAnalyzer(calculator=RandomCalculator())
    first = analyzer.calculate(l10_feni, is_relaxed=True)
    second = analyzer.calculate(other_structure, is_relaxed=True)

    assert second["elemental_references"]["Fe"] is first["elemental_references"]["Fe"]


def test_calculate_uses_custom_transformation(l10_feni) -> None:
    """calculate() uses the transformation instance passed to __init__, not a fresh default one."""
    from materialsframework.calculators.random import RandomCalculator

    class RecordingTransformation(FormationEnergyTransformation):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def apply_transformation(self, structure):
            self.calls += 1
            return super().apply_transformation(structure)

    transformation = RecordingTransformation()
    analyzer = FormationEnergyAnalyzer(calculator=RandomCalculator(), formation_energy_transformation=transformation)
    analyzer.calculate(l10_feni, is_relaxed=True)

    assert analyzer.formation_energy_transformation is transformation
    assert transformation.calls == 1
