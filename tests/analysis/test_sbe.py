"""Tests for SBEAnalyzer."""

from __future__ import annotations

import numpy as np
import pytest

from materialsframework.analysis.sbe import SBEAnalyzer
from materialsframework.calculators.random import RandomCalculator
from materialsframework.transformations.sbe import SBETransformation


def _fast_kwargs() -> dict:
    """Small slab/vacuum/supercell sizes shared by tests that need SBEAnalyzer.calculate() to run fast."""
    return {"max_index": 1, "min_slab_size": 6.0, "min_vacuum_size": 8.0, "supercell_size": [2, 2, 1]}


@pytest.fixture(scope="module")
def analyzer(calc):
    """SBEAnalyzer with a real calculator and small slab/vacuum/supercell sizes for integration tests."""
    return SBEAnalyzer(calculator=calc, **_fast_kwargs())


@pytest.fixture(scope="module")
def result(analyzer, bcc_fe):
    """Single SBE calculation result shared by all result-checking integration tests."""
    return analyzer.calculate(bcc_fe, is_relaxed=True)


def test_default_params() -> None:
    """SBEAnalyzer stores the documented default values."""
    analyzer = SBEAnalyzer()
    assert analyzer.max_index == 1
    assert analyzer.min_slab_size == pytest.approx(10.0)
    assert analyzer.min_vacuum_size == pytest.approx(10.0)
    assert analyzer.height == pytest.approx(1.0)
    assert analyzer.supercell_size == [4, 4, 1]
    assert analyzer._calculator is None
    assert analyzer._sbe_transformation is None


def test_sbe_transformation_lazy_property() -> None:
    """Accessing .sbe_transformation creates an SBETransformation seeded from the analyzer's params."""
    analyzer = SBEAnalyzer(max_index=2, min_slab_size=5.0, supercell_size=[2, 2, 1])
    transformation = analyzer.sbe_transformation
    assert isinstance(transformation, SBETransformation)
    assert transformation.max_index == 2
    assert transformation.min_slab_size == pytest.approx(5.0)
    assert transformation.supercell_size == [2, 2, 1]


def test_calculate_raises_without_energy_property(bcc_fe) -> None:
    """calculate() raises if the calculator lacks the 'energy' property, before doing any real work."""

    class _NoEnergyCalculator(RandomCalculator):
        AVAILABLE_PROPERTIES = ["forces"]

    analyzer = SBEAnalyzer(calculator=_NoEnergyCalculator())
    with pytest.raises(ValueError, match="'energy'"):
        analyzer.calculate(bcc_fe)


def test_calculate_returns_expected_keys(bcc_fe) -> None:
    """calculate() returns the documented top-level keys, with flat terminations/vacancy_results/structures populated."""
    analyzer = SBEAnalyzer(calculator=RandomCalculator(), **_fast_kwargs())
    result = analyzer.calculate(bcc_fe, is_relaxed=True)

    for key in (
        "bulk_energy_per_atom",
        "best_miller_index",
        "best_surface_energy",
        "surface_energies",
        "isolated_atom_energies",
        "terminations",
        "vacancy_results",
        "avg_surface_binding_energy_by_element",
        "avg_surface_binding_energy",
        "structures",
    ):
        assert key in result

    assert any(entry["miller_index"] == result["best_miller_index"] for entry in result["surface_energies"])
    assert result["best_surface_energy"] == pytest.approx(min(entry["surface_energy"] for entry in result["surface_energies"]))
    assert "Fe" in result["isolated_atom_energies"]

    assert len(result["terminations"]) > 0
    for termination in result["terminations"]:
        for key in ("miller_index", "termination_index", "supercell_slab_energy", "avg_surface_binding_energy_by_element"):
            assert key in termination
        assert termination["miller_index"] == result["best_miller_index"]

    assert len(result["vacancy_results"]) > 0
    for vacancy in result["vacancy_results"]:
        for key in (
            "miller_index",
            "termination_index",
            "site_index",
            "element",
            "vacancy_energy",
            "surface_binding_energy",
        ):
            assert key in vacancy

    structures = result["structures"]
    assert "bulk_structure" in structures
    assert len(structures["slabs"]) == len(result["surface_energies"])
    assert len(structures["supercells"]) == len(result["terminations"])
    assert len(structures["vacancies"]) == len(result["vacancy_results"])
    for key in ("miller_index", "termination_index", "slab", "relaxed_slab"):
        assert key in structures["slabs"][0]
    for key in ("miller_index", "termination_index", "supercell_slab"):
        assert key in structures["supercells"][0]
    for key in ("miller_index", "termination_index", "site_index", "structure", "relaxed_structure"):
        assert key in structures["vacancies"][0]

    assert isinstance(result["avg_surface_binding_energy"], (float, np.floating))


def test_calculate_restores_relax_cell(bcc_fe) -> None:
    """calculate() restores the calculator's relax_cell setting afterward, instead of leaking it."""
    random_calc = RandomCalculator()
    random_calc.relax_cell = True
    analyzer = SBEAnalyzer(calculator=random_calc, **_fast_kwargs())

    analyzer.calculate(bcc_fe, is_relaxed=True)

    assert random_calc.relax_cell is True


def test_calculate_caches_isolated_atom_energy_across_calls(bcc_fe) -> None:
    """A second calculate() call reuses the cached isolated-atom energy instead of recomputing it.

    RandomCalculator returns a fresh random energy on every call, so a matching value across two
    separate calculate() calls is strong evidence the cache (not a lucky coincidence) is doing its job.
    """
    analyzer = SBEAnalyzer(calculator=RandomCalculator(), **_fast_kwargs())

    first = analyzer.calculate(bcc_fe, is_relaxed=True)
    second = analyzer.calculate(bcc_fe, is_relaxed=True)

    assert first["isolated_atom_energies"]["Fe"] == second["isolated_atom_energies"]["Fe"]


def test_calculate_no_slabs_raises(bcc_fe, monkeypatch) -> None:
    """calculate() raises a clear error if the transformation produces no slabs."""
    analyzer = SBEAnalyzer(calculator=RandomCalculator(), **_fast_kwargs())
    monkeypatch.setattr(analyzer.sbe_transformation, "apply_transformation", lambda structure: [])

    with pytest.raises(ValueError, match="No slabs were generated"):
        analyzer.calculate(bcc_fe, is_relaxed=True)


@pytest.mark.integration
def test_avg_surface_binding_energy_is_finite_float(result) -> None:
    """The overall average SBE is a finite float."""
    assert isinstance(result["avg_surface_binding_energy"], (float, np.floating))
    assert np.isfinite(result["avg_surface_binding_energy"])


@pytest.mark.integration
def test_best_surface_energy_is_the_global_minimum(result) -> None:
    """best_surface_energy is the minimum surface_energy across every screened termination."""
    all_gammas = [entry["surface_energy"] for entry in result["surface_energies"]]

    assert result["best_surface_energy"] == pytest.approx(min(all_gammas))


@pytest.mark.integration
def test_best_miller_index_owns_the_best_surface_energy(result) -> None:
    """best_miller_index has at least one termination whose surface_energy equals best_surface_energy."""
    best_gammas = [
        entry["surface_energy"] for entry in result["surface_energies"] if entry["miller_index"] == result["best_miller_index"]
    ]

    assert min(best_gammas) == pytest.approx(result["best_surface_energy"])


@pytest.mark.integration
def test_vacancy_results_join_to_structures_by_key(result) -> None:
    """Every vacancy_results entry has a matching structures['vacancies'] entry with the same join keys."""
    structure_keys = {
        (entry["miller_index"], entry["termination_index"], entry["site_index"]) for entry in result["structures"]["vacancies"]
    }
    for vacancy in result["vacancy_results"]:
        assert (vacancy["miller_index"], vacancy["termination_index"], vacancy["site_index"]) in structure_keys
