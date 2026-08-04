"""Tests for NEBAnalyzer."""

from __future__ import annotations

import numpy as np
import pytest
from ase.calculators.emt import EMT
from ase.mep import NEB

from materialsframework.analysis.neb import NEBAnalyzer
from materialsframework.tools.calculator import BaseCalculator
from materialsframework.transformations.neb import NEBTransformation

_N_IMAGES = 2


class _EMTCalculator(BaseCalculator):
    """Minimal concrete `BaseCalculator` backed by ASE's dependency-free EMT calculator.

    Caches a single `EMT` instance, mirroring how the framework's real MLIP calculators share one
    calculator instance across all NEB images.
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
def ni_endpoints(fcc_ni):
    """Endpoint structures for a small, single-atom hop in FCC Ni (an EMT-supported element)."""
    final = fcc_ni.copy()
    final.translate_sites(indices=[0], vector=[0.1, 0.0, 0.0], frac_coords=False, to_unit_cell=True)
    return fcc_ni, final


def test_default_params() -> None:
    """Analyzer stores the documented default values."""
    analyzer = NEBAnalyzer()
    assert analyzer.spring_constant == pytest.approx(0.1)
    assert analyzer.climb is False
    assert analyzer.remove_rotation_and_translation is False
    assert analyzer.method == "improvedtangent"
    assert analyzer.n_images == 5
    assert analyzer.neb is None
    assert analyzer._calculator is None
    assert analyzer._neb_transformation is None


def test_neb_transformation_lazy_property() -> None:
    """Accessing .neb_transformation creates a NEBTransformation seeded from the analyzer's params."""
    analyzer = NEBAnalyzer(n_images=3, autosort_tol=0.2, pbc=False)
    transformation = analyzer.neb_transformation
    assert isinstance(transformation, NEBTransformation)
    assert transformation.n_images == 3
    assert transformation.autosort_tol == pytest.approx(0.2)
    assert transformation.pbc is False


def test_calculate_raises_without_energy_property(fcc_ni) -> None:
    """calculate() raises if the calculator lacks the 'energy' property, before doing any real work."""
    from materialsframework.calculators.random import RandomCalculator

    class _NoEnergyCalculator(RandomCalculator):
        AVAILABLE_PROPERTIES = ["forces"]

    analyzer = NEBAnalyzer(calculator=_NoEnergyCalculator())
    with pytest.raises(ValueError, match="'energy'"):
        analyzer.calculate(fcc_ni, fcc_ni)


def test_calculate_returns_expected_keys(ni_endpoints) -> None:
    """calculate() returns images/energies/barrier/reverse_barrier/reaction_energy/converged."""
    initial, final = ni_endpoints
    analyzer = NEBAnalyzer(calculator=_EMTCalculator(fmax=1.0, steps=5), n_images=_N_IMAGES)
    result = analyzer.calculate(initial, final, is_relaxed=True)

    for key in ("images", "energies", "barrier", "reverse_barrier", "reaction_energy", "converged"):
        assert key in result

    assert len(result["images"]) == _N_IMAGES + 1
    assert len(result["energies"]) == _N_IMAGES + 1
    assert all(isinstance(e, (float, np.floating)) for e in result["energies"])
    assert isinstance(result["converged"], bool)
    assert isinstance(analyzer.neb, NEB)


def test_calculate_reaction_energy_matches_endpoint_energies(ni_endpoints) -> None:
    """reaction_energy is the final image's energy minus the initial image's energy."""
    initial, final = ni_endpoints
    analyzer = NEBAnalyzer(calculator=_EMTCalculator(fmax=1.0, steps=5), n_images=_N_IMAGES)
    result = analyzer.calculate(initial, final, is_relaxed=True)

    assert result["reaction_energy"] == pytest.approx(result["energies"][-1] - result["energies"][0])


def test_calculate_barrier_is_non_negative(ni_endpoints) -> None:
    """The forward and reverse barriers are non-negative, since they're measured from the path maximum."""
    initial, final = ni_endpoints
    analyzer = NEBAnalyzer(calculator=_EMTCalculator(fmax=1.0, steps=5), n_images=_N_IMAGES)
    result = analyzer.calculate(initial, final, is_relaxed=True)

    assert result["barrier"] >= 0
    assert result["reverse_barrier"] >= 0


def test_calculate_with_climb_enables_climbing_image_on_neb(ni_endpoints) -> None:
    """climb=True enables the climbing image on `self.neb` after the first optimizer run."""
    initial, final = ni_endpoints
    analyzer = NEBAnalyzer(calculator=_EMTCalculator(fmax=1.0, steps=5), n_images=_N_IMAGES, climb=True)
    analyzer.calculate(initial, final, is_relaxed=True)

    assert analyzer.neb is not None
    assert analyzer.neb.climb is True


def test_calculate_without_climb_leaves_climbing_image_disabled(ni_endpoints) -> None:
    """climb=False (the default) never enables the climbing image."""
    initial, final = ni_endpoints
    analyzer = NEBAnalyzer(calculator=_EMTCalculator(fmax=1.0, steps=5), n_images=_N_IMAGES)
    analyzer.calculate(initial, final, is_relaxed=True)

    assert analyzer.neb is not None
    assert analyzer.neb.climb is False
