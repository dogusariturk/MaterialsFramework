"""Tests for ANNNIStackingFaultAnalyzer."""

from __future__ import annotations

import pytest

pytest.importorskip("sqsgenerator")

from materialsframework.analysis.annni import ANNNIStackingFaultAnalyzer
from materialsframework.transformations.annni import ANNNIStackingFaultTransformation


class _SmallANNNITransformation(ANNNIStackingFaultTransformation):
    """ANNNI transformation that always uses minimal supercells for fast testing.

    fcc_prim (2,2,1) gives 4 atoms; hcp (2,1,1) gives 4 atoms; dhcp (1,1,1) gives 4 atoms.
    All sizes are large enough to accommodate a binary 50/50 composition and to let
    sqsgenerator resolve the requested coordination shells.
    """

    def apply_transformation(self, composition, **kwargs):  # type: ignore[override]
        """Run the transformation with minimal supercell sizes for fast testing."""
        return super().apply_transformation(
            composition,
            fcc_supercell_size=(2, 2, 1),
            hcp_supercell_size=(2, 1, 1),
            dhcp_supercell_size=(1, 1, 1),
        )


@pytest.fixture(scope="module")
def analyzer(calc):
    """ANNNIStackingFaultAnalyzer with CHGNet and small supercell transformation."""
    return ANNNIStackingFaultAnalyzer(
        calculator=calc,
        annni_transformation=_SmallANNNITransformation(),
    )


@pytest.fixture(scope="module")
def result(analyzer):
    """Single ANNNI SFE calculation shared by all result-checking tests."""
    return analyzer.calculate("Fe0.5Co0.5")


def test_default_params() -> None:
    """Analyzer initialises with no calculator and no transformation."""
    analyzer = ANNNIStackingFaultAnalyzer()
    assert analyzer._calculator is None
    assert analyzer._annni_transformation is None


def test_annni_transformation_lazy_property() -> None:
    """Accessing .annni_transformation creates a default ANNNIStackingFaultTransformation."""
    analyzer = ANNNIStackingFaultAnalyzer()
    assert isinstance(analyzer.annni_transformation, ANNNIStackingFaultTransformation)


def test_calculate_raises_without_energy_property() -> None:
    """calculate() raises if the calculator lacks the 'energy' property, before doing any real work."""
    from materialsframework.calculators.random import RandomCalculator

    class _NoEnergyCalculator(RandomCalculator):
        AVAILABLE_PROPERTIES = ["forces"]

    analyzer = ANNNIStackingFaultAnalyzer(calculator=_NoEnergyCalculator())
    with pytest.raises(ValueError, match="'energy'"):
        analyzer.calculate("Fe0.5Co0.5")


@pytest.mark.integration
def test_calculate_returns_isfe_and_esfe(result) -> None:
    """calculate() returns a dict with 'isfe' and 'esfe' float values."""
    assert "isfe" in result
    assert "esfe" in result
    assert isinstance(result["isfe"], float)
    assert isinstance(result["esfe"], float)


@pytest.mark.integration
def test_hcp_dhcp_scaled_to_fcc_per_atom_volume(analyzer) -> None:
    """HCP and DHCP structures are scaled to the same per-atom volume as the relaxed FCC."""
    structures = analyzer.annni_transformation.apply_transformation("Fe0.5Co0.5")

    fcc_struct = structures["fcc"]
    fcc_result = analyzer.calculator.relax(fcc_struct)
    fcc_vol_per_atom = fcc_result["final_structure"].volume / fcc_result["final_structure"].num_sites

    hcp_struct = structures["hcp"]
    hcp_scaled = hcp_struct.scale_lattice(fcc_vol_per_atom * hcp_struct.num_sites)
    hcp_vol_per_atom = hcp_scaled.volume / hcp_scaled.num_sites

    dhcp_struct = structures["dhcp"]
    dhcp_scaled = dhcp_struct.scale_lattice(fcc_vol_per_atom * dhcp_struct.num_sites)
    dhcp_vol_per_atom = dhcp_scaled.volume / dhcp_scaled.num_sites

    assert hcp_vol_per_atom == pytest.approx(fcc_vol_per_atom, rel=1e-6)
    assert dhcp_vol_per_atom == pytest.approx(fcc_vol_per_atom, rel=1e-6)
