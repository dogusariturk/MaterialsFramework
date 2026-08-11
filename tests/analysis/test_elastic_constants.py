"""Tests for ElasticConstantsAnalyzer."""

from __future__ import annotations

import math

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.cell import Cell

from materialsframework.analysis.elastic_constants import ElasticConstantsAnalyzer
from materialsframework.tools import elastic
from materialsframework.tools.calculator import BaseCalculator
from materialsframework.tools.elastic import get_lattice_type
from materialsframework.transformations.elastic_constants import ElasticConstantsDeformationTransformation


def _make_atoms(a, b, c, alpha, beta, gamma) -> Atoms:
    atoms = Atoms("H", positions=[[0, 0, 0]])
    atoms.cell = Cell.fromcellpar([a, b, c, alpha, beta, gamma])
    atoms.pbc = True
    return atoms


class _RecordingCalculator(BaseCalculator):
    """Minimal calculator that records the settings used for each relaxation."""

    AVAILABLE_PROPERTIES = ["energy", "stress", "forces"]

    def __init__(self) -> None:
        super().__init__(fmax=0.1, relax_cell=True)
        self.relax_calls: list[tuple[bool, float]] = []
        self._ase_calculator = Calculator()

    @property
    def calculator(self) -> Calculator:
        """Return the stand-in ASE calculator."""
        return self._ase_calculator

    def relax(self, structure, **kwargs):
        """Return the input unchanged after recording the active relaxation settings."""
        self.relax_calls.append((self.relax_cell, self.fmax))
        return {"final_structure": structure.copy()}


class _TwoDistortionTransformation(ElasticConstantsDeformationTransformation):
    """Return two copies so relaxation behavior can be tested without real deformations."""

    def apply_transformation(self, structure):
        """Return two stand-in distorted structures."""
        return [structure.copy(), structure.copy()]


class _ElasticTensorStub:
    """Deterministic elastic properties used to test the newly returned values."""

    y_mod = 210e9
    k_voigt = 160.0
    g_voigt = 80.0
    k_reuss = 140.0
    g_reuss = 70.0
    k_vrh = 150.0
    g_vrh = 75.0
    homogeneous_poisson = 0.25

    @staticmethod
    def debye_temperature(structure) -> float:
        """Return a deterministic Debye temperature."""
        return 450.0

    @staticmethod
    def long_v(structure) -> float:
        """Return a deterministic longitudinal sound velocity."""
        return 6000.0

    @staticmethod
    def trans_v(structure) -> float:
        """Return a deterministic transverse sound velocity."""
        return 3000.0


def _stubbed_analyzer(monkeypatch) -> tuple[ElasticConstantsAnalyzer, _RecordingCalculator]:
    """Build an analyzer whose elastic fit is deterministic and inexpensive."""
    calculator = _RecordingCalculator()
    analyzer = ElasticConstantsAnalyzer(
        calculator=calculator,
        elastic_constant_transformation=_TwoDistortionTransformation(),
    )
    tensor = _ElasticTensorStub()

    monkeypatch.setattr(elastic, "get_cij_order", lambda structure: ("C_11", "C_12", "C_44"))
    monkeypatch.setattr(elastic, "get_elastic_tensor", lambda **kwargs: ([1.0, 0.5, 0.25], None))
    monkeypatch.setattr(analyzer, "_build_elastic_tensor", lambda *args: tensor)

    return analyzer, calculator


@pytest.mark.parametrize(
    "atoms, expected_brav",
    [
        (_make_atoms(2.83, 2.83, 2.83, 90, 90, 90), "Cubic"),
        (_make_atoms(3.25, 3.25, 4.95, 90, 90, 90), "Tetragonal"),
        (_make_atoms(3.21, 3.21, 5.21, 90, 90, 120), "Hexagonal"),
        (_make_atoms(3.0, 4.0, 5.0, 90, 90, 90), "Orthorombic"),
        (_make_atoms(4.0, 4.0, 4.0, 70, 70, 70), "Trigonal"),
        (_make_atoms(3.0, 4.0, 5.0, 90, 110, 90), "Monoclinic"),
        (_make_atoms(3.0, 4.0, 5.0, 80, 90, 100), "Triclinic"),
    ],
)
def test_get_lattice_type(atoms, expected_brav) -> None:
    """get_lattice_type correctly classifies all seven crystal systems."""
    _, brav, _, _ = get_lattice_type(atoms)
    assert brav == expected_brav


@pytest.fixture(scope="module")
def analyzer(calc):
    """ElasticConstantsAnalyzer with reduced deformations for fast tests."""
    return ElasticConstantsAnalyzer(num_deform=3, max_deform=1.0, calculator=calc)


@pytest.fixture(scope="module")
def result(analyzer, fcc_ni):
    """Single stable elastic constant calculation shared by all result-checking tests."""
    return analyzer.calculate(fcc_ni, is_relaxed=True)


def test_default_params() -> None:
    """Analyzer stores the correct default parameters."""
    analyzer = ElasticConstantsAnalyzer()
    assert analyzer.num_deform == 5
    assert analyzer.max_deform == pytest.approx(2)
    assert analyzer._calculator is None
    assert analyzer._elastic_constant_transformation is None


def test_elastic_constants_transformation_lazy_property() -> None:
    """Accessing .elastic_constants_transformation creates an ElasticConstantsDeformationTransformation."""
    analyzer = ElasticConstantsAnalyzer()
    assert isinstance(analyzer.elastic_constants_transformation, ElasticConstantsDeformationTransformation)


def test_build_cubic_elastic_tensor_preserves_fitted_constants() -> None:
    """Cubic symmetry copies each fitted constant to its equivalent tensor entries."""
    analyzer = ElasticConstantsAnalyzer()
    structure = _make_atoms(2.83, 2.83, 2.83, 90, 90, 90)

    tensor = analyzer._build_elastic_tensor([300.0, 120.0, 90.0], ["C_11", "C_12", "C_44"], structure)

    expected = np.array(
        [
            [300, 120, 120, 0, 0, 0],
            [120, 300, 120, 0, 0, 0],
            [120, 120, 300, 0, 0, 0],
            [0, 0, 0, 90, 0, 0],
            [0, 0, 0, 0, 90, 0],
            [0, 0, 0, 0, 0, 90],
        ]
    )
    np.testing.assert_allclose(tensor.voigt, expected)
    assert tensor.k_vrh == pytest.approx(180.0)
    assert tensor.g_vrh == pytest.approx(90.0)


def test_build_tetragonal_elastic_tensor_preserves_fitted_constants() -> None:
    """Tetragonal symmetry copies equivalent constants while preserving independent C66."""
    analyzer = ElasticConstantsAnalyzer()
    structure = _make_atoms(3.25, 3.25, 4.95, 90, 90, 90)

    tensor = analyzer._build_elastic_tensor(
        [200.0, 250.0, 80.0, 70.0, 60.0, 50.0],
        ["C_11", "C_33", "C_12", "C_13", "C_44", "C_66"],
        structure,
    )

    expected = np.array(
        [
            [200, 80, 70, 0, 0, 0],
            [80, 200, 70, 0, 0, 0],
            [70, 70, 250, 0, 0, 0],
            [0, 0, 0, 60, 0, 0],
            [0, 0, 0, 0, 60, 0],
            [0, 0, 0, 0, 0, 50],
        ]
    )
    np.testing.assert_allclose(tensor.voigt, expected)


def test_build_hexagonal_elastic_tensor_derives_c66() -> None:
    """Hexagonal symmetry derives C66 from C11 and C12 after copying equivalent constants."""
    analyzer = ElasticConstantsAnalyzer()
    structure = _make_atoms(3.21, 3.21, 5.21, 90, 90, 120)

    tensor = analyzer._build_elastic_tensor(
        [200.0, 250.0, 80.0, 70.0, 60.0],
        ["C_11", "C_33", "C_12", "C_13", "C_44"],
        structure,
    )

    expected = np.array(
        [
            [200, 80, 70, 0, 0, 0],
            [80, 200, 70, 0, 0, 0],
            [70, 70, 250, 0, 0, 0],
            [0, 0, 0, 60, 0, 0],
            [0, 0, 0, 0, 60, 0],
            [0, 0, 0, 0, 0, 60],
        ]
    )
    np.testing.assert_allclose(tensor.voigt, expected)


def test_calculate_raises_without_required_properties(bcc_fe) -> None:
    """calculate() raises before running when required properties are unavailable."""
    from materialsframework.calculators.random import RandomCalculator

    class _NoEnergyOrStressCalculator(RandomCalculator):
        AVAILABLE_PROPERTIES = ["forces"]

    analyzer = ElasticConstantsAnalyzer(calculator=_NoEnergyOrStressCalculator())
    with pytest.raises(ValueError, match="'energy' and 'stress' and 'forces'"):
        analyzer.calculate(bcc_fe)


def test_relax_ions_true_relaxes_distortions(bcc_fe, monkeypatch) -> None:
    """relax_ions=True relaxes each distorted structure, honors fmax_distort, and restores settings after."""
    analyzer, calculator = _stubbed_analyzer(monkeypatch)

    analyzer.calculate(bcc_fe, relax_ions=True, fmax_distort=0.02)

    assert calculator.relax_calls == [(True, 0.1), (False, 0.02), (False, 0.02)]
    assert calculator.relax_cell is True
    assert calculator.fmax == pytest.approx(0.1)


def test_relax_ions_defaults_to_false(bcc_fe, monkeypatch) -> None:
    """The default relaxes the input structure but not the distortions."""
    analyzer, calculator = _stubbed_analyzer(monkeypatch)

    analyzer.calculate(bcc_fe, fmax_distort=0.02)

    assert calculator.relax_calls == [(True, 0.1)]
    assert calculator.relax_cell is True
    assert calculator.fmax == pytest.approx(0.1)


def test_calculate_restores_settings_after_error(bcc_fe, monkeypatch) -> None:
    """Temporary relax_cell and fmax_distort values do not leak when elastic fitting raises."""
    analyzer, calculator = _stubbed_analyzer(monkeypatch)

    def raise_fit_error(**kwargs):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(elastic, "get_elastic_tensor", raise_fit_error)

    with pytest.raises(RuntimeError, match="fit failed"):
        analyzer.calculate(bcc_fe, is_relaxed=True, fmax_distort=0.02)

    assert calculator.relax_cell is True
    assert calculator.fmax == pytest.approx(0.1)


def test_calculate_returns_acoustic_properties(bcc_fe, monkeypatch) -> None:
    """The new Debye, Gruneisen, and sound-velocity results use the documented formulas and units."""
    analyzer, calculator = _stubbed_analyzer(monkeypatch)

    result = analyzer.calculate(bcc_fe, is_relaxed=True, relax_ions=False)

    assert calculator.relax_calls == []
    expected_mean_velocity = 3.0 ** (1.0 / 3.0) * (1.0 / 6000.0**3 + 2.0 / 3000.0**3) ** (-1.0 / 3.0)
    expected_gruneisen = 1.5 * (1.0 + 0.25) / (2.0 - 3.0 * 0.25)
    assert result["debye_temperature"] == pytest.approx(450.0)
    assert result["gruneisen_approx"] == pytest.approx(expected_gruneisen)
    assert result["v_longitudinal"] == pytest.approx(6000.0)
    assert result["v_transverse"] == pytest.approx(3000.0)
    assert result["v_mean"] == pytest.approx(expected_mean_velocity)


@pytest.mark.integration
def test_calculate_contains_cij_for_cubic(result) -> None:
    """For a cubic structure the result contains C_11, C_12, and C_44."""
    for key in ("C_11", "C_12", "C_44"):
        assert key in result


@pytest.mark.integration
def test_calculate_returns_moduli(result) -> None:
    """calculate() returns all expected mechanical, thermoelastic, and acoustic properties."""
    for key in (
        "youngs_modulus",
        "voigt_bulk_modulus",
        "voigt_shear_modulus",
        "reuss_bulk_modulus",
        "reuss_shear_modulus",
        "voigt_reuss_hill_bulk_modulus",
        "voigt_reuss_hill_shear_modulus",
        "poisson_ratio",
        "pugh_ratio",
        "chen_vickers_hardness",
        "debye_temperature",
        "gruneisen_approx",
        "v_longitudinal",
        "v_transverse",
        "v_mean",
    ):
        assert key in result


@pytest.mark.integration
def test_bulk_modulus_positive(result) -> None:
    """Voigt bulk modulus should be positive for FCC Ni."""
    assert result["voigt_bulk_modulus"] > 0


@pytest.mark.integration
def test_chen_vickers_hardness_matches_formula(result) -> None:
    """chen_vickers_hardness follows Hv = 2 * (k^2 * G)^0.585 - 3 for the returned Pugh ratio and G_VRH.

    The Chen model is only defined for a positive k^2 * G; for a mechanically unstable fit (negative
    shear modulus) it legitimately evaluates to NaN, so that case is asserted explicitly rather than
    compared with pytest.approx (which treats NaN as never equal by default).
    """
    base = result["pugh_ratio"] ** 2 * result["voigt_reuss_hill_shear_modulus"]
    if base < 0:
        assert math.isnan(result["chen_vickers_hardness"])
    else:
        assert result["chen_vickers_hardness"] == pytest.approx(2.0 * base**0.585 - 3.0)
