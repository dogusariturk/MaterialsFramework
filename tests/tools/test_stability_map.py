"""Tests for StabilityMap and CoherentStabilityMap static/pure helpers, plus one end-to-end check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.constants import Avogadro

pytest.importorskip("pycalphad")

pytestmark = pytest.mark.integration

from materialsframework.tools.stability_map import CoherentStabilityMap, StabilityMap


def test_generate_compositions_sums_to_one() -> None:
    """All generated compositions sum to 1.0."""
    comps = StabilityMap._generate_compositions(["A", "B", "C"], step=0.5)
    assert not comps.empty
    for _, row in comps.iterrows():
        assert abs(row.sum() - 1.0) < 1e-6


def test_generate_compositions_excludes_pure_components() -> None:
    """No row has a component value of exactly 1.0."""
    comps = StabilityMap._generate_compositions(["A", "B"], step=0.1)
    for _, row in comps.iterrows():
        assert not any(val == 1.0 for val in row)


def test_generate_compositions_columns() -> None:
    """DataFrame columns match the provided element list."""
    elements = ["Co", "Cr", "Fe", "Ni"]
    comps = StabilityMap._generate_compositions(elements, step=0.5)
    assert list(comps.columns) == elements


def test_determine_section_no_negatives() -> None:
    """_determine_section returns 0 when all eigenvalues are positive."""
    row = pd.Series({"eigenvalue_1": 1.0, "eigenvalue_2": 2.0, "other": 0.5})
    assert StabilityMap._determine_section(row) == 0


def test_determine_section_one_negative() -> None:
    """_determine_section returns 1 when exactly one eigenvalue is negative."""
    row = pd.Series({"eigenvalue_1": -1.0, "eigenvalue_2": 2.0})
    assert StabilityMap._determine_section(row) == 1


def test_determine_section_all_negative() -> None:
    """_determine_section counts all negative eigenvalues."""
    row = pd.Series({"eigenvalue_1": -1.0, "eigenvalue_2": -0.5, "eigenvalue_3": -0.1})
    assert StabilityMap._determine_section(row) == 3


def test_orthogonalization_matrix_shape() -> None:
    """ORTHOGONALIZATION class attribute has the expected (9, 9) shape."""
    mat = StabilityMap.ORTHOGONALIZATION
    assert mat.shape == (9, 9)


def test_zener_modulus_positive_anisotropy() -> None:
    """_zener_modulus uses the first closed-form expression when anisotropy is positive."""
    c11, c12, c44 = 200e9, 100e9, 80e9
    assert 2 * c44 - c11 + c12 > 0
    assert CoherentStabilityMap._zener_modulus(c11, c12, c44) == pytest.approx(200e9)


def test_zener_modulus_negative_anisotropy() -> None:
    """_zener_modulus uses the second closed-form expression when anisotropy is negative."""
    c11, c12, c44 = 200e9, 100e9, 10e9
    assert 2 * c44 - c11 + c12 < 0
    assert CoherentStabilityMap._zener_modulus(c11, c12, c44) == pytest.approx(2.4e22 / 4.4e11)


def test_zener_modulus_zero_anisotropy_does_not_return_none() -> None:
    """At the anisotropy boundary (== 0), both closed-form expressions agree and neither is skipped."""
    c11, c12, c44 = 200e9, 0.0, 100e9
    assert 2 * c44 - c11 + c12 == 0
    assert CoherentStabilityMap._zener_modulus(c11, c12, c44) == pytest.approx(200e9)


def test_coherent_hessian_outer_product() -> None:
    """_coherent_hessian returns 2 * y * molar_volume * outer(eta, eta)."""
    eta = np.array([[1.0, 2.0]])
    result = CoherentStabilityMap._coherent_hessian(eta, y=3.0, molar_volume=4.0)
    assert np.allclose(result, [[24.0, 48.0], [48.0, 96.0]])


def test_molar_volume_from_lattice_parameter() -> None:
    """_molar_volume_from_lattice_parameter converts Angstrom^3/cell to m^3/mol using Avogadro's number."""
    lattice_parameter = 2.0
    expected = (lattice_parameter * 1e-10) ** 3 * Avogadro / 2
    assert CoherentStabilityMap._molar_volume_from_lattice_parameter(lattice_parameter, atoms_per_cell=2) == pytest.approx(expected)


def test_count_negative_no_negatives() -> None:
    """_count_negative returns 0 when all matching columns are positive."""
    row = pd.Series({"chem_eigenvalue_1": 1.0, "chem_eigenvalue_2": 2.0, "coherent_eigenvalue_1": 3.0})
    assert CoherentStabilityMap._count_negative(row, "chem_eigenvalue") == 0


def test_count_negative_one_negative() -> None:
    """_count_negative only counts columns matching the given prefix."""
    row = pd.Series({"chem_eigenvalue_1": -1.0, "chem_eigenvalue_2": 2.0, "coherent_eigenvalue_1": -5.0})
    assert CoherentStabilityMap._count_negative(row, "chem_eigenvalue") == 1


def test_count_negative_all_negative() -> None:
    """_count_negative counts every negative value among matching columns."""
    row = pd.Series({"coherent_eigenvalue_1": -1.0, "coherent_eigenvalue_2": -0.5, "coherent_eigenvalue_3": -0.1})
    assert CoherentStabilityMap._count_negative(row, "coherent_eigenvalue") == 3


def _bundled_bcc_tdb_path() -> str:
    """Path to the small Cr-Fe-Ni BCC_A2 database bundled with pycalphad's own test suite."""
    pycalphad_tests = pytest.importorskip("pycalphad.tests")
    path = Path(pycalphad_tests.__file__).parent / "databases" / "Cr-Fe-Ni_shallow_bcc.tdb"
    if not path.exists():
        pytest.skip("pycalphad's bundled test databases are not available in this install")
    return str(path)


@pytest.mark.slow
def test_coherent_stability_map_process_row_end_to_end() -> None:
    """_process_row runs the full SQS + relax + elastic-constants + CALPHAD pipeline for one composition.

    This exercises every piece CoherentStabilityMap adds over StabilityMap: SQS generation,
    relaxation, elastic-constant fitting, the finite-difference lattice-parameter gradient, and
    the coherent-elastic Hessian correction. It's slow (SQS optimization plus several relaxations
    and elastic-constant deformations with a real MLIP calculator), so it's excluded from the
    default test run.
    """
    pytest.importorskip("sqsgenerator")
    pytest.importorskip("chgnet")
    from materialsframework.calculators.chgnet import CHGNetCalculator

    stability_map = CoherentStabilityMap(
        db=_bundled_bcc_tdb_path(),
        elements=["CR", "FE", "NI"],
        phase="BCC_A2",
        step=0.5,
        temperature=1000,
        calculator=CHGNetCalculator(device="cpu"),
        supercell_size=(3, 3, 3),
        finite_diff_step=0.05,
    )
    row = pd.Series({"CR": 0.0, "FE": 0.5, "NI": 0.5})
    result = stability_map._process_row(row)

    assert len(result) == 4
    assert all(value is not None for value in result)
    chem_eigenvalues, coherent_eigenvalues = result[:2], result[2:]
    assert chem_eigenvalues != coherent_eigenvalues
