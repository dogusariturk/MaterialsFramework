"""Integration tests for ProphetCalculator.

ProphetCalculator requires a model checkpoint path.
Set the PROPHET_MODEL env var to the path of a Prophet checkpoint without spin conditioning, the
PROPHET_SPIN_MODEL env var to the path of one with spin conditioning, and the PROPHET_MAGMOM_HEAD env var to
the path of a magnetic-moment prediction bundle, before running these tests.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("prophet_spin")

from materialsframework.calculators.prophet import ProphetCalculator

MODEL_PATH = os.environ.get("PROPHET_MODEL", "")
SPIN_MODEL_PATH = os.environ.get("PROPHET_SPIN_MODEL", "")
MAGMOM_HEAD_PATH = os.environ.get("PROPHET_MAGMOM_HEAD", "")


@pytest.fixture(scope="module")
def calc() -> ProphetCalculator:
    """ProphetCalculator loaded from the env-var path of a checkpoint without spin conditioning, skipped if var is unset."""
    if not MODEL_PATH:
        pytest.skip(f"Prophet model not found at {MODEL_PATH}")
    return ProphetCalculator(model=MODEL_PATH)


@pytest.fixture(scope="module")
def spin_calc() -> ProphetCalculator:
    """ProphetCalculator loaded from the env-var path of a spin-conditioned checkpoint, skipped if var is unset."""
    if not SPIN_MODEL_PATH:
        pytest.skip(f"Spin-conditioned Prophet model not found at {SPIN_MODEL_PATH}")
    return ProphetCalculator(model=SPIN_MODEL_PATH)


@pytest.fixture(scope="module")
def magmom_calc() -> ProphetCalculator:
    """ProphetCalculator with a magnetic-moment prediction bundle, skipped if either env var is unset."""
    if not SPIN_MODEL_PATH or not MAGMOM_HEAD_PATH:
        pytest.skip("Spin-conditioned Prophet model or magnetic-moment prediction bundle not set")
    return ProphetCalculator(model=SPIN_MODEL_PATH, magmom_head=MAGMOM_HEAD_PATH)


@pytest.mark.integration
def test_lazy_load_contract() -> None:
    """_calculator must be None until the calculator property is accessed."""
    c = ProphetCalculator(model="does-not-exist.pt")
    assert c._calculator is None


@pytest.mark.integration
def test_available_properties() -> None:
    """AVAILABLE_PROPERTIES is exactly ['energy', 'free_energy', 'forces', 'stress']."""
    assert ProphetCalculator.AVAILABLE_PROPERTIES == ["energy", "free_energy", "forces", "stress"]


@pytest.mark.integration
def test_available_properties_with_magmom_head() -> None:
    """Passing `magmom_head` adds 'magmoms' to the instance's AVAILABLE_PROPERTIES without changing the class's."""
    c = ProphetCalculator(model="does-not-exist.pt", magmom_head="does-not-exist-head.pt")
    assert c.AVAILABLE_PROPERTIES == ["energy", "free_energy", "forces", "stress", "magmoms"]
    assert ProphetCalculator.AVAILABLE_PROPERTIES == ["energy", "free_energy", "forces", "stress"]


@pytest.mark.integration
def test_magmom_head_enables_include_magmoms() -> None:
    """Passing `magmom_head` defaults `include_magmoms` to True, an explicit False still wins, and it stays False otherwise."""
    assert ProphetCalculator(model="does-not-exist.pt", magmom_head="does-not-exist-head.pt").include_magmoms is True
    assert (
        ProphetCalculator(model="does-not-exist.pt", magmom_head="does-not-exist-head.pt", include_magmoms=False).include_magmoms is False
    )
    assert ProphetCalculator(model="does-not-exist.pt").include_magmoms is False


@pytest.mark.integration
def test_calculate_energy(calc: ProphetCalculator, bcc_fe) -> None:
    """calculate() returns a float energy for BCC Fe."""
    result = calc.calculate(bcc_fe)
    assert "energy" in result
    assert isinstance(result["energy"], (float, np.floating))
    assert isinstance(result["free_energy"], (float, np.floating))


@pytest.mark.integration
def test_calculate_forces_shape(calc: ProphetCalculator, bcc_fe) -> None:
    """calculate() returns forces with shape (n_atoms, 3)."""
    result = calc.calculate(bcc_fe)
    forces = np.array(result["forces"])
    assert forces.shape == (len(bcc_fe), 3)


@pytest.mark.integration
def test_relax_returns_structure(calc: ProphetCalculator, bcc_fe) -> None:
    """relax() returns a dict with 'final_structure' and 'trajectory'."""
    result = calc.relax(bcc_fe)
    assert "final_structure" in result
    assert "trajectory" in result


@pytest.mark.integration
def test_spin_calculate_energy(spin_calc: ProphetCalculator, bcc_fe) -> None:
    """calculate() with a spin-conditioned checkpoint returns a float energy and (n_atoms, 3) forces for BCC Fe."""
    result = spin_calc.calculate(bcc_fe)
    assert isinstance(result["energy"], (float, np.floating))
    assert np.array(result["forces"]).shape == (len(bcc_fe), 3)


@pytest.mark.integration
def test_spin_energy_depends_on_magmoms(spin_calc: ProphetCalculator, bcc_fe) -> None:
    """A spin-conditioned checkpoint reads the structure's `magmom` site property: FM BCC Fe is lower in energy than AFM."""
    ferromagnetic = bcc_fe.copy(site_properties={"magmom": [2.2, 2.2]})
    antiferromagnetic = bcc_fe.copy(site_properties={"magmom": [2.2, -2.2]})
    e_fm = spin_calc.calculate(ferromagnetic)["energy"]
    e_afm = spin_calc.calculate(antiferromagnetic)["energy"]
    assert e_fm < e_afm


@pytest.mark.integration
def test_spin_relax_returns_structure(spin_calc: ProphetCalculator, bcc_fe) -> None:
    """relax() with a spin-conditioned checkpoint returns a dict with 'final_structure' and 'trajectory'."""
    result = spin_calc.relax(bcc_fe.copy(site_properties={"magmom": [2.2, 2.2]}))
    assert "final_structure" in result
    assert "trajectory" in result


@pytest.mark.integration
def test_magmom_calculate_returns_magmoms(magmom_calc: ProphetCalculator, bcc_fe) -> None:
    """calculate() with `magmom_head` returns a float energy and one predicted moment per atom, following the seed signs."""
    seeds = [2.2, -2.2]
    result = magmom_calc.calculate(bcc_fe.copy(site_properties={"magmom": seeds}))
    assert isinstance(result["energy"], (float, np.floating))
    magmoms = np.array(result["magmoms"])
    assert magmoms.shape == (len(bcc_fe),)
    assert np.array_equal(np.sign(magmoms), np.sign(seeds))


@pytest.mark.integration
def test_magmom_relax_records_magmoms(magmom_calc: ProphetCalculator, bcc_fe) -> None:
    """relax() with `magmom_head` returns the predicted moments and records them in the trajectory."""
    result = magmom_calc.relax(bcc_fe.copy(site_properties={"magmom": [2.2, 2.2]}))
    assert np.array(result["magmoms"]).shape == (len(bcc_fe),)
    assert len(result["trajectory"].magmoms) > 0


@pytest.mark.integration
def test_magmom_head_requires_spin_model() -> None:
    """Passing `magmom_head` with a checkpoint that has no spin conditioning raises a clear ValueError."""
    if not MODEL_PATH or not MAGMOM_HEAD_PATH:
        pytest.skip("Prophet model without spin conditioning or magnetic-moment prediction bundle not set")
    with pytest.raises(ValueError, match="spin-conditioned"):
        _ = ProphetCalculator(model=MODEL_PATH, magmom_head=MAGMOM_HEAD_PATH).calculator
