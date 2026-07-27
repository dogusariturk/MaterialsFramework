"""Tests for `BaseMDCalculator` and its supported ASE-integrator-backed ensembles."""

from __future__ import annotations

import re
import warnings

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md import VelocityVerlet
from ase.md.andersen import Andersen
from ase.md.bussi import Bussi
from ase.md.langevin import Langevin
from ase.md.melchionna import MelchionnaNPT
from ase.md.nose_hoover_chain import MTKNPT, IsotropicMTKNPT, MaskedMTKNPT, NoseHooverChainNVT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import thermalize_momenta

from materialsframework.tools.md import BaseMDCalculator

# Ensembles that require a `pressure` (and, for one, a `mask`) to be meaningful.
NPT_LIKE_ENSEMBLES = {
    "npt_nose_hoover",
    "isotropic_mtk_npt",
    "mtk_npt",
    "masked_mtk_npt",
    "npt_berendsen",
    "inhomogeneous_npt_berendsen",
}

# Every ensemble mapped to the ASE dynamics class it must construct.
ENSEMBLE_DYNAMICS_TYPES = {
    "nve": VelocityVerlet,
    "nvt_nose_hoover": MelchionnaNPT,
    "langevin": Langevin,
    "andersen": Andersen,
    "bussi": Bussi,
    "nvt_berendsen": NVTBerendsen,
    "nose_hoover_chain_nvt": NoseHooverChainNVT,
    "npt_nose_hoover": MelchionnaNPT,
    "isotropic_mtk_npt": IsotropicMTKNPT,
    "mtk_npt": MTKNPT,
    "masked_mtk_npt": MaskedMTKNPT,
    "npt_berendsen": NPTBerendsen,
    "inhomogeneous_npt_berendsen": Inhomogeneous_NPTBerendsen,
}

ALL_ENSEMBLES = list(ENSEMBLE_DYNAMICS_TYPES)


def _kwargs_for(ensemble: str) -> dict:
    """Builds the extra constructor kwargs a given ensemble needs to run meaningfully."""
    kwargs = {"pressure": 0.0} if ensemble in NPT_LIKE_ENSEMBLES else {}
    if ensemble == "masked_mtk_npt":
        kwargs["mask"] = (1, 1, 0)
    return kwargs


class _EMTMDCalculator(BaseMDCalculator):
    """Minimal concrete `BaseMDCalculator` backed by ASE's dependency-free EMT calculator."""

    @property
    def calculator(self):
        return EMT()


@pytest.fixture(scope="module")
def cu_atoms():
    """4-atom cubic FCC Cu cell; already upper-triangular, so cell-shape conversion is a no-op."""
    return bulk("Cu", "fcc", a=3.6, cubic=True)


@pytest.fixture(scope="module")
def primitive_cu_atoms():
    """8-atom, non-cubic, non-triangular FCC Cu cell (replicated primitive cell)."""
    return bulk("Cu", "fcc", a=3.6) * (2, 2, 2)


@pytest.mark.parametrize("ensemble", ALL_ENSEMBLES)
def test_run_returns_expected_keys(cu_atoms, ensemble) -> None:
    """run() returns the documented dictionary keys for every supported ensemble."""
    calc = _EMTMDCalculator(ensemble=ensemble, timestep=1.0, temperature=300, **_kwargs_for(ensemble))
    result = calc.run(cu_atoms, steps=3)

    assert set(result) == {
        "total_energy",
        "potential_energy",
        "kinetic_energy",
        "forces",
        "stresses",
        "temperature",
        "velocities",
        "final_structure",
    }


@pytest.mark.parametrize("ensemble", ALL_ENSEMBLES)
def test_run_records_one_step_per_interval(cu_atoms, ensemble) -> None:
    """Each MD step is recorded once, plus the initial state, for every ensemble."""
    calc = _EMTMDCalculator(ensemble=ensemble, timestep=1.0, temperature=300, **_kwargs_for(ensemble))
    result = calc.run(cu_atoms, steps=3)

    assert len(result["total_energy"]) == 4
    assert len(result["temperature"]) == 4


@pytest.mark.parametrize("ensemble", ALL_ENSEMBLES)
def test_run_raises_no_ase_deprecation_warnings(cu_atoms, ensemble) -> None:
    """None of the supported ensembles trigger a DeprecationWarning/FutureWarning from ASE."""
    calc = _EMTMDCalculator(ensemble=ensemble, timestep=1.0, temperature=300, **_kwargs_for(ensemble))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calc.run(cu_atoms, steps=3)

    deprecation_warnings = [w for w in caught if issubclass(w.category, (DeprecationWarning, FutureWarning))]
    assert not deprecation_warnings


@pytest.mark.parametrize(("ensemble", "expected_type"), ENSEMBLE_DYNAMICS_TYPES.items())
def test_initializer_constructs_expected_dynamics_class(cu_atoms, ensemble, expected_type) -> None:
    """Each `_initialize_<ensemble>` helper constructs the ASE dynamics class it's meant to.

    This guards `run()`'s ensemble-to-initializer dispatch table against a mixed-up mapping between
    two ensembles that happen to take identical keyword arguments (e.g. the two MTK NPT variants),
    which a generic "does it run" smoke test would not catch.
    """
    ase_atoms = cu_atoms.copy()
    calc = _EMTMDCalculator(ensemble=ensemble, timestep=1.0, temperature=300, **_kwargs_for(ensemble))
    thermalize_momenta(ase_atoms, temperature_K=calc.temperature)
    ase_atoms.calc = calc.calculator

    dyn = getattr(calc, f"_initialize_{ensemble}")(ase_atoms)

    assert isinstance(dyn, expected_type)


def test_melchionna_ensembles_convert_non_triangular_cell(primitive_cu_atoms) -> None:
    """The Melchionna-backed ensembles convert a non-triangular cell before running, instead of raising."""
    for ensemble in ("nvt_nose_hoover", "npt_nose_hoover"):
        calc = _EMTMDCalculator(ensemble=ensemble, timestep=1.0, temperature=300, pressure=0.0)
        result = calc.run(primitive_cu_atoms, steps=2)
        assert len(result["total_energy"]) == 3


def test_masked_mtk_npt_mask_is_a_boolean_tuple(cu_atoms) -> None:
    """`_initialize_masked_mtk_npt` converts the int-based `mask` attribute to real Python bools.

    `MaskedMTKNPT` uses `mask` for boolean fancy-indexing (`self._p_c[self.mask]`); passing the plain
    int tuple used elsewhere in this class (e.g. for `Inhomogeneous_NPTBerendsen`) would silently
    switch that indexing from "select axes where True" to integer-position indexing.
    """
    ase_atoms = cu_atoms.copy()
    calc = _EMTMDCalculator(ensemble="masked_mtk_npt", pressure=0.0, mask=(1, 1, 0))
    ase_atoms.calc = calc.calculator

    dyn = calc._initialize_masked_mtk_npt(ase_atoms)

    assert dyn.mask == (True, True, False)
    assert all(isinstance(value, bool) for value in dyn.mask)


def test_init_stores_new_ensemble_parameters() -> None:
    """The Langevin/Andersen-specific constructor parameters are stored on the instance."""
    calc = _EMTMDCalculator(friction=0.05, andersen_prob=0.02)

    assert calc.friction == pytest.approx(0.05)
    assert calc.andersen_prob == pytest.approx(0.02)


def test_invalid_ensemble_raises_value_error() -> None:
    """An unsupported ensemble name raises ValueError."""
    with pytest.raises(ValueError, match="Ensemble must be one of"):
        _EMTMDCalculator(ensemble="bogus")  # ty: ignore[invalid-argument-type]


def test_invalid_ensemble_message_lists_every_supported_ensemble() -> None:
    """The ValueError message enumerates exactly the ensembles accepted by the constructor.

    Guards against the hand-maintained message string drifting out of sync with the actual
    validation list as ensembles are added or renamed.
    """
    with pytest.raises(ValueError) as exc_info:
        _EMTMDCalculator(ensemble="bogus")  # ty: ignore[invalid-argument-type]

    quoted = set(re.findall(r"'([^']+)'", str(exc_info.value)))
    assert quoted == set(ALL_ENSEMBLES)
