"""Tests for ElasticConstantsDeformationTransformation."""

from __future__ import annotations

from ase import Atoms

from materialsframework.transformations.elastic_constants import (
    ElasticConstantsDeformationTransformation,
)


def test_default_params() -> None:
    """Default num_deform and max_deform are stored correctly."""
    t = ElasticConstantsDeformationTransformation()
    assert t.num_deform == 5
    assert t.max_deform == 2


def test_custom_params() -> None:
    """Custom num_deform and max_deform are stored correctly."""
    t = ElasticConstantsDeformationTransformation(num_deform=3, max_deform=1.0)
    assert t.num_deform == 3
    assert t.max_deform == 1.0


def test_apply_transformation_returns_ase_atoms(bcc_fe) -> None:
    """apply_transformation returns a list of ASE Atoms objects."""
    t = ElasticConstantsDeformationTransformation(num_deform=3, max_deform=1.0)
    result = t.apply_transformation(bcc_fe)
    assert len(result) > 0
    for atoms in result:
        assert isinstance(atoms, Atoms)


def test_apply_transformation_cubic_deformation_count(bcc_fe) -> None:
    """Cubic BCC Fe generates the expected number of elementary deformations.

    Cubic axes = [0, 3]: axis 0 (tetragonal) -> n points; axis 3 (shear) -> n points.
    With n=3: 3 + 3 = 6 deformations total.
    """
    t = ElasticConstantsDeformationTransformation(num_deform=3, max_deform=1.0)
    result = t.apply_transformation(bcc_fe)
    assert len(result) == 6
