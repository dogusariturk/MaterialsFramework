"""Tests for SBETransformation."""

from __future__ import annotations

import pytest
from pymatgen.core import Structure
from pymatgen.core.surface import Slab

from materialsframework.transformations.sbe import SBETransformation


@pytest.fixture(scope="module")
def transformation():
    """SBETransformation with small slab/vacuum/supercell sizes for fast tests."""
    return SBETransformation(max_index=1, min_slab_size=6.0, min_vacuum_size=8.0, supercell_size=[2, 2, 1])


@pytest.fixture(scope="module")
def slabs(transformation, bcc_fe):
    """Slabs generated from the BCC Fe fixture, shared across tests that only read them."""
    return transformation.apply_transformation(bcc_fe)


def test_default_params() -> None:
    """SBETransformation stores the documented default values."""
    t = SBETransformation()
    assert t.max_index == 1
    assert t.min_slab_size == pytest.approx(10.0)
    assert t.min_vacuum_size == pytest.approx(10.0)
    assert t.height == pytest.approx(1.0)
    assert t.supercell_size == [4, 4, 1]
    assert t.isolated_atom_box_size == pytest.approx(20.0)


def test_custom_params() -> None:
    """Custom constructor parameters are stored correctly."""
    t = SBETransformation(
        max_index=2,
        min_slab_size=5.0,
        min_vacuum_size=15.0,
        height=0.5,
        supercell_size=[2, 3, 1],
        isolated_atom_box_size=25.0,
    )
    assert t.max_index == 2
    assert t.min_slab_size == pytest.approx(5.0)
    assert t.min_vacuum_size == pytest.approx(15.0)
    assert t.height == pytest.approx(0.5)
    assert t.supercell_size == [2, 3, 1]
    assert t.isolated_atom_box_size == pytest.approx(25.0)


def test_apply_transformation_returns_slabs(slabs) -> None:
    """apply_transformation returns a non-empty list of pymatgen Slab objects."""
    assert len(slabs) > 0
    for slab in slabs:
        assert isinstance(slab, Slab)
        assert slab.surface_area > 0
        assert max(abs(index) for index in slab.miller_index) <= 1


def test_apply_transformation_independent_calls(transformation, bcc_fe) -> None:
    """Calling apply_transformation() twice returns independent lists, not an accumulated one."""
    first = transformation.apply_transformation(bcc_fe)
    second = transformation.apply_transformation(bcc_fe)

    assert first is not second
    assert len(first) == len(second)


def test_apply_transformation_slab_arg_replicates_lateral_directions(transformation, slabs) -> None:
    """The `slab` argument scales the a/b site count by supercell_size while leaving c (1 layer) alone."""
    slab = slabs[0]
    supercell = transformation.apply_transformation(slab=slab)

    multiplier = transformation.supercell_size[0] * transformation.supercell_size[1] * transformation.supercell_size[2]
    assert supercell.num_sites == slab.num_sites * multiplier
    assert isinstance(supercell, Structure)


def test_apply_transformation_slab_arg_independent_of_input(transformation, slabs) -> None:
    """The `slab` argument does not mutate the input slab (in_place=False)."""
    slab = slabs[0]
    original_num_sites = slab.num_sites

    transformation.apply_transformation(slab=slab)

    assert slab.num_sites == original_num_sites


def test_apply_transformation_supercell_slab_arg_returns_valid_entries(transformation, slabs) -> None:
    """Each vacancy entry has one fewer site than the supercell and a valid element/site index."""
    supercell = transformation.apply_transformation(slab=slabs[0])
    vacancies = transformation.apply_transformation(supercell_slab=supercell)

    assert len(vacancies) > 0
    valid_elements = {el.symbol for el in supercell.composition.elements}
    for vacancy in vacancies:
        assert vacancy["structure"].num_sites == supercell.num_sites - 1
        assert 0 <= vacancy["site_index"] < supercell.num_sites
        assert vacancy["element"] in valid_elements
        assert isinstance(vacancy["structure"], Structure)


def test_apply_transformation_supercell_slab_arg_independent_of_input(transformation, slabs) -> None:
    """The `supercell_slab` argument does not mutate the supercell it reads sites from."""
    supercell = transformation.apply_transformation(slab=slabs[0])
    original_num_sites = supercell.num_sites

    transformation.apply_transformation(supercell_slab=supercell)

    assert supercell.num_sites == original_num_sites


def test_apply_transformation_element_arg_returns_single_atom_in_a_box() -> None:
    """The `element` argument returns one atom of the requested element in a cubic box."""
    t = SBETransformation(isolated_atom_box_size=15.0)
    structure = t.apply_transformation(element="Fe")

    assert structure.num_sites == 1
    assert structure.elements[0].symbol == "Fe"
    assert structure.lattice.a == pytest.approx(15.0)
    assert structure.lattice.b == pytest.approx(15.0)
    assert structure.lattice.c == pytest.approx(15.0)


def test_apply_transformation_no_args_raises() -> None:
    """Calling apply_transformation with no arguments raises ValueError."""
    t = SBETransformation()
    with pytest.raises(ValueError, match="Exactly one of"):
        t.apply_transformation()


def test_apply_transformation_multiple_args_raises(bcc_fe) -> None:
    """Calling apply_transformation with more than one argument raises ValueError."""
    t = SBETransformation()
    with pytest.raises(ValueError, match="Exactly one of"):
        t.apply_transformation(structure=bcc_fe, element="Fe")
