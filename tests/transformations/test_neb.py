"""Tests for NEBTransformation."""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from materialsframework.transformations.neb import NEBTransformation

_CUSTOM_N_IMAGES = 3


@pytest.fixture
def fe_endpoints(bcc_fe) -> tuple[Structure, Structure]:
    """Endpoint structures for a small, single-atom hop in BCC Fe."""
    final = bcc_fe.copy()
    final.translate_sites(indices=[1], vector=[0.1, 0.0, 0.0], frac_coords=False, to_unit_cell=True)
    return bcc_fe, final


def test_default_params() -> None:
    """Default interpolation parameters are stored correctly."""
    t = NEBTransformation()
    assert t.n_images == 5
    assert t.interpolate_lattices is False
    assert t.pbc is True
    assert t.autosort_tol == pytest.approx(0.5)
    assert t.end_amplitude == pytest.approx(1)


def test_custom_params() -> None:
    """Custom interpolation parameters are stored correctly."""
    t = NEBTransformation(
        n_images=_CUSTOM_N_IMAGES,
        interpolate_lattices=True,
        pbc=False,
        autosort_tol=0.0,
        end_amplitude=0.5,
    )
    assert t.n_images == _CUSTOM_N_IMAGES
    assert t.interpolate_lattices is True
    assert t.pbc is False
    assert t.autosort_tol == pytest.approx(0.0)
    assert t.end_amplitude == pytest.approx(0.5)


def test_apply_transformation_returns_structures(fe_endpoints) -> None:
    """apply_transformation returns n_images + 1 pymatgen Structures, including both endpoints."""
    initial, final = fe_endpoints
    t = NEBTransformation(n_images=_CUSTOM_N_IMAGES)
    images = t.apply_transformation(initial, final)

    assert len(images) == _CUSTOM_N_IMAGES + 1
    for image in images:
        assert isinstance(image, Structure)
        assert len(image) == len(initial)


def test_apply_transformation_endpoints_match_input(fe_endpoints) -> None:
    """The first and last interpolated images match the given initial/final structures."""
    initial, final = fe_endpoints
    t = NEBTransformation(n_images=2)
    images = t.apply_transformation(initial, final)

    assert images[0].frac_coords == pytest.approx(initial.frac_coords)
    assert images[-1].frac_coords == pytest.approx(final.frac_coords)


def test_apply_transformation_intermediate_image_is_between_endpoints(fe_endpoints) -> None:
    """A single intermediate image (n_images=2) sits at the halfway point between the endpoints."""
    initial, final = fe_endpoints
    t = NEBTransformation(n_images=2)
    images = t.apply_transformation(initial, final)

    assert len(images) == 3
    midpoint = (initial.frac_coords + final.frac_coords) / 2
    assert images[1].frac_coords == pytest.approx(midpoint)


def test_apply_transformation_different_lattice_raises(bcc_fe) -> None:
    """Interpolating between structures with different lattices and interpolate_lattices=False raises."""
    different_lattice = Structure(Lattice.cubic(3.0), bcc_fe.species, bcc_fe.frac_coords)
    t = NEBTransformation()
    with pytest.raises(ValueError, match="different lattice"):
        t.apply_transformation(bcc_fe, different_lattice)


def test_apply_transformation_different_length_raises(bcc_fe, fcc_ni) -> None:
    """Interpolating between structures with a different number of sites raises."""
    t = NEBTransformation()
    with pytest.raises(ValueError, match="different lengths"):
        t.apply_transformation(bcc_fe, fcc_ni)
