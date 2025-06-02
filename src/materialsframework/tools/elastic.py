"""
This module provides functions to calculate the elastic tensor components
from the strain-stress relation for various crystal symmetries.

This file is based on code from the `elastic` package by Paweł T. Jochym (Copyright 1998-2017).
Original source: https://github.com/jochym/elastic

Modifications have been made to adapt and extend the original implementation for use in this project.
Please refer to the original license (GPLv3 or later) for terms of use.

Original docstring follows:

Elastic is a module for calculation of :math:`C_{ij}` components of elastic
tensor from the strain-stress relation.

The strain components here are ordered in standard way which is different
to ordering in previous versions of the code (up to 4.0).
The ordering is: :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}`.

The general ordering of :math:`C_{ij}` components is (except for triclinic
symmetry and taking into account customary names of constants - e.g.
:math:`C_{16} \\rightarrow C_{14}`):

.. math::
   C_{11}, C_{22}, C_{33}, C_{12}, C_{13}, C_{23},
   C_{44}, C_{55}, C_{66}, C_{16}, C_{26}, C_{36}, C_{45}

The functions with the name of bravais lattices define the symmetry of the
:math:`C_{ij}` matrix. The matrix is N columns by 6 rows where the columns
corespond to independent elastic constants of the given crystal, while the rows
corespond to the canonical deformations of a crystal. The elements are the
second partial derivatives of the free energy formula for the crystal written
down as a quadratic form of the deformations with respect to elastic constant
and deformation.

*Note:*
The elements for deformations :math:`u_{xy}, u_{xz}, u_{yz}`
have to be divided by 2 to properly match the usual definition
of elastic constants.

See: [LL]_ L.D. Landau, E.M. Lifszyc, "Theory of elasticity"

There is some usefull summary also at:
`ScienceWorld <https://scienceworld.wolfram.com/physics/Elasticity.html>`_
"""
from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

from ase.atoms import Atoms
from numpy import array, diag, dot, linspace, mean, ones, reshape
from numpy.linalg import inv
from scipy.linalg import lstsq

if TYPE_CHECKING:
    from numpy import ndarray


def regular(u: Sequence[float]) -> ndarray:
    """
    Generate the equation matrix for the regular (cubic) lattice.

    The order of constants is as follows: C_{11}, C_{12}, C_{44}.

    Args:
        u (Sequence[float]): Deformation vector [u_xx, u_yy, u_zz, u_yz, u_xz, u_xy].

    Returns:
        ndarray: Stress-strain equation matrix for cubic symmetry.
    """
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
        [
            [uxx, uyy + uzz, 0],
            [uyy, uxx + uzz, 0],
            [uzz, uxx + uyy, 0],
            [0, 0, 2 * uyz],
            [0, 0, 2 * uxz],
            [0, 0, 2 * uxy],
        ]
    )


def tetragonal(u: Sequence[float]) -> ndarray:
    """
    Generate the equation matrix for the tetragonal lattice.

    The order of constants is: C_{11}, C_{33}, C_{12}, C_{13}, C_{44}, C_{66}.

    Args:
        u (Sequence[float]): Deformation vector [u_xx, u_yy, u_zz, u_yz, u_xz, u_xy].

    Returns:
        ndarray: Stress-strain equation matrix for tetragonal symmetry.
    """
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
        [
            [uxx, 0, uyy, uzz, 0, 0],
            [uyy, 0, uxx, uzz, 0, 0],
            [0, uzz, 0, uxx + uyy, 0, 0],
            [0, 0, 0, 0, 2 * uxz, 0],
            [0, 0, 0, 0, 2 * uyz, 0],
            [0, 0, 0, 0, 0, 2 * uxy],
        ]
    )


def orthorombic(u: Sequence[float]) -> ndarray:
    """
    Generate the equation matrix for the orthorhombic lattice.

    The order of constants is: C_{11}, C_{22}, C_{33}, C_{12}, C_{13}, C_{23},
    C_{44}, C_{55}, C_{66}.

    Args:
        u (Sequence[float]): Deformation vector [u_xx, u_yy, u_zz, u_yz, u_xz, u_xy].

    Returns:
        ndarray: Stress-strain equation matrix for orthorhombic symmetry.
    """
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
        [
            [uxx, 0, 0, uyy, uzz, 0, 0, 0, 0],
            [0, uyy, 0, uxx, 0, uzz, 0, 0, 0],
            [0, 0, uzz, 0, uxx, uyy, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 2 * uyz, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2 * uxz, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 2 * uxy],
        ]
    )


def trigonal(u: Sequence[float]) -> ndarray:
    """
    Construct the equation matrix for the trigonal lattice based on L&L using
    auxiliary coordinates.

    The order of constants is: C_{11}, C_{33}, C_{12}, C_{13}, C_{44}, C_{14}.

    Args:
        u (Sequence[float]): Deformation vector [u_xx, u_yy, u_zz, u_yz, u_xz, u_xy].

    Returns:
        ndarray: Stress-strain equation matrix for trigonal symmetry.
    """
    # TODO: Not tested yet.
    # TODO: There is still some doubt about the :math:`C_{14}` constant.
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
        [
            [uxx, 0, uyy, uzz, 0, 2 * uxz],
            [uyy, 0, uxx, uzz, 0, -2 * uxz],
            [0, uzz, 0, uxx + uyy, 0, 0],
            [0, 0, 0, 0, 2 * uyz, -4 * uxy],
            [0, 0, 0, 0, 2 * uxz, 2 * (uxx - uyy)],
            [2 * uxy, 0, -2 * uxy, 0, 0, -4 * uyz],
        ]
    )


def hexagonal(u: Sequence[float]) -> ndarray:
    """
    Construct the equation matrix for the hexagonal lattice using auxiliary coordinates.

    The order of constants is: C_{11}, C_{33}, C_{12}, C_{13}, C_{44}.

    Args:
        u (Sequence[float]): Deformation vector [u_xx, u_yy, u_zz, u_yz, u_xz, u_xy].

    Returns:
        ndarray: Stress-strain equation matrix for hexagonal symmetry.
    """
    # TODO: Still needs good verification
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
        [
            [uxx, 0, uyy, uzz, 0],
            [uyy, 0, uxx, uzz, 0],
            [0, uzz, 0, uxx + uyy, 0],
            [0, 0, 0, 0, 2 * uyz],
            [0, 0, 0, 0, 2 * uxz],
            [uxy, 0, -uxy, 0, 0],
        ]
    )


def monoclinic(u: Sequence[float]) -> ndarray:
    """
    Generate the equation matrix for the monoclinic lattice.

    The ordering of constants is:
    C_{11}, C_{22}, C_{33}, C_{12}, C_{13}, C_{23},
    C_{44}, C_{55}, C_{66}, C_{16}, C_{26}, C_{36}, C_{45}.

    Args:
        u (Sequence[float]): Deformation vector [u_xx, u_yy, u_zz, u_yz, u_xz, u_xy].

    Returns:
        ndarray: Stress-strain equation matrix for monoclinic symmetry.
    """
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
        [
            [uxx, 0, 0, uyy, uzz, 0, 0, 0, 0, uxy, 0, 0, 0],
            [0, uyy, 0, uxx, 0, uzz, 0, 0, 0, 0, uxy, 0, 0],
            [0, 0, uzz, 0, uxx, uyy, 0, 0, 0, 0, 0, uxy, 0],
            [0, 0, 0, 0, 0, 0, 2 * uyz, 0, 0, 0, 0, 0, uxz],
            [0, 0, 0, 0, 0, 0, 0, 2 * uxz, 0, 0, 0, 0, uyz],
            [0, 0, 0, 0, 0, 0, 0, 0, 2 * uxy, uxx, uyy, uzz, 0],
        ]
    )


def triclinic(u: Sequence[float]) -> ndarray:
    """
    Construct the equation matrix for triclinic crystals.

    *Note*: This implementation is untested.

    The ordering of constants is:
    C_{11}, C_{22}, C_{33},
    C_{12}, C_{13}, C_{23},
    C_{44}, C_{55}, C_{66},
    C_{16}, C_{26}, C_{36}, C_{46}, C_{56},
    C_{14}, C_{15}, C_{25}, C_{45}.

    Args:
        u (Sequence[float]): Deformation vector [u_xx, u_yy, u_zz, u_yz, u_xz, u_xy].

    Returns:
        ndarray: Stress-strain equation matrix for triclinic symmetry.
    """
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return array(
        [
            [uxx, 0, 0, uyy, uzz, 0, 0, 0, 0, uxy, 0, 0, 0, 0, uyz, uxz, 0, 0],
            [0, uyy, 0, uxx, 0, uzz, 0, 0, 0, 0, uxy, 0, 0, 0, 0, 0, uxz, 0],
            [0, 0, uzz, 0, uxx, uyy, 0, 0, 0, 0, 0, uxy, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 2 * uyz, 0, 0, 0, 0, 0, uxy, 0, uxx, 0, 0, uxz],
            [0, 0, 0, 0, 0, 0, 0, 2 * uxz, 0, 0, 0, 0, 0, uxy, 0, uxx, uyy, uyz],
            [0, 0, 0, 0, 0, 0, 0, 0, 2 * uxy, uxx, uyy, uzz, uyz, uxz, 0, 0, 0, 0],
        ]
    )


def get_cij_order(cryst: Atoms) -> tuple[str, ...]:
    """
    Get the order of elastic constants for the structure.

    Args:
        cryst (Atoms): The ASE atoms object representing the structure.

    Returns:
        uple[str, ...]: Order of elastic constants as a tuple of strings.
    """
    orders = {
        1: ("C_11", "C_22", "C_33",
            "C_12", "C_13", "C_23",
            "C_44", "C_55", "C_66",
            "C_16", "C_26", "C_36",
            "C_46", "C_56", "C_14",
            "C_15", "C_25", "C_45",),
        2: ("C_11", "C_22", "C_33",
            "C_12", "C_13", "C_23",
            "C_44", "C_55", "C_66",
            "C_16", "C_26", "C_36",
            "C_45",),
        3: ("C_11", "C_22", "C_33",
            "C_12", "C_13", "C_23",
            "C_44", "C_55", "C_66"),
        4: ("C_11", "C_33", "C_12",
            "C_13", "C_44", "C_66"),
        5: ("C_11", "C_33", "C_12",
            "C_13", "C_44", "C_14"),
        6: ("C_11", "C_33", "C_12",
            "C_13", "C_44"),
        7: ("C_11", "C_12", "C_44"),
    }
    return orders[get_lattice_type(cryst)[0]]


def get_lattice_type(
        structure: Atoms,
        length_tol: float = 1e-4,
        angle_tol: float = 1.0
) -> tuple[int, str, None, None]:
    """
    Classify the lattice type and Bravais lattice name of a crystal structure
    based on its unit cell parameters.

    This function has been modified to determine the lattice type solely from
    the unit cell lengths and angles, without relying on external symmetry
    analysis tools like spglib. It uses specified tolerances to compare lattice
    parameters and classify the crystal system accordingly.

    Args:
        structure (Atoms): The ASE Atoms object representing the crystal structure.
        length_tol (float): Tolerance for comparing lattice lengths.
        angle_tol (float): Tolerance for comparing lattice angles in degrees.

    Returns:
        tuple[int, str, None, None]: A tuple containing:
            - lattice type number (1-7):
                1: Triclinic
                2: Monoclinic
                3: Orthorhombic
                4: Tetragonal
                5: Trigonal
                6: Hexagonal
                7: Cubic
            - Bravais lattice name (str)
            - None (placeholder for space group name)
            - None (placeholder for space group number)

    """
    def approx_equal(x, y, tol):
        return abs(x - y) < tol

    a, b, c = structure.cell.lengths()
    alpha, beta, gamma = structure.cell.angles()

    if approx_equal(alpha, 90, angle_tol) and approx_equal(beta, 90, angle_tol) and approx_equal(gamma, 90, angle_tol):
        if approx_equal(a, b, length_tol) and approx_equal(b, c, length_tol):
            return 7, "Cubic", None, None
        elif approx_equal(a, b, length_tol):
            return 4, "Tetragonal", None, None
        elif approx_equal(gamma, 120, angle_tol):
            return 6, "Hexagonal", None, None
        else:
            return 3, "Orthorombic", None, None
    else:
        if approx_equal(alpha, beta, angle_tol) and approx_equal(beta, gamma, angle_tol):
            return 5, "Trigonal", None, None
        else:
            return 1, "Triclinic", None, None


def get_pressure(s: ndarray) -> float:
    """
    Return external isotropic (hydrostatic) pressure in ASE units.

    The pressure is taken as negative of the mean of the first three stress components.

    Args:
        s (ndarray): Stress tensor as a 6 component array.

    Returns:
        float: External hydrostatic pressure.
    """
    return -mean(s[:3])


def get_elementary_deformations(cryst: Atoms, n: int = 5, d: float = 2) -> list[Atoms]:
    """
    Generate elementary deformations for elastic tensor calculation based on crystal symmetry.

    Args:
        cryst (Atoms): Base structure.
        n (int): Number of deformations along each axis.
        d (float): Deformation magnitude.

    Returns:
        List[Atoms]: A list of deformed structures.
    """
    deform = {
        "Cubic": [[0, 3], regular],
        "Hexagonal": [[0, 2, 3, 5], hexagonal],
        "Trigonal": [[0, 1, 2, 3, 4, 5], trigonal],
        "Tetragonal": [[0, 2, 3, 5], tetragonal],
        "Orthorombic": [[0, 1, 2, 3, 4, 5], orthorombic],
        "Monoclinic": [[0, 1, 2, 3, 4, 5], monoclinic],
        "Triclinic": [[0, 1, 2, 3, 4, 5], triclinic],
    }

    lattyp, brav, sg_name, sg_nr = get_lattice_type(cryst)
    axis, symm = deform[brav]

    systems = []
    for a in axis:
        if a < 3:  # tetragonal deformation
            for dx in linspace(-d, d, n):
                systems.append(get_cart_deformed_cell(cryst, axis=a, size=dx))
        elif a < 6:  # sheer deformation (skip the zero angle)
            for dx in linspace(d / 10.0, d, n):
                systems.append(get_cart_deformed_cell(cryst, axis=a, size=dx))
    return systems


def get_elastic_tensor(cryst: Atoms, systems: list[Atoms]) -> tuple[ndarray, tuple[ndarray, ...]]:
    """
    Calculate the elastic tensor of the crystal using stress-strain fitting.

    The elastic tensor is derived from the stress-strain relation
    using a least squares fit.

    Args:
        cryst (Atoms): Reference structure with calculated stress.
        systems (List[Atoms]): List of deformed structures with stresses computed.

    Returns:
        Tuple[ndarray, Tuple[np.ndarray, ...]]:
            - Elastic tensor coefficients (C_{ij})
            - Fitting results (solution, residuals, rank, singular values)
    """
    deform = {
        "Cubic": [[0, 3], regular],
        "Hexagonal": [[0, 2, 3, 5], hexagonal],
        "Trigonal": [[0, 1, 2, 3, 4, 5], trigonal],
        "Tetragonal": [[0, 2, 3, 5], tetragonal],
        "Orthorombic": [[0, 1, 2, 3, 4, 5], orthorombic],
        "Monoclinic": [[0, 1, 2, 3, 4, 5], monoclinic],
        "Triclinic": [[0, 1, 2, 3, 4, 5], triclinic],
    }

    lattyp, brav, sg_name, sg_nr = get_lattice_type(cryst)
    axis, symm = deform[brav]

    ul = []
    sl = []
    p = get_pressure(cryst.get_stress())
    for g in systems:
        ul.append(get_strain(g, refcell=cryst))
        # Remove the ambient pressure from the stress tensor
        sl.append(g.get_stress() - array([p, p, p, 0, 0, 0]))
    eqm = array([symm(u) for u in ul])
    eqm = reshape(eqm, (eqm.shape[0] * eqm.shape[1], eqm.shape[2]))
    slm = reshape(array(sl), (-1,))
    Bij = lstsq(eqm, slm)
    # TODO: Check the sign of the pressure array in the B <=> C relation
    if symm == orthorombic:
        Cij = Bij[0] - array([-p, -p, -p, p, p, p, -p, -p, -p])
    elif symm == tetragonal:
        Cij = Bij[0] - array([-p, -p, p, p, -p, -p])
    elif symm == regular:
        Cij = Bij[0] - array([-p, p, -p])
    elif symm == trigonal:
        Cij = Bij[0] - array([-p, -p, p, p, -p, p])
    elif symm == hexagonal:
        Cij = Bij[0] - array([-p, -p, p, p, -p])
    elif symm == monoclinic:
        # TODO: verify this pressure array
        Cij = Bij[0] - array([-p, -p, -p, p, p, p, -p, -p, -p, p, p, p, p])
    elif symm == triclinic:
        # TODO: verify this pressure array
        Cij = Bij[0] - array(
            [-p, -p, -p, p, p, p, -p, -p, -p, p, p, p, p, p, p, p, p, p]
        )
    return Cij, Bij


def get_cart_deformed_cell(base_cryst: Atoms, axis: int = 0, size: float = 1) -> Atoms:
    """
    Return a cell deformed along one of the Cartesian or shear directions.

    The deformation is performed by applying a linear deformation matrix
    to the base structure's cell.

    Args:
        base_cryst (Atoms): The base structure.
        axis (int): Deformation axis (0, 1, 2 for x, y, z; 3,4,5 for shear).
        size (float): Magnitude of the deformation (percentage for axial deformation
                      and degrees for shear deformation).

    Returns:
        Atoms: New structure with deformed cell.
    """
    cryst = base_cryst.copy()
    uc = base_cryst.get_cell()
    s = size / 100.0
    L = diag(ones(3))
    if axis < 3:
        L[axis, axis] += s
    else:
        if axis == 3:
            L[1, 2] += s
        elif axis == 4:
            L[0, 2] += s
        else:
            L[0, 1] += s
    uc = dot(uc, L)
    cryst.set_cell(uc, scale_atoms=True)
    return cryst


def get_strain(cryst: Atoms, refcell: Atoms | None = None) -> ndarray:
    """
    Calculate the strain tensor in Voigt notation based on a reference cell.

    The strain tensor is computed as a symmetric tensor represented as a 6-element vector.

    Args:
        cryst (Atoms): The deformed structure.
        refcell (Atoms, optional): The reference structure. If None, uses cryst.

    Returns:
        ndarray: 6-element strain vector [u_xx, u_yy, u_zz, u_yz, u_xz, u_xy].
    """
    if refcell is None:
        refcell = cryst
    du = cryst.get_cell() - refcell.get_cell()
    m = refcell.get_cell()
    m = inv(m)
    u = dot(m, du)
    u = (u + u.T) / 2
    return array([u[0, 0], u[1, 1], u[2, 2], u[2, 1], u[2, 0], u[1, 0]])
