# Phonopy

!!! info "Optional dependency"

    `PhonopyAnalyzer` requires the `phonopy` extra.

    === "uv"

        ```bash
        uv add "materialsframework[phonopy]"
        ```

    === "pip"

        ```bash
        pip install "materialsframework[phonopy]"
        ```

Computes harmonic phonon frequencies, density of states, and thermal properties from
finite-displacement force constants.

## Overview

`PhonopyAnalyzer` builds a supercell (default 2×2×2), displaces one atom at a time by a small
distance (default 0.01 Å), and evaluates the resulting forces with the provided calculator. Phonopy
uses these forces to build the harmonic force constants, then diagonalizes the dynamical matrix on a
q-point mesh to get phonon frequencies everywhere in the Brillouin zone. From there it derives the
total phonon density of states (DOS), the site-projected DOS (PDOS), and thermodynamic quantities
(free energy, entropy, heat capacity) under the harmonic approximation.

The harmonic approximation treats phonons as non-interacting: it is exact only to second order in
atomic displacement, so it ignores phonon-phonon scattering. That's sufficient for vibrational
thermodynamics and DOS/band structure, but not for finite thermal conductivity. See
[Phono3py](phono3py.md) for the anharmonic (third-order) extension.

## Theory

### Force constants and the dynamical matrix

The harmonic force constants are the second derivatives of the potential energy with respect to
atomic displacements:

$$
\Phi_{\alpha\beta}(l\kappa, l'\kappa') = -\frac{\partial F_\alpha(l\kappa)}{\partial u_\beta(l'\kappa')}
$$

where \(F_\alpha(l\kappa)\) is the force on atom \(\kappa\) in cell \(l\) along Cartesian direction
\(\alpha\), and \(u_\beta(l'\kappa')\) is a displacement of atom \(\kappa'\) in cell \(l'\) along
\(\beta\). `PhonopyAnalyzer` estimates \(\Phi\) numerically: it displaces each inequivalent atom by
`distance` and reads off the force response from the calculator, then uses the supercell's symmetry
to fill in the remaining components.

The mass-weighted, Fourier-transformed force constants give the dynamical matrix at wavevector
\(\mathbf{q}\):

$$
D_{\alpha\beta}(\kappa\kappa', \mathbf{q}) = \frac{1}{\sqrt{m_\kappa m_{\kappa'}}}
\sum_{l'} \Phi_{\alpha\beta}(0\kappa, l'\kappa')\, e^{i\mathbf{q}\cdot\mathbf{r}(l')}
$$

Diagonalizing \(D(\mathbf{q})\) gives the phonon frequencies \(\omega_s(\mathbf{q})\) and
eigenvectors (polarization vectors) for each band \(s\):

$$
D(\mathbf{q})\, \mathbf{e}_s(\mathbf{q}) = \omega_s(\mathbf{q})^2\, \mathbf{e}_s(\mathbf{q})
$$

Sampling \(\omega_s(\mathbf{q})\) on the `mesh` grid and histogramming (optionally Gaussian-smeared
by `sigma`) gives the total DOS; resolving each eigenvector's atomic-site weight on `pdos_mesh`
gives the site-projected DOS.

### Thermal properties

For each phonon mode \(\lambda = (\mathbf{q}, s)\) with frequency \(\omega_\lambda\), the harmonic
oscillator partition function gives the vibrational free energy, entropy, and constant-volume heat
capacity at temperature \(T\):

$$
F(T) = \frac{1}{2}\sum_\lambda \hbar\omega_\lambda
+ k_B T \sum_\lambda \ln\left(1 - e^{-\hbar\omega_\lambda / k_B T}\right)
$$

$$
S(T) = \sum_\lambda \left[
\frac{\hbar\omega_\lambda / k_B T}{e^{\hbar\omega_\lambda / k_B T} - 1}
- \ln\left(1 - e^{-\hbar\omega_\lambda / k_B T}\right)
\right] k_B
$$

$$
C_V(T) = k_B \sum_\lambda \left(\frac{\hbar\omega_\lambda}{k_B T}\right)^2
\frac{e^{\hbar\omega_\lambda / k_B T}}{\left(e^{\hbar\omega_\lambda / k_B T} - 1\right)^2}
$$

These sums run over every mode on the `mesh` grid and are evaluated at each temperature between
`t_min` and `t_max` in steps of `t_step`.

## References

- Phonopy: [https://doi.org/10.1016/j.scriptamat.2015.07.021](https://doi.org/10.1016/j.scriptamat.2015.07.021)
