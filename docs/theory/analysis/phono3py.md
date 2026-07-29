# Phono3py

!!! info "Optional dependency"

    `Phono3pyAnalyzer` requires the `phono3py` extra.

    === "uv"

        ```bash
        uv add "materialsframework[phono3py]"
        ```

    === "pip"

        ```bash
        pip install "materialsframework[phono3py]"
        ```

Computes third-order (anharmonic) force constants and uses them to solve for lattice thermal conductivity.

## Overview

Harmonic phonons (see [Phonopy](phonopy.md)) don't scatter off each other, so on their own they predict infinite thermal conductivity. There's no mechanism to relax a phonon population back toward equilibrium. `Phono3pyAnalyzer` adds the third-order force constants needed to describe phonon-phonon scattering, then solves a phonon Boltzmann transport equation to get a finite lattice thermal conductivity tensor.

It builds two supercells: a larger one (`supercell_matrix`, default 2x2x2) for the third-order force constants, and a separate one (`phonon_supercell_matrix`, default 3x3x3) for the second-order (harmonic) force constants that phono3py also needs for the group velocities and mode heat capacities entering the transport equation. Atoms in both are displaced by `distance` (default 0.01 Å) and evaluated with the calculator, the same finite-displacement idea as `PhonopyAnalyzer` but doubled up for the extra derivative order.

## Theory

### Third-order force constants

The third-order force constants are the third derivatives of the potential energy with respect to three atomic displacements:

$$
\Phi_{\alpha\beta\gamma}(l\kappa, l'\kappa', l''\kappa'') =
\frac{\partial^3 U}{\partial u_\alpha(l\kappa)\, \partial u_\beta(l'\kappa')\, \partial u_\gamma(l''\kappa'')}
$$

Phono3py estimates \(\Phi\) numerically from the forces measured on systematically displaced pairs of atoms in the `supercell_matrix` supercell, the anharmonic analogue of the finite-displacement scheme used for the ordinary (second-order) force constants.

### Phonon scattering and lifetimes

\(\Phi\) enters a three-phonon scattering rate for each mode \(\lambda = (\mathbf{q}, s)\), derived from Fermi's golden rule for absorption and emission of phonon pairs. The mode's linewidth \(\Gamma_\lambda\) sets its lifetime, \(\tau_\lambda = 1/(2\Gamma_\lambda)\): the shorter the lifetime, the more strongly that mode scatters and the less it contributes to heat transport. By default (`is_lbte=False`), lifetimes are obtained under the single-mode relaxation-time approximation (RTA), which treats every mode's population as relaxing independently back to equilibrium. Setting `is_lbte=True` instead solves the linearized Boltzmann transport equation (LBTE) directly, which additionally captures collective (normal-process) phonon drift and is more accurate, at significantly higher computational cost. `is_isotope=True` adds isotopic mass-disorder scattering, and `boundary_mfp` adds a simple grain/sample-boundary scattering contribution on top of the intrinsic phonon-phonon rates.

### Thermal conductivity

Once mode lifetimes, group velocities \(\mathbf{v}_\lambda\), and mode heat capacities \(C_\lambda\) are known (the last two from the harmonic phonons on `phonon_supercell_matrix`), the lattice thermal conductivity tensor follows the standard BTE-RTA form:

$$
\kappa_{\alpha\beta}(T) = \frac{1}{N V_0}\sum_\lambda C_\lambda(T)\, v_{\lambda,\alpha}\, v_{\lambda,\beta}\, \tau_\lambda(T)
$$

where \(N\) is the number of unit cells sampled on the `mesh` grid and \(V_0\) is the unit cell volume. `transport_type` selects an inter-band correction layered on top of this intra-band result: `"SMM19"` is the Wigner transport formulation of Simoncelli, Marzari & Mauri, which adds a coherent tunneling contribution between nearly degenerate bands (important in materials with low, glass-like conductivity); `"IBDB19"` is the quasi-harmonic Green-Kubo approach of Isaeva, Barbalinardo, Donadio & Baroni; `"NJC23"` is a further Green-Kubo-based formulation. `kappa` holds the resulting tensor with shape `(sigmas, temperatures, 6)`, the last axis being the independent Voigt components (xx, yy, zz, yz, xz, xy).

## References

- Phono3py / phonon lifetimes: Togo, A., Chaput, L., & Tanaka, I. (2015). Distributions of phonon lifetimes in Brillouin zones. *Physical Review B*, 91, 094306. <https://doi.org/10.1103/PhysRevB.91.094306>
- Wigner transport equation ("SMM19"): Simoncelli, M., Marzari, N., & Mauri, F. (2019). Unified theory of thermal transport in crystals and glasses. *Nature Physics*, 15, 809-813. <https://doi.org/10.1038/s41567-019-0520-x>
- Quasi-harmonic Green-Kubo ("IBDB19"): Isaeva, L., Barbalinardo, G., Donadio, D., & Baroni, S. (2019). Modeling heat transport in crystals and glasses from a unified lattice-dynamical approach. *Nature Communications*, 10, 3853. <https://doi.org/10.1038/s41467-019-11572-4>
