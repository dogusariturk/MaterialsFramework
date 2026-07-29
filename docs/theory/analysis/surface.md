# Surface Energy

Computes the surface (cleavage) energy of one or more slab terminations for a given Miller index.

## Overview

`SurfaceAnalyzer` relaxes the bulk structure (cell and atoms) to obtain a reference energy per atom, then uses `SurfaceTransformation` to build one or more inequivalent, vacuum-padded slab terminations for the configured `miller_index` (via `pymatgen.core.surface.SlabGenerator`). Each slab is relaxed with the cell fixed (only atomic positions move, so the vacuum spacing and slab thickness are preserved exactly as generated), and its surface energy is extracted from the difference between the relaxed slab energy and the energy an equivalent slab of bulk material would have.

## Theory

For a slab of \(N\) atoms and total energy \(E_{\text{slab}}\), with bulk reference energy per atom \(E_{\text{bulk}}\) and surface area \(A\) exposed on *each* of the slab's two faces, the surface energy is

$$
\gamma = \frac{E_{\text{slab}} - N E_{\text{bulk}}}{2A}
$$

the standard supercell-method surface energy. Division by \(2A\) rather than \(A\) accounts for the two free surfaces a slab necessarily creates. This estimate is only meaningful once `min_slab_size` is large enough that the slab's interior recovers bulk-like coordination (so the two surfaces don't interact through the slab) and `min_vacuum_size` is large enough that a slab doesn't interact with its own periodic image across the vacuum gap. Both are hkl-plane-count or Angstrom thicknesses depending on `in_unit_planes`. `symmetrize=True` (the default) restricts slab generation to terminations with the same surface on both faces, which is what makes the simple "divide by \(2A\)" form above valid without needing to separately resolve two different surface energies for an asymmetric slab. `gamma_J_m2` reports the same quantity converted from eV/Å² to J/m².

## References

- Boettger, J.C. (1994). Nonconvergence of surface energies obtained from thin-film calculations. *Physical Review B*, 49(23), 16798-16800. <https://doi.org/10.1103/PhysRevB.49.16798>
- Fiorentini, V., & Methfessel, M. (1996). Extracting convergent surface energies from slab calculations. *Journal of Physics: Condensed Matter*, 8(36), 6525-6529. <https://doi.org/10.1088/0953-8984/8/36/005>
