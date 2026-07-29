# Surface Binding Energy

Screens slab terminations for the lowest-surface-energy one, then computes the per-site energy needed to remove a single surface atom to an isolated reference state: a first-principles proxy for that surface's resistance to physical sputtering.

## Overview

`SBEAnalyzer` follows the surface-binding-energy (SBE) workflow of Fedrigucci, Marzari, & Ricci (2024), developed there to screen candidate plasma-facing materials (PFMs) for the divertor of a fusion reactor, where erosion of the exposed surface by physical sputtering limits a material's usable lifetime. SBE is used in that context as a cheaper, first-principles-consistent stand-in for the elemental sublimation enthalpies that sputtering yield estimates have traditionally relied on.

The workflow: the bulk structure is relaxed first for a reference energy per atom. Then, exactly like `SurfaceAnalyzer`, `SBETransformation` generates every inequivalent slab termination across every Miller index up to `max_index` (via `pymatgen.core.surface.generate_all_slabs`), each is relaxed with the cell fixed, and its surface energy is computed the same way. The single termination with the lowest surface energy, across every screened Miller index, is kept as `best_miller_index`. For every termination of that best Miller index, a larger supercell is built from a fixed `supercell_size` (default `[4, 4, 1]`); the paper instead sizes each supercell to keep periodic vacancy images at least 9 Angstrom apart, a criterion that scales with a termination's in-plane lattice vectors instead of using a fixed replication factor. Surface sites of that supercell are identified with `pymatgen.analysis.adsorption.AdsorbateSiteFinder` (sites within `height` Angstrom of the topmost site along the surface normal), and a single-atom vacancy structure is generated for each one and relaxed with the cell fixed. The paper leaves that vacancy structure unrelaxed "for efficiency" (Sec. II B 1, step 4); `SBEAnalyzer` relaxes it anyway, trading some of that efficiency for a more physical vacancy energy at the cost of one extra relaxation per surface site.

## Theory

### Why surface binding energy tracks sputtering

Physical sputtering ejects a surface atom when a plasma particle transfers enough kinetic energy to it in a head-on elastic collision. For a projectile of mass \(m_p\) striking a surface atom of mass \(m_a\) bound with energy \(E_{\text{SB}}\), momentum and energy conservation give the minimum projectile energy needed:

$$
E_p \ge E_T = E_{\text{SB}} \, \frac{(m_p + m_a)^2}{4 \, m_p \, m_a}
$$

so a larger \(E_{\text{SB}}\) directly raises the sputtering threshold. The energy distribution of already-sputtered atoms follows the Thompson energy spectrum,

$$
\Phi(E_e) \propto \frac{E_e}{(E_e + E_{\text{SB}})^3},
$$

which peaks at \(E_e = E_{\text{SB}}/2\), again tied directly to \(E_{\text{SB}}\). Both relations are why `SBEAnalyzer` treats SBE as a proxy for a material's erosion resistance rather than as an end in itself.

### Surface energy screening

For each generated slab termination of a given Miller index \(hkl\), with \(N_s\) atoms, total energy \(E_s\), and surface area \(A_s\) exposed on each of its two faces, the surface energy is

$$
\gamma^{hkl} = \frac{E_s^{hkl} - \dfrac{N_s^{hkl}}{N_b} E_b}{2 A_s^{hkl}}
$$

where \(E_b\) and \(N_b\) are the relaxed bulk structure's total energy and atom count (so \(E_b / N_b\) is `bulk_energy_per_atom`). Across every screened Miller index, the termination with the lowest \(\gamma\) is kept as `best_miller_index`, and every termination belonging to that Miller index goes on to the vacancy step below.

### Surface binding energy

For a supercell built from a kept termination, with perfect-supercell energy \(E_s\) and the energy of that supercell with one surface atom of element \(a\) removed \(E_{s+v}\), the surface binding energy at that site is

$$
E_{\text{SB}} = E_a + E_{s+v} - E_s
$$

where \(E_a\) is the energy of an isolated atom of that element, computed in a large, non-interacting cubic cell (`isolated_atom_box_size`, 20 Angstrom by default here, versus the paper's cell of side \(4000^{1/3} \approx 15.87\) Angstrom). This is the same defect-formation-energy form used for bulk point defects, applied at a surface site and referenced to an isolated (gas-phase-like) atom rather than to the bulk chemical potential. A large, positive \(E_{\text{SB}}\) means that surface atom is strongly bound in place; a small or negative value flags a weakly-bound site.

`calculate()` reports \(E_{\text{SB}}\) for every surface site of every termination of `best_miller_index` (`vacancy_results`) and each termination's per-element mean (`avg_surface_binding_energy_by_element` inside `terminations`), matching how the paper itself averages over inequivalent sites of the same element within one termination (Sec. II B 1, step 6). `avg_surface_binding_energy` then goes one step further than the paper and averages those per-termination means across every termination of `best_miller_index`, weighting each termination equally regardless of how many surface sites it has.

## References

- Fedrigucci, A., Marzari, N., & Ricci, P. (2024). Comprehensive screening of plasma-facing materials for nuclear fusion. *PRX Energy*, 3, 043002. <https://doi.org/10.1103/PRXEnergy.3.043002>
- Thompson, M.W. (1968). II. The energy spectrum of ejected atoms during the high energy sputtering of gold. *Philosophical Magazine*, 18(152), 377-414. <https://doi.org/10.1080/14786436808227358>
- Montoya, J.H., & Persson, K.A. (2017). A high-throughput framework for determining adsorption energies on solid surfaces. *npj Computational Materials*, 3, 14. <https://doi.org/10.1038/s41524-017-0017-z>
- Freysoldt, C., Grabowski, B., Hickel, T., Neugebauer, J., Kresse, G., Janotti, A., & Van de Walle, C.G. (2014). First-principles calculations for point defects in solids. *Reviews of Modern Physics*, 86(1), 253-305. <https://doi.org/10.1103/RevModPhys.86.253>
