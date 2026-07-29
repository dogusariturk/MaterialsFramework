# Formation Energy

Computes the formation energy per atom of a compound relative to the relaxed ground states of its
constituent elements.

## Overview

Formation energy measures how much more (or less) stable a compound is than a mechanical mixture of
its constituent elements in their reference states. A negative value means the compound is
thermodynamically favorable to form from the pure elements. `FormationEnergyAnalyzer` relaxes the
input compound, relaxes an elemental reference structure for every element present, and combines the
two into an energy per atom.

For each element, `FormationEnergyTransformation` first tries to build its experimentally-tabulated
ground state (`ase.data.reference_states`, the table `ase.build.bulk` also uses): gas-phase diatomic
elements (H, N, O, F) become an isolated dimer in a vacuum box, noble gases become an isolated atom,
and everything else with a simple Bravais lattice (FCC, BCC, HCP, diamond cubic, etc.) is built
directly from its known symmetry. A handful of elements (e.g. Mn, P, S, Ga, and elements with no
tabulated reference state at all) have no ground state `ase.build.bulk` can construct from a formula
alone. For these, several candidate high-symmetry lattices (FCC, BCC, HCP, diamond, simple cubic)
are estimated from the element's atomic radius, all are relaxed with the same calculator, and the
lowest-energy candidate is used as the reference. This is why the results dictionary flags each
element with `is_guessed`.

Each element's relaxed reference energy is cached on the analyzer instance the first time it's
needed, so calling `calculate()` again (even on a completely different structure that shares an
element) reuses the cached reference instead of relaxing it again. Construct a new
`FormationEnergyAnalyzer` to force fresh relaxations.

## Theory

For a compound with total energy \(E_{\text{compound}}\) and \(N\) atoms, containing \(n_i\) atoms
of element \(i\) with elemental reference energy per atom \(E_i^{\text{ref}}\):

$$
E_f = \frac{E_{\text{compound}} - \sum_i n_i E_i^{\text{ref}}}{N}
$$

\(E_f\) is the formation energy per atom (eV/atom). Both \(E_{\text{compound}}\) and each
\(E_i^{\text{ref}}\) come from relaxing the respective structure with the same calculator, so
systematic errors in the underlying MLIP largely cancel between the compound and its references.
\(E_i^{\text{ref}}\) is the lowest energy per atom found among the candidate reference structures
for element \(i\) (a single, known ground state in the common case, or the best of several guessed
high-symmetry candidates otherwise).

## References

- Ground-state crystal structures: `ase.data.reference_states` (the same table used by `ase.build.bulk`).
