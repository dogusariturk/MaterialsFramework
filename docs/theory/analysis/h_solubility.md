# H-Solubility

Evaluates the energetics of inserting a single hydrogen atom into octahedral and tetrahedral
interstitial sites of a BCC host, and reports the lowest-energy (preferred) site.

## Overview

Interstitial solubility of hydrogen governs hydrogen embrittlement and storage behavior in metals.
In a BCC lattice the two candidate interstitial families are octahedral sites (6 per conventional
cell, each with 2 nearest neighbors close and 4 farther away, the geometry relevant to H trapping
in ferritic steels) and tetrahedral sites (12 per conventional cell, 4 equidistant nearest
neighbors). `HSolubilityAnalyzer` inserts a hydrogen atom at each candidate site of each requested
family, relaxes (or, with `is_relaxed=True`, single-point evaluates) the resulting host+H structure,
and compares its energy against the clean host and a hydrogen reference. The host itself can be
supplied directly or generated for a given `composition` via `SqsGenerator` (through
`HSolubilityTransformation`). When the host is relaxed from scratch (`is_relaxed=False`), interstitial
sites are re-generated on the *relaxed* host so site positions reflect the relaxed lattice, not the
initial guess.

## Theory

For a host of energy \(E_{\text{host}}\) and a host-plus-hydrogen structure of energy
\(E_{\text{host+H}}\), the insertion energy at that site is

$$
E_{\text{ins}} = E_{\text{host+H}} - E_{\text{host}} - E_{\text{H}}^{\text{ref}}
$$

where \(E_{\text{H}}^{\text{ref}}\) is the `hydrogen_reference_energy` supplied at construction:
typically half the energy of an isolated H\(_2\) molecule, \(\tfrac{1}{2} E(\text{H}_2)\), so that
\(E_{\text{ins}}\) measures the energy cost of moving a hydrogen atom from the gas-phase molecule
into the lattice. \(E_{\text{ins}}\) is computed for every generated site of every requested
`site_types` family, and the minimum within each family is kept:

$$
E_{\text{ins}}^{\text{min}} = \min_{\text{sites}} E_{\text{ins}}
$$

The overall `solution_energy` is the smaller of the octahedral and tetrahedral minima, and
`preferred_site_type` records which family it came from. Octahedral sites win when their minimum is
less than or equal to the tetrahedral one. A lower solution energy means hydrogen more readily
dissolves into that site family.

## References

- Kirchheim, R. (1988). Hydrogen solubility and diffusivity in defective and amorphous metals.
  *Progress in Materials Science*, 32, 261-325. <https://doi.org/10.1016/0079-6425(88)90010-2>
