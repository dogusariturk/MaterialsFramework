# ANNNI Stacking Faults

Estimates intrinsic and extrinsic stacking fault energies of an FCC-forming composition from the
relaxed energies of three bulk polytypes, using the second-order axial next-nearest-neighbor Ising
(ANNNI) model.

## Overview

Unlike [USFE](usfe.md), which shears a supercell rigidly across a slip plane and evaluates the
resulting energy landscape directly, the ANNNI approach never builds a faulted supercell at all. A
stacking fault is a local change in the sequence of close-packed planes, and the ANNNI model treats
each plane as an Ising-like spin whose interactions with nearby planes are inferred from the
energies of small-period stacking sequences: FCC (`ABCABC...`), HCP (`ABAB...`), and DHCP
(`ABACABAC...`, a doubled HCP-like period). To second order in the axial interactions, the intrinsic
and extrinsic stacking fault energies can be written purely in terms of the (bulk) energies of these
three polytypes. No explicit fault geometry or supercell is needed. `ANNNIStackingFaultAnalyzer`
builds all three structures for a given composition via `SqsGenerator` (through
`ANNNIStackingFaultTransformation`), relaxes the FCC cell, and evaluates HCP/DHCP at the same
volume per atom as the relaxed FCC cell so all three energies are compared at a common density.

## Theory

Let \(E_{\text{fcc}}\), \(E_{\text{hcp}}\), and \(E_{\text{dhcp}}\) be the relaxed energies per atom
of the FCC, HCP, and DHCP polytypes, all evaluated at the FCC equilibrium volume per atom. Let
\(a\) be the conventional FCC lattice parameter recovered from that volume (\(V_{\text{atom}} =
a^3/4\), i.e. 4 atoms per conventional cubic cell), and

$$
A = \frac{\sqrt{3}}{4}\,a^2
$$

the area per atom of an FCC \(\{111\}\) plane. The second-order ANNNI formulae for the intrinsic
stacking fault energy (ISFE) and extrinsic stacking fault energy (ESFE) are:

$$
\gamma_{\text{ISFE}} = \frac{E_{\text{hcp}} + 2 E_{\text{dhcp}} - 3 E_{\text{fcc}}}{A}
$$

$$
\gamma_{\text{ESFE}} = \frac{4 \left( E_{\text{dhcp}} - E_{\text{fcc}} \right)}{A}
$$

Both are reported in eV/Å². An intrinsic fault corresponds to a single missing plane in the FCC
stacking sequence (locally HCP-like across one plane); an extrinsic fault corresponds to a single
inserted plane (locally HCP-like across two planes, i.e. a thin DHCP-like slab). This is why ISFE
mixes all three polytypes while ESFE only contrasts DHCP against FCC.

## References

- Denteneer, P. J. H., & van Haeringen, W. (1987). Stacking-fault energies in semiconductors from
  first-principles calculations. *Journal of Physics C: Solid State Physics*, 20(32), L883-L887.
  <https://doi.org/10.1088/0022-3719/20/32/001>
