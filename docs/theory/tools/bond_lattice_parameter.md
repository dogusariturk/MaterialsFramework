# Bond Lattice Parameter

Predicts the lattice parameter of an FCC, BCC, or HCP alloy from pairwise nearest-neighbor bond lengths extracted via MLIP relaxations, rather than by linearly interpolating pure-element lattice parameters.

## Overview

Vegard's law (a linear interpolation of pure-element lattice parameters by composition) ignores that differently-sized atoms don't simply average their sizes when they actually bond to each other; real alloys often show bond-length contraction or expansion between dissimilar species relative to that naive average. `BondLatticeParameter` instead relaxes a small set of reference cells with a `BaseCalculator` (the pure element in its target structure, plus every binary pair as an ordered intermetallic: L1₀ for FCC, B2 for BCC, D0₁₉ for HCP), extracts each cell's true first-nearest-neighbor (FNN) bond length, and combines those bond lengths, weighted by composition, into a predicted alloy lattice parameter. This requires only \(N\) pure-element and \(\binom{N}{2}\) binary relaxations for an \(N\)-element system, rather than relaxing every alloy composition of interest directly.

## Theory

### Reference bond lengths

For each element \(i\), `calculate()` relaxes a conventional pure cell and extracts its FNN bond length \(d_{ii}\) from the relaxed lattice constant \(a\) (and \(c\) for HCP):

$$
d_{ii} = \begin{cases}
a/\sqrt{2} & \text{FCC} \\
a\sqrt{3}/2 & \text{BCC} \\
\sqrt{a^2/3 + c^2/4} & \text{HCP}
\end{cases}
$$

For each pair \((i, j)\), it relaxes a binary intermetallic built at the average of the two pure lattice parameters (L1₀ for FCC, B2 for BCC, D0₁₉ for HCP) and extracts the unlike-species FNN bond length \(d_{ij}\) the same way (L1₀ uses its tetragonal \(a\), \(c\); B2 uses the same cubic formula as BCC; D0₁₉ uses the same hexagonal formula as HCP, evaluated on the minority-majority bond).

### Alloy prediction

`predict(composition)` combines the bond table into a composition-weighted average bond length,

$$
\bar{d} = \sum_{i,j} x_i\, x_j\, d_{ij}
$$

summed over every ordered pair (including \(i=j\)) with mole fractions \(x_i\), then converts \(\bar d\) back to a lattice parameter with the same structure-specific relation used to extract \(d_{ii}\) above (inverted): \(a = \sqrt{2}\,\bar d\) for FCC, \(a = 2\bar d/\sqrt{3}\) for BCC, and \(a = \bar d\) for HCP (exact for an ideal \(c/a\) ratio). Because \(d_{ij}\) is measured directly from a relaxed unlike-pair cell rather than assumed, this captures bond-length non-ideality that a plain Vegard average misses. For comparison, `vegard(composition)` computes that plain Vegard's-law baseline directly from the pure-element lattice parameters:

$$
\bar{a}_{\text{Vegard}} = \sum_i x_i\, a_i
$$

`from_csv()` builds a prediction-only model from a precomputed symmetric bond-length matrix, skipping the relaxation step entirely; pure lattice parameters are recovered from the diagonal (\(a_i = d_{ii}\sqrt{2}\) for FCC, \(a_i = 2d_{ii}/\sqrt{3}\) for BCC, \(a_i = d_{ii}\) for HCP).

## References

- Tandoc, C., Qi, L., & Hu, Y.-J. (2025). A bond-based model for accurate prediction of lattice parameters of bcc solid solution alloys. *Materialia*, 40, 102410. <https://doi.org/10.1016/j.mtla.2025.102410>
