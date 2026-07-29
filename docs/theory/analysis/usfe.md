# Unstable Stacking Fault Energy (USFE)

Computes the generalized stacking fault energy (GSFE) curve for a rigid shear along a BCC slip
system, and extracts the unstable stacking fault energy (USFE) as its maximum.

## Overview

The GSFE curve traces the energy cost of shearing a crystal rigidly across a candidate slip plane,
one atomic block sliding over the other, as a function of the shift between them. Its peak (the
USFE) is the energy barrier a dislocation must overcome to glide across that plane, and is a
standard proxy for a material's resistance to slip (lower USFE tends to mean easier dislocation
glide and lower ductile-to-brittle strength contrast). `USFEAnalyzer` supports the `"110"` and
`"112"` BCC slip systems. The host structure is relaxed first (unless `is_relaxed=True`), then
`USFETransformation` builds the sheared configurations that get evaluated with the calculator.

## Theory

For slip system `slip_plane`, the transformation resolves a Cartesian unit normal \(\hat{n}\) to
the Miller plane and a Cartesian unit vector \(\hat{s}\) along the corresponding slip direction,
projected into the plane (component along \(\hat{n}\) removed, then re-normalized). Atoms on one
side of the plane (split at the median of their projection onto \(\hat{n}\)) are rigidly translated
by

$$
\mathbf{d}(f) = f \cdot (0.5\,\hat{s}), \qquad f \in [\text{start}, \text{stop}]
$$

where \(f\) is the displacement fraction (`num_steps` evenly spaced values between `start` and
`stop`, default \([0, 1]\)). Note the shift magnitude at \(f=1\) is fixed at 0.5 Å along \(\hat{s}\)
regardless of the lattice parameter. Pick `stop` to reach the physical repeat distance of interest
for your slip system rather than assuming \(f=1\) always spans a full period.

For each \(f\), the excess energy per unit fault area is

$$
\gamma(f) = \frac{E(f) - E(f_0)}{A}
$$

with \(f_0\) the first sampled fraction (`start`, by default the undisplaced structure), and \(A\)
the fault-plane cross-sectional area (the magnitude of the cross product of the two lattice vectors
not aligned with \(\hat{n}\)). \(\gamma\) is converted from eV/Å² to mJ/m² (\(1\ \text{eV/Å}^2 =
16021.76634\ \text{mJ/m}^2\)).

The unstable stacking fault energy is the maximum of the GSFE curve:

$$
\gamma_{\text{us}} = \max_f \gamma(f)
$$

reported together with the fraction \(f\) at which it occurs.

## References

- Vitek, V. (1968). Intrinsic stacking faults in body-centred cubic crystals. *Philosophical
  Magazine*, 18(154), 773-786. <https://doi.org/10.1080/14786436808227500>
