# Elastic Constants

Computes the full second-order elastic constant tensor of a structure from stress-strain data, along
with the isotropic mechanical properties derived from it.

## Overview

`ElasticConstantsAnalyzer` applies a series of small Cartesian and shear deformations to the
(optionally pre-relaxed) input cell, evaluates the stress response of each with the calculator, and
fits the elastic tensor \(C_{ij}\) by least squares. Unlike
[`CubicElasticConstantsAnalyzer`](cubic_elastic.md), it is not restricted to cubic cells: the crystal
system is detected automatically from the cell shape (triclinic through cubic), and only the
independent \(C_{ij}\) components required for that symmetry are fit, e.g. 3 for cubic
(\(C_{11}, C_{12}, C_{44}\)), 5 for hexagonal, 6 for tetragonal, up to 21 for triclinic.

From the fitted tensor, the analyzer also reports the standard polycrystalline-average mechanical
properties: bulk and shear modulus (Voigt, Reuss, and Voigt-Reuss-Hill), Young's modulus, Poisson's
ratio, Pugh's ratio, and Chen-Vickers hardness.

## Theory

### Elastic tensor fit

Generalized Hooke's law relates stress and strain (Voigt notation, indices 1-6) linearly:

$$
\sigma_i = \sum_{j=1}^{6} C_{ij}\,\varepsilon_j
$$

For each of `num_deform` deformation magnitudes (up to `max_deform`, in percent for axial strain and
degrees for shear) along every Cartesian/shear direction allowed by the cell's symmetry, the analyzer
records the resulting strain \(\varepsilon\) (relative to the reference cell) and stress \(\sigma\)
(with the ambient pressure of the reference cell subtracted out). Stacking these into one linear
system per symmetry class and solving by least squares yields the independent \(C_{ij}\) for that
Bravais lattice. Constants are reported in GPa.

### Voigt, Reuss, and Hill averages

A real polycrystal is an aggregate of randomly oriented single-crystal grains, so it has no single
well-defined bulk/shear modulus, only bounds. The Voigt average assumes uniform strain across grains
(an upper bound):

$$
K_V = \tfrac{1}{9}\big[(C_{11}+C_{22}+C_{33}) + 2(C_{12}+C_{13}+C_{23})\big], \qquad
G_V = \tfrac{1}{15}\big[(C_{11}+C_{22}+C_{33}) - (C_{12}+C_{13}+C_{23}) + 3(C_{44}+C_{55}+C_{66})\big]
$$

The Reuss average assumes uniform stress instead (a lower bound), using the compliance tensor
\(S = C^{-1}\):

$$
K_R = \big[(S_{11}+S_{22}+S_{33}) + 2(S_{12}+S_{13}+S_{23})\big]^{-1}, \qquad
\frac{1}{G_R} = \tfrac{1}{15}\big[4(S_{11}+S_{22}+S_{33}) - 4(S_{12}+S_{13}+S_{23}) + 3(S_{44}+S_{55}+S_{66})\big]
$$

The Voigt-Reuss-Hill (VRH) average, used as the "best estimate" polycrystalline modulus, is simply
their arithmetic mean: \(K_{VRH} = (K_V+K_R)/2\), \(G_{VRH} = (G_V+G_R)/2\).

### Derived mechanical properties

From \(K_{VRH}\) and \(G_{VRH}\):

$$
E = \frac{9 K_{VRH} G_{VRH}}{3K_{VRH}+G_{VRH}}, \qquad
\nu = \frac{3K_{VRH}-2G_{VRH}}{2(3K_{VRH}+G_{VRH})}, \qquad
k = \frac{G_{VRH}}{K_{VRH}}
$$

\(E\) is Young's modulus (GPa) and \(\nu\) is Poisson's ratio. \(k\) is Pugh's ratio. Empirically,
metals with \(k \lesssim 0.57\) tend to be ductile, and above that, brittle. Finally, the
Chen-Vickers hardness estimate is

$$
H_v = 2\,(k^2 G_{VRH})^{0.585} - 3
$$

in GPa.

## References

- Hill, R. (1952). The elastic behaviour of a crystalline aggregate. *Proceedings of the Physical
  Society. Section A*, 65(5), 349-354. <https://doi.org/10.1088/0370-1298/65/5/307>
- Pugh, S.F. (1954). Relations between the elastic moduli and the plastic properties of polycrystalline
  pure metals. *Philosophical Magazine*, 45(367), 823-843. <https://doi.org/10.1080/14786440808520496>
- Chen, X.-Q., Niu, H., Li, D., & Li, Y. (2011). Modeling hardness of polycrystalline materials and
  bulk metallic glasses. *Intermetallics*, 19(9), 1275-1281. <https://doi.org/10.1016/j.intermet.2011.03.026>
