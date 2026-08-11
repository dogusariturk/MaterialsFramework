# Elastic Constants

Computes the full second-order elastic constant tensor of a structure from stress-strain data, along with the isotropic mechanical properties derived from it.

## Overview

`ElasticConstantsAnalyzer` applies a series of small Cartesian and shear deformations to the (optionally pre-relaxed) input cell, then measures the stress response of each deformed structure with the calculator. By default, the atoms are not allowed to move during that measurement (see [Letting the atoms relax during deformation](#letting-the-atoms-relax-during-deformation) below); the resulting stress-strain data is then fit to the elastic tensor \(C_{ij}\) by least squares. Unlike [`CubicElasticConstantsAnalyzer`](cubic_elastic.md), it is not restricted to cubic cells: the crystal system is detected automatically from the cell shape (triclinic through cubic), and only the independent \(C_{ij}\) components required for that symmetry are fit, e.g. 3 for cubic (\(C_{11}, C_{12}, C_{44}\)), 5 for hexagonal, 6 for tetragonal, up to 21 for triclinic.

From the fitted tensor, the analyzer also reports the standard polycrystalline-average mechanical properties (bulk and shear modulus, Young's modulus, Poisson's ratio, Pugh's ratio, Chen-Vickers hardness), plus the Debye temperature, a Grüneisen parameter approximation, and elastic wave speeds.

## Theory

### Elastic tensor fit

Generalized Hooke's law relates stress and strain (Voigt notation, indices 1-6) linearly:

$$
\sigma_i = \sum_{j=1}^{6} C_{ij}\,\varepsilon_j
$$

For each of `num_deform` deformation magnitudes (up to `max_deform`, in percent for axial strain and degrees for shear) along every Cartesian/shear direction allowed by the cell's symmetry, the analyzer records the resulting strain \(\varepsilon\) and stress increment relative to the reference cell. If the reference cell is already under stress, the measured Cauchy-stress increment also contains a geometric initial-stress contribution from the change of reference frame. The analyzer removes the linearized Piola-transformation term, including its rotation term for non-hydrostatic stress, so the fit yields the thermodynamic elastic constants \(C_{ij}\). Stacking the corrected responses into one linear system per symmetry class and solving by least squares yields the independent \(C_{ij}\). Constants are reported in GPa.

### Letting the atoms relax during deformation

An applied strain can also shift the atoms inside a cell with more than one atom per primitive cell, not just its overall shape. By default (`relax_ions=False`), the analyzer measures each deformed structure's stress without ionic relaxation. `relax_ions=True` instead lets the atoms settle into new positions at that same, fixed cell shape first, producing *relaxed-ion* elastic constants at the cost of one extra relaxation per deformation. For a single-atom primitive cell (e.g. elemental fcc/bcc metals), the two approaches coincide. `fmax_distort` overrides the calculator's `fmax` for these relaxations only, restoring it afterward; it has no effect when `relax_ions=False`.

### Voigt, Reuss, and Hill averages

A real polycrystal is an aggregate of randomly oriented single-crystal grains, so it has no single well-defined bulk/shear modulus, only bounds. The Voigt average assumes uniform strain across grains (an upper bound):

$$
K_V = \tfrac{1}{9}\big[(C_{11}+C_{22}+C_{33}) + 2(C_{12}+C_{13}+C_{23})\big], \qquad
G_V = \tfrac{1}{15}\big[(C_{11}+C_{22}+C_{33}) - (C_{12}+C_{13}+C_{23}) + 3(C_{44}+C_{55}+C_{66})\big]
$$

The Reuss average assumes uniform stress instead (a lower bound), using the compliance tensor \(S = C^{-1}\):

$$
K_R = \big[(S_{11}+S_{22}+S_{33}) + 2(S_{12}+S_{13}+S_{23})\big]^{-1}, \qquad
\frac{1}{G_R} = \tfrac{1}{15}\big[4(S_{11}+S_{22}+S_{33}) - 4(S_{12}+S_{13}+S_{23}) + 3(S_{44}+S_{55}+S_{66})\big]
$$

The Voigt-Reuss-Hill (VRH) average, used as the "best estimate" polycrystalline modulus, is simply their arithmetic mean: \(K_{VRH} = (K_V+K_R)/2\), \(G_{VRH} = (G_V+G_R)/2\).

### Derived mechanical properties

From \(K_{VRH}\) and \(G_{VRH}\):

$$
E = \frac{9 K_{VRH} G_{VRH}}{3K_{VRH}+G_{VRH}}, \qquad
\nu = \frac{3K_{VRH}-2G_{VRH}}{2(3K_{VRH}+G_{VRH})}, \qquad
k = \frac{G_{VRH}}{K_{VRH}}
$$

\(E\) is Young's modulus (GPa) and \(\nu\) is Poisson's ratio. \(k\) is Pugh's ratio. Empirically, metals with \(k \lesssim 0.57\) tend to be ductile, and above that, brittle. Finally, the Chen-Vickers hardness estimate is

$$
H_v = 2\,(k^2 G_{VRH})^{0.585} - 3
$$

in GPa.

### Elastic wave speeds and Debye temperature

Assuming an isotropic elastic continuum of mass density \(\rho\), the transverse and longitudinal sound velocities are

$$
v_t = \sqrt{\frac{G_{VRH}}{\rho}}, \qquad
v_l = \sqrt{\frac{K_{VRH} + \tfrac{4}{3}G_{VRH}}{\rho}}
$$

reported as `v_transverse` and `v_longitudinal` (m/s), with their weighted mean

$$
v_m = \left[\frac{1}{3}\left(\frac{2}{v_t^3} + \frac{1}{v_l^3}\right)\right]^{-1/3}
$$

as `v_mean`. The Debye temperature, reported as `debye_temperature` (K), follows from \(v_m\) and the atomic number density \(n\):

$$
\Theta_D = \frac{\hbar}{k_B}\,v_m\,\left(6\pi^2 n\right)^{1/3}
$$

### Grüneisen parameter approximation

The Grüneisen parameter \(\gamma\) describes how vibrational frequencies change with volume, but its rigorous calculation requires volume-dependent vibrational data. `gruneisen_approx` uses the Belomestnykh–Tesleva relation to estimate the acoustic Grüneisen parameter of a quasi-isotropic solid from its homogeneous Poisson ratio:

$$
\gamma \approx \frac{3}{2}\,\frac{1+\nu}{2-3\nu}
$$

\(\gamma\) is dimensionless.

## References

- Hill, R. (1952). The elastic behaviour of a crystalline aggregate. *Proceedings of the Physical Society. Section A*, 65(5), 349-354. <https://doi.org/10.1088/0370-1298/65/5/307>
- Pugh, S.F. (1954). Relations between the elastic moduli and the plastic properties of polycrystalline pure metals. *Philosophical Magazine*, 45(367), 823-843. <https://doi.org/10.1080/14786440808520496>
- Chen, X.-Q., Niu, H., Li, D., & Li, Y. (2011). Modeling hardness of polycrystalline materials and bulk metallic glasses. *Intermetallics*, 19(9), 1275-1281. <https://doi.org/10.1016/j.intermet.2011.03.026>
- Anderson, O.L. (1963). A simplified method for calculating the Debye temperature from elastic constants. *Journal of Physics and Chemistry of Solids*, 24(7), 909-917. <https://doi.org/10.1016/0022-3697(63)90067-2>
- Belomestnykh, V.N., & Tesleva, E.P. (2004). Interrelation between anharmonicity and lateral strain in quasi-isotropic polycrystalline solids. *Technical Physics*, 49(8), 1098-1100. <https://doi.org/10.1134/1.1787679>
- Mouhat, F., & Coudert, F.-X. (2014). Necessary and sufficient elastic stability conditions in various crystal systems. *Physical Review B*, 90(22), 224104. <https://doi.org/10.1103/PhysRevB.90.224104>
- Luo, C., Tromp, J., & Wentzcovitch, R.M. (2022). Ab initio calculations of third-order elastic coefficients. *Physical Review B*, 106(21), 214104. <https://doi.org/10.1103/PhysRevB.106.214104>
