# Cubic Elastic Constants

Computes \(C_{11}\), \(C_{12}\), and \(C_{44}\) for a cubic (or orthogonal) cell from three targeted energy-volume/energy-strain deformation modes, rather than a general stress-strain fit.

## Overview

`CubicElasticConstantsAnalyzer` is a lighter-weight, cubic-only alternative to the general [`ElasticConstantsAnalyzer`](elastic.md). Instead of fitting a full symmetry-dependent stress-strain system, it applies three specific distortion families to the input cell: uniform (volumetric), orthorhombic (tetragonal shear), and monoclinic (pure shear), and extracts the bulk modulus and the two independent shear-related moduli directly from the curvature of energy vs. distortion. This only requires the `energy` property (not `stress`), at the cost of only being valid for cubic or orthogonal cells.

!!! caution
    This analyzer only works with cubic or orthogonal cells.

## Theory

### Bulk modulus from uniform compression/expansion

The `uniform` distortion scales the cell isotropically by \((1+\delta)\) along all three axes for a range of \(\delta\) (`-delta_max` to `delta_max`, step `step_size`). The energy-volume curve is fit to an equation of state (`eos_name`, default Birch-Murnaghan; see [Equation of State](eos.md) for the functional form) to obtain the bulk modulus \(B\) directly in GPa.

### Shear moduli from volume-conserving distortions

The `orthorhombic` distortion (diag\((1+\delta, 1-\delta, 1/(1-\delta^2))\)) and the `monoclinic` distortion (a shear of magnitude \(\delta\) between the \(x\)/\(y\) axes, with \(z\) rescaled to preserve volume) are both volume-conserving to second order. For either family, the energy is quadratic in \(\delta\) near \(\delta=0\):

$$
E(\delta) \approx E(0) + a\,\delta^{2}
$$

A degree-2 polynomial fit to \(E(\delta)\) over the sampled \(\delta\) range gives the leading coefficient \(a\), from which the corresponding modulus follows as

$$
M = \frac{a}{2\,V_0}
$$

(in eV/Å³, converted to GPa) where \(V_0\) is the undistorted cell volume. Applied to the orthorhombic distortion this gives the tetragonal shear modulus \(G' = \tfrac{1}{2}(C_{11}-C_{12})\); applied to the monoclinic distortion it gives the shear modulus \(C_{44}\) directly.

### Assembling C11, C12, C44

$$
C_{11} = B + \tfrac{4}{3}G', \qquad C_{12} = B - \tfrac{2}{3}G', \qquad C_{44} = C_{44}
$$

From these three constants, the same Voigt-Reuss-Hill machinery described in [Elastic Constants](elastic.md#voigt-reuss-and-hill-averages) is used to derive the bulk/shear/Young's moduli, Poisson's ratio, Pugh's ratio, and Chen-Vickers hardness.

## References

- Hill, R. (1952). The elastic behaviour of a crystalline aggregate. *Proceedings of the Physical Society. Section A*, 65(5), 349-354. <https://doi.org/10.1088/0370-1298/65/5/307>
- Pugh, S.F. (1954). Relations between the elastic moduli and the plastic properties of polycrystalline pure metals. *Philosophical Magazine*, 45(367), 823-843. <https://doi.org/10.1080/14786440808520496>
- Chen, X.-Q., Niu, H., Li, D., & Li, Y. (2011). Modeling hardness of polycrystalline materials and bulk metallic glasses. *Intermetallics*, 19(9), 1275-1281. <https://doi.org/10.1016/j.intermet.2011.03.026>
