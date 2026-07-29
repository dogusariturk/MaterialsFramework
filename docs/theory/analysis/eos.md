# Equation of State

Fits energy-volume data to an equation of state (EOS) to extract equilibrium volume, equilibrium energy, and bulk modulus.

## Overview

`EOSAnalyzer` generates a series of isotropically strained copies of a structure, computes the potential energy of each at fixed cell shape (only ionic positions relax; `EOSTransformation` scales the lattice vectors directly), and fits the resulting energy-volume curve to a chosen analytic equation of state via `pymatgen.analysis.eos.EOS`. The fit yields the equilibrium volume \(V_0\), equilibrium energy \(E_0\), bulk modulus \(B_0\), and the pressure derivative of the bulk modulus \(B_0'\), the same quantities normally extracted from a cold-compression curve.

`EOSTransformation` applies `num` evenly spaced linear strains between `start` and `stop` (defaults: 11 points from -10% to +10%) to every lattice vector uniformly, so the deformed volumes span \((1+\text{start})^3 V \le V \le (1+\text{stop})^3 V\). At least 3 points are required to fit an EOS.

## Theory

The default `eos_name="birch_murnaghan"` fits the third-order Birch-Murnaghan form (as implemented in `pymatgen.analysis.eos`, following Phys. Rev. B 70, 224107):

$$
E(V) = E_0 + \frac{9 B_0 V_0}{16} \left(\eta^2 - 1\right)^2 \left[6 + B_0'\left(\eta^2 - 1\right) - 4\eta^2\right],
\qquad \eta = \left(\frac{V_0}{V}\right)^{1/3}
$$

- \(E_0\): equilibrium (minimum) energy.
- \(V_0\): equilibrium volume, where \(dE/dV = 0\).
- \(B_0 = V \, d^2E/dV^2 \big|_{V_0}\): the bulk modulus at equilibrium, i.e. the curvature of the
  energy-volume well; stiffer materials have deeper, narrower wells and larger \(B_0\).
- \(B_0' = dB/dP \big|_{V_0}\): the pressure derivative of the bulk modulus, describing how quickly
  the material stiffens under compression.

Other supported forms (`eos_name="murnaghan"`, `"birch"`, `"pourier_tarantola"`, `"vinet"`, `"deltafactor"`, `"numerical_eos"`) fit the same \(E(V)\) data with a different functional form for \(E(V)\); they return the same four quantities and are useful when the default Birch-Murnaghan fit is a poor match to the underlying data (e.g. under strong compression, where `"vinet"` tends to be more robust).

## References

- Birch-Murnaghan equation of state: Phys. Rev. B 70, 224107 (2004).
- `pymatgen.analysis.eos`: <https://doi.org/10.1016/j.commatsci.2012.10.028>
