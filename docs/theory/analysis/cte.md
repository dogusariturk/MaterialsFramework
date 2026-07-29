# Coefficient of Thermal Expansion

Estimates the volumetric coefficient of thermal expansion from NPT molecular dynamics run at
several temperatures.

## Overview

The coefficient of thermal expansion (CTE) measures how much a material's volume changes with
temperature at constant pressure. `CTEAnalyzer` gets there empirically rather than from a
closed-form model: it runs an NPT MD simulation at each temperature in `temperatures` (default
300, 600, 900 K), lets the cell reach its equilibrium volume under the target pressure, and fits a
line through the resulting volume-temperature points. This requires an MD-capable calculator (a
`BaseMDCalculator` subclass); a plain `BaseCalculator` without a `run()` method raises. The MD
ensemble (`ensemble`, default `"npt_berendsen"`) and target `pressure` (default 1.0 atm) are applied
identically at every temperature.

## Theory

The volumetric CTE is the thermodynamic derivative

$$
\alpha_V = \frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_P
$$

`CTEAnalyzer` approximates this by running NPT-MD at each configured temperature \(T_i\), reading
off the final cell volume \(V_i\) once the run completes, and fitting a straight line through the
\((T_i, V_i)\) points:

$$
V(T) \approx V_{\text{ref}} + m\,(T - T_{\text{ref}})
$$

where \(m = dV/dT\) is the fitted slope and \(V_{\text{ref}}\) is the volume at the lowest sampled
temperature \(T_{\text{ref}}\). The volumetric CTE is then the slope normalized by that reference
volume:

$$
\alpha_V \approx \frac{m}{V_{\text{ref}}}
$$

reported both in \(\text{K}^{-1}\) (`cte`) and in ppm/K (`cte_ppm`, \(\alpha_V \times 10^6\)). Because
this is a linear fit, accuracy improves with more temperature points and longer runs (`steps`) that
let the cell fully equilibrate at each temperature; at least two distinct temperatures are required.
