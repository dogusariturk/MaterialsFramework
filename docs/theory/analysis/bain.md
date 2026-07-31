# Bain Path

Traces the potential energy along the Bain transformation path connecting the BCC and FCC structures through a continuous tetragonal distortion.

## Overview

The Bain path is the classical continuous strain path between body-centered cubic (BCC) and face-centered cubic (FCC): both can be described as the same body-centered tetragonal (BCT) lattice at different \(c/a\) ratios (\(c/a=1\) is BCC; \(c/a=\sqrt{2}\approx1.41\) is FCC), so a single volume-conserving tetragonal strain interpolates between them. `BainPathAnalyzer` standardizes the input structure to a conventional cell, applies this strain over a range of \(c/a\), and evaluates the potential energy at each point with the calculator. The resulting energy vs. \(c/a\) curve is a common tool for reasoning about martensitic (BCC/FCC-type) transformations: local minima mark mechanically stable structures, and the energy barrier between them is a proxy for the transformation's resistance.

## Theory

For a target ratio \(c/a\), the transformation is applied as a volume-conserving tetragonal deformation of the conventional cell:

$$
\mathbf{F}(\delta) =
\begin{pmatrix}
1+\delta & 0 & 0 \\
0 & 1+\delta & 0 \\
0 & 0 & \dfrac{1}{(1+\delta)^2}
\end{pmatrix},
\qquad
\delta = \sqrt[3]{\dfrac{1}{c/a}} - 1
$$

\(\delta\) is chosen so that the in-plane axes both scale by \(1+\delta\) and the out-of-plane axis by \((1+\delta)^{-2}\), keeping the cell volume fixed while realizing the requested \(c/a\) ratio. This isolates the shape change from any volume relaxation effect on the energy. `c_a_list` (0.89 up to, but excluding, 1.5 by default) and the corresponding `energy_list` are returned together, ready to plot as the Bain energy curve.

## References

- Bain, E.C. (1924). The nature of martensite. *Transactions of the American Institute of Mining and Metallurgical Engineers*, 70, 25-47.
