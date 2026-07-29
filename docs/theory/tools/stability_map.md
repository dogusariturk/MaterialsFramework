# StabilityMap

!!! info "Optional dependency"

    `StabilityMap` requires `calphad`. `CoherentStabilityMap` additionally requires `sqsgen`. Plotting/visualization support additionally requires `plots`.

    === "uv"

        ```bash
        # StabilityMap
        uv add "materialsframework[calphad]"

        # CoherentStabilityMap
        uv add "materialsframework[calphad,sqsgen]"

        # For plotting/visualization support
        uv add "materialsframework[calphad,plots]"
        ```

    === "pip"

        ```bash
        # StabilityMap
        pip install "materialsframework[calphad]"

        # CoherentStabilityMap
        pip install "materialsframework[calphad,sqsgen]"

        # For plotting/visualization support
        pip install "materialsframework[calphad,plots]"
        ```

Maps thermodynamic stability and phase-separation (spinodal) regions across a composition space,
using a CALPHAD database as the source of the Gibbs energy.

## Overview

`StabilityMap` evaluates a CALPHAD `Database` across a grid of compositions for one phase at fixed
temperature and pressure, and classifies each composition by the local curvature of the Gibbs
energy: convex points are single-phase stable, points where the curvature turns non-convex along
some composition direction lie inside the spinodal (unstable to infinitesimal composition
fluctuations, i.e. prone to spontaneous phase separation). `fit()` parallelizes this per-composition
evaluation across processes; `plot()` visualizes the result, but only for exactly 4 elements
(projected onto a 2D pentagon-like layout).

`CoherentStabilityMap` extends `StabilityMap` with an elastic penalty term that accounts for
coherency strain: a composition fluctuation embedded in a coherent (defect-free interface) matrix
costs additional elastic energy that a stress-free fluctuation would not, which shrinks the unstable
region relative to the purely chemical spinodal. This requires generating and relaxing an SQS
structure and computing its elastic constants (see [Elastic Constants](../analysis/elastic.md)) at
every composition point, so `fit()` here runs sequentially rather than process-pooled: MLIP
calculators typically hold GPU/PyTorch state that doesn't pickle across processes cleanly.

## Theory

### Chemical (incoherent) stability

For a phase with independent composition variables \(x_1, \dots, x_{n-1}\) (one component
eliminated via Gibbs-Duhem), stability at fixed \(T\), \(P\) is governed by the curvature of
\(G(x_1, \dots, x_{n-1})\). Since \(\mu_i - \mu_n = \partial G/\partial x_i\) for the dependent
component \(n\), the raw composition Hessian is

$$
H_{ij} = \frac{\partial(\mu_i - \mu_n)}{\partial x_j} = \frac{\partial^2 G}{\partial x_i\,\partial x_j}
$$

which `StabilityMap` builds from `pycalphad`'s \(\partial \mu_i/\partial x_j\) at each grid point.
Because mole fractions are not an orthonormal coordinate system once \(n > 2\), the eigenvalues of
\(H\) itself are not gauge-invariant. `ORTHOGONALIZATION` is a fixed change-of-basis matrix
(supporting up to 10 components) that maps composition differences onto an orthonormal basis of the
Gibbs simplex; with \(Q\) the relevant leading block of that matrix,

$$
\tilde{H} = Q^{T} H Q
$$

has eigenvalues that are directly stability-diagnostic: the composition is locally stable if every
eigenvalue of \(\tilde H\) is positive (convex \(G\)), and lies inside the spinodal (unstable along
at least one composition direction) if any eigenvalue is negative. `fit()` records, for each grid
composition, how many eigenvalues of \(\tilde H\) are negative (`negative_eigenvalues`); `plot()`
colors points by that count.

### Coherent correction

An elastically coherent solid solution resists composition fluctuations more strongly than an
unconstrained one, because the fluctuation must strain against its coherent surroundings. Following
Cahn's coherent-spinodal treatment, `CoherentStabilityMap` adds a positive semi-definite elastic term
to the chemical Hessian before orthogonalizing and diagonalizing it:

$$
H^{\text{coh}} = H + 2\,Y\,V_m\,\big(\mathbf{\eta}^{T}\mathbf{\eta}\big)
$$

\(\eta_i = \dfrac{1}{a}\dfrac{\partial a}{\partial x_i}\) is the fractional composition-dependence of
the lattice parameter along independent direction \(i\), estimated by relaxing an SQS structure at
the base composition and at a small perturbation of it (forward finite difference). \(V_m\) is the
molar volume of the corresponding conventional cell. \(Y\) is the biaxial modulus, a combination of
\(C_{11}\), \(C_{12}\), \(C_{44}\) that reduces to the isotropic case when the elastic-anisotropy
factor \(2C_{44} - C_{11} + C_{12}\) vanishes:

$$
Y = \begin{cases}
\dfrac{(C_{11}+2C_{12})(C_{11}-C_{12})}{C_{11}}, & 2C_{44} - C_{11} + C_{12} \ge 0 \\[8pt]
\dfrac{6(C_{11}+2C_{12})C_{44}}{4C_{44}+C_{11}+2C_{12}}, & \text{otherwise}
\end{cases}
$$

Since \(H^{\text{coh}} - H\) is positive semi-definite, the coherent spinodal is always nested inside
(no larger than) the chemical spinodal. `fit()` reports both `negative_eigenvalues_chem` and
`negative_eigenvalues_coherent` per composition.

## References

- Kunselman, C., Sarıtürk, D., Zhu, S., Attari, V., & Arroyave, R. (2026). From MLIPs to
  microstructure: A high-throughput computational framework to design spinodal alloys in
  high-dimensional composition spaces via analytic derivatives of CALPHAD model predictions.
  *arXiv preprint*. <https://doi.org/10.48550/arXiv.2607.20077>
- Kadirvel, K., Koneru, S.R., & Wang, Y. (2022). Exploration of spinodal decomposition in
  multi-principal element alloys (MPEAs) using CALPHAD modeling. *Scripta Materialia*, 214, 114657.
  <https://doi.org/10.1016/j.scriptamat.2022.114657>
- Cahn, J.W. (1961). On spinodal decomposition. *Acta Metallurgica*, 9(9), 795-801.
  <https://doi.org/10.1016/0001-6160(61)90182-1>
- Zener, C. (1948). *Elasticity and Anelasticity of Metals*. University of Chicago Press.
