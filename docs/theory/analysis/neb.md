# Nudged Elastic Band

Interpolates a series of images between two endpoint structures and relaxes them into a minimum
energy path (MEP), reporting the forward and reverse reaction barriers.

## Overview

The nudged elastic band (NEB) method finds the minimum energy path connecting two known local minima
of the potential energy surface (for example, an atom hopping between two lattice sites, or a
vacancy migrating through a host). `NEBAnalyzer` relaxes both endpoints first (unless
`is_relaxed=True`), interpolates `n_images` intermediate images between them (via
`NEBTransformation`, using `pymatgen`'s image interpolation with optional lattice interpolation and
periodic-boundary-aware site matching), then optimizes the resulting band with ASE's `NEB`
implementation. The path is first converged without a climbing image; if `climb=True`, it is then
re-optimized with the climbing image enabled, following standard NEB practice of only turning it on
once the band is already close to converged.

## Theory

Each image \(i\) (excluding the fixed endpoints) is connected to its neighbors by fictitious springs
of constant \(k\) (`spring_constant`), and the true inter-atomic forces are decomposed relative to
the local path tangent \(\hat{\tau}_i\). The NEB force applied to image \(i\) is the spring force
along the tangent plus the true force perpendicular to it (the "nudging" that keeps images evenly
spaced without the springs corrupting the true energy landscape):

$$
\mathbf{F}_i = \big(\mathbf{F}_i^{\text{spring}} \cdot \hat{\tau}_i\big)\hat{\tau}_i
+ \Big[\mathbf{F}_i^{\text{true}} - \big(\mathbf{F}_i^{\text{true}} \cdot \hat{\tau}_i\big)\hat{\tau}_i\Big],
\qquad
\mathbf{F}_i^{\text{spring}} = k\big(|\mathbf{R}_{i+1} - \mathbf{R}_i| - |\mathbf{R}_i - \mathbf{R}_{i-1}|\big)\hat{\tau}_i
$$

`method` selects how ASE estimates \(\hat{\tau}_i\) and assembles these forces (`"aseneb"`,
`"improvedtangent"` (the default, generally more robust tangent estimate), `"eb"`, `"spline"`, or
`"string"`); see `ase.mep.NEB` for the differences between them. With `climb=True`, after the
band first converges the highest-energy image is switched to climbing-image forces,

$$
\mathbf{F}_i^{\text{climb}} = \mathbf{F}_i^{\text{true}} - 2\big(\mathbf{F}_i^{\text{true}} \cdot \hat{\tau}_i\big)\hat{\tau}_i,
$$

which inverts the force component along the tangent so that image drives itself to the exact saddle
point rather than settling for whichever discrete image happens to sit highest.

The forward barrier and reaction energy are extracted from a cubic-spline fit through the converged
images' energies and forces (`ase.mep.NEBTools.get_barrier`), not simply the maximum of the discrete
image energies:

$$
E_b = \max_{s} \tilde{E}(s) - E_0, \qquad \Delta E_r = E_N - E_0, \qquad E_b^{\text{rev}} = E_b - \Delta E_r
$$

where \(\tilde{E}(s)\) is the fitted energy along the path, \(E_0\) and \(E_N\) are the (already
relaxed) initial and final endpoint energies, \(E_b\) is the forward barrier, and \(E_b^{\text{rev}}\)
is the reverse barrier (the barrier seen going from the final structure back to the initial one).

## References

- Henkelman, G., Uberuaga, B.P., & Jónsson, H. (2000). A climbing image nudged elastic band method
  for finding saddle points and minimum energy paths. *Journal of Chemical Physics*, 113(22),
  9901-9904. <https://doi.org/10.1063/1.1329672>
- Henkelman, G., & Jónsson, H. (2000). Improved tangent estimate in the nudged elastic band method
  for finding minimum energy paths and saddle points. *Journal of Chemical Physics*, 113(22),
  9978-9985. <https://doi.org/10.1063/1.1323224>
