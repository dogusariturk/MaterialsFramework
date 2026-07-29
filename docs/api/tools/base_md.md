# BaseMDCalculator

The `BaseMDCalculator` abstract base class provides the `run()` method for molecular dynamics simulations: NVE, six NVT thermostats (Nosé-Hoover, Langevin, Andersen, Bussi, Berendsen, and a Nosé-Hoover chain), and six NPT/barostat variants (Nosé-Hoover, isotropic MTK, MTK, masked MTK, Berendsen, and inhomogeneous Berendsen). It is a sibling ABC to `BaseCalculator`, not a subclass of it. Most MLIP calculators implement both via multiple inheritance.

See the [Base Classes theory](../../theory/calculators/base.md) for each thermostat/barostat's equations of motion, or the full API documentation on the [Base Classes](../calculators/base.md#materialsframework.tools.md.BaseMDCalculator) page.
