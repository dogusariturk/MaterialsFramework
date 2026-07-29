# Tools

The physics and derivations behind `MaterialsFramework`'s standalone tools. See [Usage](../../usage/tools/index.md) for runnable examples, or [API Reference](../../api/tools/index.md) for the full parameter list.

| Topic                                                 | Description                                                           |
|-------------------------------------------------------|-----------------------------------------------------------------------|
| [Bond Lattice Parameter](bond_lattice_parameter.md)   | Lattice parameter estimation from bond lengths for FCC/BCC/HCP alloys |
| [Phase Field Model (Cahn-Hilliard)](cahn_hilliard.md) | Cahn-Hilliard phase-field simulations                                 |
| [Cluster Expansion](cluster_expansion.md)             | Cluster expansion model construction and fitting                      |
| [PhaseForge](sqs2tdb.md)                              | Converts SQS output to TDB format for CALPHAD workflows               |
| [Special Quasirandom Structures](sqsgen.md)           | SQS generation                                                        |
| [StabilityMap](stability_map.md)                      | Composition-temperature stability map generation                      |

`TrajectoryObserver` has no dedicated theory page. It only records what the calculator already computed at each step; see its [Usage](../../usage/tools/trajectory.md) page.
