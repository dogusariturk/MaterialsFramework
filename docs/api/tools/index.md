# Tools

Standalone utilities and abstract base classes in the `tools` package.

## Abstract Base Classes

| Class              | Module       | Description                                                                                     |
|--------------------|--------------|-------------------------------------------------------------------------------------------------|
| `BaseCalculator`   | `calculator` | Abstract base for geometry optimization (`relax()`) and single-point evaluation (`calculate()`) |
| `BaseMDCalculator` | `md`         | Abstract base for molecular dynamics (`run()`)                                                  |

## Utilities

| Class                  | Module                   | Description                                                             |
|------------------------|--------------------------|-------------------------------------------------------------------------|
| `TrajectoryObserver`   | `trajectory`             | Records atomic positions, energies, and forces during relaxation or MD  |
| `StabilityMap`         | `stability_map`          | Phase stability analysis and visualization                              |
| `CoherentStabilityMap` | `stability_map`          | `StabilityMap` with a coherent-elastic Gibbs energy correction          |
| `PhaseFieldModel`      | `cahn_hilliard`          | Cahn-Hilliard phase field simulation                                    |
| `SimulationGrid`       | `cahn_hilliard`          | Grid definition for phase field simulations                             |
| `MaterialParameters`   | `cahn_hilliard`          | Material parameter container for phase field                            |
| `Sqs2tdb`              | `sqs2tdb`                | Convert SQS structures to TDB thermodynamic database files (PhaseForge) |
| `SqsGenerator`         | `sqsgen`                 | Generate Special Quasirandom Structures (SQS)                           |
| `ClusterExpansion`     | `cluster_expansion`      | Cluster expansion construction and fitting                              |
| `BondLatticeParameter` | `bond_lattice_parameter` | Bond-length-based lattice parameter estimation                          |
