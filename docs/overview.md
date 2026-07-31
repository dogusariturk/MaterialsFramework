# Overview

How `MaterialsFramework`'s four packages (`calculators`, `analysis`, `transformations`, and `tools`) fit together, and why the API is shaped the way it is.

## Entry-point-based registries

Calculators, analyzers, transformations, and tools are each registered as Python package entry-points. Every one of the four packages exposes the same pair of lookup functions: `list_x()` to enumerate what's available and `get_x(name, **kwargs)` to lazily import and instantiate it by name:

```python
from materialsframework.calculators import get_calculator, list_calculators

list_calculators()          # every registered calculator name
calc = get_calculator("chgnet")
```

Nothing MLIP-specific is imported until it's actually requested. This is what lets `import materialsframework` stay cheap even though 20+ MLIP backends are registered. Installing only the `chgnet` extra and calling `get_calculator("chgnet")` never touches MACE's, ORB's, or any other backend's dependencies.

## Calculators: one interface, many backends

Nearly every calculator subclasses `BaseCalculator`, an ABC wrapping an ASE `Calculator`, and most additionally subclass `BaseMDCalculator`, a sibling ABC. Together they provide three entry points shared by nearly every MLIP backend:

- `relax(structure)`: structure optimization (atoms and, optionally, cell)
- `calculate(structure)`: single-point energy/forces/stress evaluation, no optimization
- `run(structure, steps)`: molecular dynamics with NVE and multiple NVT/NPT thermostats and barostats

`RandomCalculator` and `VASPCalculator` are exceptions that implement only `BaseCalculator`, with no MD story: `RandomCalculator` is a dependency-free stub useful as a minimal reference implementation and for tests that shouldn't require a real MLIP installed, while `VASPCalculator` wraps a licensed VASP installation instead of an ML model as a ground-truth reference calculator, not one of the ML backends. `MEGNetCalculator` is a further outlier that implements neither ABC: it only exposes `calculate(structure)`, predicting a single scalar formation energy rather than the energy/forces/stress properties other calculators return, with no `relax()`/`run()` at all.

For calculators that implement the base interfaces, swapping one MLIP for another usually means changing only the calculator you instantiate. Available properties and backend-specific setup still vary; VASP, Random, and MEGNet have the exceptions described above.

```python
from materialsframework.calculators import CHGNetCalculator, MACECalculator

calc = CHGNetCalculator()   # or MACECalculator(), or any other backend
result = calc.relax(structure)
```

See [Base Classes theory](theory/calculators/base.md) for the relaxation/MD contract in detail, or [Geometry Optimization](usage/relaxation.md) / [Molecular Dynamics](usage/md.md) for usage.

## Analyzers and Transformations: a 1:1 pairing

Each property analyzer (e.g. `FormationEnergyAnalyzer`) is paired with a transformation module of the same name (e.g. `FormationEnergyTransformation`). The split is deliberate:

- The *transformation* generates the structures a calculation needs: candidate elemental reference structures, deformed/displaced structures for elastic constants or stacking faults, slab terminations for surface energies, and so on.
- The *analyzer* orchestrates calling a calculator on those structures and combines the results into the physical property.

```python
analyzer = SomeAnalyzer(calculator=calc)
result = analyzer.calculate(structure)
```

An analyzer takes an optional `calculator` and an optional `<name>_transformation` in its constructor, both lazily default-constructed if not supplied. See [Theory](theory/index.md) for the physics behind each analyzer, or [Usage](usage/analysis/index.md) for runnable examples.

## Tools

Standalone utilities that don't fit the calculator or analyzer/transformation pattern: special quasirandom structure (SQS) generation, cluster expansion, Cahn-Hilliard phase-field simulation, composition-temperature stability maps, and SQS-to-TDB conversion for CALPHAD workflows. See [Theory](theory/tools/index.md) and [Usage](usage/tools/index.md).
