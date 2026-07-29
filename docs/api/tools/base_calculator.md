# BaseCalculator

The `BaseCalculator` abstract base class is the foundation for all ML calculator implementations. It provides `relax()` for geometry optimization (atomic positions and, optionally, the cell) and `calculate()` for single-point energy/force/stress evaluation, with no MLIP-specific code. Each subclass only supplies an ASE `Calculator` and its `AVAILABLE_PROPERTIES`.

See the [Base Classes theory](../../theory/calculators/base.md) for the convergence criteria, cell relaxation via `FrechetCellFilter`, and symmetry constraints behind `relax()`/`calculate()`, or the full API documentation on the [Base Classes](../calculators/base.md#materialsframework.tools.calculator.BaseCalculator) page.
