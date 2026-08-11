# Elastic Constants

Computes the full second-order elastic constant tensor of a structure from stress-strain data, along with the isotropic mechanical properties derived from it.

```python
from ase.build import bulk
from materialsframework.analysis import ElasticConstantsAnalyzer
from materialsframework.calculators import GraceCalculator

struct = bulk("Cu", "fcc", a=3.6, cubic=True)
calc = GraceCalculator()

ec = ElasticConstantsAnalyzer(calculator=calc)
res = ec.calculate(struct)

print(res["C_11"], res["C_12"], res["C_44"])  # GPa, keys depend on detected symmetry
print(res["voigt_reuss_hill_bulk_modulus"])
print(res["poisson_ratio"])

# Derived thermodynamic/acoustic properties
print(res["debye_temperature"])  # K
print(res["gruneisen_approx"])  # dimensionless
print(res["v_longitudinal"], res["v_transverse"], res["v_mean"])  # m/s
```

!!! tip "Relax ions within distorted cells"

    By default, `calculate()` measures each distorted structure without ionic relaxation. Passing `relax_ions=True` first relaxes the atoms within that same, fixed cell shape, producing relaxed-ion elastic constants (see [Theory](../../theory/analysis/elastic.md#letting-the-atoms-relax-during-deformation)). This adds a potentially multi-step optimization for every deformation, so runtime can increase substantially. For single-atom primitive cells, the two approaches coincide.

Enable ion relaxation with the calculator's existing `fmax`, or override it for distorted structures with `fmax_distort`:

```python
res = ec.calculate(struct, relax_ions=True)
res = ec.calculate(struct, relax_ions=True, fmax_distort=0.2)
```

See [Theory](../../theory/analysis/elastic.md) for the derivation, or the [API Reference](../../api/analysis/elastic.md) for the full parameter list.
