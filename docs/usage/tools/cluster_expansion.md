# Cluster Expansion

!!! info "Optional dependency"

    `ClusterExpansion` requires the `ce` extra.

    === "uv"

        ```bash
        uv add "materialsframework[ce]"
        ```

    === "pip"

        ```bash
        pip install "materialsframework[ce]"
        ```

Fits a cluster expansion: a linear model that predicts a configuration-dependent property (e.g. mixing energy) for any atomic decoration of a fixed lattice, trained on a set of structures evaluated with a `BaseCalculator`.

```python
from ase.build import bulk
from materialsframework.calculators import RandomCalculator
from materialsframework.tools import ClusterExpansion

primitive = bulk("Cu", "fcc", a=3.6)
training_structures = [...]  # list of pymatgen Structures decorating the same lattice

ce = ClusterExpansion(fit_method="ardr", calculator=RandomCalculator())
ce.fit(
    structures=training_structures,
    primitive_structure=primitive,
    cutoffs=[8.0, 6.0],       # pair, triplet cutoff radii (Å)
    chemical_symbols=["Cu", "Ni"],
    properties=["energy"],
    fit_property="energy",
)

print(ce.cluster_expansion)                       # fitted icet.ClusterExpansion
new_config = primitive.repeat((2, 2, 2))          # any ase.Atoms sharing the same cluster space
prediction = ce.cluster_expansion.predict(new_config)
```

See [Theory](../../theory/tools/cluster_expansion.md) for the derivation, or the [API Reference](../../api/tools/cluster_expansion.md) for the full parameter list.
