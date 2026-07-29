# Special Quasirandom Structures

!!! info "Optional dependency"

    `SqsGenerator` requires the `sqsgen` extra.

    === "uv"

        ```bash
        uv add "materialsframework[sqsgen]"
        ```

    === "pip"

        ```bash
        pip install "materialsframework[sqsgen]"
        ```

Generates a finite supercell whose short-range order mimics that of a truly random substitutional alloy, so disordered-alloy properties can be modeled with a small, deterministic cell instead of a much larger explicitly-disordered one.

```python
from materialsframework.tools.sqsgen import SqsGenerator

sqs = SqsGenerator(iterations=5000, mode="random")
result = sqs.generate(composition="Fe0.5Ni0.5", crystal_structure="BCC", supercell_size=(4, 4, 4))

structure = result["structure"]   # pymatgen Structure
print(result["objective"])        # final SQS objective value (0 = exact match to random target)
```

See [Theory](../../theory/tools/sqsgen.md) for the derivation, or the [API Reference](../../api/tools/sqsgen.md) for the full parameter list.
