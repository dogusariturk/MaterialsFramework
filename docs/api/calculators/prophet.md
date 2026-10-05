# Prophet

!!! info "Optional dependency"

    `ProphetCalculator` has no `materialsframework` extra. Its upstream `prophet-mlip` package isn't published on PyPI, so install it from the upstream git repository. Then download a checkpoint from the [Prophet model repository](https://huggingface.co/kairosmaterial/prophet) on Hugging Face and pass its path as `model`. See [Non-Extra Calculators](../../installation.md#non-extra-calculators) for details.

    === "uv"

        ```bash
        uv pip install "prophet-mlip @ git+https://github.com/kairosmaterial/prophet.git"
        ```

    === "pip"

        ```bash
        pip install "prophet-mlip @ git+https://github.com/kairosmaterial/prophet.git"
        ```

!!! note "Checkpoints"

    `ProphetCalculator` reads each checkpoint's JSON header to select the matching upstream calculator, so there's no flag to set for the checkpoint type. Checkpoints with spin conditioning also take local magnetic moments as input. They're read from the `magmom` site property of a pymatgen `Structure` or from `Atoms.set_initial_magnetic_moments()`. Unset moments are passed as zeros, which means a non-magnetic configuration. Different checkpoints aren't guaranteed to share an energy reference, so don't compare energies across them.

!!! note "Magnetic-moment prediction"

    To predict local magnetic moments instead of supplying them, pass a magnetic-moment prediction bundle as `magmom_head` together with a spin-conditioned `model`. The bundle isn't a model checkpoint, so don't pass it as `model` itself. The structure's initial magnetic moments then act as seeds that encode the magnetic ordering. They must be collinear and not all zero. Each atom's moment is predicted first, and energy, forces, and stress are evaluated at the predicted moments. `calculate()` and `relax()` also return the predicted moments under `"magmoms"`, and relaxations record them in the trajectory by default (pass `include_magmoms=False` to turn that off).

    ```python
    from materialsframework.calculators import ProphetCalculator

    calc = ProphetCalculator(model="<spin-conditioned checkpoint>.pt", magmom_head="<prediction bundle>.pt")
    structure.add_site_property("magmom", [2.0, -2.0])  # seeds: antiferromagnetic ordering
    result = calc.calculate(structure)
    result["magmoms"]  # predicted moments
    ```

::: materialsframework.calculators.prophet.ProphetCalculator
