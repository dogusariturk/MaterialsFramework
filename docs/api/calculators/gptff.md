# GPTFF

!!! info "Optional dependency"

    `GPTFFCalculator` has no `materialsframework` extra. The pinned upstream revision requires `ase<3.29`, but MaterialsFramework requires `ase>=3.29`; a normal install would downgrade ASE and break MaterialsFramework's MD imports. The verified sequence bypasses GPTFF's dependency metadata and installs its non-core runtime dependencies explicitly. See [Non-Extra Calculators](../../installation.md#non-extra-calculators) for details.

    === "uv"

        ```bash
        uv pip install --no-deps "gptff @ git+https://github.com/atomly-materials-research-lab/GPTFF.git@8a03afa5c9a09411bc4e769a0efc947ee52e32c9"
        uv pip install "torch>=2.0" scikit-learn psutil tqdm
        ```

    === "pip"

        ```bash
        pip install --no-deps "gptff @ git+https://github.com/atomly-materials-research-lab/GPTFF.git@8a03afa5c9a09411bc4e769a0efc947ee52e32c9"
        pip install "torch>=2.0" scikit-learn psutil tqdm
        ```

::: materialsframework.calculators.gptff.GPTFFCalculator
