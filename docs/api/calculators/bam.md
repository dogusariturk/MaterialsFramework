# BAM

!!! info "Optional dependency"

    `BAMCalculator` has no `materialsframework` extra. Install it from the upstream git repository, then pass the path of a trained BAM model as `model`, such as [BAM-MP-core](https://matbench-discovery.materialsproject.org/models/bam-mp-core). See [Non-Extra Calculators](../../installation.md#non-extra-calculators) for details.

    === "uv"

        ```bash
        uv pip install "BAM-torch @ git+https://github.com/myung-group/BAM-torch.git"
        ```

    === "pip"

        ```bash
        pip install "BAM-torch @ git+https://github.com/myung-group/BAM-torch.git"
        ```

::: materialsframework.calculators.bam.BAMCalculator
