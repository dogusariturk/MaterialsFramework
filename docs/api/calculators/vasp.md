# VASP

!!! info "Optional dependency"

    `VASPCalculator` has no `materialsframework` extra. It wraps a separately installed, licensed VASP binary, configured via the `command` argument or the `ASE_VASP_COMMAND`/`VASP_COMMAND`/`VASP_SCRIPT` env vars, plus `VASP_PP_PATH` for pseudopotentials. See [Non-Extra Calculators](../../installation.md#non-extra-calculators) or ASE's [VASP documentation](https://docs.ase-lib.org/ase/calculators/vasp.html) for details.

::: materialsframework.calculators.vasp.VASPCalculator
