# PhaseForge

!!! info "Optional dependency"

    `Sqs2tdb` requires the `calphad` extra, plus the `sqs2tdb` binary (part of [ATAT](https://axelvandewalle.github.io/www-avdw/atat/)) on `PATH`, with its SQS database configured via `~/.atat.rc`.

    === "uv"

        ```bash
        uv add "materialsframework[calphad]"
        ```

    === "pip"

        ```bash
        pip install "materialsframework[calphad]"
        ```

Fits a CALPHAD thermodynamic database (`.tdb`) solution model directly from SQS energies computed
with a MaterialsFramework calculator, bridging MLIP energetics into standard CALPHAD-format
phase-stability data.

## Overview

CALPHAD phase diagrams and stability maps (see [StabilityMap](stability_map.md)) are built from
Gibbs-energy solution models whose parameters are traditionally fit to a mix of experimental data and
DFT formation energies. `Sqs2tdb` automates a materials-informatics shortcut: it pulls Special
Quasirandom Structures from ATAT's built-in SQS database for each requested lattice and composition
(every lattice in that database if `lattices` is left as `None`), evaluates their formation energies
with a `BaseCalculator` (or, for the `"LIQUID"` lattice, by equilibrating with NPT-then-NVT molecular
dynamics via a `BaseMDCalculator`, on a supercell scaled by `md_scaling_matrix`), and hands those
energies to ATAT's `sqs2tdb` script to regress a substitutional solution model, written out as a
standard `.tdb` file readable by `pycalphad` or Thermo-Calc. Structures are converted from ATAT's own
format straight to POSCAR with `pymatgen.io.atat.Mcsqs`, so no VASP installation is needed even though
the underlying script is ATAT's VASP-oriented `sqs2tdb`.

## Theory

For each lattice being fit (e.g. `"BCC_A2"`, `"FCC_A1"`, `"LIQUID"`), the molar Gibbs energy of the
substitutional solution phase follows the standard CALPHAD Redlich-Kister-Muggianu form:

$$
G_m = \sum_i x_i\,{}^{0}G_i + RT \sum_i x_i \ln x_i + \sum_{i<j} x_i x_j \sum_{v} {}^{v}L_{ij}\,(x_i - x_j)^v
$$

where \(x_i\) is the mole fraction of component \(i\), \({}^{0}G_i\) is the end-member reference
lattice stability, the middle term is the ideal configurational entropy of mixing, and \({}^{v}L_{ij}\)
are Redlich-Kister interaction parameters of order \(v\) between components \(i\) and \(j\). `fit()`'s
`terms` argument controls which of these are regressed for each lattice, via `(order, level)` pairs:
`order=1` fits an end-member-like linear-in-composition term, `order=2` fits a binary interaction
\({}^{v}L_{ij}\) at Redlich-Kister degree `level`, and `order=3` extends this to ternary interactions.
`terms` can be a single string or list of pairs applied to every lattice, or a dict keyed by lattice
name for per-lattice control, with any lattice left out of the dict falling back to the built-in
default. The regression uses the SQS formation energies, relaxed at 0 K or time-averaged over the
last 20% of an MD trajectory for `"LIQUID"`, as the enthalpic data fit against these
composition-dependent terms over the temperature window `[t_min, t_max]`; `sro=True` additionally
accounts for short-range order beyond the ideal-mixing entropy term.

## References

- Zhu, S., Sarıtürk, D., & Arróyave, R. (2025). Machine learning potentials for alloys: A detailed
  workflow to predict phase diagrams and benchmark accuracy. *npj Computational Materials*, 11, 340.
  <https://doi.org/10.1038/s41524-025-01814-z>
- Zhu, S., Sarıtürk, D., & Arróyave, R. (2025). Accelerating CALPHAD-based phase diagram predictions
  in complex alloys using universal machine learning potentials: Opportunities and challenges.
  *Acta Materialia*, 286, 120747. <https://doi.org/10.1016/j.actamat.2025.120747>
- Sarıtürk, D., Zhu, S., & Arróyave, R. (2025). PhaseForge (v1.0.0) [Software]. Zenodo.
  <https://doi.org/10.5281/zenodo.15730911>
- van de Walle, A., Sun, R., Hong, Q.-J., & Kadkhodaei, S. (2017). Software tools for high-throughput
  CALPHAD from first-principles data. *Calphad*, 58, 70-81. <https://doi.org/10.1016/j.calphad.2017.05.005>
