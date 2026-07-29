<div align="center" markdown>

# MaterialsFramework

[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](https://spdx.org/licenses/GPL-3.0-or-later.html)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20macos-lightgrey)

[![Tests](https://github.com/dogusariturk/MaterialsFramework/actions/workflows/tests.yml/badge.svg)](https://github.com/dogusariturk/MaterialsFramework/actions/workflows/tests.yml)
[![Lint](https://github.com/dogusariturk/MaterialsFramework/actions/workflows/lint.yml/badge.svg)](https://github.com/dogusariturk/MaterialsFramework/actions/workflows/lint.yml)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15731044.svg)](https://doi.org/10.5281/zenodo.15731044)

`MaterialsFramework` provides a single, uniform API for 20+ machine learning interatomic potentials (MLIPs), covering single-point calculations, structure relaxation, and molecular dynamics, plus the property analyzers and structure-generation tools that build on them. Swapping one MLIP for another, or for the licensed `VASPCalculator`, means changing one line of code.

<p>
  <a href="https://github.com/dogusariturk/MaterialsFramework/issues/new?labels=bug">Report a Bug</a> |
  <a href="https://github.com/dogusariturk/MaterialsFramework/issues/new?labels=enhancement">Request a Feature</a>
</p>

</div>

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } __Installation__

    ---

    Set up `uv` or `pip` and install the MLIP extras you need.

    [:octicons-arrow-right-24: Installation](installation.md)

-   :material-lightning-bolt:{ .lg .middle } __Quick Start__

    ---

    Relax a crystal structure in a few lines of code.

    [:octicons-arrow-right-24: Quick Start](quickstart.md)

-   :material-axis-arrow:{ .lg .middle } __Usage__

    ---

    Task-oriented walkthroughs for calculators, analyzers, and tools.

    [:octicons-arrow-right-24: Usage](usage/index.md)

-   :material-atom:{ .lg .middle } __Theory__

    ---

    The physics and derivations behind each analyzer and tool.

    [:octicons-arrow-right-24: Theory](theory/index.md)

</div>

---

## Key Features

- Run single-point calculations and structure relaxations across 20+ ML interatomic potentials through one shared `BaseCalculator` interface, or swap in the licensed `VASPCalculator` without changing calling code
- Accept `ase.Atoms`, `pymatgen.Structure`, and `pymatgen.Molecule` interchangeably as calculator input
- Run molecular dynamics (NVE, NVT/NPT Nose-Hoover, NPT/Inhomogeneous-NPT Berendsen) on any calculator that supports it
- Compute formation energy, elastic constants, phonons, stacking faults, surface/binding energies, and reaction barriers with 14 property analyzers, each paired with a transformation that generates the structures it needs
- Generate special quasirandom structures, cluster expansion models, phase-field simulations, and stability maps with the built-in tools
- Look up calculators, analyzers, transformations, and tools by name, without importing every MLIP backend at once

---

## Supported MLIPs
<div align="center" markdown>
| MLIP      | Extra       | Package            | API                                 | Repository                                                         | Paper                                                       |
|-----------|-------------|--------------------|-------------------------------------|--------------------------------------------------------------------|-------------------------------------------------------------|
| ALIGNN    | `alignn`    | `alignn`           | [API](api/calculators/alignn.md)    | [Repo](https://github.com/usnistgov/alignn)                        | [Paper](https://arxiv.org/abs/2106.01829)                   |
| Allegro   | `allegro`   | `nequip-allegro`   | [API](api/calculators/allegro.md)   | [Repo](https://github.com/mir-group/allegro)                       | [Paper](https://doi.org/10.1038/s41467-023-36329-y)         |
| AlphaNet  | `alphanet`  | `msc-alphanet`     | [API](api/calculators/alphanet.md)  | [Repo](https://github.com/zmyybc/AlphaNet)                         | [Paper](https://arxiv.org/abs/2501.07155)                   |
| CHGNet    | `chgnet`    | `chgnet`           | [API](api/calculators/chgnet.md)    | [Repo](https://github.com/CederGroupHub/chgnet)                    | [Paper](https://arxiv.org/abs/2302.14231)                   |
| DeePMD    | `deepmd`    | `deepmd-kit`       | [API](api/calculators/deepmd.md)    | [Repo](https://github.com/deepmodeling/deepmd-kit)                 | [Paper](https://doi.org/10.1016/j.cpc.2018.03.016)          |
| EqNorm    | `eqnorm`    | `eqnorm`           | [API](api/calculators/eqnorm.md)    | [Repo](https://github.com/yzchen08/eqnorm)                         | N/A                                                         |
| EquFlash  | N/A         | `GGNN` (git-only)  | [API](api/calculators/equflash.md)  | [Repo](https://github.com/SamsungDS/GGNN)                          | N/A                                                         |
| EqV2      | `eqv2`      | `fairchem-core`    | [API](api/calculators/eqv2.md)      | [Repo](https://github.com/facebookresearch/fairchem)                      | [Paper](https://arxiv.org/abs/2306.12059)                   |
| eSEN      | `esen`      | `fairchem-core`    | [API](api/calculators/esen.md)      | [Repo](https://github.com/facebookresearch/fairchem)                      | [Paper](https://arxiv.org/abs/2502.12147)                   |
| GPTFF     | N/A         | `gptff` (git-only) | [API](api/calculators/gptff.md)     | [Repo](https://github.com/atomly-materials-research-lab/GPTFF)     | [Paper](https://doi.org/10.1016/j.scib.2024.08.039)         |
| GRACE     | `grace`     | `tensorpotential`  | [API](api/calculators/grace.md)     | [Repo](https://github.com/ICAMS/grace-tensorpotential)             | [Paper](https://arxiv.org/abs/2508.17936)                   |
| HIENet    | `hienet`    | `hienet`           | [API](api/calculators/hienet.md)    | [Repo](https://github.com/divelab/AIRS/tree/main/OpenMat/HIENet)   | [Paper](https://arxiv.org/abs/2503.05771)                   |
| M3GNet    | `matgl`     | `matgl`            | [API](api/calculators/m3gnet.md)    | [Repo](https://github.com/materialsvirtuallab/m3gnet)              | [Paper](https://arxiv.org/abs/2202.02450)                   |
| MACE      | `mace`      | `mace-torch`       | [API](api/calculators/mace.md)      | [Repo](https://github.com/ACEsuit/mace)                            | [Paper](https://arxiv.org/abs/2401.00096)                   |
| MatRIS    | `matris`    | `matris`           | [API](api/calculators/matris.md)    | [Repo](https://github.com/HPC-AI-Team/MatRIS)                      | [Paper](https://arxiv.org/abs/2603.02002)                   |
| MatterSim | `mattersim` | `mattersim`        | [API](api/calculators/mattersim.md) | [Repo](https://github.com/microsoft/mattersim)                     | [Paper](https://arxiv.org/abs/2405.04967)                   |
| MEGNet    | `matgl`     | `matgl`            | [API](api/calculators/megnet.md)    | [Repo](https://github.com/materialsvirtuallab/megnet)              | [Paper](https://arxiv.org/abs/1812.05055)                   |
| NequIP    | `nequip`    | `nequip`           | [API](api/calculators/nequip.md)    | [Repo](https://github.com/mir-group/nequip)                        | [Paper](https://arxiv.org/abs/2101.03164)                   |
| Nequix    | `nequix`    | `nequix`           | [API](api/calculators/nequix.md)    | [Repo](https://github.com/atomicarchitects/nequix)                 | [Paper](https://arxiv.org/abs/2508.16067)                   |
| NewtonNet | `newtonnet` | `newtonnet`        | [API](api/calculators/newtonnet.md) | [Repo](https://github.com/THGLab/NewtonNet)                        | [Paper](https://doi.org/10.1039/D2DD00008C)                 |
| ORB       | `orb`       | `orb-models`       | [API](api/calculators/orb.md)       | [Repo](https://github.com/orbital-materials/orb-models)            | [Paper](https://arxiv.org/abs/2504.06231)                   |
| PetMad    | `petmad`    | `upet`             | [API](api/calculators/petmad.md)    | [Repo](https://github.com/lab-cosmo/upet)                          | [Paper](https://www.nature.com/articles/s41467-025-65662-7) |
| PosEGNN   | N/A         | N/A                | [API](api/calculators/posegnn.md)   | [Repo](https://github.com/IBM/materials/tree/main/models/pos_egnn) | N/A                                                         |
| SevenNet  | `sevennet`  | `sevenn`           | [API](api/calculators/sevennet.md)  | [Repo](https://github.com/MDIL-SNU/SevenNet)                       | [Paper](https://arxiv.org/abs/2510.11241)                   |
| TACE      | `tace`      | `TACE`             | [API](api/calculators/tace.md)      | [Repo](https://github.com/xvzemin/tace)                            | [Paper](https://arxiv.org/abs/2509.14961)                   |
| UMA       | `uma`       | `fairchem-core`    | [API](api/calculators/uma.md)       | [Repo](https://github.com/facebookresearch/fairchem)                      | [Paper](https://arxiv.org/abs/2506.23971)                   |
</div>

Non-MLIP calculators: `RandomCalculator` (dependency-free testing stub) and `VASPCalculator` (external licensed VASP backend).

??? info "EquFlash"

    *EquFlash* is only installable from its upstream git repository:

    ```bash
    uv pip install "GGNN @ git+https://github.com/SamsungDS/GGNN.git"
    ```

    That alone won't work: `fairchem-core` is hard-imported but never installed, so `EquFlashCalculator().calculator` raises `ModuleNotFoundError`. See [installation](installation.md) for the CPU-only dependency set that's confirmed to work, and [MLIP Conflicts](mlip-conflicts.md) for details.

??? info "GPTFF"

    *GPTFF* is only installable from its upstream git repository:

    ```bash
    uv pip install "gptff @ git+https://github.com/atomly-materials-research-lab/GPTFF.git"
    ```

??? info "PosEGNN"

    *PosEGNN* has no installable package on any public index. Clone the repository and add the module directory to `PYTHONPATH` manually:

    ```bash
    git clone --depth 1 https://github.com/IBM/materials.git
    export PYTHONPATH="$PWD/materials/models/pos_egnn:$PYTHONPATH"
    ```

---

## Property Analyzers

| Analyzer                        | Description                                                                  |
|---------------------------------|------------------------------------------------------------------------------|
| `ANNNIStackingFaultAnalyzer`    | ANNNI-based intrinsic and extrinsic stacking fault energies                  |
| `BainPathAnalyzer`              | Energy along the FCC-to-BCC Bain transformation path                         |
| `CTEAnalyzer`                   | Coefficient of thermal expansion from NPT-MD volume trends                   |
| `CubicElasticConstantsAnalyzer` | Cubic elastic constants and derived moduli (B, G, E, ν)                      |
| `ElasticConstantsAnalyzer`      | Full elastic tensor and Voigt-Reuss-Hill averages                            |
| `EOSAnalyzer`                   | Equation-of-state curve fitting from E-V data                                |
| `FormationEnergyAnalyzer`       | Formation energy per atom                                                    |
| `HSolubilityAnalyzer`           | Hydrogen insertion and solution energies                                     |
| `NEBAnalyzer`                   | Nudged elastic band minimum energy path and reaction barrier                 |
| `PhonopyAnalyzer`               | Phonon DOS, band structure, and thermal properties                           |
| `Phono3pyAnalyzer`              | Anharmonic force constants and lattice thermal conductivity                  |
| `SBEAnalyzer`                   | Surface binding energies, a first-principles proxy for sputtering resistance |
| `SurfaceAnalyzer`               | Slab surface energies for a given Miller index                               |
| `USFEAnalyzer`                  | Generalized stacking fault energy curves and unstable SFE                    |

---

## Tools

| Tool                   | Description                                                                             |
|------------------------|-----------------------------------------------------------------------------------------|
| `BondLatticeParameter` | Lattice parameter estimation from bond lengths for FCC/BCC/HCP alloys                   |
| `ClusterExpansion`     | Cluster expansion model construction and fitting                                        |
| `CoherentStabilityMap` | Stability map generation with a coherent-elastic correction to the Gibbs energy Hessian |
| `PhaseFieldModel`      | Cahn-Hilliard phase-field simulations                                                   |
| `Sqs2tdb`              | Converts SQS output files to TDB format for CALPHAD workflows (PhaseForge)              |
| `SqsGenerator`         | Special quasirandom structure generation                                                |
| `StabilityMap`         | Composition-temperature stability map generation                                        |
| `TrajectoryObserver`   | Records energies, forces, stresses, and trajectory frames during relaxation or MD       |

---

## Citing

If you use `MaterialsFramework` in your research, please cite:

> Sarıtürk, D. (2025). *MaterialsFramework*. Zenodo. <https://doi.org/10.5281/zenodo.15731044>

???+ quote "BibTeX"
    ```bibtex
    @software{sariturk_2025_15731044,
      author    = {Sarıtürk, Doğuhan and Arroyave, Raymundo},
      title     = {MaterialsFramework},
      month     = jun,
      year      = 2025,
      publisher = {Zenodo},
      doi       = {10.5281/zenodo.15731044},
      url       = {https://doi.org/10.5281/zenodo.15731044},
    }
    ```
