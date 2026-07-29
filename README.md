<div align="center">

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
  <a href="https://github.com/dogusariturk/MaterialsFramework/issues/new?labels=enhancement">Request a Feature</a> |
  <a href="https://dogusariturk.github.io/MaterialsFramework">Documentation</a>
</p>

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

| MLIP      | Extra       | Package            | API                                      | Repository                                                         | Paper                                                       |
|-----------|-------------|--------------------|------------------------------------------|--------------------------------------------------------------------|-------------------------------------------------------------|
| ALIGNN    | `alignn`    | `alignn`           | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/alignn/)    | [Repo](https://github.com/usnistgov/alignn)                        | [Paper](https://arxiv.org/abs/2106.01829)                   |
| Allegro   | `allegro`   | `nequip-allegro`   | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/allegro/)   | [Repo](https://github.com/mir-group/allegro)                       | [Paper](https://doi.org/10.1038/s41467-023-36329-y)         |
| AlphaNet  | `alphanet`  | `msc-alphanet`     | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/alphanet/)  | [Repo](https://github.com/zmyybc/AlphaNet)                         | [Paper](https://arxiv.org/abs/2501.07155)                   |
| CHGNet    | `chgnet`    | `chgnet`           | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/chgnet/)    | [Repo](https://github.com/CederGroupHub/chgnet)                    | [Paper](https://arxiv.org/abs/2302.14231)                   |
| DeePMD    | `deepmd`    | `deepmd-kit`       | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/deepmd/)    | [Repo](https://github.com/deepmodeling/deepmd-kit)                 | [Paper](https://arxiv.org/abs/2506.01686)                   |
| EqNorm    | `eqnorm`    | `eqnorm`           | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/eqnorm/)    | [Repo](https://github.com/yzchen08/eqnorm)                         | N/A                                                         |
| EquFlash  | N/A         | `GGNN` (git-only)  | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/equflash/)  | [Repo](https://github.com/SamsungDS/GGNN)                          | N/A                                                         |
| EqV2      | `eqv2`      | `fairchem-core`    | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/eqv2/)      | [Repo](https://github.com/facebookresearch/fairchem)                      | [Paper](https://arxiv.org/abs/2306.12059)                   |
| eSEN      | `esen`      | `fairchem-core`    | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/esen/)      | [Repo](https://github.com/facebookresearch/fairchem)                      | [Paper](https://arxiv.org/abs/2502.12147)                   |
| GPTFF     | N/A         | `gptff` (git-only) | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/gptff/)     | [Repo](https://github.com/atomly-materials-research-lab/GPTFF)     | [Paper](https://doi.org/10.1016/j.scib.2024.08.039)         |
| GRACE     | `grace`     | `tensorpotential`  | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/grace/)     | [Repo](https://github.com/ICAMS/grace-tensorpotential)             | [Paper](https://arxiv.org/abs/2508.17936)                   |
| HIENet    | `hienet`    | `hienet`           | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/hienet/)    | [Repo](https://github.com/divelab/AIRS/tree/main/OpenMat/HIENet)   | [Paper](https://arxiv.org/abs/2503.05771)                   |
| M3GNet    | `matgl`     | `matgl`            | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/m3gnet/)    | [Repo](https://github.com/materialsvirtuallab/m3gnet)              | [Paper](https://arxiv.org/abs/2202.02450)                   |
| MACE      | `mace`      | `mace-torch`       | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/mace/)      | [Repo](https://github.com/ACEsuit/mace)                            | [Paper](https://arxiv.org/abs/2401.00096)                   |
| MatRIS    | `matris`    | `matris`           | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/matris/)    | [Repo](https://github.com/HPC-AI-Team/MatRIS)                      | [Paper](https://arxiv.org/abs/2603.02002)                   |
| MatterSim | `mattersim` | `mattersim`        | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/mattersim/) | [Repo](https://github.com/microsoft/mattersim)                     | [Paper](https://arxiv.org/abs/2405.04967)                   |
| MEGNet    | `matgl`     | `matgl`            | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/megnet/)    | [Repo](https://github.com/materialsvirtuallab/megnet)              | [Paper](https://arxiv.org/abs/1812.05055)                   |
| NequIP    | `nequip`    | `nequip`           | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/nequip/)    | [Repo](https://github.com/mir-group/nequip)                        | [Paper](https://arxiv.org/abs/2504.16068)                   |
| Nequix    | `nequix`    | `nequix`           | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/nequix/)    | [Repo](https://github.com/atomicarchitects/nequix)                 | [Paper](https://arxiv.org/abs/2508.16067)                   |
| NewtonNet | `newtonnet` | `newtonnet`        | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/newtonnet/) | [Repo](https://github.com/THGLab/NewtonNet)                        | [Paper](https://doi.org/10.1039/D2DD00008C)                 |
| ORB       | `orb`       | `orb-models`       | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/orb/)       | [Repo](https://github.com/orbital-materials/orb-models)            | [Paper](https://arxiv.org/abs/2504.06231)                   |
| PetMad    | `petmad`    | `upet`             | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/petmad/)    | [Repo](https://github.com/lab-cosmo/upet)                          | [Paper](https://www.nature.com/articles/s41467-025-65662-7) |
| PosEGNN   | N/A         | N/A                | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/posegnn/)   | [Repo](https://github.com/IBM/materials/tree/main/models/pos_egnn) | N/A                                                         |
| SevenNet  | `sevennet`  | `sevenn`           | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/sevennet/)  | [Repo](https://github.com/MDIL-SNU/SevenNet)                       | [Paper](https://arxiv.org/abs/2510.11241)                   |
| TACE      | `tace`      | `TACE`             | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/tace/)      | [Repo](https://github.com/xvzemin/tace)                            | [Paper](https://arxiv.org/abs/2509.14961)                   |
| UMA       | `uma`       | `fairchem-core`    | [API](https://dogusariturk.github.io/MaterialsFramework/api/calculators/uma/)       | [Repo](https://github.com/facebookresearch/fairchem)                      | [Paper](https://arxiv.org/abs/2506.23971)                   |

Non-MLIP calculators: `RandomCalculator` (dependency-free testing stub) and `VASPCalculator` (external licensed VASP backend).

> [!WARNING]
> **PosEGNN** has no installable package on any public index. Clone the repository and add the module directory to `PYTHONPATH` manually:
> ```bash
> git clone --depth 1 https://github.com/IBM/materials.git
> export PYTHONPATH="$PWD/materials/models/pos_egnn:$PYTHONPATH"
> ```

> [!WARNING]
> **GPTFF** is only installable from its upstream git repository:
> ```bash
> uv pip install "gptff @ git+https://github.com/atomly-materials-research-lab/GPTFF.git"
> ```

> [!WARNING]
> **EquFlash** is only installable from its upstream git repository, and the bare install below leaves
> `fairchem-core` missing, so `EquFlashCalculator().calculator` raises `ModuleNotFoundError`. See
> [installation](https://dogusariturk.github.io/MaterialsFramework/installation/) for the CPU-only dependency set that's confirmed to work.
> ```bash
> uv pip install "GGNN @ git+https://github.com/SamsungDS/GGNN.git"
> ```

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

## Installation

We recommend [uv](https://docs.astral.sh/uv/) for dependency management, though a plain `pip` install also works. Use the `Extra` column in the [Supported MLIPs](#supported-mlips) table above to pick which MLIP extras to add.

### uv

```bash
uv add materialsframework
```

Add one or more compatible MLIP extras:

```bash
# Single MLIP
uv add "materialsframework[chgnet]"

# Compatible multi-MLIP stack
uv add "materialsframework[chgnet,matgl,sevennet]"
```

### pip

```bash
pip install materialsframework
```

Add an MLIP extra the same way:

```bash
pip install "materialsframework[chgnet]"
```

See the [installation guide](https://dogusariturk.github.io/MaterialsFramework/installation/) for full setup instructions and [MLIP Conflicts](https://dogusariturk.github.io/MaterialsFramework/mlip-conflicts/) for conflict details.

---

## Quickstart

### Calculators

Every calculator except `MEGNetCalculator` accepts `ase.Atoms` or `pymatgen.Structure` and exposes the same `relax()`/`calculate()` interface, regardless of which MLIP backs it.

```python
from ase.build import bulk
from materialsframework.calculators import MACECalculator

structure = bulk("Cu", crystalstructure="fcc", a=3.6, cubic=True)
calc = MACECalculator()

result = calc.relax(structure)
print(result["final_structure"])
print(result["energy"])
```

`calculate()` evaluates the same properties on the structure exactly as given, with no relaxation step:

```python
result = calc.calculate(structure)
print(result["energy"])
print(result["forces"])
```

### Molecular Dynamics

Calculators that subclass `BaseMDCalculator` add a `run()` method for NVE, NVT/NPT Nose-Hoover, and NPT/Inhomogeneous-NPT Berendsen molecular dynamics.

```python
from ase.build import bulk
from materialsframework.calculators import CHGNetCalculator

structure = bulk("Fe", crystalstructure="bcc", a=2.87, cubic=True)
calc = CHGNetCalculator(ensemble="nvt_nose_hoover", temperature=300)

result = calc.run(structure, steps=1000)
print(result["final_structure"])
```

### Property Analyzers

Analyzers pair with a transformation of the same name: the transformation generates the structures a calculation needs, and the analyzer orchestrates the calculator calls and combines the results.

```python
from ase.build import bulk
from materialsframework.analysis import FormationEnergyAnalyzer
from materialsframework.calculators import CHGNetCalculator

structure = bulk("NaCl", crystalstructure="rocksalt", a=5.64)
analyzer = FormationEnergyAnalyzer(calculator=CHGNetCalculator())

result = analyzer.calculate(structure, is_relaxed=True)
print(result["formation_energy"])
```

### Tools

Standalone utilities such as special quasirandom structure generation, cluster expansion, and phase-field modeling live in `materialsframework.tools`.

```python
from materialsframework.tools import SqsGenerator

generator = SqsGenerator(iterations=1000)
result = generator.generate("Fe0.5Co0.5", crystal_structure="bcc", supercell_size=(2, 2, 2))
print(result["structure"])
print(result["objective"])
```

### Registries

Look up calculators, analyzers, transformations, and tools by name to swap in a new backend without importing every MLIP dependency up front.

```python
from materialsframework.calculators import get_calculator

calc = get_calculator("chgnet")
```

---

## License

Distributed under the GPL-3.0-or-later License. See [GPL-3.0](https://github.com/dogusariturk/MaterialsFramework/blob/main/LICENSE) for details.

---

## Citation

If you use MaterialsFramework in your research, please cite:

> Sarıtürk, D., & Arroyave, R. (2025). MaterialsFramework. Zenodo. https://doi.org/10.5281/zenodo.15731044

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
