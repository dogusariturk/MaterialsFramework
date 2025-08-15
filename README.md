<div align="center">

# MaterialsFramework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15731044.svg)](https://doi.org/10.5281/zenodo.15731044)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://opensource.org/license/gpl-3-0)
[![Python](https://img.shields.io/badge/python-3.11+-brightgreen.svg)](https://www.python.org/)

<p>
  A modular and extensible framework for deploying, benchmarking, and experimenting with state-of-the-art machine learning potentials in materials science.
</p>

<p>
  <a href="https://github.com/dogusariturk/MaterialsFramework/issues/new?labels=bug">Report a Bug</a> |
  <a href="https://github.com/dogusariturk/MaterialsFramework/issues/new?labels=enhancement">Request a Feature</a>
</p>

</div>

---

## Getting Started

Follow the steps below to get a local copy of the project up and running.


### Prerequisites

This project uses `conda` for managing dependencies. Several `environment.yml` files are provided to support different model groups.

| Environment File          | Supported Models                                                                                                                       |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `environment.yml`         | M3GNet / MEGNet                                                                                                                        |
| `main-environment.yml`    | AlphaNet, CHGNet, DeepMD, Eqnorm, EqV2 / eSEN, GPTFF, GRACE, HIENet, M3GNet / MEGNet, MatterSim, NewtonNet, PET-MAD, PosEGNN, SevenNet |
| `orb-uma-environment.yml` | ORB, UMA                                                                                                                               |
| `mace-environment.yml`    | MACE                                                                                                                                   |
| `alignn-environment.yml`  | ALIGNN-FF                                                                                                                              |                                                                                               |

 ### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dogusariturk/MaterialsFramework.git

2.  **Navigate into the project directory:**
    ```sh
    cd MaterialsFramework
    ```
3.  **Create a conda environment from the desired file:**
    ```sh
    conda env create -f <environment_file.yml>
    ```
4.  **Activate the environment:**
    ```sh
    conda activate <environment_name>
    ```
5.  **Install the framework in editable mode:**
    ```sh
    pip install -e .
    ```

## Modules

Below are the main modules and classes for analysis and tools:

### `analysis`

- **`ANNNIStackingFaultAnalyzer`**  
  Tools for simulating and analyzing the Axial Next-Nearest-Neighbor Ising (ANNNI) model, useful for studying magnetic and structural phase transitions.


- **`BainPathAnalyzer`**  
  Implements the Bain transformation, providing utilities to analyze and visualize the transformation path between fcc and bcc crystal structures.


- **`CubicElasticConstantsAnalyzer`**
  Provides methods to calculate cubic elastic constants (C11, C12, C44) and derived properties like Young's modulus, bulk modulus, shear modulus, Poisson's ratio, and Pugh's ratio.

> [!CAUTION]
> `CubicElasticConstantsAnalyzer` only works with cubic/orthogonal cells. 


- **`ElasticConstantsAnalyzer`**
  General tools for calculating elastic constants from stress-strain data, including methods for fitting and extracting Voigt-Reuss-Hill averages.


- **`EOSAnalyzer`**  
  Equation of State (EOS) fitting and analysis tools, including routines to fit energy-volume data and extract bulk properties.


- **`FormationEnergyAnalyzer`**
  Computes the formation energy of a material based on its composition and structure. The class can be used to analyze the stability of materials.


- **`PhonopyAnalyzer`**  
  Interfaces and helpers for phonon calculations using the Phonopy package, including phonon band structure and density of states analysis.


- **`Phono3pyAnalyzer`**  
  Tools for third-order phonon calculations with Phono3py, enabling analysis of lattice thermal conductivity and anharmonic effects.


### `tools`

- **`ClusterExpansion`**  
  Provides tools for constructing and analyzing cluster expansions, including fitting methods and validation routines.


- **`PhaseFieldModel`**  
  Implements the Cahn-Hilliard equation for simulating phase separation and microstructure evolution in materials.


- **`Sqs2tdb`**  
  Converts Special Quasirandom Structures (SQS) data to thermodynamic database (TDB) files for use in thermodynamic modeling.


- **`StabilityMap`**
    Tools for generating stability maps of materials, visualizing phase stability as a function of composition and temperature.


## Example Workflows

The following example scripts demonstrate typical use cases:

- **Geometry Optimization**
    ```python
    from ase.build import bulk
    from materialsframework.calculators import GraceCalculator
    
    struct = bulk(name="Cu", crystalstructure="fcc", a=3.6, cubic=True)
    
    calc = GraceCalculator()
    res = calc.relax(struct)
    
    print(res["final_structure"])
    print(res["forces"])
    print(res["stress"])
    ```

- **Cubic Elastic Constants**  
    ```python
    from ase.build import bulk
    from materialsframework.calculators import GraceCalculator
    from materialsframework.analysis import CubicElasticConstantsAnalyzer
    
    struct = bulk(name="Cu", crystalstructure="fcc", a=3.6, cubic=True)
    
    calc = GraceCalculator()
    cubic_elastic_constants = CubicElasticConstantsAnalyzer(calculator=calc)
    res = cubic_elastic_constants.calculate(struct)
    
    print(res["c11"])
    print(res["c12"])
    print(res["c44"])
    print(res["youngs_modulus"])
    print(res["voigt_reuss_hill_bulk_modulus"])
    print(res["voigt_reuss_hill_shear_modulus"])
    print(res["poisson_ratio"])
    print(res["pugh_ratio"])
    ```

- **Equation Of State Analysis**  
    ```python
    from ase.build import bulk
    from materialsframework.calculators import GraceCalculator
    from materialsframework.analysis import EOSAnalyzer
    
    struct = bulk(name="Cu", crystalstructure="fcc", a=3.6, cubic=True)
    
    calc = GraceCalculator()
    eos_analyzer = EOSAnalyzer(calculator=calc)
    res = eos_analyzer.calculate(struct)
    
    print(res["e0"])
    print(res["b0"])
    print(res["b0_GPa"])
    print(res["b1"])
    print(res["v0"])
    ```

- **Phonon Calculations**
    ```python
    from ase.build import bulk
    from materialsframework.calculators import GraceCalculator
    from materialsframework.analysis import PhonopyAnalyzer
    
    struct = bulk(name="Cu", crystalstructure="fcc", a=3.6, cubic=True)
    
    calc = GraceCalculator()
    phonopy_analyzer = PhonopyAnalyzer(calculator=calc)
    res = phonopy_analyzer.calculate(struct)
    
    print(res["total_dos"])
    print(res["projected_dos"])
    print(res["thermal_properties"])
    ```

- **Molecular Dynamics**
    ```python
    from ase.build import bulk
    from materialsframework.calculators import GraceCalculator
    
    struct = bulk(name="Cu", crystalstructure="fcc", a=3.6, cubic=True)
    
    calc = GraceCalculator(ensemble="nvt_nose_hoover", verbose=True, temperature=300)
    res = calc.run(structure=struct, steps=1000)
    
    print(res["total_energy"])
    print(res["potential_energy"])
    print(res["kinetic_energy"])
    print(res["temperature"])
    print(res["final_structure"])
    ```

## Citing

We are currently preparing a preprint for publication. If you use MaterialsFramework in your research, please cite the following:

> Sarıtürk, D., & Arroyave, R. (2025). MaterialsFramework. Zenodo. https://doi.org/10.5281/zenodo.15731044

```bibtex
@software{sariturk_2025_15731044,
  author       = {Sarıtürk, Doğuhan and Arroyave, Raymundo},
  title        = {MaterialsFramework},
  month        = jun,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15731044},
  url          = {https://doi.org/10.5281/zenodo.15731044},
}
```
## License

Distributed under the GPLv3 License. See [GPLv3 License](https://opensource.org/license/gpl-3-0) for more information.

## Contact

Doguhan Sariturk - [doguhan.sariturk@gmail.com](mailto:doguhan.sariturk@gmail.com)

