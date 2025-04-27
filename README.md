<div align="center">

# MaterialsFramework

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://opensource.org/license/gpl-3-0)
[![Python](https://img.shields.io/badge/python-3.8+-brightgreen.svg)](https://www.python.org/)

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

| Environment File           | Supported Models                                                                                            |
|----------------------------|-------------------------------------------------------------------------------------------------------------|
| `environment.yml`          | M3GNet / MEGNet                                                                                             |
| `main-environment.yml`     | CHGNet, DeepMD, EqV2 / eSEN, GPTFF, GRACE, HIENet, M3GNet / MEGNet, MatterSim, NewtonNet, PosEGNN, SevenNet |
| `orb-environment.yml`      | ORB                                                                                                         |
| `mace-environment.yml`     | MACE                                                                                                        |
| `alignn-environment.yml`   | ALIGNN-FF                                                                                                   |
| `alphanet-environment.yml` | AlphaNet                                                                                                    |

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

## License

Distributed under the GPLv3 License. See [GPLv3 License](https://opensource.org/license/gpl-3-0) for more information.

## Contact

Doguhan Sariturk - [doguhan.sariturk@gmail.com](mailto:doguhan.sariturk@gmail.com)

