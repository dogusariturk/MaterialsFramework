# Analyzers

`MaterialsFramework` provides analyzer classes for common materials property calculations. Instantiate an analyzer with a calculator, then call its `calculate()` method. Most analyzers take one structure; ANNNI takes a composition, and NEB takes initial and final structures.

```python
analyzer = SomeAnalyzer(calculator=calc)
results = analyzer.calculate(structure)
```

<div class="grid cards" markdown>

-   __ANNNI Stacking Faults__

    ---

    ANNNI-based intrinsic and extrinsic stacking fault energies.

    [:octicons-arrow-right-24: ANNNI Stacking Faults](annni.md)

-   __Bain Path__

    ---

    Energy along the FCC-to-BCC Bain transformation path.

    [:octicons-arrow-right-24: Bain Path](bain.md)

-   __Coefficient of Thermal Expansion__

    ---

    Coefficient of thermal expansion from NPT-MD V-T data.

    [:octicons-arrow-right-24: Coefficient of Thermal Expansion](cte.md)

-   __Cubic Elastic Constants__

    ---

    Cubic elastic constants and derived moduli (B, G, E, ν).

    [:octicons-arrow-right-24: Cubic Elastic Constants](cubic_elastic.md)

-   __Elastic Constants__

    ---

    Full elastic tensor and Voigt-Reuss-Hill averages.

    [:octicons-arrow-right-24: Elastic Constants](elastic.md)

-   __Equation of State__

    ---

    Equation-of-state curve fitting from E-V data.

    [:octicons-arrow-right-24: Equation of State](eos.md)

-   __Formation Energy__

    ---

    Formation energy per atom.

    [:octicons-arrow-right-24: Formation Energy](formation_energy.md)

-   __H-Solubility__

    ---

    Hydrogen insertion and solution energies.

    [:octicons-arrow-right-24: H-Solubility](h_solubility.md)

-   __Nudged Elastic Band__

    ---

    Nudged elastic band minimum energy path and reaction barrier.

    [:octicons-arrow-right-24: Nudged Elastic Band](neb.md)

-   __Phonopy__

    ---

    Total/projected phonon DOS and thermal properties.

    [:octicons-arrow-right-24: Phonopy](phonopy.md)

-   __Phono3py__

    ---

    Anharmonic force constants and lattice thermal conductivity.

    [:octicons-arrow-right-24: Phono3py](phono3py.md)

-   __Surface Binding Energy__

    ---

    Surface binding energies, a first-principles proxy for sputtering resistance.

    [:octicons-arrow-right-24: Surface Binding Energy](sbe.md)

-   __Surface Energy__

    ---

    Slab surface energies for a given Miller index.

    [:octicons-arrow-right-24: Surface Energy](surface.md)

-   __USFE__

    ---

    Generalized stacking fault energy curves and unstable SFE.

    [:octicons-arrow-right-24: USFE](usfe.md)

</div>
