"""Shared physical-unit conversion factors used across the analysis package."""

from __future__ import annotations

from ase import units

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"

EV_A3_TO_GPA: float = 1 / units.GPa
EV_A2_TO_J_M2: float = units.m**2 / units.J
EV_A2_TO_MJ_M2: float = EV_A2_TO_J_M2 * 1000
