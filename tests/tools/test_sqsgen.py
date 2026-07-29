"""Tests for SqsGenerator."""

from __future__ import annotations

import pytest
from pymatgen.core import Structure

pytest.importorskip("sqsgenerator")

from materialsframework.tools.sqsgen import SqsGenerator


def test_default_params() -> None:
    """Default iteration count and mode are stored correctly."""
    t = SqsGenerator()
    assert t._iterations == 1000
    assert t._mode == "random"
    assert t._structure_format == "pymatgen"


@pytest.mark.integration
def test_generate_returns_structure_and_objective() -> None:
    """generate() returns a dict with 'structure' and 'objective' keys."""
    t = SqsGenerator(iterations=50)
    result = t.generate("Fe0.5Co0.5", crystal_structure="bcc", supercell_size=(2, 2, 2))
    assert "structure" in result
    assert "objective" in result
    assert isinstance(result["structure"], Structure)
    assert isinstance(result["objective"], float)


@pytest.mark.integration
def test_generate_independent_calls() -> None:
    """Two calls to generate() with different compositions return independent results."""
    t = SqsGenerator(iterations=50)
    result1 = t.generate("Fe0.5Co0.5", crystal_structure="bcc", supercell_size=(2, 2, 2))
    result2 = t.generate("Fe0.25Co0.75", crystal_structure="bcc", supercell_size=(2, 2, 2))
    assert isinstance(result1["structure"], Structure)
    assert isinstance(result2["structure"], Structure)
    assert result1 is not result2


@pytest.mark.integration
def test_generate_composition_matches(bcc_fe) -> None:
    """Generated structure contains the expected elements."""
    t = SqsGenerator(iterations=50)
    result = t.generate("Fe0.5Co0.5", crystal_structure="bcc", supercell_size=(2, 2, 2))
    elements = {str(el) for el in result["structure"].elements}
    assert "Fe" in elements
    assert "Co" in elements
