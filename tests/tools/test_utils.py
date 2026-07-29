"""Tests for tools/utils.py, no external dependencies."""

from __future__ import annotations

from materialsframework.tools.utils import generate_compositions


def test_all_compositions_sum_to_100() -> None:
    """Every returned composition sums to exactly 100."""
    for comp in generate_compositions(num_elements=2, step=50):
        assert abs(sum(comp) - 100) < 1e-8


def test_binary_step50_count() -> None:
    """Binary system at step=50 has exactly 3 compositions (0+100, 50+50, 100+0)."""
    result = generate_compositions(num_elements=2, step=50)
    assert len(result) == 3


def test_returns_list_of_tuples() -> None:
    """Return type is a list of tuples."""
    result = generate_compositions(num_elements=2, step=50)
    assert isinstance(result, list)
    assert all(isinstance(c, tuple) for c in result)


def test_element_count_matches_num_elements() -> None:
    """Each composition tuple has num_elements entries."""
    for comp in generate_compositions(num_elements=3, step=50):
        assert len(comp) == 3
