"""
This module contains utility functions for MaterialsFramework.
"""
import numpy as np
from itertools import combinations_with_replacement, permutations

__author__ = "Doguhan Sariturk"
__email__ = "dogu.sariturk@gmail.com"


def generate_compositions(num_elements=5, step=12.5):
    """
    Generate all unique compositions of a given number of elements that sum to 100.

    Args:
        num_elements (int): Number of elements in the composition.
        step (float): Increment for each component's value.

    Returns:
        List of unique compositions as tuples.
    """
    composition_set = set()

    results = [combo for combo in combinations_with_replacement(np.arange(0, 100 + step, step), num_elements)
               if round(sum(combo), 10) == 100]

    for result in results:
        composition_set.update(permutations(result))

    return list(composition_set)
