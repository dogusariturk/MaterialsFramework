"""Base class for all relaxers"""

from abc import ABC, abstractmethod

from pymatgen.core import Structure


class Relaxer(ABC):
    """Base class for all relaxers"""

    @abstractmethod
    def relax(self, structure: Structure):
        ...
