"""Explicit orthonormal Fock coordinates; projection never renormalizes a ket.

An arbitrary coordinate sector is not a symmetry certificate for synthesis.
The existing synthesis API continues to accept only its supported FockBasis.
"""

from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from types import MappingProxyType
import numpy as np
from .basis import FermionSpace


def _check_capacity(size, maximum):
    if isinstance(maximum, (bool, np.bool_)) or not isinstance(maximum, (int, np.integer)) or maximum < 1:
        raise ValueError("max_dimension must be a positive integer")
    if size > maximum:
        raise ValueError(f"Requested {size} coordinates exceeds max_dimension={maximum}")


@dataclass(frozen=True)
class OccupationSector:
    space: FermionSpace
    occupations: tuple
    _indices: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.space, FermionSpace):
            raise TypeError("space must be FermionSpace metadata")
        masks = tuple(self.space.validate_occupation(x) for x in self.occupations)
        if not masks or len(set(masks)) != len(masks):
            raise ValueError("Sector must have nonempty, distinct occupation masks")
        object.__setattr__(self, "occupations", masks)
        object.__setattr__(self, "_indices", MappingProxyType({mask: i for i, mask in enumerate(masks)}))

    @property
    def n(self):
        return self.space.n

    @property
    def parity(self):
        return self.space.parity

    @property
    def coordinate_dimension(self):
        return len(self.occupations)

    @property
    def dimension(self):
        return self.coordinate_dimension

    @property
    def ambient_dimension(self):
        return self.space.ambient_dimension

    def index(self, occupation):
        occupation = self.space.validate_occupation(occupation)
        try:
            return self._indices[occupation]
        except KeyError as error:
            raise ValueError("Occupation is outside this coordinate sector") from error

    def vector(self, values, *, nonzero=False):
        result = np.array(values, dtype=np.complex128, copy=True)
        if result.shape != (self.dimension,) or not np.all(np.isfinite(result)):
            raise ValueError(f"Expected {self.dimension} finite complex coordinates")
        norm = np.linalg.norm(result)
        if not np.isfinite(norm) or (nonzero and norm == 0):
            raise ValueError("Vector must have finite, nonzero norm")
        return result

    def ket(self, occupation):
        result = np.zeros(self.dimension, dtype=np.complex128)
        result[self.index(occupation)] = 1
        return result

    def embed(self, values, *, max_dimension):
        """Embed into ascending-bitmask parity coordinates, with an explicit cap."""
        _check_capacity(self.ambient_dimension, max_dimension)
        values = self.vector(values)
        result = np.zeros(self.ambient_dimension, dtype=np.complex128)
        # Every consecutive pair of integer masks contains exactly one of each parity.
        result[[mask >> 1 for mask in self.occupations]] = values
        return result

    def restrict(self, full_values):
        """Select from ascending-bitmask parity coordinates, without normalization."""
        values = np.asarray(full_values, dtype=np.complex128)
        if values.shape != (self.ambient_dimension,) or not np.all(np.isfinite(values)):
            raise ValueError("Full vector has the wrong dimension or nonfinite values")
        return values[[mask >> 1 for mask in self.occupations]].copy()

    def embed_into(self, values, destination, *, max_dimension):
        """Embed into another explicit coordinate sector containing this one."""
        if self.space != destination.space:
            raise ValueError("Source and destination must share the physical space")
        _check_capacity(destination.dimension, max_dimension)
        indices = [destination.index(mask) for mask in self.occupations]
        result = np.zeros(destination.dimension, dtype=np.complex128)
        result[indices] = self.vector(values)
        return result


class NumberSector(OccupationSector):
    """All Fock configurations of a fixed electron number, including unpaired ones."""

    def __init__(self, space, particle_number, *, max_dimension=100_000):
        if not isinstance(space, FermionSpace):
            raise TypeError("space must be FermionSpace metadata")
        if (isinstance(particle_number, (bool, np.bool_))
                or not isinstance(particle_number, (int, np.integer))
                or not 0 <= particle_number <= space.n):
            raise ValueError("particle_number must be an integer between 0 and n")
        particle_number = int(particle_number)
        if particle_number % 2 != int(space.parity == "odd"):
            raise ValueError("Particle number is incompatible with the space parity")
        _check_capacity(comb(space.n, particle_number), max_dimension)
        masks = tuple(sorted(sum(1 << j for j in indices)
                             for indices in combinations(range(space.n), particle_number)))
        super().__init__(space, masks)
        object.__setattr__(self, "particle_number", particle_number)
