"""BCS-specific pairing of physical modes, built on generic Fock coordinates."""

from dataclasses import dataclass, field
from itertools import combinations
from math import comb
import numpy as np
from gaussian_synthesis import FermionSpace, OccupationSector


@dataclass(frozen=True)
class PairBasis:
    L: int
    m: int
    max_dimension: int = 10000
    pair_masks: tuple = field(init=False)
    sector: OccupationSector = field(init=False)
    occupations: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        for name in ('L', 'm', 'max_dimension'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f'{name} must be an integer')
        if not 1 <= self.L <= 24 or not 0 <= self.m <= self.L or self.max_dimension < 1:
            raise ValueError('Require 1 <= L <= 24, 0 <= m <= L and positive capacity')
        if comb(self.L, self.m) > self.max_dimension:
            raise ValueError('Pair basis exceeds the coordinate budget')
        masks = tuple(sorted(sum(1 << j for j in indices)
                             for indices in combinations(range(self.L), self.m)))
        full = tuple(sum(3 << (2*j) for j in range(self.L) if mask & (1 << j)) for mask in masks)
        occupations = np.array([[bool(mask & (1 << j)) for j in range(self.L)] for mask in masks], dtype=int)
        occupations.flags.writeable = False
        object.__setattr__(self, 'pair_masks', masks)
        object.__setattr__(self, 'sector', OccupationSector(FermionSpace(2*self.L), full))
        object.__setattr__(self, 'occupations', occupations)

    @property
    def dimension(self):
        return self.sector.dimension
