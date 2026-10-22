"""Occupation basis: bit j labels mode j; creation operators in ascending order."""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class FockBasis:
    n: int = 4
    parity: str = "even"
    occupations: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.n, bool) or not isinstance(self.n, (int, np.integer)):
            raise ValueError("n must be an integer")
        if not 2 <= self.n <= 12:
            raise ValueError("This dense v1 basis supports 2 <= n <= 12")
        if self.parity not in ("even", "odd"):
            raise ValueError("parity must be 'even' or 'odd'")
        bit = int(self.parity == "odd")
        object.__setattr__(self, "occupations", tuple(
            x for x in range(1 << self.n) if x.bit_count() % 2 == bit
        ))

    @property
    def dimension(self):
        return 1 << (self.n - 1)

    def index(self, occupation):
        try:
            return self.occupations.index(occupation)
        except ValueError as error:
            raise ValueError("Occupation is outside this parity sector") from error

    def ket(self, occupation):
        result = np.zeros(self.dimension, dtype=np.complex128)
        result[self.index(occupation)] = 1
        return result

    def vector(self, value, *, nonzero=False):
        result = np.array(value, dtype=np.complex128, copy=True)
        if result.shape != (self.dimension,):
            raise ValueError(f"Expected vector shape ({self.dimension},)")
        if not np.all(np.isfinite(result)):
            raise ValueError("State coefficients must be finite")
        norm = np.linalg.norm(result)
        if not np.isfinite(norm) or (nonzero and norm == 0):
            raise ValueError("State must have a finite, nonzero norm")
        return result

    def require_four_mode_even(self):
        if self.n != 4 or self.parity != "even":
            raise NotImplementedError("The v1 Haar sampler supports n=4, even parity only")
