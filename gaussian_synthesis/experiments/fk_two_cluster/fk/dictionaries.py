"""Full and local-product Gaussian dictionaries with explicit frame constants."""

from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
from .gaussian import GaussianBatch, _masks, sample_gaussian


def _log_dimension(n, k):
    return sum(math.log1p(k/(2*n-i-j))
               for i in range(1, n+1) for j in range(i+1, n+1))


@lru_cache(maxsize=2)
def _product_destinations(parity):
    local = _masks(4, parity)
    # Local A occupies low bits.  Tensor coefficients are indexed [B,A].
    combined = ((local[:, None] << 4) | local[None, :]).reshape(-1)
    return np.searchsorted(_masks(8, 0), combined)


@dataclass(frozen=True)
class Dictionary:
    name: str = "full"

    def __post_init__(self):
        if self.name not in ("full", "product"):
            raise ValueError("dictionary name must be 'full' or 'product'")

    def log_frame_constant(self, k):
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        if self.name == "full":
            return _log_dimension(8, k)
        # Equal probability on even-even and odd-odd.  Each block is a product
        # of two four-mode Gaussian frames.  The factor 2 cancels the mixture
        # probability, yielding the sum of the two orthogonal replica blocks.
        return math.log(2) + 2*_log_dimension(4, k)

    def sample(self, count, rng):
        if self.name == "full":
            return sample_gaussian(8, count, rng)
        if (isinstance(count, bool) or not isinstance(count, (int, np.integer))
                or count < 1):
            raise ValueError("count must be a positive integer")
        # Independent Bernoulli choices are essential to the iid validation
        # assumptions.  Do not replace with an exactly balanced batch.
        sectors = rng.integers(0, 2, size=count, dtype=np.int8)
        kets = np.zeros((count, 128), dtype=np.complex128)
        covariances = np.zeros((count, 16, 16), dtype=float)
        for parity in (0, 1):
            rows = np.flatnonzero(sectors == parity)
            if len(rows) == 0:
                continue
            left = sample_gaussian(4, len(rows), rng, parity)
            right = sample_gaussian(4, len(rows), rng, parity)
            products = (right.kets[:, :, None] * left.kets[:, None, :]).reshape(len(rows), 64)
            kets[np.ix_(rows, _product_destinations(parity))] = products
            covariances[rows, :8, :8] = left.covariances
            covariances[rows, 8:, 8:] = right.covariances
        return GaussianBatch(kets, covariances, sectors)

