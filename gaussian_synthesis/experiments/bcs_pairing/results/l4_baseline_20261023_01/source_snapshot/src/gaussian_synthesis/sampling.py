"""Independent normalized invariant sampling of four-mode even Gaussian rays."""

import numpy as np
from .basis import FockBasis
from .gaussian import spin_to_fock


class HaarFourModeSampler:
    max_batch = 8192

    def __init__(self, rng, basis=None):
        self.basis = basis or FockBasis()
        self.basis.require_four_mode_even()
        self.rng = rng

    def sample(self, count):
        if (isinstance(count, bool) or not isinstance(count, (int, np.integer))
                or not 1 <= count <= self.max_batch):
            raise ValueError("Batch size must be an integer in [1, 8192]")
        r = self.rng.standard_normal((count, 8))
        norms = np.linalg.norm(r, axis=1)
        # Degeneracy has zero probability in the mathematical sampler. Redraw
        # numerically degenerate rows rather than biasing the whole batch.
        bad = norms < 1e-14
        while np.any(bad):
            r[bad] = self.rng.standard_normal((int(bad.sum()), 8))
            norms = np.linalg.norm(r, axis=1)
            bad = norms < 1e-14
        r /= norms[:, None]
        s = self.rng.standard_normal((count, 8))
        s -= np.einsum("bi,bi->b", r, s)[:, None]*r
        norms = np.linalg.norm(s, axis=1)
        bad = norms < 1e-14
        while np.any(bad):
            s[bad] = self.rng.standard_normal((int(bad.sum()), 8))
            s[bad] -= np.einsum("bi,bi->b", r[bad], s[bad])[:, None]*r[bad]
            norms = np.linalg.norm(s, axis=1)
            bad = norms < 1e-14
        s /= norms[:, None]
        return spin_to_fock((r+1j*s)/np.sqrt(2))
