"""Half-filled Hubbard chain, Sz=0 sector, exact diagonalization in the Fock basis.

Conventions
-----------
* L sites, open chain, hopping ``t`` (we set t=1), on-site repulsion ``U``.
* Spin-orbitals: up orbitals 0..L-1 (site i = orbital i),
  down orbitals L..2L-1 (site i = orbital L+i).
* A determinant is an integer bitmask over the 2L spin-orbitals.
* Half filling with Sz=0: N_up = N_down = L/2 electrons.

All Hamiltonians are real symmetric. We build a dense matrix in the full
Sz=0 sector (dim = C(L, L/2)^2, <= 400 for L<=6), so exact diagonalization
is trivial and serves as the reference energy.

This module is self-contained (numpy only) and read-only w.r.t. the rest of
the repository.
"""
from __future__ import annotations

import itertools
import numpy as np


def _parity(n: int) -> int:
    """±1 Jordan-Wigner parity of a bitmask (popcount mod 2)."""
    return -1 if (n & 0x55555555).bit_count() & 1 else 1


class HubbardModel:
    def __init__(self, L: int, t: float = 1.0, U: float = 0.0):
        assert L % 2 == 0, "half-filled chain uses even L"
        self.L = L
        self.t = float(t)
        self.U = float(U)
        self.n_up = L // 2
        self.n_down = L // 2
        self._build_basis()
        self._build_hamiltonian()

    # ---- basis -----------------------------------------------------------
    def _build_basis(self):
        L = self.L
        ups = [sum(1 << i for i in comb)
               for comb in itertools.combinations(range(L), self.n_up)]
        downs = [sum(1 << (L + i) for i in comb)
                 for comb in itertools.combinations(range(L), self.n_down)]
        # each determinant = up-bits | down-bits
        self.states = np.array([u | d for u in ups for d in downs], dtype=np.int64)
        self.index = {int(m): i for i, m in enumerate(self.states)}
        self.dim = len(self.states)

    def _hop_sign(self, mask: int, a: int, b: int) -> float:
        """Sign for c_b^dag c_a acting on `mask` (a occupied, b empty).
        Removes bit a, then adds bit b; Jordan-Wigner parity from orbitals
        removed/added in canonical order."""
        # annihilate a: parity = popcount of orbitals below a that are occupied
        s = -1.0 if (mask & ((1 << a) - 1)).bit_count() & 1 else 1.0
        m2 = mask & ~(1 << a)
        # create b: parity = popcount of orbitals below b occupied in m2
        s *= -1.0 if (m2 & ((1 << b) - 1)).bit_count() & 1 else 1.0
        return s

    def _build_hamiltonian(self):
        L, t, U = self.L, self.t, self.U
        H = np.zeros((self.dim, self.dim))
        for idx, mask in enumerate(self.states):
            mask = int(mask)
            # diagonal interaction: U * number of doubly occupied sites
            dbl = 0
            for i in range(L):
                if (mask >> i) & 1 and (mask >> (L + i)) & 1:
                    dbl += 1
            H[idx, idx] += U * dbl
            # hopping on each bond, both spins
            for i in range(L - 1):
                for a in (i, L + i):          # occupied orbital
                    b = i + 1 if a < L else L + i + 1
                    if ((mask >> a) & 1) and not ((mask >> b) & 1):
                        s = self._hop_sign(mask, a, b)
                        m2 = (mask & ~(1 << a)) | (1 << b)
                        j = self.index[m2]
                        H[idx, j] += -t * s
                        H[j, idx] += -t * s
        self.H = H

    # ---- exact diagonalization ------------------------------------------
    def exact_ground(self):
        w, v = np.linalg.eigh(self.H)
        return w[0], v[:, 0]

    # ---- determinants ----------------------------------------------------
    def hf_determinant(self) -> int:
        """Single-determinant seed. We use the localized Neel product
        (up on even sites, down on odd sites), which is Sz=0 and concrete;
        additional randomized seeds are supplied by the driver."""
        return self.neel_determinant()

    def neel_determinant(self) -> int:
        """Localized Neel product: up on even sites, down on odd sites
        (Sz=0). A concrete, reproducible single-determinant seed."""
        m = 0
        for i in range(self.L):
            if i % 2 == 0:
                m |= 1 << i            # up on even
            else:
                m |= 1 << (self.L + i)  # down on odd
        return m

    def all_determinants(self):
        return self.states

    # ---- excitation generation ------------------------------------------
    def external_candidates(self, selected_idx):
        """All determinants reachable by at most 2 spin-conserving
        excitations from any selected determinant, excluding selected ones.

        Returns a sorted numpy array of basis indices.
        """
        sel = set(int(i) for i in selected_idx)
        cand = set()
        for si in selected_idx:
            mask = int(self.states[si])
            occ = [b for b in range(2 * self.L) if (mask >> b) & 1]
            unocc = [b for b in range(2 * self.L) if not ((mask >> b) & 1)]
            # singles
            for a in occ:
                for b in unocc:
                    if (a % self.L) == (b % self.L) and (a < self.L) == (b < self.L):
                        # same spin channel: allowed
                        pass
                    if (a < self.L) != (b < self.L):
                        continue  # spin flip not allowed in Sz=0
                    m2 = (mask & ~(1 << a)) | (1 << b)
                    j = self.index.get(m2)
                    if j is not None and j not in sel:
                        cand.add(j)
            # doubles
            for a1, a2 in itertools.combinations(occ, 2):
                for b1, b2 in itertools.combinations(unocc, 2):
                    if (a1 < self.L) != (b1 < self.L) or (a2 < self.L) != (b2 < self.L):
                        continue
                    m2 = mask
                    m2 = (m2 & ~(1 << a1)) | (1 << b1)
                    m2 = (m2 & ~(1 << a2)) | (1 << b2)
                    j = self.index.get(m2)
                    if j is not None and j not in sel:
                        cand.add(j)
        return np.array(sorted(cand), dtype=np.int64)
