"""Feasibility demo: H2 / STO-3G, neutral singlet, 2e in a 2-spatial-orbital
active space (the minimal basis itself). pyscf supplies RHF one-/two-body
integrals; we build the second-quantized Fock Hamiltonian ourselves in the
Sz=0 sector so the SAME selected-CI / NOCI machinery can run on it.

This is a small feasibility demonstration, NOT a quantum-chemistry benchmark:
at (2e,2orb) the sector has dim = C(2,1)*C(2,1) = 4.

pyscf is allowed (shared venv, wheel 2.14.0). Requirements recorded in
results/molecule_requirements.txt.
"""
from __future__ import annotations

import itertools
import numpy as np


class ActiveSpaceHamiltonian:
    """Generic real second-quantized Hamiltonian in a fixed Sz=0 Fock basis.

    spin-orbitals: up channels 0..nsp-1, down channels nsp..2*nsp-1.
    h[p,q] one-body; eri[p,q,r,s] chemists' (pq|rs) in spin-orbital block form.
    """

    def __init__(self, h, eri, nspatial, n_up, n_down):
        self.L = nspatial
        self.n_up = n_up
        self.n_down = n_down
        M = 2 * nspatial
        ups = [sum(1 << i for i in c)
               for c in itertools.combinations(range(nspatial), n_up)]
        dns = [sum(1 << (nspatial + i) for i in c)
               for c in itertools.combinations(range(nspatial), n_down)]
        self.states = np.array([u | d for u in ups for d in dns], dtype=np.int64)
        self.index = {int(m): i for i, m in enumerate(self.states)}
        self.dim = len(self.states)
        self.H = self._build(h, eri, M)

    def _sign(self, mask, orb):
        # annihilate orb: parity of occupied orbitals below it
        return -1.0 if (int(mask) & ((1 << orb) - 1)).bit_count() & 1 else 1.0

    def _build(self, h, eri, M):
        H = np.zeros((self.dim, self.dim))
        for idx, mask in enumerate(self.states):
            mask = int(mask)
            # one-body h_{pq} c_p^dag c_q
            for q in range(M):
                if not ((mask >> q) & 1):
                    continue
                sq = self._sign(mask, q)
                m1 = mask & ~(1 << q)
                for p in range(M):
                    if (m1 >> p) & 1:
                        continue
                    sp = self._sign(m1 | (1 << p), p)
                    m2 = m1 | (1 << p)
                    j = self.index.get(m2)
                    if j is not None:
                        H[idx, j] += h[p, q] * sq * sp
            # two-body: 1/2 sum_{pqrs} eri[p,q,r,s] a_p^dag a_r^dag a_s a_q
            # (chemists' (pq|rs); spin(p)=spin(q), spin(r)=spin(s))
            for q in range(M):
                if not (mask >> q) & 1:
                    continue
                sq = self._sign(mask, q)
                m1 = mask & ~(1 << q)
                for s in range(M):
                    if not (m1 >> s) & 1:
                        continue
                    ss = self._sign(m1, s)
                    m2 = m1 & ~(1 << s)
                    for r in range(M):
                        if (m2 >> r) & 1:
                            continue
                        sr = self._sign(m2 | (1 << r), r)
                        m3 = m2 | (1 << r)
                        for p in range(M):
                            if (m3 >> p) & 1:
                                continue
                            sp = self._sign(m3 | (1 << p), p)
                            m4 = m3 | (1 << p)
                            j = self.index.get(m4)
                            if j is not None:
                                H[idx, j] += 0.5 * eri[p, q, r, s] * sq * ss * sr * sp
        return (H + H.T) / 2

    def exact_ground(self):
        w, v = np.linalg.eigh(self.H)
        return w[0], v[:, 0]

    def external_candidates(self, selected_idx):
        # tiny sector: just return all non-selected states
        sel = set(int(i) for i in selected_idx)
        return np.array(sorted(set(range(self.dim)) - sel), dtype=np.int64)

    def neel_determinant(self):
        # RHF reference: occupy the lowest spatial orbital with both spins
        m = (1 << 0) | (1 << self.L)
        return m

    def hf_determinant(self):
        return self.neel_determinant()


def h2_active(R_angstrom: float, basis: str = "sto-3g"):
    """Return (ActiveSpaceHamiltonian, fci_energy, rhf_energy, mf)."""
    from pyscf import gto, scf, fci
    from pyscf import ao2mo

    atom = f"H 0 0 0; H 0 0 {R_angstrom}"
    mol = gto.M(atom=atom, basis=basis, spin=0, charge=0, verbose=0)
    mf = scf.RHF(mol).run()
    nsp = mol.nao  # minimal basis: 2 AOs -> 2 spatial MOs
    # active space = all MOs (2e, 2 orb)
    mo = mf.mo_coeff
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = ao2mo.kernel(mol, mo, aosym="s1").reshape(nsp, nsp, nsp, nsp)
    # embed spin: spin-orbitals 0..nsp-1 up, nsp..2nsp-1 down
    M = 2 * nsp
    h = np.zeros((M, M))
    h[:nsp, :nsp] = h1e
    h[nsp:, nsp:] = h1e
    # spin-orbital eri: chemists' (pq|rs)=eri[p,q,r,s], nonzero only when
    # spin(p)=spin(q) and spin(r)=spin(s) (operator a_p^dag a_r^dag a_s a_q).
    E = np.zeros((M, M, M, M))
    for a in range(nsp):
        for b in range(nsp):
            for c in range(nsp):
                for d in range(nsp):
                    val = eri[a, b, c, d]
                    for sig in (0, 1):       # spin(p)=spin(q)
                        for tau in (0, 1):   # spin(r)=spin(s)
                            p = a + nsp * sig
                            q = b + nsp * sig
                            r = c + nsp * tau
                            s = d + nsp * tau
                            E[p, q, r, s] = val
    ham = ActiveSpaceHamiltonian(h, E, nsp, n_up=1, n_down=1)
    cisolver = fci.FCI(mf)
    fci_e = cisolver.kernel()[0]
    return ham, float(fci_e), float(mf.e_tot), mf
