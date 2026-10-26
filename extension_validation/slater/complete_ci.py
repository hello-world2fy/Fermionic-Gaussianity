"""Fock-complete residual-coverage selected CI.

Pool at every step
------------------
A = [ all external orthonormal Fock basis states (span(r) guaranteed) ]
    + [ 40 non-orthogonal Slater atoms around the dominant Ritz determinant ].
Because every Fock component of the residual is itself an orthonormal column,
span(A) contains r exactly: proj_residual = 0 and the convex-J KKT residual
||C(u)-r||/||r|| is a genuine optimization accuracy (target < 1e-5), not a
span gap.

Methods, SAME pool, pick ONE atom per step to budget (8/12):
  * ``cipsi``            add Fock det with largest PT2 weight |<d|r>|^2/(H_dd-E).
  * ``greedy_noci``      add pool atom with largest 2x2 Ritz drop.
  * ``replica_coef``     ABLATION: convex-J k=2, add atom with |c_a| max
                         (coefficient-only; NOT the full energy-aware rule).
  * ``replica_energy``   convex-J k=2, score |c_a|^2 / (|<a|H|a>-E|+eps) and add
                         the best-scoring atom (energy-aware replica-guided rule).

Accounting
----------
n_det records the size of the Ritz space whose energy E is reported (len(V)
BEFORE the next append). A final Ritz row is emitted at n_det=budget. Timing
buckets belong to the step that produced that n_det state.

Duplicate rejection
-------------------
An atom is rejected as already-in-span when its projection residual onto the
current orthonormal Ritz basis Q is tiny: ||a - Q Q^T a|| < 1e-10 (not a raw
overlap threshold, which would let QR invent arbitrary span extension). Picks
are tried in score order until a linearly-independent atom is found.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from hubbard import HubbardModel
import noci

EPS_DEN = 1e-4


@dataclass
class CRec:
    n_det: int
    E: float
    dE: float
    residual: float
    kkt_err: float
    proj_residual: float
    gauge: float
    t_pool: float
    t_residual: float
    t_optimize: float
    t_ritz: float
    pool_size: int
    picked_nonorth: bool


def _orthonormal(V):
    Vm = np.array(V).T
    Q, _ = np.linalg.qr(Vm)
    return Q


def _ritz_energy(Q, H):
    return float(np.linalg.eigvalsh(Q.T @ H @ Q)[0])


def run_complete(model: HubbardModel, seed_mask: int, method: str,
                 budget: int = 12, k: float = 2.0, E_exact=None,
                 n_atoms: int = 40, eps: float = 0.6) -> list:
    H = model.H; dim = model.dim
    if E_exact is None:
        E_exact, _ = model.exact_ground()
    V = [np.zeros(dim)]
    V[0][model.index[int(seed_mask)]] = 1.0
    sel_fock = {model.index[int(seed_mask)]}
    records = []

    while True:
        n = len(V)
        # ---- Ritz energy of the CURRENT n-determinant space ----
        t = time.perf_counter()
        Q = _orthonormal(V)
        Hm = Q.T @ H @ Q
        ev, wv = np.linalg.eigh(Hm)
        E = float(ev[0]); psi = Q @ wv[:, 0]
        t_ritz = time.perf_counter() - t

        t = time.perf_counter()
        r_full = H @ psi - E * psi
        rnorm = float(np.linalg.norm(r_full)); r = r_full / (rnorm + 1e-30)
        t_res = time.perf_counter() - t

        kkt_err = proj_res = gauge = np.nan
        t_pool = t_opt = 0.0
        order = None  # ranked atom indices to try

        if n < budget:
            t = time.perf_counter()
            fock_cols = [j for j in range(dim) if j not in sel_fock]
            Af = np.zeros((dim, len(fock_cols)))
            for jj, j in enumerate(fock_cols):
                Af[j, jj] = 1.0
            dom_mask = int(model.states[int(np.argmax(np.abs(psi)))])
            An = noci.sample_dictionary(model, dom_mask, n_atoms, eps,
                                        seed=n * 17 + 3)
            A = np.hstack([Af, An])
            nf = Af.shape[1]
            pool_size = A.shape[1]
            t_pool = time.perf_counter() - t

            t = time.perf_counter()
            Hdiag = np.einsum('ij,ij->j', A, H @ A)   # <a|H|a> for every atom
            if method == "cipsi":
                u = H @ psi
                sc = np.zeros(A.shape[1])
                for jj in range(nf):
                    d = fock_cols[jj]
                    den = Hdiag[jj] - E
                    sc[jj] = (u[d] ** 2) / (abs(den) + 1e-12) * (1 if den > 0 else 0)
            elif method == "greedy_noci":
                sc = np.array([-noci.two_dim_ritz_drop(H, psi, E, A[:, a])
                               for a in range(A.shape[1])])
            else:  # replica variants
                out = noci.convex_j_synthesize(A, r, k=k)
                cw = out["c"]
                kkt_err = out["kkt_err"]; proj_res = out["proj_residual"]
                gauge = out["gauge"]
                if method == "replica_coef":           # ablation
                    sc = np.abs(cw)
                else:                                   # replica_energy
                    sc = (np.abs(cw) ** 2) / (np.abs(Hdiag - E) + EPS_DEN)
            order = list(np.argsort(sc)[::-1])
            t_opt = time.perf_counter() - t
        else:
            nf = 0; pool_size = 0; A = np.zeros((dim, 0))

        # record THIS n-determinant Ritz state (before appending)
        records.append(CRec(n, E, E - E_exact, rnorm, kkt_err, proj_res,
                            gauge, t_pool, t_res, t_opt, t_ritz, pool_size,
                            False))
        if n >= budget:
            break

        # ---- pick the first linearly-independent atom in score order ----
        picked_nonorth = False
        for pick in order:
            atom = A[:, pick] / np.linalg.norm(A[:, pick])
            proj = Q @ (Q.T @ atom)
            resid = float(np.linalg.norm(atom - proj))
            if resid < 1e-10:
                continue                       # already in span(Q)
            V.append(atom)
            picked_nonorth = bool(pick >= nf)
            if not picked_nonorth:
                sel_fock.add(fock_cols[pick])
            break
    return records
