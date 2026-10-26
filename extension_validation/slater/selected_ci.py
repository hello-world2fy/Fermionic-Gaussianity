"""Corrected selected-CI loop. Methods, all adding k_new Fock determinants to
the SAME orthonormal selected space at matched determinant count:

Fock pool (<=2 excitations from current space):
  * ``fock_greedy``  rank by |<d|r>| (myopic residual).
  * ``pt2``          CIPSI/PT2: |<d|H|psi>|^2/(H_dd-E).

Non-orthogonal Slater dictionary pool (M nearby rotated determinants around the
dominant Ritz determinant) -- SAME pool for both:
  * ``noci_greedy``  strong greedy: pick the atom with the largest 2x2 Ritz
                     energy drop in span{psi_s, atom}, harvest its Fock weights.
  * ``replica``      the actual convex-J synthesis: optimize the dual
                     max <w,r>/N_k(w) over the non-orthogonal dictionary, get
                     KKT atom weights c_a = t|z_a|^{2k-2} z_a and the Fock-space
                     reconstruction r_rec; report KKT reconstruction error, gauge
                     tau=sum|c_a|; harvest external Fock determinants by |r_rec[d]|
                     re-ranked by the PT2 energy denominator.

``power_weighted_heuristic`` is the OLD orthonormal power rule kept only as an
ablation; it is NOT a replica result.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from hubbard import HubbardModel
import noci


@dataclass
class StepRecord:
    n_det: int
    E: float
    dE: float
    residual_norm: float
    overlap: float
    double_occ: float
    t_pool: float
    t_residual: float
    t_optimize: float
    t_ritz: float
    kkt_err: float = np.nan
    gauge: float = np.nan
    proj_residual: float = np.nan
    opt_iters: float = np.nan
    pool_size: int = 0


def _external_fock(model, sel):
    sel_set = set(int(i) for i in sel)
    cand = set()
    for si in sel:
        mask = int(model.states[si])
        occ = [b for b in range(2 * model.L) if (mask >> b) & 1]
        unocc = [b for b in range(2 * model.L) if not ((mask >> b) & 1)]
        for a in occ:
            for b in unocc:
                if (a < model.L) != (b < model.L):
                    continue
                m2 = (mask & ~(1 << a)) | (1 << b)
                j = model.index.get(m2)
                if j is not None and j not in sel_set:
                    cand.add(j)
        for a1, a2 in __import__("itertools").combinations(occ, 2):
            for b1, b2 in __import__("itertools").combinations(unocc, 2):
                if (a1 < model.L) != (b1 < model.L) or (a2 < model.L) != (b2 < model.L):
                    continue
                m2 = (mask & ~(1 << a1)) | (1 << b1)
                m2 = (m2 & ~(1 << a2)) | (1 << b2)
                j = model.index.get(m2)
                if j is not None and j not in sel_set:
                    cand.add(j)
    return np.array(sorted(cand), dtype=np.int64)


def _harvest(r_rec, cand, psi, Hdiag, E, k_new):
    """Pick k_new external Fock determinants by |r_rec[d]| re-ranked by PT2."""
    u = H[cand] @ psi if False else None  # placeholder
    return None


def run_selected_ci(model: HubbardModel, seed_mask: int, method: str,
                    k_new: int = 2, max_det: int = None, k_rep: float = 2.0,
                    E_exact: float = None, psi_exact: np.ndarray = None,
                    n_atoms: int = 60, dict_eps: float = 0.6,
                    dict_seed: int = 0) -> dict:
    H = model.H
    dim = model.dim
    if E_exact is None or psi_exact is None:
        E_exact, psi_exact = model.exact_ground()
    if max_det is None:
        max_det = dim
    sel = np.array([model.index[int(seed_mask)]], dtype=np.int64)
    records = []

    while len(sel) < max_det:
        # --- Ritz ---
        t0 = time.perf_counter()
        Hsub = H[np.ix_(sel, sel)]
        evals, evecs = np.linalg.eigh(Hsub)
        E = evals[0]; c = evecs[:, 0]
        psi = np.zeros(dim); psi[sel] = c
        t_ritz = time.perf_counter() - t0

        # --- residual ---
        t1 = time.perf_counter()
        r_full = H @ psi - E * psi
        rnorm = float(np.linalg.norm(r_full))
        ov = float(abs(np.vdot(psi, psi_exact)))
        dbl = 0.0
        for idx, wgt in zip(sel, c ** 2):
            mask = int(model.states[idx])
            for i in range(model.L):
                if (mask >> i) & 1 and (mask >> (model.L + i)) & 1:
                    dbl += wgt
        t_res = time.perf_counter() - t1

        kkt_err = gauge = proj_res = opt_it = np.nan
        t_pool = t_opt = 0.0

        if method in ("fock_greedy", "pt2", "power_weighted_heuristic"):
            t1 = time.perf_counter()
            cand = _external_fock(model, sel)
            t_pool = time.perf_counter() - t1
            if cand.size == 0:
                records.append(StepRecord(len(sel), E, E - E_exact, rnorm, ov,
                                          dbl, t_pool, t_res, 0.0, t_ritz,
                                          pool_size=0))
                break
            u = H[cand] @ psi
            Hdiag = np.diag(H)[cand]
            denom = np.clip(Hdiag - E, 1e-12, None)
            if method == "fock_greedy":
                scores = np.abs(u)
            elif method == "pt2":
                scores = u ** 2 / denom
            else:  # power_weighted_heuristic (old, orthonormal)
                scores = np.abs(u) ** (2 * k_rep)
            t2 = time.perf_counter()
            order = np.argsort(scores)[::-1][:min(k_new, cand.size)]
            t_opt = time.perf_counter() - t2
            picked = cand[order]
            pool_size = int(cand.size)

        elif method in ("noci_greedy", "replica"):
            t1 = time.perf_counter()
            dom = int(sel[int(np.argmax(np.abs(c)))])  # dominant Ritz determinant
            dom_mask = int(model.states[dom])
            A = noci.sample_dictionary(model, dom_mask, n_atoms, dict_eps,
                                       seed=dict_seed + len(sel))
            cand = _external_fock(model, sel)
            pool_size = int(cand.size)
            t_pool = time.perf_counter() - t1

            t2 = time.perf_counter()
            r = r_full / (rnorm + 1e-30)
            if method == "noci_greedy":
                best_drop = None; best_atom_vec = None
                for ai in range(A.shape[1]):
                    e2 = noci.two_dim_ritz_drop(H, psi, E, A[:, ai])
                    if best_drop is None or e2 < best_drop:
                        best_drop = e2; best_atom_vec = A[:, ai]
                r_rec = best_atom_vec
                opt_it = A.shape[1]
            else:  # replica: real convex-J optimization
                out = noci.convex_j_synthesize(A, r, k=k_rep)
                r_rec = out["synth"]
                kkt_err = out["kkt_err"]; gauge = out["gauge"]
                proj_res = out["proj_residual"]; opt_it = out["iters"]
            t_opt = time.perf_counter() - t2

            # harvest external Fock determinants from reconstruction weights,
            # re-ranked by PT2 energy denominator
            u = H[cand] @ psi
            Hdiag = np.diag(H)[cand]
            denom = np.clip(Hdiag - E, 1e-12, None)
            harvest = np.abs(r_rec[cand])
            scores = harvest * (u ** 2 / denom)
            order = np.argsort(scores)[::-1][:min(k_new, cand.size)]
            picked = cand[order]
        else:
            raise ValueError(method)

        records.append(StepRecord(len(sel), E, E - E_exact, rnorm, ov, dbl,
                                  t_pool, t_res, t_opt, t_ritz, kkt_err, gauge,
                                  proj_res, opt_it, pool_size))
        sel = np.unique(np.concatenate([sel, picked]))
    return dict(method=method, records=records, E_exact=E_exact)
