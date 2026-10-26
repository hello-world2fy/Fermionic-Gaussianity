"""Fixed-wall-clock AND fixed-detector-budget selected CI.

Fairness / hard deadline
------------------------
* Models, H matrices and E_score (FCI energy) are preloaded OUTSIDE the clock.
  Start vectors are also precomputed outside the clock (their normalization is
  timed); FCI eigenvectors are NEVER passed to the worker -- only E_score.
* t0 = perf_counter() AFTER all preload. Method-specific prepare (pool build,
  L-BFGS convex-J, orbital optimization, 2x2 Ritz) is inside the clock.
* A hard SIGALRM deadline interrupts scipy L-BFGS mid-optimization; the
  exception is caught and the loop rolls back to the last fully-committed Ritz
  state (timestamp <= deadline). Checkpoint = Ritz diagonalize AND residual
  both completed before the deadline.
* If even the initial Ritz is not finished by the deadline, the run is marked
  initialization_not_completed and NO energy is reported.
"""
from __future__ import annotations

import csv
import signal
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from hubbard import HubbardModel
import noci
import molecule

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TIME_BUDGETS = [0.3, 0.8, 1.5]
METHODS = ["cipsi", "greedy_noci", "orbopt_noci", "replica_energy"]


class _Deadline(Exception):
    pass


def _alarm(signum, frame):
    raise _Deadline()


@dataclass
class TRun:
    system: str
    mode: str
    method: str
    seed: int
    budget: str
    clock_start: float
    deadline: float
    actual_stop: float
    last_committed_time: float
    elapsed: float
    overrun: float
    timeout_stage: str
    rollback: bool
    initialization_not_completed: bool
    completed_n: int
    energy: float
    dE: float
    residual: float
    start_fci_overlap: float
    checkpoint: str
    run_version: str = "ritzfix_v2"


def _starts(model, seed, E_fci_vec=None):
    dim = model.dim
    if getattr(model, "is_h2", False):
        mask = model.neel_determinant()
        Uup0, Udn0 = noci.reference_U(model, mask)
        if seed == 0:
            v = noci.slater_fock_vector(model, Uup0, Udn0)
        elif seed == 1:
            v = noci.rotated_slater(model, Uup0, Udn0, np.array([[0.6]]), np.array([[0.2]]))
        else:
            v = noci.rotated_slater(model, Uup0, Udn0, np.array([[-0.4]]), np.array([[0.8]]))
        ov = abs(float(np.vdot(E_fci_vec, v))) if E_fci_vec is not None else float("nan")
        return v, ov
    masks = [model.neel_determinant()]
    m = 0
    for i in range(model.L):
        m |= (1 << (i if i % 2 else model.L + i))
    masks.append(m)
    rng = np.random.default_rng(100 + seed)
    up = rng.choice(model.L, model.n_up, replace=False)
    dn = rng.choice(model.L, model.n_down, replace=False)
    mm = 0
    for i in up: mm |= 1 << int(i)
    for i in dn: mm |= 1 << (model.L + int(i))
    masks.append(mm)
    v = np.zeros(dim); v[model.index[int(masks[seed % 3])]] = 1.0
    ov = abs(float(np.vdot(E_fci_vec, v))) if E_fci_vec is not None else float("nan")
    return v, ov


def _save_ckpt(system, mode, method, seed, budget_lbl, model, psi, E, n, stage,
               E_hist=None, n_hist=None, t_hist=None, r_hist=None):
    d = RES / "ckpt"
    d.mkdir(exist_ok=True)
    fn = d / f"{system}_{mode}_{method}_s{seed}_{budget_lbl}.npz"
    np.savez(fn, H=model.H, psi=psi, E=E, n=n, stage=stage,
             residual=float(np.linalg.norm(model.H @ psi - E * psi)),
             E_hist=np.array(E_hist or []), n_hist=np.array(n_hist or []),
             t_hist=np.array(t_hist or []), r_hist=np.array(r_hist or []))
    return fn.name


def _run_once(model, E_score, method, start_vec, start_overlap,
              deadline, mode, budget_n, budget_lbl, k=2.0, n_atoms=40, eps=0.6, seed=0):
    H = model.H; dim = model.dim
    t0 = time.perf_counter()
    V = [start_vec / np.linalg.norm(start_vec)]   # normalization timed
    sel_fock = set(np.where(np.abs(V[0]) > 0.99)[0])
    committed = []   # (n, E, resid, t, psi.copy())
    timeout_stage = "completed"
    init_done = False
    psi = np.zeros(dim); E = float("nan")
    try:
        while True:
            now = time.perf_counter()
            if now > deadline:
                timeout_stage = "pre_ritz" if not init_done else "post_committed"
                break
            Q, _ = np.linalg.qr(np.array(V).T)
            ev, wv = np.linalg.eigh(Q.T @ H @ Q)
            E = float(ev[0]); psi = Q @ wv[:, 0]
            r_full = H @ psi - E * psi
            resid = float(np.linalg.norm(r_full))
            t_commit = time.perf_counter()
            if t_commit > deadline:
                timeout_stage = "ritz_residual" if not init_done else "post_committed"
                break
            committed.append((len(V), E, resid, t_commit, psi.copy()))
            # variational guard: Ritz energy must be non-increasing
            if len(committed) >= 2 and E > committed[-2][1] + 1e-9:
                timeout_stage = "variolation_violation"; break
            init_done = True
            if len(V) >= dim or resid < 1e-9:
                timeout_stage = "full_space" if len(V) >= dim else "residual_zero"
                break
            if mode == "fixed_n" and len(V) >= budget_n:
                timeout_stage = "budget_reached"; break

            r = r_full / (resid + 1e-30)
            if method == "orbopt_noci":
                dom = int(np.argmax(np.abs(psi)))
                atom, _ = noci.orbital_optimize(
                    model, H, psi, E, int(model.states[dom]),
                    deadline=deadline, seed=seed + len(V))
                if atom is None or np.linalg.norm(atom - Q @ (Q.T @ atom)) < 1e-10:
                    timeout_stage = "orbopt_none"; break
                V.append(atom / np.linalg.norm(atom)); continue

            fock_cols = [j for j in range(dim) if j not in sel_fock]
            Af = np.zeros((dim, len(fock_cols)))
            for jj, j in enumerate(fock_cols):
                Af[j, jj] = 1.0
            dom = int(np.argmax(np.abs(psi)))
            An = noci.sample_dictionary(model, int(model.states[dom]), n_atoms,
                                        eps, seed=len(V) * 17 + seed + 3)
            A = np.hstack([Af, An]); nf = Af.shape[1]
            Hdiag = np.einsum("ij,ij->j", A, H @ A)
            if method == "cipsi":
                u = H @ psi
                sc = np.zeros(A.shape[1])
                for jj in range(nf):
                    d = fock_cols[jj]; den = Hdiag[jj] - E
                    sc[jj] = (u[d] ** 2) / (abs(den) + 1e-12) * (den > 0)
            elif method == "greedy_noci":
                sc = np.array([-noci.two_dim_ritz_drop(H, psi, E, A[:, a])
                               for a in range(A.shape[1])])
            else:
                out = noci.convex_j_synthesize(A, r, k=k)
                cw = out["c"]
                sc = (np.abs(cw) ** 2) / (np.abs(Hdiag - E) + 1e-4)
            picked = False
            for pick in np.argsort(sc)[::-1]:
                atom = A[:, pick] / np.linalg.norm(A[:, pick])
                if np.linalg.norm(atom - Q @ (Q.T @ atom)) < 1e-10:
                    continue
                V.append(atom)
                if pick < nf:
                    sel_fock.add(fock_cols[int(pick)])
                picked = True; break
            if not picked:
                timeout_stage = "no_independent"; break
    except _Deadline:
        timeout_stage = "hard_deadline"
    except Exception:
        timeout_stage = "compute_error"
    actual_stop = time.perf_counter()   # stop clock BEFORE any file IO
    signal.setitimer(signal.ITIMER_REAL, 0)   # never let npz IO raise Deadline
    if not init_done:
        return TRun(getattr(model, "tag", "?"), mode, method, seed, budget_lbl,
                    t0, deadline, actual_stop, t0, actual_stop - t0,
                    actual_stop - deadline, timeout_stage, True, True,
                    0, float("nan"), float("nan"), float("nan"), start_overlap, "")
    n_last, E_last, res_last, t_last, psi_last = committed[-1]
    E_hist = [c[1] for c in committed]; n_hist = [c[0] for c in committed]
    t_hist = [c[3] for c in committed]; r_hist = [c[2] for c in committed]
    ck = ""
    try:
        ck = _save_ckpt(getattr(model, "tag", "?"), mode, method, seed,
                        budget_lbl, model, psi_last, E_last, n_last, timeout_stage,
                        E_hist, n_hist, t_hist, r_hist)
    except Exception:
        ck = ""
    return TRun(getattr(model, "tag", "?"), mode, method, seed, budget_lbl,
                t0, deadline, actual_stop, t_last, actual_stop - t0,
                max(0.0, actual_stop - deadline), timeout_stage,
                len(V) > n_last, False, n_last, E_last, E_last - E_score,
                res_last, start_overlap, ck)


def timed_run(model, E_score, method, seed, budget, start_vec, start_overlap):
    lbl = f"{budget:g}s"
    signal.signal(signal.SIGALRM, _alarm)
    t0 = time.perf_counter()
    signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        rec = _run_once(model, E_score, method, start_vec, start_overlap,
                        t0 + budget, "timed", 0, lbl, seed=seed)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return rec


def fixed_n_run(model, E_score, method, seed, n_budget, start_vec, start_overlap):
    t0 = time.perf_counter()
    rec = _run_once(model, E_score, method, start_vec, start_overlap,
                    t0 + 60.0, "fixed_n", n_budget, f"n{n_budget}", seed=seed)
    return rec


def main():
    systems = []
    for L, U in [(4, 1.0), (4, 8.0), (6, 1.0), (6, 8.0)]:
        m = HubbardModel(L=L, t=1.0, U=U)
        E, vec = m.exact_ground()
        m.tag = f"HUBBARD_L{L}_U{int(U)}"; m.is_h2 = False
        systems.append((m, float(E), vec))
    for R in [0.74, 1.5, 2.0]:
        ham, fci_e, rhf_e, mf = molecule.h2_active(R)
        E, vec = ham.exact_ground()
        ham.tag = f"H2_R{R}"; ham.is_h2 = True
        systems.append((ham, fci_e - mf.energy_nuc(), vec))

    rows = []
    for model, E_score, E_vec in systems:
        starts = [_starts(model, s, E_vec) for s in [0, 1, 2]]
        is_h2 = getattr(model, "is_h2", False)
        n_budgets = [2, 3, 4] if is_h2 else [8, 12]
        # fixed-n runs
        for method in METHODS:
            for seed in [0, 1, 2]:
                sv, sov = starts[seed]
                for nb in n_budgets:
                    rec = fixed_n_run(model, E_score, method, seed, nb, sv, sov)
                    rows.append(asdict(rec))
        # timed runs (H2 saturates fast; keep short budgets)
        for method in METHODS:
            for seed in [0, 1, 2]:
                sv, sov = starts[seed]
                for b in TIME_BUDGETS:
                    rec = timed_run(model, E_score, method, seed, b, sv, sov)
                    rows.append(asdict(rec))
                    print(f"[{rec.system:14s}] {method:13s} s{seed} b={b}s "
                          f"n={rec.completed_n:3d} dE={rec.dE:.3e} "
                          f"stop={rec.timeout_stage} overrun={rec.overrun:.3f}")
    with open(RES / "timed_bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[done] {len(rows)} rows -> results/timed_bench.csv")


if __name__ == "__main__":
    main()
