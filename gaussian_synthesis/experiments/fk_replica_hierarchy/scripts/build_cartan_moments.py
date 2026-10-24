#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4 import paths
from fk4.io import write_json, source_manifest
from fk4.cartan_sparse import casimir_defect, replica_frame, chebyshev_project, CartanMoments, independent_full_reconstruction, CAYLEY_MASKS, CAYLEY_SIGNS, EVEN_MASKS
from fk4.energy_windows import hopping_eigenbasis
from fk4.solver import solve
from fk.model import build_model
from fk2.exact_moments import ExactK2Moments


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, required=True, choices=[2, 3, 4])
    p.add_argument("--degree", type=int, default=100)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"manifest.json", source_manifest(out/"source_snapshot"))
    write_json(out/"config.json", {"k": args.k, "degree": args.degree})
    started = time.monotonic()
    A, codes, tuples, report = casimir_defect(args.k)
    print('matrix', report, 'seconds', time.monotonic()-started, flush=True)
    write_json(out/"construction.json", report)
    V, rows, powers, frame_error = replica_frame(args.k, tuples)
    projected, diagnostics = chebyshev_project(A, V, minimum=report["theoretical_nonzero_defect_lower"],
                                                maximum=report["defect_max"], degree=args.degree,
                                                callback=lambda m: print('degree', m, 'seconds', time.monotonic()-started, flush=True))
    raw = V.T@projected
    matrix = .5*(raw+raw.T)
    projected_rows = projected[rows]
    np.savez_compressed(out/"moment_matrix.npz", matrix=matrix, powers=powers,
                        support_tuples=tuples[rows], projected_support_rows=projected_rows,
                        frame_nonzero_values=V[rows].toarray())
    backend = CartanMoments(args.k, matrix)
    model = build_model()
    summary = {"status": "running", "construction": report, "projection": diagnostics,
               "frame_orthogonality_error": frame_error,
               "moment_symmetry_error": float(np.linalg.norm(raw-raw.T)),
               "moment_eigenvalues": np.linalg.eigvalsh(matrix), "rows": [],
               "uses_monte_carlo": False, "strict_interval_certified": False}
    # Independently construct the Gaussian Fock basis from physical Majoranas.
    B = np.zeros((256, 256), complex)
    B[:, 0] = model.embed(model.S@hopping_eigenbasis()[:, 0])
    for mask in range(1, 256):
        bit = mask & -mask
        j = bit.bit_length()-1
        previous = B[:, mask^bit]
        B[:, mask] = (model.gammas[j]@previous-1j*model.gammas[j+8]@previous)/2
    U = B[np.ix_(model.even_masks, model.even_masks)]
    exact_basis = np.zeros((128, 3))
    exact_basis[0, 0], exact_basis[-1, 2] = 1, -1
    exact_basis[np.searchsorted(EVEN_MASKS, CAYLEY_MASKS), 1] = CAYLEY_SIGNS/np.sqrt(14)
    summary["hopping_fock_unitarity_error"] = float(np.linalg.norm(U.conj().T@U-np.eye(128)))
    summary["cayley_basis_match_error"] = float(np.linalg.norm(U.conj().T@model.S@hopping_eigenbasis()-exact_basis))
    if args.k == 2:
        old = ExactK2Moments(model)
        rng = np.random.default_rng(2901)
        summary["independent_k2_errors"] = []
        for _ in range(12):
            z = rng.normal(size=3)+1j*rng.normal(size=3)
            z /= np.linalg.norm(z)
            a, b = backend.evaluate(z), old.evaluate(z)
            summary["independent_k2_errors"].append(max(abs(a.q-b.q), np.linalg.norm(a.gradient-b.gradient), np.linalg.norm(a.hessian-b.hessian)))
        if max(summary["independent_k2_errors"]) > 1e-10:
            write_json(out/"summary.json", summary)
            raise ArithmeticError("independent exact order2 comparison failed")
    for t in [0, 5, 9, 9.1, 10, 10.9225, 11, 12, 20]:
        target = model.target(t)
        target_z = model.S.conj().T@target
        solution = solve(backend, target_z)
        z = solution["z"]
        Cnew = independent_full_reconstruction(args.k, z, powers, tuples[rows], projected_rows)
        fullC = U@Cnew
        residual = float(np.linalg.norm(fullC-target))
        key = f"t{t:g}_k{args.k}"
        np.savez_compressed(out/f"{key}_frozen.npz", z=z, witness=model.S@z, target=target,
                            target_z=target_z, full_reconstruction=fullC)
        row = {"t": t, "k": args.k, "q": backend.evaluate(z, hessian=False).q,
               "L_algebraic_point": float(np.vdot(z, target_z).real/backend.evaluate(z, hessian=False).q**(1/(2*args.k))),
               "full_reconstruction_residual": residual, "optimizer_gradient_norm": solution["gradient_norm"],
               "optimizer_converged": solution["converged"], "B": None,
               "B_status": "half_integer_mass_requires_independent_integration"}
        summary["rows"].append(row)
        write_json(out/"summary.json", summary)
        print(row, flush=True)
    summary.update(status="complete", elapsed_seconds=time.monotonic()-started)
    write_json(out/"summary.json", summary)


if __name__ == "__main__":
    main()
