#!/usr/bin/env python3
"""Validate generic tempered integration against analytic independent references."""
from pathlib import Path
import argparse
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import t as student_t
from fk4 import paths
from fk4.io import write_json, source_manifest
from fk4.reference import log_dimension, magic_log_z, magic_reference, gaussian_reference
from fk4.tempering import sample_tempered
from fk3.discovery import covariance_from_kets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32768)
    parser.add_argument("--max-k", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[24101, 24102, 24103, 24104, 24105])
    parser.add_argument("--cases", nargs="+", default=["magic4", "gaussian4", "gaussian8"])
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--backend", choices=['cpu','cuda'], default='cpu')
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    os.environ['FK_SAMPLER_BACKEND'] = args.backend
    os.environ['FK_CUDA_DEVICE'] = str(args.device)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config["output"] = str(out)
    write_json(out/"config.json", config)
    write_json(out/"manifest.json", source_manifest(out/"source_snapshot"))
    orders = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= args.max_k]
    summary = {"status": "running", "rows": [], "integration_is_certified": False,
               "acceptance_is_empirical": True, "cases": args.cases, "orders": orders}
    started = time.monotonic()
    for case in args.cases:
        n = int(case[-1])
        u = np.zeros(2**(n-1), complex)
        if case == "magic4":
            u[[0, -1]] = 1/np.sqrt(2)
        elif case in ("gaussian4", "gaussian8"):
            u[0] = 1
        else:
            raise ValueError(case)
        for seed in args.seeds:
            row = {"case": case, "n": n, "seed": seed, "count": args.count, "observations": []}
            arrays = {"reference_state": u}
            def snapshot(kets, beta, log_z, log_r, diagnostic):
                key = f"beta{beta:g}"
                analytic_log_z = magic_log_z(beta) if case == "magic4" else -log_dimension(n, beta)
                cov = covariance_from_kets(kets[:64], n=n)
                purity = float(np.max(abs(cov@cov+np.eye(2*n))))
                overlap = kets.conj()@u
                inverse_moment = np.mean(kets/overlap.conj()[:, None], axis=0)
                arrays[key+"_inverse_moment"] = inverse_moment
                arrays[key+"_sample_kets"] = kets[:256]
                arrays[key+"_log_r"] = log_r.copy()
                observation = {"beta": beta, "log_z": log_z, "analytic_log_z": analytic_log_z,
                               "relative_Z_error": float(np.expm1(log_z-analytic_log_z)),
                               "overlap_squared_mean": float(np.exp(log_r).mean()),
                               "purity_error": purity, "elapsed_seconds": diagnostic["elapsed_seconds"]}
                if beta in orders:
                    k = int(beta)
                    ref = magic_reference(k) if case == "magic4" else gaussian_reference(n, k)
                    factor = np.exp(log_dimension(n, k)+log_z-ref["log_q"])
                    observation.update(k=k, q=float(np.exp(log_dimension(n, k)+log_z)),
                                       L=float(np.exp(-(log_dimension(n, k)+log_z)/(2*k))),
                                       reference=ref,
                                       raw_reconstruction_residual=float(np.linalg.norm(factor*inverse_moment-u)),
                                       stabilizer_averaged_reconstruction_residual=float(abs(factor-1)))
                elif beta+.5 in orders:
                    k = int(beta+.5)
                    ref = magic_reference(k) if case == "magic4" else gaussian_reference(n, k)
                    observation.update(k=k, B=float(np.exp(log_dimension(n, k)+log_z+(2*k-1)*np.log(ref["radius"]))),
                                       reference_B=ref["B"])
                    if case == "magic4":
                        observation.update(mass_squared_mean=float(np.mean(2*np.exp(log_r))),
                                           reference_mass_squared_mean=ref["mass_squared_mean"])
                row["observations"].append(observation)
                write_json(out/f"{case}_{seed}.json", row)
                np.savez_compressed(out/f"{case}_{seed}.npz", **arrays)
                print(f"{case} seed={seed} beta={beta:g} relative_Z_error={observation['relative_Z_error']:.5g} elapsed={diagnostic['elapsed_seconds']:.1f}s", flush=True)
            result = sample_tempered(u, max(orders), args.count, np.random.default_rng(seed),
                                     n=n, sweeps=args.sweeps,
                                     checkpoints=[b for k in orders for b in (k-.5, k)],
                                     state_callback=snapshot)
            row["diagnostics"] = result.diagnostics
            write_json(out/f"{case}_{seed}.json", row)
            summary["rows"].append(row)
            write_json(out/"summary.json", summary)
    comparisons = []
    for case in args.cases:
        for beta in sorted({b for k in orders for b in (k-.5, k)}):
            observations = [o for r in summary["rows"] if r["case"] == case
                            for o in r["observations"] if o["beta"] == beta]
            errors = np.array([o["relative_Z_error"] for o in observations])
            se = float(np.std(errors, ddof=1)/np.sqrt(len(errors))) if len(errors) > 1 else None
            halfwidth = float(student_t.ppf(.975, len(errors)-1)*se) if se is not None else None
            comparisons.append({"case": case, "beta": beta, "seeds": len(errors),
                                "mean_relative_error": float(errors.mean()), "repeat_standard_error": se,
                                "repeat_95_percent_halfwidth": halfwidth,
                                "interval_kind": "Student-t across independent SMC runs; not a rigorous coverage certificate",
                                "all_purity_pass": all(o["purity_error"] < 1e-9 for o in observations),
                                "one_percent_empirical_gate": len(errors) >= 5 and abs(errors.mean()) < .01 and halfwidth < .01,
                                "analytic_compatibility_gate": len(errors) >= 5 and abs(errors.mean()) <= max(.003, 4*se)})
    summary.update(status="complete", elapsed_seconds=time.monotonic()-started, comparisons=comparisons)
    write_json(out/"summary.json", summary)


if __name__ == "__main__":
    main()
