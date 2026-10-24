#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4 import paths
from fk4.io import write_json, source_manifest
from fk4.columns import column_generation, independent_arc_dictionary, fit_orbits, finite_group_expansion
from fk4.finite_twirl import FiniteSpin7Twirl
from fk.model import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--t", type=float, nargs="+", default=[0, 5, 9, 9.1, 10, 10.9225, 11, 12, 20])
    p.add_argument("--seeds", type=int, nargs="+", default=[25101, 25102, 25103, 25104, 25105])
    p.add_argument("--starts", type=int, default=32)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--sweeps", type=int, default=40)
    p.add_argument("--arc-count", type=int, default=129)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    write_json(out/"manifest.json", source_manifest(out/"source_snapshot"))
    model = build_model()
    twirl = FiniteSpin7Twirl(model)
    started = time.monotonic()
    arc, angles = independent_arc_dictionary(model, args.arc_count)
    np.savez_compressed(out/"model_and_prior.npz", S=model.S, arc_atoms=arc, arc_angles=angles,
                        even_masks=model.even_masks)
    summary = {"status": "running", "rows": [], "prior_rows": [],
               "pricing_oracle_is_local": True, "complete_gaussian_optimum_proved": False}
    for t in args.t:
        target = model.target(t)
        atoms = np.concatenate((np.eye(128, dtype=complex), arc))
        fit = fit_orbits(model.S, atoms, target)
        finite, arrays = finite_group_expansion(model, atoms, fit["coefficients"], target, twirl=twirl)
        key = f"prior_t{t:g}"
        np.savez_compressed(out/f"{key}.npz", atoms=atoms, coefficients=fit["coefficients"],
                            target=target, dual_z=fit["dual_z"], **arrays)
        row = {"t": t, "key": key, "method": "explicit_physics_prior",
               "U_orbit_repaired": fit["U_orbit_repaired"], **finite}
        summary["prior_rows"].append(row)
        write_json(out/f"{key}.json", row)
        for seed in args.seeds:
            report, arrays = column_generation(model, t, seed, starts=args.starts,
                                               iterations=args.iterations, sweeps=args.sweeps)
            finite, finite_arrays = finite_group_expansion(model, arrays["atoms"], arrays["coefficients"], target, twirl=twirl)
            report["finite_expansion"] = finite
            key = f"columns_t{t:g}_seed{seed}"
            report["key"] = key
            np.savez_compressed(out/f"{key}.npz", **arrays, **finite_arrays)
            write_json(out/f"{key}.json", report)
            summary["rows"].append(report)
            write_json(out/"summary.json", summary)
            print(f"t={t:g} seed={seed} U_orbit={report['U_orbit_repaired']:.9g} U_finite={finite['U_finite_repaired']:.9g} local_price={report['history'][-1]['local_pricing_max']:.7g}", flush=True)
    summary.update(status="complete", elapsed_seconds=time.monotonic()-started)
    write_json(out/"summary.json", summary)


if __name__ == "__main__":
    main()
