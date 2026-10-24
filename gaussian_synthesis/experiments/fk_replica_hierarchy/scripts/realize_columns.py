#!/usr/bin/env python3
"""Reify previously saved Gaussian orbit coefficients without reoptimizing."""
from pathlib import Path
import argparse
import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4 import paths
from fk4.io import write_json, source_manifest
from fk4.columns import finite_group_expansion
from fk4.finite_twirl import FiniteSpin7Twirl
from fk.model import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"manifest.json", source_manifest(out/"source_snapshot"))
    source = json.loads((args.source/"summary.json").read_text())
    model = build_model()
    twirl = FiniteSpin7Twirl(model)
    rows = []
    for row in source["prior_rows"]+source["rows"]:
        key = row["key"]
        data = np.load(args.source/f"{key}.npz")
        result, arrays = finite_group_expansion(model, data["atoms"], data["coefficients"], data["target"], twirl=twirl)
        np.savez_compressed(out/f"{key}.npz", **arrays, target=data["target"])
        result.update(t=row["t"], key=key, source_U_orbit=row["U_orbit_repaired"], source_run=str(args.source))
        rows.append(result)
        write_json(out/f"{key}.json", result)
        print(f"{key}: U={result['U_finite_repaired']:.9g}, actual branches={result['main_gaussian_branch_count']}, residual={result['raw_reconstruction_residual']:.3g}", flush=True)
    write_json(out/"summary.json", {"status": "complete", "rows": rows,
                                   "full_twirl_operator_error": twirl.operator_error})


if __name__ == "__main__":
    main()
