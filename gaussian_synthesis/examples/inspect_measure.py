#!/usr/bin/env python3
"""Load a saved continuous measure and query it, without optimization or sampling."""

import os
for _name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

import numpy as np
from gaussian_synthesis import FockBasis, GaussianState, GaussianCoefficientMeasure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=ROOT/"results"/"magic_state")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    summary = json.loads((args.result/"summary.json").read_text())
    basis = FockBasis(summary["basis"]["n"], summary["basis"]["parity"])
    stage = next((s for s in summary["stages"] if s["k"] == args.k), None)
    if stage is None:
        parser.error("Requested order is absent from saved results")
    with np.load(args.result/"vectors.npz", allow_pickle=False) as saved:
        measure = GaussianCoefficientMeasure(basis, args.k, saved[f"u_k{args.k}"])
    mass = stage["validation"]["mass_estimate"] if stage["validation"] else None
    rows = []
    for occupation in (0, 3, 15):
        atom = GaussianState(basis, basis.ket(occupation))
        aligned = measure.aligned_state(atom).to_fock()
        rows.append({
            "occupation_bitmask": occupation,
            "density_relative_to_mu": measure.density(atom),
            "estimated_probability_density_relative_to_mu": (
                measure.probability_density_estimate(atom, mass_estimate=mass) if mass else None),
            "aligned_amplitudes_real": aligned.real.tolist(),
            "aligned_amplitudes_imag": aligned.imag.tolist(),
        })
    print(json.dumps({"k": args.k, "reference_measure": measure.reference_measure,
        "note": "These are densities relative to continuous mu, not point probabilities.",
        "validation": stage["validation"], "queries": rows}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
