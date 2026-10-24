#!/usr/bin/env python3
"""Add a separate order-two interval certificate to frozen high-order vectors."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4 import paths
from fk4.io import write_json, source_manifest
from fk4.certificates import certify_order2_normalization
from fk2.exact_moments import integer_fk_matrix_proof


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/"manifest.json", source_manifest(args.output/"source_snapshot"))
    proof = integer_fk_matrix_proof()
    if not proof["verified"]:
        raise ArithmeticError("signed-integer projector proof failed")
    summary = json.loads((args.run/"summary.json").read_text())
    rows = []
    for r in summary["rows"]:
        key = f"t{r['t']:g}_k{r['k']}"
        data = np.load(args.run/f"{key}_frozen.npz")
        certified = certify_order2_normalization(data["z"], data["target"])
        z = certified.pop("z_feasible")
        np.savez_compressed(args.output/f"{key}_certificate.npz", z_feasible=z, target=data["target"])
        certified.update(t=r["t"], source_order=r["k"], source_key=key,
                         source_witness_sha256=r["witness_sha256"])
        rows.append(certified)
    write_json(args.output/"certificates.json", {"rows": rows, "integer_proof_verified": True,
                                                "source_run": str(args.run), "status": "complete"})
    print(json.dumps([{k:r[k] for k in ("t", "source_order", "L_certified_saved_target", "full_dictionary_feasible")} for r in rows]))


if __name__ == "__main__":
    main()
