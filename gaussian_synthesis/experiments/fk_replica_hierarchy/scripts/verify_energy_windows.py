#!/usr/bin/env python3
"""Verify saved decimal coefficients directly, without recomputing their solver."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mpmath as mp
from fk4.io import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    args = p.parse_args()
    run = args.run
    manifest = json.loads((run/"manifest.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((run/"source_snapshot"/name).read_bytes()).hexdigest() == expected
    data = json.loads((run/"summary.json").read_text())
    assert data["status"] == "complete"
    mp.mp.dps = 100
    worst = mp.mpf(0)
    for r in data["rows"]:
        c = [mp.mpc(*v) for v in r["coefficient_decimal_pairs"]]
        q = [mp.mpc(*v) for v in r["target_hopping_coordinates_decimal_pairs"]]
        x = [mp.mpf(v) for v in r["x_decimal"]]
        reconstructed = [sum(cj*(1-y/2)**2 for cj, y in zip(c, x)),
                         sum(cj*3*(2*y-y*y)/(2*mp.sqrt(14)) for cj, y in zip(c, x)),
                         sum(cj*y*y/4 for cj, y in zip(c, x))]
        err = mp.sqrt(sum(abs(a-b)**2 for a, b in zip(reconstructed, q)))
        assert err <= mp.mpf(r["error_allowance"])+mp.mpf("1e-60")
        if r["error_allowance"] == 0:
            worst = max(worst, err)
        for y in x:
            assert y >= 0
            assert 18*y*y+4*(mp.mpf(r["t"])-9)*y < mp.mpf(r["delta"])
        cost = float(sum(abs(v) for v in c))
        assert abs(cost-r["upper_cost"]) < 1e-12*max(1., cost)
        assert r["lower_cost"] <= r["upper_cost"]
        assert not r["uses_out_of_window_fock_repair"]
    result = {"artifact_verification_passed": True, "constructions_checked": len(data["rows"]),
              "largest_zero_allowance_reconstruction_error": float(worst),
              "all_saved_gaussian_orbit_parameters_inside_window": True,
              "checked_explicit_decimal_coefficients_not_solver_output_alone": True,
              "strict_interval_certified": False, "complete_E3": False}
    write_json(run/"verification.json", result)
    print(result)


if __name__ == "__main__":
    main()
