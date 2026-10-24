#!/usr/bin/env python3
"""Independent checks of saved observations, kets, and frozen source hashes.

This verifier does not certify Monte Carlo integration error or promote a
failed reconstruction gate. A passing artifact check and a passing physical
analysis gate are intentionally different outcomes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4 import paths
from fk4.io import write_json
from fk4.reference import log_dimension, magic_log_z
from fk3.discovery import covariance_from_kets
from fk4.symmetry import time_reverse
from fk4.distributions import full_distribution
from fk.model import build_model
from fk4.radial import target_overlap,coefficient_mass


def close(a, b, *, atol=1e-11, rtol=1e-10):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    args = p.parse_args()
    run = args.run
    manifest = json.loads((run/"manifest.json").read_text())
    for name, digest in manifest["source_sha256"].items():
        source = run/"source_snapshot"/name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, name
    summary = json.loads((run/"summary.json").read_text())
    assert summary["status"] == "complete", "partial runs cannot pass final artifact verification"
    checked = 0
    if "comparisons" in summary:
        for row in summary["rows"]:
            arrays = np.load(run/f"{row['case']}_{row['seed']}.npz")
            u = arrays["reference_state"]
            n = row["n"]
            for o in row["observations"]:
                beta = o["beta"]
                key = f"beta{beta:g}"
                kets = arrays[key+"_sample_kets"]
                close(np.linalg.norm(kets, axis=1), 1., atol=1e-10)
                close(2*np.log(abs(kets.conj()@u)), arrays[key+"_log_r"][:len(kets)])
                cov = covariance_from_kets(kets[:64], n=n)
                assert np.max(abs(cov@cov+np.eye(2*n))) < 1e-9
                exact = magic_log_z(beta) if row["case"] == "magic4" else -log_dimension(n, beta)
                close(o["relative_Z_error"], np.expm1(o["log_z"]-exact))
                if "q" in o:
                    close(o["q"], np.exp(log_dimension(n, int(beta))+o["log_z"]), atol=1e-100)
                    factor = np.exp(log_dimension(n, int(beta))+o["log_z"]-o["reference"]["log_q"])
                    close(o["raw_reconstruction_residual"], np.linalg.norm(factor*arrays[key+"_inverse_moment"]-u))
                    close(o["stabilizer_averaged_reconstruction_residual"], abs(factor-1))
                checked += 1
        scientific_pass = all(r["one_percent_empirical_gate"] and r["all_purity_pass"] for r in summary["comparisons"])
    else:
        model = np.load(run/"model.npz")
        S = model["S"]
        clifford = build_model().j_matrices if any(r.get("all_particle_distribution_saved") for r in summary["rows"]) else None
        close(S.conj().T@S, np.eye(3))
        for row in summary["rows"]:
            key = f"t{row['t']:g}_k{row['k']}"
            frozen = np.load(run/f"{key}_frozen.npz")
            u, target = frozen["witness"], frozen["target"]
            assert hashlib.sha256(u.tobytes()).hexdigest() == row["witness_sha256"]
            close(S@frozen["z"], u)
            Cs = []
            for o in row["validation_rows"]:
                a = np.load(run/f"{key}_validation_seed{o['seed']}.npz")
                particles=a
                if 'P_source_file' in o:
                    source_file=Path(o['P_source_file'])
                    assert hashlib.sha256(source_file.read_bytes()).hexdigest()==o['P_source_sha256']
                    particles=np.load(source_file)
                    close(a['C_fixed_numeric_u'],particles['C'])
                    close(a['C_RB_fixed_numeric_u'],particles['C_RB'])
                if row.get('radial_elimination',False):
                    factor=target_overlap(u,target)/o['q']
                    close(a['C'],factor*a['C_fixed_numeric_u'])
                    close(a['C_RB'],factor*a['C_RB_fixed_numeric_u'])
                    close(o['B'],coefficient_mass(u,target,particles['P_all_kets']))
                close(a["witness"], u)
                reference = S@(S.conj().T@a["C"])
                if row.get("time_reversal_averaged", False):
                    close(time_reverse(u), u)
                    reference = .5*(reference+time_reverse(reference))
                close(a["C_RB"], reference)
                close(o["rho_RB"], np.linalg.norm(a["C_RB"]-target))
                close(o["rho_raw"], np.linalg.norm(a["C"]-target))
                close(o["L_point"], np.vdot(u, target).real/o["q"]**(1/(2*row["k"])))
                close(covariance_from_kets(particles["P_kets"]), particles["P_covariances"], atol=1e-10)
                if row.get("all_particle_distribution_saved"):
                    features, feature_report = full_distribution(particles["P_all_kets"], clifford)
                    for name in ("features", "purity_errors", "mass_a_eta_counts", "bin_edges"):
                        close(particles["P_all_"+name], features[name], atol=1e-10)
                    close(o["P_full_distribution"]["feature_means"], feature_report["feature_means"])
                    close(np.linalg.norm(particles["P_all_kets"], axis=1), 1., atol=1e-10)
                assert not o["L_point_is_certified_lower"]
                assert not o["canonical_B_is_certified_upper"]
                Cs.append(a["C_RB"])
                checked += 1
            close(row["rho_RB_pooled"], np.linalg.norm(np.mean(Cs, axis=0)-target))
            assert row["reconstruction_pass"] == (row["rho_RB_pooled"] < 1e-3)
            assert row["all_seeds_reconstruction_pass"] == all(o["rho_RB"] < 1e-3 for o in row["validation_rows"])
        scientific_pass = summary["next_order_gate_pass"]
    result = {"artifact_verification_passed": True, "saved_observations_checked": checked,
              "frozen_sources_checked": len(manifest["source_sha256"]),
              "empirical_scientific_gate_passed": scientific_pass,
              "integration_error_certified": False, "full_objective_complete": False}
    write_json(run/"verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
