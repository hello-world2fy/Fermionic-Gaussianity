#!/usr/bin/env python3
"""Audit a saved FK run without generating samples or rerunning optimization.

Usage: .venv/bin/python scripts/verify_results.py OUTPUT_DIR
Writes OUTPUT_DIR/verification.json. Sampling precision failures are preserved
scientific outcomes, not audit failures; inconsistent values or claims fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from fk.exact import run_exact
from fk.model import build_model
from fk.runtime import source_hashes


def log_dimension(n, order):
    return sum(math.log1p(order/(2*n-i-j))
               for i in range(1,n+1) for j in range(i+1,n+1))


def log_frame(name, order):
    if name == "full":
        return log_dimension(8, order)
    if name == "product":
        return math.log(2)+2*log_dimension(4, order)
    raise ValueError(f"Unknown dictionary: {name}")


def mm(a, b):
    return np.einsum("ij,jk->ik", a, b, optimize=False)


def mv(a, b):
    return np.einsum("ij,j->i", a, b, optimize=False)


class Audit:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.checks = 0
        self.failures = []
        self.warnings = []
        self.counts = {}

    def check(self, label, condition, detail=None):
        self.checks += 1
        if not bool(condition):
            entry = {"check": label}
            if detail is not None:
                entry["detail"] = detail
            self.failures.append(entry)

    def close(self, label, observed, expected, atol=1e-9, rtol=1e-9):
        observed, expected = np.asarray(observed), np.asarray(expected)
        # Sample budgets can exceed int64 even though no such allocation is
        # attempted. Compare those Python integers as finite floating scalars.
        if observed.dtype.kind == "O":
            observed = observed.astype(float)
        if expected.dtype.kind == "O":
            expected = expected.astype(float)
        if observed.shape != expected.shape:
            self.check(label, False, {"observed_shape": list(observed.shape),
                                      "expected_shape": list(expected.shape)})
            return
        finite = np.all(np.isfinite(observed)) and np.all(np.isfinite(expected))
        passed = finite and np.allclose(observed, expected, atol=atol, rtol=rtol)
        error = float(np.max(np.abs(observed-expected))) if finite and observed.size else None
        self.check(label, passed, {"maximum_absolute_error": error,
                                  "atol": atol, "rtol": rtol})

    def read(self, filename):
        with (self.output/filename).open() as handle:
            return json.load(handle)

    def section(self, label, action):
        try:
            action()
        except Exception as exc:
            self.check(label+".exception", False,
                       {"error": str(exc), "traceback": traceback.format_exc()})


def audit_manifest(audit, manifest):
    stored = manifest["source_sha256"]
    current = source_hashes(ROOT)
    missing = sorted(set(stored)-set(current))
    added = sorted(set(current)-set(stored))
    changed = sorted(key for key in stored.keys() & current.keys() if stored[key] != current[key])
    audit.check("manifest.current_source_hashes", not (missing or added or changed),
                {"missing": missing, "added": added, "changed": changed})
    audit.check("manifest.source_unchanged_during_run",
                manifest.get("source_unchanged_during_run") is True)
    audit.check("manifest.single_process", manifest.get("numerical_processes") == 1)
    audit.check("manifest.single_thread", manifest.get("threads") == 1)
    audit.counts["source_files"] = len(stored)


def audit_exact(audit, model, config):
    stored = audit.read("exact.json")
    expected = run_exact(model, config["t_values"],
                         peak_bounds=stored["wick_peak"]["search_bounds"])
    audit.check("exact.row_count", len(stored["points"]) == len(config["t_values"]))
    fields = ("t", "t_over_v", "energy", "full_ground_energy", "gap", "entropy",
              "schmidt_l1", "mass_correlation", "wick_prediction", "wick_deviation",
              "eta", "gaussian_variational_energy", "eigenstate_residual")
    for index, (row, reference) in enumerate(zip(stored["points"], expected["points"])):
        for field in fields:
            audit.close(f"exact.{index}.{field}", row[field], reference[field])
    audit.close("exact.peak_location", stored["wick_peak"]["t_over_v"],
                expected["wick_peak"]["t_over_v"], atol=1e-5, rtol=0)
    audit.close("exact.peak_value", stored["wick_peak"]["wick_deviation"],
                expected["wick_peak"]["wick_deviation"])
    audit.check("exact.structure_verified", stored["structure"]["verified"] is True
                and expected["structure"]["verified"] is True)
    audit.check("exact.fixed_space_dimension",
                stored["structure"]["symmetry"]["fixed_space_dimension"] == 3)
    with np.load(audit.output/"exact_vectors.npz", allow_pickle=False) as arrays:
        audit.close("exact.saved_S", arrays["S"], model.S, atol=1e-12, rtol=0)
        audit.close("exact.even_masks", arrays["even_masks"], model.even_masks, atol=0, rtol=0)
        audit.close("exact.targets", arrays["targets"],
                    np.array([model.target(t) for t in config["t_values"]]), atol=1e-12, rtol=0)
        audit.close("exact.reduced_interaction", arrays["reduced_interaction"], model.reduced_interaction)
        audit.close("exact.reduced_hopping", arrays["reduced_hopping"], model.reduced_hopping)
    audit.counts["exact_rows"] = len(stored["points"])


def schmidt_norm(model, state):
    return float(np.sum(np.linalg.svd(model.embed(state).reshape(16,16), compute_uv=False)))


def audit_baselines(audit, model, config):
    report = audit.read("baselines.json")
    audit.check("baselines.finite_dictionary_only", report["finite_dictionary_optimality_only"] is True)
    keys = set()
    with np.load(audit.output/"baseline_vectors.npz", allow_pickle=False) as arrays:
        for name in config["dictionaries"]:
            if name+"_atoms" not in arrays:
                audit.check(f"baselines.{name}.documented_missing",
                            report["status"] == "budget_exhausted")
                continue
            atoms = arrays[name+"_atoms"]
            audit.check(f"baselines.{name}.atom_shape", atoms.ndim == 2 and atoms.shape[1] == 128)
            audit.close(f"baselines.{name}.normalized_atoms", np.linalg.norm(atoms, axis=1),
                        np.ones(len(atoms)), atol=1e-10, rtol=0)
            projected = mm(model.S.conj().T, atoms.T)
            audit.close(f"baselines.{name}.unnormalized_projection",
                        arrays[name+"_projected_atoms"], projected, atol=1e-11, rtol=0)
        for row in report["rows"]:
            name, t = row["dictionary"], float(row["t"])
            key = f"{name}_t{t:g}".replace(".", "p")
            audit.check(key+".unique", key not in keys)
            keys.add(key)
            target = arrays[key+"_target"]
            audit.close(key+".target", target, model.target(t), atol=1e-11, rtol=0)
            atoms = arrays[name+"_atoms"]
            projected = mm(model.S.conj().T, atoms.T)
            coeff = arrays[key+"_coefficients"]
            reconstruction = mv(model.S, mv(projected, coeff))
            residual = target-reconstruction
            audit.check(key+".atom_count", row["atom_count"] == len(atoms) == len(coeff))
            audit.close(key+".reconstruction", arrays[key+"_reconstruction"], reconstruction)
            audit.close(key+".residual", arrays[key+"_residual"], residual)
            cost = float(np.sum(np.abs(coeff)))
            state_error = float(row["state_error"])
            schmidt = schmidt_norm(model, target)
            l_lower = max(0., schmidt-4*state_error)
            upper = cost+float(np.sum(np.abs(residual)))+math.sqrt(128)*state_error
            restricted_atom_cap = (1+math.sqrt(row["eta0"]))**2
            lower = l_lower/restricted_atom_cap
            values = {"coefficient_l1": cost, "fock_residual_l1": np.sum(np.abs(residual)),
                      "residual_l2": np.linalg.norm(residual), "cost_upper": upper,
                      "schmidt_norm": schmidt, "restricted_cost_lower": lower,
                      "fock_cost": np.sum(np.abs(target)), "separation_gap": lower-upper,
                      "schmidt_norm_lower": l_lower, "product_schmidt_lower": l_lower,
                      "structure_cost_cap": upper,
                      "mean_eta_necessary_lower": max(math.sqrt(l_lower/upper)-1,0.)**2,
                      "eta_above_threshold_mass_fraction_lower":
                          max((l_lower/upper-restricted_atom_cap)/(4-restricted_atom_cap),0.)
                          if row["eta0"] < 1 else 0.}
            for field, value in values.items():
                audit.close(key+"."+field, row[field], value)
            audit.check(key+".active_atoms", row["active_atoms_1e_7"] == int(np.sum(np.abs(coeff)>1e-7)))
            audit.check(key+".solver_status", row["solver_status"] in ("optimal","optimal_inaccurate"))
            audit.close(key+".state_error_consistent", state_error, row["target_state_error_estimate"])
            margin = float(row["numerical_margin"])
            separated = name == "full" and upper+margin < lower-margin
            audit.check(key+".numerical_separation_flag", bool(row["numerical_separation"]) == separated)
            expected_level = ("product_upper_bound_only" if name == "product" else
                              "numerical_evidence" if separated else "not_separated")
            audit.check(key+".evidence_level", row["evidence_level"] == expected_level)
            audit.check(key+".not_interval_certified", row["strict_interval_certified"] is False
                        and row["target_state_error_interval_certified"] is False)
            audit.check(key+".not_statistically_certified", row["statistically_certified"] is False)
            if name == "full" and t == 20:
                audit.counts["t20_numerical_separation"] = bool(separated)
                audit.counts["t20_cost_upper"] = float(upper)
                audit.counts["t20_restricted_lower"] = float(lower)
    expected_count = len(config["dictionaries"])*len(config["t_values"])
    audit.check("baselines.row_count", len(report["rows"]) == expected_count
                or report["status"] == "budget_exhausted")
    audit.counts["baseline_rows"] = len(report["rows"])


def audit_calibrations(audit, config, summary):
    final_passes = {}
    row_count = 0
    for name in config["dictionaries"]:
        for seed in config["seeds"]:
            label = f"calibration_{name}_{seed}"
            path = audit.output/(label+".json")
            if not path.exists():
                documented = any(s["stage"] == label and s["status"] != "complete"
                                 for s in summary["stages"])
                audit.check(label+".documented_missing", documented)
                audit.warnings.append(label+" missing; no completed calibration to audit")
                continue
            report = audit.read(path.name)
            rows = report["rows"]
            expected_keys = {(int(k),int(n)) for k in config["orders"] for n in config["sample_sizes"]}
            observed_keys = {(int(r["k"]),int(r["samples"])) for r in rows}
            audit.check(label+".row_keys", observed_keys == expected_keys and len(rows) == len(expected_keys))
            for row in rows:
                k, n = int(row["k"]), int(row["samples"])
                ld = log_frame(name,k)
                mass = math.exp(ld-log_frame(name,k-.5))
                q_second = math.exp(2*ld-log_frame(name,2*k))
                coefficient_second = math.exp(2*ld-log_frame(name,2*k-1))
                qse = math.sqrt(max(0.,q_second-1)/n)
                bse = math.sqrt(max(0.,coefficient_second-mass*mass)/n)
                crms = math.sqrt(max(0.,coefficient_second-1)/n)
                prefix = f"{label}.k{k}.n{n}"
                for field, value in (("q_expected",1.),("mass_expected",mass),
                                     ("q_standard_error_exact",qse),
                                     ("mass_standard_error_exact",bse),
                                     ("reconstruction_rmse_exact",crms)):
                    audit.close(prefix+"."+field,row[field],value)
                passed = (abs(row["q"]-1) <= 5*qse+1e-12
                          and abs(row["mass"]-mass) <= 5*bse+1e-12
                          and row["sampled_full_residual"] <= 6*crms+1e-12)
                audit.check(prefix+".diagnostic_flag", bool(row["diagnostic_calibration_passed"]) == passed)
                audit.check(prefix+".precision_flag", bool(row["raw_precision_target_reached"]) ==
                            (row["sampled_full_residual"] <= config["reconstruction_tolerance"]))
                if n == max(config["sample_sizes"]):
                    final_passes[(name,seed,k)] = passed
            final_rows = [r for r in rows if r["samples"] == max(config["sample_sizes"])]
            audit.check(label+".status", report["status"] ==
                        ("passed" if all(r["diagnostic_calibration_passed"] for r in final_rows)
                         else "calibration_failed"))
            row_count += len(rows)
    allowed = {name:[k for k in config["orders"] if all(final_passes.get((name,seed,k),False)
                                                       for seed in config["seeds"])]
               for name in config["dictionaries"]}
    audit.check("calibrations.allowed_orders", summary["calibrated_orders"] == allowed)
    audit.counts["calibration_rows"] = row_count
    audit.counts["planned_calibration_rows"] = (len(config["dictionaries"])*len(config["seeds"])
                                                *len(config["orders"])*len(config["sample_sizes"]))
    audit.counts["all_available_final_calibrations_passed"] = bool(final_passes) and all(final_passes.values())
    return allowed


def audit_pilot_row(audit, model, config, row, arrays, planned_count):
    name, key, k = row["dictionary"], row["key"], int(row["k"])
    prefix = f"pilot.{name}.{row['seed']}.{key}"
    u, z = arrays[key+"_u"], arrays[key+"_z"]
    target = model.target(row["t"])
    audit.check(prefix+".witness_sha256", hashlib.sha256(u.tobytes()).hexdigest() == row["witness_sha256"])
    audit.close(prefix+".three_dimensional_embedding", u, mv(model.S,z), atol=1e-12, rtol=0)
    count = int(row["train_samples"])
    h = arrays["training_overlaps"][:count]
    a = np.einsum("ni,i->n",h,z,optimize=False)
    weights = math.exp(log_frame(name,k))*np.abs(a)**(2*k-1)
    audit.close(prefix+".q_train",row["q_train"],np.mean(weights*np.abs(a)))
    audit.close(prefix+".mass_train",row["mass_train"],np.mean(weights))
    audit.close(prefix+".training_mass_ess",row["training_mass_ess"],weights.sum()**2/np.sum(weights**2))
    if k == 1:
        audit.close(prefix+".analytic_witness",u,target,atol=1e-11,rtol=0)
        audit.check(prefix+".gradient_kind",row["gradient_kind"] == "exact_frame")
        audit.close(prefix+".gradient_norm",row["gradient_norm"],0.)
    else:
        coefficients = math.exp(log_frame(name,k))*np.abs(a)**(2*k-2)*a
        gradient = np.einsum("n,ni->i",coefficients,h.conj(),optimize=False)/count
        target_z = mv(model.S.conj().T,target)
        audit.close(prefix+".gradient_norm",row["gradient_norm"],np.linalg.norm(gradient-target_z),atol=1e-10,rtol=1e-6)
    reconstruction = arrays[key+"_heldout_reconstruction"]
    diff = reconstruction-target
    projected = mv(model.S.conj().T,diff)
    outside = diff-mv(model.S,projected)
    raw = float(np.linalg.norm(diff))
    for field,value in (("sampled_full_residual",raw),
                        ("validation_singlet_residual",np.linalg.norm(projected)),
                        ("validation_outside_singlet_norm",np.linalg.norm(outside)),
                        ("validation_full_residual",0. if k==1 else raw),
                        ("q_validation",np.vdot(u,reconstruction).real)):
        audit.close(prefix+"."+field,row[field],value)
    pythagoras = abs(raw*raw-float(np.linalg.norm(projected))**2-float(np.linalg.norm(outside))**2)
    audit.close(prefix+".pythagoras_report",row["sampled_residual_pythagoras_error"],pythagoras,atol=1e-12,rtol=0)
    audit.check(prefix+".pythagoras_identity",pythagoras < 1e-10)
    mass, n = float(row["mass_validation"]), int(row["validation_samples"])
    audit.check(prefix+".validation_sample_cap",n == max(config["sample_sizes"]) and n <= 65536)
    audit.check(prefix+".mass_dominates_norm",mass+1e-10 >= np.linalg.norm(reconstruction))
    audit.check(prefix+".ess_range",1-1e-10 <= row["mass_ess"] <= n+1e-8)
    audit.check(prefix+".weight_share_range",1/n-1e-12 <= row["max_weight_share"] <= 1+1e-12)
    second = n*mass*mass/row["mass_ess"]
    audit.close(prefix+".mass_standard_error",row["mass_standard_error"],
                math.sqrt(max(0.,second-mass*mass)/max(1,n-1)))
    audit.close(prefix+".residual_mc_rmse_estimate",row["residual_mc_rmse_estimate"],
                math.sqrt(max(0.,second-float(np.vdot(reconstruction,reconstruction).real))/n))
    audit.close(prefix+".empirical_repaired_cost",row["empirical_fock_repaired_cost"],mass+np.sum(np.abs(diff)))
    accepted = bool(row["optimizer_converged"] and row["validation_full_residual"] <= config["reconstruction_tolerance"])
    audit.check(prefix+".acceptance_flag",bool(row["reconstruction_accepted"]) == accepted)
    plan = row["statistical_plan"]
    delta = config["total_failure_probability"]/(2*planned_count)
    bound = math.exp(2*log_frame(name,k)-log_frame(name,2*k-1)+(4*k-2)*math.log(np.linalg.norm(u)))
    required = math.ceil(bound/(delta*config["statistical_radius_target"]**2))
    for field,value in (("second_moment_bound",bound),("failure_probability_per_event",delta),
                        ("statistical_radius_target",config["statistical_radius_target"]),
                        ("sample_cap",max(config["sample_sizes"]))):
        audit.close(prefix+".plan."+field,plan[field],value)
    audit.close(prefix+".plan.required_samples",plan["required_samples"],required,atol=2,rtol=1e-12)
    ready = required <= n
    audit.check(prefix+".plan.status",plan["status"] == ("ready" if ready else "validation_budget_exceeded"))
    if ready:
        radius = math.sqrt(bound/(delta*n))
        upper = 0. if k==1 else raw+radius
        audit.close(prefix+".formal_residual_upper",row["formal_residual_upper"],upper)
        audit.close(prefix+".formal_mass_upper",row["formal_mass_upper"],mass+radius)
        audit.close(prefix+".formal_repaired_cost_upper",row["formal_repaired_cost_upper"],mass+radius+math.sqrt(128)*upper)
        certified = upper <= config["reconstruction_tolerance"]
        audit.check(prefix+".statistical_flag",bool(row["statistically_certified"]) == certified)
    else:
        audit.check(prefix+".no_budget_no_certificate",row["statistically_certified"] is False)
    audit.check(prefix+".distribution_gate",not row["distribution_interpretation_allowed"]
                or (accepted and row["coherent_bin_closure_residual"] <= config["reconstruction_tolerance"]))


def audit_pilots(audit, model, config, summary, allowed):
    all_rows = []
    planned = (len(config["dictionaries"])*len(config["seeds"])*len(config["sample_sizes"])
               *len(config["t_values"])*len(config["orders"]))
    for name in config["dictionaries"]:
        for seed in config["seeds"]:
            label = f"{name}_{seed}"
            path = audit.output/("pilot_"+label+".json")
            if not path.exists():
                documented = any(s["stage"] in ("training_"+label,"validation_"+label)
                                 and s["status"] != "complete" for s in summary["stages"])
                audit.check(label+".documented_missing_pilot",documented)
                audit.warnings.append(label+" has no completed pilot; checkpoint remains separate")
                continue
            report = audit.read(path.name)
            rows = report["rows"]
            expected_keys = {(int(n),float(t),int(k)) for n in config["sample_sizes"]
                             for t in config["t_values"] for k in allowed[name]}
            actual_keys = {(int(r["train_samples"]),float(r["t"]),int(r["k"])) for r in rows}
            audit.check(label+".pilot_row_keys",actual_keys == expected_keys and len(rows) == len(expected_keys))
            if not rows:
                audit.check(label+".skipped_status",report["status"] == "skipped_calibration_failed")
                continue
            audit.check(label+".planned_count",report["planned_candidate_count"] == planned)
            audit.check(label+".streams",report["random_streams"] == {
                "calibration":[seed,int(name=="product"),0],
                "training":[seed,int(name=="product"),1],
                "heldout_validation":[seed,int(name=="product"),2]})
            training = audit.read("training_"+label+".json")
            audit.check(label+".training_row_count",len(training["rows"]) == len(rows))
            training_by_key = {r["key"]:r for r in training["rows"]}
            with np.load(audit.output/("witnesses_"+label+".npz"),allow_pickle=False) as arrays:
                for row in rows:
                    audit.check(label+"."+row["key"]+".frozen_training_witness",
                                training_by_key[row["key"]]["witness_sha256"] == row["witness_sha256"])
                    audit_pilot_row(audit,model,config,row,arrays,planned)
                distribution_path = audit.output/("distributions_"+label+".npz")
                if distribution_path.exists():
                    with np.load(distribution_path,allow_pickle=False) as distributions:
                        operator = model.mass_correlation_operator[np.ix_(model.even_masks,model.even_masks)]
                        for row in rows:
                            key = row["key"]
                            if key+"_coherent_vectors" not in distributions:
                                continue
                            phi = distributions[key+"_coherent_vectors"]
                            total = np.sum(phi,axis=0)
                            audit.close(label+"."+key+".coherent_bin_sum",total,arrays[key+"_heldout_reconstruction"])
                            kernel = np.einsum("bi,ij,cj->bc",phi.conj(),operator,phi,optimize=False)
                            audit.close(label+"."+key+".coherent_kernel",distributions[key+"_correlation_kernel"],kernel)
                            histogram = distributions[key+"_histogram"]
                            audit.close(label+"."+key+".histogram_normalization",np.sum(histogram),1.)
                            audit.check(label+"."+key+".histogram_nonnegative",np.all(histogram >= -1e-14))
            all_rows.extend(rows)
    counts = {"candidate_count":len(all_rows),
              "reconstruction_accepted_count":sum(r["reconstruction_accepted"] for r in all_rows),
              "k_ge_2_reconstruction_accepted_count":sum(r["reconstruction_accepted"] and r["k"]>=2 for r in all_rows),
              "statistically_certified_count":sum(r["statistically_certified"] for r in all_rows),
              "distribution_interpretation_allowed_count":sum(r["distribution_interpretation_allowed"] for r in all_rows)}
    for field,value in counts.items():
        audit.check("summary."+field,summary[field] == value)
    audit.counts.update(counts)
    audit.counts["planned_pilot_rows"] = planned
    audit.counts["pilot_precision_failures"] = sum(not r["reconstruction_accepted"] for r in all_rows)


def verify(output):
    audit = Audit(output)
    config = audit.read("config.json")
    summary = audit.read("summary.json")
    manifest = audit.read("manifest.json")
    model = build_model()
    audit.section("manifest",lambda:audit_manifest(audit,manifest))
    audit.section("exact",lambda:audit_exact(audit,model,config))
    if (audit.output/"baselines.json").exists():
        audit.section("baselines",lambda:audit_baselines(audit,model,config))
    else:
        documented = any(s["stage"] == "baselines" and s["status"] != "complete"
                         for s in summary["stages"])
        audit.check("baselines.documented_missing",documented)
        audit.warnings.append("No completed baseline stage to audit")
    allowed_box = {}
    audit.section("calibrations",lambda:allowed_box.update(audit_calibrations(audit,config,summary)))
    if allowed_box:
        audit.section("pilots",lambda:audit_pilots(audit,model,config,summary,allowed_box))
    result = {"audit_passed":not audit.failures,"checks":audit.checks,
              "failure_count":len(audit.failures),"failures":audit.failures,
              "warnings":audit.warnings,"counts":audit.counts,
              "run_status":summary["status"],"output_dir":str(audit.output),
              "monte_carlo_rerun":False,"optimization_rerun":False,
              "scope":"Saved-array and algebraic consistency audit with deterministic model/ED recomputation. No new samples or SOCP solves.",
              "limitations":["Stored calibration scalar estimates cannot be regenerated without original samples; their analytic references and decision flags are checked.",
                             "This audit checks numerical reproducibility and evidence labels, not interval-arithmetic correctness.",
                             "Sampling precision failures and declared resource-limited omissions are outcomes, not audit inconsistencies."]}
    (audit.output/"verification.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir",type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.output_dir)
    except Exception as exc:
        result = {"audit_passed":False,"failure_count":1,"failures":[{
            "check":"audit_setup","error":str(exc),"traceback":traceback.format_exc()}],
            "monte_carlo_rerun":False,"optimization_rerun":False}
        if args.output_dir.is_dir():
            (args.output_dir/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({key:result[key] for key in ("audit_passed","failure_count")},indent=2))
    return 0 if result["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
