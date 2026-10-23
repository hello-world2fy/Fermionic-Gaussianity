#!/usr/bin/env python3
"""Deterministically audit a saved follow-up; never resample or rerun SOCP.

Usage: python scripts/verify_results.py OUTPUT_DIR
Writes OUTPUT_DIR/verification.json and exits nonzero on inconsistencies.
Failed sampling precision is a scientific result, not an audit failure.
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

for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT.parent/"fk_two_cluster"
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(BACKEND))
import numpy as np
from fk.model import build_model
from fk.conventions import parity_masks
from fk.runtime import source_hashes
from fk2.exact_moments import (ExactK2Moments, integer_fk_matrix_proof,
                              closed_form_fk_matrix, full_tensor_reconstruction)


def digest(value):
    return hashlib.sha256(np.asarray(value,dtype=np.complex128).tobytes()).hexdigest()


def file_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()
            and not any(part.startswith(".") for part in p.relative_to(root).parts)}


def mv(a,b):
    return np.einsum("ij,j->i",a,b,optimize=False)


def project(model,vector):
    return np.einsum("ij,...i->...j",model.S.conj(),vector,optimize=False)


def embed(model,coordinates):
    return np.einsum("ij,...j->...i",model.S,coordinates,optimize=False)


def log_frame(name,k):
    logd = lambda n:sum(math.log1p(k/(2*n-i-j)) for i in range(1,n+1) for j in range(i+1,n+1))
    return logd(8) if name == "full" else math.log(2)+2*logd(4)


class Audit:
    def __init__(self,output):
        self.output = Path(output).resolve()
        self.checks = 0
        self.failures = []
        self.counts = {}
        self.warnings = []

    def check(self,name,passed,detail=None):
        self.checks += 1
        if not bool(passed):
            self.failures.append({"check":name,"detail":detail})

    def close(self,name,actual,expected,atol=1e-10,rtol=1e-10):
        actual,expected = np.asarray(actual),np.asarray(expected)
        if actual.dtype.kind == "O": actual = actual.astype(float)
        if expected.dtype.kind == "O": expected = expected.astype(float)
        if actual.shape != expected.shape:
            self.check(name,False,{"actual_shape":list(actual.shape),"expected_shape":list(expected.shape)})
            return
        finite = np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
        passed = finite and np.allclose(actual,expected,atol=atol,rtol=rtol)
        maximum = float(np.max(np.abs(actual-expected))) if finite and actual.size else None
        self.check(name,passed,{"max_absolute_error":maximum,"atol":atol,"rtol":rtol})

    def load(self,name):
        return json.loads((self.output/name).read_text())

    def section(self,name,call):
        try:
            return call()
        except Exception as error:
            self.check(name+".exception",False,{"error":str(error),"traceback":traceback.format_exc()})
            return None


def hash_check(audit,label,stored,current):
    missing = sorted(set(stored)-set(current))
    added = sorted(set(current)-set(stored))
    changed = sorted(key for key in set(stored)&set(current) if stored[key] != current[key])
    audit.check(label,not (missing or added or changed),{"missing":missing,"added":added,"changed":changed})


def audit_hashes(audit,manifest,source):
    hash_check(audit,"hash.own",manifest["source_sha256"],source_hashes(ROOT))
    hash_check(audit,"hash.backend",manifest["backend_sha256"],source_hashes(BACKEND))
    hash_check(audit,"hash.inputs",manifest["input_sha256"],file_hashes(source))
    for key in ("source_unchanged_during_run","backend_unchanged_during_run","inputs_unchanged_during_run"):
        audit.check("manifest."+key,manifest[key] is True)
    audit.check("manifest.single_thread",manifest["numerical_threads"] == 1)
    audit.check("manifest.single_process",manifest["numerical_processes"] == 1)


def dense_projector_reference(model,tensor):
    """Independent dense local-block implementation, never a 16384² matrix."""
    even,odd = parity_masks(8,0),parity_masks(8,1)
    blocks = [model.gammas[:,odd][:,:,even],model.gammas[:,even][:,:,odd]]
    def action(value,parity):
        result = np.zeros_like(value)
        for gamma in blocks[parity]:
            result += gamma@value@gamma.T
        return result
    squared = action(action(tensor,0),1)
    fourth = action(action(squared,0),1)
    return tensor-(5/256)*squared+fourth/16384


def audit_exact(audit,model,config):
    report = audit.load("exact_k2.json")
    proof = integer_fk_matrix_proof()
    backend = ExactK2Moments(model)
    audit.check("exact.integer_proof",proof["verified"] is True and report["integer_closed_form_proof_verified"] is True)
    audit.check("exact.verified",report["verified"] is True)
    audit.check("exact.row_count",len(report["rows"]) == len(config["t_values"]))
    with np.load(audit.output/"model.npz",allow_pickle=False) as saved:
        audit.close("model.S",saved["S"],model.S)
        audit.close("model.even_masks",saved["even_masks"],model.even_masks,atol=0,rtol=0)
    with np.load(audit.output/"exact_k2.npz",allow_pickle=False) as saved:
        audit.close("exact.integer_gram",saved["integer_gram"],proof["integer_gram"],atol=0,rtol=0)
        for key,value in (("moment_matrix",backend.matrix),("haar_moment_matrix",backend.matrix/6435),
                          ("closed_form_matrix",closed_form_fk_matrix()),("exact_radical_basis",proof["basis"])):
            audit.close("exact."+key,saved[key],value)
        # A complex probe exercises mixed real/imaginary polynomial entries.
        z = np.array([.31+.17j,-.23+.11j,.19-.29j])
        u = embed(model,z)
        tensor = np.outer(u,u)
        projected = dense_projector_reference(model,tensor)
        audit.close("exact.independent_dense_q",np.vdot(tensor,projected).real,backend.evaluate(z).q)
        audit.close("exact.independent_dense_C",mv(projected,u.conj()),backend.reconstruction(z))
        for row in report["rows"]:
            t,key = row["t"],f"t{row['t']:g}"
            target = model.target(t)
            u,z = saved["witness_"+key],saved["z_"+key]
            direct = full_tensor_reconstruction(u)
            dense_direct = mv(dense_projector_reference(model,np.outer(u,u)),u.conj())
            reduced = backend.reconstruction(z)
            audit.close(key+".witness_embedding",u,embed(model,z))
            audit.close(key+".target",saved["target_"+key],target)
            audit.close(key+".dense_independent_C",dense_direct,direct)
            audit.close(key+".full_tensor_C",saved["full_tensor_reconstruction_"+key],direct)
            audit.close(key+".reduced_C",saved["reconstruction_"+key],reduced)
            audit.close(key+".q",row["q"],backend.evaluate(z).q)
            audit.close(key+".full_residual",row["full_reconstruction_residual"],np.linalg.norm(direct-target))
            audit.close(key+".singlet_closure",row["full_tensor_singlet_closure_residual"],np.linalg.norm(direct-reduced))
            audit.check(key+".actual_residual_accepted",np.linalg.norm(direct-target) <= config["structural_tolerance"])
            audit.check(key+".no_mass_claim",row["coefficient_mass"] is None)
            audit.check(key+".no_interval_claim",row["strict_interval_certified"] is False)
    audit.counts["exact_reference_rows"] = len(report["rows"])
    return backend


def audit_rb(audit,model,config,source,backend,summary):
    rows_all = []
    source_count = 0
    stage_count = 0
    with np.load(audit.output/"exact_k2.npz",allow_pickle=False) as exact_saved:
        for name in config["dictionaries"]:
            for seed in config["seeds"]:
                label = f"rb_{name}_{seed}"
                report = audit.load(label+".json")
                stage_count += 1
                audit.check(label+".no_formal_claim",report["formal_confidence_guarantee"] is False)
                audit.check(label+".no_source_mutation",report["original_source_files_modified"] is False)
                audit.check(label+".rng",report["rng_seed_sequence"] == [seed,int(name=="product"),config["rng_purpose"]])
                provenance = report["source"]
                for filename_key,hash_key in (("pilot_file","pilot_file_sha256"),("witness_file","witness_file_sha256")):
                    audit.check(label+"."+hash_key,hashlib.sha256((source/provenance[filename_key]).read_bytes()).hexdigest() == provenance[hash_key])
                expected_candidates = len(config["t_values"])*len(config["orders"])
                if name == "full": expected_candidates += len(config["orders"])+len(config["t_values"])
                audit.check(label+".candidate_count",report["candidate_count"] == expected_candidates)
                audit.check(label+".row_count",len(report["rows"]) == expected_candidates*len(config["sample_sizes"]))
                source_count += expected_candidates
                with np.load(audit.output/(label+".npz"),allow_pickle=False) as saved, \
                     np.load(source/f"witnesses_{name}_{seed}.npz",allow_pickle=False) as original:
                    labels = list(map(str,saved["candidate_labels"]))
                    audit.check(label+".unique_labels",len(set(labels)) == expected_candidates)
                    lookup = {key:i for i,key in enumerate(labels)}
                    witnesses,targets = saved["witnesses"],saved["targets"]
                    audit.close(label+".S",saved["basis"],model.S)
                    for row in report["rows"]:
                        i = lookup[row["candidate"]]
                        n = row["samples"]
                        prefix = f"{label}.{row['candidate']}.n{n}"
                        u,target = witnesses[i],targets[i]
                        raw = saved[f"n{n}_raw_C"][i]
                        orbit = saved[f"n{n}_orbit_C"][i]
                        coordinates = saved[f"n{n}_projected_C"][i]
                        audit.check(prefix+".witness_hash",digest(u) == row["source_witness_sha256"])
                        audit.check(prefix+".target_hash",digest(target) == row["target_sha256"])
                        audit.close(prefix+".invariant_witness",u,embed(model,project(model,u)))
                        audit.close(prefix+".projected_coordinates",coordinates,project(model,raw))
                        audit.close(prefix+".orbit_is_full_projection",orbit,embed(model,project(model,raw)))
                        raw_res,orbit_res,complement = map(float,(np.linalg.norm(raw-target),np.linalg.norm(orbit-target),np.linalg.norm(raw-orbit)))
                        for field,value in (("raw_full_residual",raw_res),("projected_residual",np.linalg.norm(coordinates-project(model,target))),
                                            ("orbit_full_residual",orbit_res),("complement_residual",complement),
                                            ("q_estimate",np.vdot(u,raw).real)):
                            audit.close(prefix+"."+field,row[field],value)
                        pythagoras = abs(raw_res**2-orbit_res**2-complement**2)
                        audit.close(prefix+".pythagoras_report",row["paired_pythagoras_error"],pythagoras)
                        audit.check(prefix+".pythagoras",pythagoras <= 1e-10*max(1,raw_res**2))
                        audit.check(prefix+".raw_gate",bool(row["raw_within_tolerance"]) == (raw_res <= config["reconstruction_tolerance"]))
                        audit.check(prefix+".orbit_gate",bool(row["orbit_within_tolerance"]) == (orbit_res <= config["reconstruction_tolerance"]))
                        audit.check(prefix+".no_certificate",row["formally_certified"] is False)
                        audit.check(prefix+".old_acceptance_unchanged",row["original_first_round_acceptance_changed"] is False)
                        if row["sourcekind"] == "frozen_pilot":
                            audit.close(prefix+".source_witness",u,original[row["candidate"]+"_u"],atol=0,rtol=0)
                        elif row["sourcekind"] == "deterministic_k2":
                            audit.close(prefix+".exact_witness",u,exact_saved[f"witness_t{row['t']:g}"],atol=0,rtol=0)
                        if row["t"] is not None:
                            audit.close(prefix+".physical_target",target,model.target(row["t"]))
                        exact_c = None
                        if name == "full" and row["k"] == 2:
                            exact_c = full_tensor_reconstruction(u)
                        elif row["sourcekind"] == "hopping_gaussian_probe":
                            exact_c = target
                        if exact_c is not None:
                            audit.check(prefix+".exact_reference_available",bool(saved["exact_reconstruction_available"][i]))
                            audit.close(prefix+".saved_exact_C",saved["exact_reconstructions"][i],exact_c)
                            bias,raw_noise,orbit_noise = exact_c-target,raw-exact_c,orbit-exact_c
                            for field,value in (("exact_bias_l2",np.linalg.norm(bias)),("raw_error_to_exact_C",np.linalg.norm(raw_noise)),
                                                ("orbit_error_to_exact_C",np.linalg.norm(orbit_noise)),("rb_error_to_exact_C",np.linalg.norm(orbit_noise)),
                                                ("raw_bias_noise_cross_term",2*np.vdot(raw_noise,bias).real),
                                                ("orbit_bias_noise_cross_term",2*np.vdot(orbit_noise,bias).real)):
                                audit.close(prefix+"."+field,row[field],value)
                        if row["sourcekind"] == "hopping_gaussian_probe":
                            k = row["k"]
                            ld = log_frame("full",k)
                            expected_mass = math.exp(ld-log_frame("full",k-.5))
                            qse = math.sqrt((math.exp(2*ld-log_frame("full",2*k))-1)/n)
                            crms = math.sqrt((math.exp(2*ld-log_frame("full",2*k-1))-1)/n)
                            bse = math.sqrt((math.exp(2*ld-log_frame("full",2*k-1))-expected_mass**2)/n)
                            passed = abs(row["q_estimate"]-1) <= 5*qse+1e-12 and abs(row["mass_estimate"]-expected_mass) <= 5*bse+1e-12 and raw_res <= 6*crms+1e-12
                            audit.check(prefix+".calibration_flag",bool(row["probe_diagnostic_calibration_passed"]) == passed)
                        audit.check(prefix+".mass_dominates_norm",row["mass_estimate"]+1e-10 >= np.linalg.norm(raw))
                        audit.check(prefix+".ESS_range",1-1e-10 <= row["mass_ess"] <= n+1e-8)
                        audit.close(prefix+".raw_repaired_cost",row["raw_fock_repaired_cost_numeric"],row["mass_estimate"]+np.sum(np.abs(target-raw)))
                        audit.close(prefix+".orbit_repaired_cost",row["orbit_fock_repaired_cost_numeric"],row["mass_estimate"]+np.sum(np.abs(target-orbit)))
                        if n == int(saved["completed_samples"]):
                            audit.close(prefix+".raw_accumulator",raw,saved["raw_sum"][i]/n)
                            audit.close(prefix+".projected_accumulator",coordinates,saved["projected_sum"][i]/n)
                            raw_var = max(0.,(saved["raw_norm2_sum"][i]-n*np.vdot(raw,raw).real)/(n-1))
                            rb_var = max(0.,(saved["projected_norm2_sum"][i]-n*np.vdot(coordinates,coordinates).real)/(n-1))
                            mass = saved["mass_sum"][i]/n
                            mass_var = max(0.,(saved["mass_norm2_sum"][i]-n*mass**2)/(n-1))
                            values = {"raw_sample_variance_trace":raw_var,"orbit_sample_variance_trace":rb_var,
                                      "complement_sample_variance_trace":max(0.,raw_var-rb_var),
                                      "raw_mean_rmse_estimate":math.sqrt(raw_var/n),"orbit_mean_rmse_estimate":math.sqrt(rb_var/n),
                                      "mass_estimate":mass,"mass_standard_error_estimate":math.sqrt(mass_var/n),
                                      "mass_ess":saved["mass_sum"][i]**2/saved["mass_norm2_sum"][i],
                                      "max_mass_weight_share":saved["max_mass_weight"][i]/saved["mass_sum"][i],
                                      "q_estimate":saved["q_sum"][i]/n}
                            if rb_var > 0: values["paired_variance_ratio"] = raw_var/rb_var
                            if raw_var > 0: values["variance_reduction_fraction"] = 1-rb_var/raw_var
                            for field,value in values.items():audit.close(prefix+"."+field,row[field],value)
                rows_all.extend(report["rows"])
    finals = [r for r in rows_all if r["samples"] == max(config["sample_sizes"])]
    frozen = [r for r in finals if r["sourcekind"] == "frozen_pilot"]
    counts = {"rb_record_count":len(rows_all),"rb_final_record_count":len(finals),
              "frozen_final_count":len(frozen),"frozen_final_raw_accepted":sum(r["raw_within_tolerance"] for r in frozen),
              "frozen_final_orbit_accepted":sum(r["orbit_within_tolerance"] for r in frozen),
              "rb_statistical_certificates":sum(r["formally_certified"] for r in rows_all),"rb_stage_count":stage_count}
    for key,value in counts.items():audit.check("summary."+key,summary[key] == value)
    audit.counts.update(counts)
    audit.counts["planned_rb_sources"] = source_count


def actual_covariances(model,kets):
    full = np.zeros((len(kets),256),complex)
    full[:,model.even_masks] = kets
    actions = np.einsum("aij,nj->nai",model.gammas,full,optimize=False)
    corr = np.einsum("nai,nbi->nab",actions.conj(),actions,optimize=False)
    covariance = (1j*corr).real
    covariance[:,np.arange(16),np.arange(16)] = 0.
    return (covariance-covariance.transpose(0,2,1))/2


def physical_values(model,state,t):
    norm = float(np.linalg.norm(state))
    if norm < 1e-14:return None
    state = state/norm
    full = model.embed(state)
    cov = actual_covariances(model,state[None,:])[0]
    a,b,c = cov[:8,:8],cov[8:,8:],cov[:8,8:]
    ma = np.einsum("kab,ab->k",model.j_matrices,a,optimize=False)/8
    mb = np.einsum("kab,ab->k",model.j_matrices,b,optimize=False)/8
    eta = np.sum(c*c)/8
    wick = 4*np.dot(ma,mb)-np.einsum("kab,kcd,ac,bd->",model.j_matrices,model.j_matrices,c,c,optimize=False)/8
    cq = np.vdot(full,mv(model.mass_correlation_operator,full)).real
    sv = np.linalg.svd(full.reshape(16,16),compute_uv=False)
    probabilities = sv*sv
    positive = probabilities[probabilities > 0]
    return {"energy":float(np.vdot(full,mv(model.full_hamiltonian(t),full)).real),
            "entropy":float(-np.sum(positive*np.log2(positive))),"schmidt_l1":float(sv.sum()),
            "mass_correlation":float(cq),"wick_prediction":float(wick),"wick_deviation":float(cq-wick),
            "eta":float(eta),"mass_a":float(np.dot(ma,ma)),"mass_b":float(np.dot(mb,mb)),"mass_dot":float(np.dot(ma,mb))}


def audit_cost(audit,prefix,row,coefficients,residual,schmidt):
    mass = float(np.sum(np.abs(coefficients)))
    error = float(row["state_error"])
    upper = mass+np.sum(np.abs(residual))+math.sqrt(128)*error
    l_lower = max(0.,schmidt-4*error)
    atom_cap = (1+math.sqrt(row["eta0"]))**2
    lower = l_lower/atom_cap
    values = {"coefficient_l1":mass,"fock_residual_l1":np.sum(np.abs(residual)),"residual_l2":np.linalg.norm(residual),
              "cost_upper":upper,"restricted_cost_lower":lower,"separation_gap":lower-upper,
              "schmidt_norm_lower":l_lower,"structure_cost_cap":upper,
              "mean_eta_necessary_lower":max(math.sqrt(l_lower/upper)-1,0.)**2,
              "eta_above_threshold_mass_fraction_lower":max((l_lower/upper-atom_cap)/(4-atom_cap),0.)}
    for field,value in values.items():audit.close(prefix+"."+field,row[field],value)
    separated = upper+row["numerical_margin"] < lower-row["numerical_margin"]
    audit.check(prefix+".separation_flag",bool(row["numerical_separation"]) == separated)
    audit.check(prefix+".evidence_level",row["evidence_level"] == ("numerical_evidence" if separated else "not_separated"))
    audit.check(prefix+".no_interval_claim",row["strict_interval_certified"] is False)
    audit.check(prefix+".no_statistical_claim",row["statistically_certified"] is False)


def audit_ablation(audit,model,config,source_arrays):
    report = audit.load("ablation.json")
    expected = 4*len(config["t_values"])
    audit.check("ablation.row_count",len(report["rows"]) == expected)
    with np.load(audit.output/"ablation.npz",allow_pickle=False) as saved:
        for row in report["rows"]:
            key,t = row["key"],row["t"]
            indices = saved[key+"_indices"]
            atoms = source_arrays["full_atoms"][indices]
            coeff = saved[key+"_coefficients"]
            target = model.target(t)
            projected = project(model,atoms)
            reconstruction = embed(model,np.einsum("n,ni->i",coeff,projected,optimize=False))
            residual = target-reconstruction
            audit.close(key+".atom_normalization",np.linalg.norm(atoms,axis=1),np.ones(len(atoms)))
            audit.close(key+".residual",saved[key+"_residual"],residual)
            audit.check(key+".atom_count",row["atom_count"] == len(atoms))
            schmidt = physical_values(model,target,t)["schmidt_l1"]
            audit_cost(audit,key,row,coeff,residual,schmidt)
    audit.counts["ablation_rows"] = len(report["rows"])


def audit_finite(audit,model,config,source_arrays):
    report = audit.load("finite_orbits.json")
    t = report["t"]
    target = model.target(t)
    reference = physical_values(model,target,t)
    count = report["representative_count"]
    expected_rows = 1+sum(r < count*8 for r in config["orbit_sizes"])+len(config["seeds"])*len(config["orbit_sizes"])
    audit.check("finite.row_count",len(report["rows"]) == expected_rows)
    audit.check("finite.no_interval_claim",report["strict_interval_certified"] is False)
    with np.load(audit.output/"finite_orbits.npz",allow_pickle=False) as saved:
        indices = saved["source_representative_indices"]
        reps = saved["source_representative_kets"]
        audit.close("finite.source_representatives",reps,source_arrays["full_atoms"][indices])
        directions = saved["simplex_directions"]
        audit.close("finite.simplex_mean",directions.mean(axis=0),np.zeros(7))
        audit.close("finite.simplex_second_moment",directions.T@directions/8,np.eye(7)/7)
        audit.close("finite.simplex_norm",np.linalg.norm(directions,axis=1),np.ones(8))
        design = saved["exact_simplex_kets"]
        audit.check("finite.simplex_branch_count",len(design) == 8*count)
        means = design.reshape(8,count,128).mean(axis=0)
        projected = embed(model,project(model,reps))
        errors = np.linalg.norm(means-projected,axis=1)
        audit.close("finite.simplex_orbit_means",saved["simplex_orbit_means"],means)
        audit.close("finite.simplex_orbit_average_errors",report["simplex_orbit_average_errors"],errors)
        audit.check("finite.simplex_specific_orbits_flag",bool(report["simplex_specific_orbits_verified"]) ==
                    (np.max(errors) < config["structural_tolerance"]))
        audit.close("finite.simplex_constructive_coefficients",saved["exact_simplex_coefficients"],
                    np.tile(saved["refitted_projected_coefficients"]/8,8))
        for row in report["rows"]:
            key = row["key"]
            kets,coeff = saved[key+"_kets"],saved[key+"_coefficients"]
            raw = np.einsum("n,ni->i",coeff,kets,optimize=False)
            repair = saved[key+"_fock_repair_coefficients"]
            residual = target-raw
            audit.close(key+".target",saved[key+"_target"],target)
            audit.close(key+".raw_reconstruction",saved[key+"_reconstruction"],raw)
            audit.close(key+".fock_repair",repair,residual)
            audit.close(key+".repaired_target",raw+repair,target)
            audit.check(key+".total_branch_count",row["R"] == len(kets) == row["finite_gaussian_count"])
            audit.check(key+".active_count",row["active_gaussian_count_1e_7"] == int(np.sum(np.abs(coeff)>1e-7)))
            covariance = actual_covariances(model,kets)
            audit.close(key+".actual_covariance",saved[key+"_covariances"],covariance)
            norm_error = np.max(np.abs(np.linalg.norm(kets,axis=1)-1))
            purity = np.max(np.linalg.norm(covariance@covariance+np.eye(16),axis=(1,2)))
            audit.close(key+".norm_error",row["max_atom_norm_error"],norm_error)
            audit.close(key+".purity_error",row["max_gaussian_purity_error"],purity)
            audit.check(key+".actual_atoms_are_gaussian",norm_error < 1e-10 and purity < 1e-10)
            audit.check(key+".gaussian_check_flag",row["gaussian_checks_passed"] is True)
            audit.close(key+".raw_residual",row["raw_full_residual"],np.linalg.norm(residual))
            audit.close(key+".repaired_residual",row["repaired_full_residual"],np.linalg.norm(raw+repair-target))
            audit.check(key+".fock_repair_count",row["fock_repair_count_nonzero"] == int(np.count_nonzero(repair)))
            audit.check(key+".fock_repair_count_threshold",row["fock_repair_count_above_1e_12"] == int(np.sum(np.abs(repair)>1e-12)))
            audit_cost(audit,key,row,coeff,residual,reference["schmidt_l1"])
            for kind,state in (("raw",raw),("repaired",raw+repair)):
                values = physical_values(model,state,t)
                if values is None:
                    audit.check(key+"."+kind+"_undefined",row[kind+"_observables"] is None)
                    continue
                for field,value in values.items():
                    audit.close(key+"."+kind+"."+field,row[kind+"_observables"][field],value)
                    audit.close(key+"."+kind+".error."+field,row[kind+"_observable_absolute_errors"][field],abs(value-reference[field]))
    audit.counts["finite_rows"] = len(report["rows"])
    audit.counts["simplex_gaussian_branches"] = count*8


def verify(output):
    audit = Audit(output)
    config,summary,manifest = audit.load("config.json"),audit.load("summary.json"),audit.load("manifest.json")
    source = Path(manifest["source_run"])
    model = build_model()
    audit.section("hashes",lambda:audit_hashes(audit,manifest,source))
    backend = audit.section("exact",lambda:audit_exact(audit,model,config))
    if backend is not None:
        audit.section("RB",lambda:audit_rb(audit,model,config,source,backend,summary))
    with np.load(source/"baseline_vectors.npz",allow_pickle=False) as saved:
        source_arrays = {key:saved[key] for key in saved.files}
    audit.section("ablation",lambda:audit_ablation(audit,model,config,source_arrays))
    audit.section("finite_orbits",lambda:audit_finite(audit,model,config,source_arrays))
    result = {"audit_passed":not audit.failures,"checks":audit.checks,"failure_count":len(audit.failures),
              "failures":audit.failures,"warnings":audit.warnings,"counts":audit.counts,
              "source_run":str(source),"output_dir":str(audit.output),"run_status":summary["status"],
              "monte_carlo_rerun":False,"optimization_rerun":False,
              "strict_interval_certified":False,"global_optimality_certified":False,
              "scope":"Hash consistency; exact integer k2 matrix and independent dense-block full tensor references; saved paired MC vectors and final accumulators; explicit finite Gaussian kets, repairs, costs and observables.",
              "limitations":["Monte Carlo estimates are not regenerated; earlier sample checkpoints retain means but only the final checkpoint retains second-moment accumulators.",
                             "Tolerance-based numerical consistency does not turn variance diagnostics into confidence intervals or finite dictionaries into global optima."]}
    (audit.output/"verification.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir",type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.output_dir)
    except Exception as error:
        result = {"audit_passed":False,"failure_count":1,"failures":[{"check":"setup","error":str(error),"traceback":traceback.format_exc()}],
                  "monte_carlo_rerun":False,"optimization_rerun":False}
        if args.output_dir.is_dir():
            (args.output_dir/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({key:result[key] for key in ("audit_passed","failure_count")},indent=2))
    return 0 if result["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
