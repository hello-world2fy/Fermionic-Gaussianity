"""Audit saved evidence without regenerating Haar samples or solving programs.

This validates algebraic provenance, frozen witnesses, sufficient statistics,
reported gates, and finite orbit decompositions. It cannot certify that every
saved sufficient statistic came from the claimed random stream. One saved
Gaussian rotation path per seed and physical target is replayed independently.
Floating checks and statistical assumptions are not interval certificates.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT.parent/"fk_two_cluster"
FOLLOWUP = ROOT.parent/"fk_two_cluster_followup"
for entry in (ROOT, BACKEND, FOLLOWUP):
    sys.path.insert(0, str(entry))

import numpy as np
from fk.model import build_model
from fk.observables import exact_metrics
from fk.runtime import save_json
from fk2.exact_moments import closed_form_fk_matrix, full_tensor_reconstruction, integer_fk_matrix_proof


class Audit:
    def __init__(self):
        self.checks = 0
        self.failures = []

    def check(self, condition, label):
        self.checks += 1
        if not bool(condition):
            self.failures.append(label)

    def close(self, actual, expected, label, atol=2e-9, rtol=2e-9):
        a, b = np.asarray(actual), np.asarray(expected)
        self.check(a.shape == b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b))
                   and np.allclose(a, b, atol=atol, rtol=rtol), label)

    def block(self, label, operation):
        try:
            operation()
        except Exception as exc:
            self.check(False, f"{label}: {type(exc).__name__}: {exc}")


def digest(vector):
    return hashlib.sha256(np.asarray(vector, np.complex128).tobytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def dimension(order):
    return math.exp(sum(math.log1p(order/(16-i-j)) for i in range(1, 9) for j in range(i+1, 9)))


def own_covariances(states, model):
    """Independent dense-model Majorana permutations, batched into small blocks."""
    states = np.asarray(states, complex)
    permutations = np.argmax(abs(model.gammas), axis=2)
    phases = np.take_along_axis(model.gammas, permutations[:, :, None], axis=2)[..., 0]
    out = []
    for start in range(0, len(states), 32):
        full = np.zeros((len(states[start:start+32]), 256), complex)
        full[:, model.even_masks] = states[start:start+32]
        acted = full[:, permutations]*phases
        cov = (1j*np.einsum("nai,nbi->nab", acted.conj(), acted, optimize=True)).real
        cov[:, np.arange(16), np.arange(16)] = 0
        out.append((cov-cov.transpose(0, 2, 1))/2)
    return np.concatenate(out)


def own_invariants(cov, model):
    ma = np.einsum("kab,nab->nk", model.j_matrices, cov[:, :8, :8])/8
    mb = np.einsum("kab,nab->nk", model.j_matrices, cov[:, 8:, 8:])/8
    return {"eta": np.sum(cov[:, :8, 8:]**2, axis=(1, 2))/8,
            "mass_a": np.sum(ma**2, axis=1), "mass_b": np.sum(mb**2, axis=1),
            "mass_dot": np.sum(ma*mb, axis=1)}


def verify_canonical(out, model, audit):
    report = load_json(out/"canonical_k2.json")
    arrays = load_npz(out/"canonical_k2.npz")
    proof = integer_fk_matrix_proof()
    audit.close(arrays["matrix"], closed_form_fk_matrix(), "canonical closed matrix", atol=1e-12)
    audit.close(arrays["matrix"], proof["matrix"], "canonical integer matrix", atol=1e-12)
    audit.check(np.array_equal(arrays["integer_gram"], proof["integer_gram"]), "canonical exact integer Gram")
    audit.close(arrays["S"], model.S, "canonical singlet basis", atol=1e-12)
    for row in report["rows"]:
        tag = f"canonical t{row['t']:g}"
        u, target = arrays[row["witness_key"]], arrays[row["target_key"]]
        c = full_tensor_reconstruction(u)
        error = c-target
        coordinates = model.S.conj().T@error
        audit.check(digest(u) == row["witness_sha256"], tag+" witness digest")
        audit.close(target, model.target(row["t"]), tag+" target", atol=1e-12)
        audit.close(c, arrays[f"full_C_t{row['t']:g}_k2"], tag+" full tensor C", atol=1e-12)
        audit.close(np.linalg.norm(error), row["full_reconstruction_residual"], tag+" full residual")
        audit.close(np.linalg.norm(coordinates), row["projected_residual"], tag+" projected residual")
        audit.close(np.linalg.norm(error-model.S@coordinates), row["outside_residual"], tag+" outside residual")
        audit.close(np.vdot(u, c).real, row["q"], tag+" Euler moment")
        audit.check(bool(row["reconstruction_accepted"]) == (row["optimizer_converged"]
                    and np.linalg.norm(error) <= report["reconstruction_tolerance"]
                    and np.linalg.norm(c-arrays[f"reduced_C_t{row['t']:g}_k2"]) <= report["structural_tolerance"]),
                    tag+" acceptance")
    for index, row in enumerate(report["gaussian_probes"]):
        g = arrays["gaussian_probes"][index]
        audit.close(full_tensor_reconstruction(g), g, f"Gaussian exact probe {index}", atol=1e-10)
        audit.check(row["full_reconstruction_error"] <= 1e-10, f"Gaussian probe gate {index}")
    audit.check(report["verified"] == all(row["verified"] for row in report["rows"]), "canonical aggregate verified")
    return arrays


def verify_frozen_and_training(out, config, manifest, model, canonical, audit):
    for seed in config["seeds"]:
        training = load_json(out/f"training_{seed}.json")
        arrays = load_npz(out/f"training_{seed}.npz")
        audit.check(training["rng_seed_sequence"] == [seed, 910, 1], f"training {seed} stream")
        cache = arrays["h_cache"]
        for row in training["rows"]:
            n = row["train_samples"]
            u = arrays[row["witness_key"]]
            z = model.S.conj().T@u
            overlap = cache[:n]@z
            weight = dimension(4)*abs(overlap)**6*overlap
            coordinates = np.mean(weight[:, None]*cache[:n].conj(), axis=0)
            target_z = model.S.conj().T@arrays[row["target_key"]]
            tag = f"training {seed} {row['witness_key']}"
            audit.check(digest(u) == row["witness_sha256"], tag+" digest")
            audit.close(np.linalg.norm(coordinates-target_z), row["gradient_norm"], tag+" empirical gradient")
            audit.close(dimension(4)*np.mean(abs(overlap)**8), row["q_train"], tag+" q")
            audit.close(np.mean(abs(weight)), row["mass_train"], tag+" B")
            audit.check(not row["heldout_validation_completed"] and not row["reconstruction_accepted"], tag+" no training acceptance")
        frozen = load_npz(out/f"frozen_{seed}.npz")
        h = hashlib.sha256()
        for cid, t, k, u in zip(frozen["ids"], frozen["t"], frozen["k"], frozen["witnesses"]):
            matched = [row for row in training["rows"] if row["t"] == t
                       and row["train_samples"] == max(config["sample_sizes"])]
            converged = True if k == 2 else bool(matched[0]["optimizer_converged"])
            metadata = {"id": str(cid), "t": int(t), "k": int(k),
                        "source": "deterministic_integer_moment" if k == 2 else "empirical_full_Haar",
                        "deterministic_verified": bool(k == 2), "optimizer_converged": converged}
            h.update(json.dumps(metadata, sort_keys=True).encode())
            h.update(np.asarray(u, np.complex128).tobytes())
            source = (canonical[f"witness_t{t:g}_k2"] if k == 2
                      else arrays[f"witness_t{t:g}_k4_n{max(config['sample_sizes'])}"])
            audit.close(u, source, f"freeze {seed} {cid}", atol=0, rtol=0)
        expected_hash = manifest["candidate_freezes"][str(seed)]["sha256"]
        audit.check(h.hexdigest() == expected_hash, f"freeze {seed} aggregate digest")
        audit.check(manifest["final_validation_plan_frozen"]["candidate_sha256"][str(seed)] == expected_hash,
                    f"freeze {seed} validation digest")


def verify_statistics(out, name, seed, config, model, audit):
    report = load_json(out/f"{name}_{seed}.json")
    a = load_npz(out/f"{name}_{seed}.npz")
    frozen = load_npz(out/f"frozen_{seed}.npz")
    frozen_by_id = {str(cid): u for cid, u in zip(frozen["ids"], frozen["witnesses"])}
    training_rows = load_json(out/f"training_{seed}.json")["rows"]
    phase = "development" if name == "development" else "final"
    audit.check(report["rng_seed_sequence"] == [seed, 910, 2 if phase == "development" else 4], f"{name} {seed} stream")
    audit.close(a["basis"], model.S, f"{name} {seed} basis", atol=1e-12)
    op = model.S.conj().T@model.mass_correlation_operator[np.ix_(model.even_masks, model.even_masks)]@model.S
    audit.close(a["coherent_operator"], op, f"{name} {seed} coherent operator", atol=1e-12)
    ids = [str(cid) for cid in a["candidate_ids"]]
    nfinal = int(a["completed_samples"])
    sizes = sorted({row["n"] for row in report["rows"]})
    delta = config["total_failure_probability"]/config["planned_statistical_events"]
    inv_names = ("eta", "mass_a", "mass_b", "mass_dot", "saturation_a", "saturation_b")
    for row in report["rows"]:
        i = ids.index(row["id"])
        n, k = row["n"], row["k"]
        p = f"n{n}_"
        tag = f"{name} {seed} {row['id']} n{n}"
        raw, rb = a[p+"raw_C"][i], a[p+"rb_C"][i]
        u, target = a["witnesses"][i], a["targets"][i]
        phi, halves = a[p+"group_phi"][i], a[p+"group_phi_halves"][i]
        cross = (halves[0].conj()@op@halves[1].T+halves[1].conj()@op@halves[0].T)/2
        plugin = phi.conj()@op@phi.T
        audit.check(digest(u) == row["witness_sha256"], tag+" witness digest")
        audit.check(digest(target) == row["target_sha256"], tag+" target digest")
        if not row["is_probe"]:
            audit.close(u, frozen_by_id[row["id"]], tag+" frozen witness identity", atol=0, rtol=0)
            audit.close(target, model.target(row["t"]), tag+" physical target identity", atol=1e-12)
            expected_convergence = True if k == 2 else next(r["optimizer_converged"]
                for r in training_rows if r["t"] == row["t"] and r["train_samples"] == max(config["sample_sizes"]))
            audit.check(row["optimizer_converged"] == expected_convergence, tag+" frozen optimizer state")
        else:
            audit.close(u, target, tag+" Gaussian calibration witness")
            audit.close(full_tensor_reconstruction(u), target, tag+" Gaussian calibration exact moment", atol=1e-10)
        audit.close(rb, model.S@(model.S.conj().T@raw), tag+" RB projection")
        audit.close(model.S@phi.sum(axis=0), rb, tag+" group additivity")
        audit.close(cross, a[p+"K_cross"][i], tag+" cross kernel")
        audit.close(plugin, a[p+"group_kernel_plugin"][i], tag+" plugin kernel")
        audit.close(cross, cross.conj().T, tag+" cross Hermiticity")
        audit.close(np.linalg.norm(raw-target), row["raw_residual"], tag+" raw residual")
        audit.close(np.linalg.norm(rb-target), row["rb_residual"], tag+" RB residual")
        audit.close(np.linalg.norm(raw-rb), row["outside_residual"], tag+" outside residual")
        audit.close(np.linalg.norm(phi.sum(axis=0)-model.S.conj().T@target), row["coherent_closure_residual"], tag+" coherent closure")
        audit.close(np.linalg.norm(halves[0]-halves[1], axis=1), row["group_half_disagreement"], tag+" half disagreement")
        audit.close(np.vdot(u, raw).real, row["q"], tag+" q Euler identity")
        audit.close(a[p+"histogram"][i].sum(), 1., tag+" histogram total")
        audit.close(a[p+"haar_histogram"].sum(), 1., tag+" Haar histogram total")
        audit.close(.5*abs(a[p+"histogram"][i]-a[p+"haar_histogram"]).sum(), row["histogram_tv_haar"], tag+" Haar TV")
        if sizes.index(n) > 0:
            prev = a[f"n{sizes[sizes.index(n)-1]}_histogram"][i]
            audit.close(.5*abs(a[p+"histogram"][i]-prev).sum(), row["histogram_tv_previous"], tag+" previous TV")
        reference = exact_metrics(target, model)
        metrics = exact_metrics(rb/np.linalg.norm(rb), model)
        audit.close(cross.sum().real, row["CQ_cross"], tag+" CQ cross")
        audit.close(metrics["mass_correlation"], row["CQ_plugin"], tag+" CQ plugin")
        audit.close(metrics["wick_deviation"], row["W_plugin"], tag+" Wick plugin")
        audit.close(abs(cross.sum().real-reference["mass_correlation"]), row["CQ_error"], tag+" CQ error")
        audit.close(abs(metrics["wick_deviation"]-reference["wick_deviation"]), row["W_error"], tag+" W error")
        second = dimension(k)**2/dimension(2*k-1)*np.linalg.norm(u)**(4*k-2)
        radius = math.sqrt(second/(delta*n))
        envelope = row["formal_envelopes"]
        audit.close(second, row["formal_budget"]["second_moment_bound"], tag+" second moment bound")
        audit.close(radius, envelope["radius"], tag+" formal radius")
        audit.close(delta, envelope["allocated_failure_probability_each"], tag+" failure allocation", atol=1e-15)
        audit.close(row["B"]+radius, envelope["B_upper"], tag+" mass upper envelope")
        audit.close(row["rb_residual"]+radius, envelope["C_error_upper"], tag+" C envelope")
        included_in_joint_budget = phase in ("final", "validation") and not row["is_probe"]
        audit.check(envelope["allocated_events_this_candidate"] == (5 if included_in_joint_budget else 0), tag+" event count")
        audit.check(envelope["included_in_joint_budget"] == included_in_joint_budget, tag+" joint budget membership")
        audit.check(envelope["C_certified_within_tolerance"] == (included_in_joint_budget
                    and row["rb_residual"]+radius <= config["reconstruction_tolerance"]), tag+" C certification")
        audit.check(row["local_reconstruction_passed"] == (row["optimizer_converged"]
                    and (row["deterministic_verified"] or row["rb_residual"] <= config["reconstruction_tolerance"])), tag+" local gate")
        audit.check(row["optimizer_converged"] or not row["local_reconstruction_passed"], tag+" failed optimizer cannot pass")
        if row["is_probe"]:
            b = dimension(k)/dimension(k-.5)
            qs = math.sqrt(max(0., dimension(k)**2/dimension(2*k)-1)/n)
            bs = math.sqrt(max(0., dimension(k)**2/dimension(2*k-1)-b*b)/n)
            cr = math.sqrt(max(0., dimension(k)**2/dimension(2*k-1)-1)/n)
            passed = abs(row["q"]-1) <= 5*qs+1e-12 and abs(row["B"]-b) <= 5*bs+1e-12 and row["raw_residual"] <= 6*cr+1e-12
            audit.check(row["probe_passed"] == passed, tag+" probe gate")
        if n == nfinal:
            audit.close(raw, a["raw_sum"][i]/n, tag+" raw sum")
            audit.close(rb, model.S@a["rb_sum"][i]/n, tag+" RB sum")
            audit.close(phi, a["group_sum"][i]/n, tag+" group sum")
            audit.close(halves, a["group_half_sum"][i]/a["half_counts"][:, None, None], tag+" half sum")
            mass = a["mass_sum"][i]
            msq = a["mass_sq_sum"][i]
            audit.close(mass/n, row["B"], tag+" mass sum")
            audit.close(a["q_sum"][i]/n, row["q"], tag+" q sum")
            audit.close(mass*mass/msq, row["ess"], tag+" ESS")
            audit.close(math.sqrt(max(0., (msq-mass*mass/n)/(n-1))/n), row["B_se"], tag+" mass SE")
            audit.close(a["hist_sum"][i].reshape(4, 4)/mass, a[p+"histogram"][i], tag+" histogram sum")
            hess = np.divide(a["hist_sum"][i]**2, a["hist_sq_sum"][i], out=np.zeros(16), where=a["hist_sq_sum"][i] > 0)
            audit.close(hess.reshape(4, 4), a[p+"histogram_ess"][i], tag+" bin ESS")
            audit.close(a["group_mass_sum"][i]/mass, row["group_mass_fractions"], tag+" group mass")
            gvar = np.maximum(0., (a["group_norm2_sum"][i]-n*np.sum(abs(phi)**2, axis=1))/(n-1))
            audit.close(np.sqrt(gvar/n), row["group_rmse"], tag+" group RMSE")
            for j, field in enumerate(inv_names):
                mean = a["inv_sum"][i, j]/mass
                audit.close(mean, row["mean_"+field], tag+" invariant "+field)
                variance = max(0., a["inv_w2x2_sum"][i, j]-2*mean*a["inv_w2_sum"][i, j]+mean*mean*msq)
                audit.close(math.sqrt(n/(n-1)*variance)/mass, row["invariant_se"][field], tag+" invariant SE "+field)
    audit.check(int(a["haar_counts"].sum()) == nfinal, f"{name} {seed} Haar count")
    audit.check(int(a["half_counts"].sum()) == nfinal, f"{name} {seed} half count")
    return report


def verify_discovery(out, seed, config, model, audit):
    report = load_json(out/f"discovery_{seed}.json")
    a = load_npz(out/f"discovery_{seed}.npz")
    audit.check(report["rng_seed_sequence"] == [seed, 910, 3], f"discovery {seed} stream")
    permutations = np.argmax(abs(model.gammas), axis=2)
    phases = np.take_along_axis(model.gammas, permutations[:, :, None], axis=2)[..., 0]
    for t in config["t_values"]:
        tag = f"discovery {seed} t{t:g}"
        tk = f"t{t:g}"
        u = a[tk+"_witness"]/np.linalg.norm(a[tk+"_witness"])
        trace = a[tk+"_guided_objective_trace_normalized_witness"]
        audit.check(np.min(np.diff(trace, axis=0)) >= -1e-10, tag+" monotone all-start objective")
        state = a[tk+"_haar_starts"][0].copy()
        angles = a[tk+"_guided_rotation_angles"][:, 0]
        pairs = a[tk+"_guided_rotation_pairs"]
        replay = [abs(np.vdot(u, state))**2]
        masks = model.even_masks
        for sweep in angles:
            for theta, (x, y) in zip(sweep, pairs):
                source = permutations[y, permutations[x, masks]]
                perm = np.searchsorted(masks, source)
                phase = phases[x, masks]*phases[y, permutations[x, masks]]
                state = np.cos(theta)*state+np.sin(theta)*state[perm]*phase
            replay.append(abs(np.vdot(u, state))**2)
        state /= np.linalg.norm(state)
        audit.close(replay, trace[:, 0], tag+" first-start objective replay", atol=1e-10)
        audit.close(state, a["guided_"+tk+"_new_raw_atoms"][0], tag+" first-start ket replay", atol=1e-10)
    for row in report["rows"]:
        key = row["key"]
        tag = f"discovery {seed} {key}"
        atoms = a[key+"_atoms"]
        cov = own_covariances(atoms, model)
        inv = own_invariants(cov, model)
        projected = model.S.conj().T@atoms.T
        coefficients = a[key+"_coefficients"]
        target = a[f"t{row['t']:g}_target"]
        reconstruction = model.S@(projected@coefficients)
        residual = target-reconstruction
        l1 = float(abs(coefficients).sum())
        state_error = row["target_diagnostics"]["target_state_error_estimate"]
        hamiltonian = model.full_hamiltonian(row["t"])
        full_target = model.embed(target)
        energy = np.vdot(full_target, hamiltonian@full_target).real
        eigenvalues = np.linalg.eigvalsh(hamiltonian)
        eigen_residual = np.linalg.norm(hamiltonian@full_target-energy*full_target)
        estimated_state_error = math.sqrt(2)*eigen_residual/(eigenvalues[1]-energy)
        audit.close(estimated_state_error, state_error, tag+" target eigenpair error", atol=1e-12)
        cost = l1+abs(residual).sum()+math.sqrt(128)*state_error
        schmidt = np.linalg.svd(model.embed(target).reshape(16, 16), compute_uv=False).sum()
        audit.close(np.linalg.norm(atoms, axis=1), np.ones(len(atoms)), tag+" Gaussian norms", atol=1e-10)
        audit.close(cov, a[key+"_covariances"], tag+" covariance from independent Majoranas", atol=1e-10)
        audit.close(cov@cov, -np.tile(np.eye(16), (len(atoms), 1, 1)), tag+" covariance purity", atol=1e-9)
        audit.close(atoms[128:], a[key+"_new_raw_atoms"]*a[key+"_phases"][:, None], tag+" phase alignment")
        audit.close(projected, a[key+"_projected_atoms"], tag+" unnormalized projections")
        audit.close(reconstruction, a[key+"_reconstruction"], tag+" reconstruction")
        audit.close(residual, a[key+"_residual"], tag+" residual")
        audit.close(l1, row["coefficient_l1"], tag+" coefficient mass")
        audit.close(cost, row["cost_upper"], tag+" repaired upper")
        audit.close(schmidt, row["schmidt_norm"], tag+" Schmidt norm")
        for field, value in inv.items():
            audit.close(value, a[key+"_"+field], tag+" invariant "+field)
        for certificate in row["certificates"]:
            eta = certificate["eta0"]
            lower_l = max(0., schmidt-4*state_error)
            atomcap = (1+math.sqrt(eta))**2
            lower = lower_l/atomcap
            ratio = lower_l/cost
            audit.close(cost, certificate["cost_upper"], tag+f" eta{eta} upper")
            audit.close(lower, certificate["restricted_cost_lower"], tag+f" eta{eta} lower")
            audit.close(max(math.sqrt(ratio)-1, 0)**2, certificate["mean_eta_necessary_lower"], tag+f" eta{eta} necessary mean")
            audit.close(max((ratio-atomcap)/(4-atomcap), 0), certificate["eta_above_threshold_mass_fraction_lower"], tag+f" eta{eta} necessary tail")
            margin = certificate["numerical_margin"]
            audit.check(certificate["numerical_separation"] == (cost+margin < lower-margin), tag+f" eta{eta} separation")
            audit.check(not certificate["strict_interval_certified"], tag+f" eta{eta} evidence level")


def verify_acceptance(out, config, development, validation, audit):
    saved = load_json(out/"acceptance.json")
    summary = load_json(out/"summary.json")
    final = [dict(row, seed=seed) for seed, doc in validation.items() for row in doc["rows"] if row["n"] == config["final_samples"]]
    dev = [dict(row, seed=seed) for seed, doc in development.items() for row in doc["rows"] if row["n"] == max(config["sample_sizes"])]
    probes = [row for row in final if row["is_probe"]]
    expected_probes = {(seed, f"probe_k{k}") for seed in config["seeds"] for k in (2, 4)}
    probes_ok = (len(probes) == len(expected_probes)
                 and {(row["seed"], row["id"]) for row in probes} == expected_probes
                 and all(row["probe_passed"] for row in probes))
    counts = [0, 0, 0]
    for row in saved["rows"]:
        rs = [r for r in final if r["id"] == row["id"] and not r["is_probe"]]
        ds = [r for r in dev if r["id"] == row["id"] and not r["is_probe"]]
        complete = (len(rs) == len(config["seeds"]) and {r["seed"] for r in rs} == set(config["seeds"])
                    and len(ds) == len(config["seeds"]) and {r["seed"] for r in ds} == set(config["seeds"]))
        reco = bool(complete and all(r["local_reconstruction_passed"] for r in rs))
        stable = bool(complete and all(r["histogram_tv_previous"] is not None and r["histogram_tv_previous"] <= config["histogram_tv_max"] for r in ds)
                      and all(r["ess"] >= config["mass_ess_min"] for r in rs)
                      and np.ptp([r["mean_eta"] for r in rs]) <= config["eta_seed_range_max"])
        distribution = bool(probes_ok and reco and stable)
        coherent = bool(distribution and all(r["coherent_closure_residual"] <= config["reconstruction_tolerance"]
                    and r["CQ_error"] <= config["observable_tolerance"] and r["W_error"] is not None
                    and r["W_error"] <= config["observable_tolerance"]
                    and max(r["group_rmse"]) <= config["group_rmse_max"]
                    and max(r["group_half_disagreement"]) <= config["group_half_disagreement_max"] for r in rs))
        for key, expected in (("reconstruction_passed", reco), ("distribution_stable", stable),
                              ("distribution_interpretation_allowed", distribution), ("coherent_interpretation_allowed", coherent)):
            audit.check(row[key] == expected, f"acceptance {row['id']} {key}")
        counts = [counts[0]+int(reco), counts[1]+int(distribution), counts[2]+int(coherent)]
    for key, expected in zip(("reconstruction_passed_count", "distribution_allowed_count", "coherent_allowed_count"), counts):
        audit.check(saved[key] == expected, "acceptance "+key)
        audit.check(summary[key] == expected, "summary "+key)
    if "statistically_certified_count" in summary:
        expected = sum(bool(row["formal_envelopes"]["C_certified_within_tolerance"]) for row in final if not row["is_probe"])
        audit.check(summary["statistically_certified_count"] == expected, "summary statistical count")
    allocated = sum(r["formal_envelopes"]["allocated_events_this_candidate"]
                    *r["formal_envelopes"]["allocated_failure_probability_each"] for r in final)
    audit.check(allocated <= config["total_failure_probability"]+1e-14, "joint final failure budget")
    audit.check(config["planned_statistical_events"] >= 5*len(config["seeds"])*(len(config["t_values"])+len(config["k4_t_values"])), "predeclared event capacity")


def verify_sources(manifest, audit):
    for field, root, unchanged in (("source_sha256", ROOT, "source_unchanged_during_run"),
                                   ("backend_sha256", BACKEND, "backend_unchanged_during_run"),
                                   ("followup_sha256", FOLLOWUP, "followup_unchanged_during_run")):
        audit.check(manifest.get(unchanged, False), unchanged)
        for relative, expected in manifest[field].items():
            path = root/relative
            audit.check(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                        f"source {root.name}/{relative}")


def verify(output):
    out = Path(output).resolve()
    started = time.monotonic()
    audit = Audit()
    config, manifest = load_json(out/"config.json"), load_json(out/"manifest.json")
    model = build_model()
    audit.block("source provenance", lambda: verify_sources(manifest, audit))
    canonical = {}
    audit.block("canonical", lambda: canonical.update(verify_canonical(out, model, audit)))
    audit.block("frozen and training", lambda: verify_frozen_and_training(out, config, manifest, model, canonical, audit))
    development, validation = {}, {}
    for seed in config["seeds"]:
        audit.block(f"development {seed}", lambda s=seed: development.update({s: verify_statistics(out, "development", s, config, model, audit)}))
        audit.block(f"validation {seed}", lambda s=seed: validation.update({s: verify_statistics(out, "validation", s, config, model, audit)}))
        audit.block(f"discovery {seed}", lambda s=seed: verify_discovery(out, s, config, model, audit))
    audit.block("acceptance", lambda: verify_acceptance(out, config, development, validation, audit))
    result = {"status": "passed" if not audit.failures else "failed", "checks": audit.checks,
              "failures": audit.failures, "elapsed_seconds": time.monotonic()-started,
              "strict_interval_certified": False,
              "scope": "Saved-array algebra, sufficient statistics, all-seed gates, provenance, Gaussian atoms and one rotation trajectory per seed/t; no optimization or Monte Carlo rerun.",
              "limits": ["Original Haar draws are not regenerated; sufficient-statistic origins are not independently certified.",
                         "Integer k2 matrix identity is exact; eigensystems, Newton solutions and remaining arithmetic use floating point.",
                         "Covariance purity and orbit costs are numerical evidence, not directed-rounding enclosures.",
                         "Empirical variance estimates and acceptance gates are not formal confidence guarantees."]}
    save_json(out/"verification.json", result)
    return result


def self_test():
    """Test that the verifier rejects tampering and uses complex coherence."""
    a = Audit()
    a.close(np.array([1+2j]), np.array([1+2j]), "match")
    a.close(np.array([1+2j]), np.array([1+2.01j]), "tampered imaginary coefficient")
    a.close(np.array([1.]), np.array([np.nan]), "nonfinite evidence")
    assert a.checks == 3 and len(a.failures) == 2
    left = np.array([[1, 1j], [.3j, .4]], complex)
    right = np.array([[.8, -.2j], [.2, .5j]], complex)
    operator = np.array([[1, .3j], [-.3j, 2]], complex)
    kernel = (left.conj()@operator@right.T+right.conj()@operator@left.T)/2
    assert np.allclose(kernel, kernel.conj().T)
    assert not np.allclose(kernel, left.conj()@operator@left.T)
    assert digest(np.array([1j])) != digest(np.array([-1j]))
    print("Verifier self-test passed: numerical tampering, nonfinite values, complex kernel, witness phase digest.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.output:
        parser.error("output directory is required unless --self-test is selected")
    result = verify(args.output)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
