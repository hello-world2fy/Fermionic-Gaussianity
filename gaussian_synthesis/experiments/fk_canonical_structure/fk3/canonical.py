"""Frozen-source audit and canonical witnesses for the structure experiment.

The previous packages are imported read-only.  The k=2 backend is an algebraic
Haar moment; coefficient mass remains an independent half-integer integral.
An exact integer matrix identity is not an interval certificate for a floating
Newton solution.  The k=4 training stream never doubles as held-out validation.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from fk.dictionaries import Dictionary
from fk.moments import ProjectedMoments
from fk.runtime import ResourceLimit
from fk.solver import solve
from fk.validation import gaussian_probe_reference
from fk2.exact_moments import (
    ExactK2Moments, _majorana_actions, apply_lambda, closed_form_fk_matrix,
    full_tensor_reconstruction, integer_fk_matrix_proof, project_gaussian_pair,
)
from fk2.rb import hopping_gaussian_probe


def _check(budget):
    if budget is not None:
        budget.check()


def _usage(budget, started):
    return budget.usage() if budget is not None else {"seconds": time.monotonic()-started}


def _digest(vector):
    return hashlib.sha256(np.asarray(vector, dtype=np.complex128).tobytes()).hexdigest()


def _history(solution):
    return np.array([[row["iteration"], row["objective"], row["gradient_norm"]]
                     for row in solution["history"]], dtype=float).reshape(-1, 3)


def _key(t):
    return f"t{float(t):g}"


def _options(config):
    return {
        "tolerance": float(config.get("gradient_tol", config.get("gradient_tolerance", 1e-10))),
        "max_steps": int(config.get("maxsteps", config.get("max_newton_steps", 200))),
    }


def _majorana_audit(model):
    """Compare signed-permutation pair backend with the full model matrices."""
    sectors = [np.array([i for i in range(256) if i.bit_count() % 2 == p])
               for p in (0, 1)]
    largest = 0.
    for parity in (0, 1):
        permutations, phases = _majorana_actions(parity)
        for a in range(16):
            reconstructed = np.zeros((128, 128), complex)
            reconstructed[np.arange(128), permutations[a]] = phases[a]
            dense = model.gammas[a][np.ix_(sectors[1-parity], sectors[parity])]
            largest = max(largest, float(np.linalg.norm(reconstructed-dense)))
    return largest


def _derivative_audit(backend, model):
    """Fixed complex probes independent of physical target solving."""
    rng = np.random.default_rng(np.random.SeedSequence([20261024, 910, 0]))
    rows = []
    for index in range(3):
        z = rng.normal(size=3) + 1j*rng.normal(size=3)
        z /= np.linalg.norm(z)
        u = np.einsum("ij,j->i", model.S, z, optimize=False)
        direct = full_tensor_reconstruction(u)
        reduced = backend.reconstruction(z)
        value = backend.evaluate(z)
        step = 1e-5
        x = np.concatenate((z.real, z.imag))
        finite_gradient = np.empty(6)
        finite_hessian = np.empty((6, 6))
        for j in range(6):
            delta = np.eye(6)[j]*step
            xp, xm = x+delta, x-delta
            plus = backend.evaluate(xp[:3]+1j*xp[3:])
            minus = backend.evaluate(xm[:3]+1j*xm[3:])
            finite_gradient[j] = (plus.q-minus.q)/(8*step)
            finite_hessian[:, j] = (plus.gradient-minus.gradient)/(2*step)
        rows.append({
            "probe": index,
            "full_tensor_vs_reduced_error": float(np.linalg.norm(direct-reduced)),
            "q_euler_error": abs(value.q-float(np.vdot(u, direct).real)),
            "gradient_finite_difference_error": float(np.linalg.norm(value.gradient-finite_gradient)),
            "hessian_finite_difference_error": float(np.linalg.norm(value.hessian-finite_hessian)),
            "hessian_min_eigenvalue": float(np.linalg.eigvalsh(value.hessian)[0]),
        })
    return rows


def _gaussian_audit(model, backend):
    """Check invariant and non-invariant Gaussian probes in full tensor space."""
    invariant, metadata = hopping_gaussian_probe(model)
    vacuum = np.zeros(128, complex)
    vacuum[0] = 1.
    rng = np.random.default_rng(np.random.SeedSequence([20261024, 910, 8]))
    random_gaussians = Dictionary("full").sample(3, rng)
    vectors = [invariant, vacuum, *random_gaussians.kets]
    labels = ["hopping_ground_state", "fock_vacuum", "haar_0", "haar_1", "haar_2"]
    rows = []
    for label, g in zip(labels, vectors):
        pair = np.outer(g, g)
        projection = project_gaussian_pair(pair)
        reconstruction = full_tensor_reconstruction(g)
        row = {
            "label": label, "normalization_error": abs(float(np.vdot(g, g).real)-1.),
            "lambda_annihilation_error": float(np.linalg.norm(apply_lambda(pair))),
            "projector_error": float(np.linalg.norm(projection-pair)),
            "full_reconstruction_error": float(np.linalg.norm(reconstruction-g)),
            "scaled_q_error": abs(float(np.vdot(pair, projection).real)-1.),
        }
        if label == "hopping_ground_state":
            z = np.einsum("ij,i->j", model.S.conj(), g, optimize=False)
            row["reduced_q_error"] = abs(backend.evaluate(z, hessian=False).q-1.)
            row.update(metadata)
        rows.append(row)
    return rows, np.array(vectors), gaussian_probe_reference(Dictionary("full"), 2)


def prepare_canonical(model, config, budget, checkpoint=None):
    """Audit the reused exact backend and solve all predeclared k=2 targets.

    Returns JSON-compatible diagnostics and raw NumPy arrays.  Every accepted
    target is independently reconstructed through all 128 tensor coordinates.
    The ``matrix`` array is D_(8,2) times the Haar moment, not the raw moment.
    """
    started = time.monotonic()
    structural_tol = float(config.get("structural_tol", config.get("structural_tolerance", 1e-10)))
    reconstruction_tol = float(config.get("reconstruction_tol", 1e-8))
    t_values = [float(t) for t in config.get("t_values", [0, 5, 9, 11, 20])]
    if not t_values or len(t_values) != len(set(t_values)) or not np.all(np.isfinite(t_values)):
        raise ValueError("t_values must be distinct finite values")
    report = {
        "status": "running", "dictionary": "full", "k": 2,
        "rows": [], "planned_t_values": t_values, "verified": False,
        "strict_interval_certified": False, "uses_monte_carlo_for_canonical_moment": False,
        "coefficient_mass_status": "not_determined_by_integer_k2_moment",
        "audit_rng_seed_sequences": [[20261024, 910, 0], [20261024, 910, 8]],
        "interpretation": "Algebraic Haar moment with signed-integer coefficient proof; floating Newton and independent full-space evaluation. No interval or optimal-atomic-cost claim.",
    }
    arrays = {"S": model.S.copy()}
    try:
        _check(budget)
        backend = ExactK2Moments(model)
        proof = integer_fk_matrix_proof()
        report["backend_diagnostics"] = backend.diagnostics
        report["audit"] = {
            "integer_closed_form_proof_verified": bool(proof["verified"]),
            "integer_arithmetic": proof["integer_arithmetic"],
            "basis_error": float(np.linalg.norm(proof["basis"]-model.S)),
            "integer_vs_tensor_matrix_error": float(np.linalg.norm(proof["matrix"]-backend.matrix)),
            "closed_form_matrix_error": float(np.linalg.norm(proof["matrix"]-closed_form_fk_matrix())),
            "majorana_action_error": _majorana_audit(model),
        }
        arrays.update(matrix=backend.matrix.copy(), haar_moment_matrix=backend.haar_moment_matrix.copy(),
                      integer_gram=proof["integer_gram"], integer_gram_expected=proof["integer_gram_expected"],
                      closed_form_matrix=closed_form_fk_matrix(), radical_basis=proof["basis"])
        _check(budget)
        report["derivative_probes"] = _derivative_audit(backend, model)
        report["gaussian_probes"], arrays["gaussian_probes"], report["gaussian_exact_reference"] = _gaussian_audit(model, backend)
        audit = report["audit"]
        errors = [audit[name] for name in ("basis_error", "integer_vs_tensor_matrix_error",
                                          "closed_form_matrix_error", "majorana_action_error")]
        errors += [backend.diagnostics[name] for name in ("orthogonality_error", "moment_hermiticity_error",
                                                          "lambda_kernel_residual", "projector_idempotence_residual")]
        errors += [row[name] for row in report["derivative_probes"]
                   for name in ("full_tensor_vs_reduced_error", "q_euler_error")]
        errors += [row[name] for row in report["gaussian_probes"]
                   for name in ("normalization_error", "lambda_annihilation_error", "projector_error",
                                "full_reconstruction_error", "scaled_q_error")]
        derivative_ok = all(row["gradient_finite_difference_error"] < 1e-8
                            and row["hessian_finite_difference_error"] < 1e-8
                            and row["hessian_min_eigenvalue"] >= -structural_tol
                            for row in report["derivative_probes"])
        report["audit_passed"] = bool(proof["verified"] and max(errors) <= structural_tol and derivative_ok)
        report["structural_tolerance"] = structural_tol
        report["reconstruction_tolerance"] = reconstruction_tol
        if not report["audit_passed"]:
            report["status"] = "audit_failed"
        else:
            for t in t_values:
                _check(budget)
                target = model.target(t)
                reduced_target = np.einsum("ij,i->j", model.S.conj(), target, optimize=False)
                solution = solve(backend, reduced_target, budget=budget, **_options(config))
                z = solution["z"]
                u = np.einsum("ij,j->i", model.S, z, optimize=False)
                direct = full_tensor_reconstruction(u)
                reduced = backend.reconstruction(z)
                error = direct-target
                projection = np.einsum("ij,i->j", model.S.conj(), error, optimize=False)
                outside = error-np.einsum("ij,j->i", model.S, projection, optimize=False)
                q = backend.evaluate(z, hessian=False).q
                name = _key(t)
                witness_key = f"witness_{name}_k2"
                arrays.update({witness_key: u, f"target_{name}": target,
                               f"z_{name}_k2": z, f"full_C_{name}_k2": direct,
                               f"reduced_C_{name}_k2": reduced, f"history_{name}_k2": _history(solution)})
                residual = float(np.linalg.norm(error))
                closure = float(np.linalg.norm(direct-reduced))
                accepted = bool(solution["converged"] and residual <= reconstruction_tol and closure <= structural_tol)
                report["rows"].append({
                    "t": t, "k": 2, "dictionary": "full", "witness_key": witness_key,
                    "target_key": f"target_{name}", "witness_sha256": _digest(u),
                    "optimizer_converged": bool(solution["converged"]), "optimizer_status": solution["status"],
                    "gradient_norm": solution["gradient_norm"], "iterations": solution["iterations"],
                    "q": q, "witness_norm": float(np.linalg.norm(u)),
                    "full_reconstruction_residual": residual,
                    "projected_residual": float(np.linalg.norm(projection)),
                    "outside_residual": float(np.linalg.norm(outside)),
                    "full_tensor_vs_reduced_error": closure,
                    "euler_error": abs(q-float(np.vdot(u, direct).real)),
                    "reconstruction_accepted": accepted, "verified": accepted,
                    "coefficient_mass": None, "strict_interval_certified": False,
                })
                report["usage"] = _usage(budget, started)
                if checkpoint:
                    checkpoint(report, arrays)
            report["verified"] = bool(len(report["rows"]) == len(t_values)
                                      and all(row["verified"] for row in report["rows"]))
            report["status"] = "complete" if report["verified"] else "reconstruction_failed"
    except ResourceLimit as error:
        report.update(status="resource_limit", error=str(error), verified=False)
    report["usage"] = _usage(budget, started)
    if checkpoint:
        checkpoint(report, arrays)
    return report, arrays


def train_k4(model, config, seed, budget, checkpoint=None):
    """Train k=4 on fresh nested Haar prefixes; never claim held-out acceptance."""
    started = time.monotonic()
    sizes = [int(n) for n in config.get("sample_sizes", [4096, 16384, 65536])]
    if not sizes or sizes != sorted(set(sizes)) or sizes[0] < 1 or sizes[-1] > 65536:
        raise ValueError("sample_sizes must be distinct increasing values in [1,65536]")
    t_values = [float(t) for t in config.get("k4_t_values", [0, 11, 20])]
    if not t_values or len(t_values) != len(set(t_values)) or not np.all(np.isfinite(t_values)):
        raise ValueError("k4_t_values must be distinct finite values")
    batch_size = int(config.get("batch_size", 256))
    if batch_size < 1 or batch_size > 256:
        raise ValueError("batch_size must lie in [1,256]")
    labels = [int(seed), 910, 1]
    rng = np.random.default_rng(np.random.SeedSequence(labels))
    dictionary = Dictionary("full")
    h = np.empty((sizes[-1], 3), complex)
    arrays = {"h_cache": h[:0].copy(), "sample_count": np.array(0)}
    report = {"status": "running", "seed": int(seed), "dictionary": "full", "k": 4,
              "planned_sample_sizes": sizes, "planned_t_values": t_values, "rows": [],
              "rng_seed_sequence": labels, "rng_state": rng.bit_generator.state,
              "completed_samples": 0, "initialization": "deterministic_k2_then_previous_sample_prefix",
              "heldout_validation_completed": False, "statistically_certified": False}
    count = 0
    try:
        _check(budget)
        exact = ExactK2Moments(model, verify=False)
        previous = {}
        for t in t_values:
            target = model.target(t)
            reduced_target = np.einsum("ij,i->j", model.S.conj(), target, optimize=False)
            initial = solve(exact, reduced_target, budget=budget, **_options(config))
            if not initial["converged"]:
                raise ArithmeticError("deterministic k=2 initialization failed")
            previous[t] = initial["z"]
            arrays[f"target_{_key(t)}"] = target
        for size in sizes:
            while count < size:
                _check(budget)
                batch = dictionary.sample(min(batch_size, size-count), rng)
                h[count:count+len(batch.kets)] = np.einsum("ni,ij->nj", batch.kets.conj(), model.S, optimize=False)
                count += len(batch.kets)
                if count % 4096 == 0 or count == size:
                    arrays.update(h_cache=h[:count].copy(), sample_count=np.array(count))
                    report.update(completed_samples=count, rng_state=rng.bit_generator.state, usage=_usage(budget, started))
                    if checkpoint:
                        checkpoint(report, arrays)
            backend = ProjectedMoments(h[:size], 4, dictionary.log_frame_constant(4))
            for t in t_values:
                _check(budget)
                target = arrays[f"target_{_key(t)}"]
                reduced_target = np.einsum("ij,i->j", model.S.conj(), target, optimize=False)
                solution = solve(backend, reduced_target, initial=previous[t], budget=budget, **_options(config))
                z = solution["z"]
                previous[t] = z
                u = np.einsum("ij,j->i", model.S, z, optimize=False)
                values = backend.evaluate(z, hessian=False)
                name = f"{_key(t)}_k4_n{size}"
                witness_key = f"witness_{name}"
                arrays.update({witness_key: u, f"z_{name}": z, f"history_{name}": _history(solution)})
                report["rows"].append({
                    "t": t, "k": 4, "seed": int(seed), "dictionary": "full", "train_samples": size,
                    "witness_key": witness_key, "target_key": f"target_{_key(t)}", "witness_sha256": _digest(u),
                    "optimizer_converged": bool(solution["converged"]), "optimizer_status": solution["status"],
                    "gradient_norm": solution["gradient_norm"], "iterations": solution["iterations"],
                    "q_train": values.q, "mass_train": values.mass, "training_mass_ess": values.ess,
                    "witness_norm": float(np.linalg.norm(u)), "heldout_validation_completed": False,
                    "gradient_kind": "empirical_projected", "reconstruction_accepted": False,
                })
                report["usage"] = _usage(budget, started)
                if checkpoint:
                    checkpoint(report, arrays)
        report["status"] = "trained" if all(row["optimizer_converged"] for row in report["rows"]) else "optimizer_failed"
    except ResourceLimit as error:
        report.update(status="resource_limit", error=str(error))
    arrays.update(h_cache=h[:count].copy(), sample_count=np.array(count))
    report.update(completed_samples=count, rng_state=rng.bit_generator.state, usage=_usage(budget, started))
    if checkpoint:
        checkpoint(report, arrays)
    return report, arrays
