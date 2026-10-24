"""Independent dual-feedback column generation and explicit Gaussian repairs.

The pricing oracle is multistart local Gaussian optimization. Its maximum is
a LOWER estimate of the support function and is never used to certify global
feasibility. A separate interval q2 normalization supplies a weaker, valid
full-dictionary witness. Orbit-average costs and actual finite ket costs are
reported separately.
"""
import time
import numpy as np
import cvxpy as cp

from . import paths
from .certificates import certify_order2_normalization
from fk.gaussian import sample_gaussian
from fk3.discovery import optimize_gaussian_overlap, covariance_from_kets, physics_arc_atoms
from fk2.orbits import DiagonalSpin7, simplex_directions
from .finite_twirl import FiniteSpin7Twirl


def fit_orbits(basis, atoms, target):
    projected = basis.conj().T@np.asarray(atoms).T
    z = basis.conj().T@target
    c = cp.Variable(len(atoms), complex=True)
    constraint = projected@c == z
    problem = cp.Problem(cp.Minimize(cp.sum(cp.abs(c))), [constraint])
    problem.solve(solver="CLARABEL", max_iter=250, max_threads=1,
                  tol_gap_abs=2e-10, tol_gap_rel=2e-10, tol_feas=2e-10)
    if c.value is None or constraint.dual_value is None:
        raise RuntimeError(f"orbit SOCP failed: {problem.status}")
    coefficient = np.asarray(c.value).ravel()
    dual = -np.asarray(constraint.dual_value).ravel()
    residual = target-basis@(projected@coefficient)
    cost = float(abs(coefficient).sum()+abs(residual).sum())
    return {"coefficients": coefficient, "dual_z": dual, "residual": residual,
            "U_orbit_repaired": cost, "solver_status": str(problem.status),
            "restricted_dual_objective": float(np.vdot(dual, z).real),
            "finite_constraint_max": float(max(abs(projected.conj().T@dual)))}


def column_generation(model, t, seed, *, starts=32, iterations=20, sweeps=40,
                      pricing_tolerance=2e-6):
    started = time.monotonic()
    target = model.target(t)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 501, int(round(t*10000))]))
    haar = sample_gaussian(8, starts, rng).kets
    atoms = np.concatenate((np.eye(128, dtype=complex), haar))
    history = []
    generated = []
    warm = haar
    for iteration in range(iterations):
        fit = fit_orbits(model.S, atoms, target)
        witness = model.S@fit["dual_z"]
        fresh = sample_gaussian(8, starts, rng).kets
        candidates, search = optimize_gaussian_overlap(witness, np.concatenate((warm, fresh)),
                                                        max_sweeps=sweeps, gradient_tolerance=1e-9)
        overlaps = abs(candidates.conj()@witness)
        order = np.argsort(overlaps)[::-1]
        best = candidates[order[:min(8, len(candidates))]]
        cov = covariance_from_kets(best)
        purity = float(np.max(abs(cov@cov+np.eye(16))))
        if purity >= 1e-9:
            raise ArithmeticError("pricing oracle produced a non-Gaussian ket")
        history.append({"iteration": iteration, "candidate_count": len(atoms),
                        "U_orbit_repaired": fit["U_orbit_repaired"],
                        "finite_constraint_max": fit["finite_constraint_max"],
                        "restricted_dual_objective": fit["restricted_dual_objective"],
                        "local_pricing_max": float(overlaps.max()),
                        "pricing_is_global_upper_bound": False,
                        "search_seconds": search["elapsed_seconds"],
                        "purity_error_new_candidates": purity,
                        "elapsed_seconds": time.monotonic()-started})
        generated.append(best)
        if overlaps.max() <= 1+pricing_tolerance:
            break
        atoms = np.concatenate((atoms, best))
        warm = candidates[order[:starts]]
    fit = fit_orbits(model.S, atoms, target)
    certificate = certify_order2_normalization(fit["dual_z"], target)
    certified_z = certificate.pop("z_feasible")
    return {"t": t, "seed": seed, "history": history,
            "U_orbit_repaired": fit["U_orbit_repaired"],
            "pricing_stationarity_reached": history[-1]["local_pricing_max"] <= 1+pricing_tolerance,
            "full_dictionary_optimum_certified": False,
            "certificate": certificate, "elapsed_seconds": time.monotonic()-started}, {
                "atoms": atoms, "coefficients": fit["coefficients"],
                "dual_z": fit["dual_z"], "certified_z": certified_z,
                "target": target, "residual": fit["residual"],
                "generated_pricing_atoms": np.concatenate(generated)}


class _NoBudget:
    def check(self):
        return None


def independent_arc_dictionary(model, count=129):
    """Explicit prior baseline, separate from the adaptive pricing oracle."""
    return physics_arc_atoms(model, count, _NoBudget())


def finite_simplex_expansion(model, atoms, coefficients, target, *, threshold=1e-8):
    """Try the known eight-direction design, then repair all actual residuals.

No assumption that this design twirls arbitrary atoms is made. The measured
twirl errors are saved, and the finite expansion is always priced with its
full Fock correction, even when that error is large.
"""
    symmetry = DiagonalSpin7(model)
    directions = simplex_directions()
    active = np.flatnonzero(abs(coefficients) > threshold)
    branches, values, errors = [], [], []
    for j in active:
        orbit = np.array([symmetry.act(atoms[j], direction)[0] for direction in directions])
        exact_projection = model.S@(model.S.conj().T@atoms[j])
        errors.append(float(np.linalg.norm(orbit.mean(axis=0)-exact_projection)))
        branches.extend(orbit)
        values.extend(np.full(len(orbit), coefficients[j]/len(orbit)))
    branches = np.asarray(branches, complex).reshape(-1, 128)
    values = np.asarray(values, complex)
    residual = target-values@branches
    cov = covariance_from_kets(branches) if len(branches) else np.empty((0, 16, 16))
    purity = float(np.max(abs(cov@cov+np.eye(16)))) if len(branches) else 0.
    if purity >= 1e-9:
        raise ArithmeticError("finite branch Gaussianity failed")
    return {"U_finite_repaired": float(abs(values).sum()+abs(residual).sum()),
            "main_gaussian_branch_count": len(branches),
            "fock_repair_count_nonzero": int(np.count_nonzero(residual)),
            "fock_repair_count_above_1e_12": int(np.sum(abs(residual) > 1e-12)),
            "fock_repair_l1": float(abs(residual).sum()),
            "raw_reconstruction_residual": float(np.linalg.norm(residual)),
            "twirl_errors": errors, "gaussian_purity_error": purity,
            "discard_threshold": threshold,
            "discarded_coefficient_mass": float(abs(coefficients[abs(coefficients) <= threshold]).sum()),
            "finite_cost_includes_all_fock_repair": True,
            "global_optimality_certified": False}, {
                "finite_kets": branches, "finite_coefficients": values,
                "fock_repair": residual, "finite_covariances": cov, "active_orbit_indices": active}


def finite_group_expansion(model, atoms, coefficients, target, *, threshold=1e-8, twirl=None):
    """Realize arbitrary retained orbit representatives as actual Gaussian kets."""
    twirl = twirl or FiniteSpin7Twirl(model)
    active = np.flatnonzero(abs(coefficients) > threshold)
    branches, values, info = [], [], []
    for j in active:
        orbit, diagnostic = twirl.orbit(atoms[j])
        branches.append(orbit)
        values.append(np.full(len(orbit), coefficients[j]/len(orbit)))
        info.append({"source_index": int(j), **diagnostic})
    branches = np.concatenate(branches) if branches else np.empty((0, 128), complex)
    values = np.concatenate(values) if values else np.empty(0, complex)
    residual = target-values@branches
    covariances = []
    for start in range(0, len(branches), 256):
        covariances.append(covariance_from_kets(branches[start:start+256]))
    cov = np.concatenate(covariances) if covariances else np.empty((0, 16, 16))
    purity = float(np.max(abs(cov@cov+np.eye(16)))) if len(cov) else 0.
    if purity >= 1e-9:
        raise ArithmeticError("finite group branches fail Gaussian purity")
    return {"U_finite_repaired": float(abs(values).sum()+abs(residual).sum()),
            "main_gaussian_branch_count": len(branches),
            "fock_repair_count_nonzero": int(np.count_nonzero(residual)),
            "fock_repair_count_above_1e_12": int(np.sum(abs(residual) > 1e-12)),
            "fock_repair_l1": float(abs(residual).sum()),
            "raw_reconstruction_residual": float(np.linalg.norm(residual)),
            "orbit_realizations": info, "full_twirl_operator_error": twirl.operator_error,
            "gaussian_purity_error": purity, "finite_cost_includes_all_fock_repair": True,
            "discarded_coefficient_mass": float(abs(coefficients[abs(coefficients) <= threshold]).sum()),
            "strict_interval_certified": False, "global_optimality_certified": False}, {
                "finite_kets": branches, "finite_coefficients": values, "fock_repair": residual,
                "finite_covariances": cov, "active_orbit_indices": active}
