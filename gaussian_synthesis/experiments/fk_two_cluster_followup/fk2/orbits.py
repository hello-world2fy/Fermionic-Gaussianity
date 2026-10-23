"""Explicit finite Gaussian sums from the frozen FK symmetry-orbit baseline.

The primary construction uses eight vertices of a regular simplex in R^7.
Their spherical 2-design property is only a motivation: every retained source
atom's finite average is checked against S S^dagger g in all 128 coordinates.
No claim is made that this quadrature twirls an arbitrary Gaussian atom.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import eigh

from fk.baselines import target_error_estimate
from fk.certificates import separation_certificate
from fk.conventions import quadratic_operator
from fk.gaussian import _bivector_action
from fk.observables import exact_metrics


def simplex_directions():
    """Eight unit vectors with zero mean and second moment I_7/7."""
    differences = np.eye(8)[:, :7]-np.eye(8)[:, 7, None]
    orthogonal, _ = np.linalg.qr(differences, mode="reduced")
    return math.sqrt(8.0/7.0)*orthogonal


def covariance_from_kets(kets):
    """Actual ket covariances, using sparse Majorana permutation actions."""
    g = np.asarray(kets, dtype=complex)
    if g.ndim == 1:
        g = g[None, :]
    if g.ndim != 2 or g.shape[1] != 128:
        raise ValueError("Expected even-sector kets with shape (N, 128)")
    result = np.zeros((len(g), 16, 16))
    for a in range(16):
        for b in range(a+1, 16):
            permutation, phase = _bivector_action(8, a, b)
            value = (1j*np.sum(g.conj()*g[:, permutation]*phase[None, :], axis=1)).real
            result[:, a, b] = value
            result[:, b, a] = -value
    return result


class DiagonalSpin7:
    """Phase-preserving local16 lifts, acting as U_B tensor U_A."""
    def __init__(self, model):
        self.model = model
        self.quadratics = np.asarray([
            quadratic_operator(model.j_matrices[0]@model.j_matrices[j],
                               model.local_gammas) for j in range(1, 7)])

    def rotation(self, direction):
        direction = np.asarray(direction, dtype=float)
        if direction.shape != (7,) or not np.isclose(np.linalg.norm(direction), 1., atol=1e-12):
            raise ValueError("Direction must be a unit vector in R^7")
        angle = float(np.arccos(np.clip(direction[0], -1., 1.)))
        axis = direction[1:].copy()
        length = float(np.linalg.norm(axis))
        if length < 1e-13:
            if angle < 1e-13:
                return np.eye(16, dtype=complex), np.eye(8)
            axis[:] = 0.
            axis[0] = 1.
        else:
            axis /= length
        Q = np.einsum("a,aij->ij", axis, self.quadratics, optimize=False)
        eigenvalues, eigenvectors = eigh(Q, check_finite=False)
        U = (eigenvectors*np.exp(-.5j*angle*eigenvalues))@eigenvectors.conj().T
        J_axis = np.einsum("a,aij->ij", axis, self.model.j_matrices[1:], optimize=False)
        K = self.model.j_matrices[0]@J_axis
        R = np.cos(angle/2)*np.eye(8)+np.sin(angle/2)*K
        return U, R

    def act(self, ket, direction):
        """Preserve the supplied ket's phase; never rephase each orbit point."""
        U, R = self.rotation(direction)
        matrix = self.model.embed(ket).reshape(16, 16)
        full = (U@matrix@U.T).reshape(-1)
        return full[self.model.even_masks], R


def compress_projected_rays(model, source_arrays, t=20, threshold=1e-7,
                            tolerance=1e-10):
    """Merge equal-norm projected rays, then refit only their 3D coefficients.

The selected representatives remain normalized ORIGINAL Gaussian kets. This
does not normalize their projections or pretend the projections are Gaussian.
"""
    key = f"full_t{float(t):g}".replace(".", "p")
    atoms = np.asarray(source_arrays["full_atoms"])
    coefficients = np.asarray(source_arrays[key+"_coefficients"])
    projected = model.S.conj().T@atoms.T
    active = np.flatnonzero(np.abs(coefficients) > threshold)
    representatives, members, merged = [], [], []
    for index in active:
        column = projected[:, index]
        found = False
        for j, representative in enumerate(representatives):
            ref = projected[:, representative]
            factor = np.vdot(ref, column)/np.vdot(ref, ref)
            if (abs(abs(factor)-1.) <= tolerance
                    and np.linalg.norm(column-factor*ref) <= tolerance):
                members[j].append(int(index))
                merged[j] += factor*coefficients[index]
                found = True
                break
        if not found:
            representatives.append(int(index))
            members.append([int(index)])
            merged.append(coefficients[index])
    if not representatives:
        raise ValueError("No source orbit coefficient exceeds the active threshold")
    representatives = np.asarray(representatives, dtype=int)
    P = projected[:, representatives]
    target = model.target(t)
    z = model.S.conj().T@target
    refitted, _, rank, _ = np.linalg.lstsq(P, z, rcond=1e-12)
    return {
        "indices": representatives, "atoms": atoms[representatives],
        "projected_atoms": P, "coefficients": refitted,
        "frozen_merged_coefficients": np.asarray(merged),
        "groups": members, "rank": int(rank),
        "projected_refit_residual": float(np.linalg.norm(P@refitted-z)),
        "discarded_source_coefficient_l1": float(np.sum(np.abs(coefficients[np.abs(coefficients) <= threshold]))),
    }


def solve_finite_with_fock_repair(kets, target, time_limit=30.):
    """Optimize explicit Gaussian coefficients plus the exact Fock repair.

This finite SOCP always has the all-Fock feasible solution. Its optimality is
only for these finite Gaussian columns and Fock atoms, not the full dictionary.
"""
    import cvxpy as cp
    G = np.asarray(kets, dtype=complex)
    target = np.asarray(target, dtype=complex)
    coefficients = cp.Variable(len(G), complex=True)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.abs(coefficients))
                                    +cp.sum(cp.abs(target-G.T@coefficients))))
    try:
        problem.solve(solver="CLARABEL", max_iter=150, max_threads=1,
                      time_limit=float(time_limit), tol_gap_abs=1e-9,
                      tol_gap_rel=1e-9, tol_feas=1e-9, verbose=False)
        value = coefficients.value
        status = str(problem.status)
    except cp.error.SolverError:
        value, status = None, "solver_error_fock_fallback"
    if value is None or not np.all(np.isfinite(value)):
        value = np.zeros(len(G), dtype=complex)
        status += "_fock_fallback"
    value = np.asarray(value, dtype=complex).reshape(-1)
    residual = target-G.T@value
    return {"coefficients": value, "residual": residual, "solver_status": status,
            "coefficient_l1": float(np.sum(np.abs(value))),
            "objective": float(np.sum(np.abs(value))+np.sum(np.abs(residual))),
            "finite_dictionary_optimality_only": True}


def _physical_values(ket, model, t):
    norm = float(np.linalg.norm(ket))
    if norm < 1e-14:
        return None
    normalized = ket/norm
    values = exact_metrics(normalized, model)
    full = model.embed(normalized)
    values["energy"] = float(np.vdot(full, model.full_hamiltonian(t)@full).real)
    return {key: float(values[key]) for key in
            ("energy", "entropy", "schmidt_l1", "mass_correlation",
             "wick_prediction", "wick_deviation", "eta", "mass_a", "mass_b", "mass_dot")}


def _diagnose(kets, coefficients, target, model, t, reference, state_error,
              expected_covariances=None):
    raw = np.einsum("n,ni->i", coefficients, kets, optimize=False)
    residual = target-raw
    repaired = raw+residual
    covariances = covariance_from_kets(kets)
    norm_error = float(np.max(np.abs(np.linalg.norm(kets, axis=1)-1.)))
    purity = float(np.max(np.linalg.norm(covariances@covariances+np.eye(16), axis=(1, 2))))
    covariance_error = (float(np.max(np.linalg.norm(covariances-expected_covariances, axis=(1, 2))))
                        if expected_covariances is not None else None)
    cert = separation_certificate(float(np.sum(np.abs(coefficients))), residual,
                                  reference["schmidt_l1"], state_error=state_error)
    raw_values = _physical_values(raw, model, t)
    repaired_values = _physical_values(repaired, model, t)
    row = {**cert, "finite_gaussian_count": int(len(kets)),
           "active_gaussian_count_1e_7": int(np.sum(np.abs(coefficients) > 1e-7)),
           "fock_repair_count_nonzero": int(np.count_nonzero(residual)),
           "fock_repair_count_above_1e_12": int(np.sum(np.abs(residual) > 1e-12)),
           "finite_count_including_all_numerical_fock_repair": int(len(kets)+np.count_nonzero(residual)),
           "finite_count_including_fock_repair_above_1e_12": int(len(kets)+np.sum(np.abs(residual) > 1e-12)),
           "raw_full_residual": float(np.linalg.norm(residual)),
           "repaired_full_residual": float(np.linalg.norm(repaired-target)),
           "raw_reconstruction_norm": float(np.linalg.norm(raw)),
           "max_atom_norm_error": norm_error, "max_gaussian_purity_error": purity,
           "max_covariance_lift_error": covariance_error,
           "gaussian_checks_passed": bool(norm_error < 1e-10 and purity < 1e-10
                                          and (covariance_error is None or covariance_error < 1e-10)),
           "raw_observables": raw_values, "repaired_observables": repaired_values,
           "raw_observable_absolute_errors": ({key: abs(value-reference[key]) for key, value in raw_values.items()}
                                               if raw_values is not None else None),
           "repaired_observable_absolute_errors": {key: abs(value-reference[key]) for key, value in repaired_values.items()},
           "finite_count_note": "The numerical Fock repair is stored in full, including roundoff-sized coefficients; both exact-nonzero and 1e-12 counts are reported."}
    arrays = {"kets": kets, "coefficients": coefficients, "covariances": covariances,
              "reconstruction": raw, "fock_repair_coefficients": residual,
              "target": target}
    return row, arrays


def run_finite_orbits(model, source_arrays, config, budget, checkpoint=None):
    """Primary 24-state simplex construction, then bounded total-R comparisons.

R is always the TOTAL number of raw Gaussian branches, never per orbit. The
random comparator samples mass directions on S^6 for the selected source
orbits; it is not full Gaussian Haar sampling. Every completed row checkpoints.
"""
    budget.check()
    t = float(config.get("orbit_t", 20))
    sizes = [int(R) for R in config.get("orbit_sizes", [8, 16, 32, 64, 128])]
    if not sizes or any(R < 1 or R > 128 for R in sizes):
        raise ValueError("orbit_sizes are total branch counts and must be between 1 and 128")
    seeds = config.get("seeds", [20261023, 20261024, 20261025])
    target = model.target(t)
    reference = _physical_values(target, model, t)
    target_diagnostic = target_error_estimate(model, t, target)
    state_error = target_diagnostic["target_state_error_estimate"]
    compressed = compress_projected_rays(model, source_arrays, t,
                                          config.get("active_threshold", 1e-7))
    representatives = compressed["atoms"]
    count = len(representatives)
    if count*8 > 128:
        raise ValueError("The exact simplex construction would exceed 128 raw branches")
    action = DiagonalSpin7(model)
    source_covariances = covariance_from_kets(representatives)
    report = {"status": "running", "t": t, "rows": [], "target_diagnostics": target_diagnostic,
              "source_active_groups": compressed["groups"],
              "representative_count": count, "projected_refit_rank": compressed["rank"],
              "projected_refit_residual": compressed["projected_refit_residual"],
              "discarded_source_coefficient_l1": compressed["discarded_source_coefficient_l1"],
              "strict_interval_certified": False,
              "scope": "Explicit finite normalized Gaussian kets; simplex quadrature checked only on the selected source orbits.",
              "R_definition": "Total raw Gaussian branch count, excluding separately reported Fock residual repair."}
    arrays = {"source_representative_indices": compressed["indices"],
              "source_representative_kets": representatives,
              "refitted_projected_coefficients": compressed["coefficients"],
              "frozen_merged_coefficients": compressed["frozen_merged_coefficients"]}

    def generate(directions, assignments):
        generated, expected = [], []
        for direction, representative in zip(directions, assignments):
            budget.check()
            ket, R = action.act(representatives[representative], direction)
            rotation = np.zeros((16, 16))
            rotation[:8, :8] = rotation[8:, 8:] = R
            generated.append(ket)
            expected.append(rotation@source_covariances[representative]@rotation.T)
        return np.asarray(generated), np.asarray(expected)

    def save_row(key, kets, coefficients, expected, metadata):
        budget.check()
        row, saved = _diagnose(kets, coefficients, target, model, t, reference,
                               state_error, expected)
        if not row['gaussian_checks_passed']:
            row['numerical_separation'] = False
            row['evidence_level'] = 'invalid_gaussian_atom_checks'
        row.update({"key": key, "t": t, **metadata})
        report["rows"].append(row)
        arrays.update({key+"_"+field: value for field, value in saved.items()})
        if checkpoint:
            checkpoint(report, arrays)

    simplex = simplex_directions()
    directions = np.repeat(simplex, count, axis=0)
    assignments = np.tile(np.arange(count), 8)
    design_kets, design_covariances = generate(directions, assignments)
    orbit_means = design_kets.reshape(8, count, 128).mean(axis=0)
    projected_full = (model.S@compressed["projected_atoms"]).T
    quadrature_errors = np.linalg.norm(orbit_means-projected_full, axis=1)
    report["simplex_orbit_average_errors"] = quadrature_errors.tolist()
    report["simplex_specific_orbits_verified"] = bool(np.max(quadrature_errors) < config.get("structural_tolerance", 1e-10))
    arrays["simplex_directions"] = simplex
    arrays["simplex_orbit_means"] = orbit_means
    # First preserve the constructive result without solving a new optimization.
    direct_coefficients = np.tile(compressed["coefficients"]/8, 8)
    save_row("exact_simplex", design_kets, direct_coefficients, design_covariances,
             {"method": "specific_orbit_simplex_2_design", "R": len(design_kets),
              "seed": None, "solver_status": "constructive_3D_refit_uniform_orbit_weights",
              "per_orbit_counts": [8]*count,
              "max_specific_orbit_average_error": float(np.max(quadrature_errors))})
    for R in sizes:
        if R >= len(design_kets):
            continue
        budget.check()
        solved = solve_finite_with_fock_repair(design_kets[:R], target)
        save_row(f"partial_simplex_R{R}", design_kets[:R], solved["coefficients"],
                 design_covariances[:R], {"method": "partial_simplex_finite_SOCP", "R": R,
                 "seed": None, "solver_status": solved["solver_status"],
                 "per_orbit_counts": np.bincount(assignments[:R], minlength=count).tolist()})
    for seed in seeds:
        budget.check()
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 891]))
        directions = rng.standard_normal((max(sizes), 7))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        assignments = np.arange(max(sizes)) % count
        generated, covariances = generate(directions, assignments)
        arrays[f"random_seed{seed}_directions"] = directions
        for R in sizes:
            budget.check()
            solved = solve_finite_with_fock_repair(generated[:R], target)
            save_row(f"random_seed{seed}_R{R}", generated[:R], solved["coefficients"], covariances[:R],
                     {"method": "random_spherical_orbit_finite_SOCP", "R": R,
                      "seed": int(seed), "solver_status": solved["solver_status"],
                      "per_orbit_counts": np.bincount(assignments[:R], minlength=count).tolist(),
                      "rng_purpose": 891})
    report["status"] = "complete"
    report["usage"] = budget.usage()
    if checkpoint:
        checkpoint(report, arrays)
    return report, arrays
