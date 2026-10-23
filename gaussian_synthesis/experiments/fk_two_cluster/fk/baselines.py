"""Finite Gaussian dictionaries and symmetry-orbit cost upper bounds.

The projection of a normalized Gaussian atom onto the invariant subspace is
its Spin(7) orbit average and has Gaussian atomic cost at most one. Projected
atoms MUST NOT be normalized: their smaller norms encode the representation
cost. Diagonal Spin(7) also preserves the product Gaussian dictionary.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.linalg import eigh

from .certificates import repaired_cost, separation_certificate


def project_atoms(basis: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    """Return columns S^dagger g_j, preserving every projected atom's norm."""
    S = np.asarray(basis, dtype=complex)
    G = np.asarray(atoms, dtype=complex)
    if S.ndim != 2 or G.ndim != 2 or G.shape[1] != S.shape[0]:
        raise ValueError("Expected basis (dimension, rank) and atoms (count, dimension)")
    if not np.allclose(S.conj().T @ S, np.eye(S.shape[1]), atol=1e-10, rtol=0):
        raise ValueError("Invariant basis must be orthonormal")
    if not np.allclose(np.linalg.norm(G, axis=1), 1.0, atol=1e-10, rtol=0):
        raise ValueError("Input Gaussian atoms must have unit norm")
    return S.conj().T @ G.T


def solve_projected_atomic_l1(basis: np.ndarray, atoms: np.ndarray,
                              target: np.ndarray) -> dict:
    """Solve a finite-orbit complex SOCP and repair its full-space residual.

Solver optimality is only optimality over the finite orbit dictionary. It is
never a claim about the complete Gaussian atomic norm.
"""
    import cvxpy as cp
    S = np.asarray(basis, dtype=complex)
    psi = np.asarray(target, dtype=complex)
    Z = project_atoms(S, atoms)
    z = S.conj().T @ psi
    c = cp.Variable(Z.shape[1], complex=True)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.abs(c))), [Z @ c == z])
    problem.solve(solver="CLARABEL", max_iter=150, max_threads=1, time_limit=30.0, tol_gap_abs=1e-10,
                  tol_gap_rel=1e-10, tol_feas=1e-10, verbose=False)
    if c.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Finite dictionary SOCP failed: {problem.status}")
    coeff = np.asarray(c.value, dtype=complex).reshape(-1)
    reconstruction = S @ (Z @ coeff)
    residual = psi - reconstruction
    cost = float(np.sum(np.abs(coeff)))
    return {
        "coefficients": coeff,
        "projected_atoms": Z,
        "reconstruction": reconstruction,
        "residual": residual,
        "coefficient_l1": cost,
        "cost_upper": repaired_cost(cost, residual),
        "solver_status": problem.status,
        "solver_objective": float(problem.value),
        "active_atoms_1e_7": int(np.sum(np.abs(coeff) > 1e-7)),
    }


def _even_operator(model, operator):
    idx = np.asarray(model.even_masks, dtype=int)
    matrix = np.asarray(operator)
    return matrix[np.ix_(idx, idx)] if matrix.shape != (len(idx), len(idx)) else matrix


def _structured_atoms(model, deadline: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quadratic ground states on the local-mass / hopping saturation arc.

For theta in [0, pi], a=cos(theta), s=sin(theta), the quadratic parent is
-a Q_A,0 + a Q_B,0 + s H_hop. Its ground covariance has the pure block form
A=aJ_0, B=-aJ_0, C=-sI. Every state is globally even; the two endpoints are
product Gaussian states. The midpoint is the hopping ground state.
"""
    QA = _even_operator(model, model.q_ops[0][0])
    QB = _even_operator(model, model.q_ops[1][0])
    hop = _even_operator(model, model.h_hop)
    angles, states = [], []
    # Generate the endpoints first, so product atoms survive any later timeout.
    theta_grid = np.concatenate(([0.0, np.pi], np.linspace(0.0, np.pi, 65)[1:-1]))
    for theta in theta_grid:
        if time.monotonic() >= deadline:
            break
        a, s = float(np.cos(theta)), float(np.sin(theta))
        quadratic = a * (-QA + QB) + s * hop
        _, vec = eigh(quadratic, subset_by_index=(0, 0), check_finite=False)
        states.append(vec[:, 0])
        angles.append(theta)
    if len(states) < 2:
        raise TimeoutError("Budget expired before constructing product endpoints")
    return np.asarray(states), np.asarray(states[:2]), np.asarray(angles)


def _schmidt_norm(model, psi):
    full = np.zeros(256, dtype=complex)
    full[np.asarray(model.even_masks, dtype=int)] = psi
    # A occupies low bits; transpose swaps A/B without changing singular values.
    return float(np.sum(np.linalg.svd(full.reshape(16, 16), compute_uv=False)))


def target_error_estimate(model, t, psi) -> dict:
    """Floating-point a-posteriori estimate, never an interval enclosure.

For exact Hermitian data and a Rayleigh value mu below the first excited
eigenvalue E1, the phase-aligned state error is at most
sqrt(2) * ||H psi - mu psi|| / (E1 - mu). The full-space first two eigenvalues
and residual here are computed in floating point, so the reported value is an
estimate conditional on their accuracy and on the specified model being exact.
"""
    full = np.zeros(256, dtype=complex)
    full[np.asarray(model.even_masks, dtype=int)] = psi
    if not np.isclose(np.linalg.norm(full), 1.0, atol=1e-10, rtol=0):
        raise ValueError("Target must be normalized before estimating its eigenpair error")
    H = np.asarray(model.full_hamiltonian(float(t)))
    energies = eigh(H, subset_by_index=(0, 1), eigvals_only=True, check_finite=False)
    action = np.einsum("ij,j->i", H, full, optimize=False)
    rayleigh = float(np.real(np.vdot(full, action)))
    residual = float(np.linalg.norm(action-rayleigh*full))
    excited_distance = float(energies[1]-rayleigh)
    if excited_distance <= 0:
        raise ValueError("Target is not numerically isolated below the first excited eigenvalue")
    error = float(np.sqrt(2.0)*min(1.0, residual/excited_distance))
    return {
        "target_rayleigh_energy": rayleigh,
        "target_ground_energy_estimate": float(energies[0]),
        "target_full_spectral_gap_estimate": float(energies[1]-energies[0]),
        "target_excited_distance_estimate": excited_distance,
        "target_eigen_residual_l2": residual,
        "target_state_error_estimate": error,
        "target_state_error_interval_certified": False,
        "target_state_error_interpretation": (
            "sqrt(2) times the full-space eigenpair residual divided by E1 minus "
            "the Rayleigh energy; a floating-point a-posteriori estimate assuming "
            "the Hamiltonian, ground-state identification, spectral isolation and "
            "computed eigenvalues/residual are accurate. No interval enclosure of "
            "roundoff, model construction, or symmetry projection is supplied."
        ),
    }


def run_baselines(model, t_values=(0, 5, 9, 11, 20), seed=20261023,
                  max_seconds=180) -> tuple[dict, dict[str, np.ndarray]]:
    """Run bounded Fock and symmetry-orbit baselines for both dictionaries.

Atoms and coefficient arrays are returned for a caller-owned NPZ archive.
Raw Gaussian atoms plus their unnormalized projections fully specify each
continuous orbit decomposition. A wall-clock budget is checked between bounded
operations; no large adaptive dictionary growth occurs.
"""
    from .dictionaries import Dictionary
    started = time.monotonic()
    deadline = started + max_seconds
    report = {"status": "complete", "seed": int(seed), "rows": [],
              "evidence_level": "numerical_evidence", "max_seconds": max_seconds,
              "orbit_projection": "Diagonal Spin(7) Haar average; projected atoms are never normalized",
              "finite_dictionary_optimality_only": True}
    arrays = {}
    target_diagnostics = {}
    fock = np.eye(len(model.even_masks), dtype=complex)
    structured, product_endpoints, angles = _structured_atoms(model, deadline)
    arrays["structured_angles"] = angles
    for name, special in (("full", structured), ("product", product_endpoints)):
        if time.monotonic() >= deadline:
            report["status"] = "budget_exhausted"
            break
        dictionary = Dictionary(name)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 701, int(name == "product")]))
        random_atoms = np.asarray(dictionary.sample(256, rng).kets)
        atoms = np.concatenate((fock, special, random_atoms), axis=0)
        arrays[f"{name}_atoms"] = atoms
        arrays[f"{name}_projected_atoms"] = project_atoms(model.S, atoms)
        for t in t_values:
            if time.monotonic() >= deadline:
                report["status"] = "budget_exhausted"
                break
            psi = np.asarray(model.target(float(t)))
            if psi.ndim != 1:
                raise ValueError("model.target(t) must return the even-sector ket")
            solved = solve_projected_atomic_l1(model.S, atoms, psi)
            schmidt = _schmidt_norm(model, psi)
            if float(t) not in target_diagnostics:
                target_diagnostics[float(t)] = target_error_estimate(model, t, psi)
            target_diagnostic = target_diagnostics[float(t)]
            state_error = target_diagnostic["target_state_error_estimate"]
            certificate = separation_certificate(solved["coefficient_l1"], solved["residual"],
                                                 schmidt, state_error=state_error)
            # The same inequality is useful diagnostically for product costs,
            # but a separation claim must come from the unrestricted dictionary.
            row = {"dictionary": name, "t": float(t), "atom_count": len(atoms),
                   "fock_cost": float(np.sum(np.abs(psi))), "schmidt_norm": schmidt,
                   "product_schmidt_lower": certificate["schmidt_norm_lower"],
                   "solver_status": solved["solver_status"],
                   "active_atoms_1e_7": solved["active_atoms_1e_7"],
                   **certificate,
                   **target_diagnostic,
                   "is_full_dictionary_upper": name == "full"}
            if name == "product":
                row["numerical_separation"] = False
                row["evidence_level"] = "product_upper_bound_only"
            report["rows"].append(row)
            key = f"{name}_t{float(t):g}".replace(".", "p")
            for field in ("coefficients", "reconstruction", "residual"):
                arrays[f"{key}_{field}"] = solved[field]
            arrays[f"{key}_target"] = psi
    report["elapsed_seconds"] = float(time.monotonic() - started)
    report["structured_atom_count"] = len(structured)
    return report, arrays
