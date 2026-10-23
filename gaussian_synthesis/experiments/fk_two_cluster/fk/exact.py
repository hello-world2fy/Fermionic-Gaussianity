"""Small, independently checked physical reference calculations."""
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

from .conventions import inner, matmul, matvec
from .observables import exact_metrics
from .symmetry import symmetry_report


def run_exact(model, t_values, peak_bounds=(0.,40.)):
    t_values = [float(t) for t in t_values]
    if not t_values or not all(np.isfinite(t) for t in t_values):
        raise ValueError("t_values must be a nonempty finite sequence")
    rows = []
    for t in t_values:
        state = model.target(t)
        full_state = model.embed(state)
        hamiltonian = model.full_hamiltonian(t)
        eigenvalues = eigh(hamiltonian, eigvals_only=True, subset_by_index=(0,1))
        energy = float(inner(full_state, matvec(hamiltonian, full_state)).real)
        row = {"t": t, "t_over_v": t, "energy": energy,
               "full_ground_energy": float(eigenvalues[0]),
               "gap": float(eigenvalues[1]-eigenvalues[0]),
               "ground_energy_crosscheck_error": abs(energy-float(eigenvalues[0])),
               "eigenstate_residual": float(np.linalg.norm(matvec(hamiltonian,full_state)-energy*full_state))}
        row.update(exact_metrics(state, model))
        row["gaussian_variational_energy"] = float(-32-2*t*t/9 if abs(t) <= 9 else -14-4*abs(t))
        rows.append(row)
    lo, hi = map(float, peak_bounds)
    if not 0 <= lo < hi or not np.isfinite(hi):
        raise ValueError("peak_bounds must satisfy 0 <= low < high < infinity")
    peak_result = minimize_scalar(
        lambda t: exact_metrics(model.target(t), model)["wick_deviation"],
        bounds=(lo,hi), method="bounded", options={"xatol": 1e-8})
    peak = {"t_over_v": float(peak_result.x),
            "wick_deviation": float(peak_result.fun),
            "absolute_value": abs(float(peak_result.fun)),
            "search_bounds": [lo,hi], "success": bool(peak_result.success),
            "method": "bounded scalar minimization of direct CQ minus general Wick prediction"}
    masks = model.even_masks
    h0 = model.h_int[np.ix_(masks,masks)]
    h1 = model.h_hop[np.ix_(masks,masks)]
    expected_h0 = np.diag([-56.,-14.,-8.])
    expected_h1 = np.array([[0.,np.sqrt(2),0.], [np.sqrt(2),0.,np.sqrt(14)],
                            [0.,np.sqrt(14),0.]])
    structure = {
        "orthonormality_error": float(np.linalg.norm(matmul(model.S.conj().T,model.S)-np.eye(3))),
        "interaction_closure_error": float(np.linalg.norm(matmul(h0,model.S)-matmul(model.S,model.reduced_interaction))),
        "hopping_closure_error": float(np.linalg.norm(matmul(h1,model.S)-matmul(model.S,model.reduced_hopping))),
        "interaction_formula_error": float(np.linalg.norm(model.reduced_interaction-expected_h0)),
        "hopping_formula_error": float(np.linalg.norm(model.reduced_hopping-expected_h1)),
        "max_ground_energy_crosscheck_error": max(row["ground_energy_crosscheck_error"] for row in rows),
        "max_wick_symmetric_formula_error": max(row["wick_symmetric_formula_error"] for row in rows),
        "symmetry": symmetry_report(model),
    }
    structure["verified"] = bool(structure["symmetry"]["verified"] and all(
        value < 1e-10 for key, value in structure.items() if key.endswith("error")))
    return {"normalization": "V=1; t is t/V", "points": rows,
            "wick_peak": peak, "structure": structure,
            "numerical_scope": "double precision numerical verification, not interval arithmetic"}
