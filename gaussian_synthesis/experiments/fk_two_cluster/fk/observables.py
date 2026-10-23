"""Exact observables and Wick predictions, using the same explicit conventions."""
import numpy as np

from .conventions import inner, matvec


def covariance(ket128, model):
    state = model.embed(ket128)
    if not np.isclose(np.linalg.norm(state), 1., rtol=1e-10, atol=1e-12):
        raise ValueError("covariance expects a normalized state")
    acted = np.einsum("aij,j->ai", model.gammas, state, optimize=False)
    two_point = np.einsum("ai,bi->ab", acted.conj(), acted, optimize=False)
    result = (1j*two_point).real
    np.fill_diagonal(result, 0.)
    return (result-result.T)/2


def atom_invariants(covariances, clifford):
    covariances = np.asarray(covariances, dtype=float)
    clifford = np.asarray(clifford, dtype=float)
    if covariances.shape[-2:] != (16,16) or clifford.shape != (7,8,8):
        raise ValueError("expected covariance (...,16,16) and Clifford (7,8,8)")
    a = covariances[..., :8, :8]
    b = covariances[..., 8:, 8:]
    cross = covariances[..., :8, 8:]
    mass_a = np.einsum("kab,...ab->...k", clifford, a, optimize=False)/8
    mass_b = np.einsum("kab,...ab->...k", clifford, b, optimize=False)/8
    return {
        "mass_a": np.sum(mass_a*mass_a, axis=-1),
        "mass_b": np.sum(mass_b*mass_b, axis=-1),
        "mass_dot": np.sum(mass_a*mass_b, axis=-1),
        "eta": np.sum(cross*cross, axis=(-2,-1))/8,
    }


def mass_correlation_wick(covariances, clifford):
    """General Wick prediction for sum_alpha Q_A,alpha Q_B,alpha.

    This retains the local disconnected means and both exchange contractions;
    it does not assume singlet symmetry or C proportional to the identity.
    """
    covariances = np.asarray(covariances, dtype=float)
    inv = atom_invariants(covariances, clifford)
    cross = covariances[..., :8, 8:]
    exchange = np.einsum("kab,kcd,...ac,...bd->...", clifford, clifford,
                         cross, cross, optimize=False)
    return 4*inv["mass_dot"]-exchange/8


def gaussian_energy_from_covariance(covariances, t, clifford):
    """Gaussian Wick energy at V=1; useful for backend and mean-field checks."""
    covariances = np.asarray(covariances, dtype=float)
    inv = atom_invariants(covariances, clifford)
    local_norm = np.sum(covariances[..., :8, :8]**2, axis=(-2,-1))
    local_norm += np.sum(covariances[..., 8:, 8:]**2, axis=(-2,-1))
    interaction = -14+3*local_norm/8-12*(inv["mass_a"]+inv["mass_b"])
    hopping = .5*float(t)*np.trace(covariances[..., :8, 8:], axis1=-2, axis2=-1)
    return interaction+hopping


def exact_metrics(state, model):
    full = model.embed(state)
    singular_values = np.linalg.svd(full.reshape(16,16), compute_uv=False)
    probabilities = singular_values**2
    positive = probabilities[probabilities > 0]
    cov = covariance(state, model)
    cq = float(inner(full, matvec(model.mass_correlation_operator, full)).real)
    wick = float(mass_correlation_wick(cov, model.j_matrices))
    inv = atom_invariants(cov, model.j_matrices)
    eta = float(inv["eta"])
    return {
        "entropy": float(-np.sum(positive*np.log2(positive))),
        "schmidt_l1": float(np.sum(singular_values)),
        "mass_correlation": cq,
        "wick_prediction": wick,
        "wick_deviation": cq-wick,
        "eta": eta,
        "wick_symmetric_formula_error": abs(wick+7*eta),
        "mass_a": float(inv["mass_a"]),
        "mass_b": float(inv["mass_b"]),
        "mass_dot": float(inv["mass_dot"]),
    }
