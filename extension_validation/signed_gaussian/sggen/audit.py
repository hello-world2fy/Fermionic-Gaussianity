"""Fail-closed Gaussianity audit for even-sector atoms (4 modes, D=8).

Every atom accepted into a dictionary MUST pass ALL of:
  1. ||ket|| = 1 (normalized, phase invariant).
  2. Even-parity quadric  q = g0*g7 - g1*g6 + g2*g5 - g3*g4 = 0  in the even-basis
     ordering (EVEN_OCC = 0,3,5,6,9,10,12,15; index 0=|0000>, 7=|1111>).
  3. rho = |ket><ket| : trace=1, purity=1, Hermitian, PSD, supported on even parity.
An illegal atom RAISES (fail closed); we never silently drop.
"""
from __future__ import annotations
import numpy as np

EVEN_OCC = (0, 3, 5, 6, 9, 10, 12, 15)
TOL_NORM = 1e-9
TOL_QUAD = 1e-9
TOL_RHO = 1e-9


def even_quadric(ket):
    """q = g0 g7 - g1 g6 + g2 g5 - g3 g4 in even-basis index order."""
    g = np.asarray(ket, dtype=complex)
    return g[0]*g[7] - g[1]*g[6] + g[2]*g[5] - g[3]*g[4]


def audit_ket(ket, tol_quad=TOL_QUAD, label=""):
    """Return audit metrics; raise ValueError if the atom is not a legal even Gaussian."""
    g = np.asarray(ket, dtype=complex)
    if g.shape != (8,):
        raise ValueError(f"[audit:{label}] ket shape {g.shape} != (8,)")
    if not (np.isfinite(g.real).all() and np.isfinite(g.imag).all()):
        raise ValueError(f"[audit:{label}] ket contains NaN/inf")
    norm = float(np.linalg.norm(g))
    q = complex(even_quadric(g))
    rho = np.outer(g, g.conj())
    tr = float(np.trace(rho).real)
    pur = float(np.trace(rho @ rho).real)
    herm_err = float(np.linalg.norm(rho - rho.conj().T))
    eig = np.linalg.eigvalsh(rho)
    psd_err = float(-eig.min())
    if not np.isfinite(norm):
        raise ValueError(f"[audit:{label}] non-finite norm")
    if abs(norm - 1.0) > TOL_NORM:
        raise ValueError(f"[audit:{label}] norm={norm:.3e} != 1")
    if abs(q) > tol_quad:
        raise ValueError(f"[audit:{label}] even quadric |q|={abs(q):.3e} != 0 (non-Gaussian)")
    if abs(tr - 1.0) > TOL_RHO:
        raise ValueError(f"[audit:{label}] trace={tr:.3e} != 1")
    if abs(pur - 1.0) > TOL_RHO:
        raise ValueError(f"[audit:{label}] purity={pur:.3e} != 1")
    if herm_err > TOL_RHO:
        raise ValueError(f"[audit:{label}] not Hermitian err={herm_err:.3e}")
    if psd_err > TOL_RHO:
        raise ValueError(f"[audit:{label}] not PSD min_eig={eig.min():.3e}")
    return {"norm": norm, "quadric": abs(q), "trace": tr, "purity": pur,
            "herm_err": herm_err, "psd_min_eig": float(eig.min())}


def audit_projector(a, tol=TOL_RHO, label=""):
    """Fail-closed audit of an ORIGINAL (unsymmetrized) projector 8x8.

    Verifies: finite; exact shape; Hermitian (original a, NOT symmetrized); trace
    real~1; purity~1; PSD; rank-1 eigenvalue structure; recovered main ket passes
    audit_ket; and ||a - |k><k|||_F <= tol on the ORIGINAL matrix.
    """
    a = np.asarray(a, dtype=complex)
    if a.shape != (8, 8):
        raise ValueError(f"[auditproj:{label}] shape {a.shape} != (8,8)")
    if not (np.isfinite(a.real).all() and np.isfinite(a.imag).all()):
        raise ValueError(f"[auditproj:{label}] NaN/inf")
    if float(np.linalg.norm(a - a.conj().T)) > tol:
        raise ValueError(f"[auditproj:{label}] not Hermitian (original) "
                         f"err={np.linalg.norm(a-a.conj().T):.2e}")
    tr = np.trace(a)
    if abs(tr.imag) > tol or abs(tr.real - 1.0) > tol:
        raise ValueError(f"[auditproj:{label}] trace={tr:.3e} != 1")
    pur = float(np.trace(a @ a).real)
    if abs(pur - 1.0) > tol:
        raise ValueError(f"[auditproj:{label}] purity={pur:.3e} != 1")
    eig = np.linalg.eigvalsh(a)
    if eig.min() < -tol:
        raise ValueError(f"[auditproj:{label}] not PSD min_eig={eig.min():.3e}")
    if abs(eig.max() - 1.0) > tol or abs(eig[:-1].sum()) > tol:
        raise ValueError(f"[auditproj:{label}] not rank-1 eigmax={eig.max():.3e} "
                         f"sumrest={eig[:-1].sum():.3e}")
    vec = np.linalg.eigh(a)[1][:, -1]
    vec /= np.linalg.norm(vec)
    audit_ket(vec, label=f"{label}:ket")
    err = float(np.linalg.norm(a - np.outer(vec, vec.conj())))
    if err > tol:
        raise ValueError(f"[auditproj:{label}] projector outer err={err:.3e}")
    return {"trace": tr.real, "purity": pur, "psd_min": float(eig.min()), "outer_err": err}


def audit_pool(kets, sources, tol_quad=TOL_QUAD):
    """Audit a whole pool. Raises on the FIRST illegal atom with its source label."""
    out = []
    for i, (k, s) in enumerate(zip(kets, sources)):
        m = audit_ket(k, tol_quad=tol_quad, label=f"{s}#{i}")
        m["source"] = s
        out.append(m)
    return out
