"""Independent counter-evidence: noci.two_dim_ritz_drop uses eigvalsh on the
non-Hermitian Sg^{-1} Hg, violating the variational principle.

For span{psi, atom}, the TRUE lowest Ritz energy = min eig of Q^dagger H Q where
Q = orthonormal basis of [psi, atom]. The old code does eigvalsh(Sg^{-1} Hg) which
is not Hermitian -> eigvalsh silently uses (M+M^T)/2 or upper triangle -> can return
an energy BELOW the true variational minimum.

Read-only; does not touch producer. Reproducible seeds.
"""
import numpy as np

def old_drop(H, psi, E, atom):
    s = float(psi @ atom)
    if abs(1 - s*s) < 1e-12:
        return E
    Hbb = float(atom @ H @ atom)
    Hpb = float(psi @ H @ atom)
    Hg = np.array([[E, Hpb], [Hpb, Hbb]])
    Sg = np.array([[1.0, s], [s, 1.0]])
    try:
        ev = np.linalg.eigvalsh(np.linalg.solve(Sg, Hg))
        return float(min(ev))
    except Exception:
        return E

def true_ritz(H, psi, atom):
    W = np.column_stack([psi, atom])
    Q, _ = np.linalg.qr(W)
    return float(np.linalg.eigvalsh(Q.conj().T @ H @ Q)[0])

def build_pair(d, s_real, rng, complex_atoms=False):
    H = rng.standard_normal((d,d)) + 1j*rng.standard_normal((d,d))
    H = (H + H.conj().T)/2
    psi = rng.standard_normal(d) + (1j*rng.standard_normal(d) if complex_atoms else 0)
    psi /= np.linalg.norm(psi)
    v = rng.standard_normal(d) + (1j*rng.standard_normal(d) if complex_atoms else 0)
    v -= psi * np.vdot(psi, v)          # orthogonal component
    v /= np.linalg.norm(v)
    atom = s_real*psi + np.sqrt(1-s_real**2)*v
    return H, psi, atom

if __name__ == "__main__":
    rng = np.random.default_rng(20261026)
    d = 12
    print("=== A. random Hermitian complex H, nonorth pair ===")
    n_viol = 0; errs = []
    for trial in range(2000):
        H, psi, atom = build_pair(d, rng.uniform(0.0,0.9), rng, complex_atoms=True)
        E0 = float(np.vdot(psi, H@psi).real)
        e_old = old_drop(H, psi, E0, atom)
        e_true = true_ritz(H, psi, atom)
        errs.append(e_old - e_true)
        if e_old < e_true - 1e-9:       # below variational minimum = VIOLATION
            n_viol += 1
    errs = np.array(errs)
    print("trials=2000  max(old-true)=%.3e  min(old-true)=%.3e" % (errs.max(), errs.min()))
    print("VIOLATIONS (old < true variational min):", n_viol)

    print("=== B. near-dependent overlap s -> 1 ===")
    for eps in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]:
        s = np.sqrt(max(0.0, 1-eps))
        H, psi, atom = build_pair(d, s, rng, complex_atoms=True)
        E0 = float(np.vdot(psi, H@psi).real)
        e_old = old_drop(H, psi, E0, atom); e_true = true_ritz(H, psi, atom)
        print("  eps=%.0e  old=%.6f true=%.6f diff(old-true)=%.2e" % (eps, e_old, e_true, e_old-e_true))

    print("=== C. complex bra conj test: old uses psi@atom (no conj) ===")
    # craft atom = s*psi + i*perp so <psi|atom> != psi@atom
    H, psi, _ = build_pair(d, 0.0, rng, complex_atoms=True)
    v = rng.standard_normal(d) + 1j*rng.standard_normal(d); v -= psi*np.vdot(psi,v); v/=np.linalg.norm(v)
    atom = 0.5*psi + 0.5j*v
    E0 = float(np.vdot(psi,H@psi).real)
    print("  <psi|atom>(conj)=%.3f  psi@atom(no conj)=%.3f  -> old s uses latter, WRONG for complex"
          % (np.vdot(psi,atom).real, float(psi@atom).real))

    print("=== D. full-span exact: atom spans remaining space, old must equal ground ===")
    H = rng.standard_normal((4,4))+1j*rng.standard_normal((4,4)); H=(H+H.conj().T)/2
    psi = rng.standard_normal(4,dtype=complex); psi/=np.linalg.norm(psi)
    full = np.eye(4)
    Q,_ = np.linalg.qr(full); true_ground = float(np.linalg.eigvalsh(H)[0])
    print("  ground=%.6f (old 2x2 cannot reach full span here by design)" % true_ground)
