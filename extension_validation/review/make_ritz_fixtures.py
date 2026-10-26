"""Generate independent fixtures for the CORRECTED two_dim_ritz_drop.

Reference E_ref uses a rank-aware SVD of [psi, atom]: if the smallest singular
value / largest < effective threshold (span effectively 1D), the true 2D Ritz
collapses to E0=<psi|H|psi>. Otherwise ref = min eig of Q^dagger H Q.
"""
import numpy as np

def ref_ritz(H, psi, atom, eps_eff=1e-8):
    W = np.column_stack([psi, atom])
    s = np.linalg.svd(W, compute_uv=False)
    E0 = float(np.vdot(psi, H@psi).real)
    if s[-1]/s[0] < eps_eff:           # effectively 1D span
        return E0
    Q, _ = np.linalg.qr(W)
    return float(np.linalg.eigvalsh(Q.conj().T @ H @ Q)[0])

def make_pair(d, s, rng, complex_atoms):
    H = rng.standard_normal((d,d)) + (1j*rng.standard_normal((d,d)) if complex_atoms else 0)
    H = (H + H.conj().T)/2
    psi = rng.standard_normal(d) + (1j*rng.standard_normal(d) if complex_atoms else 0)
    psi /= np.linalg.norm(psi)
    v = rng.standard_normal(d) + (1j*rng.standard_normal(d) if complex_atoms else 0)
    v -= psi * np.vdot(psi, v); v /= np.linalg.norm(v)
    atom = s*psi + np.sqrt(max(0.0,1-s*s))*v
    return H, psi, atom

if __name__ == "__main__":
    rng = np.random.default_rng(12345); d = 10
    Hs,P,A,E0s,Erefs,tags = [],[],[],[],[],[]
    H=np.array([[0,1],[1,2.]]); psi=np.array([1.,0.]); atom=np.array([.8,.6])
    Hs.append(H);P.append(psi);A.append(atom);E0s.append(0.0);Erefs.append(ref_ritz(H,psi,atom));tags.append("main2d")
    for _ in range(30):
        H,psi,atom=make_pair(d,rng.uniform(0,0.9),rng,False)
        E0=float(np.vdot(psi,H@psi).real)
        Hs.append(H);P.append(psi);A.append(atom);E0s.append(E0);Erefs.append(ref_ritz(H,psi,atom));tags.append("real")
    for _ in range(30):
        H,psi,atom=make_pair(d,rng.uniform(0,0.9),rng,True)
        E0=float(np.vdot(psi,H@psi).real)
        Hs.append(H);P.append(psi);A.append(atom);E0s.append(E0);Erefs.append(ref_ritz(H,psi,atom));tags.append("complex")
    for eps in [1e-4,1e-6,1e-8,1e-10,1e-12,1e-14]:
        H,psi,atom=make_pair(d,np.sqrt(1-eps),rng,True)
        E0=float(np.vdot(psi,H@psi).real)
        Hs.append(H);P.append(psi);A.append(atom);E0s.append(E0);Erefs.append(ref_ritz(H,psi,atom));tags.append("neardep")
    np.savez("ritz_fixtures.npz", H=np.array(Hs,dtype=object), psi=np.array(P,dtype=object),
             atom=np.array(A,dtype=object), E0=np.array(E0s), E_ref=np.array(Erefs), tag=np.array(tags))
    print("saved",len(tags),"fixtures")
