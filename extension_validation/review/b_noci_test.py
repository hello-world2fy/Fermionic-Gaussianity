"""Independent test of slater/noci.py convex_j_synthesize (read-only import).

Checks:
 (a) orthonormal dictionary: KKT synthesis should collapse to r itself
     (producer's own claim) -- dual carries no selection info.
 (b) non-orthogonal dictionary with r IN span(A): stationarity
     r = t * A(|z|^{2k-2}z) must hold with t=<w,r> (NOT a refit alpha), and the
     achieved dual value <w,r>/N_k(w) must match a brute-force search.
 (c) the producer's kkt_err uses a least-squares alpha=refit; compare to the
     true KKT residual with t fixed.
"""
import sys, numpy as np
sys.path.insert(0, "/Users/bytedance/Desktop/Fermionic Gaussianity/extension_validation/slater")
from noci import convex_j_synthesize


def dual_bruteforce(A, r, k, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    best = -1.0
    for _ in range(n):
        w = rng.standard_normal(A.shape[0]) + 1j*rng.standard_normal(A.shape[0])
        z = A.conj().T @ w
        N = (np.sum(np.abs(z)**(2*k)))**(1/(2*k))
        v = (w @ r).real / N
        best = max(best, v)
    return best


def main():
    out = []
    rng = np.random.default_rng(3)
    dim, m = 12, 6

    # (a) orthonormal dictionary
    Q, _ = np.linalg.qr(rng.standard_normal((dim, m)))
    r = Q @ rng.standard_normal(m)
    res = convex_j_synthesize(Q, r, k=2.0)
    t = res["t"]; c = res["c"]
    kkt_true = np.linalg.norm(r - t*(Q @ c))/np.linalg.norm(r)
    out.append(f"(a) orthonormal A: t=<w,r>={t:.4f}; ||r - t A c||/||r|| (TRUE KKT)={kkt_true:.2e}; "
               f"reported kkt_err={res['kkt_err']:.2e}; span_dim={res['span_dim']}")

    # (b) non-orthogonal dictionary, r in span
    A = rng.standard_normal((dim, m)) + 0.5*rng.standard_normal((dim, m))
    ctrue = rng.standard_normal(m)
    r = A @ ctrue; r /= np.linalg.norm(r)
    for k in [1.0, 2.0, 4.0]:
        res = convex_j_synthesize(A, r, k=k, n_iter=2000)
        t = res["t"]; c = res["c"]
        kkt_true = np.linalg.norm(r - t*(A @ c))/np.linalg.norm(r)
        bf = dual_bruteforce(A, r, k, n=60000)
        out.append(f"(b) k={k}: dual <w,r>/N_k(w)={t:.4f}; brute-force~{bf:.4f}; "
                   f"TRUE KKT resid={kkt_true:.2e}; reported kkt_err={res['kkt_err']:.2e}; iters={res['iters']}")
    out.append("  note: t = <w,r> since w is N_k-normalized; TRUE KKT uses this fixed t, not the refit alpha.")
    print("\n".join(out))


if __name__ == "__main__":
    main()
