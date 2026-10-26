"""Independent derivation + Monte-Carlo verification of the generalized frames.

A: Hartree  = Veronese coherent orbit u^{⊗N} in Sym^N(C^m)
B: Slater   = Grassmannian orbit u_1^∧...^u_N in Λ^N(C^m)
Both are SU(m) highest-weight orbits => Schur's lemma makes the k-th frame
operator a SCALAR on the k-fold Cartan module, but with orbit-specific Weyl
dimensions (NOT the four-mode spinor d_{4,k}).

Verifies numerically:
  * Hartree  d_{N,k} = dim Sym^{Nk}(C^m) = binom(Nk+m-1, m-1)
  * Slater   d_{N,k} = prod_{1<=i<=N<j<=m} (k+j-i)/(j-i)   (rectangular (k^N) diagram)
  * first moment E|g><g| = I/d on the orbit space (both cases)
  * dimer scale m=4,N=2: Slater d_k = (k+1)(k+2)^2(k+3)/12, k=1 -> 6 = binom(4,2)
Read-only; writes nothing outside review/.
"""
import math
import itertools
import numpy as np


# ---------- closed forms ----------
def hartree_dim(m, N, k=1):
    return math.comb(N * k + m - 1, m - 1)


def slater_dim(m, N, k=1):
    d = 1.0
    for i in range(1, N + 1):
        for j in range(N + 1, m + 1):
            d *= (k + j - i) / (j - i)
    return d


# ---------- MC first-moment check ----------
def hartree_first_moment(m, N, trials=200000, seed=0):
    rng = np.random.default_rng(seed)
    basis = list(itertools.combinations_with_replacement(range(m), N))
    dim = len(basis)
    # multinomial prefactor sqrt(N!/prod a_i!) for each basis tuple
    pref = []
    for a in basis:
        cnt = np.bincount(a, minlength=m)
        p = math.sqrt(math.factorial(N) / math.prod(math.factorial(c) for c in cnt))
        pref.append(p)
    pref = np.array(pref)
    F = np.zeros((dim, dim), dtype=complex)
    bs = 4096
    done = 0
    while done < trials:
        b = min(bs, trials - done)
        u = rng.standard_normal((b, m)) + 1j * rng.standard_normal((b, m))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        # g_alpha = pref_alpha * prod_i u_i^{a_i}
        powers = np.ones((b, dim), dtype=complex)
        for col, a in enumerate(basis):
            for i in a:
                powers[:, col] *= u[:, i]
        g = powers * pref[None, :]
        F += (g.conj().T @ g) / b
        done += b
    F /= (trials / bs)
    return dim, F


def slater_first_moment(m, N, trials=200000, seed=0):
    rng = np.random.default_rng(seed)
    basis = list(itertools.combinations(range(m), N))  # sorted increasing
    dim = len(basis)
    F = np.zeros((dim, dim), dtype=complex)
    g = np.zeros((len(basis),), dtype=complex)
    for _ in range(trials):
        X = rng.standard_normal((m, N)) + 1j * rng.standard_normal((m, N))
        Q, _ = np.linalg.qr(X)  # Q: (m,m); first N cols = ON frame
        Q = Q[:, :N]
        for col, I in enumerate(basis):
            g[col] = np.linalg.det(Q[list(I), :])
        F += np.outer(g, g.conj())
    F /= trials
    return dim, F


def report(name, dim, F):
    eye = np.eye(dim) / dim
    off = F - np.diag(np.diag(F))
    return (f"{name}: dim={dim}, ||E|g><g| - I/d||_F={np.linalg.norm(F-eye):.2e}, "
            f"max|offdiag|={np.abs(off).max():.2e}, diag_mean={np.diag(F).mean():.4f} (expect {1/dim:.4f})")


def main():
    out = []
    # closed forms
    for (m, N) in [(4, 2), (4, 1), (3, 2)]:
        out.append(f"[closed] Hartree m={m},N={N}: d(k=1)=binom(N+m-1,m-1)={hartree_dim(m,N,1)}; "
                   f"d(k=2)={hartree_dim(m,N,2)}; d(k=4)={hartree_dim(m,N,4)}")
    for (m, N) in [(4, 2), (4, 1), (3, 2)]:
        out.append(f"[closed] Slater  m={m},N={N}: d(k=1)={slater_dim(m,N,1):.6f} (binom={math.comb(m,N)}); "
                   f"d(k=2)={slater_dim(m,N,2):.6f}; d(k=4)={slater_dim(m,N,4):.6f}")
    # dimer Slater explicit
    dk = lambda k: (k+1)*(k+2)**2*(k+3)/12.0
    out.append(f"[dimer Slater m=4,N=2] d_k=(k+1)(k+2)^2(k+3)/12: k=1->{dk(1)}, k=2->{dk(2)}, k=4->{dk(4)}")
    out.append(f"[dimer Slater] gamma=d_k^(1/k): k=2->{dk(2)**0.5:.4f}, k=4->{dk(4)**0.25:.4f}; "
               f"c=sqrt(6)d_k^(-1/2k): k=2->{math.sqrt(6)*dk(2)**-0.25:.4f}")
    # MC first moments
    dim, Fh = hartree_first_moment(m=4, N=2, trials=120000)
    out.append(report("Hartree  Sym^2(C^4) first moment", dim, Fh))
    dim, Fs = slater_first_moment(m=4, N=2, trials=120000)
    out.append(report("Slater   Λ^2(C^4) first moment", dim, Fs))
    print("\n".join(out))


if __name__ == "__main__":
    main()
