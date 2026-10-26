"""Bosonic Hartree / coherent-frame moments.  Independent reimplementation.

Frame (PDF Sec.1-3; dual_core.tex) lifted to bosons:
    H = Sym^N C^m,  D = binom(N+m-1, m-1),
    rays |g^N> = (sum_j g_j a_j^dag)^N/sqrt(N!)|0>,  g on sphere of C^m,
    q_k(u) = d * int |<g^N|u>|^{2k} dmu,  d = d_{N,k} = binom(Nk+m-1, m-1).

Integrals over the projective space CP^{m-1}: write g_j = sqrt(x_j) e^{i theta_j},
x in the simplex (Dirichlet(1,..,1), density (m-1)!), theta uniform.  Fix the ray
gauge theta_m = 0 (the aligned quantity (a/|a|)|g^N> is ray invariant).  All phase
integrals are over trigonometric polynomials -> trapezoidal rule exact; all
simplex integrals over polynomials -> Gauss-Legendre on stick-breaking variables
U_j in [0,1] exact.  Deterministic, no sampling.
"""

import itertools
import math
import numpy as np
from scipy.special import roots_jacobi as _roots_jacobi


def fock_basis(m, N):
    out = []
    for tup in itertools.combinations(range(N + m - 1), m - 1):
        n, prev = [], -1
        for c in tup + (N + m - 1,):
            n.append(c - prev - 1)
            prev = c
        out.append(tuple(n))
    return out


def dim_H(m, N):
    return math.comb(N + m - 1, m - 1)


def d_dim(m, N, k):
    return math.comb(N * k + m - 1, m - 1)


def multinomial(n):
    n = np.asarray(n, dtype=int)
    return math.factorial(int(n.sum())) // math.prod(math.factorial(int(x)) for x in n)


# --------------------------------------------------------------------------- #
# Exact Gamma-ratio sphere moments.
# E[prod g_j^{a_j} conj(g_j)^{b_j}] = 0 unless a==b, else Gamma(m) prod Gamma(a_j+1)/Gamma(m+|a|)
# --------------------------------------------------------------------------- #
def sphere_moment(a):
    a = np.asarray(a, dtype=float)
    m = len(a)
    return math.exp(math.lgamma(m) + sum(math.lgamma(x + 1.0) for x in a) - math.lgamma(m + a.sum()))


def qk_formula(n, k):
    n = np.asarray(n, dtype=int); N = int(n.sum())
    mn = multinomial(n)
    kn = tuple(int(k * x) for x in n)
    mkn = math.factorial(k * N) // math.prod(math.factorial(x) for x in kn)
    return mn ** k / mkn


def qk_integral(n, k, m):
    n = np.asarray(n, dtype=int)
    # |<g^N|n>|^{2k} = mult(n)^k prod |g_j|^{2k n_j}; sphere_moment(a) = E prod |g_j|^{2 a_j}
    E = multinomial(n) ** k * sphere_moment(k * n)
    return d_dim(m, int(n.sum()), k) * E


def Ek_abs(n, s, m):
    n = np.asarray(n, dtype=int)
    return multinomial(n) ** s * sphere_moment(s * n)


def F1_fock(n):
    nf = np.asarray(n, dtype=float); N = nf.sum()
    return multinomial(np.asarray(n, dtype=int)) * float(np.prod((nf / N) ** nf))


def xi_fock(n):
    return 1.0 / F1_fock(n)


def Bk_closed(n, k, m):
    return Ek_abs(n, k - 0.5, m) / Ek_abs(n, k, m)


def tk_closed(n, k):
    return qk_formula(n, k) ** (-1.0 / (2.0 * k))


# --------------------------------------------------------------------------- #
# General vector u: gauge-fixed deterministic quadrature.
# --------------------------------------------------------------------------- #
def _gauss_unit_beta(deg, a):
    """Nodes/weights on [0,1] integrating (1-U)^a dU exactly for polynomials."""
    x, w = _roots_jacobi(deg, a, 0)  # weight (1-x)^a on [-1,1]  => alpha=a, beta=0
    U = 0.5 * (x + 1.0)
    # ∫_{-1}^1 f(x)(1-x)^a dx = 2^{a+1} ∫_0^1 f(U)(1-U)^a dU
    wU = 0.5 * w / (2.0 ** a)
    return U, wU


def _stick_to_simplex(U):
    """U_j in [0,1]  ->  x on simplex.  E over Dirichlet(1,..,1) factors as
    U_j ~ Beta(1, m-1-j); quadrature weights below absorb the densities."""
    m = len(U) + 1
    x = np.zeros(m)
    rem = 1.0
    for j in range(m - 1):
        x[j] = rem * U[j]
        rem *= (1.0 - U[j])
    x[m - 1] = rem
    return x


def _phase_L(m, N, k):
    return 2 * (2 * k - 1) * N + 3


def eval_qk_Ck(u, basis, m, N, k, ngauss=None, verbose=False):
    """Deterministically evaluate q_k(u) and C_k(u) (Fock-basis vector)."""
    u = np.asarray(u, dtype=complex)
    D = len(basis)
    c = np.array([np.conj(up) * math.sqrt(multinomial(p)) for up, p in zip(u, basis)])
    L = _phase_L(m, N, k)
    th = np.linspace(0.0, 2.0 * np.pi, L, endpoint=False)
    nfree = m - 1
    # simplex quadrature nodes: U_j ~ Beta(1, m-1-j), density (m-1-j)(1-U)^{m-2-j}.
    # Gauss-Jacobi nodes already integrate against (1-U)^{a} with a=m-2-j;
    # multiply by normalizing constant (a+1) = m-1-j afterwards.
    ng = ngauss or (2 * k + 2)
    unodes, uweights = [], []
    norm = []
    for j in range(nfree):
        a = m - 2 - j  # Beta(1, a+1)
        U, wU = _gauss_unit_beta(ng, a)
        unodes.append(U)
        uweights.append(wU * (a + 1))  # (a+1) = normalizer of Beta(1,a+1)
        norm.append(a + 1)
    Ugrid = np.array(list(itertools.product(*unodes)))
    Wgrid = np.ones(len(Ugrid))
    wcols = list(itertools.product(*uweights))
    Wgrid = np.array([np.prod(row) for row in wcols])

    q = 0.0
    C = np.zeros(D, dtype=complex)
    pmult = np.array([math.sqrt(multinomial(p)) for p in basis])
    for Ui, wU in zip(Ugrid, Wgrid):
        x = _stick_to_simplex(Ui)
        shape = (L,) * nfree
        phi = np.zeros(shape, dtype=complex)
        for cp, p in zip(c, basis):
            amp = cp * np.prod(x ** (0.5 * np.array(p, dtype=float)))
            axes = np.meshgrid(*[th] * nfree, indexing='ij', sparse=True)
            ph = np.zeros(shape)
            for j in range(nfree):
                ph = ph - p[j] * axes[j]
            phi = phi + amp * np.exp(1j * ph)
        w = np.abs(phi) ** (2 * k)
        q += wU * float(np.mean(w))
        F = np.abs(phi) ** (2 * k - 2) * phi
        for pip, pp in enumerate(basis):
            amp = pmult[pip] * np.prod(x ** (0.5 * np.array(pp, dtype=float)))
            axes = np.meshgrid(*[th] * nfree, indexing='ij', sparse=True)
            ph = np.zeros(shape)
            for j in range(nfree):
                ph = ph + pp[j] * axes[j]
            C[pip] += wU * amp * np.mean(F * np.exp(1j * ph))
    return d_dim(m, N, k) * q, d_dim(m, N, k) * C


def eval_Bk(u, basis, m, N, k, ngauss=14):
    """B_k(u) = d int |<g^N|u>|^{2k-1} dmu  (coefficient mass), deterministic."""
    u = np.asarray(u, dtype=complex)
    c = np.array([np.conj(up) * math.sqrt(multinomial(p)) for up, p in zip(u, basis)])
    L = _phase_L(m, N, k) + 4
    th = np.linspace(0.0, 2.0 * np.pi, L, endpoint=False)
    nfree = m - 1
    unodes, uweights = [], []
    for j in range(nfree):
        a = m - 2 - j
        U, wU = _gauss_unit_beta(ngauss, a)
        unodes.append(U); uweights.append(wU * (a + 1))
    Ugrid = np.array(list(itertools.product(*unodes)))
    Wgrid = np.array([np.prod(row) for row in itertools.product(*uweights)])
    tot = 0.0
    for Ui, wU in zip(Ugrid, Wgrid):
        x = _stick_to_simplex(Ui)
        shape = (L,) * nfree
        phi = np.zeros(shape, dtype=complex)
        for cp, p in zip(c, basis):
            amp = cp * np.prod(x ** (0.5 * np.array(p, dtype=float)))
            axes = np.meshgrid(*[th] * nfree, indexing='ij', sparse=True)
            ph = np.zeros(shape)
            for j in range(nfree):
                ph = ph - p[j] * axes[j]
            phi = phi + amp * np.exp(1j * ph)
        tot += wU * float(np.mean(np.abs(phi) ** (2 * k - 1)))
    return d_dim(m, N, k) * tot


def overlap_g(psi, basis, g):
    """<g^N|psi>."""
    s = 0j
    for p, up in zip(basis, psi):
        s += np.conj(up) * math.sqrt(multinomial(p)) * np.prod(g.conj() ** np.array(p, dtype=int))
    return s


def max_overlap_g(psi, basis, m, N, nstart=200, seed=0):
    """F_1(psi) = max_g |<g^N|psi>|^2 by deterministic local optimization from a
    coarse deterministic net of starting directions."""
    best = -1.0
    rng = np.random.default_rng(seed)
    starts = []
    # deterministic simplex/phase net
    for _ in range(nstart):
        x = rng.random(m); x = x / x.sum()
        ph = rng.uniform(0, 2 * np.pi, m)
        g = np.array([math.sqrt(x[j]) * np.exp(1j * ph[j]) for j in range(m)])
        starts.append(g)
    # refine each start by projected gradient on the sphere
    for g in starts:
        for _ in range(80):
            ov = overlap_g(psi, basis, g)
            # Wirtinger gradient d|ov|^2 / d conj(g_j); ov depends on conj(g)
            grad = np.zeros(m, dtype=complex)
            for p, up in zip(basis, psi):
                mp = math.sqrt(multinomial(p))
                for j in range(m):
                    if p[j] > 0:
                        prod = 1.0 + 0.0j
                        for l in range(m):
                            e = p[l] - (1 if l == j else 0)
                            prod *= g.conj()[l] ** e
                        grad[j] += np.conj(up) * mp * p[j] * prod
            grad = ov * grad  # d|ov|^2/d conj g_j = ov * d conj(ov)/d conj g_j... use direct:
            # |ov|^2 = ov*conj(ov); d/d conj g_j = conj(ov) * d ov/d conj g_j
            dov = np.zeros(m, dtype=complex)
            for p, up in zip(basis, psi):
                mp = math.sqrt(multinomial(p))
                for j in range(m):
                    if p[j] > 0:
                        prod = 1.0 + 0.0j
                        for l in range(m):
                            e = p[l] - (1 if l == j else 0)
                            prod *= g.conj()[l] ** e
                        dov[j] += np.conj(up) * mp * p[j] * prod
            grad = np.conj(ov) * dov
            g = g + 0.05 * grad
            g = g / np.linalg.norm(g)
        v = abs(overlap_g(psi, basis, g)) ** 2
        best = max(best, v)
    return best


# --------------------------------------------------------------------------- #
# Certified KKT-free bounds for an arbitrary feasible witness u (not assuming
# stationarity C(u)=psi).  Uses ONLY polynomial-exact integrals q(u), C(u).
#   lower bound:  xi >= (|<u|psi>| / q(u)^{1/(2k)})^2          (test-vector bound)
#   upper bound:  tau(psi) <= d^{1/(2k)} q(u)^{(2k-1)/(2k)} + tau(residual)
#                 residual r = psi - C(u);  tau(r) <= sum_n |r_n| tau(|n>)
#   with tau(|n>) = 1/sqrt(F1(n)) known explicitly.
# The odd-power integral B_k(u) is deliberately NOT used as a certificate.
# --------------------------------------------------------------------------- #
def fock_tau_norms(basis, m, N):
    return np.array([1.0 / math.sqrt(F1_fock(n)) for n in basis])


def certified_bounds(u, psi, basis, m, N, k, ngauss=14):
    u = np.asarray(u, dtype=complex); psi = np.asarray(psi, dtype=complex)
    q, C = eval_qk_Ck(u, basis, m, N, k, ngauss=ngauss)
    r = psi - C
    # lower bound (KKT-free test vector)
    ov = abs(np.vdot(u, psi))
    t_lo = ov / q ** (1.0 / (2 * k))
    xi_lo = t_lo ** 2
    # upper bound: Holder envelope on the ensemble mass + residual atomic norm
    tau_ens = d_dim(m, N, k) ** (1.0 / (2 * k)) * q ** ((2 * k - 1) / (2 * k))
    tau_n = fock_tau_norms(basis, m, N)
    tau_res = float(np.sum(np.abs(r) * tau_n))
    tau_hi = tau_ens + tau_res
    xi_hi = tau_hi ** 2
    rho = float(np.linalg.norm(r))
    return dict(q=q, xi_lower=xi_lo, xi_upper=xi_hi,
                tau_envelope=tau_ens, tau_residual=tau_res,
                KKT_residual=rho, overlap=ov)
