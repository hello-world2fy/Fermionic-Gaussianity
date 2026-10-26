"""Exp A2: real convex optimization for the bosonic finite-order dual witness.

(a) Minimize J_k(u)=q_k(u)/(2k)-Re<u|psi> over the full Fock space with L-BFGS-B
    (moments evaluated by the deterministic quadrature).  For psi=|n> we check the
    optimizer rediscovers u_k = q_k(n)^{-1/(2k-1)} |n> and the KKT residual
    ||C_k(u_k)-psi||.
(b) Independent discrete-dictionary SOCP upper bound on the atomic norm tau(|n>)
    = max_{h(y)<=1} Re<y|n>:  |<y|g_i^N>| <= 1 on a deterministic net {g_i}.
    The net relaxes the continuum constraint, so tau_SOCP >= tau_true (upper bound).
    The explicit feasible witness y*=|n>/sqrt(F1(n)) gives tau >= sqrt(xi) (lower).
"""
import os, sys, json, time, math
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boson_frame as bf
import cvxpy as cp

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
os.makedirs(RES, exist_ok=True)

rng = np.random.default_rng(20261026)


def optimize_witness(psi, basis, m, N, k, ngauss=14):
    D = len(basis)

    def unpack(z):
        return z[:D] + 1j * z[D:]

    def obj(z):
        u = unpack(z)
        q, C = bf.eval_qk_Ck(u, basis, m, N, k, ngauss=ngauss)
        return q / (2 * k) - np.vdot(u, psi).real

    def grad(z):
        u = unpack(z)
        _, C = bf.eval_qk_Ck(u, basis, m, N, k, ngauss=ngauss)
        g = C - psi  # real-Hilbert gradient
        return np.concatenate([g.real, g.imag])

    z0 = np.concatenate([psi.real, psi.imag]) * 0.5
    res = minimize(obj, z0, jac=grad, method="L-BFGS-B",
                   options={"maxiter": 200, "ftol": 1e-14, "gtol": 1e-10})
    ustar = unpack(res.x)
    _, Cstar = bf.eval_qk_Ck(ustar, basis, m, N, k, ngauss=ngauss + 4)
    kkt = float(np.linalg.norm(Cstar - psi))
    return ustar, Cstar, kkt, res.fun, res.nit


def sphere_net(m, n_per=12):
    """Deterministic net on S^{2m-1}: simplex grid + phase grid, theta_m=0 gauge."""
    pts = []
    # simplex grid
    if m == 2:
        xs = [(i / n_per, 1 - i / n_per) for i in range(n_per + 1)]
    else:
        # coarse grid on simplex via independent 1D grids
        grids = [np.linspace(0.05, 0.95, n_per) for _ in range(m - 1)]
        for combo in np.array(np.meshgrid(*grids)).T.reshape(-1, m - 1):
            x = np.zeros(m)
            x[:m - 1] = combo
            x[-1] = max(0.0, 1 - x[:m - 1].sum())
            if x[-1] >= 0.02:
                x = x / x.sum()
                pts.append(x)
        xs = pts
    out = []
    nph = 2 * n_per + 1
    for x in xs:
        # phase grid on free coords
        phis = np.linspace(0, 2 * np.pi, nph, endpoint=False)
        if m == 2:
            for ph in phis:
                g = np.zeros(m, dtype=complex)
                g[0] = np.sqrt(x[0]) * np.exp(1j * ph)
                g[1] = np.sqrt(x[1])
                out.append(g)
        else:
            # skip full tensor for m>2: use a deterministic loop over free phases
            free = np.array(np.meshgrid(*[phis] * (m - 1))).T.reshape(-1, m - 1)
            for fr in free[::max(1, len(free) // 200)]:
                g = np.zeros(m, dtype=complex)
                for j in range(m - 1):
                    g[j] = np.sqrt(x[j]) * np.exp(1j * fr[j])
                g[m - 1] = np.sqrt(x[m - 1])
                out.append(g)
    return np.array(out)


def socp_tau_upper(psi_idx, basis, m, N, net):
    D = len(basis)
    y = cp.Variable(D, complex=True)
    # <y|g^N> = sum_p y_p sqrt(mult p) conj(g)^p
    A = np.array([[math.sqrt(bf.multinomial(p)) * np.prod(g.conj() ** np.array(p, dtype=int))
                   for p in basis] for g in net])
    cons = [cp.abs(A @ y) <= 1]
    obj = cp.Maximize(cp.real(y[psi_idx]))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.CLARABEL)
    return float(prob.value), y.value


def main():
    t0 = time.time()
    out = []
    for (m, N, states) in [(2, 4, [(2, 2), (1, 3), (0, 4)]),
                           (3, 3, [(1, 1, 1), (2, 1, 0)]),
                           (3, 4, [(2, 1, 1)])]:
        basis = bf.fock_basis(m, N)
        for n in states:
            idx = basis.index(tuple(n))
            psi = np.zeros(len(basis), dtype=complex); psi[idx] = 1.0
            f1 = bf.F1_fock(n); xi = 1.0 / f1
            for k in [1, 2, 3, 5]:
                ustar, Cstar, kkt, fval, nit = optimize_witness(psi, basis, m, N, k)
                alpha_exact = bf.qk_formula(n, k) ** (-1.0 / (2 * k - 1))
                # ustar should be alpha*|n>
                err_dir = float(np.linalg.norm(ustar - alpha_exact * psi) / abs(alpha_exact))
                tk = bf.tk_closed(n, k)
                out.append(dict(m=m, N=N, n=str(n), k=k,
                                KKT_residual=kkt, ustar_direction_err=err_dir,
                                t_k=tk, sqrt_xi=math.sqrt(xi),
                                nLBFGS=nit))
            # SOCP upper bound on tau (k-independent certificate)
            net = sphere_net(m, n_per=10 if m == 2 else 6)
            tau_socp, yval = socp_tau_upper(idx, basis, m, N, net)
            # coarser net for convergence
            net2 = sphere_net(m, n_per=16 if m == 2 else 9)
            tau_socp2, _ = socp_tau_upper(idx, basis, m, N, net2)
            for r in out:
                if r["n"] == str(n):
                    r["tau_socp_net1"] = tau_socp
                    r["tau_socp_net2"] = tau_socp2
                    r["tau_lower_explicit"] = math.sqrt(xi)
                    r["tau_claimed"] = math.sqrt(xi)
    res = dict(walltime_s=time.time() - t0, rows=out)
    with open(os.path.join(RES, "exp2_convex.json"), "w") as f:
        json.dump(res, f, indent=2)
    for r in out:
        print(r)


if __name__ == "__main__":
    main()
