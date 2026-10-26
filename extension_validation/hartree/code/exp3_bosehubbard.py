"""Exp A3 (revised certification): unit-filling Bose-Hubbard ground state.

Certification framing (per independent review a_findings.md):
  * LOWER bound on xi uses the KKT-free test-vector quotient
        xi >= (|<u|psi>| / q(u)^{1/(2k)})^2
    (valid for ANY feasible u; q(u) is a polynomial integral -> exact to floating).
  * UPPER bound uses the Holder envelope + explicit residual atomic norm
        tau(psi) <= d^{1/(2k)} q(u)^{(2k-1)/(2k)} + sum_n |r_n| tau(|n>),
        r = psi - C(u), tau(|n>) = 1/sqrt(F1(n))  (C(u) polynomial -> exact).
    This avoids the odd-power integral entirely.
  * B_k from the odd-power quadrature is reported ONLY as a numerical estimate
    with a Richardson refinement margin, NOT as a certificate.
  * condensate_F1 = max_g |<g^N|psi>|^2 is a multi-start local sup -> numerical
    estimate (lower bound), not a global certificate.
"""
import os, sys, json, time, math
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boson_frame as bf

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
os.makedirs(RES, exist_ok=True)


def build_bh(basis, m, N, UoverJ=1.0, J=1.0):
    D = len(basis); idx = {p: i for i, p in enumerate(basis)}
    H = np.zeros((D, D), dtype=complex)
    for p, i in idx.items():
        H[i, i] += 0.5 * UoverJ * sum(nj * (nj - 1) for nj in p)
        for a in range(m - 1):
            if p[a] > 0:
                q = list(p); q[a] -= 1; q[a + 1] += 1; q = tuple(q)
                H[idx[q], i] += -J * math.sqrt(p[a] * (p[a + 1] + 1))
            if p[a + 1] > 0:
                q = list(p); q[a + 1] -= 1; q[a] += 1; q = tuple(q)
                H[idx[q], i] += -J * math.sqrt(p[a + 1] * (p[a] + 1))
    return (H + H.conj().T) / 2


def optimize_witness(psi, basis, m, N, k, ngauss=12):
    D = len(basis)
    def unpack(z): return z[:D] + 1j * z[D:]
    def obj(z):
        u = unpack(z)
        q, C = bf.eval_qk_Ck(u, basis, m, N, k, ngauss=ngauss)
        return q / (2 * k) - np.vdot(u, psi).real
    def grad(z):
        u = unpack(z)
        _, C = bf.eval_qk_Ck(u, basis, m, N, k, ngauss=ngauss)
        g = C - psi
        return np.concatenate([g.real, g.imag])
    z0 = np.concatenate([psi.real, psi.imag]) * 0.3
    res = minimize(obj, z0, jac=grad, method="L-BFGS-B",
                   options={"maxiter": 300, "ftol": 1e-13, "gtol": 1e-9})
    return unpack(res.x), res.nit


def one_site_entropy(psi, basis, m):
    maxn1 = max(p[0] for p in basis)
    rho = np.zeros((maxn1 + 1, maxn1 + 1), dtype=complex)
    for i, p in enumerate(basis):
        for j, q in enumerate(basis):
            if p[1:] == q[1:]:
                rho[p[0], q[0]] += psi[i] * np.conj(psi[j])
    ev = np.linalg.eigvalsh(rho); ev = ev[ev > 1e-12]
    return -float(np.sum(ev * np.log2(ev))), float(np.sum(ev ** 2))


def main():
    t0 = time.time(); out = []
    for (m, UoverJ) in [(2, 0.5), (2, 2.0), (3, 0.5), (3, 2.0), (3, 6.0)]:
        N = m; basis = bf.fock_basis(m, N); D = len(basis)
        H = build_bh(basis, m, N, UoverJ=UoverJ)
        E, vecs = np.linalg.eigh(H)
        psi = vecs[:, 0]; psi /= np.linalg.norm(psi)
        psi = psi * np.exp(-1j * np.angle(psi[np.argmax(np.abs(psi))]))
        F1 = bf.max_overlap_g(psi, basis, m, N, nstart=400, seed=1)
        S, purity = one_site_entropy(psi, basis, m)
        rec = dict(m=m, N=N, D=D, UoverJ=UoverJ, E0=float(E[0]),
                   condensate_F1_estimate=F1, F1_is_lower_bound=True,
                   entropy_site1_bits=S, purity_site1=purity, witness=[])
        for k in [1, 2, 3, 5]:
            ustar, nit = optimize_witness(psi, basis, m, N, k)
            cb = bf.certified_bounds(ustar, psi, basis, m, N, k, ngauss=15)
            # numerical B_k estimate with Richardson refinement (NOT certified)
            B1 = bf.eval_Bk(ustar, basis, m, N, k, ngauss=12)
            B2 = bf.eval_Bk(ustar, basis, m, N, k, ngauss=24)
            rec["witness"].append(dict(
                k=k, nLBFGS=nit,
                xi_lower_certified=cb["xi_lower"],
                xi_upper_certified=cb["xi_upper"],
                tau_envelope=cb["tau_envelope"],
                tau_residual=cb["tau_residual"],
                KKT_residual=cb["KKT_residual"],
                Bk_numerical_estimate=B2,
                Bk_Richardson_margin=abs(B2 - B1) / max(1e-12, abs(B2)),
                gap_rel=(cb["xi_upper"] - cb["xi_lower"]) / cb["xi_lower"]))
        out.append(rec)
        lw = rec["witness"][-1]
        print(f"m={m} U/J={UoverJ}: F1est={F1:.4f} S={S:.3f}b  "
              f"xi_certified_in [{lw['xi_lower_certified']:.3f},{lw['xi_upper_certified']:.3f}] "
              f"KKT={lw['KKT_residual']:.1e}")
    res = dict(walltime_s=time.time() - t0, certification_revision="a_findings.md",
               results=out)
    with open(os.path.join(RES, "exp3_bosehubbard.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
