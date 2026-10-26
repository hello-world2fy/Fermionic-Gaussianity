"""Independent review of Hartree (A) claims.

Checks:
 (1) q_k(n)=multinomial(n)^k/multinomial(kn) vs direct Gamma integral.
 (2) 864-case count reproduction.
 (3) B_k quadrature (eval_Bk, integrand |<g|u>|^{2k-1}, NON-trig-polynomial)
     vs the analytic Bk_closed for a Fock witness u=q^{-1/(2k-1)}|n>, where the
     analytic value is exact. Quantifies quadrature error the producers do not bound.
 (4) Lower-bound quotient: rigorous KKT-free lower bound is |<u,psi>|/q(u)^{1/(2k)},
     NOT q(u)^{(2k-1)/(2k)} which assumes stationarity C(u)=psi. Compare for an
     approximate L-BFGS witness vs exact Fock witness.
Read-only.
"""
import sys, math, itertools
import numpy as np
sys.path.insert(0, "/Users/bytedance/Desktop/Fermionic Gaussianity/extension_validation/hartree/code")
import boson_frame as bf


def count_cases():
    cases = [(2, 4), (2, 6), (3, 3), (3, 4), (4, 4)]
    tot = 0
    for (m, N) in cases:
        d = bf.dim_H(m, N)
        tot += d * 12
    return tot


def main():
    out = []
    # (1)(2)
    out.append(f"(1)(2) reproduced case count = {count_cases()} (producers claim 864)")
    for (m, N, n) in [(2, 4, (2, 2)), (3, 3, (1, 1, 1)), (4, 4, (1, 1, 1, 1))]:
        for k in [1, 2, 4]:
            qf = bf.qk_formula(np.array(n), k)
            qi = bf.qk_integral(np.array(n), k, m)
            out.append(f"  m={m} n={n} k={k}: q_formula={qf:.10f} q_integral={qi:.10f} rel={abs(qf-qi)/qf:.2e}")

    # (3) eval_Bk quadrature vs analytic Bk_closed on the exact Fock witness
    out.append("(3) B_k quadrature error (Fock witness; analytic Bk_closed is exact):")
    for (m, N, n) in [(2, 4, (2, 2)), (3, 3, (1, 1, 1)), (3, 3, (2, 1, 0))]:
        basis = bf.fock_basis(m, N)
        idx = basis.index(tuple(n))
        psi = np.zeros(len(basis), dtype=complex); psi[idx] = 1.0
        for k in [1, 2, 4]:
            q = bf.qk_formula(np.array(n), k)
            alpha = q ** (-1.0 / (2 * k - 1))
            ustar = alpha * psi
            Bquad = bf.eval_Bk(ustar, basis, m, N, k, ngauss=14)
            Banal = bf.Bk_closed(np.array(n), k, m)
            out.append(f"  m={m} n={n} k={k}: eval_Bk(quadrature)={Bquad:.8f} "
                       f"Bk_closed(analytic)={Banal:.8f} relerr={abs(Bquad-Banal)/Banal:.3e}")

    # (4) lower-bound quotient: exact Fock witness is stationary so they coincide;
    #     show the two formulas differ for a NON-stationary (random) u
    out.append("(4) lower bound on a non-stationary random u (rigorous quotient vs stationarity-assuming):")
    m, N = 3, 3
    basis = bf.fock_basis(m, N)
    idx = basis.index((1, 1, 1))
    psi = np.zeros(len(basis), dtype=complex); psi[idx] = 1.0
    rng = np.random.default_rng(0)
    for k in [2, 4]:
        u = rng.standard_normal(len(basis)) + 0j
        q, C = bf.eval_qk_Ck(u, basis, m, N, k, ngauss=14)
        quot = abs(np.vdot(u, psi)) / q ** (1 / (2 * k))           # rigorous KKT-free
        stat = q ** ((2 * k - 1) / (2 * k))                        # assumes C(u)=psi
        kkt = np.linalg.norm(C - psi)
        out.append(f"  k={k}: rigorous |<u,psi>|/q^(1/2k)={quot:.5f}  "
                   f"stationarity-form q^((2k-1)/2k)={stat:.5f}  ||C-psi||={kkt:.3f}  "
                   f"(only the rigorous one is a valid xi lower bound)")
    print("\n".join(out))


if __name__ == "__main__":
    main()
