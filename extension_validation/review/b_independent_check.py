"""Independent review checks for Experiment B (Slater).

1) Rebuild the 2-site half-filled Hubbard dimer Hamiltonian by hand (4 states),
   take the ground state, compute the Slater (Takagi/nuclear-norm) extent
   xi_S = (lam1+lam2)^2 and compare to the producer formula 1+U/sqrt(U^2+16t^2).
2) Expose the rho diagnostic flaw in selected_ci._score_candidates:
   residual amplitudes u carry SIGNS, but p_d=|u_d|^{2k}/sum >= 0; comparing a
   signed unit vector uhat to a nonnegative probability vector p is sign-dominated
   and is NOT the paper residual certificate ||psi - t s(w)||.
3) Show the KKT exponent mismatch: if a witness w truly solved the convex
   stationary point C_k(w)=residual on an orthonormal pool, the tilted atom
   weights would be |u_d|^{2k/(2k-1)}, not |u_d|^{2k}; the producer never solves
   for w at all.
Read-only.
"""
import numpy as np


def dimer_ground(U, t=1.0):
    # basis order: D0=c0c2, S1=c0c3, S2=c1c2, D1=c1c3  (spin-orbs 0..3)
    H = np.zeros((4, 4))
    H[0, 0] = U; H[3, 3] = U          # doubles
    for i, j in [(0, 1), (0, 2), (3, 1), (3, 2)]:
        H[i, j] = H[j, i] = -t
    w, v = np.linalg.eigh(H)
    return w[0], v[:, 0]


def antisym_M(psi):
    # spin-orbs: 0=up0,1=up1,2=down0,3=down1.
    # D0=bits{0,2}, S1=bits{0,3}, S2=bits{1,2}, D1=bits{1,3}
    masks = [0b0101, 0b1001, 0b0110, 0b1010]
    M = np.zeros((4, 4), dtype=complex)
    for c, m in zip(psi, masks):
        occ = [b for b in range(4) if (m >> b) & 1]
        i, j = occ
        M[i, j] += c; M[j, i] -= c
    return M


def main():
    out = []
    out.append("=== 1) independent dimer xi_S vs producer formula ===")
    for U in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0]:
        E, psi = dimer_ground(U)
        M = antisym_M(psi)
        sv = np.linalg.svd(M, compute_uv=False)
        lam1 = (sv[0] + sv[1]) / 2
        lam2 = (sv[2] + sv[3]) / 2
        xi_num = (lam1 + lam2) ** 2
        xi_an = 1 + U / np.sqrt(U ** 2 + 16.0)
        out.append(f"U={U:6.2f} E={E:9.5f} lam1={lam1:.5f} lam2={lam2:.5f} "
                   f"xi_num={xi_num:.8f} xi_an={xi_an:.8f} rel={abs(xi_num-xi_an)/xi_an:.2e}")

    out.append("")
    out.append("=== 2) rho diagnostic sign flaw (realistic residual amplitudes) ===")
    # residual amplitudes on external determinants carry arbitrary signs
    rng = np.random.default_rng(1)
    u = rng.standard_normal(50)
    uhat = u / np.linalg.norm(u)
    for krep in [1.0, 2.0, 4.0]:
        au = np.abs(u)
        p = au ** (2 * krep); p /= p.sum()
        rho = np.linalg.norm(uhat - p)
        # if p were meant to approximate the residual direction, a sign-ALIGNED
        # positive projection |uhat|/sum|u| would be the right object:
        pa = np.abs(uhat); pa /= pa.sum()
        out.append(f"k_rep={krep}: producer rho=||uhat-p||={rho:.4f}; "
                   f"note ||uhat||={np.linalg.norm(uhat):.3f}, ||p||={np.linalg.norm(p):.3f}, "
                   f"p.min()={p.min():.2e} (all>=0, uhat has signs)")
    out.append("  -> rho compares a signed unit vector to a nonneg probability vector; "
               "it is sign-orthogonality dominated, not ||psi-t s(w)||.")

    out.append("")
    out.append("=== 3) KKT exponent: true tilted weights vs producer weights ===")
    # stationary point C_k(w)=residual on orthonormal pool: |w_d|^{2k-1} w_d ~ u_d
    # => w_d ~ sign(u_d)|u_d|^{1/(2k-1)}; tilted atom weights pi_d ~ |w_d|^{2k} ~ |u_d|^{2k/(2k-1)}
    u = np.array([0.5, 0.2, -0.15, 0.05, 0.3, -0.4])
    for krep in [1.0, 2.0, 4.0]:
        prod = np.abs(u) ** (2 * krep); prod /= prod.sum()
        kkt = np.abs(u) ** (2 * krep / (2 * krep - 1)); kkt /= kkt.sum()
        ep = "2*k"; ek = "2k/(2k-1)"
        out.append(f"k={krep}: producer p_d=|u|^{ep} = {np.round(prod,3)}")
        out.append(f"        KKT   pi_d=|u|^{ek} = {np.round(kkt,3)}")
    out.append("  -> producer uses w_d ~ u_d (NOT the convex-optimized witness); "
               "no projected-SGD/stationary-point solve exists in slater/.")
    print("\n".join(out))


if __name__ == "__main__":
    main()
