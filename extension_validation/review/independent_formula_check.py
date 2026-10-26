"""Independent read-only audit of v1 closed-form identities (no producer imports).

Re-derives Weyl dimension, frame constants, and four-mode beta moments from
first principles and compares against the closed forms quoted in the tex.
Writes nothing outside the review directory.
"""
import math
import numpy as np


def d_product(n, k):
    """tex: prod_{i<j} (1 + k/(2n-i-j))."""
    p = 1.0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            p *= 1.0 + k / (2 * n - i - j)
    return p


def d_closed4(k):
    """tex dual_core: (k+1)(k+2)(k+3)^2(k+4)(k+5)/360."""
    return (k + 1) * (k + 2) * (k + 3) ** 2 * (k + 4) * (k + 5) / 360.0


def m_s(s):
    """tex: E|<g,M>|^{2s} = 2^{-s} 6/((s+1)(s+2)(s+3)), M=(0000+1111)/sqrt2."""
    return 2.0 ** (-s) * 6.0 / ((s + 1) * (s + 2) * (s + 3))


def beta_momento_s_mc(seed, trials, s):
    """Monte Carlo: X~Beta(1,3) i.e. density 3(1-x)^2; |<g,M>|^2 = X/2."""
    rng = np.random.default_rng(seed)
    u = rng.random(trials)
    # inverse CDF of Beta(1,3): F(x)=1-(1-x)^3 => x = 1-(1-u)^{1/3}
    x = 1.0 - (1.0 - u) ** (1.0 / 3.0)
    return np.mean((x / 2.0) ** s)


def main():
    out = []
    # 1) d_{4,k} closed form vs product
    for k in (1, 2, 4, 3, 7):
        dp = d_product(4, k)
        dc = d_closed4(k)
        out.append(f"n=4 k={k}: product={dp:.10f} closed={dc:.10f} relerr={abs(dp-dc)/dc:.3e}")
    # 2) gamma_{n,k}=d^{1/k} and c=sqrt(D) d^{-1/(2k)} for n=4
    n = 4
    D = 2 ** (n - 1)
    for k in (1, 2, 4):
        d = d_product(n, k)
        gamma = d ** (1.0 / k)
        c = math.sqrt(D) * d ** (-1.0 / (2 * k))
        out.append(f"n=4 k={k}: d={d:.6f} gamma=d^(1/k)={gamma:.6f} c=sqrt(D)d^(-1/2k)={c:.6f}")
    # 3) beta moment closed form vs MC
    for s in (1.0, 2.0, 3.0, 4.0, 0.5):
        mc = beta_momento_s_mc(0, 4_000_000, s)
        cf = m_s(s)
        out.append(f"s={s}: m_s closed={cf:.6f} MC={mc:.6f} relerr={abs(mc-cf)/cf:.3e}")
    # 4) q_{1,k}^{(4)}(M) = d_{4,k} m_k = 2^{-k} b_k, b_k=(k+3)(k+4)(k+5)/60
    for k in (1, 2, 4):
        q = d_product(4, k) * m_s(float(k))
        bk = (k + 3) * (k + 4) * (k + 5) / 60.0
        q2 = 2.0 ** (-k) * bk
        out.append(f"k={k}: q=d*m={q:.8f} 2^-k b_k={q2:.8f} relerr={abs(q-q2)/q2:.3e}")
    # 5) B_k = m_{k-1/2}/m_k ; B_k^2 -> 2 ; E||X||^2 = m_{k-1}/m_k = 2(k+3)/k
    for k in (2, 4, 10, 100):
        Bk = m_s(k - 0.5) / m_s(float(k))
        var = m_s(k - 1.0) / m_s(float(k))
        var_pred = 2.0 * (k + 3) / k
        out.append(f"k={k}: B_k={Bk:.6f} B_k^2={Bk**2:.6f} E||X||^2={var:.6f} pred={var_pred:.6f}")
    # 6) extent baseline: ||psi||^2 <= xi_G <= ||psi||_1^2 ; for M uniform 0/1111
    psi = np.zeros(8, dtype=complex); psi[0] = 1 / math.sqrt(2); psi[7] = 1 / math.sqrt(2)
    out.append(f"M: ||psi||^2={np.vdot(psi,psi).real:.4f} ||psi||_1^2={np.sum(np.abs(psi))**2:.4f} xi_G=2")
    print("\n".join(out))


if __name__ == "__main__":
    main()
