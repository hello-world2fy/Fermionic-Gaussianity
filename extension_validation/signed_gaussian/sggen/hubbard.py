"""Two-site Hubbard model (4 modes: a_up,a_down,b_up,b_down), exact diagonalization.

Bit j = mode j, ascending Jordan-Wigner order.  Modes:
  0 = a_up, 1 = a_down, 2 = b_up, 3 = b_down.
H = -t_hop sum_sigma (c_a_sig^dag c_b_sig + h.c.) + U (n_aup n_adown + n_bup n_bdown)
    - mu (n_a + n_b).
Half-filling particle-hole point: mu = U/2.
"""
from __future__ import annotations

import numpy as np

MODES = (0, 1, 2, 3)
AUP, ADOWN, BUP, BDOWN = 0, 1, 2, 3


def annihilator(j):
    """c_j matrix in the 16-dim occupation basis, ascending Jordan-Wigner.

    Correct annihilation: c_j |s> = sign |s without j> when j in s, else 0.
    Nonzero entry at row=sp (after removal), col=s (before removal).
    Creation c_j^dag = annihilator.conj().T.  CAR {c_i,c_j^dag}=delta_ij.
    """
    dim = 16
    mat = np.zeros((dim, dim), dtype=np.complex128)
    for s in range(dim):
        if not (s & (1 << j)):
            continue
        below = bin(s & ((1 << j) - 1)).count("1")
        sign = -1.0 if below % 2 else 1.0
        sp = s ^ (1 << j)
        mat[sp, s] = sign
    return mat


def hubbard_hamiltonian(t_hop=1.0, U=0.0, mu=0.0):
    cs = [annihilator(j) for j in range(4)]
    cd = [c.conj().T for c in cs]
    H = np.zeros((16, 16), dtype=np.complex128)
    # hopping a <-> b for each spin
    for a, b in ((AUP, BUP), (ADOWN, BDOWN)):
        H += -t_hop * (cd[a] @ cs[b] + cd[b] @ cs[a])
    # interactions
    H += U * (cd[AUP] @ cs[AUP] @ cd[ADOWN] @ cs[ADOWN])
    H += U * (cd[BUP] @ cs[BUP] @ cd[BDOWN] @ cs[BDOWN])
    # chemical potential
    for j in range(4):
        H += -mu * (cd[j] @ cs[j])
    return (H + H.conj().T) / 2


def gibbs_state(H, beta):
    w, v = np.linalg.eigh(H)
    e = np.exp(-beta * w - np.ptp(-beta * w) / 2)  # shifted for stability
    rho = (v * e) @ v.conj().T
    rho /= np.trace(rho)
    return rho


def even_block(rho16):
    from .gaussian import EVEN_OCC
    idx = list(EVEN_OCC)
    return rho16[np.ix_(idx, idx)]


def odd_block(rho16):
    from .gaussian import ODD_OCC
    idx = list(ODD_OCC)
    return rho16[np.ix_(idx, idx)]


def majorana_basis():
    """8 Majorana operators gamma_0..gamma_7 as 16x16 matrices.

    gamma_{2j} = (c_j + c_j^dag);  gamma_{2j+1} = -i(c_j - c_j^dag).
    """
    cs = [annihilator(j) for j in range(4)]
    cd = [c.conj().T for c in cs]
    gam = []
    for j in range(4):
        gam.append(cs[j] + cd[j])
        gam.append(-1j * (cs[j] - cd[j]))
    return gam
