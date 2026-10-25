"""Phase-sensitive transfer noise and its Hermitian pulse quadratures."""
import numpy as np


def distinct_transfer_square_symbol(plus,minus,W):
    """Symbol of X^2 with (s_i^+)^2=(s_i^-)^2=0 imposed first.

    Valid for NumPy arrays or Torch tensors; each retained product acts
    on four distinct spin sites. Squaring the classical X alone is wrong.
    """
    row=plus*(minus@W.T);column=minus*(plus@W)
    return row.sum(-1)**2-(row**2).sum(-1)-(column**2).sum(-1)+((plus**2@(W**2))*minus**2).sum(-1)


def transfer_operators(model):
    theta = 2*np.pi*np.arange(model.r)/model.r
    operators = []
    for q in range(4):
        kernel = np.zeros((model.L, model.L))
        kernel[model.r:, :model.r] = np.cos(q*(theta[:, None]-theta[None, :]))/model.r
        operators.append(model.pair_operator(kernel))
    return operators


def noise_observables(operators, state):
    forward = [A@state for A in operators]
    backward = [A.conj().T@state for A in operators]
    means = np.array([np.vdot(state, v) for v in forward])
    # <X_q X_p> = <X_q^dagger psi | X_p psi>.
    second = np.array([[np.vdot(a, b) for b in forward] for a in backward])
    connected = second - means[:, None]*means[None, :]
    variance = []
    for j, (a, b) in enumerate(zip(forward, backward)):
        variance.extend((np.linalg.norm(a+b)**2 - (2*means[j].real)**2,
                         np.linalg.norm(-1j*(a-b))**2 - (2*means[j].imag)**2))
    return means, connected, np.array(variance)
