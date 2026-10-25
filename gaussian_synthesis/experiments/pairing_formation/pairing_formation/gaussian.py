"""Full SO(2n)-Haar Gaussian rays, evaluated only on specified Fock strings.

Projection is used for storage only. Atoms retain their full-Fock norm; they
must not be renormalized in the selected sector when evaluating Haar moments.
"""
import numpy as np


def pfaffian_batch(matrices):
    shape = matrices.shape[:-2]
    a = np.array(matrices, dtype=complex, copy=True).reshape(-1, matrices.shape[-1], matrices.shape[-1])
    answer = np.ones(len(a), complex)
    for width in range(a.shape[-1], 1, -2):
        pivot_index = np.argmax(abs(a[:, 0, 1:]), axis=1)+1
        permutation = np.broadcast_to(np.arange(width), (len(a), width)).copy()
        permutation[:, 1] = pivot_index
        permutation[np.arange(len(a)), pivot_index] = 1
        a = np.take_along_axis(a, permutation[:, :, None], axis=1)
        a = np.take_along_axis(a, permutation[:, None, :], axis=2)
        pivot = a[:, 0, 1]
        answer *= np.where(pivot_index == 1, 1., -1.)*pivot
        if width > 2:
            first, second = a[:, 0, 2:], a[:, 1, 2:]
            # A zero pivot means a zero Pfaffian; use a safe denominator.
            denom = np.where(abs(pivot) > 1e-300, pivot, 1.)
            a = a[:, 2:, 2:]+(second[:, :, None]*first[:, None, :]
                 -first[:, :, None]*second[:, None, :])/denom[:, None, None]
    return answer.reshape(shape)


def haar_charts(n, count, rng):
    normal = rng.normal(size=(count, 2*n, 2*n))
    O, R = np.linalg.qr(normal)
    O *= np.where(np.diagonal(R, axis1=1, axis2=2) >= 0, 1., -1.)[:, None, :]
    O[:, :, -1] *= np.linalg.det(O)[:, None]
    U = .5*(O[:, ::2, ::2]+O[:, 1::2, 1::2]+1j*(O[:, 1::2, ::2]-O[:, ::2, 1::2]))
    V = .5*(O[:, ::2, ::2]-O[:, 1::2, 1::2]+1j*(O[:, 1::2, ::2]+O[:, ::2, 1::2]))
    Z = -np.linalg.solve(U, V)
    Z = .5*(Z-Z.transpose(0, 2, 1))
    vacuum = np.exp(.5*np.linalg.slogdet(U)[1])
    return Z, vacuum


def pair_sector_atoms(model, Z, vacuum):
    indices = np.array([[k for j in range(model.L) if mask & (1 << j)
                         for k in (2*j, 2*j+1)] for mask in model.masks])
    minors = Z[:, indices[:, :, None], indices[:, None, :]]
    return pfaffian_batch(minors)*vacuum[:, None]


def sample_atoms(model, count, rng):
    Z, vacuum = haar_charts(2*model.L, count, rng)
    return pair_sector_atoms(model, Z, vacuum), Z, vacuum
