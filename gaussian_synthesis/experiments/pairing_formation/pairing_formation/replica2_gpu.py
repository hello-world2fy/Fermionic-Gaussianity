"""Full-Gaussian second Cartan moment on a GPU, through 14 pair levels.

The blocks are the full-Fock Cartan projector compressed after evaluation,
as in replica2.py. This is numerical tensor contraction, not Monte Carlo and
not interval certification. A supplied invariant Q reduces witness variables,
never the Gaussian dictionary. The CPU reference implementation is unchanged.
"""
from math import comb
import numpy as np
from scipy.special import beta
from scipy.optimize import minimize
from .projected_gaussian import torch_module


class Replica2GPU:
    def __init__(self, model, device='cuda:0'):
        if model.L > 14:
            raise ValueError('This dense pair-index implementation is limited to L<=14')
        self.model = model
        self.torch = torch_module()
        self.torch.set_num_threads(1)
        self.device = device
        D = model.dim
        aa = np.repeat(np.arange(D, dtype=np.int32), D)
        bb = np.tile(np.arange(D, dtype=np.int32), D)
        ma, mb = model.masks[aa], model.masks[bb]
        difference = ma ^ mb
        keys = (ma & mb) | (difference << model.L)
        sizes = np.bitwise_count(difference)
        self.blocks = []
        self.block_shapes = []
        for s in range(0, model.L + 1, 2):
            chosen = np.flatnonzero(sizes == s)
            chosen = chosen[np.lexsort((ma[chosen], keys[chosen]))]
            width = comb(s, s // 2)
            a, b = aa[chosen].reshape(-1, width), bb[chosen].reshape(-1, width)
            masks = model.masks[a[0]]
            flipped = np.bitwise_count(masks[:, None] ^ masks[None, :])
            if len(a) > 1:
                last = model.masks[a[-1]]
                assert np.array_equal(flipped, np.bitwise_count(last[:, None] ^ last[None, :]))
            kernel = beta(flipped + .5, s - flipped + .5) / np.pi
            self.blocks.append((self.torch.as_tensor(a.astype(np.int64), device=device),
                                self.torch.as_tensor(b.astype(np.int64), device=device),
                                self.torch.as_tensor(kernel, device=device, dtype=self.torch.complex128)))
            self.block_shapes.append({'different_levels': s, 'rows': len(a), 'width': width})

    def tensor_evaluate(self, u):
        C = self.torch.zeros_like(u)
        for a, b, K in self.blocks:
            action = (u[a] * u[b]) @ K.T
            C.index_add_(0, a.flatten(), (action * u[b].conj()).flatten())
        return self.torch.vdot(u, C).real, C

    def evaluate(self, u):
        vector = self.torch.as_tensor(np.asarray(u), device=self.device, dtype=self.torch.complex128)
        q, C = self.tensor_evaluate(vector)
        return float(q.item()), C.cpu().numpy()

    def derivative(self, u, v):
        torch = self.torch
        u = torch.as_tensor(np.asarray(u), device=self.device, dtype=torch.complex128)
        v = torch.as_tensor(np.asarray(v), device=self.device, dtype=torch.complex128)
        result = torch.zeros_like(u)
        for a, b, K in self.blocks:
            action = (u[a]*u[b]) @ K.T
            change = (v[a]*u[b] + u[a]*v[b]) @ K.T
            contribution = change*u[b].conj() + action*v[b].conj()
            result.index_add_(0, a.flatten(), contribution.flatten())
        return result.cpu().numpy()

    def solve(self, target, Q=None, maxiter=500):
        torch = self.torch
        target = np.asarray(target, complex)
        Q = np.eye(len(target)) if Q is None else np.asarray(Q)
        np.testing.assert_allclose(Q.conj().T @ Q, np.eye(Q.shape[1]), atol=2e-12)
        coordinates = Q.conj().T @ target
        symmetry_error = float(np.linalg.norm(Q @ coordinates - target))
        if symmetry_error > 2e-11:
            raise ValueError('Target is outside supplied witness symmetry space')
        basis = torch.as_tensor(Q, device=self.device, dtype=torch.complex128)
        t = torch.as_tensor(target, device=self.device)
        q, _ = self.evaluate(target)
        initial = coordinates / q**(1/3)
        n = len(initial)
        history = []
        def objective(x):
            c = torch.as_tensor(x[:n]+1j*x[n:], device=self.device)
            u = basis @ c
            value, C = self.tensor_evaluate(u)
            residual = C-t
            gradient = (basis.conj().T @ residual).cpu().numpy()
            objective_value = float((value/4-torch.vdot(u,t).real).item())
            return objective_value, np.r_[gradient.real, gradient.imag]
        def callback(x):
            value, gradient = objective(x)
            history.append({'objective': value, 'reduced_residual': float(np.linalg.norm(gradient))})
        result = minimize(objective, np.r_[initial.real, initial.imag], jac=True,
                          method='L-BFGS-B', callback=callback,
                          options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 2e-10, 'maxcor': 20})
        u = Q @ (result.x[:n]+1j*result.x[n:])
        q, C = self.evaluate(u)
        return u, {'k': 2, 'full_gaussian_dictionary': True,
            'physical_modes': 2*self.model.L, 'witness_coordinates': n,
            'target_symmetry_error': symmetry_error, 'q': q,
            'residual': float(np.linalg.norm(C-target)),
            'iterations': int(result.nit), 'optimizer_success': bool(result.success),
            'moment_dual_extent_lower': float(abs(np.vdot(u,target))**2/np.sqrt(q)),
            'float64_gpu_not_interval_certificate': True, 'history': history,
            'block_shapes': self.block_shapes}
