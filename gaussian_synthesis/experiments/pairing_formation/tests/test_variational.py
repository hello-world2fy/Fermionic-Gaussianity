import sys
from pathlib import Path
import unittest
from itertools import combinations
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.gaussian import pfaffian_batch
from pairing_formation.projected_gaussian import torch_module, torch_pfaffian, number_norm_squared
from pairing_formation.coupled_cluster import objective_T


class VariationalTests(unittest.TestCase):
    def test_full_number_norm_includes_broken_pairs(self):
        torch = torch_module()
        rng = np.random.default_rng(210)
        Z = rng.normal(size=(8, 8))+1j*rng.normal(size=(8, 8))
        Z = (Z-Z.T)/5
        ids = np.array(list(combinations(range(8), 4)))
        minors = Z[ids[:, :, None], ids[:, None, :]]
        exact = np.sum(abs(pfaffian_batch(minors))**2)
        tz = torch.tensor(Z, dtype=torch.complex128, requires_grad=True)
        norm2 = number_norm_squared(tz, 2, torch)
        self.assertAlmostEqual(float(norm2.detach()), exact, places=12)
        np.testing.assert_allclose(torch_pfaffian(torch.tensor(minors), torch).detach().numpy(),
                                   pfaffian_batch(minors), atol=1e-13)
        norm2.backward()
        dz = rng.normal(size=(8, 8))+1j*rng.normal(size=(8, 8))
        dz = dz-dz.T
        h = 1e-6
        def f(x):
            return number_norm_squared(torch.tensor(x), 2, torch).item()
        derivative = (f(Z+h*dz)-f(Z-h*dz))/(2*h)
        predicted = np.vdot(tz.grad.numpy(), dz).real
        self.assertAlmostEqual(derivative, predicted, delta=1e-6)

    def test_cc_reverse_derivative(self):
        model = picket_fence(8, 2.)
        state = model.evolve([.7])[0]
        rng = np.random.default_rng(100)
        T = .1*(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)))
        V = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
        _, grad, _ = objective_T(model, state, T)
        h = 1e-6
        derivative = (objective_T(model, state, T+h*V)[0]-objective_T(model, state, T-h*V)[0])/(2*h)
        self.assertAlmostEqual(derivative, 2*np.vdot(grad, V).real, delta=1e-7)


if __name__ == '__main__':
    unittest.main()
