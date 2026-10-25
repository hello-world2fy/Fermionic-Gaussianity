import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.columns import onsite_atom, onsite_objective, onsite_to_qp_Z, FullGaussianOracle
from pairing_formation.projected_gaussian import torch_module
from pairing_formation.gaussian import pfaffian_batch


class ColumnTests(unittest.TestCase):
    def test_onsite_gradient_and_full_norm(self):
        model = picket_fence(4, 2.)
        target = model.evolve([.7])[0]
        rng = np.random.default_rng(42)
        x, direction = rng.normal(size=(2, 8))
        f, grad = onsite_objective(model, target, x)
        h = 1e-6
        diff = (onsite_objective(model, target, x+h*direction)[0]-onsite_objective(model, target, x-h*direction)[0])/(2*h)
        self.assertAlmostEqual(diff, grad@direction, places=8)
        a = onsite_atom(model, x)
        self.assertLess(np.linalg.norm(a), 1.)
        oracle = FullGaussianOracle(model)
        torch = torch_module()
        b = oracle.atom_tensor(torch.tensor(onsite_to_qp_Z(model, x))).detach().numpy()
        self.assertAlmostEqual(abs(np.vdot(a, b)), np.linalg.norm(a)*np.linalg.norm(b), places=12)
        np.testing.assert_allclose(abs(a), abs(b), atol=2e-14)

    def test_full_chart_gradient_and_fock_normalization(self):
        from itertools import combinations
        model = picket_fence(4, 2.)
        oracle = FullGaussianOracle(model)
        rng = np.random.default_rng(81)
        x = rng.normal(size=56)*.15
        v = rng.normal(size=56)
        target = model.evolve([.9])[0]
        _, grad = oracle.evaluate(x, target)
        h = 1e-6
        finite = (oracle.evaluate(x+h*v, target)[0]-oracle.evaluate(x-h*v, target)[0])/(2*h)
        self.assertAlmostEqual(finite, grad@v, places=8)
        atom, Z = oracle.evaluate(x, target, save=True)
        vacuum = np.exp(-.25*np.linalg.slogdet(np.eye(8)+Z.conj().T@Z)[1])
        norm = vacuum**2
        for number in range(2, 9, 2):
            ids = np.array(list(combinations(range(8), number)))
            norm += vacuum**2*np.sum(abs(pfaffian_batch(Z[ids[:, :, None], ids[:, None, :]]))**2)
        self.assertAlmostEqual(norm, 1., places=12)
        self.assertLess(np.linalg.norm(atom), 1.)


if __name__ == '__main__':
    unittest.main()
