import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.costs import repair_minimum_mass, matched_magic_witness


class CostTests(unittest.TestCase):
    def test_fock_dictionary_and_exact_repair(self):
        model = picket_fence(4, 2.)
        state = model.evolve([.7])[0]
        A = np.eye(model.dim, dtype=complex)
        report, c, residual, dual = repair_minimum_mass(A, state)
        np.testing.assert_allclose(A@c+residual, state, atol=1e-15)
        self.assertAlmostEqual(report['mass'], sum(abs(state)), places=7)
        self.assertLess(report['finite_dual_max_fock_overlap'], 1+1e-7)

    def test_embedded_magic_global_witness(self):
        model = picket_fence(4, 2.)
        state = model.initial.copy()
        state[model.index[6]] = 1j
        state /= np.sqrt(2)
        report, W = matched_magic_witness(model, state)
        self.assertAlmostEqual(report['extent_lower_numerical'], 2., places=10)
        self.assertAlmostEqual(abs(np.vdot(W, state))**2, 2., places=10)


if __name__ == '__main__':
    unittest.main()
