import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.columns import onsite_atom
from pairing_formation.column_prediction import evolve_span, coherent_kernels


class ColumnPredictionTests(unittest.TestCase):
    def test_complete_span_free_transport_matches_exact(self):
        model = picket_fence(4, 2.)
        state = model.evolve([.4])[0]
        times = np.array([0., .13, .7])
        Q = np.eye(model.dim, dtype=complex)
        exact = model.evolve(times, state)
        for transport in ('static', 'free'):
            actual = evolve_span(model, Q, state, times, transport)
            np.testing.assert_allclose(actual, exact, atol=1e-9)

    def test_coherent_kernel_matches_direct_observables(self):
        model = picket_fence(8, 2.)
        rng = np.random.default_rng(86)
        A = np.array([onsite_atom(model, x) for x in rng.normal(size=(5, 16))]).T
        c = rng.normal(size=5)+1j*rng.normal(size=5)
        result = coherent_kernels(model, A, c)
        state = A@c
        direct = model.observe(state/np.linalg.norm(state))
        self.assertAlmostEqual(result['coherent_mean'], direct['mean'], places=12)
        self.assertAlmostEqual(result['coherent_variance'], direct['variance'], places=12)
        self.assertAlmostEqual(result['coherent_cross'][0], direct['cross_real'], places=12)
        self.assertAlmostEqual(result['coherent_cross'][1], direct['cross_imag'], places=12)


if __name__ == '__main__':
    unittest.main()
