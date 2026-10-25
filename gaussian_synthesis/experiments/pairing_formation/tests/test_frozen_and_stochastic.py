import sys
from pathlib import Path
import unittest
from itertools import product
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.frozen import channel_basis, propagate
from pairing_formation.stochastic import numpy_rhs


class FrozenTests(unittest.TestCase):
    def test_complete_channels_recover_exact_future(self):
        model = picket_fence(8, 2.)
        state = model.evolve([.4])[0]
        T = state[model.blocks[1][1]]
        Q, _ = channel_basis(model, T, model.r)
        np.testing.assert_allclose(Q.conj().T@Q, np.eye(model.dim), atol=5e-14)
        H = Q.conj().T@(model.H@Q)
        predicted = propagate(H, Q.conj().T@state, [.2, .7])@Q.T
        truth = model.evolve([.2, .7], initial=state)
        np.testing.assert_allclose(predicted, truth, atol=2e-13)

    def test_dtwa_short_time_limitation_is_not_hidden(self):
        model = picket_fence(4, 2.)
        sample = np.zeros((4**model.L, model.L, 3))
        sample[..., :2] = np.array(list(product((-.5, .5), repeat=2*model.L))).reshape(-1, model.L, 2)
        sample[:, :model.r, 2], sample[:, model.r:, 2] = .5, -.5
        dt = 1e-4
        def rhs(s):
            return numpy_rhs(s, model.energies, model.g)
        k1 = rhs(sample)
        k2 = rhs(sample+.5*dt*k1)
        k3 = rhs(sample+.5*dt*k2)
        k4 = rhs(sample+dt*k3)
        s = sample+dt/6*(k1+2*k2+2*k3+k4)
        z = s[:, model.r:, 2]
        m = z.sum(axis=1)+model.r/2
        variance = np.mean(m*m+(.25-z*z).sum(axis=1))-m.mean()**2
        exact_leading = (model.G*dt/2)**2
        self.assertAlmostEqual(variance/exact_leading, (model.r+1)/(2*model.r), delta=1e-7)


if __name__ == '__main__':
    unittest.main()
