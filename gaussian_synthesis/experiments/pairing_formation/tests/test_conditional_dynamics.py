import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.conditional_dynamics import ConditionalDynamics


class ConditionalDynamicsTests(unittest.TestCase):
    def test_physical_derivative_and_conservation(self):
        model = picket_fence(8, 2.)
        rng = np.random.default_rng(1928)
        for conditioned in (False, True):
            dynamics = ConditionalDynamics(model, conditioned)
            z = .2*(rng.normal(size=dynamics.shape)+1j*rng.normal(size=dynamics.shape))
            initial, _ = dynamics.encode(z, model.evolve([.7])[0])
            rate, report, physical_rate = dynamics.derivative(0., initial, diagnostics=True)
            eps = 2e-6
            finite = (dynamics.state_and_blocks(initial+eps*rate)[0]-dynamics.state_and_blocks(initial-eps*rate)[0])/(2*eps)
            np.testing.assert_allclose(finite, physical_rate, atol=3e-10, rtol=3e-9)
            self.assertLess(abs(report['norm_rate']), 1e-12)
            self.assertLess(abs(report['energy_rate']), 1e-11)
            # The residual is orthogonal to each complex tangent direction.
            state, blocks = dynamics.state_and_blocks(initial)
            residual = 1j*physical_rate-model.H@state
            for rows, q, mean, centered, metric in blocks:
                self.assertLess(abs(np.vdot(q, residual[rows])), 1e-12)
            if conditioned:
                for rows, q, mean, centered, metric in blocks:
                    self.assertLess(np.linalg.norm((q.conj()*residual[rows])@centered), 1e-11)

    def test_one_body_transport_exact(self):
        model = picket_fence(8, 0.)
        rng = np.random.default_rng(12)
        for conditioned in (False, True):
            dynamics = ConditionalDynamics(model, conditioned)
            z = .3*(rng.normal(size=dynamics.shape)+1j*rng.normal(size=dynamics.shape))
            template = rng.normal(size=model.dim)+1j*rng.normal(size=model.dim)
            template /= np.linalg.norm(template)
            initial, _ = dynamics.encode(z, template)
            elapsed = np.linspace(0., .5, 4)
            states, _, report = dynamics.evolve(initial, elapsed)
            exact = model.evolve(elapsed, dynamics.state_and_blocks(initial)[0])
            np.testing.assert_allclose(states, exact, atol=3e-10)
            self.assertLess(report['max_energy_drift'], 1e-10)


if __name__ == '__main__':
    unittest.main()
