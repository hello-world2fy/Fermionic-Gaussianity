import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.fitting import excitation_features
from pairing_formation.conditional_dynamics import ConditionalDynamics
from pairing_formation.jastrow import JastrowDynamics, radial_objective


class JastrowTests(unittest.TestCase):
    def test_gradient_and_physical_projection(self):
        model = picket_fence(8, 2.)
        rng = np.random.default_rng(946)
        target = rng.normal(size=model.dim)+1j*rng.normal(size=model.dim)
        target /= np.linalg.norm(target)
        for family in ('collective_1', 'two_body'):
            dynamics = JastrowDynamics(model, family)
            d = dynamics.shape[1]
            x = rng.normal(0, .08, 2*d)
            value, gradient = radial_objective(dynamics, target, x)
            direction = rng.normal(size=2*d)
            eps = 1e-6
            finite = (radial_objective(dynamics, target, x+eps*direction)[0]
                      -radial_objective(dynamics, target, x-eps*direction)[0])/(2*eps)
            self.assertAlmostEqual(finite, gradient@direction, places=9)
            initial, _ = dynamics.encode(x[:d]+1j*x[d:], target)
            rate, report, physical = dynamics.derivative(0, initial, diagnostics=True)
            finite = (dynamics.state_and_blocks(initial+eps*rate)[0]
                      -dynamics.state_and_blocks(initial-eps*rate)[0])/(2*eps)
            np.testing.assert_allclose(finite, physical, atol=2e-9)
            self.assertLess(abs(report['norm_rate']), 1e-12)
            self.assertLess(abs(report['energy_rate']), 1e-11)
            state, data = dynamics.state_and_blocks(initial)
            residual = 1j*physical-model.H@state
            tangent_overlap = sum(initial[m].conjugate()*(q.conj()*residual[rows])@centered
                                  for m, (rows, q, mean, centered, metric) in enumerate(data))
            self.assertLess(np.linalg.norm(tangent_overlap), 1e-11)

    def test_known_subfamilies_and_exact_free_motion(self):
        model = picket_fence(8, 0.)
        rng = np.random.default_rng(538)
        common = ConditionalDynamics(model, False)
        z = .2*(rng.normal(size=model.L)+1j*rng.normal(size=model.L))
        template = rng.normal(size=model.dim)+1j*rng.normal(size=model.dim)
        template /= np.linalg.norm(template)
        initial, _ = common.encode(z, template)
        state = common.state_and_blocks(initial)[0]
        for family in ('collective_0', 'collective_1', 'collective_2', 'two_body'):
            dynamics = JastrowDynamics(model, family)
            theta, error = dynamics.project_logarithms(excitation_features(model)@z)
            self.assertLess(error, 1e-12)
            coordinates, fidelity = dynamics.encode(theta, state)
            np.testing.assert_allclose(dynamics.state_and_blocks(coordinates)[0], state, atol=1e-14)
            phase = model.occ@rng.normal(size=model.L)
            kicked = dynamics.apply_diagonal_phase(coordinates, phase)
            np.testing.assert_allclose(dynamics.state_and_blocks(kicked)[0], state*np.exp(-1j*phase), atol=1e-14)
            elapsed = np.linspace(0, .3, 3)
            states, _, report = dynamics.evolve(kicked, elapsed)
            exact = model.evolve(elapsed, state*np.exp(-1j*phase))
            np.testing.assert_allclose(states, exact, atol=2e-10)
        # The entire linear-in-m orbital exponent is a two-body density Jastrow.
        raw = excitation_features(model)@z*(2*model.m/model.r-1)
        self.assertLess(JastrowDynamics(model, 'two_body').project_logarithms(raw)[1], 1e-12)


if __name__ == '__main__':
    unittest.main()
