import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.multiorbit import MultiOrbitDynamics, block_objective
from pairing_formation.conditional_dynamics import ConditionalDynamics


class MultiOrbitTests(unittest.TestCase):
    def test_variational_gradient_and_physical_tangent(self):
        model = picket_fence(8, 2.)
        dynamics = MultiOrbitDynamics(model, rank=2)
        rng = np.random.default_rng(417)
        logs = rng.normal(0, .3, (model.r+1, 2, model.L))+1j*rng.normal(0, .3, (model.r+1, 2, model.L))
        target = model.evolve([1.])[0]
        coordinates, _ = dynamics.encode(logs, target)
        rows = dynamics.groups[2]
        x = np.r_[logs[2].real.ravel(), logs[2].imag.ravel()]
        direction = rng.normal(size=len(x)); eps = 1e-6
        loss, grad = block_objective(dynamics.features[rows], target[rows], x, 2)
        finite = (block_objective(dynamics.features[rows], target[rows], x+eps*direction, 2)[0]
                  -block_objective(dynamics.features[rows], target[rows], x-eps*direction, 2)[0])/(2*eps)
        self.assertAlmostEqual(finite, grad@direction, places=9)
        rate, report, physical = dynamics.derivative(0, coordinates, True)
        finite = (dynamics.state(coordinates+eps*rate)-dynamics.state(coordinates-eps*rate))/(2*eps)
        np.testing.assert_allclose(finite, physical, atol=2e-8, rtol=2e-8)
        self.assertLess(abs(report['norm_rate']), 1e-11)
        self.assertLess(abs(report['energy_rate']), 1e-10)

    def test_rank_one_equivalence_and_free_evolution(self):
        rng = np.random.default_rng(271)
        for coupling in (0., 2.):
            model = picket_fence(8, coupling)
            target = picket_fence(8, 2.).evolve([.8])[0]
            logs = rng.normal(0, .2, (model.r+1, 1, model.L))+1j*rng.normal(0, .2, (model.r+1, 1, model.L))
            dynamics = MultiOrbitDynamics(model, 1)
            coordinates, _ = dynamics.encode(logs, target)
            c, z = dynamics.unpack(coordinates)
            previous = ConditionalDynamics(model, True)
            old = np.r_[c.ravel(), z.ravel()]
            np.testing.assert_allclose(dynamics.state(coordinates), previous.state_and_blocks(old)[0], atol=1e-14)
            np.testing.assert_allclose(dynamics.derivative(0, coordinates, True)[2], previous.derivative(0, old, True)[2], atol=1e-12)
        model = picket_fence(8, 0.)
        dynamics = MultiOrbitDynamics(model, 2)
        logs = rng.normal(0, .2, (model.r+1, 2, model.L))+1j*rng.normal(0, .2, (model.r+1, 2, model.L))
        coordinates, _ = dynamics.encode(logs, target)
        elapsed = [0., .1, .3]
        states, _, report = dynamics.evolve(coordinates, elapsed)
        np.testing.assert_allclose(states, model.evolve(elapsed, dynamics.state(coordinates)), atol=3e-10)


if __name__ == '__main__':
    unittest.main()
