import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.fitting import excitation_features
from pairing_formation.multiorbit import MultiOrbitDynamics
from pairing_formation.orbit_response import ensemble_objective, response_basis, tangent_response_score


class OrbitResponseTests(unittest.TestCase):
    def test_multiple_target_variable_projection_gradient(self):
        rng = np.random.default_rng(20261120)
        model = picket_fence(8, 2.)
        F = excitation_features(model)[model.m == 2]
        target = rng.normal(size=(len(F), 3))+1j*rng.normal(size=(len(F), 3))
        target /= np.linalg.norm(target)
        x = rng.normal(0, .3, 4*model.L)
        loss, gradient = ensemble_objective(F, target, x, 2)
        eps = 1e-6
        numeric = []
        for j in range(len(x)):
            direction = np.zeros(len(x)); direction[j] = eps
            numeric.append((ensemble_objective(F, target, x+direction, 2)[0]-ensemble_objective(F, target, x-direction, 2)[0])/(2*eps))
        np.testing.assert_allclose(gradient, numeric, atol=3e-9, rtol=2e-5)
        # The state-averaged projector loss is invariant under a target-space unitary.
        unitary = np.linalg.qr(rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3)))[0]
        rotated, rotated_gradient = ensemble_objective(F, target@unitary, x, 2)
        self.assertAlmostEqual(loss, rotated)
        np.testing.assert_allclose(gradient, rotated_gradient, atol=1e-14)

    def test_response_operators_and_tangent_score(self):
        model = picket_fence(8, 2.)
        target = model.evolve([1.])[0]
        u, operators, report = response_basis(model, target, 'generators')
        np.testing.assert_allclose(np.column_stack([H@target for H in operators]), u, atol=1e-14)
        np.testing.assert_allclose(u.conj().T@u, np.eye(report['rank']), atol=1e-14)
        np.testing.assert_allclose(target.conj()@u, 0., atol=1e-14)
        rng = np.random.default_rng(72)
        logs = rng.normal(0, .3, (model.r+1, 2, model.L))+1j*rng.normal(0, .3, (model.r+1, 2, model.L))
        dynamics = MultiOrbitDynamics(model, 2)
        coordinates, _ = dynamics.encode(logs, target)
        measured = tangent_response_score(dynamics, coordinates, [model.H])
        _, diagnostic, _ = dynamics.derivative(0., coordinates, True)
        self.assertAlmostEqual(measured['score'], diagnostic['residual_norm']**2, delta=1e-12)


if __name__ == '__main__':
    unittest.main()
