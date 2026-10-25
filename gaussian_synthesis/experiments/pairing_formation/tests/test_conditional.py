import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.fitting import excitation_features
from pairing_formation.conditional import (product_objective, fit_conditional_products,
                                         conditioned_tangent_span, energy_resolved_observables)


class ConditionalTests(unittest.TestCase):
    def test_complex_gradient_and_independent_block_recovery(self):
        model = picket_fence(8, 2.)
        rng = np.random.default_rng(182)
        F = excitation_features(model)
        target = rng.normal(size=model.dim)+1j*rng.normal(size=model.dim)
        target /= np.linalg.norm(target)
        x, direction = rng.normal(size=(2, 2*model.L))*.3
        loss, gradient = product_objective(F, target, x)
        eps = 2e-6
        finite = (product_objective(F, target, x+eps*direction)[0]-product_objective(F, target, x-eps*direction)[0])/(2*eps)
        self.assertAlmostEqual(finite, gradient@direction, places=9)
        for m in range(model.r+1):
            rows = np.flatnonzero(model.m == m)
            z = rng.normal(size=model.L)*.2+1j*rng.normal(size=model.L)*.3
            target[rows] = (1+.2j*m)*np.exp(F[rows]@z)
        target /= np.linalg.norm(target)
        report, basis, parameters, approximation, preserved = fit_conditional_products(model, target, starts=3)
        self.assertGreater(report['fidelity'], 1-1e-10)
        np.testing.assert_allclose(model.observe(preserved)['count'], model.observe(target)['count'], atol=1e-13)

    def test_tangent_contains_density_responses_and_energy_identity(self):
        model = picket_fence(8, 2.)
        state = model.evolve([.7])[0]
        _, basis, _, _, _ = fit_conditional_products(model, state, starts=3)
        Q, dimensions = conditioned_tangent_span(model, basis, order=1)
        np.testing.assert_allclose(Q.conj().T@Q, np.eye(Q.shape[1]), atol=2e-14)
        for column in range(basis.shape[1]):
            response = model.occ*basis[:, column, None]
            np.testing.assert_allclose(Q@(Q.conj().T@response), response, atol=1e-13)
        observable = energy_resolved_observables(model, state)
        free_energy = 2*(abs(state)**2@model.occ)@model.energies-model.energies.sum()
        initial_energy = 2*model.initial.real@model.occ@model.energies-model.energies.sum()
        self.assertAlmostEqual(observable['energy_transfer_mean'], free_energy-initial_energy, places=13)


if __name__ == '__main__':
    unittest.main()
