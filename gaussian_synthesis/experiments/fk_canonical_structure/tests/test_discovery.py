"""Independent differential, Gaussian-geometry and repaired-cost checks."""
import unittest
from unittest.mock import patch

import numpy as np

from fk.gaussian import sample_gaussian
from fk.model import build_model
from fk.observables import covariance
from fk.baselines import solve_projected_atomic_l1
from fk.certificates import repaired_cost
from fk3.discovery import (covariance_from_kets, optimize_gaussian_overlap,
                           overlap_gradient, rotate_plane, phase_align)


class DiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model()

    def test_overlap_derivatives_against_finite_differences(self):
        rng = np.random.default_rng(937)
        kets = sample_gaussian(8, 2, rng).kets
        witness = rng.normal(size=128)+1j*rng.normal(size=128)
        witness /= np.linalg.norm(witness)
        gradient = overlap_gradient(witness, kets)
        pairs = [(a, b) for a in range(16) for b in range(a+1, 16)]
        for index in (0, 7, 14, 35, 72, 119):
            a, b = pairs[index]
            eps = 2e-6
            plus = abs(rotate_plane(kets, a, b, eps)@witness.conj())**2
            minus = abs(rotate_plane(kets, a, b, -eps)@witness.conj())**2
            np.testing.assert_allclose((plus-minus)/(2*eps), gradient[:, index], atol=3e-11)

    def test_rotations_match_independent_dense_covariance(self):
        rng = np.random.default_rng(938)
        sample = sample_gaussian(8, 2, rng)
        kets = rotate_plane(sample.kets, 3, 12, np.array([.3, -.4]))
        cov = covariance_from_kets(kets)
        for ket, computed in zip(kets, cov):
            np.testing.assert_allclose(computed, covariance(ket, self.model), atol=5e-13)
        np.testing.assert_allclose(cov@cov, np.broadcast_to(-np.eye(16), cov.shape), atol=8e-13)
        np.testing.assert_allclose(np.linalg.norm(kets, axis=1), 1, atol=5e-13)

    def test_guided_is_monotone_phase_invariant_and_never_uses_prior(self):
        rng = np.random.default_rng(939)
        starts = sample_gaussian(8, 2, rng).kets
        witness = self.model.target(11)
        # Any accidental call to the physical prior fails this semantic test.
        with patch('fk3.discovery.physics_arc_atoms', side_effect=AssertionError('prior leak')):
            kets, diagnostics = optimize_gaussian_overlap(witness, starts, max_sweeps=2)
            phased, _ = optimize_gaussian_overlap(witness*np.exp(.42j),
                                                  starts*np.exp(-.31j), max_sweeps=2)
        trace = diagnostics['objective_trace_normalized_witness']
        self.assertGreaterEqual(np.diff(trace, axis=0).min(), -1e-12)
        self.assertGreater((trace[-1]-trace[0]).min(), .01)
        np.testing.assert_allclose(abs(np.sum(kets.conj()*phased, axis=1)), 1, atol=2e-12)
        replay = starts.copy()
        for angles in diagnostics['rotation_angles']:
            for p, (a, b) in enumerate(diagnostics['rotation_pairs']):
                replay = rotate_plane(replay, int(a), int(b), angles[:, p])
        np.testing.assert_allclose(replay, kets, atol=2e-12)
        aligned, phases = phase_align(witness, kets)
        np.testing.assert_allclose(aligned, kets*phases[:, None], atol=1e-15)
        self.assertLess(abs((aligned@witness.conj()).imag).max(), 1e-13)
        self.assertFalse(diagnostics['globally_optimized'])

    def test_full_residual_repair_and_projected_cost_semantics(self):
        target = self.model.target(0)
        solved = solve_projected_atomic_l1(self.model.S, np.eye(128), target)
        self.assertAlmostEqual(solved['coefficient_l1'], 2, places=7)
        reconstructed = self.model.S@(solved['projected_atoms']@solved['coefficients'])
        np.testing.assert_allclose(solved['residual'], target-reconstructed, atol=1e-15)
        self.assertLess(np.linalg.norm(solved['residual']), 1e-8)
        # Fock residual components are explicit Gaussian repairs, including
        # any component perpendicular to the invariant subspace.
        residual = np.zeros(128, dtype=complex)
        residual[0], residual[3] = .03j, -.04
        cost = repaired_cost(solved['coefficient_l1'], residual, target_error=.002)
        self.assertAlmostEqual(cost, solved['coefficient_l1']+.07+np.sqrt(128)*.002)


if __name__ == '__main__':
    unittest.main()
