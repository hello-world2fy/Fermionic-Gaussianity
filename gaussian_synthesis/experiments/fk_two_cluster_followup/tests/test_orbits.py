"""Small algebraic checks for explicit finite Spin(7) orbit expansions."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent/"fk_two_cluster"))

from fk.model import build_model
from fk.observables import covariance
from fk2.orbits import (DiagonalSpin7, covariance_from_kets, simplex_directions,
                        compress_projected_rays, solve_finite_with_fock_repair)


class FiniteOrbitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model()

    def test_simplex_first_two_moments(self):
        directions = simplex_directions()
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-14)
        np.testing.assert_allclose(directions.mean(axis=0), 0., atol=1e-14)
        np.testing.assert_allclose(directions.T@directions/8, np.eye(7)/7, atol=1e-14)

    def test_actual_covariance_and_phase_preserving_lift(self):
        model = self.model
        atom = np.eye(128, dtype=complex)[59]
        phase = np.exp(.371j)
        action = DiagonalSpin7(model)
        directions = simplex_directions()
        rotated = np.array([action.act(atom, v)[0] for v in directions])
        phased = np.array([action.act(phase*atom, v)[0] for v in directions])
        np.testing.assert_allclose(phased, phase*rotated, atol=2e-14)
        actual_covariances = covariance_from_kets(rotated)
        np.testing.assert_allclose(actual_covariances[0], covariance(rotated[0], model), atol=2e-14)
        np.testing.assert_allclose(actual_covariances@actual_covariances,
                                   np.broadcast_to(-np.eye(16), (8, 16, 16)), atol=2e-14)
        projected = model.S@(model.S.conj().T@atom)
        np.testing.assert_allclose(rotated.mean(axis=0), projected, atol=2e-14)

    def test_projection_ray_merge_preserves_norm_and_original_atoms(self):
        model = self.model
        atoms = np.eye(128, dtype=complex)[[59, 68]]
        arrays = {"full_atoms": atoms, "full_t20_coefficients": np.array([.03, .04])}
        compressed = compress_projected_rays(model, arrays)
        self.assertEqual(len(compressed["indices"]), 1)
        self.assertEqual(compressed["groups"], [[0, 1]])
        self.assertAlmostEqual(np.linalg.norm(compressed["atoms"][0]), 1.)
        self.assertLess(np.linalg.norm(compressed["projected_atoms"][:, 0]), .6)
        self.assertAlmostEqual(abs(compressed["frozen_merged_coefficients"][0]), .07)

    def test_finite_fit_repairs_missing_coordinates(self):
        atom = np.array([[1., 0., 0.]])
        target = np.array([.5, .3j, -.2])
        solved = solve_finite_with_fock_repair(atom, target)
        reconstruction = atom.T@solved["coefficients"]+solved["residual"]
        np.testing.assert_allclose(reconstruction, target, atol=1e-13)
        self.assertAlmostEqual(solved["objective"], 1., places=7)


if __name__ == "__main__":
    unittest.main()
