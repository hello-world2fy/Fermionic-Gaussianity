import _bootstrap  # noqa: F401

import math
import unittest
import numpy as np

from gaussian_synthesis.gaussian import quadric_matrix
from gaussian_synthesis.sampling import HaarFourModeSampler
from gaussian_synthesis.solver import ResourceLimitError
from gaussian_synthesis.theory import dimension
from exact_solver import exact_hessian, exact_k2, solve_exact
from reference import (basis, cat_extent, cat_state, cat_witness,
                       gaussianity_defect, quadratic_state, quadratic_unitary,
                       quench_reference, quench_state)


class ExactSolverTests(unittest.TestCase):
    def test_normalization_and_complex_directional_derivatives(self):
        self.assertEqual(dimension(4, 2), 35)
        rng = np.random.default_rng(75013)
        j = quadric_matrix()
        for _ in range(8):
            u = (rng.normal(size=8)+1j*rng.normal(size=8))/3
            v = (rng.normal(size=8)+1j*rng.normal(size=8))/3
            q, c = exact_k2(u)
            self.assertAlmostEqual(q, np.linalg.norm(u)**4-abs(u@j@u)**2/8)
            self.assertAlmostEqual(q, np.vdot(u, c).real)
            epsilon = 2e-6
            q_plus, c_plus = exact_k2(u+epsilon*v)
            q_minus, c_minus = exact_k2(u-epsilon*v)
            self.assertAlmostEqual((q_plus-q_minus)/(8*epsilon), np.vdot(c, v).real, places=7)
            h = exact_hessian(u)
            np.testing.assert_allclose(h, h.T, atol=2e-14)
            self.assertGreater(np.linalg.eigvalsh(h)[0], 0)
            predicted = h@np.r_[v.real, v.imag]
            derivative = (c_plus-c_minus)/(2*epsilon)
            np.testing.assert_allclose(predicted, np.r_[derivative.real, derivative.imag], atol=2e-8)

    def test_recover_manufactured_full_complex_solutions(self):
        rng = np.random.default_rng(5410)
        for scale in (.2, .7, 1.0, 2.):
            for _ in range(3):
                truth = rng.normal(size=8)+1j*rng.normal(size=8)
                truth *= scale/np.linalg.norm(truth)
                _, psi = exact_k2(truth)
                result = solve_exact(psi, tolerance=1e-12)
                self.assertEqual(result.status, "converged")
                self.assertLessEqual(result.residual, 1e-12)
                np.testing.assert_allclose(result.u, truth, atol=2e-11)
                self.assertLess(result.iterations, 30)

    def test_zero_target_iteration_and_budget_status(self):
        result = solve_exact(np.zeros(8))
        self.assertEqual((result.status, result.iterations, result.residual), ("converged", 0, 0))
        result = solve_exact(cat_state(.2), max_iterations=0)
        self.assertEqual(result.status, "max_iterations")
        self.assertEqual(result.iterations, 0)

        class ExpiredBudget:
            def check(self):
                raise ResourceLimitError("test budget")

        result = solve_exact(cat_state(.2), budget=ExpiredBudget())
        self.assertEqual(result.status, "resource_limit")
        self.assertTrue(np.isfinite(result.residual))


class StaticReferenceTests(unittest.TestCase):
    def test_cat_family_stationarity_and_direct_solve(self):
        for r in (0, 1e-12, .01, .1, .25, .49, .5):
            with self.subTest(r=r):
                psi, u = cat_state(r), cat_witness(r)
                self.assertAlmostEqual(np.linalg.norm(psi), 1.)
                self.assertAlmostEqual(abs(gaussianity_defect(psi)), math.sqrt(r*(1-r)))
                _, c = exact_k2(u)
                np.testing.assert_allclose(c, psi, atol=5e-15)
                direct = solve_exact(psi)
                self.assertEqual(direct.status, "converged")
                np.testing.assert_allclose(direct.u, u, atol=2e-10)
                self.assertAlmostEqual(cat_extent(r), (math.sqrt(1-r)+math.sqrt(r))**2)

    def test_point_density_ratio_and_endpoint_proportionality(self):
        u = cat_witness(.1)
        self.assertAlmostEqual(float(u[0].real), .94464, places=5)
        self.assertAlmostEqual(float(u[7].real), .38634, places=5)
        self.assertAlmostEqual((abs(u[0])/abs(u[7]))**3, 14.62, delta=.01)
        psi = cat_state(.1)
        self.assertAlmostEqual((abs(psi[0])/abs(psi[7]))**3, 27.)
        np.testing.assert_array_equal(cat_witness(0), cat_state(0))
        np.testing.assert_allclose(cat_witness(.5), (8/7)**(1/3)*cat_state(.5), atol=1e-15)
        self.assertEqual(cat_extent(0), 1.)
        self.assertEqual(cat_extent(.5), 2.)

    def test_invalid_reference_parameters(self):
        for bad in (-.01, .51, math.inf, math.nan, True, .1j):
            with self.subTest(r=bad), self.assertRaises(ValueError):
                cat_state(bad)
        for bad in (math.inf, math.nan, True, 1j):
            with self.subTest(phi=bad), self.assertRaises(ValueError):
                quench_state(bad)


class DynamicalReferenceTests(unittest.TestCase):
    def test_schmidt_mapping_unitarity_quadric_and_full_witness(self):
        j = quadric_matrix()
        sampler = HaarFourModeSampler(np.random.default_rng(8196), basis)
        gaussian_samples = sampler.sample(24)
        phases = (0, 1e-12, .37, math.pi/2, math.pi-1e-10, math.pi,
                  math.pi+1e-10, 3*math.pi/2, 2*math.pi-1e-12, 2*math.pi, -1.7)
        for phi in phases:
            with self.subTest(phi=phi):
                ref = quench_reference(phi)
                u = ref.unitary
                np.testing.assert_allclose(u.conj().T@u, np.eye(8), atol=2e-14)
                np.testing.assert_allclose(u@cat_state(ref.r), ref.state, atol=2e-14)
                transformed_j = u.T@j@u
                phase = transformed_j[0, 7]
                self.assertAlmostEqual(abs(phase), 1.)
                np.testing.assert_allclose(transformed_j, phase*j, atol=2e-14)
                transformed = gaussian_samples@u.T
                np.testing.assert_allclose(np.einsum("bi,ij,bj->b", transformed, j, transformed), 0, atol=2e-14)
                _, c = exact_k2(ref.witness)
                np.testing.assert_allclose(c, ref.state, atol=2e-14)
                direct = solve_exact(ref.state)
                self.assertEqual(direct.status, "converged")
                np.testing.assert_allclose(direct.u, ref.witness, atol=2e-10)
                self.assertAlmostEqual(cat_extent(ref.r), 1+abs(math.sin(phi/2)))

    def test_quench_from_hamiltonian_defect_and_periodicity(self):
        initial = quench_state(0)
        energies = np.array([int(bool(mask & 1) and bool(mask & 4))
                             for mask in basis.occupations])
        for phi in (0, .37, math.pi/2, math.pi, 5.3, 2*math.pi):
            state = quench_state(phi)
            np.testing.assert_allclose(state, initial*np.exp(-1j*phi*energies), atol=1e-15)
            self.assertAlmostEqual(gaussianity_defect(state), (np.exp(-1j*phi)-1)/4)
            np.testing.assert_allclose(quench_state(phi+2*math.pi), state, atol=2e-15)
            np.testing.assert_allclose(quench_reference(phi+2*math.pi).witness,
                                       quench_reference(phi).witness, atol=2e-14)

    def test_quadratic_evolution_stays_gaussian(self):
        energies = np.array([0, 3, 4, 5, 5, 6, 7, 10])
        for phi in (0, .137, math.pi/2, math.pi, 2*math.pi):
            u = quadratic_unitary(phi)
            np.testing.assert_allclose(u, np.diag(np.exp(-1j*phi*energies)), atol=1e-15)
            state = quadratic_state(phi)
            self.assertLess(abs(gaussianity_defect(state)), 2e-15)
            _, c = exact_k2(state)
            np.testing.assert_allclose(c, state, atol=2e-15)


if __name__ == "__main__":
    unittest.main()
