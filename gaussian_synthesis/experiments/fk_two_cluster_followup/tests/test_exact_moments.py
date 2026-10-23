"""Light deterministic checks of the algebraic second Gaussian moment."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT.parent/"fk_two_cluster"))

import numpy as np
from scipy.linalg import eigh

from fk.conventions import majoranas, parity_masks
from fk.model import build_model
from fk.solver import solve
from fk2.exact_moments import (ExactK2Moments, _majorana_actions, apply_lambda,
                              project_gaussian_pair, full_tensor_reconstruction,
                              integer_fk_matrix_proof, closed_form_fk_matrix,
                              run_exact_reference)


class ExactMomentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model()
        cls.backend = ExactK2Moments(cls.model)

    def test_majorana_actions_match_independent_dense_convention(self):
        gammas = majoranas(8)
        for parity in (0,1):
            source, destination = parity_masks(8,parity),parity_masks(8,1-parity)
            permutations, phases = _majorana_actions(parity)
            for a in range(16):
                reconstructed = np.zeros((128,128),complex)
                reconstructed[np.arange(128),permutations[a]] = phases[a]
                np.testing.assert_array_equal(reconstructed,gammas[a][np.ix_(destination,source)])

    def test_projector_kernel_idempotence_and_positivity(self):
        diagnostics = self.backend.diagnostics
        self.assertLess(diagnostics["lambda_kernel_residual"],1e-11)
        self.assertLess(diagnostics["projector_idempotence_residual"],1e-11)
        self.assertGreaterEqual(diagnostics["minimum_moment_eigenvalue"],-1e-12)
        self.assertLessEqual(diagnostics["maximum_moment_eigenvalue"],1+1e-12)
        self.assertEqual(diagnostics["frame_constant"],6435)
        self.assertIsNone(self.backend.evaluate(np.ones(3)).mass)

    def test_gaussian_probe_and_nongaussian_projection(self):
        _, vectors = eigh(self.model.reduced_hopping)
        z = vectors[:,0]
        gaussian = self.model.S@z
        pair = np.outer(gaussian,gaussian)
        self.assertLess(np.linalg.norm(apply_lambda(pair)),1e-12)
        np.testing.assert_allclose(project_gaussian_pair(pair),pair,atol=1e-12,rtol=0)
        self.assertAlmostEqual(self.backend.evaluate(z).q,1.,places=12)
        np.testing.assert_allclose(self.backend.reconstruction(z),gaussian,atol=1e-12,rtol=0)
        product_magic = self.model.S[:,0]
        self.assertGreater(np.linalg.norm(apply_lambda(np.outer(product_magic,product_magic))),.1)
        self.assertLess(self.backend.evaluate(np.array([1,0,0])).q,1.)

    def test_gradient_and_hessian(self):
        z = np.array([.31+.12j,-.37+.08j,.23-.19j])
        x = np.concatenate((z.real,z.imag))
        result = self.backend.evaluate(z)
        step = 1e-5
        unpack = lambda value:value[:3]+1j*value[3:]
        gradient = np.zeros(6)
        hessian = np.zeros((6,6))
        for j in range(6):
            direction = np.eye(6)[j]*step
            plus,minus = self.backend.evaluate(unpack(x+direction)),self.backend.evaluate(unpack(x-direction))
            gradient[j] = (plus.q-minus.q)/(8*step)
            hessian[:,j] = (plus.gradient-minus.gradient)/(2*step)
        np.testing.assert_allclose(result.gradient,gradient,atol=1e-9,rtol=1e-7)
        np.testing.assert_allclose(result.hessian,hessian,atol=1e-9,rtol=1e-7)
        self.assertGreaterEqual(np.linalg.eigvalsh(result.hessian)[0],-1e-12)

    def test_solver_compatibility_and_exact_reconstruction(self):
        target = self.model.target(20)
        reduced = self.model.S.conj().T@target
        solution = solve(self.backend,reduced,tolerance=1e-10,max_steps=100)
        self.assertTrue(solution["converged"])
        self.assertLess(np.linalg.norm(self.backend.reconstruction(solution["z"])-target),1e-10)
        q = self.backend.evaluate(solution["z"]).q
        self.assertAlmostEqual(q,float(np.vdot(self.model.S@solution["z"],target).real),places=10)
        direct = full_tensor_reconstruction(self.model.S@solution["z"])
        np.testing.assert_allclose(direct,self.backend.reconstruction(solution["z"]),atol=1e-12,rtol=0)
        self.assertLess(np.linalg.norm(direct-target),1e-10)

    def test_signed_integer_closed_form_and_runner(self):
        proof = integer_fk_matrix_proof()
        self.assertTrue(proof["verified"])
        np.testing.assert_allclose(proof["basis"],self.model.S,atol=1e-14,rtol=0)
        np.testing.assert_allclose(proof["matrix"],closed_form_fk_matrix(),atol=1e-15,rtol=0)
        np.testing.assert_allclose(proof["matrix"],self.backend.matrix,atol=1e-14,rtol=0)
        report,arrays = run_exact_reference(self.model,{"t_values":[0,11,20]})
        self.assertTrue(report["verified"])
        self.assertEqual(len(report["rows"]),3)
        self.assertIn("witness_t20",arrays)


if __name__ == "__main__":
    unittest.main()
