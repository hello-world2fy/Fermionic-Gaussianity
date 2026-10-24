import unittest
import numpy as np

from fk4 import paths
from fk4.certificates import interval_q2, certify_order2_normalization, exact_basis_action
from fk.model import build_model
from fk2.exact_moments import ExactK2Moments, integer_fk_matrix_proof, closed_form_fk_matrix
from fk.gaussian import sample_gaussian


class CertificateTests(unittest.TestCase):
    def test_interval_matrix_agrees_with_integer_proof_and_independent_tensor(self):
        model = build_model()
        proof = integer_fk_matrix_proof()
        self.assertTrue(proof["verified"])
        np.testing.assert_allclose(proof["matrix"], closed_form_fk_matrix(), atol=1e-15)
        exact = ExactK2Moments(model)
        rng = np.random.default_rng(4501)
        for _ in range(10):
            z = rng.normal(size=3)+1j*rng.normal(size=3)
            value = interval_q2(z)
            self.assertAlmostEqual(float(value.mid), exact.evaluate(z).q, delta=2e-12)
            full = np.array([complex(float(r.mid), float(i.mid)) for r, i in exact_basis_action(z)])
            np.testing.assert_allclose(full, model.S@z, atol=1e-14)

    def test_feasible_witness_and_saved_target_bound(self):
        model = build_model()
        rng = np.random.default_rng(4502)
        atoms = sample_gaussian(8, 128, rng).kets
        for t in (0, 11, 20):
            target = model.target(t)
            z = model.S.conj().T@target
            result = certify_order2_normalization(z, target)
            w = model.S@result["z_feasible"]
            self.assertLessEqual(result["q2_normalized_interval"][1], 1.)
            self.assertLessEqual(max(abs(atoms.conj()@w)), 1.)
            self.assertLessEqual(result["L_certified_saved_target"], np.vdot(w, target).real+1e-14)
            self.assertLessEqual(result["L_certified_saved_target"], np.sum(abs(target)))
            self.assertFalse(result["high_order_moment_certified"])


if __name__ == "__main__":
    unittest.main()
