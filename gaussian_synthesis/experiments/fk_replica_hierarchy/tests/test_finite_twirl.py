import unittest
import numpy as np
from fk4 import paths
from fk4.finite_twirl import FiniteSpin7Twirl
from fk.model import build_model
from fk.gaussian import sample_gaussian
from fk3.discovery import covariance_from_kets


class FiniteTwirlTests(unittest.TestCase):
    def test_full_operator_and_generic_gaussian_orbits(self):
        m = build_model()
        twirl = FiniteSpin7Twirl(m)
        self.assertLess(twirl.operator_error, 1e-11)
        random = sample_gaussian(8, 3, np.random.default_rng(9831)).kets
        for g in random:
            atoms, info = twirl.orbit(g)
            np.testing.assert_allclose(atoms.mean(axis=0), m.S@(m.S.conj().T@g), atol=1e-12)
            self.assertEqual(info["count"], 448)
            cov = covariance_from_kets(atoms[::32])
            self.assertLess(np.max(abs(cov@cov+np.eye(16))), 1e-10)
            np.testing.assert_allclose(np.linalg.norm(atoms, axis=1), 1., atol=1e-12)


if __name__ == "__main__":
    unittest.main()
