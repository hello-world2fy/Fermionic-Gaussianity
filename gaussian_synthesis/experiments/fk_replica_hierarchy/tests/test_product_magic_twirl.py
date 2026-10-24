import unittest
import numpy as np
from fk4 import paths
from fk4.product_magic_twirl import proof_and_actions,orbit,aligned_orbit_decomposition
from fk.gaussian import sample_gaussian
from fk3.discovery import covariance_from_kets


class ProductMagicTwirlTests(unittest.TestCase):
    def test_integer_identity_and_all_actual_gaussian_branches(self):
        proof,_,_,target=proof_and_actions()
        self.assertTrue(proof['exact_Gaussian_integer_identity_checked'])
        kets=sample_gaussian(8,5,np.random.default_rng(61120)).kets
        factors,report=aligned_orbit_decomposition(kets)
        branches=np.concatenate([orbit(g) for g in kets])
        coefficients=np.repeat(factors,256)
        np.testing.assert_allclose(coefficients@branches,target,atol=2e-14)
        self.assertAlmostEqual(np.sum(abs(coefficients)),report['coefficient_cost'],places=12)
        self.assertEqual(len(branches),report['actual_Gaussian_terms'])
        cov=covariance_from_kets(branches)
        self.assertLess(np.max(abs(cov@cov+np.eye(16))),1e-11)


if __name__=='__main__':unittest.main()
