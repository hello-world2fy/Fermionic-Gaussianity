import unittest
import numpy as np
from fk4.husimi import sample_husimi
from fk4.reference import log_dimension
from fk3.discovery import covariance_from_kets


class HusimiTests(unittest.TestCase):
    def test_exact_iid_husimi_against_analytic_magic_beta_one(self):
        u=np.zeros(8,complex);u[[0,7]]=1/np.sqrt(2)
        states,report=sample_husimi(u,4096,np.random.default_rng(5919))
        values=abs(states.conj()@u)**2
        self.assertLess(abs(values.mean()-.2),6*values.std()/np.sqrt(len(values)))
        self.assertTrue(report['iid_particles'])
        cov=covariance_from_kets(states[:64],n=4)
        self.assertLess(np.max(abs(cov@cov+np.eye(8))),1e-12)
        self.assertEqual(len(states),4096)


if __name__=='__main__':unittest.main()
