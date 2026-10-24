from pathlib import Path
import sys
import unittest
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk6.rare import number_volume_constant,polynomial_integral,sample_pair_probabilities,sample_number_ball
from fk6 import paths
from fk3.discovery import covariance_from_kets


class RareTests(unittest.TestCase):
    def test_exact_volume_and_independent_polynomial_moments(self):
        self.assertEqual(number_volume_constant(),Fraction(1,23178480))
        p=sample_pair_probabilities(100000,1.,np.random.default_rng(5513))
        base=polynomial_integral(True)
        for powers in [(2,0,0,0),(1,1,0,0)]:
            exact=float(polynomial_integral(True,extra=powers)/base)
            if powers[0]==2:observed=(p*p).mean(axis=1)
            else:observed=(p.sum(axis=1)**2-(p*p).sum(axis=1))/12
            self.assertLess(abs(observed.mean()-exact),6*observed.std(ddof=1)/np.sqrt(len(p)))
        self.assertLess(abs(p.sum(axis=1).mean()-28/29),.0003)

    def test_actual_gaussian_purity_and_number(self):
        g,p=sample_number_ball(128,.8,np.random.default_rng(5514))
        cov=covariance_from_kets(g)
        self.assertLess(np.max(abs(cov@cov+np.eye(16))),1e-11)
        masks=[m for m in range(256) if m.bit_count()%2==0]
        number=abs(g)**2@np.array([m.bit_count() for m in masks])
        np.testing.assert_allclose(number,2*p.sum(axis=1),atol=3e-14)


if __name__=='__main__':unittest.main()
