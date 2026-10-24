import unittest
import numpy as np
from fk4 import paths
from fk4.product_magic import support_at_eta,necessity_certificate
from fk4.columns import independent_arc_dictionary
from fk.model import build_model
from fk.gaussian import sample_gaussian
from fk3.discovery import covariance_from_kets
from fk.observables import atom_invariants


class ProductMagicTests(unittest.TestCase):
    def test_global_envelope_and_saturation_arc(self):
        model=build_model()
        random=sample_gaussian(8,512,np.random.default_rng(28101)).kets
        arc,_=independent_arc_dictionary(model,33)
        for states,sharp in [(random,False),(arc,True)]:
            eta=atom_invariants(covariance_from_kets(states),model.j_matrices)['eta']
            value=2*abs(states.conj()@model.S[:,0])
            upper=1-eta/2
            self.assertLess(float(np.max(value-upper)),2e-12)
            if sharp:
                np.testing.assert_allclose(value,upper,atol=2e-12)

    def test_near_optimal_physical_necessity(self):
        exact=necessity_certificate(2.,.2)
        self.assertLess(exact['coefficient_mass_fraction_upper'],1e-60)
        near=necessity_certificate(2.02,.2)
        self.assertTrue(near['nontrivial'])
        self.assertLess(near['coefficient_mass_fraction_upper'],.0991)


if __name__=='__main__':
    unittest.main()
