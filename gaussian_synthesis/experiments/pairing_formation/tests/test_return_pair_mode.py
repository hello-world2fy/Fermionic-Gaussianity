import unittest
import numpy as np
from pairing_formation.return_pair_mode import return_pair_mode, fifth_order_coefficient
from pairing_formation.pairing_channels import angular_model, annihilator
from pairing_formation.dark_pairs import PairDarkSectors


class ReturnPairModeTests(unittest.TestCase):
    def test_against_original_full_hamiltonian(self):
        for L in (10,14):
            model,forms=angular_model(L,G=3.2)
            dark=PairDarkSectors(model,forms)
            coefficient=model.initial.copy()
            for n in range(1,6):coefficient=(-1j/n)*(model.H@coefficient)
            actual=dark.apply(coefficient,'either')[model.blocks[2][1]]
            np.testing.assert_allclose(actual,fifth_order_coefficient(model.r,3.2),atol=2e-14)

    def test_single_band_darkness_and_equal_schmidt_weights(self):
        for r in (5,6,7,8,9,13):
            b,d,mode=return_pair_mode(r);theta=2*np.pi*np.arange(r)/r
            for f in (np.cos(theta),np.sin(theta)):
                np.testing.assert_allclose(annihilator(r,2,f)@d,0.,atol=2e-13)
            self.assertLess(abs(np.vdot(b,d)),2e-13)
            s=np.linalg.svd(mode,compute_uv=False)
            np.testing.assert_allclose(s[:2],np.ones(2)/np.sqrt(2),atol=2e-13)
            self.assertLess(np.linalg.norm(s[2:]),2e-13)

    def test_finite_grid_six_site_norm_exception(self):
        r=6;G=3.2
        norm=np.linalg.norm(fifth_order_coefficient(r,G))**2
        generic=G**10*(r-3)**2*(r-4)/(1800*r**7)
        self.assertAlmostEqual(norm/generic,3.,places=12)
