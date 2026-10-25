import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.band_transfer import BandTransfer
from pairing_formation.linear import coupled_cluster_state


class BandTransferTests(unittest.TestCase):
    def test_independent_full_hamiltonian_action(self):
        rng=np.random.default_rng(945)
        for L in (6,10):
            model,_=angular_model(L,G=3.7)
            bands=BandTransfer(model.r,G=3.7)
            state=rng.normal(size=model.dim)+1j*rng.normal(size=model.dim)
            blocks=[state[indices] for _,indices in model.blocks]
            computed=bands.apply(blocks)
            exact=model.H@state-model.H[model.fermi_index,model.fermi_index]*state
            for m,(_,indices) in enumerate(model.blocks):
                np.testing.assert_allclose(computed[m],exact[indices],atol=3e-13)

    def test_permanent_series(self):
        model,_=angular_model(8)
        bands=BandTransfer(4)
        rng=np.random.default_rng(946)
        T=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
        coefficients=[bands.zero() for _ in range(5)]
        coefficients[0][0][0,0]=1.
        coefficients[1][1]=T
        series=bands.single_cluster_series(coefficients,4)
        reference=coupled_cluster_state(model,T)
        reference/=reference[model.fermi_index]
        for m,(_,indices) in enumerate(model.blocks):
            np.testing.assert_allclose(series[m][m],reference[indices],atol=1e-11)
            for n in range(5):
                if n!=m:self.assertLess(np.linalg.norm(series[n][m]),1e-12)


if __name__=='__main__':unittest.main()
