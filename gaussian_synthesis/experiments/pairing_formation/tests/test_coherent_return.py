import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.coherent_return import factor_observables, dephase_groups, flat_band_hamiltonian
from pairing_formation.model import PairModel
from pairing_formation.structure import collective_basis
from pairing_formation.projected_count_bound import quantum_fisher


class CoherentReturnTests(unittest.TestCase):
    def test_mixed_factor_echo_and_full_QFI(self):
        rng = np.random.default_rng(251025)
        m = np.array([0, 0, 1, 1, 1, 2, 3])
        V = rng.normal(size=(7, 3))+1j*rng.normal(size=(7, 3))
        V /= np.linalg.norm(V)
        rho = V@V.conj().T
        angle = .43
        U = np.diag(np.exp(-1j*angle*m))
        obs = factor_observables(m, V, angle)
        self.assertAlmostEqual(obs['QFI'], quantum_fisher(rho, np.diag(m)), places=11)
        expected = np.trace(rho@rho-rho@U@rho@U.conj().T).real
        self.assertAlmostEqual(obs['echo_difference'], expected, places=12)
        state = V[:, 0]/np.linalg.norm(V[:, 0])
        groups = dephase_groups(state, m)
        clean, mixed = factor_observables(m, state), factor_observables(m, groups)
        np.testing.assert_allclose(clean['count'], mixed['count'], atol=1e-13)
        self.assertLess(abs(mixed['QFI']), 1e-11)
        self.assertLess(abs(mixed['echo_difference']), 1e-12)
        self.assertAlmostEqual(clean['QFI'], 4*clean['variance'], places=11)

    def test_flat_chain_matches_full_pair_hamiltonian(self):
        for r in (2, 3, 4):
            model = PairModel(np.r_[-.5*np.ones(r), .5*np.ones(r)], 3.7)
            B = collective_basis(model, np.ones(r), np.ones(r))
            H = flat_band_hamiltonian(r, model.G).toarray()
            np.testing.assert_allclose(B.conj().T@(model.H@B), H, atol=2e-13)
            np.testing.assert_allclose(model.H@B, B@H, atol=2e-13)


if __name__ == '__main__':
    unittest.main()
