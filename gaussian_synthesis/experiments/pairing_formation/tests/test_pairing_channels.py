import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.linear import matrices
from pairing_formation.pairing_channels import PairKernelModel, linear_matrices, angular_model, ChannelObservables
from pairing_formation.structure import collective_basis


class PairingChannelTests(unittest.TestCase):
    def test_uniform_kernel_recovers_original_H_and_response(self):
        for L in (4, 8):
            original = picket_fence(L, 3.1)
            general = PairKernelModel(original.energies, np.full((L, L), 3.1/L))
            np.testing.assert_allclose(general.H.toarray(), original.H.toarray(), atol=1e-13)
            for a, b in zip(linear_matrices(general), matrices(original)):
                np.testing.assert_allclose(a, b, atol=1e-13)

    def test_annihilators_match_number_conserving_operators(self):
        model, forms = angular_model(8)
        obs = ChannelObservables(model, forms)
        x, y = np.vstack((np.zeros_like(forms), forms)).T/np.sqrt(model.r)
        for w, B in zip((x, y), obs.first):
            np.testing.assert_allclose((B.conj().T@B).toarray(),
                                       model.pair_operator(np.outer(w.conj(), w)).toarray(), atol=1e-13)
        C = 1j*(obs.first[0].conj().T@obs.first[1]-obs.first[1].conj().T@obs.first[0])
        np.testing.assert_allclose(C.toarray(), obs.chirality.toarray(), atol=1e-13)

    def test_reflection_forbids_a_mean_chirality(self):
        model, forms = angular_model(12)
        state = model.evolve([.7])[0]
        value = ChannelObservables(model, forms).evaluate(state)
        self.assertLess(abs(value['chirality_mean']), 1e-11)
        self.assertLess(abs(value['relative_phase_four_pair_imag']), 1e-11)
        np.testing.assert_allclose(value['upper_channel_occupations'][0],
                                   value['upper_channel_occupations'][1], atol=1e-11)

    def test_negative_phase_moment_is_not_a_chirality_witness(self):
        model, forms = angular_model(12)
        # A single real, nematic orbital in each band, with exactly two
        # transfers. There are no two competing orbital configurations.
        state = collective_basis(model, forms[:, 0], forms[:, 0])[:, 2]
        obs = ChannelObservables(model, forms).evaluate(state)
        self.assertAlmostEqual(obs['normalized_phase_locking'], -1., places=12)
        self.assertLess(abs(obs['chirality_mean']), 1e-12)

    def test_first_dark_channel_is_the_Pauli_cubic_source(self):
        for L in (10, 12):
            model, forms = angular_model(L)
            U, _ = np.linalg.qr(forms)
            Q = np.eye(model.r)-U@U.T
            state = model.initial.copy()
            for order in (1, 2, 3):
                state = model.H@state
                dark = Q@state[model.blocks[1][1]]@Q
                if order < 3:
                    np.testing.assert_allclose(dark, 0., atol=2e-12)
                else:
                    expected = -4*Q@(model.kernel[model.r:, :model.r]**3)@Q
                    np.testing.assert_allclose(dark, expected, atol=2e-12)


if __name__ == '__main__':
    unittest.main()
