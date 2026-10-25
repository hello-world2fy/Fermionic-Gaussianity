import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.interband import EntanglingSources, sector_entanglement, filtered_density, tomography, schmidt_dephasing_columns, PAULI, ppt_witness


class InterbandTests(unittest.TestCase):
    def test_tomography_and_number_preserving_dephasing(self):
        model = picket_fence(8, 2.)
        state = model.evolve([2.])[0]
        columns = schmidt_dephasing_columns(model, state)
        np.testing.assert_allclose(np.sum(abs(columns)**2, axis=1)@model.occ, abs(state)**2@model.occ, atol=1e-14)
        for _, indices in model.blocks:
            B = state[indices]
            mixed = sum(columns[indices, a]@columns[indices, a].conj().T for a in range(columns.shape[1]))
            np.testing.assert_allclose(B@B.conj().T, mixed, atol=1e-14)
            mixed = sum(columns[indices, a].T@columns[indices, a].conj() for a in range(columns.shape[1]))
            np.testing.assert_allclose(B.T@B.conj(), mixed, atol=1e-14)
        rho, probability = filtered_density(model, columns, [0, 3], [0, 3])
        self.assertLess(tomography(rho, probability)['negativity'], 1e-14)
        bell = np.array([1, 0, 0, 1])/np.sqrt(2)
        rho = np.outer(bell, bell)
        report = tomography(rho)
        self.assertAlmostEqual(report['negativity'], .5)
        self.assertAlmostEqual(report['tomographic_CHSH_max'], 2*np.sqrt(2))
        reconstructed = sum(report['pauli_moments'][i][j]*np.kron(a, b)/4 for i, a in enumerate(PAULI) for j, b in enumerate(PAULI))
        np.testing.assert_allclose(reconstructed, rho, atol=1e-15)

    def test_rank_two_source_and_schmidt_leakage(self):
        model = picket_fence(8, 2.)
        analyzer = EntanglingSources(model)
        report, product = analyzer.analyze(model.evolve([1.2])[0])
        self.assertLess(max(r['local_normal_norm'] for r in report['records']), 1e-13)
        self.assertLess(max(r['rank_beyond_two_norm'] for r in report['records']), 1e-13)
        eps = 2e-4
        states = model.evolve([-eps, eps], product)
        leakage = np.mean([1-sector_entanglement(model, x)['rank1_fidelity_bound'] for x in states])/eps**2
        self.assertAlmostEqual(leakage, report['total_normal_squared_norm'], delta=2e-8)

    def test_linear_witness_and_local_measurements(self):
        rng = np.random.default_rng(813)
        bell = np.array([1., 0., 0., 1.])/np.sqrt(2)
        rho = np.outer(bell, bell)
        W, coefficients, predicted = ppt_witness(rho)
        self.assertAlmostEqual(predicted, -.5)
        reconstructed = sum(coefficients[i, j]*np.kron(a, b) for i, a in enumerate(PAULI) for j, b in enumerate(PAULI))
        np.testing.assert_allclose(W, reconstructed, atol=1e-15)
        for _ in range(20):
            a = rng.normal(size=2)+1j*rng.normal(size=2); a /= np.linalg.norm(a)
            b = rng.normal(size=2)+1j*rng.normal(size=2); b /= np.linalg.norm(b)
            product = np.kron(a, b)
            self.assertGreaterEqual(np.vdot(product, W@product).real, -1e-14)


if __name__ == '__main__':
    unittest.main()
