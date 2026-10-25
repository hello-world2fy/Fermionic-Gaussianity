import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.projected_count_bound import CanonicalAGPCheck, quantum_fisher, coherence_lower_bound, moment_fidelity_bound


class ProjectedCountBoundTests(unittest.TestCase):
    def test_arbitrary_orbitals_match_full_pfaffian_and_bound_steps(self):
        rng = np.random.default_rng(517)
        model = CanonicalAGPCheck(4, 2)
        for _ in range(12):
            logs = rng.normal(size=4)+1j*rng.normal(size=4)
            U = np.linalg.qr(rng.normal(size=(8, 8))+1j*rng.normal(size=(8, 8)))[0]
            A = np.diag([0.]*4+[1.]*4)
            P = U.conj().T@A@U
            result = model.check(logs, P)
            state = model.rotated_full_state(logs, U)
            count = np.sum(model.configurations >= 4, axis=1)
            probability = abs(state)**2
            mean = float(probability@count)
            variance = float(probability@(count**2)-mean**2)
            self.assertAlmostEqual(mean, result['mean_number'], delta=1e-13)
            self.assertAlmostEqual(variance, result['variance_number'], delta=1e-13)
            self.assertLess(result['diagonal_variance'], result['diagonal_bound']+1e-12)
            self.assertLess(result['broken_pair_variance'], result['broken_pair_bound']+1e-12)
            self.assertLess(variance, result['simple_bound']+1e-12)
            self.assertLess(result['variance_decomposition_error'], 1e-12)

    def test_mixed_fisher_and_coherence_bound(self):
        rng = np.random.default_rng(612)
        model = CanonicalAGPCheck(4, 2)
        count = np.sum(model.configurations >= 4, axis=1)/2
        generator = np.diag(count)
        for _ in range(4):
            states = []
            for j in range(3):
                logs = rng.normal(size=4)+1j*rng.normal(size=4)
                U = np.linalg.qr(rng.normal(size=(8, 8))+1j*rng.normal(size=(8, 8)))[0]
                states.append(model.rotated_full_state(logs, U))
            rho = sum(np.outer(s, s.conj()) for s in states)/len(states)
            fisher = quantum_fisher(rho, generator)
            lower = coherence_lower_bound(rho, generator)
            self.assertLess(lower, fisher+1e-12)
            self.assertLess(fisher, 4*np.trace(rho@generator).real+1e-12)
            for angle in (.1, .3, 1.):
                phase = np.exp(-1j*angle*count)
                rotated = phase[:, None]*rho*phase.conj()[None, :]
                echo_difference = np.trace(rho@rho-rho@rotated).real
                self.assertLess(4*echo_difference/angle**2, lower+1e-11)
                self.assertLess(echo_difference, angle**2*np.trace(rho@generator).real+1e-11)
        # A four-fermion cat with p(m=2)=1/4 violates the proposed free-state bound.
        pure = np.array([np.sqrt(.75), .5])
        rho = np.outer(pure, pure)
        generator = np.diag([0., 2.])
        self.assertAlmostEqual(quantum_fisher(rho, generator), 3.)
        self.assertAlmostEqual(coherence_lower_bound(rho, generator), 3.)
        self.assertGreater(quantum_fisher(rho, generator), 4*np.trace(rho@generator).real)
        certificate = moment_fidelity_bound(.5, .75, 2)
        self.assertLess(certificate['fidelity_upper_bound'], 1.)

    def test_finite_angle_detects_coherence_not_classical_count_noise(self):
        pure = np.array([np.sqrt(.75), .5])
        coherent = np.outer(pure, pure)
        classical = np.diag(pure**2)
        m = np.array([0., 2.])
        angle = .2
        phase = np.exp(-1j*angle*m)
        excesses = []
        for rho in (coherent, classical):
            rotated = phase[:, None]*rho*phase.conj()[None, :]
            excesses.append(np.trace(rho@rho-rho@rotated).real-angle**2*np.diag(rho)@m)
        self.assertGreater(excesses[0], 0.)
        self.assertLess(excesses[1], 0.)


if __name__ == '__main__':
    unittest.main()
