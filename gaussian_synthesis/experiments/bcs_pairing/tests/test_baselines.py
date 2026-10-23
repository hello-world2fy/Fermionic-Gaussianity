import unittest
from itertools import combinations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from gaussian_synthesis import fidelity, phase_aligned_distance
from bcs_pairing.sectors import PairBasis
from bcs_pairing.model import BCSModel, uniform_model
from bcs_pairing.asymptotic import asymptotic_state
from bcs_pairing.exact import evolve_constant, evolve_inverse_time
from bcs_pairing.mean_field import (initial_spinors, project_spinors, evolve_mean_field,
                                    spinor_hamiltonians, mean_field_energy)
from bcs_pairing.pbcs import fit_pbcs, objective_gradient, parameters_to_spinors
from bcs_pairing.minors import minor_certificate, minor_rows
from bcs_pairing.observables import observables


def fermion_action(mask, operations):
    sign = 1
    for mode, create in operations:
        occupied = bool(mask & (1 << mode))
        if occupied == create:
            return None, 0
        sign *= (-1)**((mask & ((1 << mode)-1)).bit_count())
        mask ^= 1 << mode
    return mask, sign


def independent_fock_hamiltonian(masks, eps, g):
    matrix = np.zeros((len(masks), len(masks)))
    indices = {mask: i for i, mask in enumerate(masks)}
    for col, mask in enumerate(masks):
        matrix[col, col] = sum(eps[j]*(((mask >> (2*j)) & 1)+((mask >> (2*j+1)) & 1)-1)
                               for j in range(len(eps)))
        for i in range(len(eps)):
            for j in range(len(eps)):
                final, sign = fermion_action(mask, [(2*j, False), (2*j+1, False),
                                                    (2*i+1, True), (2*i, True)])
                if sign and final in indices:
                    matrix[indices[final], col] -= g*sign
    return matrix


class ModelTests(unittest.TestCase):
    def test_independent_fermion_hamiltonian_and_dicke(self):
        for L, m in ((2, 1), (4, 2), (4, 0), (4, 4)):
            model = uniform_model(L, m)
            actual = independent_fock_hamiltonian(model.basis.sector.occupations, model.energies, .6)
            np.testing.assert_allclose(model.hamiltonian(.6), actual, atol=1e-12)
            np.testing.assert_allclose(model.K @ model.dicke(), m*(L-m+1)*model.dicke(), atol=1e-12)
            np.testing.assert_allclose(project_spinors(model.basis, initial_spinors(model.basis)), model.dicke())

    def test_pair_mapping_and_capacity(self):
        basis = PairBasis(4, 2)
        self.assertEqual(basis.pair_masks, (3, 5, 6, 9, 10, 12))
        self.assertEqual(basis.sector.occupations, (15, 51, 60, 195, 204, 240))
        with self.assertRaises(ValueError):
            PairBasis(24, 12, max_dimension=10)
        with self.assertRaises(ValueError):
            BCSModel(basis, [0, 1, 2, np.nan])


class EvolutionTests(unittest.TestCase):
    def setUp(self):
        self.model = uniform_model()

    def test_spectral_reference_against_matrix_exponential(self):
        times = [0., .37, 2.]
        states = evolve_constant(self.model, self.model.dicke(), .6, times)
        for t, state in zip(times, states):
            np.testing.assert_allclose(state, expm(-1j*t*self.model.hamiltonian(.6)) @ self.model.dicke(), atol=1e-12)

    def test_free_evolution_and_both_mean_fields(self):
        times = np.linspace(0, 5, 11)
        exact = evolve_constant(self.model, self.model.dicke(), 0., times)
        for convention in ('td_bcs', 'product_tdvp'):
            spinors = evolve_mean_field(self.model, 0., times, convention=convention)
            for psi, spinor in zip(exact, spinors):
                self.assertLess(1-fidelity(psi, project_spinors(self.model.basis, spinor)), 1e-10)

    def test_mean_field_invariants_and_convergence(self):
        times = np.linspace(0, 20, 41)
        for convention in ('td_bcs', 'product_tdvp'):
            states = evolve_mean_field(self.model, .6, times, convention=convention)
            tight = evolve_mean_field(self.model, .6, times, convention=convention, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(np.sum(abs(states)**2, axis=2), 1, atol=2e-9)
            np.testing.assert_allclose(np.sum(abs(states[:, :, 1])**2, axis=1), 2, atol=2e-9)
            energies = [mean_field_energy(self.model.energies, s, .6, convention) for s in states]
            self.assertLess(np.ptp(energies), 2e-9)
            for a, b in zip(states, tight):
                self.assertLess(phase_aligned_distance(project_spinors(self.model.basis, a),
                                                      project_spinors(self.model.basis, b)), 1e-7)

    def test_product_tdvp_from_full_quantum_variation(self):
        rng = np.random.default_rng(10)
        spinors = rng.normal(size=(4, 2))+1j*rng.normal(size=(4, 2))
        spinors /= np.linalg.norm(spinors, axis=1)[:, None]
        masks = tuple(sum(3 << (2*j) for j in range(4) if p & (1 << j)) for p in range(16))
        matrix = independent_fock_hamiltonian(masks, self.model.energies, .6)
        occupation = np.array([[(p >> j) & 1 for j in range(4)] for p in range(16)])
        def energy(s):
            psi = np.prod(np.where(occupation, s[:, 1], s[:, 0]), axis=1)
            return float(np.vdot(psi, matrix @ psi).real)
        self.assertAlmostEqual(energy(spinors), mean_field_energy(self.model.energies, spinors, .6, 'product_tdvp'), places=12)
        h = spinor_hamiltonians(self.model.energies, spinors, .6, 'product_tdvp')
        for j in range(4):
            direction = rng.normal(size=2)+1j*rng.normal(size=2)
            direction -= spinors[j]*np.vdot(spinors[j], direction)
            shifted = []
            for sign in (1, -1):
                s = spinors.copy()
                s[j] += sign*1e-6*direction
                s[j] /= np.linalg.norm(s[j])
                shifted.append(energy(s))
            actual = (shifted[0]-shifted[1])/2e-6
            expected = 2*np.vdot(direction, h[j] @ spinors[j]).real
            self.assertAlmostEqual(actual, expected, places=7)

    def test_inverse_time_against_linear_time_ode(self):
        t0, eta = 1e-3, .8
        times = np.array([.1, .3, 1.])
        log_states = evolve_inverse_time(self.model, eta, t0, times, rtol=1e-11, atol=1e-13)
        direct = solve_ivp(lambda t, psi: -1j*(self.model.hamiltonian(eta/4/t) @ psi),
                           (t0, times[-1]), self.model.dicke(), t_eval=times,
                           method='DOP853', rtol=1e-11, atol=1e-13).y.T
        for a, b in zip(log_states, direct):
            self.assertLess(phase_aligned_distance(a, b), 1e-9)


class AsymptoticTests(unittest.TestCase):
    def setUp(self):
        self.model = uniform_model()

    def test_formula_against_explicit_scalar_sums(self):
        eps = self.model.energies
        eta, time = .8, 2.3
        coefficients = []
        for mask in self.model.basis.pair_masks:
            S = [j for j in range(4) if mask & (1 << j)]
            log_z = sum(-np.pi*(j+1)*eta/4-2j*time*eps[j]
                        -1j*eta/4*sum(np.log(abs(eps[j]-eps[a])) for a in range(4) if a != j) for j in S)
            jastrow = 2j*eta/4*sum(np.log(abs(eps[i]-eps[j])) for i, j in combinations(S, 2))
            coefficients.append(np.exp(log_z+jastrow))
        expected = np.array(coefficients)/np.linalg.norm(coefficients)
        np.testing.assert_allclose(asymptotic_state(self.model, eta, time=time, frame='physical'), expected, atol=1e-12)

    def test_same_probabilities_energy_units_and_cross_ratio(self):
        basis, eps = self.model.basis, self.model.energies
        original = asymptotic_state(self.model, .8)
        control = asymptotic_state(self.model, .8, alpha=0)
        np.testing.assert_allclose(abs(original)**2, abs(control)**2, atol=1e-14)
        self.assertLess(phase_aligned_distance(original, asymptotic_state(self.model, .8, energy_reference=17)), 1e-12)
        a, b, c, d = [basis.pair_masks.index(mask) for mask in (3, 12, 5, 10)]
        expected = np.exp(2j*.8/4*np.log(abs((eps[0]-eps[1])*(eps[2]-eps[3])/
                                                ((eps[0]-eps[2])*(eps[1]-eps[3])))))
        self.assertLess(abs(original[a]*original[b]/(original[c]*original[d])-expected), 1e-12)

    def test_degeneracy_rejected(self):
        with self.assertRaises(ValueError):
            asymptotic_state(BCSModel(self.model.basis, [0, 0, 1, 2]), .8)


class FitMinorObservableTests(unittest.TestCase):
    def setUp(self):
        self.model = uniform_model()
        self.basis = self.model.basis

    def test_pbcs_gradient_real_and_phase_directions(self):
        rng = np.random.default_rng(11)
        x = np.r_[rng.uniform(.2, 1.3, 4), rng.normal(size=3)]
        target = asymptotic_state(self.model, .8)
        _, gradient = objective_gradient(x, self.basis, target)
        for j in range(len(x)):
            dx = np.zeros_like(x)
            dx[j] = 1e-6
            numerical = (objective_gradient(x+dx, self.basis, target)[0]-objective_gradient(x-dx, self.basis, target)[0])/2e-6
            self.assertAlmostEqual(gradient[j], numerical, places=7)

    def test_random_pbcs_minor_and_fit(self):
        x = np.array([.3, .6, .9, 1.2, .7, -.4, 1.])
        spinors = parameters_to_spinors(x, 4)
        target = project_spinors(self.basis, spinors)
        certificate = minor_certificate(self.basis, target)
        self.assertLess(certificate['max_minor'], 1e-14)
        result = fit_pbcs(self.basis, target, seed=43, starts=4)
        self.assertGreater(result.best_found_fidelity, 1-1e-9)

    def test_general_target_fidelity_bracket_and_constraint_rank(self):
        target = asymptotic_state(self.model, 2.)
        fit = fit_pbcs(self.basis, target, starts=4)
        certificate = minor_certificate(self.basis, target)
        self.assertEqual(certificate['rank'], 2)
        self.assertGreater(certificate['q'], 0)
        self.assertLessEqual(fit.best_found_fidelity, certificate['upper_bound']+1e-10)
        rows = minor_rows(self.basis)
        self.assertGreater(abs(np.dot(rows[0], rows[1])), 0)
        self.assertEqual(minor_certificate(PairBasis(4, 1), np.ones(4))['upper_bound'], 1.)

    def test_boundary_fock_states_and_invalid_projection(self):
        for occupied in self.basis.occupations:
            spinors = np.column_stack((1-occupied, occupied))
            state = project_spinors(self.basis, spinors)
            self.assertEqual(np.count_nonzero(state), 1)
            fit = fit_pbcs(self.basis, state, starts=1)
            self.assertAlmostEqual(fit.best_found_fidelity, 1.)
        with self.assertRaises(ValueError):
            project_spinors(self.basis, np.tile([1, 0], (4, 1)))

    def test_observables_sum_rules_and_dicke_entropy(self):
        data = observables(self.basis, self.model.dicke())
        np.testing.assert_allclose(data['number'], .5)
        self.assertAlmostEqual(np.sum(data['number_correlation']), 4.)
        self.assertAlmostEqual(np.sum(data['pair_correlation']).real, 6.)
        np.testing.assert_allclose(data['pair_correlation'], data['pair_correlation'].conj().T)
        expected = -np.dot([1/6, 4/6, 1/6], np.log([1/6, 4/6, 1/6]))
        self.assertAlmostEqual(data['entropy'], expected, places=12)
        for mask in self.basis.sector.occupations:
            self.assertAlmostEqual(observables(self.basis, self.basis.sector.ket(mask))['entropy'], 0.)
