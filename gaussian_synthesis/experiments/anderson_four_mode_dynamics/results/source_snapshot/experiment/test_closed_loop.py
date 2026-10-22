import bootstrap
import unittest
import numpy as np
from gaussian_synthesis.gaussian import GaussianState
from gaussian_synthesis.measure import GaussianCoefficientMeasure
from gaussian_synthesis.sampling import HaarFourModeSampler
from gaussian_synthesis.moments import batch_moments
from exact_solver import exact_k2
from model import BASIS, J, Anderson, gaussian_defect, schmidt_decompose
from bridge import solve_measure, sampled_discretization, two_node_discretization, refit, symmetry_dictionary


class ClosedLoopTests(unittest.TestCase):
    def test_branch_matches_split_and_gaussianity(self):
        model = Anderson()
        state = model.initial_sum()
        vector = model.initial.copy()
        for _ in range(8):
            state = model.branch(state)
            vector = model.strang @ vector
        np.testing.assert_allclose(state.vector(), vector, atol=2e-13)
        self.assertLess(gaussian_defect(state.atoms), 2e-13)

    def test_measure_formula_uses_original_density(self):
        model = Anderson()
        target = model.strang @ model.initial
        measure, info = solve_measure(target, 'newton', 71)
        sampler = HaarFourModeSampler(np.random.default_rng(9))
        for g in sampler.sample(8):
            obj = GaussianState(BASIS, g)
            a = np.vdot(g, measure.u)
            self.assertAlmostEqual(measure.density(obj), 35*abs(a)**3, places=12)
            np.testing.assert_allclose(measure.aligned_state(obj).amplitudes, a/abs(a)*g, atol=1e-14)

    def test_two_node_preserves_moment_for_general_witness(self):
        rng = np.random.default_rng(41)
        for _ in range(20):
            u = rng.normal(size=8)+1j*rng.normal(size=8)
            u /= np.linalg.norm(u)
            measure = GaussianCoefficientMeasure(BASIS, 2, u)
            state, _ = two_node_discretization(measure)
            np.testing.assert_allclose(state.vector(), exact_k2(u)[1], atol=2e-14)
            self.assertLess(gaussian_defect(state.atoms), 2e-14)

    def test_two_node_gaussian_and_real_witness_degeneracy(self):
        vectors = [BASIS.ket(0), (BASIS.ket(0)+BASIS.ket(15))/np.sqrt(2)]
        for u in vectors:
            state, _ = two_node_discretization(GaussianCoefficientMeasure(BASIS, 2, u))
            np.testing.assert_allclose(state.vector(), exact_k2(u)[1], atol=1e-14)
            self.assertLess(gaussian_defect(state.atoms), 1e-14)

    def test_sampling_then_refit_does_not_need_exact_trajectory(self):
        model = Anderson()
        target = np.linalg.matrix_power(model.strang, 4) @ model.initial
        measure, _ = solve_measure(target, 'newton', 71)
        state, info = sampled_discretization(measure, 64, 80)
        compressed, _ = refit(state, target)
        self.assertGreater(info['proposal_samples'], 64)
        np.testing.assert_allclose(compressed.vector(), target, atol=1e-12)
        self.assertLess(gaussian_defect(compressed.atoms), 1e-12)

    def test_symmetry_projection_preserves_projected_sum(self):
        measure = GaussianCoefficientMeasure(BASIS, 2, BASIS.ket(0)+BASIS.ket(15))
        state, _ = sampled_discretization(measure, 32, 10)
        projected = symmetry_dictionary(state)
        expected = state.vector()
        expected[[0, 1, 6, 7]] = 0
        np.testing.assert_allclose(projected.vector(), expected, atol=1e-13)
        self.assertLess(gaussian_defect(projected.atoms), 1e-13)

    def test_schmidt_is_only_a_baseline(self):
        model = Anderson()
        target = np.linalg.matrix_power(model.strang, 11) @ model.initial
        state = schmidt_decompose(target)
        np.testing.assert_allclose(state.vector(), target, atol=1e-14)
        self.assertLess(gaussian_defect(state.atoms), 1e-14)

    def test_original_sgd_polish_reduces_actual_solver_error(self):
        model = Anderson(impurity_energy=.4)
        target = np.linalg.matrix_power(model.strang, 7) @ model.initial
        measure, info = solve_measure(target, 'sgd_polished', 42)
        self.assertIn('projected_sgd', info['source_function'])
        self.assertGreater(info['residual_before_polish'], 1e-6)
        self.assertLess(info['continuous_residual'], 1e-10)
        state, _ = two_node_discretization(measure)
        np.testing.assert_allclose(state.vector(), target, atol=1e-10)

    def test_original_api_is_called_inside_the_evolution_loop(self):
        from experiment import Config, run
        summary, rows, events, _, _ = run(Config('measure', backend='api'), final_time=.4)
        self.assertEqual(summary['original_solver_calls'], 2)
        self.assertEqual([e['step'] for e in events], [4, 8])
        self.assertTrue(all(e['source_function']=='gaussian_synthesis.api.synthesize' for e in events))
        self.assertGreater(summary['training_samples'], 0)
        self.assertGreater(summary['validation_samples'], 0)
        self.assertLess(summary['max_compression_distance'], 1e-10)

    def test_two_node_closed_loop_and_noninteracting_control(self):
        from experiment import Config, run
        config = Config('measure', backend='newton', discretization='two_node')
        summary, rows, _, state, _ = run(config, impurity_energy=.4)
        self.assertEqual(state.count, 2)
        self.assertLess(summary['max_compression_distance'], 2e-9)
        self.assertTrue(all(r['compression_distance'] <= r['cumulative_residual']+1e-11 for r in rows))
        control, _, _, state, _ = run(config, interaction=0.)
        self.assertEqual(state.count, 1)
        self.assertLess(control['max_infidelity'], 1e-12)


if __name__ == '__main__':
    unittest.main()
