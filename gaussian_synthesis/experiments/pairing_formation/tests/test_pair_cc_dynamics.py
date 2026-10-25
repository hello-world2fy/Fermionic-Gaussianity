import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.pairing_channels import angular_model
from pairing_formation.pair_cc_dynamics import PairCCDynamics, product_parameter_map, circulant_parameter_map, projective_rhs
from pairing_formation.stochastic import numpy_rhs
from pairing_formation.kernel_stochastic import numpy_kernel_rhs, upper_pair_symbol


class PairCCDynamicsTests(unittest.TestCase):
    def test_permanent_jacobian_and_normalized_physical_rate(self):
        model, forms = angular_model(8)
        rng = np.random.default_rng(501)
        for restricted in (False, True):
            U, _ = np.linalg.qr(forms)
            d = PairCCDynamics(model, product_parameter_map(U, U) if restricted else None)
            y = d.initial()
            y[1:] = .1*(rng.normal(size=len(y)-1)+1j*rng.normal(size=len(y)-1))
            v = rng.normal(size=len(y)-1)+1j*rng.normal(size=len(y)-1)
            raw, D = d.raw_and_jacobian(y[1:])
            step = 1e-6
            finite = (d.raw_and_jacobian(y[1:]+step*v)[0]-d.raw_and_jacobian(y[1:]-step*v)[0])/(2*step)
            np.testing.assert_allclose(D@v, finite, atol=2e-9)
            rate, diagnostic, physical_rate = d.derivative(0, y, diagnostics=True)
            finite = (d.state(y+step*rate)-d.state(y-step*rate))/(2*step)
            np.testing.assert_allclose(physical_rate, finite, atol=2e-8)
            self.assertLess(abs(diagnostic['norm_rate']), 1e-11)
            self.assertLess(abs(diagnostic['energy_rate']), 1e-10)

    def test_free_evolution_is_exact_on_arbitrary_CC_state(self):
        model = picket_fence(6, 0.)
        d = PairCCDynamics(model)
        rng = np.random.default_rng(500)
        initial = d.initial()
        initial[1:] = .2*(rng.normal(size=9)+1j*rng.normal(size=9))
        times = np.array([0., .3, .7])
        predicted, _, report = d.evolve(times, initial, rtol=2e-10, atol=2e-12)
        exact = model.evolve(times, initial=d.state(initial))
        np.testing.assert_allclose(predicted, exact, atol=2e-9)
        self.assertLess(report['max_energy_drift'], 1e-9)

    def test_initial_rate_matches_Schrodinger_and_dark_ablation(self):
        model, forms = angular_model(10)
        U, _ = np.linalg.qr(forms)
        Q = np.eye(model.r)-U@U.T
        for M in (None, product_parameter_map(U, U)):
            dynamics = PairCCDynamics(model, M)
            _, _, rate = dynamics.derivative(0, dynamics.initial(), diagnostics=True)
            np.testing.assert_allclose(rate, -1j*model.H@model.initial, atol=2e-13)
        restricted = PairCCDynamics(model, product_parameter_map(U, U))
        predicted, _, report = restricted.evolve([0., .15])
        np.testing.assert_allclose(Q@predicted[-1][model.blocks[1][1]]@Q, 0., atol=1e-13)
        self.assertLess(report['max_energy_drift'], 1e-8)

    def test_projective_equation_matches_full_Fock_similarity_projection(self):
        from pairing_formation.pairing_channels import PairKernelModel
        rng = np.random.default_rng(504)
        L = 8; r = L//2
        J = rng.normal(size=(L,L))+1j*rng.normal(size=(L,L)); J = .1*(J+J.conj().T)
        model = PairKernelModel(np.linspace(-1,1,L),J)
        dynamics = PairCCDynamics(model)
        for scale in (.01,.2,1.):
            T = scale*(rng.normal(size=(r,r))+1j*rng.normal(size=(r,r)))
            raw, _ = dynamics.raw_and_jacobian(T.ravel())
            action = model.H@raw
            expected = -1j*(action[model.blocks[1][1]]-T*action[model.fermi_index])
            np.testing.assert_allclose(projective_rhs(model,T), expected, atol=2e-13)

    def test_translation_reduction_matches_full_DF_physical_tangent(self):
        model,_ = angular_model(10)
        M = circulant_parameter_map(model.r)
        full = PairCCDynamics(model); reduced = PairCCDynamics(model,M)
        rng = np.random.default_rng(505)
        x = .2*(rng.normal(size=model.r)+1j*rng.normal(size=model.r))
        a = np.r_[1.+0j,M@x]; b = np.r_[1.+0j,x]
        np.testing.assert_allclose(full.state(a),reduced.state(b),atol=2e-14)
        np.testing.assert_allclose(full.derivative(0,a,True)[2],reduced.derivative(0,b,True)[2],atol=2e-12)


class KernelStochasticTests(unittest.TestCase):
    def test_recovers_uniform_kernel_and_classical_conservation(self):
        rng = np.random.default_rng(502)
        sample = rng.normal(size=(17, 8, 3))
        model = picket_fence(8, 2.3)
        J = np.full((8, 8), model.g)
        np.testing.assert_allclose(numpy_kernel_rhs(sample, model.energies, J), numpy_rhs(sample, model.energies, model.g), atol=2e-14)
        model, _ = angular_model(8)
        rate = numpy_kernel_rhs(sample, model.energies, model.kernel)
        np.testing.assert_allclose((sample*rate).sum(axis=-1), 0., atol=2e-14)
        np.testing.assert_allclose(rate[..., 2].sum(axis=-1), 0., atol=2e-14)

    def test_pair_symbol_agrees_with_product_quantum_expectation(self):
        # In a product spin-coherent state, different sites factorize exactly;
        # the diagonal is <s+ s->=z+1/2, not x*x+y*y.
        rng = np.random.default_rng(503)
        sample = rng.normal(size=(11, 8, 3))
        sample /= 2*np.linalg.norm(sample, axis=-1, keepdims=True)
        W = rng.normal(size=(4, 4)); W = W+W.T
        density = np.einsum('ni,nj->nij', sample[:,4:,0]+1j*sample[:,4:,1], sample[:,4:,0]-1j*sample[:,4:,1])
        density[:, np.arange(4), np.arange(4)] = sample[:,4:,2]+.5
        expected = np.einsum('ij,nij->n', W, density).real
        np.testing.assert_allclose(upper_pair_symbol(sample, W, 4), expected, atol=2e-14)


if __name__ == '__main__':
    unittest.main()
