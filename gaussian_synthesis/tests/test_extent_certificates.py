import math
import unittest
import numpy as np
from gaussian_synthesis import FockBasis
from gaussian_synthesis.certificates import extent_certificate
from gaussian_synthesis.gaussian import quadric_matrix
from gaussian_synthesis.theory import gamma_factor
from _magic_reference import magic_state, magic_q, magic_optimizer


def exact_k2(u):
    """Independent four-mode symmetric-traceless projector fixture.

    Q_2 removes the one-dimensional trace from Sym^2(C^8), hence
    q_2(u)=||u||^4-|u^T J u|^2/8. Differentiating q_2/4 gives C_2.
    """
    j = quadric_matrix()
    a = float(np.vdot(u, u).real)
    s = u @ j @ u
    return a*a-abs(s)**2/8, a*u-s*(j @ u.conj())/8


class ExtentCertificateTests(unittest.TestCase):
    def setUp(self):
        self.basis = FockBasis()
        self.psi = magic_state(self.basis)

    def test_exact_magic_optimizer(self):
        for k in (1, 2, 4):
            with self.subTest(k=k):
                u = magic_optimizer(k, self.basis)
                q = float(np.linalg.norm(u))**(2*k)*magic_q(k)
                cert = extent_certificate(self.psi, u, k, q_lower=q,
                                          q_upper=q, residual_bound=0.)
                expected = magic_q(k)**(-1/(2*k))
                self.assertAlmostEqual(cert.t_lower, expected, places=12)
                self.assertAlmostEqual(cert.t_upper, expected, places=12)
                self.assertAlmostEqual(cert.lower_bound, expected**2, places=12)
                self.assertAlmostEqual(cert.upper_bound,
                                       gamma_factor(4, k)*expected**2, places=12)
                self.assertLessEqual(cert.lower_bound, 2.)
                self.assertGreaterEqual(cert.upper_bound, 2.)

    def test_perturbed_witness_exact_polynomial(self):
        u = magic_optimizer(2, self.basis)
        u[1] += .11+.07j
        u[6] -= .03j
        q, c = exact_k2(u)
        self.assertAlmostEqual(float(np.vdot(u, c).real), q)
        rho = float(np.linalg.norm(c-self.psi))
        cert = extent_certificate(self.psi, u, 2, q_lower=q,
                                  q_upper=q, residual_bound=rho)
        exact_t = magic_q(2)**(-.25)
        self.assertLessEqual(cert.t_lower, exact_t)
        self.assertGreaterEqual(cert.t_upper, exact_t)
        self.assertLessEqual(cert.lower_bound, 2.)
        self.assertGreaterEqual(cert.upper_bound, 2.)
        # A wider validated interval must remain conservative.
        wider = extent_certificate(self.psi, u, 2, q_lower=q*.9,
                                   q_upper=q*1.1, residual_bound=rho+.02)
        self.assertLessEqual(wider.t_lower, cert.t_lower)
        self.assertGreaterEqual(wider.t_upper, cert.t_upper)

    def test_target_scaling_and_joint_global_phase(self):
        u = magic_optimizer(2, self.basis)
        u[2] += .09j
        q, c = exact_k2(u)
        rho = float(np.linalg.norm(c-self.psi))
        base = extent_certificate(self.psi, u, 2, q_lower=q,
                                  q_upper=q, residual_bound=rho)
        for scale, phase in ((1., .73), (2.3, -.91), (.4, 1.21)):
            with self.subTest(scale=scale, phase=phase):
                z = np.exp(1j*phase)
                scaled_u = scale**(1/3)*z*u
                scaled_q = scale**(4/3)*q
                cert = extent_certificate(scale*z*self.psi, scaled_u, 2,
                    q_lower=scaled_q, q_upper=scaled_q, residual_bound=scale*rho)
                self.assertAlmostEqual(cert.t_lower, scale*base.t_lower)
                self.assertAlmostEqual(cert.t_upper, scale*base.t_upper)
                self.assertAlmostEqual(cert.lower_bound, scale**2*base.lower_bound)
                self.assertAlmostEqual(cert.upper_bound, scale**2*base.upper_bound)

    def test_zero_witness(self):
        zero = np.zeros(8, dtype=complex)
        cert = extent_certificate(self.psi, zero, 4, q_lower=0., q_upper=.1,
                                  residual_bound=float(np.linalg.norm(self.psi)))
        self.assertEqual(cert.lower_bound, 0.)
        self.assertAlmostEqual(cert.upper_bound, 8.)
        cert = extent_certificate(zero, zero, 4, q_lower=0., q_upper=0.,
                                  residual_bound=0.)
        self.assertEqual((cert.lower_bound, cert.upper_bound), (0., 0.))
        for q_lower, rho in ((.01, 1.), (0., .9)):
            with self.assertRaises(ValueError):
                extent_certificate(self.psi, zero, 4, q_lower=q_lower,
                                   q_upper=.1, residual_bound=rho)

    def test_rejects_invalid_or_inconsistent_inputs(self):
        u = magic_optimizer(2, self.basis)
        q, _ = exact_k2(u)
        cases = (
            {"q_lower": -1.}, {"q_upper": math.inf},
            {"q_lower": math.nan}, {"q_lower": q+1},
            {"residual_bound": -.1}, {"q_upper": 0., "q_lower": 0.},
            {"q_upper": q*1e-4, "q_lower": 0.},
            {"q_upper": 100., "q_lower": 99.},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                kwargs = dict(q_lower=q, q_upper=q, residual_bound=0.)
                kwargs.update(changes)
                extent_certificate(self.psi, u, 2, **kwargs)
        # These are valid moment endpoints but contradict the claimed residual.
        with self.assertRaisesRegex(ValueError, "inconsistent dual interval"):
            extent_certificate(2*self.psi, u, 2, q_lower=q, q_upper=q,
                               residual_bound=0.)
        for bad_k in (0, 1.5, True):
            with self.assertRaises(ValueError):
                extent_certificate(self.psi, u, bad_k, q_lower=q, q_upper=q,
                                   residual_bound=0.)
        with self.assertRaises(ValueError):
            extent_certificate(self.psi[:7], u, 2, q_lower=q, q_upper=q,
                               residual_bound=0.)
