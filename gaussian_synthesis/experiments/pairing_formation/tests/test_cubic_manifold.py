import unittest
import numpy as np
from scipy.linalg import expm
from pairing_formation.cubic_manifold import CubicManifold


def radial_tensor(n):
    # C(c)=||c||^2 c has the complete projective Hilbert manifold.
    T=np.zeros((n,n,n,n),complex)
    for a in range(n):
        for j in range(n):T[a,a,j,j]+=.5;T[a,j,a,j]+=.5
    return T


class CubicManifoldTests(unittest.TestCase):
    def test_jacobian_nonholomorphic(self):
        rng=np.random.default_rng(1501)
        T=rng.normal(size=(7,3,3,3))+1j*rng.normal(size=(7,3,3,3));T=(T+T.swapaxes(1,2))/2
        model=CubicManifold(T,np.eye(7));c=rng.normal(size=3)+1j*rng.normal(size=3)
        v=rng.normal(size=3)+1j*rng.normal(size=3);eps=1e-5
        fd=(model.raw(c+eps*v)-model.raw(c-eps*v))/(2*eps)
        np.testing.assert_allclose(fd,model.raw_jacobian(c)@np.r_[v.real,v.imag],atol=2e-8)

    def test_full_manifold_matches_schrodinger_and_conserves_energy(self):
        rng=np.random.default_rng(1502);x=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));H=x+x.conj().T
        c=rng.normal(size=4)+1j*rng.normal(size=4);c/=np.linalg.norm(c)
        model=CubicManifold(radial_tensor(4),H)
        times=np.linspace(0,.6,7);states,_,report=model.evolve(times,c,rtol=2e-10,atol=2e-12)
        exact=np.array([expm(-1j*t*H)@c for t in times])
        overlaps=abs(np.einsum('ti,ti->t',exact.conj(),states))**2
        np.testing.assert_allclose(overlaps,1.,atol=2e-10)
        self.assertLess(report['max_energy_drift'],1e-10)
        self.assertEqual(report['physical_tangent_ranks'],[6])

    def test_coordinate_gauge_null_directions(self):
        model=CubicManifold(radial_tensor(3),np.diag([1.,2.,4.]))
        c=np.array([1.,.2j,.3]);_,J=model.tangent(c)
        np.testing.assert_allclose(J@np.r_[c.real,c.imag],0.,atol=2e-14)
        np.testing.assert_allclose(J@np.r_[-c.imag,c.real],0.,atol=2e-14)

    def test_non_kahler_action_preserves_energy(self):
        rng=np.random.default_rng(1503)
        T=rng.normal(size=(7,3,3,3))+1j*rng.normal(size=(7,3,3,3));T=(T+T.swapaxes(1,2))/2
        x=rng.normal(size=(7,7))+1j*rng.normal(size=(7,7));H=x+x.conj().T
        c=rng.normal(size=3)+1j*rng.normal(size=3);c/=np.linalg.norm(c)
        model=CubicManifold(T,H);psi,J=model.tangent(c);dc=model.rhs(0.,c)
        rate=J@np.r_[dc.real,dc.imag]
        np.testing.assert_allclose((J.conj().T@(rate+1j*H@psi)).imag,0.,atol=2e-12)
        self.assertLess(abs(2*np.vdot(H@psi,rate).real),2e-12)
        U,s,_=np.linalg.svd(np.vstack((J.real,J.imag)),full_matrices=False)
        X=U[:7,s>1e-10]+1j*U[7:,s>1e-10]
        omega=(X.conj().T@X).imag
        self.assertGreater(np.linalg.norm(omega@omega+np.eye(omega.shape[0])),.1)
