import sys
from pathlib import Path
import unittest
from math import factorial, prod
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.occupation_symbols import beta_coefficients, occupation_components
from pairing_formation.coherent_partition import coherent_parts
from pairing_formation.model import picket_fence
from pairing_formation.replica2 import Replica2


class OccupationSymbolsTests(unittest.TestCase):
    def test_independent_complex_sphere_integrals(self):
        # For n=2,3 every even pure state is Gaussian. Integrate monomials
        # over CP^(D-1) by phase balance and exact Dirichlet moments.
        rng=np.random.default_rng(1630)
        for n in (2,3):
            D=2**(n-1); occupation=np.r_[np.zeros(D//2),np.ones(D//2)]
            u=rng.normal(size=D)+1j*rng.normal(size=D);u/=np.linalg.norm(u)
            C=u.copy();B=u*np.vdot(occupation*u,u)
            components=occupation_components(C,occupation,B)
            for degree in (0,1,2,3,6):
                direct=np.zeros(D,complex)
                for i in range(D):
                    for j in range(D):
                        for k in range(D):
                            for l in range(D):
                                if sorted((i,l)) != sorted((j,k)): continue
                                alpha=np.bincount([i,l],minlength=D)
                                moment=prod(factorial(int(a)) for a in alpha)/(D*(D+1))
                                for d in range(degree):
                                    moment *= (D//2+alpha@occupation+d)/(D+2+d)
                                direct[i]+=D*(D+1)/2*moment*u[j]*u[k]*u[l].conjugate()
                np.testing.assert_allclose(beta_coefficients(n,degree)@components,direct,atol=3e-15)

    def test_quadratic_ward_identity_and_gaussian_equivariance(self):
        rng=np.random.default_rng(1631)
        for L in (4,6,8):
            model=picket_fence(L,2);replica=Replica2(model)
            u=rng.normal(size=model.dim)+1j*rng.normal(size=model.dim);u/=np.linalg.norm(u)
            diagonal=model.occ@rng.normal(size=L)
            C=replica.evaluate(u)[1];parts=coherent_parts(replica,u,diagonal*u)
            np.testing.assert_allclose(2*parts['complex_ratio']-parts['conjugate_ratio'],diagonal*C,atol=2e-14)
            phase=np.exp(-.73j*diagonal)
            np.testing.assert_allclose(replica.evaluate(phase*u)[1],phase*C,atol=2e-14)

    def test_rank_one_gaussian_vacuum_beta_weight(self):
        model=picket_fence(4,2);replica=Replica2(model);u=model.initial
        C=replica.evaluate(u)[1]
        for site in (0,3):
            P=model.occ[:,site];parts=coherent_parts(replica,u,P*u)
            components=occupation_components(C,P,parts['conjugate_ratio'])
            j=int(2*P[model.fermi_index])
            np.testing.assert_allclose(components[j],u,atol=1e-14)
            for other in set(range(3))-{j}:np.testing.assert_allclose(components[other],0,atol=1e-14)

    def test_nonstationary_density_response_against_finite_kicks(self):
        from pairing_formation.pairing_channels import angular_model
        model,_=angular_model(8);psi=model.evolve([.7])[0]
        weights=np.cos(2*np.pi*np.arange(model.r)/model.r)
        O=model.occ@np.r_[-weights,weights]
        background=model.evolve([.31],initial=psi)[0]
        response=model.evolve([.31],initial=O*psi)[0]
        expected=2*np.vdot(background,O*response).imag
        self.assertGreater(abs(expected),1e-3)
        errors=[]
        for step in (1e-2,1e-3):
            plus=model.evolve([.31],initial=np.exp(-1j*step*O)*psi)[0]
            minus=model.evolve([.31],initial=np.exp(1j*step*O)*psi)[0]
            derivative=(np.vdot(plus,O*plus).real-np.vdot(minus,O*minus).real)/(2*step)
            errors.append(abs(derivative-expected))
        self.assertLess(errors[1],errors[0]/90)
        self.assertLess(errors[1],2e-6)


if __name__=='__main__':unittest.main()
