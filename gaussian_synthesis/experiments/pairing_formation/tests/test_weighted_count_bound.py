import sys
from pathlib import Path
import unittest
from itertools import combinations
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.projected_count_bound import CanonicalAGPCheck
from pairing_formation.orbital_fluctuations import reduced_moments
from pairing_formation.weighted_count_bound import covariance_excess,weighted_statistics,full_number_spectrum,weighted_fidelity_bound


class WeightedBoundTests(unittest.TestCase):
    def test_canonical_covariance_concavity_with_signed_weights(self):
        rng=np.random.default_rng(820)
        for L,M in ((4,1),(4,2),(4,3),(6,3),(6,5)):
            check=CanonicalAGPCheck(L,M)
            for _ in range(10):
                c=check.coefficients(3*rng.normal(size=L)+1j*rng.normal(size=L));p=abs(c)**2
                n=p@check.occupations;C=check.occupations.T@(p[:,None]*check.occupations)-np.outer(n,n)
                self.assertLess(np.linalg.eigvalsh(covariance_excess(n,C,M))[-1],2e-13)

    def test_rotated_full_Pfaffian_states_in_physical_count_basis(self):
        rng=np.random.default_rng(821);check=CanonicalAGPCheck(4,2)
        for _ in range(24):
            U,_=np.linalg.qr(rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)))
            state=check.rotated_full_state(rng.normal(size=4)+1j*rng.normal(size=4),U)
            weights=rng.normal(size=8)
            A=weights[check.configurations].sum(axis=1)
            K2=(weights**2)[check.configurations].sum(axis=1)
            p=abs(state)**2;mean=p@A;variance=p@(A*A)-mean**2
            self.assertLess(variance-(2*(p@K2)-mean**2/2),2e-12)

    def test_full_number_spectrum_includes_single_occupancies(self):
        weights=np.array([-.7,-.1,.4,1.2]);X,Y=full_number_spectrum(weights,2)
        values=[]
        spin=np.repeat(weights,2)
        for subset in combinations(range(8),4):
            a=spin[list(subset)];values.append((a.sum()/2,(a*a).sum()/2))
        for x,y in values:
            self.assertLess(np.min(abs(X-x)+abs(Y-y)),1e-13)
        self.assertEqual(len(set(zip(np.round(X,12),np.round(Y,12)))),len(set((round(x,12),round(y,12)) for x,y in values)))

    def test_weight_shift_invariance_and_finite_overlap_bound(self):
        model=picket_fence(4,2.);rng=np.random.default_rng(822)
        state=model.initial.copy()/np.sqrt(2);state[model.index[12]]=1/np.sqrt(2)
        weights=np.array([-1.,-1.,1.,1.])
        stats=weighted_statistics(model,state,weights);shifted=weighted_statistics(model,state,weights+2.3)
        self.assertAlmostEqual(stats['excess'],shifted['excess'],places=12)
        self.assertGreater(stats['excess'],0.)
        result=weighted_fidelity_bound(model,state,weights)
        self.assertLess(result['fidelity_upper_bound'],1.)
        check=CanonicalAGPCheck(4,2)
        for _ in range(25):
            free=check.coefficients(rng.normal(size=4)+1j*rng.normal(size=4))
            self.assertLess(abs(np.vdot(state,free))**2,result['fidelity_upper_bound']+1e-13)


if __name__=='__main__':unittest.main()
