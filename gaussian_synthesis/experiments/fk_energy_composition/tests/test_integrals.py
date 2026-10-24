from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk6.weighted import EnergyWeightedMoments
from fk6.coherent import coherent_kernels,split_vectors
from fk4.weighted import WeightedMoments
from fk.model import build_model
from fk.gaussian import sample_gaussian


class IntegralTests(unittest.TestCase):
    def test_weighted_gradient_and_cost_identity(self):
        rng=np.random.default_rng(8871)
        h=rng.normal(size=(40,3))+1j*rng.normal(size=(40,3))
        energies=rng.uniform(0,4,40);logs=np.full(40,-np.log(40));z=rng.normal(size=3)+1j*rng.normal(size=3)
        for k in [1,2,4]:
            ordinary=WeightedMoments(h,logs,k,0)
            zero=EnergyWeightedMoments(h,logs,energies,0,k,0)
            np.testing.assert_allclose(zero.evaluate(z).gradient,ordinary.evaluate(z).gradient,atol=1e-12)
            backend=EnergyWeightedMoments(h,logs,energies,.7,k,0)
            values=backend.evaluate(z);costs=backend.coefficient_costs(z)
            self.assertAlmostEqual(costs['weighted_cost'],costs['plain_cost']+.7*costs['composition_energy'],places=9)
            direction=rng.normal(size=3)+1j*rng.normal(size=3);eps=1e-6
            difference=(backend.evaluate(z+eps*direction,False).q-backend.evaluate(z-eps*direction,False).q)/(2*eps*2*k)
            derivative=values.gradient@np.r_[direction.real,direction.imag]
            self.assertLess(abs(difference-derivative)/max(1,abs(derivative)),2e-8)

    def test_full_space_coherent_recombination(self):
        model=build_model();rng=np.random.default_rng(8872)
        g=sample_gaussian(8,12,rng).kets
        phases=np.exp(1j*rng.uniform(0,6.28,12));energy=np.arange(12.)
        rows,v=split_vectors(g,phases,2.,energy,[3.5,8.5],model)
        np.testing.assert_allclose(v['low_raw']+v['high_raw'],np.broadcast_to(v['total_raw'],(2,128)),atol=1e-15)
        low,high=v['low_raw'][0],v['high_raw'][0]
        summary,a=coherent_kernels(model,11,low,high);phi=model.embed(low+high)
        self.assertAlmostEqual(sum(summary['unnormalized_parts_low_low_high_high_cross']['energy']),np.vdot(phi,model.full_hamiltonian(11)@phi).real,places=12)
        for key,indices in [('two',a['pairs']),('four',a['quartets'])]:
            # Direct dense Majorana action is independent of permutation-kernel construction.
            for n in [0,len(indices)//2,len(indices)-1]:
                acted=phi.copy()
                for j in reversed(indices[n]):acted=model.gammas[j]@acted
                if key=='two':acted*=1j
                expected=np.vdot(phi,acted).real
                self.assertAlmostEqual(float(a[key+'_parts'][:,n].sum()),expected,places=12)
        self.assertFalse(rows[0]['conditional_sampling_pass'])


if __name__=='__main__':unittest.main()
