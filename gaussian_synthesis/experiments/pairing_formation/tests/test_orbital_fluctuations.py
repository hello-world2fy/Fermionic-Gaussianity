import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.projected_count_bound import CanonicalAGPCheck
from pairing_formation.orbital_fluctuations import reduced_moments,orbital_count,torch_count


class OrbitalCountTests(unittest.TestCase):
    def test_arbitrary_paired_state_against_full_Fock_onebody_action(self):
        rng=np.random.default_rng(701)
        model=picket_fence(4,2.)
        checker=CanonicalAGPCheck(4,2)
        for _ in range(12):
            s=rng.normal(size=model.dim)+1j*rng.normal(size=model.dim);s/=np.linalg.norm(s)
            Q,_=np.linalg.qr(rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)))
            rank=int(rng.integers(2,7));P=Q[:,:rank]@Q[:,:rank].conj().T
            result=orbital_count(reduced_moments(model,s),P)
            action=np.zeros(len(checker.configurations),complex)
            np.add.at(action,checker.dst,checker.signs*P[checker.ci,checker.cj]*s[checker.src])
            mean=float(np.vdot(s,action[checker.paired_indices]).real)
            variance=float(np.linalg.norm(action)**2-mean**2)
            self.assertAlmostEqual(result['mean'],mean/2,places=12)
            self.assertAlmostEqual(result['variance'],variance/4,places=12)

    def test_energy_subset_count_and_complement(self):
        model=picket_fence(8,2.);s=model.evolve([1.1])[0];moments=reduced_moments(model,s)
        P=np.diag(np.r_[np.zeros(8),np.ones(8)])
        a=orbital_count(moments,P);b=orbital_count(moments,np.eye(16)-P)
        obs=model.observe(s)
        self.assertAlmostEqual(a['mean'],obs['mean'],places=12)
        self.assertAlmostEqual(a['variance'],obs['variance'],places=12)
        self.assertAlmostEqual(a['variance'],b['variance'],places=12)
        self.assertAlmostEqual(a['mean']+b['mean'],model.r,places=12)

    def test_batched_complex_QR_objective_and_directional_gradient(self):
        try:
            import torch
        except ImportError:
            sys.path.append('/usr/local/lib/python3.11/dist-packages')
            try:
                import torch
            except ImportError:
                self.skipTest('Optional Torch backend unavailable')
        torch.set_num_threads(1)
        rng=np.random.default_rng(928)
        model=picket_fence(4,2.)
        state=rng.normal(size=model.dim)+1j*rng.normal(size=model.dim)
        moments=reduced_moments(model,state)
        tm={key:torch.as_tensor(value) for key,value in moments.items()}
        raw=rng.normal(size=(3,8,3))+1j*rng.normal(size=(3,8,3))
        z=torch.tensor(raw,dtype=torch.complex128,requires_grad=True)
        Q=torch.linalg.qr(z,mode='reduced').Q
        means,variances=torch_count(tm,Q@Q.mH)
        expected=[]
        for sample in raw:
            q,_=np.linalg.qr(sample)
            expected.append(orbital_count(moments,q@q.conj().T))
        np.testing.assert_allclose(means.detach(),[x['mean'] for x in expected],atol=2e-13)
        np.testing.assert_allclose(variances.detach(),[x['variance'] for x in expected],atol=2e-13)
        (variances-means).sum().backward()
        for _ in range(4):
            direction=rng.normal(size=raw.shape)+1j*rng.normal(size=raw.shape)
            direction/=np.linalg.norm(direction)
            def objective(samples):
                value=0.
                for sample in samples:
                    q,_=np.linalg.qr(sample)
                    value+=orbital_count(moments,q@q.conj().T)['excess']
                return value
            step=2e-5
            finite=(objective(raw+step*direction)-objective(raw-step*direction))/(2*step)
            analytic=float(np.vdot(z.grad.numpy(),direction).real)
            self.assertAlmostEqual(finite,analytic,delta=2e-9)


if __name__=='__main__':unittest.main()
