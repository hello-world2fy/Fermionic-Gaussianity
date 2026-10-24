import unittest
import numpy as np

from fk4 import paths
from fk4.tempering import actions
from fk4.gpu_tempering import heatbath_device, sample_tempered_gpu
from fk4.gpu_haar import sample_haar_device
from fk4.reference import log_dimension, magic_log_z
from fk.gaussian import sample_gaussian
from fk3.discovery import covariance_from_kets

try:
    import cupy as cp
    CUDA = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, RuntimeError):
    CUDA = False


@unittest.skipUnless(CUDA, 'CUDA/CuPy unavailable')
class GpuTemperingTests(unittest.TestCase):
    def test_haar_independent_frame_projector_moments_and_gaussianity(self):
        rng=np.random.default_rng(5910)
        for n in (4,8):
            states=sample_haar_device(n,65536,rng).get()
            np.testing.assert_allclose(np.linalg.norm(states,axis=1),1,atol=1e-13)
            cov=covariance_from_kets(states[:256],n=n)
            self.assertLess(np.max(abs(cov@cov+np.eye(2*n))),1e-12)
            frame=states.conj().T@states/len(states)
            self.assertLess(np.linalg.norm(frame-np.eye(states.shape[1])/states.shape[1]),.006)
            for k in (1,2,3,4):
                values=abs(states[:,0])**(2*k)
                expected=np.exp(-log_dimension(n,k))
                self.assertLess(abs(values.mean()-expected),6*values.std()/np.sqrt(len(values)))
            # Every Majorana-pair covariance has the isotropic SO(2n) variance.
            all_cov=covariance_from_kets(states[:8192],n=n)
            self.assertLess(abs(np.mean(all_cov[:,0,1]**2)-1/(2*n-1)),.012)

    def test_prescribed_rotations_match_full_cpu_kets_and_purity(self):
        rng = np.random.default_rng(5901)
        for n in (4, 8):
            original = sample_gaussian(n, 19, rng).kets
            witness = rng.normal(size=original.shape[1])+1j*rng.normal(size=original.shape[1])
            permutations, phases = actions(n)
            planes = rng.permutation(len(permutations))
            angles = rng.uniform(-np.pi, np.pi, (len(original), len(planes)))
            reference = original.copy()
            for step,p in enumerate(planes):
                moved = reference[:,permutations[p]]*phases[p]
                reference = np.cos(angles[:,step,None])*reference+np.sin(angles[:,step,None])*moved
            device = cp.asarray(original)
            heatbath_device(device, witness, 3.5, rng, n=n, planes=planes,
                            prescribed_angles=angles)
            np.testing.assert_allclose(device.get(), reference, atol=3e-14, rtol=1e-12)
            cov = covariance_from_kets(device.get(), n=n)
            self.assertLess(np.max(abs(cov@cov+np.eye(2*n))), 1e-12)

    def test_single_plane_exact_conditional_density(self):
        rng = np.random.default_rng(5902)
        original = sample_gaussian(4, 1, rng).kets[0]
        witness = rng.normal(size=8)+1j*rng.normal(size=8)
        perms, phases = actions(4)
        p = 4
        moved = phases[p]*original[perms[p]]
        a, b = np.vdot(witness,original), np.vdot(witness,moved)
        angles = np.linspace(-np.pi,np.pi,100000,endpoint=False)
        r = abs(a*np.cos(angles)+b*np.sin(angles))**2
        for beta in (0., .5, 8., 128.):
            expected = np.sum(r**(beta+1))/np.sum(r**beta)
            device = cp.asarray(np.tile(original,(32768,1)))
            heatbath_device(device,witness,beta,rng,n=4,planes=np.array([p]))
            measured = abs(device.get()@witness.conj())**2
            se = measured.std(ddof=1)/np.sqrt(len(measured))
            self.assertLess(abs(measured.mean()-expected),6*se+1e-11)

    def test_analytic_magic_and_gaussian_tempered_measures(self):
        for n,kind,beta in ((4,'magic',8),(4,'magic',128),(8,'gaussian',8)):
            witness = np.zeros(2**(n-1),complex)
            witness[0] = 1
            if kind=='magic':
                witness[-1]=1
                witness/=np.sqrt(2)
            result=sample_tempered_gpu(witness,beta,8192,np.random.default_rng(5903+beta+n),n=n)
            expected = magic_log_z(beta) if kind=='magic' else -log_dimension(n,beta)
            self.assertLess(abs(result.log_z-expected),.1)
            mean = (beta+1)/(beta+4)/2 if kind=='magic' else np.exp(log_dimension(n,beta)-log_dimension(n,beta+1))
            self.assertLess(abs(np.exp(result.log_r).mean()-mean),.01)
            cov=covariance_from_kets(result.kets[:128],n=n)
            self.assertLess(np.max(abs(cov@cov+np.eye(2*n))),1e-10)


if __name__=='__main__':
    unittest.main()
