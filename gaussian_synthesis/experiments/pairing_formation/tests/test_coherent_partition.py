import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.replica2 import Replica2
from pairing_formation.coherent_partition import coherent_parts


class CoherentPartitionTests(unittest.TestCase):
    def test_frame_hermiticity_and_symmetric_conjugate_part(self):
        model,_=angular_model(6);replica=Replica2(model)
        rng=np.random.default_rng(960);u=rng.normal(size=model.dim)+1j*rng.normal(size=model.dim)
        u/=np.linalg.norm(u)
        parts=[coherent_parts(replica,u,e) for e in np.eye(model.dim)]
        A=np.column_stack([p['complex_ratio'] for p in parts])
        B=np.column_stack([p['conjugate_ratio'] for p in parts])
        np.testing.assert_allclose(A,A.conj().T,atol=2e-14)
        np.testing.assert_allclose(B,B.T,atol=2e-14)
        self.assertGreater(np.linalg.eigvalsh(A).min(),-2e-14)
        np.testing.assert_allclose(A@u,replica.evaluate(u)[1],atol=2e-14)
        np.testing.assert_allclose(B@u.conj(),replica.evaluate(u)[1],atol=2e-14)

    def test_partition_reconstruction_and_phase_gauge(self):
        model,_=angular_model(6);replica=Replica2(model)
        rng=np.random.default_rng(961);u=rng.normal(size=model.dim)+1j*rng.normal(size=model.dim)
        v=u*(model.m>=2)
        a=coherent_parts(replica,u,v);b=coherent_parts(replica,u,u-v)
        for key in ('complex_ratio','conjugate_ratio'):
            np.testing.assert_allclose(a[key]+b[key],replica.evaluate(u)[1],atol=2e-12)
        phase=np.exp(.73j);changed=coherent_parts(replica,phase*u,phase*v)
        for key in a:np.testing.assert_allclose(changed[key],phase*a[key],atol=2e-12)
        np.testing.assert_allclose(a['real_ratio']+1j*a['imaginary_ratio'],a['complex_ratio'],atol=2e-12)


if __name__=='__main__':unittest.main()
