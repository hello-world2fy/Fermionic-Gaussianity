import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

from fk4 import paths
from fk4.cartan_sparse import CartanMoments, casimir_defect, integer_casimir_defect, exact_hopping_fock_proof
from fk4.exact_cartan import ExactCartan
from fk4.certificates import interval_q2
from fk.model import build_model
from fk2.exact_moments import ExactK2Moments, full_tensor_reconstruction


class CartanTests(unittest.TestCase):
    def test_physical_cayley_identity_in_exact_integers(self):
        self.assertTrue(exact_hopping_fock_proof()['verified'])

    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory()
        cls.directory=Path(cls.temporary.name)/'exact'
        subprocess.run([sys.executable,str(paths.ROOT/'scripts/build_exact_cartan.py'),
                        '--k','2','--output',str(cls.directory)],check=True,capture_output=True)
        cls.matrix=np.load(cls.directory/'moment_matrix.npz')['matrix']
        cls.exact=ExactCartan(cls.directory/'proof.json')
        cls.model=build_model()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complex_probes_against_independent_majorana_projector(self):
        old=ExactK2Moments(self.model)
        new=CartanMoments(2,self.matrix)
        rng=np.random.default_rng(29711)
        for _ in range(10):
            z=rng.normal(size=3)+1j*rng.normal(size=3)
            z/=np.linalg.norm(z)
            a,b=new.evaluate(z),old.evaluate(z)
            self.assertAlmostEqual(a.q,b.q,delta=2e-14)
            np.testing.assert_allclose(a.gradient,b.gradient,atol=2e-14)
            np.testing.assert_allclose(a.hessian,b.hessian,atol=2e-14)
            interval,Cz=self.exact.evaluate(z)
            reference=interval_q2(z)
            self.assertLess(abs(float(interval.mid)-float(reference.mid)),2e-15)
            actual=np.array([complex(float(r.mid),float(i.mid)) for r,i in Cz])
            np.testing.assert_allclose(self.model.S@actual,full_tensor_reconstruction(self.model.S@z),atol=2e-14)

    def test_integer_similarity_and_spectrum(self):
        A,_,_,multiplicities,_=integer_casimir_defect(2)
        B,_,_,_=casimir_defect(2)
        d=np.sqrt(multiplicities)
        np.testing.assert_allclose(d[:,None]*A.toarray()/d[None,:],B.toarray(),atol=2e-14)
        eigenvalues=np.linalg.eigvalsh(B.toarray())
        distance=np.min(abs(eigenvalues[:,None]-np.array([0,64,256])[None,:]),axis=1)
        self.assertLess(distance.max(),1e-11)

    def test_interval_reconstruction_bound_and_normalization(self):
        target=self.model.target(11)
        z=np.array([1.2+.1j,-.2+.3j,.1-.2j])
        result=self.exact.certificate(z,target)
        residual=np.linalg.norm(full_tensor_reconstruction(self.model.S@z)-target)
        self.assertLessEqual(residual,result['full_reconstruction_residual_upper']+2e-14)
        self.assertLessEqual(result['q_normalized_interval'][1],1.)
        self.assertLessEqual(result['L_certified_saved_target'],np.vdot(self.model.S@result['z_feasible'],target).real+2e-14)


if __name__=='__main__':
    unittest.main()
