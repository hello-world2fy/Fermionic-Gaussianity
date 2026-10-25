import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.linalg import expm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.pair_cc_dynamics import PairCCDynamics
from pairing_formation.cluster_dynamics import ClusterDynamics,transfer_cluster_operator,connected_two_transfer,connected_source


class ClusterTests(unittest.TestCase):
    def test_polynomial_recurrence_and_jacobian_against_matrix_exponential(self):
        model=picket_fence(6,2.);rng=np.random.default_rng(730)
        matrix=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3));matrix/=np.linalg.norm(matrix)
        dynamics=ClusterDynamics(model,[(2,matrix)])
        z=.1*(rng.normal(size=10)+1j*rng.normal(size=10))
        K=transfer_cluster_operator(model,1,z[:9].reshape(3,3))+z[9]*dynamics.operators[0]
        expected=expm(K.toarray())@model.initial
        raw,D=dynamics.raw_and_jacobian(z)
        np.testing.assert_allclose(raw,expected,atol=2e-14)
        v=rng.normal(size=10)+1j*rng.normal(size=10);step=1e-6
        derivative=(dynamics.raw_and_jacobian(z+step*v)[0]-dynamics.raw_and_jacobian(z-step*v)[0])/(2*step)
        np.testing.assert_allclose(D@v,derivative,atol=2e-10)

    def test_zero_added_clusters_recovers_pair_CC(self):
        model=picket_fence(8,2.);rng=np.random.default_rng(731)
        z=.2*(rng.normal(size=16)+1j*rng.normal(size=16))
        old=PairCCDynamics(model);new=ClusterDynamics(model,[])
        for a,b in zip(old.raw_and_jacobian(z),new.raw_and_jacobian(z)):
            np.testing.assert_allclose(a,b,atol=2e-14)
        y=np.r_[1.+0j,z]
        np.testing.assert_allclose(old.derivative(0,y),new.derivative(0,y),atol=2e-12)

    def test_connected_source_is_exact_directional_derivative(self):
        model=picket_fence(8,2.);state=model.evolve([.8])[0]
        rate=-1j*(model.H@state);step=1e-6
        finite=(connected_two_transfer(model,state+step*rate)-connected_two_transfer(model,state-step*rate))/(2*step)
        np.testing.assert_allclose(connected_source(model,state),finite,atol=2e-9)

    def test_physical_rate_and_conservation_with_extra_clusters(self):
        model=picket_fence(8,2.);rng=np.random.default_rng(732)
        matrix=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6));matrix/=np.linalg.norm(matrix)
        dynamics=ClusterDynamics(model,[(2,matrix)])
        z=dynamics.initial();z[1:]=.1*(rng.normal(size=17)+1j*rng.normal(size=17))
        rate,diagnostic,physical=dynamics.derivative(0,z,True)
        step=1e-6;finite=(dynamics.state(z+step*rate)-dynamics.state(z-step*rate))/(2*step)
        np.testing.assert_allclose(physical,finite,atol=2e-8)
        self.assertLess(abs(diagnostic['norm_rate']),1e-11)
        self.assertLess(abs(diagnostic['energy_rate']),1e-10)

    def test_free_radial_coordinates_preserve_all_count_Ehrenfest_equations(self):
        model=picket_fence(8,2.);rng=np.random.default_rng(733)
        matrix=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6));matrix/=np.linalg.norm(matrix)
        dynamics=ClusterDynamics(model,[(2,matrix)],radial=True,metric_cutoff=1e-13)
        z=dynamics.initial();z[1:]=.25*(rng.normal(size=len(z)-1)+1j*rng.normal(size=len(z)-1))
        state=dynamics.state(z);rate,_,physical=dynamics.derivative(0,z,True)
        step=1e-6;finite=(dynamics.state(z+step*rate)-dynamics.state(z-step*rate))/(2*step)
        np.testing.assert_allclose(physical,finite,atol=2e-8)
        variational=np.bincount(model.m,weights=2*np.real(state.conj()*physical),minlength=model.r+1)
        exact=np.bincount(model.m,weights=2*np.real(state.conj()*(-1j*(model.H@state))),minlength=model.r+1)
        np.testing.assert_allclose(variational,exact,atol=2e-11)

    def test_direct_SVD_solver_with_exact_coordinate_redundancies(self):
        model=picket_fence(8,2.);rng=np.random.default_rng(734)
        matrix=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6));matrix/=np.linalg.norm(matrix)
        # Repeated clusters and the radial/singles scale gauge make the
        # coordinate metric singular even at a generic physical state.
        clusters=[(2,matrix),(2,matrix)]
        gram=ClusterDynamics(model,clusters,radial=True,metric_cutoff=1e-11)
        svd=ClusterDynamics(model,clusters,radial=True,metric_cutoff=1e-11,metric_solver='svd')
        for scale in (0.,.2):
            z=svd.initial();z[1:]=scale*(rng.normal(size=len(z)-1)+1j*rng.normal(size=len(z)-1))
            rate,report,physical=svd.derivative(0,z,True)
            _,other,reference=gram.derivative(0,z,True)
            self.assertEqual(report['metric_kept'],other['metric_kept'])
            np.testing.assert_allclose(physical,reference,atol=2e-10)
            step=1e-6
            finite=(svd.state(z+step*rate)-svd.state(z-step*rate))/(2*step)
            np.testing.assert_allclose(physical,finite,atol=2e-8)
            self.assertLess(abs(report['norm_rate']),2e-12)
            self.assertLess(abs(report['energy_rate']),2e-11)
            if scale==0:
                np.testing.assert_allclose(physical,-1j*(model.H@model.initial),atol=2e-13)

    def test_physical_error_control_in_a_complete_small_manifold(self):
        model=picket_fence(4,2.)
        d=ClusterDynamics(model,[(2,np.ones((1,1)))],radial=True,metric_solver='svd')
        times=np.linspace(0.,.8,9)
        states,_,report=d.evolve(times,rtol=2e-10,atol=2e-12,max_step=.03,error_control='physical')
        np.testing.assert_allclose(states,model.evolve(times),atol=3e-9)
        self.assertLess(report['max_norm_error'],3e-10)


if __name__=='__main__':unittest.main()
