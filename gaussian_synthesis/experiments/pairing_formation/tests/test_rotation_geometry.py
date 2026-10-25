import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.rotation_geometry import spinors,matched_sector_atoms,orthogonal_columns,two_band_sector_atoms
from pairing_formation.pairing_channels import angular_model
from pairing_formation.columns import onsite_atom
from pairing_formation.frozen import channel_basis


class RotationGeometryTests(unittest.TestCase):
    def test_normalization_and_independent_full_Gaussian_amplitudes(self):
        model,forms=angular_model(10);field=forms@np.array([.8,.3j]);A=.43
        for kind in ('rotation','Thouless'):
            u,v=spinors(field,A,kind)
            np.testing.assert_allclose(abs(u)**2+abs(v)**2,1,atol=2e-15)
            z=v/u;physical=np.r_[1/z.conj(),z]
            direct=onsite_atom(model,np.r_[np.log(abs(physical)),np.angle(physical)])
            atom=matched_sector_atoms(model,field,A,kind).sum(axis=1)
            phase=np.vdot(direct,atom);phase/=abs(phase)
            np.testing.assert_allclose(direct*phase,atom,atol=2e-14)
            self.assertLess(np.linalg.norm(atom),1.)

    def test_Thouless_field_in_source_span_but_rotation_has_curvature(self):
        model,forms=angular_model(14);U,_=np.linalg.qr(forms);B,_=channel_basis(model,U@U.T,2)
        field=forms@np.array([.8,.3j])
        linear=matched_sector_atoms(model,field,.8,'Thouless')
        curved=matched_sector_atoms(model,field,.8,'rotation')
        np.testing.assert_allclose(linear,B@(B.conj().T@linear),atol=2e-13)
        self.assertGreater(np.linalg.norm(curved-B@(B.conj().T@curved)),1e-3)

    def test_cubic_harmonic_is_kinematic_not_independent_field(self):
        r=11;theta=2*np.pi*np.arange(r)/r;field=np.sqrt(2)*np.cos(theta);A=.003
        u,v=spinors(field,A);z=v/u;coefficient=np.fft.fft(z)/r
        expected=(np.sqrt(2)*A)**3/24
        self.assertLess(abs(coefficient[3]/expected-1),1e-4)

    def test_independent_particle_hole_fields_match_onsite_Gaussian(self):
        model,forms=angular_model(10);p=forms@np.array([.7,.6j]);h=forms@np.array([.2j,.9])
        up,vp=spinors(p,.7);uh,vh=spinors(h,.8)
        z=np.r_[uh/vh,vp/up];x=np.r_[np.log(abs(z)),np.angle(z)]
        direct=onsite_atom(model,x);atom=two_band_sector_atoms(model,p,h,.7,.8).sum(axis=1)
        phase=np.vdot(direct,atom);phase/=abs(phase)
        np.testing.assert_allclose(direct*phase,atom,atol=3e-14)


if __name__=='__main__':unittest.main()
