#!/usr/bin/env python3
"""Local16 spectral-projector verification, independent of global128 code."""
from pathlib import Path
import argparse
import hashlib
import json
from fractions import Fraction
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import mp,iv
from scipy.linalg import hadamard
from fk5.io import write_json
from fk4.certificates import _bounds
from fk.model import build_model
from verify_initial_certificates import independent_basis_and_polar


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    s=json.loads((args.run/'summary.json').read_text());r=s['construction'];model=build_model()
    D,Wvec=independent_basis_and_polar(model);W=Wvec.reshape(16,16)
    denom=75264;N2=np.zeros((16,16),complex);N4=N2.copy();odd1=N2.copy();odd3=N2.copy()
    maxint=0.
    for sign in hadamard(8)[:,1:]:
        N=2*np.einsum('j,jab->ab',sign,model.local_q_ops)
        square=N@N;fourth=square@square
        assert not (N@(square-28*np.eye(16))@(square-112*np.eye(16))).any()
        assert np.trace(square)==448 and np.trace(fourth)==31360
        N2+=square;N4+=fourth;odd1+=N;odd3+=N@square
        maxint=max(maxint,float(np.max(abs(fourth.real))),float(np.max(abs(fourth.imag))))
    assert not odd1.any() and not odd3.any()
    P2=N4-28*N2;P1=4*(112*N2-N4);P0=denom*np.eye(16)-P2-P1
    # All P are numerator matrices of averaged local spectral projectors.
    main=[.5*W@P2.T,.25*W@P1.T,.25*W@(P0-P2).T]
    rotate=[W@(P0-P2).T/4,W@P1.T/4,W@P2.T/2]
    expected_main=[14*D[:,0]+2*D[:,2],-14*D[:,1],-7*D[:,0]+5*D[:,2]]
    expected_rot=[-7*D[:,0]+5*D[:,2],-14*D[:,1],14*D[:,0]+2*D[:,2]]
    for left,right in zip(main,expected_main):assert np.array_equal(left.reshape(-1)*56,right*denom)
    for left,right in zip(rotate,expected_rot):assert np.array_equal(left.reshape(-1)*56,right*denom)
    # All exact identity arithmetic uses integers/dyadics far below2^53.
    assert maxint<2**45
    t=Fraction(r['t']);lo,hi=[Fraction(*v) for v in r['ground_energy_rational_bracket']]
    def cubic(e):return (-56-e)*((-14-e)*(-8-e)-14*t*t)-2*t*t*(-8-e)
    assert cubic(lo)*cubic(hi)<0
    source=Path(s['initial_certificate_source']);initial=json.loads((source/'summary.json').read_text())
    assert json.loads((source/'verification.json').read_text())['artifact_verification_passed']
    spec=next(x for x in initial['spectra'] if x['t']==20)
    assert Fraction(spec['ground_energy_interval'][0])<lo<hi<Fraction(spec['ground_energy_interval'][1])
    mp.dps=120;tt=mp.mpf(r['t'])
    E,V=mp.eigsy(mp.matrix([[-56,mp.sqrt(2)*tt,0],[mp.sqrt(2)*tt,-14,mp.sqrt(14)*tt],[0,mp.sqrt(14)*tt,-8]]))
    z=V[:,0];z*=mp.sign(z[0]);e=E[0]
    assert mp.mpf(lo.numerator)/lo.denominator<e<mp.mpf(hi.numerator)/hi.denominator
    M0=(5*z[0]+mp.sqrt(7)*z[2])/3;M1=-mp.sqrt(2)*z[1];M2=2*M0-4*z[0]
    # Independent solve: positive eigenvector of the2x2 moment matrix,
    # followed by its signed two-projector decomposition.
    eig,vec=mp.eigsy(mp.matrix([[M0,M1],[M1,M2]]));a=vec[1,1]/vec[0,1]
    cp=eig[1]/(1+a*a);cm=-eig[0]/(1+a*a);cost=cp+cm
    assert cp>0 and cm>0 and 0<a<1
    def enclosed(x,key):
        bounds=r[key];assert mp.mpf(bounds[0])<=x<=mp.mpf(bounds[1]),key
    enclosed(a,'main_arc_parameter_interval');enclosed(a*a,'main_eta_interval')
    enclosed(cp,'positive_orbit_mass_interval');enclosed(cm,'negative_orbit_mass_interval');enclosed(cost,'exact_construction_cost_interval')
    Avec=mp.matrix([(2-a*a)/4,-a/mp.sqrt(2),(2+5*a*a)/(4*mp.sqrt(7))])
    Bvec=mp.matrix([(2*a*a-1)/4,a/mp.sqrt(2),(5+2*a*a)/(4*mp.sqrt(7))])
    assert max(abs(v) for v in cp*Avec-cm*Bvec-z)<mp.mpf('1e-110')
    data=np.load(args.run/'construction_t20.npz');af=float(a);c=float(mp.sqrt((1+mp.sqrt(1-a*a))/2));d=float(mp.sqrt((1-mp.sqrt(1-a*a))/2))
    expected=[];secondary=[]
    for sign in hadamard(8)[:,1:]:
        q=np.einsum('j,jab->ab',sign,model.local_q_ops)/np.sqrt(7)
        values,basis=np.linalg.eigh(q);levels=np.rint(values).astype(int)
        amplitudes=np.array([c**(2+j)*d**(2-j) for j in levels])
        X=(basis*amplitudes)@basis.conj().T
        expected.append((W@X.T).reshape(-1)[model.even_masks])
        U=(basis*np.exp(1j*np.arccos(-af)*levels))@basis.conj().T
        secondary.append((W@U.T/4).reshape(-1)[model.even_masks])
    independent=np.array(expected+secondary)
    error=float(np.max(abs(independent-data['atoms'])))
    assert error<2e-14,error
    assert len(data['atoms'])==16 and np.linalg.matrix_rank(data['atoms'],tol=1e-10)==16
    assert np.linalg.norm(data['coefficients']@data['atoms']-model.target(20))<1e-12
    # Direct full-Fock covariance, separately from the producer's numba kernels.
    maxpurity=0.
    for atom in data['atoms']:
        full=model.embed(atom);cov=np.zeros((16,16))
        acted=model.gammas@full
        for i in range(16):
            for j in range(i+1,16):cov[i,j]=(1j*np.vdot(acted[i],acted[j])).real;cov[j,i]=-cov[i,j]
        maxpurity=max(maxpurity,float(np.max(abs(cov@cov+np.eye(16)))))
    assert maxpurity<1e-12
    iv.dps=90;lower=next(x['joint_cost_lower'] for x in initial['A1_rows'] if x['t']==20)
    gap=_bounds(1-iv.mpf(lower)/iv.mpf(r['exact_construction_cost_interval'][1]))[1]
    write_json(args.run/'verification.json',{'artifact_verification_passed':True,
        'exact_local16_spectral_projector_identities':True,'exact_full16_term_reconstruction_formula':True,
        'actual_gaussian_terms':16,'distinct_Gaussian_rays':16,'no_Fock_repair':True,
        'physical_energy_root_bracket_checked_with_exact_rationals':True,
        'independent_2x2_spectral_decomposition_decimal_digits':120,
        'independent_saved_atom_error_max':error,'direct_full_Fock_purity_error_max':maxpurity,
        'certified_cost_upper':r['exact_construction_cost_interval'][1],
        'certified_relative_gap_upper_with_A1':gap,
        'full_dictionary_matching_dual_certified':False,
        'verification_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print('Verified exact16-Gaussian construction, cost upper',r['exact_construction_cost_interval'][1],'gap upper',gap)


if __name__=='__main__':main()
