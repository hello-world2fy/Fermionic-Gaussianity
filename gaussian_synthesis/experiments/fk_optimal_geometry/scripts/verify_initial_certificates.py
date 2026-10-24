#!/usr/bin/env python3
"""Independent dyadic polar algebra, rational inertia and100-digit scalars."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import mp
from fractions import Fraction
from fk5.io import write_json
from fk5.support_validation import exact_rational_spectrum_check
from fk.model import build_model


def independent_basis_and_polar(model):
    # Scaled basis D0=2s0, D1=sqrt8*s1, D2=2sqrt7*s2.
    D=np.zeros((256,3),complex)
    for mask,sign in [(119,1),(120,-1),(135,-1),(136,1)]:D[mask,0]=sign
    for mask,sign in [(15,-1),(60,1),(90,-1),(105,1),(150,1),(165,-1),(195,1),(240,-1)]:D[mask,1]=1j*sign
    for mask,val in [(30,2),(45,-2),(75,2),(180,2),(210,-2),(225,2),(119,-1),(120,-1),(135,-1),(136,-1)]:D[mask,2]=val
    W=(D[:,0]/2-D[:,1]+D[:,2]/2).reshape(16,16)
    assert np.array_equal(W.conj().T@W,np.eye(16))
    projectors=[W.conj().T@D[:,0].reshape(16,16)/2,-W.conj().T@D[:,1].reshape(16,16),W.conj().T@D[:,2].reshape(16,16)/2]
    for P,rank in zip(projectors,[1,8,7]):
        assert np.array_equal(P,P.conj().T) and np.array_equal(P@P,P)
        assert np.trace(P)==rank
    assert np.array_equal(sum(projectors),np.eye(16))
    for i in range(3):
        for j in range(3):
            if i!=j:assert not (projectors[i]@projectors[j]).any()
    assert np.array_equal(model.h_int@D,D*np.array([-56,-14,-8]))
    assert np.array_equal(model.h_hop@D[:,0],D[:,1])
    assert np.array_equal(model.h_hop@D[:,1],2*(D[:,0]+D[:,2]))
    assert np.array_equal(model.h_hop@D[:,2],7*D[:,1])
    # All products above are small Gaussian integers/dyadics, exactly
    # representable throughout binary64 summation; no tolerance is used.
    return D,W.reshape(-1)


def scalar(z):return mp.mpc(float(np.real(z)),float(np.imag(z)))


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    s=json.loads((args.run/'summary.json').read_text());assert s['status']=='complete'
    model=build_model();D,W=independent_basis_and_polar(model);mp.dps=100
    source=Path(s['cost_source']);assert hashlib.sha256((source/'summary.json').read_bytes()).hexdigest()==s['cost_source_sha256']
    assert json.loads((source/'verification.json').read_text())['artifact_verification_passed']
    manifest=json.loads((args.run/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():assert hashlib.sha256((args.run/'source_snapshot'/name).read_bytes()).hexdigest()==digest
    spectra={r['t']:r for r in s['spectra']};rational_checks=[]
    for r in s['A1_rows']:
        t=r['t'];spectrum=spectra[t]
        rational_checks.append({'t':t,**exact_rational_spectrum_check(model,t,spectrum)})
        a=np.load(args.run/f't{t:g}_witness.npz');v=model.embed(a['target'])
        assert np.array_equal(a['omega_S_full'],W)
        z=[scalar(x) for x in v];norm2=mp.fsum(abs(x)**2 for x in z)
        Hv=[]
        for i in range(256):
            columns=np.flatnonzero((model.h_int[i]!=0)|(model.h_hop[i]!=0))
            Hv.append(mp.fsum((scalar(model.h_int[i,j])+mp.mpf(t)*scalar(model.h_hop[i,j]))*z[j] for j in columns))
        energy=mp.fsum(mp.conj(x)*y for x,y in zip(z,Hv)).real/norm2
        residual=mp.sqrt(mp.fsum(abs(y-energy*x)**2 for x,y in zip(z,Hv))/norm2)
        independent_error=abs(mp.sqrt(norm2)-1)+mp.sqrt(2)*residual/(mp.mpf(spectrum['first_excited_energy_lower'])-energy)
        olderr=mp.mpf(r['physical_state_enclosure']['saved_target_to_exact_ground_state_distance_upper'])
        assert independent_error<=olderr
        schmidt=mp.fsum((mp.conj(scalar(w))*x).real for w,x in zip(W,z))
        magic=mp.fsum((mp.conj(scalar(w))*x).real for w,x in zip(D[:,0],z))
        for value,radius,interval in [(schmidt,4*olderr,r['S_physical_interval']),(magic,2*olderr,r['L0_physical_interval'])]:
            assert mp.mpf(interval[0])<=value-radius and mp.mpf(interval[1])>=value+radius
        lam=mp.mpf(r['lambda_numerator'])/r['lambda_denominator'];assert lam>=4
        H=1+lam+2/(lam-2);norm=mp.sqrt(4*lam*lam+4*lam+16)
        N=schmidt+lam*magic-norm*olderr
        assert mp.mpf(r['support_upper'])>=H and mp.mpf(r['witness_norm_upper'])>=norm
        assert mp.mpf(r['joint_physical_overlap_lower'])<=N
        assert mp.mpf(r['joint_cost_lower'])<=N/H
        U=mp.mpf(r['cost_upper']);L=mp.mpf(r['best_cost_lower'])
        assert mp.mpf(r['relative_gap_upper'])>=1-L/U
        if t==20:
            assert Fraction(r['lambda_numerator'],r['lambda_denominator'])==Fraction(4713,1000)
            assert r['relative_gap_upper']<.01
        ml=max(mp.mpf(0),mp.sqrt(mp.mpf(r['S_physical_interval'][0])/U)-1)**2
        mh=min(mp.mpf(1),2*(1-mp.mpf(r['L0_physical_interval'][0])/U))
        assert mp.mpf(r['all_cost_le_U_mean_eta_interval'][0])<=ml
        assert mp.mpf(r['all_cost_le_U_mean_eta_interval'][1])>=mh
        variance=(H-N/U)/((lam-2)/2)
        assert mp.mpf(r['all_cost_le_U_mean_squared_sqrt_eta_displacement_upper'])>=variance
        for tail in r['sqrt_eta_tail_bounds']:
            assert mp.mpf(tail['fraction_upper'])>=min(mp.mpf(1),variance/mp.mpf(tail['delta_sqrt_eta'])**2)
    assert len(s['D1_D2_rows'])==48
    for r in s['D1_D2_rows']:
        t=mp.mpf(r['t']);delta=mp.mpf(r['delta_numerator'])/r['delta_denominator']
        X=(mp.sqrt(16*(t-9)**2+72*delta)-4*(t-9))/36
        h=1-(1-X)**2/2
        assert mp.mpf(r['window_X_interval'][0])<=X<=mp.mpf(r['window_X_interval'][1])
        assert mp.mpf(r['restricted_support_upper'])>=h
        d=max(mp.mpf(0),(mp.mpf(r['physical_overlap_lower'])-mp.mpf(r['coefficient_budget'])*h)/2)
        assert mp.mpf(r['phase_aligned_state_distance_lower'])<=d
        assert mp.mpf(r['fidelity_upper'])>=(1-d*d/2)**2
        gamma=mp.mpf(r['Hamiltonian_gap_lower'])
        assert gamma==mp.mpf(spectra[r['t']]['full_spectral_gap_lower'])
        assert mp.mpf(r['ground_energy_error_lower'])<=gamma*(d*d-d**4/4)
    write_json(args.run/'verification.json',{'artifact_verification_passed':True,
        'A1_joint_witnesses_checked':len(s['A1_rows']),'D1_D2_budget_window_bounds_checked':48,
        'exact_polar_monomial_and_three_projector_identities':True,
        'exact_reduced_Hamiltonian_invariance':True,'exact_rational_spectral_checks':rational_checks,
        'independent_scalar_decimal_digits':100,
        'verification_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Complete dictionary joint supports and scalar concentration; all normalized cost-budget energy-window states. No global contact classification inferred.'})
    print('Verified7 joint witnesses, exact polar identities,7 full spectra by rational LDL,48 fidelity/energy bounds.')


if __name__=='__main__':main()
