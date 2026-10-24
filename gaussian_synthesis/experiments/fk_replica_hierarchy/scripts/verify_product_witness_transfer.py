#!/usr/bin/env python3
"""Independent high-precision scalar checks and full saved-column diagnostics."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import mp
from fk4.io import write_json
from fk4.state_enclosure import enclose_ground_state
from fk.model import build_model
from verify_witness_certificates import ket,q4


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    s=json.loads((args.run/'summary.json').read_text());assert s['status']=='complete'
    model=build_model();mp.dps=100;w=ket([2.,0.,0.]);w_float=np.array([complex(x) for x in w])
    proof=json.loads(Path('results/cartan_exact_k4_01/proof.json').read_text())
    q4rows=json.loads(Path('results/cartan_k4_physical_certificates_02/summary.json').read_text())['rows']
    manifest=json.loads((args.run/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():assert hashlib.sha256((args.run/'source_snapshot'/name).read_bytes()).hexdigest()==digest
    columns=0;violation=-np.inf;means=[]
    for r in s['rows']:
        t=r['t'];target=np.load(args.run/f't{t:g}.npz')['target']
        enclosure=enclose_ground_state(model,t,target,digits=90)
        err=mp.mpf(r['physical_target_enclosure']['saved_target_to_exact_ground_state_distance_upper'])
        assert enclosure['saved_target_to_exact_ground_state_distance_upper']<=float(err)*(1+1e-14)
        inner=mp.fsum((mp.conj(a)*mp.mpc(float(b.real),float(b.imag))).real for a,b in zip(w,target))
        if t==0:
            integer_state=model.embed(w_float)
            assert np.array_equal(integer_state.real,np.rint(integer_state.real)) and not integer_state.imag.any()
            assert np.array_equal(model.h_int@integer_state,-56*integer_state)
            assert r['L_endpoint_witness_physical_lower']==r['U_certified']==2
        else:assert mp.mpf(r['L_endpoint_witness_physical_lower'])<=inner-2*err
        directory=Path(r['cost_source']);assert json.loads((directory/'verification.json').read_text())['artifact_verification_passed']
        assert hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest()==s['cost_certificate_sha256']
        cost=next(x for x in json.loads((directory/'summary.json').read_text())['rows'] if x['t']==t)
        if t!=0:assert r['U_certified']==cost['corrected_Gaussian_cost_upper']
        a=np.load(Path('results/cartan_k4_certified_01')/f't{t:g}_k4_frozen.npz');safe=a['z_feasible'];cw=ket(safe)
        assert q4(safe,proof)<=1
        L4=next(x['L_certified_exact_ground_state'] for x in q4rows if x['t']==t)
        overlap=mp.fsum((mp.conj(x)*mp.mpc(float(y.real),float(y.imag))).real for x,y in zip(cw,target))
        norm=mp.sqrt(mp.fsum(abs(x)**2 for x in cw));assert mp.mpf(L4)<=overlap-norm*err
        L,U=mp.mpf(r['L_endpoint_witness_physical_lower']),mp.mpf(r['U_certified'])
        eta=min(mp.mpf(1),2*(1-L/U));assert eta<=mp.mpf(r['mean_eta_upper_all_cost_le_U'])
        lower=max(mp.mpf(1),L,mp.mpf(L4));assert lower==mp.mpf(r['best_combined_physical_cost_lower'])
        assert mp.mpf(r['Gaussian_extent_lower'])<=lower**2
        assert mp.mpf(r['Gaussian_extent_upper'])>=U**2
        assert mp.mpf(r['certified_global_relative_gap_upper'])>=1-lower/U
        for tail in r['tail_bounds']:assert mp.mpf(tail['fraction_upper'])>=min(mp.mpf(1),eta/mp.mpf(tail['eta0']))
        data=np.load(cost['source_file']);cov=data['finite_covariances'];coeff=abs(data['finite_coefficients']);g=data['finite_kets']
        etas=np.sum(cov[:,:8,8:]**2,axis=(1,2))/8
        violation=max(violation,float(np.max(abs(g.conj()@w_float)-(1-etas/2))))
        average=float(coeff@etas/coeff.sum());assert average<=r['mean_eta_upper_all_cost_le_U']+1e-9
        means.append({'t':t,'original_floating_column_weighted_eta':average})
        columns+=len(g)
    assert violation<1e-9
    for r in s['window_rows']:
        t,delta=mp.mpf(r['t']),mp.mpf(r['delta']);a=4*(t-9)
        X=(mp.sqrt(a*a+72*delta)-a)/36
        eta=(1-X)**2
        arc=[(1-X/2)**2,3*(2*X-X*X)/(2*mp.sqrt(14)),X*X/4]
        overlap=arc[0]/2+mp.sqrt(14)*arc[1]/2+arc[2]/2
        assert abs(overlap-(1-eta/2))<mp.mpf('1e-90')
        assert mp.mpf(r['X_interval'][0])<=X<=mp.mpf(r['X_interval'][1])
        assert mp.mpf(r['eta_window_lower'])<=eta
        assert mp.mpf(r['restricted_support_upper'])>=1-eta/2
        bound=min(mp.mpf(1),2*(1-mp.mpf(r['L_endpoint_witness_physical_lower'])/mp.mpf(r['U_certified']))/eta)
        assert bound<=mp.mpf(r['fraction_upper'])
    write_json(args.run/'verification.json',{'artifact_verification_passed':True,'targets_checked':len(s['rows']),
        'window_bounds_checked':len(s['window_rows']),'independent_scalar_decimal_digits':100,
        'explicit_arc_saturates_ideal_restricted_support_formula':True,
        'all_saved_column_branches_support_checked':columns,'maximum_measured_sharp_support_violation':violation,
        'original_floating_column_means':means,
        'scope':'Scalar directed-bound artifacts independently checked; all-column diagnostics are numerical checks, while universal validity comes from the analytic support and energy proofs.'})
    print('verified endpoint transfer',len(s['rows']),len(s['window_rows']),columns)


if __name__=='__main__':main()
