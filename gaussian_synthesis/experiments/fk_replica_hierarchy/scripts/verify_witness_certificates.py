#!/usr/bin/env python3
"""Independently cross-check certificate artifacts at 100 decimal digits.

The producer supplies directed interval proofs. This verifier independently
evaluates the polynomial, radical basis, and window formulas with mpmath;
its high-precision numerical check is not a new interval proof.
"""
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


def c(x):return mp.mpc(float(np.real(x)),float(np.imag(x)))


def beta(z):
    a,b,d=map(c,z)
    return [(a-2*mp.sqrt(2)*b+mp.sqrt(7)*d)/4,
            (mp.sqrt(14)*a-mp.sqrt(2)*d)/4,
            (a+2*mp.sqrt(2)*b+mp.sqrt(7)*d)/4]


def q4(z,proof):
    b=beta(z);x=[b[0],b[1]/mp.sqrt(14),b[2]]
    v=[mp.fprod(x[j]**p[j] for j in range(3)) for p in proof['powers']]
    value=mp.fsum(mp.conj(v[i])*v[j]*int(proof['gram_numerator'][i][j])
                  for i in range(len(v)) for j in range(len(v)))/int(proof['projector_denominator'])
    assert abs(value.imag)<mp.mpf('1e-90')
    return value.real


def ket(z):
    a,b,d=map(c,z);v=[mp.mpc(0) for _ in range(256)]
    for j,s in [(119,1),(120,-1),(135,-1),(136,1)]:v[j]+=s*a/2
    for j,s in [(15,-1),(60,1),(90,-1),(105,1),(150,1),(165,-1),(195,1),(240,-1)]:v[j]+=s*1j*b/mp.sqrt(8)
    for j,s in [(30,2),(45,-2),(75,2),(180,2),(210,-2),(225,2),(119,-1),(120,-1),(135,-1),(136,-1)]:v[j]+=s*d/(2*mp.sqrt(7))
    return [v[j] for j in range(256) if j.bit_count()%2==0]


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path)
    p.add_argument('--moment',type=Path,default=Path('results/cartan_exact_k4_01'));args=p.parse_args()
    mp.dps=100;s=json.loads((args.run/'summary.json').read_text());assert s['status']=='complete'
    manifest=json.loads((args.run/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((args.run/'source_snapshot'/name).read_bytes()).hexdigest()==digest
    proof=json.loads((args.moment/'proof.json').read_text())
    assert json.loads((args.moment/'verification.json').read_text())['artifact_verification_passed']
    model=build_model();data={};checked=0;enclosures={}
    mode='k1' if s.get('analytic_identity_certificate') else 'window' if 'witnesses' in s else 'block'
    for row in s.get('witnesses',s['rows']):
        t=row['t']
        key=row['key'] if mode=='window' else f"t{t:g}_k{row.get('direction_order',1)}"
        suffix='.npz' if mode=='window' else '_frozen.npz' if mode=='k1' else '_certificate.npz'
        a=np.load(args.run/(key+suffix));z=a['z_feasible'];target=a['target'];w=ket(z)
        if 'source_sha256' in row and row['source_sha256']:
            assert hashlib.sha256(Path(row['source_file']).read_bytes()).hexdigest()==row['source_sha256']
            np.testing.assert_array_equal(a['direction_z'],np.load(row['source_file'])['z'])
        q=mp.fsum(abs(c(x))**2 for x in z) if mode=='k1' else q4(z,proof)
        assert q<=1
        if mode!='k1':
            bounds=row['q_normalized_interval'] if mode=='window' else row['anchor_moment_interval']
            assert mp.mpf(bounds[0])<=q<=mp.mpf(bounds[1])
        if mode=='block':
            assert row['target_moment_order']%4==0
            assert q**(row['target_moment_order']//4)<=mp.mpf(row['high_order_normalized_q_upper'])<=1
        norm=mp.sqrt(mp.fsum(abs(x)**2 for x in w))
        inner=mp.fsum((mp.conj(x)*c(y)).real for x,y in zip(w,target))
        if t not in enclosures:enclosures[t]=enclose_ground_state(model,t,target,digits=90)
        # Recomputed 90-digit directed enclosure must support the saved bound.
        enclosure=row['physical_target_enclosure'] if mode=='k1' else s['physical_target_enclosures'][str(t)]
        error=mp.mpf(enclosure['saved_target_to_exact_ground_state_distance_upper'])
        assert enclosures[t]['saved_target_to_exact_ground_state_distance_upper']<=float(error)*(1+1e-14)
        Lkey={'k1':'L_certified_physical_target','block':'L_certified_exact_physical_target','window':'L_exact_physical_ground_lower'}[mode]
        assert mp.mpf(row[Lkey])<=inner-norm*error
        if mode=='k1':
            raw=ket(a['z']);rho=mp.sqrt(mp.fsum(abs(x-c(y))**2 for x,y in zip(raw,target)))
            assert rho<=mp.mpf(row['full_reconstruction_residual_upper'])
        data[key]=(z,row);checked+=1
    if mode=='window':
        for r in s['rows']:
            if r.get('cost_threshold_kind')=='certified_constructive_upper':
                directory=Path(r['cost_certificate_source'])
                assert json.loads((directory/'verification.json').read_text())['artifact_verification_passed']
                cost=next(v for v in json.loads((directory/'summary.json').read_text())['rows'] if v['t']==r['t'])
                assert r['declared_cost_threshold']==cost['corrected_Gaussian_cost_upper']
                assert not r['existence_of_cost_U_decomposition_not_asserted_by_this_certificate']
            z,w=data[r['witness_key']];t=mp.mpf(r['t']);delta=mp.mpf(r['delta'])
            A=4*(t-9)
            # Independent quadratic-root expression (producer uses a rationalized one).
            X=min(mp.mpf(2),(mp.sqrt(A*A+72*delta)-A)/36)
            b=list(map(abs,beta(z)))
            h=b[0]+mp.sqrt(mp.mpf(3)/2)*b[1]*X+b[2]*X*X/4
            assert h<=mp.mpf(r['restricted_support_upper'])
            assert mp.mpf(r['x_cap_interval'][0])<=X<=mp.mpf(r['x_cap_interval'][1])
            hsaved=mp.mpf(r['restricted_support_upper'])
            fraction=mp.mpf(1) if hsaved>=1 else min(mp.mpf(1),max(mp.mpf(0),(1-mp.mpf(r['L_physical_lower'])/mp.mpf(r['declared_cost_threshold']))/(1-hsaved)))
            assert fraction<=mp.mpf(r['fraction_upper'])
            assert r['nontrivial']==(r['fraction_upper']<1)
    write_json(args.run/'verification.json',{'artifact_verification_passed':True,'mode':mode,
        'witnesses_checked':checked,'window_rows_checked':len(s['rows']) if mode=='window' else 0,
        'independent_polynomial_and_radical_basis_evaluation_digits':100,
        'physical_target_enclosures_recomputed_digits':90,
        'verification_scope':'Independent high-precision artifact cross-check of the producer directed bounds; no Monte Carlo normalization certified.'})
    print('verified',mode,checked,flush=True)


if __name__=='__main__':main()
