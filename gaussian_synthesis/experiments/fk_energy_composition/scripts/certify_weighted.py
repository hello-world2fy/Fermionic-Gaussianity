#!/usr/bin/env python3
"""Global full-Gaussian weighted support via a directed covering of number x."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import iv
from fk6.paths import PREVIOUS
from fk6.io import source_manifest,write_json
from fk5.two_orbit import ground_root_enclosure,interval_fraction,scaled_basis
from fk5.window_sharp_bounds import exact_hopping_coordinates
from fk4.certificates import _bounds
from fk.model import build_model


def certify(path,initial,budget20,steps):
    match=re.fullmatch(r't([\d.]+)_k(\d+)_lambda([\d.]+)_frozen.npz',path.name)
    if match is None:raise ValueError(path)
    t,k,penalty=float(match[1]),int(match[2]),float(match[3]);iv.dps=80
    f=np.load(path);z=f['z']
    if np.max(abs(z.imag))>1e-12:raise ValueError('Current certificate supports real singlet directions')
    zr=[iv.mpf(float(v.real)) for v in z]
    w=[zr[0]/4-zr[1]/iv.sqrt(2)+iv.sqrt(7)*zr[2]/4,
       iv.sqrt(14)*zr[0]/4-iv.sqrt(2)*zr[2]/4,
       zr[0]/4+zr[1]/iv.sqrt(2)+iv.sqrt(7)*zr[2]/4]
    spectrum=next(r for r in initial['spectra'] if r['t']==t)
    lo,hi=ground_root_enclosure(t,spectrum['ground_energy_interval'])
    E=iv.mpf([interval_fraction(lo).a,interval_fraction(hi).b]);corr=-14-4*t-E
    q=exact_hopping_coordinates(t,spectrum['ground_energy_interval'])
    overlap=sum(a*b for a,b in zip(w,q));norm=iv.sqrt(sum(v*v for v in zr))
    def h4(x):
        candidates=[iv.mpf(1),iv.sqrt(iv.mpf(3)/2)*x]
        if _bounds(x)[1]<.5:candidates.append(3*x/(iv.sqrt(14)*iv.sqrt(1-2*x)))
        return iv.mpf(min(_bounds(v)[1] for v in candidates))
    covers=[]
    for j in range(steps):
        a=iv.mpf(2*j)/steps;b=iv.mpf(2*(j+1))/steps
        numerator=abs(w[0])*(1-a/2)**2+abs(w[1])*h4(b)+abs(w[2])*b*b/4
        numerator=iv.mpf(min(_bounds(numerator)[1],_bounds(norm)[1]))
        denominator=1+iv.mpf(penalty)*(18*a*a+4*(iv.mpf(t)-9)*a)/corr
        covers.append(_bounds(numerator/denominator)[1])
    support=max(covers);L=_bounds(overlap/iv.mpf(support))[0]
    U=budget20 if t==20 else next(r for r in initial['A1_rows'] if r['t']==t)['cost_upper']
    A=0. if penalty==0 else max(0.,_bounds((iv.mpf(L)-iv.mpf(U))/iv.mpf(penalty))[0])
    Lapprox=_bounds((overlap-norm/100)/iv.mpf(support))[0]
    Aapprox=0. if penalty==0 else max(0.,_bounds((iv.mpf(Lapprox)-iv.mpf(U))/iv.mpf(penalty))[0])
    model=build_model();D=scaled_basis()[model.even_masks]
    exact_basis_numeric=D/np.sqrt(np.sum(abs(D)**2,axis=0))[None,:]
    discrepancy=float(np.linalg.norm(exact_basis_numeric@z.real-f['witness']))
    if discrepancy>1e-12:raise ArithmeticError('Frozen witness coordinate convention mismatch')
    return dict(t=t,k=k,penalty=penalty,interval_decimal_digits=80,partition_steps=steps,
        frozen_source=str(path.resolve()),frozen_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        certified_witness_definition='Real binary64 z coordinates lifted in exact D[:,j]/sqrt(D[:,j]^dagger D[:,j]) basis',
        coordinate_lift_vs_sampling_ket_norm_difference=discrepancy,real_z=z.real.tolist(),
        hopping_coordinate_intervals=[_bounds(v) for v in w],physical_overlap_interval=_bounds(overlap),
        gaussian_weighted_support_upper=support,global_partition_support_uppers=covers,
        weighted_cost_lower=L,reference_plain_budget=U,composition_energy_lower=A,
        composition_energy_lower_with_hilbert_error_001=Aapprox,
        strict_global_certificate=True,nontrivial_energy_certificate=A>0,
        proof='WEIGHTED_SYNTHESIS.md, FK global support covering; full Gaussian dictionary and both parities',
        weighted_Cartan_inequality_assumed=False,certificate_implies_G3=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=4096);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if args.steps<4:raise ValueError('A positive complete covering is required')
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    initial=json.loads((PREVIOUS/'results/initial_certificates_02/summary.json').read_text())
    two=json.loads((PREVIOUS/'results/two_orbit_02/summary.json').read_text())['construction']['exact_construction_cost_interval'][1]
    files=sorted(args.input.glob('*_frozen.npz'))
    if not files:raise ValueError('No frozen weighted witnesses')
    rows=[]
    for path in files:
        row=certify(path,initial,two,args.steps);rows.append(row)
        print(path.name,'L',row['weighted_cost_lower'],'U',row['reference_plain_budget'],'A>=',row['composition_energy_lower'],flush=True)
    write_json(args.output/'summary.json',dict(status='complete',rows=rows,independent_verification_required=True,
        nontrivial_count=sum(r['nontrivial_energy_certificate'] for r in rows)))


if __name__=='__main__':main()
