#!/usr/bin/env python3
"""IID rare-region integrals with known proposal volume and explicit confidence bounds."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import iv
from scipy.stats import beta
from fk6.paths import PREVIOUS
from fk6.io import write_json,source_manifest
from fk6.rare import sample_number_ball,number_volume_constant,hopping_fock_frame
from fk6.coherent import rb_vector
from fk4.reference import log_dimension
from fk4.certificates import _bounds
from fk.model import build_model


def task(args):
    t,fraction,seed,count,out=args;out=Path(out);started=time.monotonic()
    model=build_model();frame=hopping_fock_frame(model)
    ids=model.even_masks;H=model.full_hamiltonian(t)[np.ix_(ids,ids)]
    target=model.target(t);EG=-14-4*t;E0=float(np.vdot(target,H@target).real)
    delta=float(fraction*(EG-E0));iv.dps=80
    a=4*(iv.mpf(t)-9);D=iv.mpf(delta);X=2*D/(iv.sqrt(a*a+72*D)+a)
    cap=float(np.nextafter(2*_bounds(X)[1],np.inf))
    if cap>=1:raise ArithmeticError('Number simplex does not cover this window')
    volume=float(np.nextafter(float(number_volume_constant())*cap**28,np.inf))
    volume_upper=_bounds(iv.mpf(cap)**28/23178480)[1]
    rng=np.random.default_rng(np.random.SeedSequence([seed,611,round(t*10000),round(fraction*1000)]))
    kets=[];pairs=[]
    for offset in range(0,count,2048):
        g,p=sample_number_ball(min(2048,count-offset),cap,rng)
        kets.append(g@frame.T);pairs.append(p)
    g=np.concatenate(kets);p=np.concatenate(pairs)
    energy=np.einsum('ni,ni->n',g.conj(),g@H.T).real-EG
    mask=energy<=delta;hits=int(mask.sum());alpha=1e-8
    pupper=1. if hits==count else float(np.nextafter(beta.ppf(1-alpha,hits+1,count-hits),np.inf))
    rows=[];arrays={}
    old=PREVIOUS.parent/'fk_replica_hierarchy/results'
    for k in [2,4]:
        data=np.load(old/f'cartan_k{k}_certified_01/t{t:g}_k{k}_frozen.npz');u=data['witness']
        overlap=g.conj()@u
        integrand=np.where(mask,overlap*abs(overlap)**(2*k-2),0)
        d=round(np.exp(log_dimension(8,k)))
        low=d*volume*(integrand@g)/count
        mass=d*volume*float(np.sum(np.where(mask,abs(overlap)**(2*k-1),0)))/count
        norm=iv.sqrt(sum(iv.mpf(float(z.real))**2+iv.mpf(float(z.imag))**2 for z in u))
        max_integrand=_bounds(iv.mpf(d)*iv.mpf(volume_upper)*norm**(2*k-1))[1]
        whole_mass_upper=float(np.nextafter(max_integrand*pupper,np.inf))
        weights=np.where(mask,abs(overlap)**(2*k-1),0)
        ess=0. if not np.any(weights) else float(weights.sum()**2/(weights@weights))
        rows.append(dict(k=k,low_coefficient_mass_estimate=mass,low_vector_norm=float(np.linalg.norm(low)),
            low_group_importance_ess=ess,low_whole_coefficient_mass_confidence_upper=whole_mass_upper,
            low_vector_norm_confidence_upper=whole_mass_upper,global_integrand_envelope=max_integrand,
            deterministic_proposal_volume_upper=volume_upper,confidence_failure_probability=alpha,
            enough_hits_for_conditional_shape=ess>=256,
            negligible_contribution_at_absolute_002=whole_mass_upper<.002,
            no_hits_means_zero_claimed=False,confidence_is_deterministic_theorem=False))
        arrays[f'k{k}_low_raw']=low;arrays[f'k{k}_low_rb']=rb_vector(low,model)
        arrays[f'k{k}_phases']=overlap/abs(overlap)
    key=f't{t:g}_fraction{fraction:g}_seed{seed}'
    np.savez(out/f'{key}.npz',kets=g,pair_probabilities=p,energy_excess=energy,in_window=mask,**arrays)
    row=dict(t=t,fraction=fraction,seed=seed,count=count,delta=delta,number_cap=cap,
        proposal_volume=volume,hits=hits,hit_probability_upper=pupper,
        method='IID Haar conditional on sum canonical pair probabilities <= cap; true energy indicator retained',
        hit_bound='One-sided Clopper-Pearson; ideal IID-law confidence, floating sampler error not interval enclosed',
        window_results=rows,elapsed_seconds=time.monotonic()-started)
    write_json(out/f'{key}.json',row)
    print(key,'hits',hits,'upper norms',[r['low_vector_norm_confidence_upper'] for r in rows],flush=True)
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=32768);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--t',nargs='+',type=float,default=[11,20]);p.add_argument('--fractions',nargs='+',type=float,default=[.1,.25,.5,1.])
    p.add_argument('--seeds',nargs='+',type=int,default=[25101,25102,25103,25104,25105])
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    write_json(out/'manifest.json',source_manifest(out/'source_snapshot'))
    write_json(out/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    work=[(t,f,s,args.count,str(out)) for t in args.t for f in args.fractions for s in args.seeds]
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from multiprocessing import get_context
    rows=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
        for future in as_completed([pool.submit(task,x) for x in work]):
            rows.append(future.result());write_json(out/'partial_summary.json',dict(status='running',rows=rows))
    write_json(out/'summary.json',dict(status='complete',rows=rows,elapsed_seconds=time.monotonic()-started,
        every_window_contribution_bounded_below_002=all(w['negligible_contribution_at_absolute_002'] for r in rows for w in r['window_results']),
        simultaneous_failure_probability_union_bound=len(rows)*1e-8,
        conditional_shapes_certified=False,independent_verification_required=True))


if __name__=='__main__':main()
