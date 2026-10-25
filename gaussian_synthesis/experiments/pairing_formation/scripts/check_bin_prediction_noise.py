#!/usr/bin/env python3
"""Quadrature sensitivity of an inspected physical pilot, not new holdouts."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.column_prediction import evolve_span


def main(out):
    out.mkdir(parents=True,exist_ok=False);root=out.parent
    model,_=angular_model(10);source=root/'gaussian_bins_L10_01'
    data=np.load(source/'training.npz');Q=data['Q'];base=np.load(root/'gaussian_bins_assessment_01/coherent_bins.npz')['base']
    streams=[np.load(source/f'device{i}.npz') for i in range(8)]
    protocol={'classification':'Post-pilot Monte Carlo sensitivity, not independent physical predictions. No parameter or basis tuning against future states.',
      'rank':2,'grouping':'Four count bins, fixed calibration; shared F, psi(t1), C2(psi(t1)).',
      'pools':'Two independent even/odd GPU halves plus 32 block-bootstrap resamples of the full pool.',
      'bootstrap_is_rigorous_confidence_interval':False,'seed':852,'primary_time':2.}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    records=[];saved={}
    for method in ('canonical','husimi_moment','linear_frame'):
        even=np.concatenate([streams[d][method+'_blocks'] for d in (0,2,4,6)]).reshape(128,4,3,26).sum(axis=2)
        odd=np.concatenate([streams[d][method+'_blocks'] for d in (1,3,5,7)]).reshape(128,4,3,26).sum(axis=2)
        allblocks=np.concatenate((even,odd));rng=np.random.default_rng(852)
        averages=[('even_half',even.mean(axis=0)),('odd_half',odd.mean(axis=0))]
        averages += [(f'bootstrap_{i}',allblocks[rng.integers(0,256,256)].mean(axis=0)) for i in range(32)]
        for tag,A in averages:
            A-=A@base.conj()@base.T;A-=A.mean(axis=0)
            U,s,_=np.linalg.svd(A.T,full_matrices=False);V=Q@np.column_stack((base,U[:,:2]))
            key=method+'_'+tag;saved[key+'_basis']=V
            for start,initial,elapsed in (('Fermi',model.initial,2.),('restart',data['target'],1.)):
                coeff=V.conj().T@initial
                initial_error=float(np.linalg.norm(V@coeff-initial));assert initial_error<2e-12
                state=evolve_span(model,V,coeff,[elapsed])[0];saved[key+'_'+start]=state
                records.append({'method':method,'pool':tag,'start':start,'key':key+'_'+start,'initial_error':initial_error})
    np.savez_compressed(out/'frozen_quadrature_variants.npz',**saved)
    (out/'frozen_record.json').write_text(json.dumps({'sha256':hashlib.sha256((out/'frozen_quadrature_variants.npz').read_bytes()).hexdigest(),'records':records},indent=2))
    exact=np.load(root/'nonlinear_channels_01/L10_eta1_exact.npz');index=np.argmin(abs(exact['times']-2));assert abs(exact['times'][index]-2)<1e-12
    ref=exact['states'][index];ref/=np.linalg.norm(ref);p0=np.bincount(model.m,weights=abs(ref)**2,minlength=6)
    chi=data['chi'];chi0=np.linalg.norm(chi@ref)**2/25
    I0=np.array([2*np.vdot(ref[model.m==m],(model.H@ref)[model.m==m]).imag for m in range(6)])
    for r in records:
        state=saved[r['key']];state=state/np.linalg.norm(state)
        p=np.bincount(model.m,weights=abs(state)**2,minlength=6)
        I=np.array([2*np.vdot(state[model.m==m],(model.H@state)[model.m==m]).imag for m in range(6)])
        r.update({'fidelity':float(abs(np.vdot(ref,state))**2),'count_TV':float(.5*np.sum(abs(p-p0))),
          'chi_second_error':float(np.linalg.norm(chi@state)**2/25-chi0),'current_L1_error':float(np.sum(abs(I-I0)))})
    summary=[]
    for method in ('canonical','husimi_moment','linear_frame'):
        for start in ('Fermi','restart'):
            rows=[r for r in records if r['method']==method and r['start']==start]
            summary.append({'method':method,'start':start,'independent_halves':rows[:2],
              'bootstrap_quantiles_05_50_95':{k:np.quantile([r[k] for r in rows[2:]],[.05,.5,.95]).tolist()
                for k in ('fidelity','count_TV','chi_second_error','current_L1_error')}})
    (out/'sensitivity.json').write_text(json.dumps({'records':records,'summary':summary},indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();main(args.output)
