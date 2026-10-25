#!/usr/bin/env python3
"""Local Gaussian density modes, not measure draws or global support proofs."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,time,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.pairing_channels import angular_model
from pairing_formation.replica2 import Replica2
from pairing_formation.columns import onsite_objective,onsite_atom


def job(task):
    tag,kind,start,out=task;data=np.load(out/f'{tag}_training.npz');L=int(data['L'])
    model=angular_model(L)[0] if tag.startswith('angular') else picket_fence(L,2.)
    target=data[kind];target=target/np.linalg.norm(target);rng=np.random.default_rng(860+start)
    density=np.clip(abs(target)**2@model.occ,1e-6,1-1e-6)
    a=.5*np.log(density/(1-density));_,indices=model.blocks[1]
    U,s,Vh=np.linalg.svd(target[indices],full_matrices=False);phase=np.r_[-np.angle(Vh[0]),np.angle(U[:,0])]
    if start:
        a+=rng.normal(0,1.1,L)
        phase+=rng.normal(0,1. if start%2 else 2.5,L)
        if start%4==0:
            a=np.r_[-np.ones(model.r),np.ones(model.r)]*rng.uniform(-1.5,1.5)+rng.normal(0,.4,L)
    fit=minimize(lambda x:onsite_objective(model,target,x),np.r_[a,phase],jac=True,method='L-BFGS-B',
      bounds=[(-12,12)]*L+[(None,None)]*L,options={'ftol':2e-14,'gtol':2e-9,'maxiter':600,'maxcor':15})
    atom=onsite_atom(model,fit.x)
    return {'tag':tag,'kind':kind,'start':start,'parameters':fit.x,'atom':atom,
      'score_normalized_source':float(abs(np.vdot(target,atom))**2),'success':bool(fit.success),
      'iterations':int(fit.nit),'gradient_norm':float(np.linalg.norm(fit.jac)),
      'projected_atom_norm_squared':float(np.vdot(atom,atom).real),'bound_active':bool(np.max(abs(fit.x[:L]))>11.99)}


def main(out,starts,workers):
    out.mkdir(parents=True,exist_ok=False);root=out.parent;begin=time.monotonic();cases=[]
    for label,L,t in (('angular',10,.5),('angular',10,1.),('angular',10,2.),('uniform',12,1.),('uniform',12,2.15)):
        model=angular_model(L)[0] if label=='angular' else picket_fence(L,2.)
        if label=='angular' and t<=1:
            original=root/'angular_canonical_pilot_01'/f't{t:g}.npz';saved=np.load(original)
        elif label=='uniform':
            original=root/'replica2_L12_G2_01'/f't{t:g}.npz';saved=np.load(original)
        else:
            original=None;state=model.evolve([t])[0];witness,report=Replica2(model).solve(state)
            saved={'target':state,'witness':witness};(out/f'{label}_L{L}_t{t:g}_synthesis.json').write_text(json.dumps(report,indent=2))
        tag=f'{label}_L{L}_t{t:g}';np.savez_compressed(out/f'{tag}_training.npz',L=L,time=t,target=saved['target'],witness=saved['witness'])
        cases.append({'tag':tag,'source':str(original) if original else 'Exact evolution and deterministic full-Gaussian k2 solve on previously inspected condition.',
          'source_sha256':hashlib.sha256(original.read_bytes()).hexdigest() if original else None,
          'direct_witness_ray_fidelity':float(abs(np.vdot(saved['target'],saved['witness']))**2/np.linalg.norm(saved['witness'])**2)})
    protocol={'classification':'Retrospective local Gaussian-mode pilot on already inspected times. Density modes are not measure samples, their basin counts are not probabilities, and local optimization is not a global support certificate.',
      'cases':cases,'onsite_starts_per_source':starts,'sources':['target','witness'],
      'question':'Do canonical density modes require independent pairing fields, or can nonlinear spin saturation of the original fields describe their shapes? Direct-overlap and conventional-field controls mandatory.',
      'full_dictionary_witness':True,'mode_search_currently_restricted_to_onsite':True,'next_gate':'Check general-Gaussian variations before making any claim about a complete-dictionary mode.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks=[(case['tag'],kind,i,out) for case in cases for kind in ('target','witness') for i in range(starts)]
        for f in as_completed([pool.submit(job,t) for t in tasks]):results.append(f.result())
    summary=[]
    for case in cases:
        for kind in ('target','witness'):
            rows=sorted((r for r in results if r['tag']==case['tag'] and r['kind']==kind),key=lambda x:x['start'])
            np.savez_compressed(out/f"{case['tag']}_{kind}.npz",parameters=np.array([r.pop('parameters') for r in rows]),atoms=np.array([r.pop('atom') for r in rows]))
            (out/f"{case['tag']}_{kind}.json").write_text(json.dumps(rows,indent=2))
            summary.append({'tag':case['tag'],'kind':kind,'best_score':max(r['score_normalized_source'] for r in rows),
              'worst_score':min(r['score_normalized_source'] for r in rows),'successes':sum(r['success'] for r in rows),'bound_active':sum(r['bound_active'] for r in rows)})
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({'seconds':time.monotonic()-begin,'summary':summary},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--starts',type=int,default=32);p.add_argument('--workers',type=int,default=16)
    args=p.parse_args();main(args.output,args.starts,args.workers)
