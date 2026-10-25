#!/usr/bin/env python3
"""Post-pilot physical kick, numerical propagation, and capacity controls."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,argparse,hashlib,multiprocessing,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.occupation_symbols import orthogonal_span


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def kick_worker(task):
    source,out,case,step=task;begin=time.monotonic();data=np.load(source/'frozen_spaces.npz')
    exact=np.load(source/'exact_references.npz');times=exact['times'];O=data[case+'_O'];target=data['target']
    model,_=angular_model(14);means=[]
    for sign in (1,-1):
        initial=np.exp(-1j*sign*step*O)*target
        states=model.evolve(times,initial=initial)
        means.append(np.einsum('ti,i,ti->t',states.conj(),O,states).real)
    response=(means[0]-means[1])/(2*step);truth=2*exact[case+'_correlation'].imag
    path=out/f'{case}_step{step:g}.npz';np.savez_compressed(path,times=times,response=response,truth=truth,kicked_means=means)
    return {'case':case,'step':step,'max_response_error':float(np.max(abs(response-truth))),
            'seconds':time.monotonic()-begin,'file':path.name,'sha256':digest(path)}


def main(source,out):
    out.mkdir(parents=True,exist_ok=False);data=np.load(source/'frozen_spaces.npz');model,_=angular_model(14)
    exact=np.load(source/'exact_references.npz');times=exact['times'];index=int(np.argmin(abs(times-.65)))
    predictions=json.loads((source/'frozen_predictions.json').read_text())['models']
    capacity=[];refined=[];cutoffs=[]
    for row in predictions:
        path=source/row['file'];assert digest(path)==row['sha256']
        V=data[row['key']];ref=exact[row['case']];saved=np.load(path)
        projected=ref@V.conj();maximum=np.sum(abs(projected)**2,axis=1)
        capacity.append({'case':row['case'],'method':row['method'],
            'time_065_subspace_max_fidelity':float(maximum[index]),'time_115_subspace_max_fidelity':float(maximum[-1]),
            'time_065_propagation_fidelity':float(abs(np.vdot(ref[index],saved['states'][index]))**2),
            'time_115_propagation_fidelity':float(abs(np.vdot(ref[-1],saved['states'][-1]))**2)})
        if row['method'] in ('canonical_history','Hamiltonian_d9'):
            H=V.conj().T@(model.H@V);initial=data[row['case']+'_initial'];coordinates=V.conj().T@initial
            independent=expm_multiply(-1j*H,coordinates,start=0,stop=times[-1],num=len(times),traceA=-1j*np.trace(H))@V.T
            error=float(np.max(np.linalg.norm(independent-saved['states'],axis=1)))
            refined.append({'case':row['case'],'method':row['method'],'max_state_difference':error})
    # Check physical moment ranks against a 100-fold smaller SVD cutoff.
    for case in sorted(set(r['case'] for r in predictions)):
        q=int(case[1]);parity=1 if case.endswith('plus') else -1
        f=np.sqrt(2/model.r)*np.cos(2*np.pi*q*np.arange(model.r)/model.r)
        base=data[case+'_initial'][:,None]
        for method in ('canonical','unoptimized'):
            values=np.array([np.load(source/f'{method}_{i}.npz')['centered'] for i in range(4)])
            first=np.einsum('i,tij->tj',np.r_[-parity*f,f],values[:,:,0])
            second=np.einsum('i,tij->tj',np.r_[parity*f,f],values[:,:,1])
            columns=np.column_stack((first.T,second.T));columns-=base@(base.conj().T@columns)
            normalized=columns/np.linalg.norm(columns,axis=0)
            singular=np.linalg.svd(normalized,compute_uv=False)
            new=orthogonal_span(columns,cutoff=1e-13);old=data[case+'__'+method+'_history'][:,1:]
            error=float(np.linalg.norm(new-old@(old.conj().T@new)))
            cutoffs.append({'case':case,'method':method,'singular_values':singular.tolist(),
                'old_added_rank':old.shape[1],'new_added_rank':new.shape[1],'subspace_residual':error})
    tasks=[(source,out,case,step) for case in sorted(set(r['case'] for r in predictions)) for step in (1e-3,5e-4)]
    kicks=[]
    with ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(kick_worker,t) for t in tasks]):kicks.append(f.result())
    richardson=[]
    for case in sorted(set(r['case'] for r in predictions)):
        coarse=np.load(out/f'{case}_step0.001.npz');fine=np.load(out/f'{case}_step0.0005.npz')
        corrected=(4*fine['response']-coarse['response'])/3
        richardson.append({'case':case,'coarse_max_error':float(np.max(abs(coarse['response']-coarse['truth']))),
            'fine_max_error':float(np.max(abs(fine['response']-fine['truth']))),
            'Richardson_max_error':float(np.max(abs(corrected-fine['truth'])))})
    report={'classification':'Post-pilot diagnostics on already inspected conditions; not extra independent forecasts.',
        'source_hashes':{str(p):digest(p) for p in (source/'frozen_spaces.npz',source/'exact_references.npz',source/'frozen_predictions.json')},
        'capacity_scope':'Orthogonal projection gives the exact best fidelity in each saved numerical linear subspace, up to floating point; not a full-Gaussian-family statement or interval certificate.',
        'capacity':capacity,'independent_exponential_checks':refined,'SVD_cutoff_checks':cutoffs,'kicks':kicks,'physical_response_checks':richardson}
    (out/'diagnostics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'max_independent_exponential_difference':max(r['max_state_difference'] for r in refined),
        'max_cutoff_subspace_residual':max(r['subspace_residual'] for r in cutoffs),'physical_kicks':richardson,
        'canonical_capacity':[r for r in capacity if r['method']=='canonical_history']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('results/occupation_response_01'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.source,a.output)
