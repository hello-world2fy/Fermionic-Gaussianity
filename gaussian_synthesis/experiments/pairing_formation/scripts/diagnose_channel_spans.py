#!/usr/bin/env python3
"""Post-baseline Gaussian-generated span controls; not unseen forecasts.

These conventional Hamiltonian-selected dictionaries test whether releasing
the CC coefficient constraints already explains the counting discrepancy.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import sys
from pathlib import Path
import json,argparse,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.frozen import channel_basis
from run_nonlinear_channels import TIMES,name,state_rows


def job(task):
    L,eta,q,output=task
    start=time.monotonic()
    model,forms=angular_model(L,4.,eta)
    angle=2*np.pi*np.arange(model.r)/model.r
    orbitals=np.column_stack([f(k*angle) for k in ((1,) if q==2 else (1,3)) for f in (np.cos,np.sin)])
    U,_=np.linalg.qr(orbitals)
    B,_=channel_basis(model,U@U.T,q)
    action=model.H@B
    H=B.conj().T@action
    closure=float(np.linalg.norm(action-B@H)/np.linalg.norm(action))
    c=B.conj().T@model.initial
    initial_error=float(np.linalg.norm(B@c-model.initial))
    coefficients=expm_multiply(-1j*H,c,start=0.,stop=TIMES[-1],num=len(TIMES),traceA=-1j*np.trace(H))
    states=coefficients@B.T
    tag=f'{name(L,eta)}_q{q}'
    np.savez_compressed(output/f'{tag}.npz',times=TIMES,states=states,H_reduced=H,initial_coefficients=c)
    rows=state_rows(model,states,forms)
    truth=np.load(output.parent/'nonlinear_channels_01'/f'{name(L,eta)}_exact.npz')['states']
    exact=json.loads((output.parent/'nonlinear_channels_01'/f'{name(L,eta)}_evaluation.json').read_text())['exact']
    for row,ref,x,y in zip(rows,exact,states,truth):
        row['fidelity']=float(abs(np.vdot(x,y))**2)
        row['count_TV']=float(.5*np.sum(abs(np.array(row['count'])-ref['count'])))
    report={'L':L,'eta':eta,'q':q,'dimension':B.shape[1],'full_dimension':model.dim,
            'relative_H_closure_residual':closure,'initial_state_error':initial_error,
            'seconds':time.monotonic()-start,'rows':rows,
            'dictionary_source':'Known interaction form factors, plus the analytically generated third angular harmonics for q4. No canonical synthesis or exact-state fit.',
            'classification':'Post-baseline mechanism/deletion diagnostic, not an independent new prediction.'}
    (output/f'{tag}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'condition':tag,'dimension':B.shape[1],'closure':closure,'time2':rows[40],'seconds':report['seconds']}),flush=True)
    return {'condition':tag,'dimension':B.shape[1],'initial_state_error':initial_error,'seconds':report['seconds']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    tasks=[(10,1.,2),(14,1.,2),(18,1.,2),(14,.85,2),(14,1.,4),(14,.85,4)]
    (args.output/'protocol.json').write_text(json.dumps({'tasks':tasks,'scope':'Post-baseline diagnostic on already inspected physical conditions. Do not count as new-parameter prediction.',
       'question':'Does allowing arbitrary coherent coefficients within the original two-channel Gaussian span address the failure of pair-CC? What changes when the known Pauli-generated directions are added?',
       'same_initial':'Exact Fermi sea in every subspace','limitation':'q4 has 2233/3432 states at L14, so success is not evidence for an inexpensive or canonical-specific representation.'},indent=2))
    summary=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(job,(*t,args.output)) for t in tasks]):summary.append(f.result())
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
