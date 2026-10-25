#!/usr/bin/env python3
"""Retrospective nonlinear-region witnesses and frozen-manifold obstruction."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,hashlib,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.sparse.linalg import LinearOperator,cg
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.replica2_gpu import Replica2GPU
from pairing_formation.cartan_monotonicity import numerical_image_obstruction


def job(task):
    time,device,source,out=task;data=np.load(source/'frozen_bases.npz');Q=data['Q'];model,_=angular_model(14)
    raw=np.load(source.parent/'nonlinear_channels_01/L14_eta1_exact.npz')['states'][round(time*20)]
    target=raw/np.linalg.norm(raw);replica=Replica2GPU(model,f'cuda:{device}');u,report=replica.solve(target,Q);n=Q.shape[1]
    history=[]
    for _ in range(3):
        residual=replica.evaluate(u)[1]-target;norm=float(np.linalg.norm(residual));history.append(norm)
        if norm<2e-13:break
        def action(x):
            z=Q.T@replica.derivative(u,Q@(x[:n]+1j*x[n:]));return np.r_[z.real,z.imag]
        op=LinearOperator((2*n,2*n),matvec=action,dtype=float);r=-Q.T@residual
        solution,status=cg(op,np.r_[r.real,r.imag],rtol=2e-11,atol=1e-15,maxiter=100);assert status==0
        u+=Q@(solution[:n]+1j*solution[n:])
    residual=float(np.linalg.norm(replica.evaluate(u)[1]-target));assert residual<2e-12
    results={}
    for method in ('canonical','direct','Hamiltonian','common'):
        results[method]=numerical_image_obstruction(14,u,Q@data[method],residual)
    path=out/f't{time:.2f}.npz';np.savez_compressed(path,witness=u,target=target)
    answer={'time':time,'residual':residual,'polish_history':history,
        'direct_witness_ray_fidelity':float(abs(np.vdot(target,u))**2/np.linalg.norm(u)**2),
        'obstructions':results,'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (out/f't{time:.2f}.json').write_text(json.dumps(answer,indent=2));print(json.dumps(answer),flush=True);return answer


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps({'scope':'Post-prediction diagnostic at already inspected t2 and t2.5. These witnesses are never used to modify frozen prediction structures.',
        'bound_scope':'Exact analytic coercivity theorem for the fixed-N paired Cartan map; numerical evaluation is float64, not interval certification or a certified exact-physical-state fidelity bound.',
        'source_bases_sha256':hashlib.sha256((args.source/'frozen_bases.npz').read_bytes()).hexdigest()},indent=2))
    records=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(job,(time,device,args.source,args.output)) for device,time in enumerate((2.,2.5))]):records.append(f.result())
    (args.output/'summary.json').write_text(json.dumps(records,indent=2))
