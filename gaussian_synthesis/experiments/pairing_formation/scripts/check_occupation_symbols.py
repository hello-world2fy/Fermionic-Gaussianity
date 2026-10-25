#!/usr/bin/env python3
"""Independent full-SO Haar check of the deterministic occupation insertion."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,argparse,multiprocessing,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from math import comb
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.replica2 import Replica2
from pairing_formation.gaussian_bins import haar_atoms
from pairing_formation.projected_gaussian import torch_module
from pairing_formation.occupation_symbols import occupation_moments,beta_coefficients


def worker(task):
    out,device,L,fermi,count=task;begin=time.monotonic()
    torch=torch_module();torch.set_num_threads(1);torch.manual_seed(1640+device);dev=f'cuda:{device}'
    n=2*L
    if fermi:
        model=SimpleNamespace(L=L,masks=np.array([(1<<(L//2))-1]));u=np.ones(1,complex)
        expected=np.array([beta_coefficients(n,d)[2:3] for d in range(5)])
        linear=(.5+.5/(2*n-1))*u;ward=0.
    else:
        model=picket_fence(L,2);replica=Replica2(model);rng=np.random.default_rng(1639+L)
        u=rng.normal(size=model.dim)+1j*rng.normal(size=model.dim);u/=np.linalg.norm(u)
        expected,check,_=occupation_moments(replica,u,model.occ[:,0],range(5));ward=check['ward_residual']
        linear=(.5+(model.occ[:,0]-.5)/(2*n-1))*u
    blocks=[];linear_blocks=[];probability=[];u_t=torch.as_tensor(u,device=dev)
    d2=.5*comb(2*n,n);d1=2**(n-1)
    current=torch.zeros((5,len(u)),device=dev,dtype=torch.complex128)
    current_linear=torch.zeros(len(u),device=dev,dtype=torch.complex128)
    last=time.monotonic()
    for offset in range(0,count,256):
        atoms,occupation=haar_atoms(model,256,dev,torch,return_occupations=True)
        a=atoms.conj()@u_t;p=occupation[:,0]
        features=torch.stack([p**d for d in range(5)],dim=1).to(torch.complex128)
        current+=features.T@((d2*a.abs()**2*a)[:,None]*atoms)
        current_linear+=(d1*a*p)@atoms
        probability.append(torch.stack([p.mean(),((p-.5)**2).mean()]).cpu().numpy())
        if (offset+256)%4096==0:
            blocks.append((current/4096).cpu().numpy());current.zero_()
            linear_blocks.append((current_linear/4096).cpu().numpy());current_linear.zero_()
        if time.monotonic()-last>20:
            print(json.dumps({'device':device,'L':L,'samples':offset+256}),flush=True);last=time.monotonic()
    blocks=np.array(blocks);linear_blocks=np.array(linear_blocks)
    errors=[]
    for d in range(5):
        error=np.linalg.norm(blocks[:,d].mean(axis=0)-expected[d])
        se=np.sqrt(np.sum(np.var(blocks[:,d],axis=0,ddof=1))/len(blocks))
        errors.append({'degree':d,'error':float(error),'Frobenius_SE':float(se),'error_over_SE':float(error/se),
                       'relative_error':float(error/np.linalg.norm(expected[d]))})
    linear_error=float(np.linalg.norm(linear_blocks.mean(axis=0)-linear))
    linear_se=float(np.sqrt(np.sum(np.var(linear_blocks,axis=0,ddof=1))/len(blocks)))
    path=out/f'device{device}.npz';np.savez_compressed(path,blocks=blocks,expected=expected,
        linear_blocks=linear_blocks,linear_expected=linear,occupation_statistics=np.array(probability),u=u)
    row={'device':device,'L':L,'fermi':fermi,'samples':count,'seconds':time.monotonic()-begin,
         'ward_residual':ward,'moments':errors,'linear_error_over_SE':linear_error/linear_se,
         'Haar_mean':float(np.mean(probability,axis=0)[0]),'Haar_variance':float(np.mean(probability,axis=0)[1]),
         'expected_Haar_variance':1/(4*(2*n-1)),'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    print(json.dumps(row),flush=True);return row


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--per-device',type=int,default=131072);a=p.parse_args()
    assert a.per_device%4096==0;a.output.mkdir(parents=True,exist_ok=False)
    tasks=[(a.output,d,L,fermi,a.per_device) for d,(L,fermi) in enumerate([(4,False),(4,False),(6,False),(6,False),(8,False),(8,False),(10,True),(14,True)])]
    records=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(worker,t) for t in tasks]):records.append(f.result())
    (a.output/'summary.json').write_text(json.dumps({'scope':'Numerical integration checks and estimated standard errors, not rigorous confidence certificates.','jobs':records},indent=2))
