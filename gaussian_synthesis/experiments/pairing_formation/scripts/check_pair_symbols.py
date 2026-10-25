#!/usr/bin/env python3
"""Independent Haar/Wick checks of actual Gaussian pair-correlator insertions."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,argparse,multiprocessing,hashlib
from pathlib import Path
from types import SimpleNamespace
from math import comb
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.pair_symbols import PairSymbol2
from pairing_formation.gaussian_bins import haar_atoms


def worker(task):
    out,device,L,fermi,count,seed=task;begin=time.monotonic();model=picket_fence(L,2)
    backend=PairSymbol2(model,device=f'cuda:{device}');torch=backend.torch;torch.manual_seed(seed+device)
    if fermi:u=model.initial.copy()
    else:
        rng=np.random.default_rng(1709+L);u=rng.normal(size=model.dim)+1j*rng.normal(size=model.dim);u/=np.linalg.norm(u)
    backend.prepare(u);pairs=[(0,0),(model.r,model.r),(model.r,0),(0,model.r),(model.r+1,0),(0,1)]
    methods=('full','anomalous','normal_exchange');parts=[backend.symbol_parts(i,j) for i,j in pairs]
    expected=np.array([backend.C]+[part[method] for part in parts for method in methods])
    if fermi:
        other=model.index[((1<<model.r)-1)^(1<<0)^(1<<model.r)]
        selected=np.array([model.fermi_index,other]);sample_model=SimpleNamespace(L=L,masks=model.masks[selected])
        expected=expected[:,selected];u=u[selected]
    else:sample_model=model
    dtype=torch.complex128;dev=f'cuda:{device}';ut=torch.as_tensor(u,device=dev)
    current=torch.zeros((3*len(pairs)+1,len(u)),device=dev,dtype=dtype);blocks=[];last=time.monotonic()
    for offset in range(0,count,256):
        atoms,rho,kappa=haar_atoms(sample_model,256,dev,torch,return_covariances=True)
        pair=kappa[:,1::2,::2].diagonal(dim1=1,dim2=2)
        features=[torch.ones(256,device=dev,dtype=dtype)]
        for i,j in pairs:
            anomalous=pair[:,i].conj()*pair[:,j]
            normal=rho[:,2*i,2*j]*rho[:,2*i+1,2*j+1]-rho[:,2*i,2*j+1]*rho[:,2*i+1,2*j]
            features.extend((anomalous+normal,anomalous,normal))
        labels=torch.stack(features,dim=1);overlap=atoms.conj()@ut
        coefficients=(comb(4*L,2*L)/2)*overlap.abs()**2*overlap
        current+=labels.T@(coefficients[:,None]*atoms)
        if (offset+256)%4096==0:blocks.append((current/4096).cpu().numpy());current.zero_()
        if time.monotonic()-last>20:
            print(json.dumps({'device':device,'samples':offset+256,'seconds':time.monotonic()-begin}),flush=True);last=time.monotonic()
    blocks=np.array(blocks);rows=[]
    for j,name in enumerate(['synthesis']+[str(p)+'_'+method for p in pairs for method in methods]):
        error=np.linalg.norm(blocks[:,j].mean(axis=0)-expected[j]);se=np.sqrt(np.sum(np.var(blocks[:,j],axis=0,ddof=1))/len(blocks))
        rows.append({'symbol':name,'error':float(error),'Frobenius_SE':float(se),'error_over_SE':float(error/se),
                     'expected_norm':float(np.linalg.norm(expected[j]))})
    path=out/f'device{device}.npz';np.savez_compressed(path,blocks=blocks,expected=expected,u=u,pairs=pairs)
    record={'device':device,'L':L,'fermi':fermi,'seed':seed+device,'samples':count,'seconds':time.monotonic()-begin,'checks':rows,
        'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    print(json.dumps(record),flush=True);return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--per-device',type=int,default=131072)
    p.add_argument('--seed',type=int,default=1710);a=p.parse_args()
    assert a.per_device%4096==0;a.output.mkdir(parents=True,exist_ok=False)
    tasks=[(a.output,d,L,fermi,a.per_device,a.seed) for d,(L,fermi) in enumerate([(4,False),(4,False),(6,False),(6,False),(8,False),(8,False),(14,True),(14,True)])]
    records=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(worker,t) for t in tasks]):records.append(f.result())
    (a.output/'summary.json').write_text(json.dumps({'scope':'Independent numerical Gaussian Wick/Haar integration, with estimated standard errors; not a confidence certificate.','jobs':records},indent=2))
