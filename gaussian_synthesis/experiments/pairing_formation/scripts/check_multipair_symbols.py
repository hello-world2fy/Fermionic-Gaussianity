#!/usr/bin/env python3
"""Independent full-Gaussian Wick/Pfaffian integration of eight-fermion labels."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,time,argparse,multiprocessing,hashlib
from pathlib import Path
from types import SimpleNamespace
from math import comb
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.multipair_symbols import MultiPairSymbol2,pair_word
from pairing_formation.gaussian_bins import haar_atoms
from pairing_formation.projected_gaussian import torch_pfaffian


def wick_label(word,rho,kappa,torch):
    length=len(word)
    matrix=torch.zeros((len(rho),length,length),device=rho.device,dtype=rho.dtype)
    for i,(a,dagger_a) in enumerate(word):
        for j in range(i+1,length):
            b,dagger_b=word[j]
            if dagger_a and dagger_b:value=kappa[:,b,a].conj()
            elif dagger_a:value=rho[:,a,b]
            elif dagger_b:value=(1. if a==b else 0.)-rho[:,b,a]
            else:value=kappa[:,a,b]
            matrix[:,i,j]=value;matrix[:,j,i]=-value
    return torch_pfaffian(matrix,torch)


def worker(task):
    out,device,L,fermi,count,seed=task;begin=time.monotonic();model=picket_fence(L,2)
    b=MultiPairSymbol2(model,f'cuda:{device}');torch=b.torch;torch.manual_seed(seed+device)
    if fermi:u=model.initial.copy()
    else:
        rng=np.random.default_rng(1909+L);u=rng.normal(size=model.dim)+1j*rng.normal(size=model.dim);u/=np.linalg.norm(u)
    b.prepare(u);pairs=[((model.r,model.r+1),(0,1)),((0,1),(model.r,model.r+1)),((0,model.r),(1,model.r+1))]
    expected=np.array([b.C]+[b.monomial_symbol(*p) for p in pairs])
    if fermi:
        selected=np.array([model.fermi_index,model.index[((1<<model.r)-1)^3^(3<<model.r)]])
        sample_model=SimpleNamespace(L=L,masks=model.masks[selected]);expected=expected[:,selected];u=u[selected]
    else:sample_model=model
    dev=f'cuda:{device}';ut=torch.as_tensor(u,device=dev);current=torch.zeros((4,len(u)),device=dev,dtype=torch.complex128)
    blocks=[];last=time.monotonic()
    for offset in range(0,count,256):
        atoms,rho,kappa=haar_atoms(sample_model,256,dev,torch,return_covariances=True)
        labels=torch.stack([torch.ones(256,device=dev,dtype=torch.complex128)]+[wick_label(pair_word(*p),rho,kappa,torch) for p in pairs],dim=1)
        overlap=atoms.conj()@ut;weight=comb(4*L,2*L)/2*overlap.abs()**2*overlap
        current+=labels.T@(weight[:,None]*atoms)
        if (offset+256)%4096==0:blocks.append((current/4096).cpu().numpy());current.zero_()
        if time.monotonic()-last>20:
            print(json.dumps({'device':device,'samples':offset+256,'seconds':time.monotonic()-begin}),flush=True);last=time.monotonic()
    blocks=np.array(blocks);checks=[]
    for i,name in enumerate(['synthesis']+[str(p) for p in pairs]):
        error=np.linalg.norm(blocks[:,i].mean(axis=0)-expected[i]);se=np.sqrt(np.sum(np.var(blocks[:,i],axis=0,ddof=1))/len(blocks))
        checks.append({'symbol':name,'error':float(error),'estimated_Frobenius_SE':float(se),'error_over_SE':float(error/se),'expected_norm':float(np.linalg.norm(expected[i]))})
    path=out/f'device{device}.npz';np.savez_compressed(path,blocks=blocks,expected=expected,u=u,pairs=pairs)
    row={'device':device,'L':L,'fermi':fermi,'seed':seed+device,'samples':count,'seconds':time.monotonic()-begin,'checks':checks,
         'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    print(json.dumps(row),flush=True);return row


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--per-device',type=int,default=131072);p.add_argument('--seed',type=int,default=2010);a=p.parse_args()
    assert a.per_device%4096==0;a.output.mkdir(parents=True,exist_ok=False)
    tasks=[(a.output,d,L,fermi,a.per_device,a.seed) for d,(L,fermi) in enumerate([(4,False),(4,False),(6,False),(6,False),(8,False),(8,False),(14,True),(14,True)])]
    rows=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(worker,t) for t in tasks]):rows.append(f.result())
    (a.output/'summary.json').write_text(json.dumps({'scope':'Independent full-SO Haar/Pfaffian check with estimated standard errors, not a confidence certificate.','jobs':rows},indent=2))
