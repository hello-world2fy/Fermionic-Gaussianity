#!/usr/bin/env python3
"""Exact arithmetic audit of cancellation in the saved finite column model."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
from fractions import Fraction
import argparse,json,hashlib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.sparse import load_npz
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.rotation_geometry import matched_sector_atoms
from pairing_formation.finite_dictionary_cost import FiniteDictionaryCertificate


def job(task):
    L,out=task;begin=time.monotonic();source=out.parent/'rotation_geometry_01';model,forms=angular_model(L)
    Q=load_npz(source/f'L{L}_symmetry.npz');data=np.load(source/f'L{L}_frozen_spaces.npz');base=data['source_q2']
    A=np.concatenate([Q.T@matched_sector_atoms(model,forms@c,a) for c in data['profile_coefficients'] for a in data['amplitudes']],axis=1)
    state=np.load(source/f'L{L}_rotation_profiles16.npz')['states'][40]
    R=A-base@(base.conj().T@A);U,s,_=np.linalg.svd(R,full_matrices=False)
    dual=U-base@(base.conj().T@U);dual/=np.maximum(np.linalg.norm(dual,axis=0),1e-300)
    support=np.max(abs(dual.conj().T@A),axis=1);overlap=abs(dual.conj().T@state)
    selected=[];errors=[Fraction(0),Fraction(1,100),Fraction(1,20),Fraction(1,10)]
    for e in errors:
        scores=np.where(support>1e-12,np.maximum(overlap-float(e),0)/np.maximum(support,1e-300),0)
        w=dual[:,np.argmax(scores)].copy();w*=np.exp(1j*np.angle(np.vdot(w,state)));selected.append(w)
    np.savez_compressed(out/f'L{L}_certificate_inputs.npz',atoms=A,free_basis=base,state=state,witnesses=selected)
    evaluator=FiniteDictionaryCertificate(A,base,state);certs=[evaluator.evaluate(w,e) for w,e in zip(selected,errors)]
    result={'L':L,'source':'rotation_geometry_01, matched 16-profile time-2 predicted state',
      'input_sha256':hashlib.sha256((out/f'L{L}_certificate_inputs.npz').read_bytes()).hexdigest(),
      'certificates':certs,'seconds':time.monotonic()-begin,
      'limitation':'Exact dyadic finite-matrix certificate. Gaussian spinor evaluation and symmetry reduction originally used float64, so this is not an interval certificate for ideal Gaussian atoms or any continuous/full Gaussian dictionary.'}
    (out/f'L{L}_certificate.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'L':L,'seconds':result['seconds'],'bounds':[c['coefficient_l1_lower']['float'] for c in certs]}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,(L,args.output)) for L in (14,18)]):f.result()
