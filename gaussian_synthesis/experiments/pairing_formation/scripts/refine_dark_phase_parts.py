#!/usr/bin/env python3
"""Polish the k2 equation and recheck the two principal phase-part models."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from scipy.sparse.linalg import LinearOperator,cg,expm_multiply
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.replica2_gpu import Replica2GPU
from pairing_formation.dark_pairs import PairDarkSectors
from pairing_formation.coherent_partition import coherent_parts
from pilot_dark_phase_parts import augment


def main(out):
    out.mkdir(parents=True,exist_ok=False);root=out.parent
    source=root/'dark_gaussian_training_01/training.npz';data=np.load(source)
    original=root/'dark_phase_parts_01';spaces=np.load(original/'frozen_spaces.npz')
    model,forms=angular_model(14);Q=data['Q'];u=data['witness'].copy();target=data['target'];n=Q.shape[1]
    replica=Replica2GPU(model);history=[]
    for step in range(3):
        residual=replica.evaluate(u)[1]-target;norm=float(np.linalg.norm(residual))
        record={'step':step,'residual':norm};history.append(record)
        if norm<2e-13:break
        def action(x):
            v=Q@(x[:n]+1j*x[n:]);answer=Q.T@replica.derivative(u,v)
            return np.r_[answer.real,answer.imag]
        op=LinearOperator((2*n,2*n),matvec=action,dtype=float)
        rhs=-Q.T@residual
        solution,status=cg(op,np.r_[rhs.real,rhs.imag],rtol=1e-10,atol=1e-15,maxiter=100)
        record['cg_status']=int(status);assert status==0
        u+=Q@(solution[:n]+1j*solution[n:])
    final=float(np.linalg.norm(replica.evaluate(u)[1]-target));assert final<2e-12
    dark=PairDarkSectors(model,forms);projected=dark.apply(u,'either');A=[];B=[]
    for m in (2,3):
        parts=coherent_parts(replica,u,projected*(model.m==m))
        A.append(Q.T@parts['complex_ratio']);B.append(Q.T@parts['conjugate_ratio'])
    base=spaces['shared'];H=Q.T@(model.H@Q);checks=[];saved={'witness':u}
    for name,candidates,rank in [('canonical_complex_q2',np.column_stack(A),2),
                                  ('canonical_phase_pair_q4',np.column_stack(A+B),4)]:
        V,_=augment(base,candidates,rank);saved[name+'_basis']=V
        old=spaces[name]
        projector_difference=float(np.linalg.norm(V@V.conj().T-old@old.conj().T))
        Hr=V.conj().T@H@V
        for start in ('Fermi','restart'):
            data0=np.load(original/f'{name}_{start}.npz');times=data0['times']
            initial=Q.T@(model.initial if start=='Fermi' else target)
            states=expm_multiply(-1j*Hr,V.conj().T@initial,start=0.,stop=times[-1]-times[0],
                                  num=len(times),traceA=-1j*np.trace(Hr))@V.T
            saved[name+'_'+start+'_states']=states
            checks.append({'name':name,'start':start,'basis_projector_difference':projector_difference,
                'max_state_difference':float(np.max(np.linalg.norm(states-data0['states'],axis=1))),
                'max_norm_error':float(np.max(abs(np.sum(abs(states)**2,axis=1)-1)))})
    path=out/'refinement.npz';np.savez_compressed(path,**saved)
    report={'history':history,'final_synthesis_residual':final,
        'witness_change':float(np.linalg.norm(u-data['witness'])),
        'checks':checks,'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'scope':'Numerical stability of the small conjugate phase-part directions; four repeated trajectories, not new physical predictions.'}
    (out/'refinement.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();main(args.output)
