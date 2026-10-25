#!/usr/bin/env python3
"""Equal dimension controls for matched and independent pair-field shapes."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.rotation_geometry import two_band_sector_atoms


def job(task):
    L,out=task;root=out.parent;model,forms=angular_model(L);source=root/'rotation_geometry_01'
    Q=load_npz(source/f'L{L}_symmetry.npz');H=load_npz(source/f'L{L}_Hamiltonian.npz');F=Q.T@model.initial
    original=np.load(source/f'L{L}_frozen_spaces.npz');base=original['source_q2'];dimension=original['rotation_profiles16'].shape[1];q=dimension-base.shape[1]
    spaces={};spectra={}
    for family in ('matched','shifted','independent'):
        path=source if family=='matched' else root/f'rotation_geometry_{family}_01';data=np.load(path/f'L{L}_frozen_spaces.npz')
        pc=data['profile_coefficients'];hc=data['hole_profile_coefficients'] if 'hole_profile_coefficients' in data else pc.conj()
        A=np.concatenate([Q.T@two_band_sector_atoms(model,forms@p,forms@h,a,a) for p,h in zip(pc,hc) for a in data['amplitudes']],axis=1)
        norms=np.linalg.norm(A,axis=0);A=A[:,norms>1e-14]/norms[norms>1e-14]
        R=A-base@(base.conj().T@A);U,s,_=np.linalg.svd(R,full_matrices=False)
        assert s[q-1]>1e-13*s[0]
        added=U[:,:q]-base@(base.conj().T@U[:,:q]);added,_=np.linalg.qr(added)
        spaces[family]=np.column_stack((base,added));spectra[family]=s.tolist()
    V=base.copy();block=base.copy();steps=[]
    while V.shape[1]<dimension:
        R=H@block
        for _ in range(2):R-=V@(V.conj().T@R)
        U,s,_=np.linalg.svd(R,full_matrices=False);rank=int(min(sum(s>1e-11*s[0]),dimension-V.shape[1]))
        if rank==0:raise RuntimeError('Block Hamiltonian closure exhausted before the requested dimension')
        block=U[:,:rank]-V@(V.conj().T@U[:,:rank]);block,_=np.linalg.qr(block)
        V=np.column_stack((V,block));steps.append(rank)
    spaces['Hamiltonian_block_response']=V
    np.savez_compressed(out/f'L{L}_frozen_spaces.npz',**spaces)
    predictions={};initial_errors=[]
    for name,V in spaces.items():
        np.testing.assert_allclose(V.conj().T@V,np.eye(dimension),atol=2e-12)
        c=V.conj().T@F;initial_errors.append(float(np.linalg.norm(V@c-F)));Hr=V.conj().T@(H@V)
        predictions[name]=expm_multiply(-1j*Hr,c,start=0.,stop=2.5,num=51,traceA=-1j*np.trace(Hr))@V.T
    np.savez_compressed(out/f'L{L}_frozen_predictions.npz',times=np.arange(51)/20,**predictions)
    frozen={'dimension':dimension,'common_dimension':base.shape[1],'extra_rank':q,'Hamiltonian_added_block_ranks':steps,
      'singular_values':spectra,'max_initial_error':max(initial_errors),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/f'L{L}_frozen_spaces.npz',out/f'L{L}_frozen_predictions.npz')}}
    (out/f'L{L}_frozen.json').write_text(json.dumps(frozen,indent=2))
    exact=np.load(root/'nonlinear_channels_01'/f'L{L}_eta1_exact.npz')['states']@Q
    counts=np.rint(np.asarray(Q.power(2).T@model.m).ravel()).astype(int);models=[]
    for name,states in predictions.items():
        rows=[]
        for t,state,ref in zip(np.arange(51)/20,states,exact):
            p=np.bincount(counts,weights=abs(state)**2,minlength=model.r+1);p0=np.bincount(counts,weights=abs(ref)**2,minlength=model.r+1)
            I=[2*np.vdot(state[counts==m],(H@state)[counts==m]).imag for m in range(model.r+1)]
            I0=[2*np.vdot(ref[counts==m],(H@ref)[counts==m]).imag for m in range(model.r+1)]
            rows.append({'time':float(t),'fidelity':float(abs(np.vdot(ref,state))**2),'count_TV':float(.5*np.sum(abs(p-p0))),
              'current_L1_error':float(np.sum(abs(np.array(I)-I0)))})
        models.append({'name':name,'dimension':dimension,'rows':rows})
    result={'L':L,'frozen':frozen,'models':models};(out/f'L{L}_evaluation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'L':L,'time2':[{'name':r['name'],'dimension':dimension,**r['rows'][40]} for r in models]}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps({'classification':'Post-pilot equal-rank control on previously inspected conditions. No unseen physical forecasts.',
      'rank':'Match the original matched-16-profile dimension, retaining the same source-q2 basis and top singular directions of equally normalized residual Gaussian-sector columns.',
      'ordinary_control':'Block Hamiltonian response from the same entire source-q2 basis, orthogonalized until the same dimension. No exact training or future state used in selection.',
      'initial':'Identical exact Fermi sea.'},indent=2))
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,(L,args.output)) for L in (14,18)]):f.result()
