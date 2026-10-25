#!/usr/bin/env python3
"""Training-only coherent phase partitions, then identical-state controls."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.dark_pairs import PairDarkSectors
from pairing_formation.replica2_gpu import Replica2GPU
from pairing_formation.coherent_partition import coherent_parts
from pairing_formation.rotation_geometry import orthogonal_columns


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def augment(base,candidates,rank):
    residual=candidates-base@(base.conj().T@candidates)
    # Every candidate has a fixed physical definition. Normalize each to avoid
    # selecting exclusively by arbitrary label scaling across m sectors.
    norm=np.linalg.norm(residual,axis=0)
    selected=norm>1e-10
    if sum(selected)<rank:return None,None
    U,s,_=np.linalg.svd(residual[:,selected]/norm[selected],full_matrices=False)
    if s[rank-1]<1e-8*s[0]:return None,None
    return np.column_stack((base,U[:,:rank])),s


def train(out,source):
    out.mkdir(parents=True,exist_ok=False)
    data=np.load(source/'training.npz');model,forms=angular_model(14);Q=data['Q']
    psi=data['target'];u=data['witness'];replica=Replica2GPU(model);dark=PairDarkSectors(model,forms)
    q2=np.load(out.parent/'rotation_geometry_01/L14_frozen_spaces.npz')['source_q2']
    counts=np.rint((Q*Q).T@model.m).astype(int)
    one=np.eye(Q.shape[1])[:,counts==1]
    base,_=orthogonal_columns(np.column_stack((q2,one,Q.T@psi,Q.T@data['nonlinear_reference'])),1e-11)
    candidates={};metadata={};saved={'base':base,'Q':Q,'target':psi}
    for label,w in (('canonical',u),('unoptimized_cubic',psi)):
        A=[];B=[];norms=[]
        projected=dark.apply(w,'either')
        for m in (2,3):
            v=projected*(model.m==m)
            parts=coherent_parts(replica,w,v)
            A.append(Q.T@parts['complex_ratio']);B.append(Q.T@parts['conjugate_ratio'])
            norms.append({'m':m,'input_norm':float(np.linalg.norm(v)),
                          'A_norm':float(np.linalg.norm(A[-1])),'B_norm':float(np.linalg.norm(B[-1]))})
        candidates[label+'_complex']=np.column_stack(A)
        candidates[label+'_phase_pair']=np.column_stack(A+B)
        metadata[label]={'sectors':norms,
            'partition_sum_error':float(np.linalg.norm(coherent_parts(replica,w,projected)['complex_ratio']+
                coherent_parts(replica,w,w-projected)['complex_ratio']-replica.evaluate(w)[1]))}
    Dpsi=dark.apply(psi,'either')
    candidates['physical_dark']=Q.T@np.column_stack([Dpsi*(model.m==m) for m in (2,3)])
    m=model.m;physical=np.column_stack((Dpsi*(m==2),Dpsi*(m==3),m*psi,m*m*psi))
    candidates['physical_dark_count']=Q.T@physical
    H=Q.T@(model.H@Q);target=Q.T@psi
    # Two strong conventional controls: ordered Krylov from the training state
    # and top singular directions of one Hamiltonian response of the shared base.
    v=target.copy();ordered=[]
    for _ in range(4):
        v=H@v
        residual=v-base@(base.conj().T@v)
        for old in ordered:residual-=old*np.vdot(old,residual)
        if np.linalg.norm(residual)>1e-10:ordered.append(residual/np.linalg.norm(residual))
    candidates['Hamiltonian_block']=H@base
    spaces={'shared':base};details={}
    for name,A in candidates.items():
        saved[name+'_candidates']=A
        for rank in (1,2,4):
            if rank>A.shape[1]:continue
            V,s=augment(base,A,rank)
            if V is not None:
                spaces[f'{name}_q{rank}']=V;details[f'{name}_q{rank}']={'singular_values':s.tolist()}
    for rank in (1,2,4):
        if len(ordered)>=rank:spaces[f'Hamiltonian_Krylov_q{rank}']=np.column_stack((base,*ordered[:rank]))
    spaces['exact_symmetry']=np.eye(Q.shape[1])
    np.savez_compressed(out/'frozen_spaces.npz',**spaces,**{k:v for k,v in saved.items() if k!='base'})
    geometry={}
    for rank in (1,2):
        A=spaces[f'canonical_complex_q{rank}'][:,base.shape[1]:]
        for method in ('unoptimized_cubic_complex','physical_dark'):
            B=spaces[f'{method}_q{rank}'][:,base.shape[1]:]
            geometry[f'q{rank}_vs_{method}']=np.linalg.svd(A.conj().T@B,compute_uv=False).tolist()
    protocol={'classification':'Retrospective L14 t1 training and known-condition future-time pilot; not unseen physical holdouts.',
        'source':str(source),'training_sha256':digest(source/'training.npz'),
        'frozen_sha256':digest(out/'frozen_spaces.npz'),'base_dimension':base.shape[1],
        'common':'Full source-q2 space, entire symmetry-allowed m1 sector, exact training psi(t1), and C2(psi(t1)).',
        'actual_coherent_integrals':'A_u(P_m D u), B_u(P_m D u) obtained exactly as polynomial Cartan contractions via complex derivatives; full Gaussian dictionary retained.',
        'labels':'Complex overlap ratios and their conjugates. Not positive subset labels, selected Gaussian atoms or iid measure samples.',
        'primary_time':2.,'initial':'Exact Fermi sea at t0 or exact training psi at t1; every model includes both.',
        'gate':'Require simultaneous improvement in count TV, current and higher-transfer dark population versus both strong Hamiltonian response controls before new physical holdouts.',
        'dimensions':{k:V.shape[1] for k,V in spaces.items()},'part_diagnostics':metadata,'geometry':geometry,'details':details}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps({'spaces':len(spaces),'base_dimension':base.shape[1],'geometry':geometry}),flush=True)


def predict(out):
    protocol=json.loads((out/'protocol.json').read_text());assert digest(out/'frozen_spaces.npz')==protocol['frozen_sha256']
    data=np.load(out/'frozen_spaces.npz');model,_=angular_model(14);Q=data['Q'];H=Q.T@(model.H@Q);records=[]
    for name,dimension in protocol['dimensions'].items():
        V=data[name];Hr=V.conj().T@H@V
        for start in ('Fermi','restart'):
            time0=0. if start=='Fermi' else 1.
            initial=Q.T@(model.initial if start=='Fermi' else data['target'])
            c=V.conj().T@initial;error=float(np.linalg.norm(V@c-initial));assert error<3e-12
            times=np.arange(round(time0*20),51)/20
            states=expm_multiply(-1j*Hr,c,start=0.,stop=times[-1]-time0,num=len(times),traceA=-1j*np.trace(Hr))@V.T
            path=out/f'{name}_{start}.npz';assert not path.exists()
            np.savez_compressed(path,times=times,states=states)
            records.append({'name':name,'dimension':dimension,'start':start,'file':path.name,
                            'sha256':digest(path),'initial_error':error,
                            'norm_error':float(np.max(abs(np.sum(abs(states)**2,axis=1)-1)))})
    (out/'frozen_predictions.json').write_text(json.dumps(records,indent=2));print(json.dumps({'predictions':len(records)}),flush=True)


def evaluate(out):
    records=json.loads((out/'frozen_predictions.json').read_text());data=np.load(out/'frozen_spaces.npz')
    model,forms=angular_model(14);dark=PairDarkSectors(model,forms);Q=data['Q'];H=Q.T@(model.H@Q)
    exact=np.load(out.parent/'nonlinear_channels_01/L14_eta1_exact.npz')['states'];refs=exact@Q
    counts=np.rint((Q*Q).T@model.m).astype(int);results=[]
    truth=[dark.evaluate(s)['multiple_transfer_either_dark'] for s in exact]
    for record in records:
        path=out/record['file'];assert digest(path)==record['sha256'];prediction=np.load(path);rows=[]
        for t,s in zip(prediction['times'],prediction['states']):
            index=round(float(t)*20);ref=refs[index]
            p=np.bincount(counts,weights=abs(s)**2,minlength=model.r+1);p0=np.bincount(counts,weights=abs(ref)**2,minlength=model.r+1)
            x,y=H@s,H@ref
            I=np.array([2*np.vdot(s[counts==m],x[counts==m]).imag for m in range(model.r+1)])
            I0=np.array([2*np.vdot(ref[counts==m],y[counts==m]).imag for m in range(model.r+1)])
            d=dark.evaluate(Q@s)['multiple_transfer_either_dark']
            rows.append({'time':float(t),'fidelity':float(abs(np.vdot(ref,s))**2),'count_TV':float(.5*np.sum(abs(p-p0))),
                         'current_L1_error':float(np.sum(abs(I-I0))),'multiple_dark_probability':d,'multiple_dark_error':d-truth[index]})
        results.append({**record,'rows':rows})
    (out/'evaluation.json').write_text(json.dumps({'classification':'Known-condition pilot; not a fresh physical forecast.','models':results},indent=2))
    print(json.dumps([{**{k:r[k] for k in ('name','dimension','start')},**next(x for x in r['rows'] if abs(x['time']-2)<1e-12)} for r in results],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train','predict','evaluate'],required=True)
    p.add_argument('--source',type=Path,default=Path('results/dark_gaussian_training_01'));p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.stage=='train':train(args.output,args.source)
    elif args.stage=='predict':predict(args.output)
    else:evaluate(args.output)
