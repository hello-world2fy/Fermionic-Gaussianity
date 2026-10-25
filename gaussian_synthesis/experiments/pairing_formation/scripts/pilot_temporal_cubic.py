#!/usr/bin/env python3
"""Frozen temporal full-Gaussian cubic manifolds with ordinary controls."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,hashlib,time,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import expm_multiply,LinearOperator,cg
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.replica2_gpu import Replica2GPU
from pairing_formation.coherent_response import ordered_krylov
from pairing_formation.cubic_manifold import CubicManifold
from pairing_formation.dark_pairs import PairDarkSectors

TIMES=np.arange(6,28,3)/20


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def basis(core,columns,dimension=10):
    V=np.linalg.qr(core)[0]
    residual=columns-V@(V.conj().T@columns)
    U,s,_=np.linalg.svd(residual,full_matrices=False)
    needed=dimension-V.shape[1]
    if s[needed-1]<1e-11:raise RuntimeError(f'Unresolved history rank: {s}')
    answer=np.column_stack((V,U[:,:needed]))
    assert np.linalg.norm(answer.conj().T@answer-np.eye(dimension))<2e-9
    return answer,s


def synthesis(task):
    index,out=task;start=time.monotonic();data=np.load(out/'early_states.npz');Q=data['Q'];target=data['states'][index]
    model,_=angular_model(14);replica=Replica2GPU(model,f'cuda:{index}')
    u,report=replica.solve(target,Q);n=Q.shape[1];polish=[]
    for step in range(3):
        residual=replica.evaluate(u)[1]-target;norm=float(np.linalg.norm(residual));polish.append(norm)
        if norm<2e-13:break
        def action(x):
            v=Q@(x[:n]+1j*x[n:]);z=Q.T@replica.derivative(u,v);return np.r_[z.real,z.imag]
        op=LinearOperator((2*n,2*n),matvec=action,dtype=float);rhs=-Q.T@residual
        solution,status=cg(op,np.r_[rhs.real,rhs.imag],rtol=2e-11,atol=1e-15,maxiter=100)
        assert status==0
        u+=Q@(solution[:n]+1j*solution[n:])
    residual=float(np.linalg.norm(replica.evaluate(u)[1]-target));assert residual<2e-12
    path=out/f'synthesis_{index}.npz';np.savez_compressed(path,witness=u,target=target)
    report.update({'time':float(TIMES[index]),'index':index,'polished_residual':residual,'polish_history':polish,
        'direct_witness_ray_fidelity':float(abs(np.vdot(target,u))**2/np.linalg.norm(u)**2),
        'seconds':time.monotonic()-start,'file':path.name,'sha256':digest(path)})
    (out/f'synthesis_{index}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('index','time','polished_residual','direct_witness_ray_fidelity','seconds')}),flush=True)
    return report


def train(out):
    out.mkdir(parents=True,exist_ok=False);model,_=angular_model(14)
    Q=load_npz(out.parent/'rotation_geometry_01/L14_symmetry.npz').toarray();H=Q.T@(model.H@Q);F=Q.T@model.initial
    initial=expm_multiply(-1j*TIMES[0]*H,F,traceA=-1j*TIMES[0]*np.trace(H))
    states=expm_multiply(-1j*H,initial,start=0.,stop=TIMES[-1]-TIMES[0],num=len(TIMES),traceA=-1j*np.trace(H))@Q.T
    np.savez_compressed(out/'early_states.npz',times=TIMES,states=states,Q=Q,H=H,F=F)
    (out/'protocol.json').write_text(json.dumps({'classification':'Retrospective angular L14 G4 eta1 temporal pilot; no new physical holdout claim.',
        'training_times':TIMES.tolist(),'prediction_primary_time':2.,'prediction_final_time':2.5,
        'training_source':'Fresh Hamiltonian propagation only through t1.35. No later reference files are read in training.',
        'map':'Actual full-Gaussian k2 Cartan synthesis restricted to a frozen ten-complex-dimensional witness-coordinate space; not a fitted polynomial.',
        'common_information':'All cubic manifolds contain F and the same final canonical witness. Canonical history adds earlier witnesses plus final physical state; direct history adds the same earlier physical states. This is an endpoint-canonical-plus-direct-history control, not a wholly unoptimized method.',
        'dynamics':'Real Hamiltonian variational principle with normalized projective tangent and symplectic pullback, because C2 is nonholomorphic. Degenerate symplectic tangents stop rather than being silently regularized.',
        'same_initial_states':['Exact Fermi sea t0','Exact final training state t1.35'],
        'strong_controls':['direct-history cubic manifold','Hamiltonian-seeded cubic manifold','ordinary history-plus-response linear POD','ordinary Hamiltonian linear Krylov'],
        'deletion':'Two-coordinate common cubic manifold contains the same F and final witness, deleting all additional history directions.',
        'gate':'Before new physical holdouts, require canonical-specific simultaneous count TV, sector current L1 and dark-population improvement over direct-history cubic and both ordinary linear controls at the primary time, in both starts, with stable numerical dynamics and a physically interpretable added structure.',
        'early_states_sha256':digest(out/'early_states.npz')},indent=2))
    reports=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(synthesis,(i,out)) for i in range(8)]):reports.append(f.result())
    (out/'synthesis_summary.json').write_text(json.dumps(sorted(reports,key=lambda x:x['index']),indent=2))
    witnesses=np.column_stack([np.load(out/f'synthesis_{i}.npz')['witness'] for i in range(8)])
    psi=Q.T@states[-1];u=Q.T@witnesses[:,-1];history=Q.T@states.T;whistory=Q.T@witnesses
    core=np.column_stack((F,u));arrays={'Q':Q,'H':H,'F':F,'target':psi,'endpoint_witness':u};metadata={}
    arrays['canonical'],s=basis(core,np.column_stack((whistory,psi)));metadata['canonical']=s.tolist()
    arrays['direct'],s=basis(core,history);metadata['direct']=s.tolist()
    K=ordered_krylov(H,psi,np.linalg.qr(core)[0],8)
    arrays['Hamiltonian']=np.column_stack((np.linalg.qr(core)[0],K));metadata['Hamiltonian']={'directions':K.shape[1]}
    arrays['common']=np.linalg.qr(core)[0]
    physical_core=np.column_stack((F,psi));V0=np.linalg.qr(physical_core)[0]
    arrays['linear_history'],s=basis(physical_core,np.column_stack((history,H@history)));metadata['linear_history']=s.tolist()
    arrays['linear_Hamiltonian']=np.column_stack((V0,ordered_krylov(H,psi,V0,8)))
    arrays['linear_common']=V0;arrays['exact_symmetry']=np.eye(len(F))
    np.savez_compressed(out/'frozen_bases.npz',**arrays)
    (out/'basis_geometry.json').write_text(json.dumps({'history_singular_values':metadata,
        'dimensions':{k:arrays[k].shape[1] for k in ['canonical','direct','Hamiltonian','common','linear_history','linear_Hamiltonian','linear_common','exact_symmetry']},
        'canonical_direct_cosines':np.linalg.svd(arrays['canonical'].conj().T@arrays['direct'],compute_uv=False).tolist(),
        'frozen_bases_sha256':digest(out/'frozen_bases.npz')},indent=2))


def tensor_worker(task):
    out,method,chunk,chunks,device=task;start=time.monotonic();data=np.load(out/'frozen_bases.npz')
    Q=data['Q'];W=data[method];model,_=angular_model(14);replica=Replica2GPU(model,f'cuda:{device}');torch=replica.torch
    U=torch.as_tensor(Q@W,device=replica.device,dtype=torch.complex128)
    q=torch.as_tensor(Q,device=replica.device,dtype=torch.complex128)
    pairs=[(i,j) for i in range(W.shape[1]) for j in range(i,W.shape[1])][chunk::chunks]
    values=[]
    for i,j in pairs:
        answer=torch.zeros((model.dim,W.shape[1]),device=replica.device,dtype=torch.complex128)
        for a,b,K in replica.blocks:
            action=((U[a,i]*U[b,j]+U[a,j]*U[b,i])/2)@K.T
            contribution=action[:,:,None]*U[b].conj()
            answer.index_add_(0,a.flatten(),contribution.reshape(-1,W.shape[1]))
        values.append((q.conj().T@answer).cpu().numpy())
    path=out/f'{method}_tensor_chunk{chunk}.npz';assert not path.exists()
    np.savez_compressed(path,pairs=np.array(pairs),values=np.array(values))
    record={'method':method,'chunk':chunk,'chunks':chunks,'device':device,'seconds':time.monotonic()-start,
        'file':path.name,'sha256':digest(path)}
    print(json.dumps(record),flush=True);return record


def tensors(out):
    tasks=[]
    for method,chunks,offset in [('canonical',3,0),('direct',3,3),('Hamiltonian',2,6)]:
        tasks +=[(out,method,chunk,chunks,offset+chunk) for chunk in range(chunks)]
    reports=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(tensor_worker,t) for t in tasks]):reports.append(f.result())
    (out/'tensor_jobs.json').write_text(json.dumps(reports,indent=2))
    data=np.load(out/'frozen_bases.npz');arrays={};checks=[];model,_=angular_model(14);replica=Replica2GPU(model)
    rng=np.random.default_rng(1541)
    for method,chunks in [('canonical',3),('direct',3),('Hamiltonian',2)]:
        W=data[method];k=W.shape[1];T=np.empty((W.shape[0],k,k,k),complex)
        for chunk in range(chunks):
            source=np.load(out/f'{method}_tensor_chunk{chunk}.npz')
            for (i,j),value in zip(source['pairs'],source['values']):T[:,i,j,:]=value;T[:,j,i,:]=value
        manifold=CubicManifold(T,data['H'])
        for _ in range(3):
            c=rng.normal(size=k)+1j*rng.normal(size=k);c/=np.linalg.norm(c)
            full=replica.evaluate(data['Q']@W@c)[1];error=float(np.linalg.norm(data['Q']@manifold.raw(c)-full))
            assert error<2e-12
            checks.append({'method':method,'random_reconstruction_error':error})
        for name,target in [('Fermi',data['F']),('restart',data['endpoint_witness'])]:
            c=W.conj().T@target
            expected=data['F'] if name=='Fermi' else data['target']
            error=float(np.linalg.norm(manifold.state(c)-expected))
            assert error<3e-11,(method,name,error)
            checks.append({'method':method,'start':name,'initial_state_error':error})
        arrays[method]=T
    arrays['common']=arrays['canonical'][:,:2,:2,:2]
    np.savez_compressed(out/'frozen_tensors.npz',**arrays)
    (out/'tensor_checks.json').write_text(json.dumps({'checks':checks,'frozen_tensors_sha256':digest(out/'frozen_tensors.npz')},indent=2))


def propagate(task):
    out,method,start,refine=task;begin=time.monotonic();data=np.load(out/'frozen_bases.npz')
    times=np.arange(0 if start=='Fermi' else 27,51)/20;H=data['H']
    physical=data['F'] if start=='Fermi' else data['target']
    if method in ('canonical','direct','Hamiltonian','common'):
        W=data[method];w=data['F'] if start=='Fermi' else data['endpoint_witness'];initial=W.conj().T@w
        tensor=np.load(out/'frozen_tensors.npz')[method];manifold=CubicManifold(tensor,H)
        states,coordinates,report=manifold.evolve(times,initial,rtol=2e-10 if refine else 2e-8,
            atol=2e-12 if refine else 2e-10,max_step=.0125 if refine else .025)
    else:
        V=data[method];Hr=V.conj().T@H@V
        states=expm_multiply(-1j*Hr,V.conj().T@physical,start=0.,stop=times[-1]-times[0],num=len(times),traceA=-1j*np.trace(Hr))@V.T
        coordinates=np.empty((0,));energy=np.einsum('ti,ij,tj->t',states.conj(),H,states).real
        report={'max_norm_error':float(np.max(abs(np.sum(abs(states)**2,axis=1)-1))),
                'max_energy_drift':float(np.max(abs(energy-energy[0])))}
    tag=f'{method}_{start}'+('_refined' if refine else '')
    path=out/f'{tag}.npz';assert not path.exists()
    np.savez_compressed(path,times=times,states=states,coordinates=coordinates)
    report.update({'method':method,'start':start,'refined':refine,'seconds':time.monotonic()-begin,'file':path.name,
        'sha256':digest(path),'initial_state_error':float(np.linalg.norm(states[0]-physical))})
    (out/f'{tag}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True);return report


def predict(out,refine=False):
    geometry=json.loads((out/'basis_geometry.json').read_text());assert digest(out/'frozen_bases.npz')==geometry['frozen_bases_sha256']
    checks=json.loads((out/'tensor_checks.json').read_text());assert digest(out/'frozen_tensors.npz')==checks['frozen_tensors_sha256']
    methods=['canonical','direct','Hamiltonian','common'] if refine else list(geometry['dimensions'])
    jobs=[(out,method,start,refine) for method in methods for start in ('Fermi','restart')];records=[]
    with ProcessPoolExecutor(max_workers=16) as pool:
        for f in as_completed([pool.submit(propagate,t) for t in jobs]):records.append(f.result())
    path=out/('refined_predictions.json' if refine else 'frozen_predictions.json');assert not path.exists()
    path.write_text(json.dumps(records,indent=2))


def evaluate(out):
    records=json.loads((out/'frozen_predictions.json').read_text());data=np.load(out/'frozen_bases.npz');Q=data['Q'];H=data['H']
    model,forms=angular_model(14);dark=PairDarkSectors(model,forms)
    reference=np.load(out.parent/'nonlinear_channels_01/L14_eta1_exact.npz')['states']@Q
    counts=np.rint((Q*Q).T@model.m).astype(int);D=Q.T@dark.apply(Q,'either');D*=((counts>=2)[:,None]&(counts>=2)[None,:])
    results=[]
    for record in records:
        path=out/record['file'];assert digest(path)==record['sha256'];predicted=np.load(path);rows=[]
        for t,s in zip(predicted['times'],predicted['states']):
            ref=reference[round(float(t)*20)];p=np.bincount(counts,weights=abs(s)**2,minlength=model.r+1);p0=np.bincount(counts,weights=abs(ref)**2,minlength=model.r+1)
            hs,hr=H@s,H@ref
            I=np.array([2*np.vdot(s[counts==m],hs[counts==m]).imag for m in range(model.r+1)])
            I0=np.array([2*np.vdot(ref[counts==m],hr[counts==m]).imag for m in range(model.r+1)])
            d=float(np.vdot(s,D@s).real);d0=float(np.vdot(ref,D@ref).real)
            rows.append({'time':float(t),'fidelity':float(abs(np.vdot(ref,s))**2),'count_TV':float(.5*np.sum(abs(p-p0))),
                'current_L1_error':float(np.sum(abs(I-I0))),'multiple_dark_probability':d,'multiple_dark_error':d-d0})
        results.append({**record,'rows':rows})
    primary={f"{r['method']}_{r['start']}":next(x for x in r['rows'] if x['time']==2.) for r in results}
    gates={}
    for start in ('Fermi','restart'):
        row=primary[f'canonical_{start}'];controls=['direct','linear_history','linear_Hamiltonian']
        gates[start]=all(abs(row[k])<abs(primary[f'{c}_{start}'][k])-1e-8 for c in controls for k in ['count_TV','current_L1_error','multiple_dark_error'])
    (out/'evaluation.json').write_text(json.dumps({'classification':'Known-condition temporal nonlinear pilot.','models':results,'primary':primary,'gate':gates},indent=2))
    print(json.dumps({'primary':primary,'gate':gates},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['train','tensors','predict','refine','evaluate'],required=True)
    args=p.parse_args()
    if args.stage=='refine':predict(args.output,True)
    else:globals()[args.stage](args.output)
