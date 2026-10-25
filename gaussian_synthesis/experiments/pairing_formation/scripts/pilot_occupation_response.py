#!/usr/bin/env python3
"""Physical Gaussian occupation coordinates in previously untrained response sectors.

All cases use the same known t=1.35 state. A common 32-vector ordinary
Hamiltonian background evolves that state; competing response spaces evolve
the identical normalized O psi. Only the evaluation stage evolves full
response references. This is a selection pilot, not a new-parameter holdout.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,time,argparse,hashlib,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.linalg import eigh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.replica2_gpu import Replica2GPU
from pairing_formation.occupation_symbols import occupation_moments,orthogonal_span
from pairing_formation.coherent_response import ordered_krylov
from pairing_formation.gaussian_bins import angular_invariant_basis


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(task):
    out,method,index,device=task;begin=time.monotonic();data=np.load(out/'training.npz')
    model,_=angular_model(14);replica=Replica2GPU(model,device=f'cuda:{device}')
    u=data[method][index];moments=[];wards=[]
    for site in range(model.L):
        value,check,_=occupation_moments(replica,u,model.occ[:,site])
        moments.append(value);wards.append(check['ward_residual'])
    moments=np.array(moments);C=moments[0,0]
    centered=np.stack((moments[:,1]-.5*moments[:,0],
        moments[:,2]-moments[:,1]+.25*moments[:,0]),axis=1)
    path=out/f'{method}_{index}.npz';assert not path.exists()
    np.savez_compressed(path,moments=moments,centered=centered,C=C)
    row={'method':method,'index':index,'time':float(data['times'][index]),'device':device,
         'seconds':time.monotonic()-begin,'max_ward_residual':max(wards),
         'synthesis_residual':float(np.linalg.norm(C-data['unoptimized'][index])),
         'file':path.name,'sha256':digest(path)}
    print(json.dumps(row),flush=True);return row


def train(out):
    out.mkdir(parents=True,exist_ok=False);source=out.parent/'temporal_cubic_01';old=np.load(source/'early_states.npz')
    indices=[0,2,4,7];canonical=[];paths=[source/'early_states.npz']
    for i in indices:
        path=source/f'synthesis_{i}.npz';paths.append(path);canonical.append(np.load(path)['witness'])
    target=old['states'][indices];target=target/np.linalg.norm(target,axis=1)[:,None]
    np.savez_compressed(out/'training.npz',times=old['times'][indices],canonical=canonical,unoptimized=target)
    protocol={'classification':'New response-observable selection pilot at previously studied L14 G4 eta1; no new physical parameter holdouts.',
        'training_times':old['times'][indices].tolist(),'training_sha256':digest(out/'training.npz'),
        'input_hashes':{str(p):digest(p) for p in paths},
        'coordinate':'Full-Fock Gaussian single-physical-mode p_i=<n_i>. Use centered first and second moments; spin-up/down moments agree after coherent contraction for these paired targets.',
        'response':'For q=1,2,3 and particle/hole exchange parity +/-: O=sum_j sqrt(2/r) cos(2 pi q j/r) [n_upper,j-parity*n_lower,j]. Here each n is one physical spin occupation (equals pair occupation on the target sector).',
        'initial':'Known exact psi(1.35) for the background; every response model starts from the same O psi / ||O psi||.',
        'background':'Shared 32-dimensional ordinary H-Krylov space. All response methods use this identical background for C(tau)=<psi(tau)|O|exp(-i H tau) O psi>. Chi=2 Im C is the linear density response to exp(-i alpha O).',
        'selection':'Endpoint moments or all four early-time moments. All columns fixed before any full response propagation. Ordinary time-history plus H responses and equal-dimension H Krylov controls.',
        'primary_elapsed_time':.65,'final_elapsed_time':1.15,'step':.025,
        'gate':'At a given q/parity and moment family, canonical must improve primary response-state count TV, chi RMS over [0,.65], and Re C RMS over [0,.65] over both the same-family unoptimized and same-dimension H-Krylov controls by >1e-9. History also must beat matched ordinary history. Passing requires physical/numerical followup before genuinely new conditions.',
        'deletion':'Remove all occupation-moment directions while keeping the identical response initial state. This tests a representation intervention, not a physical removal of particles.',
        'scope':'Physical atom-coordinate coherent moments, not iid canonical draws or low-cost Gaussian propagation. All contraction and response work still uses an exponential paired state space.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    tasks=[(out,method,index,4*j+index) for j,method in enumerate(('canonical','unoptimized')) for index in range(4)]
    rows=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(worker,t) for t in tasks]):rows.append(f.result())
    (out/'moment_jobs.json').write_text(json.dumps(rows,indent=2));assemble(out)


def append_space(base,columns):
    residual=columns-base@(base.conj().T@columns)
    V=orthogonal_span(residual)
    # Reorthogonalize after removing almost-dependent histories.
    V=orthogonal_span(V-base@(base.conj().T@V))
    return np.column_stack((base,V))


def assemble(out):
    data=np.load(out/'training.npz');protocol=json.loads((out/'protocol.json').read_text())
    assert digest(out/'training.npz')==protocol['training_sha256']
    for row in json.loads((out/'moment_jobs.json').read_text()):assert digest(out/row['file'])==row['sha256']
    model,_=angular_model(14);H=model.H;target=data['unoptimized'][-1]
    Q,permutations=angular_invariant_basis(model)
    background=np.column_stack((target,ordered_krylov(H,target,target[:,None],31)))
    arrays={'background':background,'target':target};cases=[];geometry=[]
    moments={method:np.array([np.load(out/f'{method}_{i}.npz')['centered'] for i in range(4)]) for method in ('canonical','unoptimized')}
    for q in (1,2,3):
        f=np.sqrt(2/model.r)*np.cos(2*np.pi*q*np.arange(model.r)/model.r)
        for parity in (1,-1):
            case=f'q{q}_'+('plus' if parity==1 else 'minus')
            weights=np.r_[-parity*f,f];O=model.occ@weights
            source=O*target;strength=np.linalg.norm(source);initial=source/strength;base=initial[:,None]
            assert abs(np.vdot(target,source))<2e-13
            arrays[case+'_O']=O;arrays[case+'_initial']=initial
            spaces={'deletion':base}
            for method,values in moments.items():
                first=np.einsum('i,tij->tj',weights,values[:,:,0])
                second=np.einsum('i,tij->tj',np.r_[parity*f,f],values[:,:,1])
                spaces[method+'_endpoint']=append_space(base,np.column_stack((first[-1],second[-1])))
                spaces[method+'_history']=append_space(base,np.column_stack((first.T,second.T)))
            ordinary=(data['unoptimized']*O).T
            candidates=np.column_stack((ordinary,H@ordinary,H@(H@source)))
            spaces['ordinary_history']=append_space(base,candidates)
            dimensions=sorted(set(V.shape[1] for name,V in spaces.items() if name!='deletion'))
            for dimension in dimensions:
                spaces[f'Hamiltonian_d{dimension}']=np.column_stack((base,ordered_krylov(H,initial,base,dimension-1)))
            for name,V in spaces.items():
                gram=float(np.linalg.norm(V.conj().T@V-np.eye(V.shape[1])))
                assert gram<1e-9,(case,name,gram)
                # All response spaces should be outside the old invariant sector
                # and have the declared reflection and band-exchange parity.
                invariant=float(np.linalg.norm(Q.T@V))
                reflection=float(np.linalg.norm(V[permutations[1]]-V))
                exchange=float(np.linalg.norm(V[permutations[2]]-parity*V))
                assert max(invariant,reflection,exchange)<2e-7,(case,name,invariant,reflection,exchange)
                key=case+'__'+name;arrays[key]=V
                cases.append({'case':case,'q':q,'parity':parity,'method':name,'key':key,'dimension':V.shape[1],
                              'source_variance':float(strength**2),'Gram_error':gram})
            for family in ('endpoint','history'):
                A=spaces['canonical_'+family][:,1:];B=spaces['unoptimized_'+family][:,1:]
                geometry.append({'case':case,'family':family,'principal_cosines':np.linalg.svd(A.conj().T@B,compute_uv=False).tolist()})
    np.savez_compressed(out/'frozen_spaces.npz',**arrays)
    protocol['frozen_sha256']=digest(out/'frozen_spaces.npz');protocol['models']=cases
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'geometry.json').write_text(json.dumps(geometry,indent=2))
    print(json.dumps({'frozen_models':len(cases),'dimensions':sorted(set(r['dimension'] for r in cases)),'background_dimension':background.shape[1]}),flush=True)


def propagate(H,V,initial,times):
    Hr=V.conj().T@(H@V);energy,X=eigh((Hr+Hr.conj().T)/2)
    coefficients=X.conj().T@(V.conj().T@initial)
    states=(np.exp(-1j*times[:,None]*energy)*coefficients)@(V@X).T
    return states,float(np.linalg.norm(states[0]-initial))


def predict(out):
    protocol=json.loads((out/'protocol.json').read_text());assert digest(out/'frozen_spaces.npz')==protocol['frozen_sha256']
    data=np.load(out/'frozen_spaces.npz');model,_=angular_model(14)
    times=np.arange(round(protocol['final_elapsed_time']/protocol['step'])+1)*protocol['step']
    background,error=propagate(model.H,data['background'],data['target'],times)
    assert error<2e-12;path=out/'background_prediction.npz';assert not path.exists()
    np.savez_compressed(path,times=times,states=background)
    records=[]
    for row in protocol['models']:
        V=data[row['key']];initial=data[row['case']+'_initial'];O=data[row['case']+'_O']
        states,error=propagate(model.H,V,initial,times);assert error<2e-10
        correlation=np.einsum('ti,i,ti->t',background.conj(),O,states)*np.sqrt(row['source_variance'])
        path=out/(row['key']+'.npz');assert not path.exists()
        np.savez_compressed(path,times=times,states=states,correlation=correlation)
        record={**row,'file':path.name,'sha256':digest(path),'initial_error':error,
                'norm_error':float(np.max(abs(np.sum(abs(states)**2,axis=1)-1)))}
        records.append(record)
    (out/'frozen_predictions.json').write_text(json.dumps({'background_sha256':digest(out/'background_prediction.npz'),'models':records},indent=2))
    print(json.dumps({'predictions':len(records),'max_initial_error':max(r['initial_error'] for r in records)}),flush=True)


def evaluate(out):
    protocol=json.loads((out/'protocol.json').read_text());frozen=json.loads((out/'frozen_predictions.json').read_text())
    data=np.load(out/'frozen_spaces.npz');model,_=angular_model(14)
    saved_background=np.load(out/'background_prediction.npz');times=saved_background['times']
    assert digest(out/'background_prediction.npz')==frozen['background_sha256']
    for record in frozen['models']:assert digest(out/record['file'])==record['sha256']
    background=model.evolve(times,initial=data['target']);reference={};correlations={};results=[]
    for case in sorted(set(r['case'] for r in frozen['models'])):
        initial=data[case+'_initial'];O=data[case+'_O'];strength=np.linalg.norm(O*data['target'])
        exact=model.evolve(times,initial=initial);reference[case]=exact
        correlations[case]=np.einsum('ti,i,ti->t',background.conj(),O,exact)*strength
    np.savez_compressed(out/'exact_references.npz',times=times,background=background,**reference,
                        **{case+'_correlation':c for case,c in correlations.items()})
    mask=times<=protocol['primary_elapsed_time']+1e-12;primary_index=np.flatnonzero(mask)[-1]
    for record in frozen['models']:
        saved=np.load(out/record['file']);exact=reference[record['case']];C=correlations[record['case']]
        states=saved['states'];rows=[]
        for t,v,w,c,cr in zip(times,states,exact,saved['correlation'],C):
            p=np.bincount(model.m,weights=abs(v)**2,minlength=model.r+1)
            p0=np.bincount(model.m,weights=abs(w)**2,minlength=model.r+1)
            rows.append({'elapsed':float(t),'fidelity':float(abs(np.vdot(w,v))**2),
                         'count_TV':float(.5*np.sum(abs(p-p0))),'chi':float(2*c.imag),'chi_exact':float(2*cr.imag),
                         'real_correlation':float(c.real),'real_correlation_exact':float(cr.real)})
        delta=saved['correlation'][mask]-C[mask]
        primary={**rows[primary_index],'chi_RMS_error':float(np.sqrt(np.mean((2*delta.imag)**2))),
                 'real_correlation_RMS_error':float(np.sqrt(np.mean(delta.real**2))),
                 'complex_correlation_relative_RMS':float(np.linalg.norm(delta)/np.linalg.norm(C[mask]))}
        results.append({**record,'primary':primary,'rows':rows})
    lookup={(r['case'],r['method']):r for r in results};gates=[]
    for r in results:
        if not r['method'].startswith('canonical'):continue
        controls=[r['method'].replace('canonical','unoptimized'),f"Hamiltonian_d{r['dimension']}"]
        if r['method'].endswith('history'):controls.append('ordinary_history')
        passing=all(all(r['primary'][k]<lookup[r['case'],c]['primary'][k]-1e-9
                   for k in ('count_TV','chi_RMS_error','real_correlation_RMS_error')) for c in controls)
        gates.append({'case':r['case'],'method':r['method'],'controls':controls,'passes':passing})
    checks={'max_background_state_error':float(np.max(np.linalg.norm(background-saved_background['states'],axis=1))),
            'max_exact_norm_error':float(max(np.max(abs(np.sum(abs(v)**2,axis=1)-1)) for v in reference.values())),
            'max_initial_state_error':max(r['initial_error'] for r in results),
            'max_prediction_norm_error':max(r['norm_error'] for r in results)}
    (out/'evaluation.json').write_text(json.dumps({'classification':protocol['classification'],'checks':checks,'models':results,'gate':gates},indent=2))
    print(json.dumps({'checks':checks,'gate':gates,'primary':[{'case':r['case'],'method':r['method'],**r['primary']} for r in results]},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['train','assemble','predict','evaluate'],required=True);a=p.parse_args();globals()[a.stage](a.output)
