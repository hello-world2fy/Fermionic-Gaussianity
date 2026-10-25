#!/usr/bin/env python3
"""Frozen anomalous-field and normal-exchange coherent coordinates."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,argparse,time,hashlib,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.pair_symbols import PairSymbol2
from pairing_formation.occupation_symbols import orthogonal_span
from pairing_formation.coherent_response import ordered_krylov
from pairing_formation.dark_pairs import PairDarkSectors
from scripts.pilot_occupation_response import propagate


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def selected_pairs(r):
    return [(r+d,0) for d in range(r//2+1)]+[(0,r+d) for d in range(r//2+1)]+[(r,r+d) for d in range(r//2+1)]+[(0,d) for d in range(r//2+1)]


def physical_action(model,vector,i,j):
    if i==j:return model.occ[:,i]*vector
    dst,src=model.transitions[i,j];out=np.zeros(model.dim,complex);out[dst]=vector[src];return out


def worker(task):
    out,method,index,device=task;begin=time.monotonic();data=np.load(out/'training.npz');model,_=angular_model(14)
    backend=PairSymbol2(model,device=f'cuda:{device}');backend.prepare(data[method][index]);Q=data['Q']
    labels=selected_pairs(model.r);moments={name:[] for name in ('full','anomalous','normal_exchange')};ordinary=[]
    for i,j in labels:
        parts=backend.symbol_parts(i,j)
        for name,value in parts.items():
            mean=0.
            if i==j:
                mean={'full':.25,'anomalous':1/(4*(2*backend.n-1)),'normal_exchange':.25-1/(4*(2*backend.n-1))}[name]
            moments[name].append(Q.T@(value-mean*backend.C))
        ordinary.append(Q.T@physical_action(model,data['unoptimized'][index],i,j))
    path=out/f'{method}_{index}.npz';assert not path.exists()
    np.savez_compressed(path,**{k:np.array(v) for k,v in moments.items()},ordinary=np.array(ordinary),C=Q.T@backend.C,pairs=labels)
    row={'method':method,'index':index,'time':float(data['times'][index]),'device':device,
         'seconds':time.monotonic()-begin,'pairs':len(labels),'file':path.name,'sha256':digest(path),
         'synthesis_residual':float(np.linalg.norm(backend.C-data['unoptimized'][index])),
         'split_sum_residual':float(np.linalg.norm(np.array(moments['full'])-np.array(moments['anomalous'])-np.array(moments['normal_exchange'])))}
    print(json.dumps(row),flush=True);return row


def train(out):
    out.mkdir(parents=True,exist_ok=False);source=out.parent/'temporal_cubic_01';early=np.load(source/'early_states.npz');indices=[0,2,4,7]
    canonical=[];paths=[source/'early_states.npz']
    for i in indices:
        path=source/f'synthesis_{i}.npz';canonical.append(np.load(path)['witness']);paths.append(path)
    targets=early['states'][indices];targets/=np.linalg.norm(targets,axis=1)[:,None]
    np.savez_compressed(out/'training.npz',times=early['times'][indices],Q=early['Q'],canonical=canonical,unoptimized=targets)
    protocol={'classification':'Retrospective physical-coordinate selection pilot at L14 G4 eta1. No new physical parameter holdouts.',
        'training_times':early['times'][indices].tolist(),'training_sha256':digest(out/'training.npz'),'input_hashes':{str(p):digest(p) for p in paths},
        'physical_labels':'Full-Fock Gaussian pair correlator <s_i+ s_j-> = Delta_i* Delta_j + rho_iup,jup rho_idown,jdown - rho_iup,jdown rho_idown,jup. Integrate full, anomalous-field product and normal-exchange labels separately; subtract their Haar means. These are coordinate-weighted coherent vectors, not target observable marginals.',
        'quadratic_intermediates':'Full Cartan contraction includes r-1/r+1 paired sectors and complementary two-unpaired-electron sectors. The normal-exchange physical label is derived by symbol decomposition, not identified with one intermediate block.',
        'labels':'Four intershell angular separations and their conjugates; four separations in each shell, including double occupation. Exact invariant output projection corresponds to averaging the labels under the same Gaussian symmetry group.',
        'shared_space':'All endpoint transfer-count profiles plus H psi_endpoint. Contains exactly the Fermi initial state and endpoint target, and the endpoint Hamiltonian tangent.',
        'selection':'Normalize each nonzero residual coordinate column, then take leading SVD directions. Eight anomalous, normal or full directions. Sixteen full directions; nested split adds eight residual normal directions to the eight anomalous directions. Ordinary operator-history and ordinary H Krylov controls use the same dimensions.',
        'deletion':'Nested split -> anomalous removes precisely the eight added normal-coordinate directions while preserving both initial states. Anomalous-plus-eight-H control tests whether ordinary dynamics repairs the same omission.',
        'primary_time':2.,'last_time':2.5,'step':.05,
        'gate':'Each canonical candidate must improve count TV, four-point pair-channel L2 error and multiple-transfer dark-probability error over same-family unoptimized, equal-dimension ordinary operator-history and H Krylov at t2. Split must additionally beat full16, anomalous-only deletion and anomalous-plus-H16. Both identical Fermi and endpoint starts are reported. No new physical conditions until numerical and mechanism checks pass.',
        'scope':'Float64 full-Gaussian integrals with exponential paired-space contractions; no iid canonical sampling, interval certificate or equal-compute advantage claim.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));records=[]
    tasks=[(out,name,index,4*k+index) for k,name in enumerate(('canonical','unoptimized')) for index in range(4)]
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(worker,t) for t in tasks]):records.append(f.result())
    (out/'moment_jobs.json').write_text(json.dumps(records,indent=2));assemble(out)


def modes(base,columns,number):
    residual=columns-base@(base.conj().T@columns);norm=np.linalg.norm(residual,axis=0);keep=norm>1e-12
    normalized=residual[:,keep]/norm[keep];U,s,_=np.linalg.svd(normalized,full_matrices=False)
    if len(s)<number or s[number-1]<1e-10*s[0]:raise ValueError(('Insufficient feature rank',number,s))
    added=orthogonal_span(U[:,:number]-base@(base.conj().T@U[:,:number]))
    assert added.shape[1]==number
    return np.column_stack((base,added)),s


def assemble(out):
    data=np.load(out/'training.npz');protocol=json.loads((out/'protocol.json').read_text());Q=data['Q'];model,forms=angular_model(14)
    assert digest(out/'training.npz')==protocol['training_sha256']
    for row in json.loads((out/'moment_jobs.json').read_text()):assert digest(out/row['file'])==row['sha256']
    H=Q.T@(model.H@Q);target=Q.T@data['unoptimized'][-1]
    base=orthogonal_span(np.column_stack([Q.T@(data['unoptimized'][-1]*(model.m==m)) for m in range(model.r+1)]+[H@target]))
    spaces={'shared':base};arrays={'Q':Q,'H':H,'target':target,'F':Q.T@model.initial};spectra={}
    for method in ('canonical','unoptimized'):
        features={name:np.concatenate([np.load(out/f'{method}_{i}.npz')[name] for i in range(4)]).T for name in ('full','anomalous','normal_exchange')}
        for kind in features:
            name=method+'_'+kind+'_q8';spaces[name],s=modes(base,features[kind],8);spectra[name]=s.tolist()
        name=method+'_full_q16';spaces[name],s=modes(base,features['full'],16);spectra[name]=s.tolist()
        A=spaces[method+'_anomalous_q8'];name=method+'_split_q16'
        spaces[name],s=modes(A,features['normal_exchange'],8);spectra[name]=s.tolist()
        spaces[method+'_anomalous_H_q16']=np.column_stack((A,ordered_krylov(H,target,A,8)))
    ordinary=np.column_stack([np.load(out/f'unoptimized_{i}.npz')['ordinary'].T for i in range(4)]+[(data['unoptimized']@Q).T])
    for rank in (8,16):
        spaces[f'ordinary_q{rank}'],s=modes(base,ordinary,rank);spectra[f'ordinary_q{rank}']=s.tolist()
        spaces[f'Hamiltonian_q{rank}']=np.column_stack((base,ordered_krylov(H,target,base,rank)))
    spaces['exact_symmetry']=np.eye(Q.shape[1]);geometry={}
    for kind in ('full_q8','anomalous_q8','normal_exchange_q8','full_q16','split_q16'):
        A=spaces['canonical_'+kind][:,base.shape[1]:];B=spaces['unoptimized_'+kind][:,base.shape[1]:]
        geometry[kind]=np.linalg.svd(A.conj().T@B,compute_uv=False).tolist()
    D=Q.T@PairDarkSectors(model,forms).apply(Q,'either');counts=np.rint((Q*Q).T@model.m).astype(int)
    D*=((counts>=2)[:,None]&(counts>=2)[None,:]);operators=[]
    theta=2*np.pi*np.arange(model.r)/model.r
    for q in range(4):
        kernel=np.zeros((model.L,model.L));kernel[model.r:,:model.r]=np.cos(q*(theta[:,None]-theta[None,:]))/model.r
        operators.append(Q.T@(model.pair_operator(kernel)@Q))
    arrays.update(spaces);arrays.update({'dark':D,'counts':counts,'pair_channels':np.array(operators)})
    for name,V in spaces.items():assert np.linalg.norm(V.conj().T@V-np.eye(V.shape[1]))<2e-10,name
    np.savez_compressed(out/'frozen_spaces.npz',**arrays);protocol['frozen_sha256']=digest(out/'frozen_spaces.npz')
    protocol['base_dimension']=base.shape[1];protocol['dimensions']={k:v.shape[1] for k,v in spaces.items()}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'geometry.json').write_text(json.dumps({'spectra':spectra,'canonical_vs_unoptimized':geometry},indent=2))
    print(json.dumps({'frozen_spaces':len(spaces),'base_dimension':base.shape[1],'dimensions':protocol['dimensions']}),flush=True)


def predict(out):
    protocol=json.loads((out/'protocol.json').read_text());assert digest(out/'frozen_spaces.npz')==protocol['frozen_sha256']
    data=np.load(out/'frozen_spaces.npz');H=data['H'];records=[]
    for name,dimension in protocol['dimensions'].items():
        for start in ('Fermi','restart'):
            begin=0. if start=='Fermi' else 1.35;times=np.arange(round(begin/.05),51)*.05
            initial=data['F'] if start=='Fermi' else data['target'];states,error=propagate(H,data[name],initial,times-times[0]);assert error<2e-11
            path=out/f'{name}_{start}.npz';assert not path.exists();np.savez_compressed(path,times=times,states=states)
            records.append({'method':name,'dimension':dimension,'start':start,'file':path.name,'sha256':digest(path),
                            'initial_error':error,'norm_error':float(np.max(abs(np.sum(abs(states)**2,axis=1)-1)))})
    (out/'frozen_predictions.json').write_text(json.dumps(records,indent=2));print(json.dumps({'predictions':len(records)}),flush=True)


def evaluate(out):
    protocol=json.loads((out/'protocol.json').read_text());records=json.loads((out/'frozen_predictions.json').read_text());data=np.load(out/'frozen_spaces.npz')
    for row in records:assert digest(out/row['file'])==row['sha256']
    H=data['H'];counts=data['counts'];D=data['dark'];operators=data['pair_channels']
    times=np.arange(51)*.05;exact=expm_multiply(-1j*H,data['F'],start=0.,stop=2.5,num=51,traceA=-1j*np.trace(H))
    np.savez_compressed(out/'exact_references.npz',times=times,states=exact);results=[]
    for row in records:
        saved=np.load(out/row['file']);rows=[]
        for t,v in zip(saved['times'],saved['states']):
            ref=exact[round(float(t)*20)];p=np.bincount(counts,weights=abs(v)**2,minlength=8);p0=np.bincount(counts,weights=abs(ref)**2,minlength=8)
            channels=np.einsum('i,qij,j->q',v.conj(),operators,v);channels0=np.einsum('i,qij,j->q',ref.conj(),operators,ref)
            dark=float(np.vdot(v,D@v).real);dark0=float(np.vdot(ref,D@ref).real)
            current=np.array([2*np.vdot(v[counts==m],(H@v)[counts==m]).imag for m in range(8)])
            current0=np.array([2*np.vdot(ref[counts==m],(H@ref)[counts==m]).imag for m in range(8)])
            rows.append({'time':float(t),'fidelity':float(abs(np.vdot(ref,v))**2),'count_TV':float(.5*np.sum(abs(p-p0))),
                'pair_channel_L2_error':float(np.linalg.norm(channels-channels0)),'current_L1_error':float(np.sum(abs(current-current0))),
                'multiple_dark_probability':dark,'multiple_dark_error':dark-dark0,'channels_real':channels.real.tolist(),'channels_imag':channels.imag.tolist()})
        results.append({**row,'rows':rows,'primary':next(r for r in rows if abs(r['time']-2)<1e-12)})
    lookup={(r['method'],r['start']):r['primary'] for r in results};gates=[]
    if 'baseline_evaluation' in protocol:
        path=Path(protocol['baseline_evaluation']);assert digest(path)==protocol['baseline_evaluation_sha256']
        for r in json.loads(path.read_text())['models']:lookup.setdefault((r['method'],r['start']),r['primary'])
    for row in results:
        name=row['method'];start=row['start']
        if not name.startswith('canonical_') or '_H_' in name:continue
        rank=int(name.rsplit('q',1)[1]);controls=[name.replace('canonical','unoptimized'),f'ordinary_q{rank}',f'Hamiltonian_q{rank}']
        if 'split' in name:controls+=['canonical_full_q16','canonical_anomalous_q8','canonical_anomalous_H_q16']
        passing=all(all(abs(row['primary'][k])<abs(lookup[c,start][k])-1e-9 for k in ('count_TV','pair_channel_L2_error','multiple_dark_error')) for c in controls)
        gates.append({'method':name,'start':start,'controls':controls,'passes':passing})
    (out/'evaluation.json').write_text(json.dumps({'classification':protocol['classification'],'models':results,'gate':gates},indent=2))
    print(json.dumps({'gate':gates,'primary_restart':[{'method':r['method'],**r['primary']} for r in results if r['start']=='restart']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train','assemble','predict','evaluate'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();globals()[a.stage](a.output)
