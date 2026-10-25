#!/usr/bin/env python3
"""Frozen two-transfer Gaussian symbols and physically measurable transfer noise."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,time,argparse,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.multipair_symbols import MultiPairSymbol2
from pairing_formation.multipair_noise import noise_observables
from pairing_formation.coherent_response import ordered_krylov
from scripts.pilot_pair_symbols import digest,modes
from scripts.pilot_occupation_response import propagate


def labels(r):
    forward=[((r+x,r+(x+dp)%r),(0,dh)) for x in (0,1) for dp in (1,2,3) for dh in (1,2,3)]
    return forward+[(minus,plus) for plus,minus in forward]


def action(model,v,plus,minus):
    add=sum(1<<j for j in plus);remove=sum(1<<j for j in minus)
    chosen=np.flatnonzero(((model.masks&add)==0)&((model.masks&remove)==remove))
    destination=np.array([model.index[int(model.masks[j])^add^remove] for j in chosen])
    result=np.zeros_like(v);result[destination]=v[chosen];return result


def worker(task):
    out,method,index,device=task;begin=time.monotonic();data=np.load(out/'training.npz');model,_=angular_model(14);Q=data['Q']
    backend=MultiPairSymbol2(model,f'cuda:{device}');backend.prepare(data[method][index])
    coordinates={'full':[],'cross_copy':[],'ordinary':[]};term_arrays={};last=time.monotonic()
    for number,(plus,minus) in enumerate(labels(model.r)):
        full,terms=backend.monomial_symbol(plus,minus,return_terms=True)
        cross=full-terms['(0, 0)']-terms['(8, 0)']
        coordinates['full'].append(Q.T@full);coordinates['cross_copy'].append(Q.T@cross)
        coordinates['ordinary'].append(Q.T@action(model,data['unoptimized'][index],plus,minus))
        for name,value in terms.items():term_arrays.setdefault(name,[]).append(Q.T@value)
        if time.monotonic()-last>20:
            print(json.dumps({'method':method,'index':index,'completed_labels':number+1,'seconds':time.monotonic()-begin}),flush=True);last=time.monotonic()
    path=out/f'{method}_{index}.npz';assert not path.exists()
    np.savez_compressed(path,**{k:np.array(v) for k,v in coordinates.items()},**{'sector_'+k:np.array(v) for k,v in term_arrays.items()},C=Q.T@backend.C,labels=labels(model.r))
    row={'method':method,'index':index,'device':device,'time':float(data['times'][index]),'seconds':time.monotonic()-begin,
        'labels':len(labels(model.r)),'cached_intermediate_sectors':len(backend._sector_cache),
        'synthesis_residual':float(np.linalg.norm(backend.C-data['unoptimized'][index])),
        'cross_fraction':float(np.linalg.norm(coordinates['cross_copy'])/np.linalg.norm(coordinates['full'])),
        'file':path.name,'sha256':digest(path)}
    print(json.dumps(row),flush=True);return row


def train(out):
    out.mkdir(parents=True,exist_ok=False);old=out.parent/'pair_symbol_pilot_01';prior=json.loads((old/'protocol.json').read_text())
    assert digest(old/'training.npz')==prior['training_sha256'];data=np.load(old/'training.npz')
    np.savez_compressed(out/'training.npz',**{k:data[k] for k in data.files})
    inputs=[old/'training.npz',old/'frozen_spaces.npz',out.parent/'pair_contrast_01/frozen_spaces.npz']
    protocol={'classification':'Retrospective L14 G4 eta1 two-transfer physical-coordinate pilot; no new physical parameter holdouts.',
        'training_times':data['times'].tolist(),'training_sha256':digest(out/'training.npz'),'input_hashes':{str(p):digest(p) for p in inputs},
        'labels':'36 disjoint two-pair-transfer monomials:18 forward configurations (upper-pair offset0/1,upper and lower separations1/2/3), plus adjoints. Full-Fock Gaussian eight-fermion symbols, with invariant output projection.',
        'cross_copy':'Remove the two endpoint terms105[O C(u)+B_u(O^dagger u)]/D4 from the full symbol. This is a derived coherent coproduct contribution, not a positive atom bin or the physical connected-noise cumulant. It vanishes on paired occupation-basis Slater witnesses.',
        'selection':'Leading8 or16 normalized residual training directions, shared9 endpoint-count/H-tangent space unchanged. Joint adds8 cross-copy directions to previous17-dimensional single-transfer contrast space. Same8 ordinary H directions provide a nested repair control.',
        'same_initial_states':'Exact Fermi at0 or exact endpoint at1.35. Training-interval Fermi results are retrospective reconstruction, not forecasts.',
        'observables':'Connected complex pair-transfer noise C_qp=<X_q X_p>-<X_q><X_p>, q,p0..3; eight Hermitian pair-pulse quadrature variances; counts and ordinary pair means. Var(X+Xdag)-Var(-i(X-Xdag))=4 Re C_qq.',
        'primary_time':2.,'last_time':2.5,'step':.05,
        'gate':'Each canonical selection must improve countTV, connected-transfer-noise L2 and pulse-variance L2 against matched unoptimized, ordinary two-transfer history and H directions. Joint must additionally beat same-initial single-transfer deletion and single-transfer-plus-H repair. Both identical starts retained; no new conditions before the mechanism survives.',
        'scope':'k2 synthesis; deterministic exponential paired-space contractions. No iid canonical sampling or new physics assumed.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
    tasks=[(out,method,i,4*j+i) for j,method in enumerate(('canonical','unoptimized')) for i in range(4)]
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(worker,t) for t in tasks]):rows.append(f.result())
    (out/'moment_jobs.json').write_text(json.dumps(rows,indent=2));assemble(out)


def assemble(out):
    protocol=json.loads((out/'protocol.json').read_text());old=np.load(out.parent/'pair_symbol_pilot_01/frozen_spaces.npz');contrast=np.load(out.parent/'pair_contrast_01/frozen_spaces.npz')
    assert digest(out/'training.npz')==protocol['training_sha256']
    for row in json.loads((out/'moment_jobs.json').read_text()):assert digest(out/row['file'])==row['sha256']
    base=old['shared'];H=old['H'];target=old['target'];spaces={'shared':base};spectra={}
    for method in ('canonical','unoptimized'):
        for kind in ('full','cross_copy'):
            columns=np.column_stack([np.load(out/f'{method}_{i}.npz')[kind].T for i in range(4)])
            for rank in (8,16):
                name=f'{method}_{kind}_q{rank}';spaces[name],s=modes(base,columns,rank);spectra[name]=s.tolist()
        A=contrast[method+'_contrast_q8'];name=method+'_joint_q16'
        spaces[name],s=modes(A,columns,8);spectra[name]=s.tolist()
        spaces[method+'_single_H_q16']=np.column_stack((A,ordered_krylov(H,target,A,8)))
        spaces[method+'_contrast_q8']=A
        spaces[method+'_contrast_q16']=contrast[method+'_contrast_q16']
    ordinary=np.column_stack([np.load(out/f'unoptimized_{i}.npz')['ordinary'].T for i in range(4)])
    for rank in (8,16):
        spaces[f'ordinary2_q{rank}'],s=modes(base,ordinary,rank);spectra[f'ordinary2_q{rank}']=s.tolist()
        spaces[f'Hamiltonian_q{rank}']=old[f'Hamiltonian_q{rank}']
    spaces['exact_symmetry']=old['exact_symmetry']
    arrays={k:old[k] for k in ('Q','H','F','target','counts','pair_channels','dark')};arrays.update(spaces)
    for name,V in spaces.items():assert np.linalg.norm(V.conj().T@V-np.eye(V.shape[1]))<2e-10,name
    np.savez_compressed(out/'frozen_spaces.npz',**arrays)
    protocol.update(frozen_sha256=digest(out/'frozen_spaces.npz'),dimensions={k:V.shape[1] for k,V in spaces.items()})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    geometry={}
    for kind in ('full_q8','full_q16','cross_copy_q8','cross_copy_q16','joint_q16'):
        A=spaces['canonical_'+kind][:,9:];B=spaces['unoptimized_'+kind][:,9:]
        geometry[kind]=np.linalg.svd(A.conj().T@B,compute_uv=False).tolist()
    (out/'geometry.json').write_text(json.dumps({'spectra':spectra,'canonical_vs_unoptimized':geometry},indent=2))
    print(json.dumps({'spaces':len(spaces),'dimensions':protocol['dimensions']}),flush=True)


def predict(out):
    protocol=json.loads((out/'protocol.json').read_text());assert digest(out/'frozen_spaces.npz')==protocol['frozen_sha256'];data=np.load(out/'frozen_spaces.npz');records=[]
    for method,dimension in protocol['dimensions'].items():
        if method in ('shared','Hamiltonian_q8','Hamiltonian_q16','exact_symmetry'):reuse='pair_symbol_pilot_01'
        elif '_contrast_' in method:reuse='pair_contrast_01'
        else:reuse=None
        for start in ('Fermi','restart'):
            initial=data['F'] if start=='Fermi' else data['target']
            if reuse:
                source=out.parent/reuse;prior=np.load(source/'frozen_spaces.npz');assert np.linalg.norm(data[method]-prior[method])<1e-13
                row=next(r for r in json.loads((source/'frozen_predictions.json').read_text()) if r['method']==method and r['start']==start)
                assert digest(source/row['file'])==row['sha256']
                records.append({**row,'file':str(Path('..')/reuse/row['file']),'reused':True});continue
            begin=0. if start=='Fermi' else 1.35;times=np.arange(round(begin/.05),51)*.05
            states,error=propagate(data['H'],data[method],initial,times-times[0]);assert error<2e-11
            path=out/f'{method}_{start}.npz';assert not path.exists();np.savez_compressed(path,times=times,states=states)
            records.append({'method':method,'dimension':dimension,'start':start,'initial_error':error,
                'norm_error':float(np.max(abs(np.sum(abs(states)**2,axis=1)-1))),
                'file':path.name,'sha256':digest(path),'reused':False})
    (out/'frozen_predictions.json').write_text(json.dumps(records,indent=2))
    print(json.dumps({'evaluated_trajectories':len(records),'new_trajectories':sum(not r['reused'] for r in records),'reused_trajectories':sum(r['reused'] for r in records)}),flush=True)


def evaluate(out):
    protocol=json.loads((out/'protocol.json').read_text());data=np.load(out/'frozen_spaces.npz');records=json.loads((out/'frozen_predictions.json').read_text())
    H=data['H'];operators=data['pair_channels'];counts=data['counts'];times=np.arange(51)*.05
    exact=expm_multiply(-1j*H,data['F'],start=0,stop=2.5,num=51,traceA=-1j*np.trace(H))
    np.savez_compressed(out/'exact_references.npz',times=times,states=exact)
    reference=[noise_observables(operators,v) for v in exact];rows=[]
    for record in records:
        path=out/record['file'];assert digest(path)==record['sha256'];saved=np.load(path);results=[]
        for t,v in zip(saved['times'],saved['states']):
            k=round(float(t)*20);ref=reference[k];means,connected,variance=noise_observables(operators,v)
            p=np.bincount(counts,weights=abs(v)**2,minlength=8);p0=np.bincount(counts,weights=abs(exact[k])**2,minlength=8)
            results.append({'time':float(t),'fidelity':float(abs(np.vdot(exact[k],v))**2),'count_TV':float(.5*np.sum(abs(p-p0))),
                'pair_channel_L2_error':float(np.linalg.norm(means-ref[0])),
                'two_transfer_connected_L2_error':float(np.linalg.norm(connected-ref[1])),
                'pair_pulse_variance_L2_error':float(np.linalg.norm(variance-ref[2])),
                'connected_real':connected.real.tolist(),'connected_imag':connected.imag.tolist(),
                'pulse_variances':variance.tolist(),
                'variance_identity_error':float(np.max(abs(variance[::2]-variance[1::2]-4*np.diag(connected).real)))})
        rows.append({**record,'rows':results,'primary':next(r for r in results if abs(r['time']-2)<1e-12)})
    lookup={(r['method'],r['start']):r for r in rows};gates=[]
    for row in rows:
        name=row['method'];start=row['start']
        if not name.startswith('canonical_') or '_H_' in name or '_contrast_' in name:continue
        rank=int(name.rsplit('q',1)[1]);controls=[name.replace('canonical','unoptimized'),f'ordinary2_q{rank}',f'Hamiltonian_q{rank}']
        if '_joint_' in name:controls+=['canonical_contrast_q8','canonical_single_H_q16','canonical_contrast_q16']
        passing=all(all(row['primary'][key]<lookup[control,start]['primary'][key]-1e-9 for key in ('count_TV','two_transfer_connected_L2_error','pair_pulse_variance_L2_error')) for control in controls)
        gates.append({'method':name,'start':start,'controls':controls,'passes':passing})
    (out/'evaluation.json').write_text(json.dumps({'classification':protocol['classification'],'models':rows,'gate':gates},indent=2))
    print(json.dumps({'gate':gates,'primary_restart':[{k:r[k] for k in ('method','dimension','primary')} for r in rows if r['start']=='restart']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['train','assemble','predict','evaluate'],required=True);a=p.parse_args();globals()[a.stage](a.output)
