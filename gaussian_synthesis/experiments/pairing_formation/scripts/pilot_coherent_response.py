#!/usr/bin/env python3
"""Complete symmetry-sector coherent response, followed by frozen pilots."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse, hashlib, json, time, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.replica2_gpu import Replica2GPU
from pairing_formation.coherent_partition import coherent_parts
from pairing_formation.coherent_response import response_modes, ordered_krylov
from pairing_formation.dark_pairs import PairDarkSectors


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(task):
    output, method, chunk, device = task
    started = time.monotonic()
    data = np.load(output/'training.npz')
    Q = data['Q']; u = data[method]
    model, _ = angular_model(14)
    replica = Replica2GPU(model, device=f'cuda:{device}')
    columns = np.arange(chunk, Q.shape[1], 4)
    A = []; B = []
    for col in columns:
        parts = coherent_parts(replica, u, Q[:, col])
        A.append(Q.conj().T @ parts['complex_ratio'])
        B.append(Q.conj().T @ parts['conjugate_ratio'])
    path = output/f'{method}_chunk{chunk}.npz'
    assert not path.exists()
    np.savez_compressed(path, columns=columns, A=np.column_stack(A), B=np.column_stack(B))
    record = {'method':method, 'chunk':chunk, 'device':device, 'columns':len(columns),
              'seconds':time.monotonic()-started, 'file':path.name, 'sha256':digest(path)}
    print(json.dumps(record), flush=True)
    return record


def train(output):
    output.mkdir(parents=True, exist_ok=False)
    root = output.parent
    source = root/'dark_gaussian_training_01/training.npz'
    refined = root/'dark_phase_refinement_01/refinement.npz'
    shared = root/'dark_phase_parts_01/frozen_spaces.npz'
    data = np.load(source); old = np.load(shared)
    np.savez_compressed(output/'training.npz', Q=data['Q'], target=data['target'],
                        canonical=np.load(refined)['witness'], unoptimized=data['target'],
                        base=old['shared'])
    protocol = {'classification':'Retrospective L14 G4 eta1 t1 training and known-condition propagation; no new parameter holdouts.',
        'physical_dictionary':'Full even Gaussian dictionary on 28 physical modes, k2; witness and output compressed only by exact symmetry.',
        'input_hashes':{str(p):digest(p) for p in (source, refined, shared)},
        'training_sha256':digest(output/'training.npz'),
        'selection':'Complete A and B on all 170 symmetry coordinates. Leading residual A eigenvectors, residual B singular vectors, and singular vectors of cross coupling [A base, B base*]. No future target states enter selection.',
        'ranks':[1,2,4,8,16], 'base_dimension':int(old['shared'].shape[1]),
        'primary_time':2., 'primary_start':'restart from the identical exact t1 training state',
        'gate':'A canonical selection must improve count TV, sector-current L1 and dark-probability error over its same-rank unoptimized counterpart and both Hamiltonian controls before new physical holdouts. Improvements must survive numerical and physical interpretation controls.',
        'additional_controls':'Same exact Fermi sea initial state at t0; all cases retain shared source and complete one-transfer sector. Dropping added directions recovers the shared-space deletion.',
        'scope':'Modes are coherent integral responses, not selected Gaussian atoms, probabilities, or a general cost claim.'}
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    tasks = [(output, method, chunk, 4*j+chunk) for j,method in enumerate(('canonical','unoptimized')) for chunk in range(4)]
    records=[]
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(worker, task) for task in tasks]):
            records.append(f.result())
    (output/'kernel_jobs.json').write_text(json.dumps(records, indent=2))
    assemble(output)


def assemble(output):
    data=np.load(output/'training.npz'); base=data['base']; Q=data['Q']; n=Q.shape[1]
    protocol=json.loads((output/'protocol.json').read_text())
    assert digest(output/'training.npz')==protocol['training_sha256']
    jobs=json.loads((output/'kernel_jobs.json').read_text())
    for row in jobs: assert digest(output/row['file'])==row['sha256']
    model,forms=angular_model(14); H=Q.T@(model.H@Q)
    dark=PairDarkSectors(model,forms)
    projectors=Q.T@dark.apply(Q,'either')
    counts=np.rint((Q*Q).T@model.m).astype(int)
    projectors*=((counts>=2)[:,None] & (counts>=2)[None,:])
    arrays={'Q':Q,'target':data['target']}; spaces={'shared':base}; reports={}; modes_all={}
    for method in ('canonical','unoptimized'):
        A=np.empty((n,n),complex); B=np.empty((n,n),complex)
        for chunk in range(4):
            x=np.load(output/f'{method}_chunk{chunk}.npz')
            A[:,x['columns']]=x['A']; B[:,x['columns']]=x['B']
        diagnostics={'Hermitian_error':float(np.linalg.norm(A-A.conj().T)),
                     'symmetric_error':float(np.linalg.norm(B-B.T)),
                     'minimum_A_eigenvalue':float(np.linalg.eigvalsh((A+A.conj().T)/2)[0])}
        assert diagnostics['Hermitian_error']<2e-11 and diagnostics['symmetric_error']<2e-11
        assert diagnostics['minimum_A_eigenvalue']>-2e-12
        w=Q.T@data[method]
        diagnostics['self_A_B_difference']=float(np.linalg.norm(A@w-B@w.conj()))
        if method=='canonical':
            diagnostics['self_reconstruction_error']=float(np.linalg.norm(A@w-Q.T@data['target']))
            assert diagnostics['self_reconstruction_error']<2e-12
        arrays[method+'_A']=A; arrays[method+'_B']=B
        modes,spectra=response_modes(A,B,base)
        diagnostics['spectra']={k:v.tolist() for k,v in spectra.items()}
        for kind,V in modes.items():
            modes_all[method+'_'+kind]=V
            for rank in protocol['ranks']:
                if rank>V.shape[1] or spectra[kind][rank-1]<1e-12: continue
                name=f'{method}_{kind}_q{rank}'
                spaces[name]=np.column_stack((base,V[:,:rank]))
        reports[method]=diagnostics
    residual=H@base-base@(base.conj().T@H@base)
    hmode,hs,_=np.linalg.svd(residual,full_matrices=False)
    kmode=ordered_krylov(H,Q.T@data['target'],base,16)
    for rank in protocol['ranks']:
        if hs[rank-1]>1e-10:
            spaces[f'Hamiltonian_block_q{rank}']=np.column_stack((base,hmode[:,:rank]))
        if kmode.shape[1]>=rank:
            spaces[f'Hamiltonian_Krylov_q{rank}']=np.column_stack((base,kmode[:,:rank]))
    spaces['exact_symmetry']=np.eye(n)
    geometry={}; content={}
    for kind in ('eigen','anomalous','cross'):
        for rank in protocol['ranks']:
            a=modes_all['canonical_'+kind][:,:rank]; b=modes_all['unoptimized_'+kind][:,:rank]
            geometry[f'{kind}_q{rank}']=np.linalg.svd(a.conj().T@b,compute_uv=False).tolist()
    for name,V in spaces.items():
        gram=float(np.linalg.norm(V.conj().T@V-np.eye(V.shape[1])))
        assert gram<2e-10,(name,gram)
        W=V[:,base.shape[1]:]
        content[name]={'dimension':V.shape[1],'Gram_error':gram,
                      'added_dark_trace':float(np.trace(W.conj().T@projectors@W).real),
                      'added_sector_trace':[float(np.sum(abs(W[counts==m])**2)) for m in range(model.r+1)]}
    arrays.update(spaces); arrays.update({k+'_modes':v for k,v in modes_all.items()})
    path=output/'frozen_spaces.npz'; assert not path.exists()
    np.savez_compressed(path,**arrays)
    protocol['frozen_sha256']=digest(path);protocol['dimensions']={k:v.shape[1] for k,v in spaces.items()}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (output/'geometry.json').write_text(json.dumps({'kernels':reports,'canonical_vs_unoptimized':geometry,'content':content},indent=2))
    print(json.dumps({'spaces':len(spaces),'kernel_checks':{k:{j:x for j,x in v.items() if j!='spectra'} for k,v in reports.items()},'geometry':geometry}),flush=True)


def predict(output):
    protocol=json.loads((output/'protocol.json').read_text())
    assert digest(output/'frozen_spaces.npz')==protocol['frozen_sha256']
    data=np.load(output/'frozen_spaces.npz'); Q=data['Q'];model,_=angular_model(14);H=Q.T@(model.H@Q)
    records=[]
    for name,dimension in protocol['dimensions'].items():
        V=data[name]; Hr=V.conj().T@H@V
        for start in ('Fermi','restart'):
            times=np.arange(0 if start=='Fermi' else 20,51)/20
            initial=Q.T@(model.initial if start=='Fermi' else data['target'])
            coefficients=V.conj().T@initial
            error=float(np.linalg.norm(V@coefficients-initial));assert error<3e-12
            states=expm_multiply(-1j*Hr,coefficients,start=0.,stop=times[-1]-times[0],num=len(times),traceA=-1j*np.trace(Hr))@V.T
            path=output/f'{name}_{start}.npz';assert not path.exists()
            np.savez_compressed(path,times=times,states=states)
            records.append({'name':name,'dimension':dimension,'start':start,'file':path.name,'sha256':digest(path),
                'initial_error':error,'norm_error':float(np.max(abs(np.sum(abs(states)**2,axis=1)-1)))})
    (output/'frozen_predictions.json').write_text(json.dumps(records,indent=2))
    print(json.dumps({'predictions':len(records)}),flush=True)


def evaluate(output):
    records=json.loads((output/'frozen_predictions.json').read_text());data=np.load(output/'frozen_spaces.npz')
    model,forms=angular_model(14);dark=PairDarkSectors(model,forms);Q=data['Q'];H=Q.T@(model.H@Q)
    exact=np.load(output.parent/'nonlinear_channels_01/L14_eta1_exact.npz')['states'];refs=exact@Q
    counts=np.rint((Q*Q).T@model.m).astype(int)
    D=Q.T@dark.apply(Q,'either');D*=((counts>=2)[:,None]&(counts>=2)[None,:])
    truth=np.einsum('ti,ij,tj->t',refs.conj(),D,refs).real
    results=[]
    for record in records:
        path=output/record['file'];assert digest(path)==record['sha256'];prediction=np.load(path);rows=[]
        for t,s in zip(prediction['times'],prediction['states']):
            index=round(float(t)*20);ref=refs[index]
            p=np.bincount(counts,weights=abs(s)**2,minlength=model.r+1)
            p0=np.bincount(counts,weights=abs(ref)**2,minlength=model.r+1)
            x,y=H@s,H@ref
            current=np.array([2*np.vdot(s[counts==m],x[counts==m]).imag for m in range(model.r+1)])
            current0=np.array([2*np.vdot(ref[counts==m],y[counts==m]).imag for m in range(model.r+1)])
            d=float(np.vdot(s,D@s).real)
            rows.append({'time':float(t),'fidelity':float(abs(np.vdot(ref,s))**2),'count_TV':float(.5*np.sum(abs(p-p0))),
                'current_L1_error':float(np.sum(abs(current-current0))),'multiple_dark_probability':d,'multiple_dark_error':d-truth[index]})
        results.append({**record,'rows':rows})
    primary={r['name']:next(x for x in r['rows'] if abs(x['time']-2)<1e-12) for r in results if r['start']=='restart'}
    gate=[]
    for name,row in primary.items():
        if not name.startswith('canonical_'):continue
        rank=int(name.rsplit('q',1)[1]);baseline=name.replace('canonical_','unoptimized_')
        controls=[baseline,f'Hamiltonian_block_q{rank}',f'Hamiltonian_Krylov_q{rank}']
        valid=all(c in primary for c in controls)
        for c in controls:
            if c not in primary:continue
            valid &= all(abs(row[k])<abs(primary[c][k])-1e-9 for k in ('count_TV','current_L1_error','multiple_dark_error'))
        gate.append({'method':name,'passes_primary_all_controls':bool(valid),'controls':controls})
    (output/'evaluation.json').write_text(json.dumps({'classification':'Known-condition selection pilot; not independent physical holdouts.','models':results,'gate':gate},indent=2))
    print(json.dumps({'primary':primary,'gate':gate},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['train','assemble','predict','evaluate'],required=True)
    args=p.parse_args();globals()[args.stage](args.output)
