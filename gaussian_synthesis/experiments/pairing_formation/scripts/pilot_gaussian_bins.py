#!/usr/bin/env python3
"""Known-condition frozen coherent-bin pilot with identical initial states."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.column_prediction import orthonormal_span,evolve_span


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def augment(base,candidates,q):
    residual=candidates-base@(base.conj().T@candidates)
    U,s,_=np.linalg.svd(residual,full_matrices=False)
    assert s[q-1]>1e-9*s[0]
    return np.column_stack((base,U[:,:q]))


def train(source,out):
    out.mkdir(parents=True,exist_ok=False)
    assessment=source/'assessment.json';audit=json.loads(assessment.read_text());bins=np.load(source/'coherent_bins.npz')
    training=source.parent/'gaussian_bins_L10_01/training.npz';data=np.load(training)
    model,_=angular_model(10);Q=data['Q'];psi=Q.T@data['target'];F=Q.T@model.initial
    H=Q.T@(model.H@Q);m=Q.T@(model.m[:,None]*Q);X2=Q.T@(data['chi']@data['chi']@Q)
    base=bins['base'];spaces={'shared':base};geometry={}
    for method in ('canonical','husimi_moment','linear_frame'):
        for label,ranks in (('count_only',(1,2)),('count_and_circulation',(1,))):
            for q in ranks:
                gate=next(x for x in audit['methods'][method][label]['rank_checks'] if x['rank']==q)
                assert gate['passes_declared_gate']
                extra=bins[method+'_'+label+'_basis'][:,:q]
                spaces[f'{method}_{label}_q{q}']=np.column_stack((base,extra))
    count=np.column_stack((m@psi,m@m@psi))
    response=np.column_stack((H@psi,H@H@psi))
    for q in (1,2):
        # Ordered polynomial and Hamiltonian response spaces; no future fit.
        spaces[f'count_polynomial_q{q}']=augment(base,count[:,:q],q)
        spaces[f'Hamiltonian_response_q{q}']=augment(base,response[:,:q],q)
        v=F.copy();vectors=[]
        for k in range(3+q):vectors.append(v);v=H@v
        spaces[f'Fermi_Krylov_dim{3+q}']=orthonormal_span(np.column_stack(vectors))[0]
    spaces['exact_symmetry_space']=np.eye(Q.shape[1])
    # The Fermi Krylov spaces contain F, but generally do not contain psi.
    # Only use them for F starts; restarted trials require identical psi.
    ordinary=augment(base,np.column_stack((count,X2@psi)),3)
    for method in ('canonical','husimi_moment','linear_frame'):
        B=bins[method+'_count_only_basis'][:,:2]
        geometry[method]={'squared_projection_on_shared_count_and_chi2_span':np.sum(abs(ordinary.conj().T@B)**2,axis=0).tolist(),
          'cosines_with_linear_frame_count_rank2':np.linalg.svd(B.conj().T@bins['linear_frame_count_only_basis'][:,:2],compute_uv=False).tolist()}
    np.savez_compressed(out/'frozen_spaces.npz',Q=Q,target=data['target'],**spaces)
    protocol={'classification':'Retrospective pilot: L10 G4 eta1 has already been inspected. This is not an independent new-parameter prediction.',
      'source':str(source),'source_hash':digest(assessment),'training_hash':digest(training),
      'spaces_hash':digest(out/'frozen_spaces.npz'),'training_time':1.,
      'selection':'Only ranks passing the previously saved independent-half Monte Carlo gate. Common F, psi(t1), C2(psi(t1)) in all bin and response methods.',
      'coherent_bins_are_single_Gaussians':False,
      'initial_conditions':'Exact Fermi sea at t0 for all spaces; exact psi(t1) restart for every space except Fermi-only Krylov. Initial differences explicitly checked.',
      'deletion':'Shared-only space deletes all learned bin directions while keeping precisely the same physical initial state.',
      'controls':'Linear Haar frame, unoptimized cubic/Husimi moment, ordinary count-polynomial response, Hamiltonian response, Fermi Krylov, exact 26-dimensional symmetry closure.',
      'primary_time':2.,'times':'0:.05:2.5, or restart at 1:.05:2.5',
      'geometry':geometry,'dimensions':{k:v.shape[1] for k,v in spaces.items()}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps({'spaces':len(spaces),'geometry':geometry}),flush=True)


def predict(out):
    data=np.load(out/'frozen_spaces.npz');protocol=json.loads((out/'protocol.json').read_text())
    assert digest(out/'frozen_spaces.npz')==protocol['spaces_hash']
    model,_=angular_model(10);Q=data['Q'];records=[]
    for name in protocol['dimensions']:
        V=Q@data[name]
        np.testing.assert_allclose(V.conj().T@V,np.eye(V.shape[1]),atol=1e-10)
        for start in ('Fermi','restart'):
            if start=='restart' and name.startswith('Fermi_Krylov'):continue
            time0=0. if start=='Fermi' else 1.
            initial=model.initial if start=='Fermi' else data['target']
            c=V.conj().T@initial;error=float(np.linalg.norm(V@c-initial));assert error<2e-12
            times=np.arange(round(time0*20),51)/20
            states=evolve_span(model,V,c,times-time0)
            filename=f'{name}_{start}.npz';assert not (out/filename).exists()
            np.savez_compressed(out/filename,times=times,states=states)
            records.append({'name':name,'start':start,'dimension':V.shape[1],'file':filename,
              'sha256':digest(out/filename),'initial_error':error,
              'max_norm_error':float(np.max(abs(np.sum(abs(states)**2,axis=1)-1)))})
    (out/'frozen_predictions.json').write_text(json.dumps(records,indent=2))
    print(json.dumps({'predictions':len(records),'max_initial_error':max(x['initial_error'] for x in records)}),flush=True)


def evaluate(out):
    records=json.loads((out/'frozen_predictions.json').read_text());model,_=angular_model(10)
    training=np.load(out.parent/'gaussian_bins_L10_01/training.npz');chi=training['chi'];data=np.load(out/'frozen_spaces.npz')
    assert digest(out/'frozen_spaces.npz')==json.loads((out/'protocol.json').read_text())['spaces_hash']
    # Existing exact trajectories are first accessed only after predictions saved.
    exact=np.load(out.parent/'nonlinear_channels_01/L10_eta1_exact.npz')
    refs={round(float(t),10):state for t,state in zip(exact['times'],exact['states'])}
    results=[]
    for record in records:
        assert digest(out/record['file'])==record['sha256']
        forecast=np.load(out/record['file']);rows=[]
        for t,state in zip(forecast['times'],forecast['states']):
            state=state/np.linalg.norm(state);ref=refs[round(float(t),10)];ref=ref/np.linalg.norm(ref)
            p=np.bincount(model.m,weights=abs(state)**2,minlength=6);p0=np.bincount(model.m,weights=abs(ref)**2,minlength=6)
            current=np.array([2*np.vdot(state[model.m==m],(model.H@state)[model.m==m]).imag for m in range(6)])
            current0=np.array([2*np.vdot(ref[model.m==m],(model.H@ref)[model.m==m]).imag for m in range(6)])
            second=float(np.linalg.norm(chi@state)**2/model.r**2);second0=float(np.linalg.norm(chi@ref)**2/model.r**2)
            rows.append({'time':float(t),'fidelity':float(abs(np.vdot(ref,state))**2),
              'count_TV':float(.5*np.sum(abs(p-p0))),'count_mean_error':float((p-p0)@np.arange(6)),
              'count_variance':float(p@np.arange(6)**2-(p@np.arange(6))**2),
              'chi_second_per_r_squared':second,'chi_second_error':second-second0,
              'current_L1_error':float(np.sum(abs(current-current0)))})
        results.append({**record,'rows':rows})
    audit={'classification':'Known-condition pilot, not fresh holdout. Full-matrix evolution; no scalability or equal-total-cost advantage.',
      'verified_prediction_files':len(records),'max_initial_error':max(x['initial_error'] for x in records),
      'max_norm_error':max(x['max_norm_error'] for x in records),'models':results}
    (out/'evaluation.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps([{'name':x['name'],'start':x['start'],'dimension':x['dimension'],
      **next(y for y in x['rows'] if abs(y['time']-2)<1e-10)} for x in results],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train','predict','evaluate'],required=True)
    p.add_argument('--source',type=Path,default=Path('results/gaussian_bins_assessment_01'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.stage=='train':train(args.source,args.output)
    elif args.stage=='predict':predict(args.output)
    else:evaluate(args.output)
