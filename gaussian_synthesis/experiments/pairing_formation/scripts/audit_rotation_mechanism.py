#!/usr/bin/env python3
"""Retrospective capacity, coefficient-cost and missing-mode controls."""
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
from pairing_formation.frozen import channel_basis
from pairing_formation.rotation_geometry import matched_sector_atoms,orthogonal_columns


def job(task):
    L,out=task;root=out.parent;source=root/'rotation_geometry_01';model,forms=angular_model(L)
    Q=load_npz(source/f'L{L}_symmetry.npz');H=load_npz(source/f'L{L}_Hamiltonian.npz').toarray();F=Q.T@model.initial
    data=np.load(source/f'L{L}_frozen_spaces.npz');base=data['source_q2'];V=data['rotation_profiles16']
    count=np.rint(np.asarray(Q.power(2).T@model.m).ravel()).astype(int)
    full_vectors=[];_,ids=model.blocks[1];angle=2*np.pi*np.arange(model.r)/model.r
    for k in range((model.r+1)//2):
        vec=np.zeros(model.dim,complex);vec[ids]=np.cos(k*(angle[:,None]-angle[None,:]))/model.r*(1. if k==0 else np.sqrt(2))
        full_vectors.append(np.asarray(Q.T@vec))
    dark=np.column_stack(full_vectors);np.testing.assert_allclose(dark.conj().T@dark,np.eye(dark.shape[1]),atol=2e-13)
    spaces={};q4comparison={}
    if L==14:
        modes=np.column_stack([f(k*angle) for k in (1,3) for f in (np.cos,np.sin)]);U,_=np.linalg.qr(modes)
        original,_=channel_basis(model,U@U.T,4);B,s=orthogonal_columns(Q.T@original);del original
        spaces['q4']=B
        spaces['q4_plus_uniform_m1']=orthogonal_columns(np.column_stack((B,dark[:,0])))[0]
        spaces['q4_plus_all_m1']=orthogonal_columns(np.column_stack((B,dark)))[0]
        q4comparison={'original_ambient_dimension':2233,'exact_invariant_dimension':B.shape[1],
          'nonzero_projection_singular_values':s[s>1e-9].tolist()}
    preds={};prediction_records=[]
    for name,B in spaces.items():
        Hr=B.conj().T@H@B;c=B.conj().T@F;assert np.linalg.norm(B@c-F)<2e-12
        states=expm_multiply(-1j*Hr,c,start=0.,stop=2.5,num=51,traceA=-1j*np.trace(Hr))@B.T
        preds[name]=states;prediction_records.append({'name':name,'dimension':B.shape[1]})
    for name,removed in (('delete_uniform_m1',dark[:,:1]),('delete_all_nonsource_m1',dark[:,np.arange(dark.shape[1])!=1])):
        P=np.eye(Q.shape[1])-removed@removed.conj().T;np.testing.assert_allclose(P@F,F,atol=2e-13)
        Hr=P@H@P;states=expm_multiply(-1j*Hr,F,start=0.,stop=2.5,num=51,traceA=-1j*np.trace(Hr))
        preds[name]=states;prediction_records.append({'name':name,'dimension':Q.shape[1]-removed.shape[1],
          'max_deleted_mode_amplitude':float(np.max(abs(states@removed.conj())))})
    np.savez_compressed(out/f'L{L}_frozen_predictions.npz',times=np.arange(51)/20,dark_modes=dark,**preds)
    prediction_hash=hashlib.sha256((out/f'L{L}_frozen_predictions.npz').read_bytes()).hexdigest()
    # Capacity and coefficient cost use the original fixed dictionary.
    raw=[]
    for c in data['profile_coefficients']:
        for A in data['amplitudes']:raw.append(Q.T@matched_sector_atoms(model,forms@c,A))
    A=np.concatenate(raw,axis=1);R=A-base@(base.conj().T@A);U,s,Vh=np.linalg.svd(R,full_matrices=False)
    predicted=np.load(source/f'L{L}_rotation_profiles16.npz')['states'][40]
    residual=predicted-base@(base.conj().T@predicted)
    overlaps=U.conj().T@residual;support=np.max(abs(U.conj().T@R),axis=1)
    # Dual vectors are projected away from the free source space again.
    dual=U-base@(base.conj().T@U);dual/=np.maximum(np.linalg.norm(dual,axis=0),1e-300)
    support=np.max(abs(dual.conj().T@A),axis=1);dual_values=abs(dual.conj().T@predicted)
    valid=(support>1e-12)&(np.linalg.norm(base.conj().T@dual,axis=0)<1e-10)
    costs=[]
    for epsilon in (0.,.01,.05,.1):
        lower=np.where(valid,np.maximum(dual_values-epsilon,0)/np.maximum(support,1e-300),0)
        j=int(np.argmax(lower));costs.append({'allowed_state_norm_error':epsilon,'restricted_dictionary_l1_mass_lower_float64':float(lower[j]),
          'chosen_dual_overlap':float(dual_values[j]),'chosen_dual_support':float(support[j]),
          'free_source_space_leakage':float(np.linalg.norm(base.conj().T@dual[:,j]))})
    exact=np.load(root/'nonlinear_channels_01'/f'L{L}_eta1_exact.npz');refs=exact['states']@Q
    projection=np.sum(abs(refs@V.conj())**2,axis=1)
    for row in prediction_records:
        states=preds[row['name']];rows=[]
        for t,state,ref in zip(np.arange(51)/20,states,refs):
            p=np.bincount(count,weights=abs(state)**2,minlength=model.r+1);p0=np.bincount(count,weights=abs(ref)**2,minlength=model.r+1)
            I=np.array([2*np.vdot(state[count==m],(H@state)[count==m]).imag for m in range(model.r+1)])
            I0=np.array([2*np.vdot(ref[count==m],(H@ref)[count==m]).imag for m in range(model.r+1)])
            rows.append({'time':float(t),'fidelity':float(abs(np.vdot(ref,state))**2),'count_TV':float(.5*np.sum(abs(p-p0))),
              'current_L1_error':float(np.sum(abs(I-I0))),'uniform_m1_probability':float(abs(np.vdot(dark[:,0],state))**2),
              'exact_uniform_m1_probability':float(abs(np.vdot(dark[:,0],ref))**2)})
        row['rows']=rows
    if L==14:
        old=np.load(root/'nonlinear_channel_spans_01/L14_eta1_q4.npz')['states']@Q
        q4comparison['max_old_new_state_difference']=float(np.max(np.linalg.norm(old-preds['q4'],axis=1)))
    result={'L':L,'predictions_sha256':prediction_hash,'q4_symmetry_comparison':q4comparison,
      'rotation_static_projection_fidelity_time2':float(projection[40]),
      'rotation_min_static_projection_fidelity':float(min(projection)),
      'coefficient_cost_scope':'Finite specified rotation-sector orbit dictionary only, with source-q2 space free. Not a full-Gaussian extent lower bound or a continuous-manifold certificate. Floating-point dual evaluations, not interval certified.',
      'coefficient_cost_lower_bounds':costs,'models':prediction_records}
    (out/f'L{L}_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({'L':L,'costs':costs,'time2':[{'name':r['name'],'dimension':r['dimension'],**r['rows'][40]} for r in prediction_records]}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps({'classification':'Post-pilot mechanism audit on inspected conditions. No unseen forecasts.',
      'deletions':'Same exact Fermi state. Hamiltonian restricted to the complement of one-pair uniform mode, or all nonsource one-pair Fourier modes. Finite-grid indices; no continuum s-wave claim.',
      'cost':'Coefficient-mass lower bound for the saved finite rotation-sector dictionary with the source-q2 span treated as free. Does not bound the full Gaussian extent.',
      'question':'Is the improved observable caused just by restoring missing one-pair grid modes, and does the apparent low-coordinate representation have a large cancellation cost?'},indent=2))
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,(L,args.output)) for L in (14,18)]):f.result()
