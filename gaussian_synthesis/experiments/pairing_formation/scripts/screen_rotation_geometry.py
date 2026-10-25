#!/usr/bin/env python3
"""Conventional curved-field screen before any canonical discovery claim."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,time,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.sparse import csr_matrix,save_npz,load_npz
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.gaussian_bins import angular_invariant_basis
from pairing_formation.rotation_geometry import matched_sector_atoms,two_band_sector_atoms,orthogonal_columns,profile_directions
from pairing_formation.frozen import channel_basis
from pairing_formation.chirality_spectrum import ChiralitySpectrum


def predict(task):
    L,out,profiles,seed,tolerance,matching=task;begin=time.monotonic();model,forms=angular_model(L);dense,_=angular_invariant_basis(model);Q=csr_matrix(dense);del dense
    H=(Q.T@model.H@Q).tocsr();F=Q.T@model.initial;times=np.arange(51)/20
    U,_=np.linalg.qr(forms);B,_=channel_basis(model,U@U.T,2);base,s=orthogonal_columns(Q.T@B,tolerance);del B
    save_npz(out/f'L{L}_symmetry.npz',Q);save_npz(out/f'L{L}_Hamiltonian.npz',H)
    coeff=profile_directions(profiles,seed);amplitudes=[.25,.6,1.,1.4];pool=[];records=[]
    if matching=='matched':hole_coeff=coeff.conj()
    elif matching=='shifted':
        d=np.pi/4;R=np.array([[np.cos(d),-np.sin(d)],[np.sin(d),np.cos(d)]])
        hole_coeff=coeff.conj()@R.T
    elif matching=='independent':hole_coeff=profile_directions(profiles,seed+1001).conj()
    else:raise ValueError(matching)
    frozen={'source_q2':base,'exact_symmetry':np.eye(Q.shape[1])};spectra={'source_q2':s.tolist()}
    for i,c in enumerate(coeff):
        field=forms@c;hole_field=forms@hole_coeff[i]
        for A in amplitudes:pool.append(Q.T@two_band_sector_atoms(model,field,hole_field,A,A))
        if i+1 in (1,2,4,8,16,32,64,128) or i+1==profiles:
            name=f'rotation_profiles{i+1}';V,s=orthogonal_columns(np.column_stack([base,*pool]),tolerance)
            frozen[name]=V;spectra[name]=s.tolist()
    np.savez_compressed(out/f'L{L}_frozen_spaces.npz',**frozen,profile_coefficients=coeff,hole_profile_coefficients=hole_coeff,amplitudes=amplitudes)
    for name,V in frozen.items():
        Hr=V.conj().T@(H@V);initial=V.conj().T@F;error=float(np.linalg.norm(V@initial-F));assert error<2e-11
        propagated=expm_multiply(-1j*Hr,initial,start=0.,stop=2.5,num=51,traceA=-1j*np.trace(Hr))
        states=propagated@V.T;filename=f'L{L}_{name}.npz'
        np.savez_compressed(out/filename,times=times,states=states)
        records.append({'L':L,'name':name,'dimension':V.shape[1],'symmetry_dimension':Q.shape[1],
          'initial_error':error,'max_norm_error':float(np.max(abs(np.sum(abs(states)**2,axis=1)-1))),
          'file':filename,'sha256':hashlib.sha256((out/filename).read_bytes()).hexdigest()})
    metadata={'L':L,'full_dimension':model.dim,'symmetry_dimension':Q.shape[1],'source_q2_dimension':base.shape[1],
      'singular_values':spectra,'frozen_spaces_sha256':hashlib.sha256((out/f'L{L}_frozen_spaces.npz').read_bytes()).hexdigest(),
      'predictions':records,'seconds':time.monotonic()-begin}
    (out/f'L{L}_frozen.json').write_text(json.dumps(metadata,indent=2));print(json.dumps({k:metadata[k] for k in ('L','full_dimension','symmetry_dimension','source_q2_dimension','seconds')}),flush=True)
    return metadata


def evaluate(task):
    L,out=task;model,forms=angular_model(L);Q=load_npz(out/f'L{L}_symmetry.npz');H=load_npz(out/f'L{L}_Hamiltonian.npz')
    record=json.loads((out/f'L{L}_frozen.json').read_text());exact=np.load(out.parent/'nonlinear_channels_01'/f'L{L}_eta1_exact.npz')
    assert hashlib.sha256((out/f'L{L}_frozen_spaces.npz').read_bytes()).hexdigest()==record['frozen_spaces_sha256']
    refs=exact['states']@Q
    # Each normalized orbit has constant transfer count; compute by weights.
    counts=np.rint(np.asarray(Q.power(2).T@model.m).ravel()).astype(int)
    probe=ChiralitySpectrum(model);K=np.zeros((L,L),complex);K[:model.r,:model.r]=probe.kernel;K[model.r:,model.r:]=probe.kernel
    action=model.pair_operator(K)@Q
    reference_chi=np.sum(abs((action@refs.T).T)**2,axis=1)/model.r**2
    reference_counts=np.array([np.bincount(counts,weights=abs(s)**2,minlength=model.r+1) for s in refs])
    reference_current=np.array([[2*np.vdot(s[counts==m],(H@s)[counts==m]).imag for m in range(model.r+1)] for s in refs])
    pspace,_=np.linalg.qr(forms);_,m1ids=model.blocks[1];selected=Q[m1ids.ravel()]
    def dark(s):
        C=(selected@s).reshape(model.r,model.r);source=pspace@(pspace.T@C@pspace)@pspace.T
        return float(np.linalg.norm(C-source)**2)
    reference_dark=[dark(s) for s in refs];models=[]
    for row in record['predictions']:
        path=out/row['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        data=np.load(path);states=data['states'];chi=np.sum(abs((action@states.T).T)**2,axis=1)/model.r**2;rows=[]
        for k,(t,s,ref) in enumerate(zip(data['times'],states,refs)):
            p=np.bincount(counts,weights=abs(s)**2,minlength=model.r+1)
            current=np.array([2*np.vdot(s[counts==m],(H@s)[counts==m]).imag for m in range(model.r+1)])
            rows.append({'time':float(t),'fidelity':float(abs(np.vdot(ref,s))**2),'count_TV':float(.5*np.sum(abs(p-reference_counts[k]))),
              'current_L1_error':float(np.sum(abs(current-reference_current[k]))),'chi_second_error':float(chi[k]-reference_chi[k]),
              'm1_dark_probability':dark(s),'exact_m1_dark_probability':reference_dark[k]})
        models.append({**row,'rows':rows})
    result={'L':L,'symmetry_dimension':Q.shape[1],'models':models,'max_exact_symmetry_projection_error':float(np.max(np.linalg.norm(exact['states']-(Q@refs.T).T,axis=1)))}
    (out/f'L{L}_evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps({'L':L,'time2':[{'name':r['name'],'dimension':r['dimension'],**r['rows'][40]} for r in models]}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['predict','evaluate'],required=True);p.add_argument('--levels',nargs='+',type=int,default=[14,18]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profiles',type=int,default=16);p.add_argument('--seed',type=int,default=861);p.add_argument('--tolerance',type=float,default=1e-9)
    p.add_argument('--matching',choices=['matched','shifted','independent'],default='matched');args=p.parse_args()
    if args.stage=='predict':
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'protocol.json').write_text(json.dumps({'classification':'Known-condition conventional-field mechanism screen, not canonical measure sampling and not new physical holdouts.',
          'levels':args.levels,'coupling':4.,'eta':1.,'matching':args.matching,'profiles':'Particle/hole Gaussian spinors with two known source fields; nonlinear unitary spin rotation versus linear Thouless coordinates. Matching is conjugate, conjugate with pi/4 relative rotation, or independently chosen after the same first four profiles.',
          'directions':args.profiles,'seed':args.seed,'amplitudes':[.25,.6,1.,1.4],'shared':'Exact source-q2 Gaussian span, exact Fermi initial state, all complex count-sector coefficients free.',
          'deletion':'Remove nonlinear rotated profiles but retain exact F and source-q2 span. No future-state-dependent selection.',
          'tolerance':args.tolerance,'warning':'A fixed number of nonlinear collective coordinates may still span the entire finite Hilbert space. Record ranks and avoid a polynomial-cost or independent-field-count claim.'},indent=2))
    function=predict if args.stage=='predict' else evaluate
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks=[(L,args.output,args.profiles,args.seed,args.tolerance,args.matching) if args.stage=='predict' else (L,args.output) for L in args.levels]
        for f in as_completed([pool.submit(function,t) for t in tasks]):f.result()
