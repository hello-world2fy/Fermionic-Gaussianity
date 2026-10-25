#!/usr/bin/env python3
"""Phase-twirl controls with identical counts and all paired four-point data."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,argparse,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.sparse import load_npz,csr_matrix
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.gaussian_bins import angular_invariant_basis
from pairing_formation.multipair_noise import transfer_operators
from scripts.pilot_pair_symbols import digest


def pair_matrix(model,ensemble):
    R=np.diag(model.occ.T@np.sum(abs(ensemble)**2,axis=1)).astype(complex)
    for (i,j),(dst,src) in model.transitions.items():R[i,j]=np.sum(ensemble[dst].conj()*ensemble[src])
    return R


def worker(task):
    out,L,q=task;root=out.parent;model,_=angular_model(L)
    if L==10:Q=csr_matrix(angular_invariant_basis(model)[0])
    else:Q=load_npz(root/f'rotation_geometry_01/L{L}_symmetry.npz')
    counts=np.rint(np.asarray(Q.power(2).T@model.m)).astype(int);X=Q.T@(transfer_operators(model)[q]@Q)
    rows=[];checks=[];hashes={}
    for method in ('exact','full_cc'):
        path=root/f'nonlinear_channels_01/L{L}_eta1_{method}.npz';hashes[str(path)]=digest(path);data=np.load(path)
        full=data['states'][np.argmin(abs(data['times']-2))];full=full/np.linalg.norm(full);u=Q.T@full
        residual=float(np.linalg.norm(Q@u-full));assert residual<1e-10
        even=np.column_stack([u*(counts%2==p) for p in (0,1)])
        diagonal=np.column_stack([u*(counts==m) for m in range(model.r+1)])
        # These are unnormalized ensemble columns with total trace one.
        e_full=Q@even;d_full=Q@diagonal
        checks.append({'L':L,'q':q,'method':method,'symmetry_residual':residual,
            'initial_diagonal_difference':float(np.max(abs(np.sum(abs(e_full)**2,axis=1)-np.sum(abs(d_full)**2,axis=1)))),
            'all_pair_correlator_difference':float(np.max(abs(pair_matrix(model,e_full)-pair_matrix(model,d_full))))})
        initial=np.column_stack((even,diagonal));second=np.vdot(u,X@X@u)
        for phi in (0.,np.pi/4):
            pulse=np.exp(-1j*phi)*X+np.exp(1j*phi)*X.conj().T
            for epsilon in (.02,.01,.1,.2,.4):
                moved=expm_multiply(-1j*epsilon*pulse,initial,traceA=0.)
                probabilities=[];variances=[];means=[]
                for selected in (moved[:,:2],moved[:,2:]):
                    p=np.bincount(counts,weights=np.sum(abs(selected)**2,axis=1),minlength=model.r+1)
                    m=np.arange(model.r+1);mean=float(p@m);variance=float(p@(m*m)-mean**2)
                    probabilities.append(p.tolist());means.append(mean);variances.append(variance)
                delta=variances[0]-variances[1]
                rows.append({'L':L,'q':q,'method':method,'phase':float(phi),'epsilon':epsilon,
                    'before_count_TV':0.,'after_count_TV':float(.5*np.sum(abs(np.array(probabilities[0])-probabilities[1]))),
                    'variance_difference':delta,'mean_difference':means[0]-means[1],
                    'variance_coefficient':delta/epsilon**2,
                    'analytic_coefficient':float(-2*np.real(np.exp(-2j*phi)*second)),
                    'even_twirl_counts':probabilities[0],'full_twirl_counts':probabilities[1],
                    'max_trace_error':float(max(abs(np.sum(p)-1) for p in probabilities))})
    for method in ('exact','full_cc'):
        for phi in (0.,np.pi/4):
            small=[r for r in rows if r['method']==method and r['phase']==phi and r['epsilon']<.03]
            lookup={r['epsilon']:r for r in small};extrapolated=(4*lookup[.01]['variance_coefficient']-lookup[.02]['variance_coefficient'])/3
            checks.append({'L':L,'q':q,'method':method,'phase':float(phi),'Richardson_variance_coefficient_error':abs(extrapolated-small[0]['analytic_coefficient'])})
    path=out/f'L{L}_q{q}.json';path.write_text(json.dumps({'rows':rows,'checks':checks,'input_hashes':hashes},indent=2))
    return {'L':L,'q':q,'file':path.name,'sha256':digest(path),'pulse_pairs':len(rows)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(worker,(a.output,L,q)) for L in (10,14,18) for q in (1,3)]):
            rows.append(f.result());print(json.dumps(rows[-1]),flush=True)
    (a.output/'summary.json').write_text(json.dumps({'classification':'Retrospective physical readout controls on previously inspected sizes. Twirl channels preserve all counts and four-fermion data; additional pulse exposes even-transfer coherence. Not a canonical-specific discovery or new prediction.','jobs':rows},indent=2))
