#!/usr/bin/env python3
"""Formula checks and one-direction deletion for the exchange-return mode."""
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
from pairing_formation.return_pair_mode import return_pair_mode,fifth_order_coefficient
from pairing_formation.band_transfer import BandTransfer
from pairing_formation.pairing_channels import angular_model
from pairing_formation.dark_pairs import PairDarkSectors


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def formula_checks(out):
    records=[]
    # r=6 is deliberately an aliasing exception candidate: harmonic 3 equals
    # its negative. The inferred odd-r formula was not fitted to these tests.
    for r in (5,6,7,8,9,10,11,12,13,15):
        G=3.2;gap=.9;model=BandTransfer(r,G,maximum=3,shell_gap=gap)
        c=model.initial()
        for n in range(1,6):c=[-1j*x/n for x in model.apply(c)]
        actual=model.projected_dark(c[2],2);predicted=fifth_order_coefficient(r,G)
        b,d,v=return_pair_mode(r)
        singular=np.linalg.svd(actual,compute_uv=False)
        records.append({'r':r,'G':G,'gap':gap,'coefficient_norm':float(np.linalg.norm(actual)),
            'formula_error':float(np.linalg.norm(actual-predicted)),
            'relative_formula_error':float(np.linalg.norm(actual-predicted)/np.linalg.norm(actual)),
            'source_dark_error':max(float(np.linalg.norm(L@d)) for L in model.lowering[2]),
            'bright_dark_inner_product':float(abs(np.vdot(b,d))),
            'singular_values':singular[:6].tolist(),
            'odd_size_norm_formula':float(G**10*(r-3)**2*(r-4)/(1800*r**7)),
            'actual_norm_squared':float(np.linalg.norm(actual)**2)})
    (out/'formula_checks.json').write_text(json.dumps(records,indent=2))
    print(json.dumps({'formula_checks':records}),flush=True)


def job(task):
    L,out=task;root=out.parent;model,forms=angular_model(L);dark=PairDarkSectors(model,forms)
    Q=load_npz(root/'rotation_geometry_01'/f'L{L}_symmetry.npz')
    H=(Q.T@model.H@Q).toarray();F=Q.T@model.initial
    b,d,mode=return_pair_mode(model.r)
    physical=np.zeros(model.dim,complex);physical[model.blocks[2][1]]=mode
    phi=Q.T@physical
    symmetry_error=float(np.linalg.norm(Q@phi-physical));assert symmetry_error<1e-12
    Hp=H@phi;energy=np.vdot(phi,Hp)
    reduced=H-np.outer(phi,Hp.conj())-np.outer(Hp,phi.conj())+energy*np.outer(phi,phi.conj())
    times=np.arange(51)/20
    deleted=expm_multiply(-1j*reduced,F,start=0.,stop=2.5,num=51,traceA=-1j*np.trace(reduced))
    frozen=out/f'L{L}_frozen.npz';np.savez_compressed(frozen,times=times,mode=phi,states=deleted)
    meta={'L':L,'symmetry_error':symmetry_error,'frozen_sha256':digest(frozen),
          'initial_error':float(np.linalg.norm(deleted[0]-F)),
          'norm_error':float(np.max(abs(np.sum(abs(deleted)**2,axis=1)-1))),
          'deleted_mode_population_max':float(np.max(abs(deleted@phi.conj())**2))}
    (out/f'L{L}_frozen.json').write_text(json.dumps(meta,indent=2))
    exact_full=np.load(root/'nonlinear_channels_01'/f'L{L}_eta1_exact.npz')['states'];exact=exact_full@Q
    counts=np.rint(np.asarray(Q.power(2).T@model.m).ravel()).astype(int)
    rows=[]
    for t,s,ref,full in zip(times,deleted,exact,exact_full):
        p=np.bincount(counts,weights=abs(s)**2,minlength=model.r+1)
        p0=np.bincount(counts,weights=abs(ref)**2,minlength=model.r+1)
        h,h0=H@s,H@ref
        I=np.array([2*np.vdot(s[counts==m],h[counts==m]).imag for m in range(model.r+1)])
        I0=np.array([2*np.vdot(ref[counts==m],h0[counts==m]).imag for m in range(model.r+1)])
        C=full[model.blocks[2][1]];D=dark.projectors[2]
        darkC=D@C+C@D.T-D@C@D.T
        probability=float(abs(np.vdot(phi,ref))**2);m2dark=float(np.linalg.norm(darkC)**2)
        whole=dark.evaluate(full)['multiple_transfer_either_dark']
        rows.append({'time':float(t),'return_mode_probability':probability,'m2_dark_probability':m2dark,
            'fraction_m2_dark':probability/m2dark if m2dark>1e-16 else None,
            'fraction_all_multiple_dark':probability/whole if whole>1e-16 else None,
            'deletion_fidelity':float(abs(np.vdot(ref,s))**2),'deletion_count_TV':float(.5*np.sum(abs(p-p0))),
            'deletion_current_L1':float(np.sum(abs(I-I0)))})
    result={'checks':meta,'rows':rows}
    (out/f'L{L}_evaluation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'L':L,'checks':meta,'time1':rows[20],'time2':rows[40]}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps({'inferred_from':'r7 and r9 G4 gap1 fifth-order Hamiltonian path coefficients only.',
        'formula':'i G^5 (r-3)/(15 r^5) [d b^T + b d^T], b_ij=cos(theta_i-theta_j), d_ij=cos(3(theta_i-theta_j))+2b_ij/(r-2).',
        'norm_conjecture_odd_r':'G^10 (r-3)^2 (r-4)/(1800 r^7)',
        'formula_validation':'Different G and gap; additional odd and even finite sizes. This is not a general-size proof.',
        'physical_test':'Retrospective L14,L18 exact Fermi initial state; project out just the normalized symmetric d-b mode in m2, retaining every other symmetry direction.',
        'independent_forecast':False,'canonical_selection':False,
        'interpretation':'A source-dark collective two-pair mode paired with a bright mode in the opposite band. Exchange symmetry forces the symmetric combination; its Schmidt rank alone is not a new phase.'},indent=2))
    formula_checks(args.output)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,(L,args.output)) for L in (14,18)]):f.result()
