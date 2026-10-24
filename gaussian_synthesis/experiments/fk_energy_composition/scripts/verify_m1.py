#!/usr/bin/env python3
"""Independent raw-ket reconstruction, Clifford-kernel and rare-mass audit."""
from pathlib import Path
from itertools import combinations
from math import factorial
from fractions import Fraction
import argparse
import hashlib
import json
import os
import sys
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import mp
from fk6.io import write_json
from fk6.paths import ROOT
from fk.model import build_model
from fk4.tempering import actions
from fk3.discovery import covariance_from_kets


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def verify_run(task):
    path,row=task;path=Path(path);model=build_model();t=row['t'];k=row['k'];seed=row['seed']
    key=f't{t:g}_k{k}_seed{seed}';particle_file=path/f'{key}_particles.npz'
    f=np.load(particle_file);g=f['kets'];u=f['witness'];target=f['target'];phases=f['aligned_phases'];B=float(f['coefficient_mass'])
    a=g.conj()@u;assert np.max(abs(phases-a/abs(a)))<2e-14
    assert np.max(abs(np.linalg.norm(g,axis=1)-1))<2e-10
    C=B*(phases@g)/len(g);assert np.linalg.norm(C-f['total_raw'])<3e-13
    # T acts by complex conjugation in the independently reconstructed S basis.
    Crb=model.S@((model.S.conj().T@C).real)
    assert np.linalg.norm(Crb-f['total_rb'])<3e-13
    rho=float(np.linalg.norm(Crb-target));assert abs(rho-row['rho_rb'])<2e-13
    assert rho<.001
    indices=np.linspace(0,len(g)-1,512,dtype=int);sample=g[indices]
    cov=covariance_from_kets(sample);purity=float(np.max(abs(cov@cov+np.eye(16))))
    assert purity<2e-10
    H=model.full_hamiltonian(t)[np.ix_(model.even_masks,model.even_masks)]
    energies=np.einsum('ni,ij,nj->n',sample.conj(),H,sample).real
    assert np.max(abs(energies-f['energies'][indices]))<3e-10
    P=model.S@model.S.conj().T;assert np.max(abs(P@H-H@P))<1e-12
    q=np.load(path/f'{key}_kernels.npz');perms,factors=actions(8)
    pairs=list(combinations(range(16),2));lookup={pair:i for i,pair in enumerate(pairs)}
    kernel_count=0
    for window in range(4):
        low=f['low_rb'][window];high=f['high_rb'][window];v=np.stack([low,high])
        assert np.linalg.norm(v.sum(axis=0)-Crb)<3e-13
        K2=np.array([v.conj()@(1j*v[:,perm]*factor).T for perm,factor in zip(perms,factors)])
        K4=[]
        for a,b,c,d in combinations(range(16),4):
            first=lookup[(a,b)];second=lookup[(c,d)]
            acted=v[:,perms[second]]*factors[second]
            acted=acted[:,perms[first]]*factors[first]
            K4.append(v.conj()@acted.T)
        K4=np.array(K4)
        assert np.max(abs(K2-q[f'window{window}_two_kernels']))<2e-13
        assert np.max(abs(K4-q[f'window{window}_four_kernels']))<2e-13
        assert np.max(abs(v.conj()@H@v.T-q[f'window{window}_energy']))<2e-11
        kernel_count+=len(K2)+len(K4)
    result=dict(t=t,k=k,seed=seed,raw_ket_count=len(g),all_norms_and_phase_reconstruction_checked=True,
        rho_rb=rho,independent_purity_and_energy_subset=len(indices),maximum_subset_purity_error=purity,
        independently_recomputed_kernel_matrices=kernel_count,particle_sha256=file_hash(particle_file))
    print('verified',key,'rho',rho,flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--unweighted',type=Path,required=True)
    p.add_argument('--rare',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    data=json.loads((args.unweighted/'summary.json').read_text());rare=json.loads((args.rare/'summary.json').read_text())
    assert data['status']==rare['status']=='complete'
    assert len(data['rows'])==20 and len(rare['rows'])==40
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn')) as pool:
        rows=list(pool.map(verify_run,[(str(args.unweighted),r) for r in data['rows']]))
    # Independent factorial form of the polynomial-integral normalization.
    C=Fraction(1,factorial(28))
    for j in range(4):C*=Fraction(factorial(7+2*j),factorial(2*j))
    assert C==Fraction(1,23178480)
    mp.dps=100;tail_rows=[]
    for r in rare['rows']:
        assert r['hits']==0  # Nonzero hits need general binomial inversion in this independent verifier.
        alpha=mp.mpf('0.00000001');probability=-mp.expm1(mp.log(alpha)/r['count'])
        for w in r['window_results']:
            bound=float(np.nextafter(float(probability*mp.mpf(w['global_integrand_envelope'])),np.inf))
            assert bound<.002
            tail_rows.append(dict(t=r['t'],k=w['k'],fraction=r['fraction'],seed=r['seed'],
                low_vector_and_mass_upper=bound,failure_probability=float(alpha),
                confidence_under_ideal_IID_proposal=True,conditional_shape_resolved=False))
    result=dict(passed=True,M1_numerical_integration_gate_pass=True,
        reconstructed_runs=rows,total_raw_gaussian_kets=sum(r['raw_ket_count'] for r in rows),
        total_independently_checked_kernel_matrices=sum(r['independently_recomputed_kernel_matrices'] for r in rows),
        rare_contribution_bounds=tail_rows,max_rare_vector_bound=max(r['low_vector_and_mass_upper'] for r in tail_rows),
        rare_joint_failure_probability_upper=len(rare['rows'])*1e-8,
        rare_confidence_not_deterministic_interval_certificate=True,
        tail_shape_not_measured=True,all_gaussian_energy_values_independently_rechecked=False,
        no_fock_repair=True,no_projected_vectors_counted_as_gaussian_atoms=True,
        G1_unprompted_discovery_pass=False,G2_distribution_to_theorem_pass=False,G3_independent_gain_pass=False,
        source_summary_sha256=file_hash(args.unweighted/'summary.json'),rare_summary_sha256=file_hash(args.rare/'summary.json'),
        interpretation='Numerical integration is adequate at the declared absolute precision; negligible windows have probability-bounded contributions, not measured conditional shapes. Full scientific milestones remain separate.')
    write_json(args.output/'verification.json',result)


if __name__=='__main__':main()
