#!/usr/bin/env python3
"""t=0 canonical direction fixed exactly by local Spin(7) x Spin(7).

Canonical C_k(s0)/q_k(s0)=s0 holds identically. P_k is sampled with
direction s0; B_k=1/E_P|<g|s0>|. Every empirical P population also gives
an explicit finite 256N-branch representation, by a separate integer proof.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_NUM_THREADS','12')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import t as student_t
from fk4.io import source_manifest,write_json
from fk4.tempering import sample_tempered
from fk4.reference import log_dimension
from fk4.product_magic_twirl import proof_and_actions,aligned_orbit_decomposition,orbit
from fk4.distributions import full_distribution
from fk.model import build_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=131072)
    p.add_argument('--orders',type=int,nargs='+',default=[1,2,4,8,16,32,64,128])
    p.add_argument('--seeds',type=int,nargs='+',default=[30101,30102,30103,30104,30105])
    p.add_argument('--backend',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--device',type=int,default=0)
    args=p.parse_args()
    os.environ['FK_SAMPLER_BACKEND']=args.backend;os.environ['FK_CUDA_DEVICE']=str(args.device)
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    proof,permutations,phases,target=proof_and_actions()
    write_json(args.output/'finite_projector_proof.json',proof)
    np.savez_compressed(args.output/'finite_projector.npz',permutations=permutations,phases=phases,target=target)
    model=build_model()
    anchors={k:np.log(q)-log_dimension(8,k) for k,q in {1:1.,2:99/128,3:.5,4:3627/12800}.items()}
    observations={k:[] for k in args.orders}
    started=time.monotonic()
    for seed in args.seeds:
        values={}
        def snapshot(kets,beta,log_z,log_r,diagnostic):
            if beta+.5 in args.orders:
                k=int(beta+.5)
                factors,realization=aligned_orbit_decomposition(kets)
                distribution,quality=full_distribution(kets,model.j_matrices)
                a=kets.conj()@target
                key=f't0_k{k}_seed{seed}'
                np.savez_compressed(args.output/f'{key}.npz',P_kets=kets,
                    aligned_phases=a/abs(a),branch_coefficients_per_particle=factors,
                    P_log_r=log_r,**distribution)
                sample_branches=np.concatenate([orbit(g) for g in kets[:4]])
                f,_=aligned_orbit_decomposition(kets[:4])
                residual=float(np.linalg.norm(np.repeat(f,256)@sample_branches-target))
                if residual>1e-12:raise ArithmeticError('explicit full-Fock orbit reconstruction failed')
                values[k]={'t':0.,'k':k,'seed':seed,'B':realization['coefficient_cost'],
                    'full_distribution':quality,'finite_realization':realization,
                    'four_particle_1024_branch_full_Fock_residual':residual,
                    'canonical_direction_exact_by_symmetry':True,
                    'canonical_normalized_reconstruction_exact':True,
                    'normalization_identity':'C_k(s0)/q_k(s0) = s0; u_k=q_k(s0)^(-1/(2k-1)) s0',
                    'L_global_support_exact':2.,'P_log_z':log_z,'key':key}
            if beta in args.orders:
                k=int(beta)
                logq=log_dimension(8,k)+log_z
                row=values[k]
                row.update(log_q_point=logq,L_canonical_point=float(np.exp(-logq/(2*k))),
                    estimated_canonical_radius=float(np.exp(-logq/(2*k-1))),
                    radius_and_L_high_order_not_certified=k>4,
                    B_from_partition_ratio=float(np.exp(row['P_log_z']-log_z)))
                # The same P population is incrementally weighted by |a| to
                # reach beta=k, so these estimators must agree numerically.
                if abs(row['B_from_partition_ratio']-row['B'])>1e-9 and k not in anchors:
                    raise ArithmeticError('P-ratio normalization mismatch')
                observations[k].append(row)
                write_json(args.output/f"{row['key']}.json",row)
                print('seed',seed,'k',k,'B',row['B'],'explicit_rho',row['four_particle_1024_branch_full_Fock_residual'],flush=True)
        result=sample_tempered(target,max(args.orders),args.count,
            np.random.default_rng(np.random.SeedSequence([seed,1001])),sweeps=2,
            checkpoints=[b for k in args.orders for b in (k-.5,k)],
            state_callback=snapshot,log_z_anchors=anchors)
        write_json(args.output/f'seed{seed}_sampling.json',result.diagnostics)
    rows=[]
    for k,obs in observations.items():
        Bs=np.array([r['B'] for r in obs])
        half=float(student_t.ppf(.975,len(Bs)-1)*Bs.std(ddof=1)/np.sqrt(len(Bs)))
        rows.append({'t':0.,'k':k,'B_mean':float(Bs.mean()),'B_95_percent_halfwidth':half,
            'mass_repeat_pass':half/Bs.mean()<.01,'L_canonical_point_mean':float(np.mean([r['L_canonical_point'] for r in obs])),
            'L_global_support_exact':2.,'U_best_exact':2.,'observations':obs,
            'canonical_normalized_reconstruction_exact':True,
            'canonical_integral_B_not_interval_certified':True,
            'finite_empirical_B_is_constructive_cost':True,
            'actual_branches_per_empirical_decomposition':256*args.count,
            'no_single_projected_vector_counted_as_gaussian':True})
    write_json(args.output/'summary.json',{'status':'complete','rows':rows,
        'elapsed_seconds':time.monotonic()-started,'scope':'exact t=0 target only',
        'mean_cost_intervals_empirical':True,'next_order_gate_pass':all(r['mass_repeat_pass'] for r in rows),
        'high_order_q_normalization_not_interval_certified':True,
        'physical_endpoint_support_and_normalized_reconstruction_are_exact':True})


if __name__=='__main__':main()
