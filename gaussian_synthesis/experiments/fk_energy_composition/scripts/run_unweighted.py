#!/usr/bin/env python3
"""Independent M1 validation of certified k2/k4 directions, including rare-tail gates."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
os.environ['NUMBA_NUM_THREADS']='2'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import t as student_t
from fk6.paths import ROOT,PREVIOUS
from fk6.io import write_json,source_manifest
from fk6.coherent import split_vectors,coherent_kernels
from fk4.gpu_tempering import sample_tempered_gpu
from fk4.distributions import full_distribution
from fk4.reference import log_dimension
from fk4.energy_windows import hopping_eigenbasis
from fk.model import build_model

DEVICE=0


def initialize(queue):
    global DEVICE
    DEVICE=queue.get()


def run_task(task):
    t,k,seed,count,sweeps,out=task
    out=Path(out);started=time.monotonic()
    model=build_model()
    old=PREVIOUS.parent/'fk_replica_hierarchy/results'
    canonical=old/f'cartan_k{k}_certified_01'
    frozen=canonical/f't{t:g}_k{k}_frozen.npz'
    saved=np.load(frozen)
    u,target=saved['witness'],saved['target']
    cert=next(r for r in json.loads((canonical/'summary.json').read_text())['rows'] if r['t']==t)
    q=float(np.mean(cert['q_interval']))
    rng=np.random.default_rng(np.random.SeedSequence([seed,610,round(t*10000),k]))
    key=f't{t:g}_k{k}_seed{seed}'
    def progress(row):
        write_json(out/f'{key}_progress.json',dict(status='sampling',pid=os.getpid(),device=DEVICE,**row))
    result=sample_tempered_gpu(u,k-.5,count,rng,device=DEVICE,sweeps=sweeps,callback=progress,
        log_z_anchors={1.:np.log(np.vdot(u,u).real)-log_dimension(8,1)})
    kets=result.kets
    import cupy as cp
    H=model.full_hamiltonian(t)[np.ix_(model.even_masks,model.even_masks)]
    with cp.cuda.Device(DEVICE):
        g=cp.asarray(kets);h=cp.asarray(H)
        energies=cp.asnumpy(cp.einsum('ni,ni->n',g.conj(),g@h.T).real)
    EG=-14-4*t;E0=float(np.vdot(target,H@target).real);corr=EG-E0
    if energies.min()<EG-1e-9:
        raise ArithmeticError('Gaussian energy theorem violated')
    a=kets.conj()@u;phases=a/abs(a);B=q/float(np.mean(abs(a)))
    fractions=np.array([.1,.25,.5,1.]);windows=fractions*corr
    groups,vectors=split_vectors(kets,phases,B,energies-EG,windows,model)
    feature_arrays,quality=full_distribution(kets,model.j_matrices)
    kernel_rows=[];kernel_arrays={}
    # Hopping coordinates are evaluated only as post-hoc analytic reference;
    # neither the sampler nor a discovery routine receives them.
    ref=model.S@hopping_eigenbasis()
    for i,row in enumerate(groups):
        row['correlation_fraction']=float(fractions[i])
        row['coherent_low_norm']=float(np.linalg.norm(vectors['low_rb'][i]))
        row['coherent_high_norm']=float(np.linalg.norm(vectors['high_rb'][i]))
        info,arrays=coherent_kernels(model,t,vectors['low_rb'][i],vectors['high_rb'][i])
        kernel_rows.append(info)
        kernel_arrays.update({f'window{i}_{name}':value for name,value in arrays.items()})
    np.savez(out/f'{key}_particles.npz',kets=kets,aligned_phases=phases,energies=energies,
        energy_excess=energies-EG,features=feature_arrays['features'],purity_errors=feature_arrays['purity_errors'],
        witness=u,target=target,coefficient_mass=B,probability_weight=1/count,windows=windows,
        reference_hopping_basis=ref,reference_target_amplitudes=ref.conj().T@target,
        reference_low_amplitudes=vectors['low_rb']@ref.conj(),reference_high_amplitudes=vectors['high_rb']@ref.conj(),**vectors)
    np.savez_compressed(out/f'{key}_kernels.npz',**kernel_arrays)
    row=dict(t=t,k=k,seed=seed,count=count,device=DEVICE,EG=EG,E0=E0,correlation_energy=corr,
        mass_estimate=B,mass_is_strict_upper=False,mass_estimator='certified_q / sample_P_mean_abs_overlap',
        exact_continuum_reconstruction_bound=cert['full_reconstruction_residual_upper'],
        canonical_certificate=str(canonical/'summary.json'),witness_sha256=hashlib.sha256(u.tobytes()).hexdigest(),
        frozen_file_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
        rho_raw=float(np.linalg.norm(vectors['total_raw']-target)),
        rho_rb=float(np.linalg.norm(vectors['total_rb']-target)),
        conditional_groups=groups,kernels=kernel_rows,gaussian_quality=quality,sampler=result.diagnostics,
        min_observed_gaussian_excess=float(energies.min()-EG),
        fock_repair_cost=0,projected_vector_counted_as_gaussian=False,
        conditional_projection_scope='Exact Spin(7) and time-reversal conditional expectation; H and window invariant',
        elapsed_seconds=time.monotonic()-started,discovery_claimed=False)
    write_json(out/f'{key}.json',row)
    write_json(out/f'{key}_progress.json',dict(status='complete',pid=os.getpid(),elapsed_seconds=row['elapsed_seconds']))
    print(key,'B',B,'rho_RB',row['rho_rb'],'low counts',[r['low_count'] for r in groups],flush=True)
    return row


def aggregate(rows,out):
    groups=[]
    for t,k in sorted(set((r['t'],r['k']) for r in rows)):
        chosen=[r for r in rows if r['t']==t and r['k']==k]
        vectors=[];low=[];high=[]
        for row in chosen:
            f=np.load(out/f't{t:g}_k{k}_seed{row["seed"]}_particles.npz')
            vectors.append(f['total_rb']);low.append(f['low_rb']);high.append(f['high_rb']);target=f['target']
        vectors=np.array(vectors);low=np.array(low);high=np.array(high)
        count=len(chosen);half=None
        if count>=2:
            # Component-wise real/imag intervals; no false simultaneous-coverage claim.
            factor=float(student_t.ppf(.975,count-1))/np.sqrt(count)
            half=np.maximum(low.real.std(axis=0,ddof=1),low.imag.std(axis=0,ddof=1))*factor
        groups.append(dict(t=t,k=k,repeats=count,pooled_rb_residual=float(np.linalg.norm(vectors.mean(axis=0)-target)),
            largest_repeat_rb_residual=max(r['rho_rb'] for r in chosen),
            max_low_component_95_halfwidth=None if half is None else float(half.max()),
            conditional_sampling_pass=all(g['conditional_sampling_pass'] for r in chosen for g in r['conditional_groups']),
            min_low_group_count=min(g['low_count'] for r in chosen for g in r['conditional_groups']),
            reconstruction_pass=bool(count>=5 and np.linalg.norm(vectors.mean(axis=0)-target)<1e-3 and max(r['rho_rb'] for r in chosen)<1e-3),
            zero_count_interval_interpretation='A zero repeat standard deviation for an unobserved group is not an uncertainty bound',
            uncertainty='Empirical Student-t across independent SMC runs; particle ESS is not an independent sample size'))
    return dict(status='complete',rows=rows,groups=groups,
        M1_pass=all(g['reconstruction_pass'] and g['conditional_sampling_pass'] and g['max_low_component_95_halfwidth'] is not None and g['max_low_component_95_halfwidth']<.002 for g in groups),
        empirical_integrals_are_rigorous_certificates=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--t',nargs='+',type=float,default=[11,20]);p.add_argument('--k',nargs='+',type=int,default=[2,4])
    p.add_argument('--seeds',nargs='+',type=int,default=[25101,25102,25103,25104,25105])
    p.add_argument('--devices',nargs='+',type=int,default=list(range(8)))
    p.add_argument('--count',type=int,default=131072);p.add_argument('--sweeps',type=int,default=3)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if not set(args.k)<={2,4}:raise ValueError('M1 runner only supports independently certified orders 2,4')
    write_json(out/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    write_json(out/'manifest.json',source_manifest(out/'source_snapshot'))
    tasks=[(t,k,s,args.count,args.sweeps,str(out)) for t in args.t for k in args.k for s in args.seeds]
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from multiprocessing import get_context
    context=get_context('spawn');queue=context.Queue()
    for device in args.devices:queue.put(device)
    rows=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=len(args.devices),mp_context=context,initializer=initialize,initargs=(queue,)) as pool:
        for future in as_completed([pool.submit(run_task,task) for task in tasks]):
            rows.append(future.result());write_json(out/'partial_summary.json',dict(status='running',completed=len(rows),total=len(tasks),rows=rows))
    result=aggregate(rows,out);result['elapsed_seconds']=time.monotonic()-started
    write_json(out/'summary.json',result)
    print('M1_PASS',result['M1_pass'],'elapsed',result['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
