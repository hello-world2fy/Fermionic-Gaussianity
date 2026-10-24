#!/usr/bin/env python3
"""Frozen-direction weighted synthesis, independent validation and explicit failure gates."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import t as student_t
from fk6.paths import ROOT,PREVIOUS
from fk6.io import source_manifest,write_json
from fk6.weighted import EnergyWeightedMoments
from fk6.weighted_sampler import sample_weighted
from fk6.coherent import rb_vector
from fk4.symmetry import TimeReversalAveraged
from fk4.solver import solve
from fk4.reference import log_dimension
from fk4.distributions import full_distribution
from fk.model import build_model

DEVICE=0


def initialize(queue):
    global DEVICE
    DEVICE=queue.get()


def worker(task):
    stage,t,k,penalty,seed,iteration,count,moves,w,out,compact=task;out=Path(out);started=time.monotonic()
    model=build_model();target=model.target(t);ids=model.even_masks;H=model.full_hamiltonian(t)[np.ix_(ids,ids)]
    EG=-14-4*t;E0=float(np.vdot(target,H@target).real);corr=EG-E0
    key=f't{t:g}_k{k}_lambda{penalty:g}_{stage}{iteration}_seed{seed}'
    rng=np.random.default_rng(np.random.SeedSequence([seed,612 if stage=='train' else 613,round(t*10000),k,round(penalty*10000),iteration]))
    def progress(row):write_json(out/f'{key}_progress.json',dict(stage=stage,pid=os.getpid(),device=DEVICE,**row))
    values={};arrays={}
    def snapshot(g,beta,log_z,log_r,diag):
        if stage=='train' or abs(beta-(k-.5))>1e-10:return
        a=g.conj()@w;energy=np.einsum('ni,ni->n',g.conj(),g@H.T).real
        e=(energy-EG)/corr;v=1+penalty*e;s=float(np.vdot(w,target).real)
        Bv=s/float(np.mean(abs(a)/v));coeff=Bv*(a/abs(a))/v/len(g)
        C=coeff@g;plain=float(np.sum(abs(coeff)));A=float(abs(coeff)@e)
        windows=np.array([.1,.25,.5,1.]);low=np.array([coeff[e<=x]@g[e<=x] for x in windows]);high=C-low
        features,quality=full_distribution(g,model.j_matrices)
        values.update(B_weighted=Bv,B_plain=plain,composition_energy=A,
            cost_identity_error=abs(Bv-plain-penalty*A),rho_P_raw=float(np.linalg.norm(C-target)),
            rho_P_rb=float(np.linalg.norm(rb_vector(C,model)-target)),
            low_group_counts=[int(np.sum(e<=x)) for x in windows],
            weighted_low_probabilities=[float(np.mean(e<=x)) for x in windows],
            plain_low_probabilities=[float(np.sum(abs(coeff[e<=x]))/plain) for x in windows],
            mean_weighted_e=float(np.mean(e)),mean_plain_e=A/plain,quality=quality)
        arrays.update(phases=a/abs(a),ordinary_coefficients=coeff,weighted_probabilities=np.full(len(g),1/len(g)),
            energy_excess_dimensionless=e,v=v,features=features['features'],purity_errors=features['purity_errors'],
            low_raw=low,high_raw=high,low_rb=np.array([rb_vector(x,model) for x in low]),
            high_rb=np.array([rb_vector(x,model) for x in high]),C_P_raw=C,C_P_rb=rb_vector(C,model))
        if compact:
            retention=np.random.default_rng(np.random.SeedSequence([seed,614,k,round(t*10000),round(penalty*10000)]))
            keep=retention.choice(len(g),min(8192,len(g)),replace=False)
            arrays.update(P_kets_subsample=g[keep],P_subsample_indices=keep,P_singlet_overlaps=g.conj()@model.S)
        else:arrays['P_kets']=g
        values['stored_actual_P_gaussian_count']=min(8192,len(g)) if compact else len(g)
        values['full_P_kets_saved']=not compact
    result=sample_weighted(w,H,EG,corr,penalty,k,count,rng,device=DEVICE,moves=moves,
        callback=progress,checkpoints=[k-.5] if stage=='validation' else [],state_callback=snapshot)
    g=result.kets
    if stage=='train':
        energy=np.einsum('ni,ni->n',g.conj(),g@H.T).real
        h=g.conj()@model.S;e=np.maximum((energy-EG)/corr,0)
        logs=result.log_z-k*result.log_r-np.log(len(g))
        np.savez_compressed(out/f'{key}.npz',overlaps=h,energies=e,log_measure_weights=logs,witness=w)
        row=dict(stage=stage,t=t,k=k,penalty=penalty,seed=seed,iteration=iteration,sampler=result.diagnostics,
                 elapsed_seconds=time.monotonic()-started,quadrature_file=f'{key}.npz')
    else:
        a=g.conj()@w;s=float(np.vdot(w,target).real)
        C=s*np.mean(g/a.conj()[:,None],axis=0);Crb=rb_vector(C,model)
        row=dict(stage=stage,t=t,k=k,penalty=penalty,seed=seed,count=count,
            rho_gradient_raw=float(np.linalg.norm(C-target)),rho_gradient_rb=float(np.linalg.norm(Crb-target)),
            implicit_exact_radial_normalization=True,numeric_radial_scale_certified=False,
            weighted_support_bound_certified=False,L_from_MC_claimed=False,sampler=result.diagnostics,
            elapsed_seconds=time.monotonic()-started,**values)
        if compact:
            arrays['gradient_kets_subsample']=g[:512]
            arrays['gradient_singlet_overlaps']=g.conj()@model.S
            arrays['gradient_log_r']=result.log_r
        else:arrays['gradient_kets']=g
        np.savez(out/f'{key}.npz',witness=w,target=target,C_gradient_raw=C,C_gradient_rb=Crb,**arrays)
    write_json(out/f'{key}.json',row)
    print(key,'elapsed',row['elapsed_seconds'],'rho',row.get('rho_gradient_rb'),flush=True)
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--t',type=float,nargs='+',default=[11,20]);p.add_argument('--k',type=int,default=2)
    p.add_argument('--penalties',type=float,nargs='+',default=[.25])
    p.add_argument('--count',type=int,default=32768);p.add_argument('--validation-count',type=int,default=32768)
    p.add_argument('--rounds',type=int,default=2);p.add_argument('--moves',type=int,default=48)
    p.add_argument('--seeds',type=int,nargs='+',default=[25101,25102,25103,25104,25105])
    p.add_argument('--devices',type=int,nargs='+',default=list(range(8)))
    p.add_argument('--initial',type=Path)
    p.add_argument('--frozen',type=Path,help='Independently validate previously frozen directions without retraining')
    p.add_argument('--compact-particles',action='store_true',help='Save all energies/phases/features/projected overlaps and 8192 actual P kets; preserve full reconstruction vectors and seeds')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    gate=ROOT/'results/m1_verification_01/verification.json'
    if not json.loads(gate.read_text())['M1_numerical_integration_gate_pass']:raise ArithmeticError('M1 gate not passed')
    write_json(out/'manifest.json',source_manifest(out/'source_snapshot'))
    write_json(out/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    context=get_context('spawn');queue=context.Queue()
    for device in args.devices:queue.put(device)
    model=build_model();summaries=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=len(args.devices),mp_context=context,initializer=initialize,initargs=(queue,)) as pool:
        for t in args.t:
            initial=args.initial or PREVIOUS.parent/f'fk_replica_hierarchy/results/cartan_k{min(args.k,4)}_certified_01'
            z=np.load(initial/f't{t:g}_k{min(args.k,4)}_frozen.npz')['z'];z=z/np.linalg.norm(z)
            target=model.S.conj().T@model.target(t)
            for penalty in args.penalties:
                training=[]
                key=f't{t:g}_k{args.k}_lambda{penalty:g}'
                if args.frozen:
                    z=np.load(args.frozen/f'{key}_frozen.npz')['z']
                exact_zero_source=None
                if penalty==0 and args.k in (2,4) and not args.frozen:
                    exact_zero_source=PREVIOUS.parent/f'fk_replica_hierarchy/results/cartan_k{args.k}_certified_01/t{t:g}_k{args.k}_frozen.npz'
                    z=np.load(exact_zero_source)['z'];z=z/np.linalg.norm(z)
                for iteration in range(0 if args.frozen or exact_zero_source is not None else args.rounds):
                    w=model.S@z
                    work=[('train',t,args.k,penalty,s,iteration,args.count,args.moves,w,str(out),args.compact_particles) for s in args.seeds]
                    rows=list(pool.map(worker,work));hs=[];es=[];logs=[]
                    for row in rows:
                        data=np.load(out/row['quadrature_file']);hs.append(data['overlaps']);es.append(data['energies']);logs.append(data['log_measure_weights']-np.log(len(rows)))
                    backend=EnergyWeightedMoments(np.concatenate(hs),np.concatenate(logs),np.concatenate(es),penalty,args.k,log_dimension(8,args.k))
                    solution=solve(TimeReversalAveraged(backend),target,initial=z)
                    if not solution['converged']:raise ArithmeticError('Weighted empirical Newton failed: '+solution['status'])
                    z=solution['z'];z=z/np.linalg.norm(z);z*=np.exp(1j*np.angle(np.vdot(z,target)))
                    training.append(dict(optimizer={name:value for name,value in solution.items() if name!='z'},sampling=rows))
                w=model.S@z
                np.savez(out/f'{key}_frozen.npz',z=z,witness=w,target=model.target(t))
                write_json(out/f'{key}_frozen.json',dict(witness_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
                    training=training,validation_has_not_started=True,previous_frozen_source=None if args.frozen is None else str(args.frozen.resolve()),
                    exact_zero_penalty_source=None if exact_zero_source is None else str(exact_zero_source.resolve())))
                rows=list(pool.map(worker,[('validation',t,args.k,penalty,s,0,args.validation_count,args.moves,w,str(out),args.compact_particles) for s in args.seeds]))
                costs=np.array([[r['B_weighted'],r['B_plain'],r['composition_energy']] for r in rows])
                half=student_t.ppf(.975,len(rows)-1)*costs.std(axis=0,ddof=1)/np.sqrt(len(rows)) if len(rows)>1 else np.full(3,np.nan)
                record=dict(t=t,k=args.k,penalty=penalty,repeats=len(rows),mean_costs_weighted_plain_energy=costs.mean(axis=0),
                    cost_95_halfwidths=None if len(rows)<2 else half,validation_rows=rows,
                    reconstruction_pass=all(r['rho_gradient_rb']<.001 and r['rho_P_rb']<.001 for r in rows),
                    global_weighted_certificate_pass=False,M2_scientific_gate_pass=False,
                    conditional_tail_integrals_pending=True,training=training,
                    previous_frozen_source=None if args.frozen is None else str(args.frozen.resolve()),
                    exact_zero_penalty_source=None if exact_zero_source is None else str(exact_zero_source.resolve()),
                    compact_particle_storage=args.compact_particles)
                summaries.append(record);write_json(out/'summary.json',dict(status='running',rows=summaries))
    write_json(out/'summary.json',dict(status='complete',rows=summaries,elapsed_seconds=time.monotonic()-started,
        M2_scientific_gate_pass=False,weighted_curves_alone_are_not_success=True))


if __name__=='__main__':main()
