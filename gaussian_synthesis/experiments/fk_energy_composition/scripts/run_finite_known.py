#!/usr/bin/env python3
"""Paired known-state R/B comparisons using only saved actual Gaussian kets."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk6.paths import ROOT
from fk6.io import source_manifest,write_json
from fk6.finite_benchmarks import take_rows,fit_fidelity,adaptive_known,evaluate
from fk.model import build_model
from fk3.discovery import covariance_from_kets
from fk4.husimi import sample_husimi
from fk4.variational import diverse_order
from fk5.two_orbit import actual_arc_atoms


def worker(task):
    t,seed,out,maximum=task;out=Path(out);model=build_model();target=model.target(t)
    H=model.full_hamiltonian(t)[np.ix_(model.even_masks,model.even_masks)]
    blind=ROOT/'results/m3_blind_directions_01';meta=json.loads((blind/f't{t}_k2_seed{seed}_fraction0.25.json').read_text())
    reference=np.load(blind/(meta['key']+'_frozen.npz'))['reference'];U=meta['budget'];rows=[]
    dictionaries={};rng=np.random.default_rng(np.random.SeedSequence([seed,631]))
    indices=rng.choice(1048576,maximum,replace=False)
    for k in [2,4]:
        source=ROOT/'results/m1_unweighted_n1048576_s5_02'/f't{t}_k{k}_seed{seed}_particles.npz'
        atoms=take_rows(source,'kets',indices);phase=take_rows(source,'aligned_phases',indices)
        data=json.loads(source.with_name(source.name.replace('_particles.npz','.json')).read_text())
        dictionaries[f'canonical_k{k}']=(atoms,phase,dict(source=str(source),sample_indices=indices,
            upstream_sampling_seconds=data['elapsed_seconds'],upstream_optimization_seconds_not_included=True,
            reconstruction_gate_pass=True,selection='uniform paired sample indices'))
    weighted=ROOT/'results/m2_k8_lambda1_validation_n1048576_02'
    source=weighted/f't{t}_k8_lambda1_validation0_seed{seed}.npz'
    selected=np.random.default_rng(np.random.SeedSequence([seed,631])).choice(8192,maximum,replace=False)
    atoms=take_rows(source,'P_kets_subsample',selected)
    original=take_rows(source,'P_subsample_indices',selected)
    coefficients=take_rows(source,'ordinary_coefficients',original)
    data=json.loads(source.with_suffix('.json').read_text())
    trained=json.loads((ROOT/'results/m2_k8_lambda1_pilot_01'/f't{t}_k8_lambda1_frozen.json').read_text())
    training_seconds=sum(s['elapsed_seconds'] for r in trained['training'] for s in r['sampling'])
    dictionaries['weighted_k8_lambda1']=(atoms,coefficients,dict(source=str(source),sample_indices=original,
        upstream_sampling_seconds=data['elapsed_seconds'],upstream_training_worker_seconds=training_seconds,
        upstream_optimization_seconds_not_included=True,
        reconstruction_gate_pass=t==11,selection='uniform paired retained-particle indices'))
    start=time.monotonic();husimi,info=sample_husimi(target,maximum,np.random.default_rng(np.random.SeedSequence([seed,633])))
    dictionaries['Husimi']=(husimi,None,dict(info,upstream_sampling_seconds=time.monotonic()-start))
    start=time.monotonic()
    library=np.concatenate([reference[None,:]]+[actual_arc_atoms(model,float(a)) for a in np.linspace(-.98,.98,65)])
    order=diverse_order(library,maximum)
    dictionaries['physical_prior']=(library[order],None,dict(upstream_sampling_seconds=time.monotonic()-start,
        prior='Fixed mass/hopping family, 65 uniform parameters; full-ket pivoting; no target-dependent parameter',
        uses_exact_target_for_dictionary=False,library_candidates=len(library)))
    for fraction in [.1,.25,.5,1.]:
        m=json.loads((blind/f't{t}_k2_seed{seed}_fraction{fraction:g}.json').read_text())
        library=np.load(blind/(m['key']+'_frozen.npz'))['low_atoms'];order=diverse_order(library,maximum)
        dictionaries[f'low_window_{fraction:g}']=(library[order],None,dict(upstream_sampling_seconds=m['hfb']['elapsed_seconds']+m['low_oracle']['elapsed_seconds'],
            physical_window_fraction=fraction,source='Hamiltonian-only low-energy Gaussian oracle',
            continuous_samples_inside_window=0,empty_sample_deletion_has_no_nonzero_normalized_state=True))
    atoms,info=adaptive_known(target,reference,seed,max_terms=maximum)
    dictionaries['adaptive_ordinary']=(atoms,None,dict(info,upstream_sampling_seconds=info['elapsed_seconds']))
    for factor in [1.,1.1,1.25]:
        B=U*factor;atoms,info=adaptive_known(target,reference,seed,max_terms=maximum,budget=B)
        dictionaries[f'adaptive_cost_{factor:g}']=(atoms,None,dict(info,upstream_sampling_seconds=info['elapsed_seconds'],only_budget_factor=factor))
    for method,(all_atoms,direct,info) in dictionaries.items():
        cov=covariance_from_kets(all_atoms);purity=float(np.max(abs(cov@cov+np.eye(16))))
        if purity>2e-8:raise ArithmeticError('Non-Gaussian benchmark atom: '+method)
        dictfile=out/f't{t}_{method}_seed{seed}_dictionary.npz'
        np.savez_compressed(dictfile,kets=all_atoms,covariances=cov,reference=reference,
                            direct_coefficients=np.array([]) if direct is None else direct)
        write_json(dictfile.with_suffix('.json'),dict(method=method,t=t,seed=seed,purity_error=purity,**info))
        for R in [8,16,32,64]:
            if R>maximum:continue
            atoms=all_atoms[:R]
            for factor in [1.,1.1,1.25]:
                if info.get('only_budget_factor',factor)!=factor:continue
                B=U*factor;methods=['reoptimized']+(['direct'] if direct is not None else [])
                for fit_method in methods:
                    if fit_method=='direct':
                        start=time.monotonic();c=direct[:len(atoms)].copy();c/=np.linalg.norm(c@atoms)
                        fit=dict(elapsed_seconds=time.monotonic()-start,method='original relative phases and weights, normalized')
                    else:c,fit=fit_fidelity(atoms,target,B)
                    metrics=evaluate(atoms,c,target,H,model,B)
                    key=f't{t}_{method}_{fit_method}_seed{seed}_R{R}_B{factor:g}'
                    np.savez_compressed(out/(key+'.npz'),kets=atoms,coefficients=c,coefficient_phases=np.exp(1j*np.angle(c)),state=c@atoms)
                    row=dict(key=key,t=t,seed=seed,method=method,coefficient_method=fit_method,R_budget=R,
                        budget_factor=factor,metrics=metrics,fit=fit,offline=info,atom_purity_error=purity,
                        permission='known-state',all_methods_allowed_same_target=True,
                        numerical_comparison_not_global_optimality_certificate=True,
                        full_offline_cost_accounting_complete=not info.get('upstream_optimization_seconds_not_included',False))
                    rows.append(row);write_json(out/(key+'.json'),row)
        print('finished',t,seed,method,flush=True)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--t',type=int,nargs='+',default=[11,20]);p.add_argument('--seeds',type=int,nargs='+',default=[25101,25102,25103,25104,25105])
    p.add_argument('--workers',type=int,default=4);p.add_argument('--maximum',type=int,default=64)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from multiprocessing import get_context
    rows=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
        jobs=[pool.submit(worker,(t,s,str(args.output),args.maximum)) for t in args.t for s in args.seeds]
        for future in as_completed(jobs):
            rows.extend(future.result());write_json(args.output/'partial_summary.json',dict(status='running',rows=rows))
    write_json(args.output/'summary.json',dict(status='complete_pending_independent_audit',rows=rows,
        elapsed_seconds=time.monotonic()-start,information_mode='known-state',G3_pass_claimed=False,
        Hamiltonian_only_comparison_still_required=True,full_offline_accounting_still_required=True))


if __name__=='__main__':main()
