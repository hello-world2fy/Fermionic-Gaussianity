#!/usr/bin/env python3
"""Freeze actual Gaussian kets before constructing any perturbed Hamiltonian.

Training uses epsilon=0 only. Column/P candidates are drawn at matched pool
size; all selection uses a target-independent column-pivoted QR. The test-state
Husimi oracle is deliberately absent from this preparation stage.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('NUMBA_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4 import paths
from fk4.io import source_manifest,write_json
from fk4.variational import diverse_order
from fk4.finite_twirl import FiniteSpin7Twirl
from fk4.distributions import full_distribution
from fk4.costs import recorded_source_cost
from fk4.columns import independent_arc_dictionary
from fk.gaussian import sample_gaussian
from fk.model import build_model
from fk3.discovery import optimize_gaussian_overlap


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--columns',type=Path,required=True)
    p.add_argument('--replica-runs',type=Path,nargs='+',required=True)
    p.add_argument('--t',type=float,nargs='+',default=[9,10,10.9225,11,12,20])
    p.add_argument('--replicates',type=int,nargs='+',default=[1,2,3,4,5])
    p.add_argument('--candidates',type=int,default=128)
    p.add_argument('--maximum-terms',type=int,default=64)
    p.add_argument('--include-exploratory',action='store_true')
    args=p.parse_args()
    if args.candidates!=128:
        raise ValueError('this preregistered prior pool uses 16 arcs times 8 actual branches =128')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    write_json(args.output/'config.json',{key:[str(v) for v in value] if key=='replica_runs' else str(value) if isinstance(value,Path) else value for key,value in vars(args).items()})
    prior_started=time.monotonic()
    model=build_model()
    twirl=FiniteSpin7Twirl(model)
    prior,_=independent_arc_dictionary(model,129)
    prior_indices=np.linspace(1,len(prior)-2,16).round().astype(int)
    prior_pool=np.concatenate([twirl.orbit(prior[j])[0] for j in prior_indices])
    prior_generation_seconds=time.monotonic()-prior_started
    column_summary=json.loads((args.columns/'summary.json').read_text())
    assert json.loads((args.columns/'verification.json').read_text())['artifact_verification_passed']
    sources=[]
    source_ledgers={}
    for directory in args.replica_runs:
        summary=json.loads((directory/'summary.json').read_text())
        assert summary['status']=='complete'
        if not summary['next_order_gate_pass'] and not args.include_exploratory:
            raise ValueError(f'{directory}: reconstruction gate has not passed')
        sources.append((directory,summary,json.loads((directory/'config.json').read_text())))
        source_ledgers[str(directory)]=recorded_source_cost(directory)
    np.testing.assert_array_equal(prior,np.load(args.columns/'model_and_prior.npz')['arc_atoms'])
    if len(prior_pool)!=128:
        raise ArithmeticError('prior pool does not have 128 actual Gaussian branches')
    rows=[]
    started=time.monotonic()
    for t in args.t:
        for replicate in args.replicates:
            rng=np.random.default_rng(np.random.SeedSequence([28100+replicate,801,int(round(t*10000))]))
            tick=time.monotonic()
            haar=sample_gaussian(8,args.candidates,rng).kets
            pools={'random':(haar,{'generation_seconds':time.monotonic()-tick,'source_generation_seconds_cold':0.})}
            pools['physics_prior']=(prior_pool[rng.permutation(len(prior_pool))],
                    {'generation_seconds':0.,'source_generation_seconds_cold':prior_generation_seconds,
                     'source_cost_scope':'Independent analytic prior, including model/group initialization and all 128 actual branches; no column optimization charged'})
            source_key=f'columns_t{t:g}_seed{25100+replicate}'
            src=np.load(args.columns/f'{source_key}.npz')
            probability=abs(src['finite_coefficients'])
            probability/=probability.sum()
            chosen=rng.choice(len(probability),size=args.candidates,replace=False,p=probability)
            pools['dual_columns']=(src['finite_kets'][chosen],{'generation_seconds':0.,'source_key':source_key,
                 'source_generation_seconds_cold':column_summary['elapsed_seconds'],
                 'source_cost_scope':'Full prior/column run; conservative shared cold-start cost',
                 'orbit_representatives_counted_as_one':False})
            guided,search=optimize_gaussian_overlap(model.target(t),haar,max_sweeps=60,gradient_tolerance=1e-9)
            pools['husimi_train_frozen']=(guided,{'generation_seconds':search['elapsed_seconds'],
                 'source_generation_seconds_cold':0.,'training_target_used':True,'test_target_used':False,
                 'search_stationarity_max':float(np.max(abs(search['gradient_final_normalized_witness']))),
                 'search_globally_optimal_proved':False})
            for directory,summary,config in sources:
                row=next(r for r in summary['rows'] if r['t']==t)
                k=row['k']
                if 'observations' in row:
                    observation=row['observations'][replicate-1]
                    source_file=directory/f"t{t:g}_k{k}_seed{observation['seed']}.npz"
                    sample=np.load(source_file)
                    if 'source_kets_file' in observation:
                        allg=np.load(observation['source_kets_file'])[observation['source_kets_key']]
                        probability=sample['weights']
                    else:
                        allg=sample['P_kets']
                        probability=None
                else:
                    observation=row['validation_rows'][replicate-1]
                    source_file=directory/f"t{t:g}_k{k}_validation_seed{observation['seed']}.npz"
                    if 'P_source_file' in observation:
                        source_file=Path(observation['P_source_file'])
                        assert hashlib.sha256(source_file.read_bytes()).hexdigest()==observation['P_source_sha256']
                    allg=np.load(source_file)[observation.get('P_source_key','P_all_kets')]
                    probability=None
                indices=rng.choice(len(allg),args.candidates,replace=False,p=probability)
                # Include measured matrix/witness construction as a shared cold
                # cost, in addition to the complete upstream sampling run.
                ledger=source_ledgers[str(directory)]
                pools[f'replica_k{k}_P']=(allg[indices],{'generation_seconds':0.,
                      'source_generation_seconds_cold':ledger['recorded_seconds'],
                      'source_cost_ledger':ledger,
                      'source_cost_scope':'Complete recorded upstream dependency graph; exact-moment construction and previous-order initialization counted once per cold start',
                      'source_run':str(directory.resolve()),'source_file':str(source_file.resolve()),
                      'source_reconstruction_gate_pass':summary['next_order_gate_pass'],
                      'source_particle_indices':indices.tolist(),'sampling_kind':'canonical coefficient P, with raw Gaussian phases retained'})
            for method,(candidates,metadata) in pools.items():
                selection_started=time.monotonic()
                order=diverse_order(candidates,args.maximum_terms)
                states=candidates[order]
                features,quality=full_distribution(states,model.j_matrices)
                name=f't{t:g}_rep{replicate}_{method}'
                filename=args.output/f'{name}.npz'
                np.savez_compressed(filename,atoms=states,candidate_atoms=candidates,selected_indices=order,
                                    features=features['features'],paired_haar_starts=haar)
                selection_seconds=time.monotonic()-selection_started
                entry={'t':t,'replicate':replicate,'method':method,'key':name,
                       'candidate_count':len(candidates),'actual_gaussian_terms_available':len(states),
                       'no_fock_completion_atoms_added':True,'selection':'column-pivoted QR on full actual kets',
                       'selection_seconds':selection_seconds,'Gaussian_quality':quality,
                       'uses_perturbed_target':False,'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       'file_sha256':hashlib.sha256(filename.read_bytes()).hexdigest(),**metadata}
                rows.append(entry)
            write_json(args.output/'summary.json',{'status':'running','rows':rows})
            print('frozen',t,replicate,','.join(pools),flush=True)
    write_json(args.output/'summary.json',{'status':'complete','rows':rows,'elapsed_seconds':time.monotonic()-started,
               'analytic_prior_setup_seconds':prior_generation_seconds,
               'all_dictionaries_frozen_before_test_hamiltonians':True,'actual_branch_counts_used':True,
               'test_oracle_not_included':True})


if __name__=='__main__':
    main()
