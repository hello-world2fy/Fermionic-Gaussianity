#!/usr/bin/env python3
"""Frozen-dictionary E4/E5 evaluation; ED follows all non-oracle predictions."""
from pathlib import Path
import argparse
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
from fk4.variational import perturbed_hamiltonian,predict,coherent_kernels,coherent_and_incoherent_metrics,optimize_gaussian_energy,diverse_order
from fk4.distributions import full_distribution
from fk4.husimi import sample_husimi
from fk.model import build_model
from fk.observables import exact_metrics
from fk3.discovery import covariance_from_kets,optimize_gaussian_overlap


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--t',type=float,nargs='+',default=[9,10,10.9225,11,12,20])
    p.add_argument('--epsilon',type=float,nargs='+',default=[0,.02,.05,.1,.2])
    p.add_argument('--budgets',type=int,nargs='+',default=[1,2,4,8,16,32,64])
    p.add_argument('--replicates',type=int,nargs='+',default=[1,2,3,4,5])
    p.add_argument('--no-test-oracle',action='store_true')
    p.add_argument('--hfb-starts',type=int,default=32)
    p.add_argument('--oracle-device',type=int,default=None)
    args=p.parse_args()
    frozen=json.loads((args.frozen/'summary.json').read_text())
    if frozen['status']!='complete':
        raise ValueError('finish freezing every requested dictionary before any test ED')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    model=build_model()
    masks=model.even_masks
    mass=model.mass_correlation_operator[np.ix_(masks,masks)]
    operators={'mass_correlation':mass}
    for indices in [(0,1,8,9),(0,2,8,10),(0,1,2,3)]:
        O=np.eye(256,dtype=complex)
        for j in indices:
            O=O@model.gammas[j]
        operators['fourpoint_'+'_'.join(map(str,indices))]=O[np.ix_(masks,masks)]
    loaded={}
    for entry in frozen['rows']:
        if entry['t'] not in args.t or entry['replicate'] not in args.replicates:
            continue
        path=args.frozen/f"{entry['key']}.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['file_sha256']
        data=np.load(path)
        atoms=data['atoms']
        loaded[entry['key']]={'entry':entry,'atoms':atoms,'covariances':covariance_from_kets(atoms),
                              'starts':data['paired_haar_starts'],'kernels':coherent_kernels(atoms,operators)}
    # All hashes and actual kets are loaded before constructing a new H.
    rows=[]
    targets=[]
    started=time.monotonic()
    for t in args.t:
        for epsilon in args.epsilon:
            H=perturbed_hamiltonian(model,t,epsilon)
            predicted=[]
            def append_prediction(method,replicate,atoms,budget,metadata,*,cov=None,kernels=None):
                states=atoms[:budget]
                report,data=predict(states,H)
                Ks=coherent_kernels(states,{'hamiltonian':H,**operators}) if kernels is None else {name:K[:budget,:budget] for name,K in kernels.items()}
                Ks['hamiltonian']=data['hamiltonian_kernel']
                metrics,_=coherent_and_incoherent_metrics(states,data['coefficients'],H,model,
                                  kernels=Ks,covariances=None if cov is None else cov[:budget])
                metadata={**metadata,'prediction_seconds':report['elapsed_seconds']}
                key=f't{t:g}_eps{epsilon:g}_rep{replicate}_{method}_m{len(states)}'
                np.savez_compressed(args.output/f'{key}.npz',state=data['state'],coefficients=data['coefficients'],
                                     atoms=states,**Ks)
                predicted.append(({'t':t,'epsilon':epsilon,'replicate':replicate,'method':method,
                                   'budget':budget,'key':key,**report,**metrics,**metadata},data['state']))
            for cache in loaded.values():
                entry=cache['entry']
                if entry['t']!=t:
                    continue
                for budget in args.budgets:
                    if budget>len(cache['atoms']):
                        continue
                    append_prediction(entry['method'],entry['replicate'],cache['atoms'],budget,
                       {'test_target_used_for_candidates':False,'frozen_source_key':entry['key'],
                        'frozen_source_sha256':entry['file_sha256'],
                        'source_generation_seconds_cold':entry['source_generation_seconds_cold'],
                        'dictionary_generation_seconds':entry['generation_seconds'],
                        'dictionary_selection_seconds':entry['selection_seconds']},
                       cov=cache['covariances'],kernels=cache['kernels'])
            for replicate in args.replicates:
                base=next(c for c in loaded.values() if c['entry']['t']==t and c['entry']['replicate']==replicate)
                states,hfb=optimize_gaussian_energy(H,base['starts'][:args.hfb_starts],sweeps=80)
                index=int(np.argmin(hfb['energies']))
                append_prediction('single_gaussian_HFB',replicate,states[index:index+1],1,
                    {'test_target_used_for_candidates':False,'source_generation_seconds_cold':0.,
                     'dictionary_generation_seconds':hfb['elapsed_seconds'],'dictionary_selection_seconds':0.,
                     'HFB_multistarts':args.hfb_starts,'HFB_gradient_max':float(hfb['gradient_max'][index]),
                     'HFB_global_optimality_certified':False})
            # ED occurs only after all ordinary predictions at this Hamiltonian.
            tick=time.monotonic()
            energies,vectors=np.linalg.eigh(H)
            target=vectors[:,0]
            full_energies=np.linalg.eigvalsh(perturbed_hamiltonian(model,t,epsilon,even_only=False))
            ED_seconds=time.monotonic()-tick
            if energies[0]>full_energies[0]+1e-9:
                raise ArithmeticError('physical ground state left even parity; benchmark assumptions changed')
            exact=exact_metrics(target,model)
            target_row={'t':t,'epsilon':epsilon,'energy':float(energies[0]),'metrics':exact,
                        'ED_validation_seconds':ED_seconds,'full_parity_ground_energy':float(full_energies[0]),
                        'all_nonoracle_predictions_completed_before_ED':True}
            targets.append(target_row)
            np.savez_compressed(args.output/f't{t:g}_eps{epsilon:g}_target.npz',target=target,energies=energies)
            if not args.no_test_oracle:
                for replicate in args.replicates:
                    base=next(c for c in loaded.values() if c['entry']['t']==t and c['entry']['replicate']==replicate)
                    candidates,search=optimize_gaussian_overlap(target,base['starts'],max_sweeps=60,gradient_tolerance=1e-9)
                    tick=time.monotonic()
                    order=diverse_order(candidates,max(args.budgets))
                    atoms=candidates[order]
                    selection_seconds=time.monotonic()-tick
                    _,quality=full_distribution(atoms,model.j_matrices)
                    cov=covariance_from_kets(atoms)
                    kernels=coherent_kernels(atoms,operators)
                    for budget in args.budgets:
                        append_prediction('husimi_localmax_test_oracle',replicate,atoms,budget,
                           {'test_target_used_for_candidates':True,'oracle_is_a_prediction_method':False,
                            'source_generation_seconds_cold':ED_seconds,'dictionary_generation_seconds':search['elapsed_seconds'],
                            'dictionary_selection_seconds':selection_seconds,'Gaussian_quality':quality},cov=cov,kernels=kernels)
                    # The proper Husimi distribution provides a separate iid
                    # oracle, avoiding the rank collapse of local maxima.
                    rng=np.random.default_rng(np.random.SeedSequence([29100+replicate,901,int(round(t*10000)),int(round(epsilon*10000))]))
                    candidates,search=sample_husimi(target,len(base['starts']),rng,device=args.oracle_device)
                    tick=time.monotonic()
                    atoms=candidates[diverse_order(candidates,max(args.budgets))]
                    selection_seconds=time.monotonic()-tick
                    _,quality=full_distribution(atoms,model.j_matrices)
                    cov=covariance_from_kets(atoms)
                    kernels=coherent_kernels(atoms,operators)
                    for budget in args.budgets:
                        append_prediction('husimi_test_oracle',replicate,atoms,budget,
                           {'test_target_used_for_candidates':True,'oracle_is_a_prediction_method':False,
                            'source_generation_seconds_cold':ED_seconds,'dictionary_generation_seconds':search['elapsed_seconds'],
                            'dictionary_selection_seconds':selection_seconds,'Gaussian_quality':quality,
                            'Husimi_sampling':search},cov=cov,kernels=kernels)
            for record,state in predicted:
                record.update(fidelity=float(abs(np.vdot(target,state))**2),
                              energy_error=record['energy']-float(energies[0]),
                              mass_correlation_error=abs(record['coherent']['mass_correlation']-exact['mass_correlation']),
                              wick_deviation_error=abs(record['coherent']['wick_deviation']-exact['wick_deviation']),
                              total_cold_prediction_seconds=record['source_generation_seconds_cold']+record['dictionary_generation_seconds']+record['dictionary_selection_seconds']+record['prediction_seconds'])
                rows.append(record)
            write_json(args.output/'summary.json',{'status':'running','rows':rows,'targets':targets})
            print('evaluated',t,epsilon,'predictions',len(predicted),flush=True)
    write_json(args.output/'summary.json',{'status':'complete','rows':rows,'targets':targets,
               'elapsed_seconds':time.monotonic()-started,'source_freezing_seconds':frozen['elapsed_seconds'],
               'budgets_count_actual_gaussians':True,'HFB_has_one_actual_term':True,
               'timing_note':'Cold source cost includes complete shared upstream runs; also report marginal selection/prediction separately. It is not a matched wall-clock benchmark.',
               'requested_prediction_grid_complete':True,'independent_verification_required':True})


if __name__=='__main__':
    main()
