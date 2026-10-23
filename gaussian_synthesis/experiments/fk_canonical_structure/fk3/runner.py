"""Sequential runner for the frozen canonical-structure protocol."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
from fk.runtime import Budget, ResourceLimit, peak_mib, save_json, save_arrays, source_hashes

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT.parent/'fk_two_cluster'
FOLLOWUP = ROOT.parent/'fk_two_cluster_followup'


def validate_config(c):
    for name,cap in [('max_seconds',180),('max_total_seconds',1800),
                     ('max_memory_mib',512),('batch_size',256),('final_samples',65536)]:
        if not 0 < c[name] <= cap:
            raise ValueError(f'{name} exceeds approved scope')
    if c['t_values'] != [0,5,9,11,20] or c['k4_t_values'] != [0,11,20]:
        raise ValueError('Fixed physical parameter grid required')
    if c['sample_sizes'] != sorted(set(c['sample_sizes'])) or not 0 < min(c['sample_sizes']) <= max(c['sample_sizes']) <= 65536:
        raise ValueError('Invalid development sample grid')
    if len(c['seeds']) != len(set(c['seeds'])) or not c['seeds']:
        raise ValueError('Seeds must be distinct')
    if c['stream_namespace'] != 910 or c['discovery']['atom_count'] > 32 or c['discovery']['max_sweeps'] > 30:
        raise ValueError('Random stream or discovery scope changed')
    if c['planned_statistical_events'] < 5*len(c['seeds'])*(len(c['t_values'])+len(c['k4_t_values'])):
        raise ValueError('Insufficient union-bound allocation')
    if not 0 < c['total_failure_probability'] < 1:
        raise ValueError('Invalid failure probability')


def candidate_digest(candidates):
    h = hashlib.sha256()
    for row in candidates:
        h.update(json.dumps({k:row[k] for k in ('id','t','k','source','deterministic_verified','optimizer_converged')},sort_keys=True).encode())
        h.update(np.asarray(row['u'],dtype=np.complex128).tobytes())
    return h.hexdigest()


def aggregate_gates(development, final, config):
    """Across-seed gates; missing data fail rather than selecting survivors."""
    seeds = config['seeds']
    cap = config['final_samples']
    devcap = max(config['sample_sizes'])
    final_rows = [dict(r,seed=seed) for seed,report in final.items()
                  for r in report.get('rows',[]) if r.get('n') == cap]
    dev_rows = [dict(r,seed=seed) for seed,report in development.items()
                for r in report.get('rows',[]) if r.get('n') == devcap]
    probe_rows = [r for r in final_rows if r.get('is_probe')]
    expected_probes = {(seed,f'probe_k{k}') for seed in seeds for k in (2,4)}
    probes_ok = (len(probe_rows)==len(expected_probes)
                 and {(r['seed'],r['id']) for r in probe_rows}==expected_probes
                 and all(r.get('probe_passed',False) for r in probe_rows))
    expected = [(f't{t:g}_k2',t,2) for t in config['t_values']]+[(f't{t:g}_k4',t,4) for t in config['k4_t_values']]
    rows=[]
    for key,t,k in expected:
        r=[r for r in final_rows if not r.get('is_probe') and r['id']==key]
        d=[r for r in dev_rows if not r.get('is_probe') and r['id']==key]
        complete=(len(r)==len(seeds) and {x['seed'] for x in r}==set(seeds)
                  and len(d)==len(seeds) and {x['seed'] for x in d}==set(seeds))
        eta_range=float(np.ptp([x['mean_eta'] for x in r])) if r else None
        tv=[x.get('histogram_tv_previous') for x in d]
        reconstruction=bool(complete and all(x.get('local_reconstruction_passed',False) for x in r))
        stable=bool(complete and all(v is not None and v<=config['histogram_tv_max'] for v in tv)
                    and all(x['ess']>=config['mass_ess_min'] for x in r)
                    and eta_range<=config['eta_seed_range_max'])
        distribution=bool(probes_ok and reconstruction and stable)
        coherent=bool(distribution and all(
            x['coherent_closure_residual']<=config['reconstruction_tolerance']
            and x['CQ_error']<=config['observable_tolerance']
            and x['W_error'] is not None and x['W_error']<=config['observable_tolerance']
            and np.max(x['group_rmse'])<=config['group_rmse_max']
            and np.max(x['group_half_disagreement'])<=config['group_half_disagreement_max'] for x in r))
        rows.append({'id':key,'t':t,'k':k,'complete_three_seeds':complete,
                     'probe_calibration_passed':probes_ok,'reconstruction_passed':reconstruction,
                     'distribution_stable':stable,'distribution_interpretation_allowed':distribution,
                     'coherent_interpretation_allowed':coherent,'mean_eta_seed_range':eta_range,
                     'development_tv_values':tv,'minimum_final_ess':min([x['ess'] for x in r],default=None),
                     'evidence_level':'numerical_development_gates','statistically_certified':False})
    return {'rows':rows,'probe_calibration_passed':probes_ok,
            'reconstruction_passed_count':sum(r['reconstruction_passed'] for r in rows),
            'distribution_allowed_count':sum(r['distribution_interpretation_allowed'] for r in rows),
            'coherent_allowed_count':sum(r['coherent_interpretation_allowed'] for r in rows),
            'interpretation':'All configured seeds are required; these stability thresholds are not confidence guarantees.'}


def run(output,config):
    validate_config(config)
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    save_json(output/'config.json',config)
    source_hash=source_hashes(ROOT)
    backend_hash=source_hashes(BACKEND)
    followup_hash=source_hashes(FOLLOWUP)
    freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
    (output/'requirements-lock.txt').write_text(freeze)
    manifest={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'python':sys.version,'platform':platform.platform(),'command':sys.argv,
              'source_sha256':source_hash,'backend_sha256':backend_hash,'followup_sha256':followup_hash,
              'dependencies':{n:importlib.metadata.version(n) for n in ('numpy','scipy','matplotlib','cvxpy','clarabel')},
              'numerical_processes':1,'numerical_threads':1,'candidate_freezes':{},
              'evidence_policy':'Numerical residuals, stability, confidence bounds and rigorous intervals are distinct.'}
    save_json(output/'manifest.json',manifest)
    started=time.monotonic()
    deadline=started+config['max_total_seconds']
    summary={'status':'running','output':str(output),'stages':[]}

    def checkpoint(name):
        def write(report,arrays):
            save_json(output/f'partial_{name}.json',report)
            save_arrays(output/f'partial_{name}.npz',**arrays)
        return write

    def stage(name,action,finalization=False):
        print(f'[{name}] starting',flush=True)
        budget=Budget(config['max_seconds'],config['max_memory_mib'],None if finalization else deadline)
        result=None
        try:
            with budget:
                result=action(budget)
                budget.check()
            entry={'stage':name,'status':'complete',**budget.usage()}
            if isinstance(result,tuple) and isinstance(result[0],dict):
                save_json(output/f'{name}.json',result[0])
                save_arrays(output/f'{name}.npz',**result[1])
                entry['status']=result[0].get('status','complete')
        except Exception as exc:
            entry={'stage':name,'status':'resource_limit' if isinstance(exc,ResourceLimit) else 'error',
                   'error':str(exc),**budget.usage()}
            (output/f'{name}_error.log').write_text(traceback.format_exc())
            result=None
        summary['stages'].append(entry)
        save_json(output/'summary.json',summary)
        print(f'[{name}] {entry["status"]} in {entry["seconds"]:.2f}s',flush=True)
        return result

    from fk.model import build_model
    from fk.exact import run_exact
    model=stage('model',lambda b:build_model())
    if model is None:
        summary['status']='failed_model'
        save_json(output/'summary.json',summary)
        return summary
    physics=stage('physics',lambda b:run_exact(model,config['t_values']))
    if physics is not None:
        save_json(output/'physics.json',physics)
    save_arrays(output/'model.npz',S=model.S,even_masks=model.even_masks,
                targets=np.array([model.target(t) for t in config['t_values']]))
    structure_ok=physics is not None and physics['structure']['verified']
    exact=None
    development={}; final={}; candidate_sets={}; discovery=[]
    if structure_ok:
        from .canonical import prepare_canonical,train_k4
        exact=stage('canonical_k2',lambda b:prepare_canonical(model,config,b,checkpoint('canonical_k2')))
        if exact is not None and exact[0].get('verified',False):
            k2w={t:exact[1][f'witness_t{t:g}_k2'] for t in config['t_values']}
            for seed in config['seeds']:
                name=f'training_{seed}'
                trained=stage(name,lambda b:train_k4(model,config,seed,b,checkpoint(name)))
                candidates=[{'id':f't{t:g}_k2','t':t,'k':2,'u':k2w[t],
                             'source':'deterministic_integer_moment','deterministic_verified':True,
                             'optimizer_converged':True} for t in config['t_values']]
                if trained is not None:
                    for t in config['k4_t_values']:
                        key=f'witness_t{t:g}_k4_n{max(config["sample_sizes"])}'
                        if key in trained[1]:
                            tr=next(r for r in trained[0]['rows'] if r['witness_key']==key)
                            candidates.append({'id':f't{t:g}_k4','t':t,'k':4,'u':trained[1][key],
                                'source':'empirical_full_Haar','deterministic_verified':False,
                                'optimizer_converged':bool(tr['optimizer_converged'])})
                candidate_sets[seed]=candidates
                manifest['candidate_freezes'][str(seed)]={'sha256':candidate_digest(candidates),'ids':[x['id'] for x in candidates]}
                save_arrays(output/f'frozen_{seed}.npz',witnesses=np.array([c['u'] for c in candidates]),
                            t=np.array([c['t'] for c in candidates]),k=np.array([c['k'] for c in candidates]),
                            ids=np.array([c['id'] for c in candidates]))
            save_json(output/'manifest.json',manifest)
            from .statistics import run_statistics
            for seed,candidates in candidate_sets.items():
                name=f'development_{seed}'
                result=stage(name,lambda b:run_statistics(model,candidates,config,seed,b,checkpoint(name),phase='development'))
                if result is not None:
                    development[seed]=result[0]
            # No development result is used to select/change witnesses, bins, N or k.
            manifest['final_validation_plan_frozen']={'samples':config['final_samples'],
                'candidate_sha256':{str(s):candidate_digest(cs) for s,cs in candidate_sets.items()},
                'planned_events':config['planned_statistical_events'],
                'total_failure_probability':config['total_failure_probability']}
            save_json(output/'manifest.json',manifest)
            for seed,candidates in candidate_sets.items():
                name=f'validation_{seed}'
                if seed not in development or development[seed].get('probe_status')!='passed':
                    summary['stages'].append({'stage':name,'status':'skipped_failed_development_calibration'})
                    save_json(output/'summary.json',summary)
                    continue
                result=stage(name,lambda b:run_statistics(model,candidates,config,seed,b,checkpoint(name),phase='final'))
                if result is not None:
                    final[seed]=result[0]
            from .discovery import run_discovery
            for seed in config['seeds']:
                name=f'discovery_{seed}'
                result=stage(name,lambda b:run_discovery(model,k2w,config,seed,b,checkpoint(name)))
                if result is not None:
                    discovery.append(result[0])
        else:
            summary['stages'].append({'stage':'dependent_canonical_stages','status':'skipped_failed_exact_audit'})
    else:
        summary['stages'].append({'stage':'dependent_stages','status':'skipped_invalid_model'})
    gates=aggregate_gates(development,final,config)
    save_json(output/'acceptance.json',gates)
    certified_count=sum(bool(r['formal_envelopes']['C_certified_within_tolerance'])
                        for report in final.values() for r in report.get('rows',[])
                        if r.get('n')==config['final_samples'] and not r['is_probe'])
    expected_certifications=len(config['seeds'])*(len(config['t_values'])+len(config['k4_t_values']))
    stage_failure=any(r['status'] not in ('complete','passed','trained') for r in summary['stages'])
    summary.update({'status':'complete_with_limits' if stage_failure or gates['coherent_allowed_count']<8 or certified_count<expected_certifications else 'complete',
                    'structure_verified':structure_ok,'canonical_k2_verified':bool(exact and exact[0].get('verified',False)),
                    'development_stages':len(development),'validation_stages':len(final),'discovery_stages':len(discovery),
                    'reconstruction_passed_count':gates['reconstruction_passed_count'],
                    'distribution_allowed_count':gates['distribution_allowed_count'],
                    'coherent_allowed_count':gates['coherent_allowed_count'],
                    'statistically_certified_count':certified_count,
                    'elapsed_seconds':time.monotonic()-started,'peak_mib':peak_mib()})
    save_json(output/'summary.json',summary)
    from .report import make_report
    stage('report',lambda b:make_report(output),finalization=True)
    summary.update({'elapsed_seconds':time.monotonic()-started,'peak_mib':peak_mib()})
    if summary['stages'][-1]['status']!='complete':
        summary['status']='complete_with_limits'
    manifest['source_unchanged_during_run']=source_hash==source_hashes(ROOT)
    manifest['backend_unchanged_during_run']=backend_hash==source_hashes(BACKEND)
    manifest['followup_unchanged_during_run']=followup_hash==source_hashes(FOLLOWUP)
    save_json(output/'manifest.json',manifest)
    save_json(output/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--config',default=str(ROOT/'configs/canonical.json'))
    args=parser.parse_args()
    result=run(args.output,json.loads(Path(args.config).read_text()))
    print(json.dumps({'status':result['status'],'output':result['output']},indent=2))
    return int(result['status'].startswith('failed'))
