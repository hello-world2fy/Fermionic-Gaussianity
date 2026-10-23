"""Sequential, checkpointed experiments with immutable first-round inputs."""
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
from fk.runtime import Budget, ResourceLimit, save_json, save_arrays, source_hashes, peak_mib

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT.parent/'fk_two_cluster'


def file_hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob('*')) if p.is_file()
            and not any(part.startswith('.') for part in p.relative_to(root).parts)}


def validate_config(c):
    for name, cap in [('max_seconds',180),('max_total_seconds',1800),
                      ('max_memory_mib',512),('batch_size',256)]:
        if not 0 < c[name] <= cap:
            raise ValueError(f'{name} exceeds the bounded follow-up scope')
    if c['dictionaries'] != ['full','product'] or c['orders'] != [2,4]:
        raise ValueError('Compare both dictionaries at k=2,4 only')
    if c['t_values'] != [0,11,20]:
        raise ValueError('This round keeps exactly t=0,11,20')
    if c['source_train_samples'] != 65536:
        raise ValueError('Freeze the largest first-round witness')
    if c['sample_sizes'] != sorted(set(c['sample_sizes'])) or not 0 < min(c['sample_sizes']) <= max(c['sample_sizes']) <= 65536:
        raise ValueError('Invalid bounded sample grid')
    if c['orbit_sizes'] != sorted(set(c['orbit_sizes'])) or not 1 <= min(c['orbit_sizes']) <= max(c['orbit_sizes']) <= 128:
        raise ValueError('Orbit sizes are TOTAL Gaussian counts, capped at128')
    if c['rng_purpose'] < 800:
        raise ValueError('Second-round streams must differ from first-round streams')
    if not 0 < c['total_failure_probability'] < 1:
        raise ValueError('Invalid joint failure budget')


def run(output, config):
    validate_config(config)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    source = (BACKEND/config['source_run']).resolve()
    if not (source/'baseline_vectors.npz').is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True)
    save_json(output/'config.json', config)
    frozen_dependencies=subprocess.run([sys.executable,'-m','pip','freeze'],
                                       check=True,capture_output=True,text=True).stdout
    (output/'requirements-lock.txt').write_text(frozen_dependencies)
    started = time.monotonic()
    deadline = started+config['max_total_seconds']
    manifest = {'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                'python':sys.version, 'platform':platform.platform(), 'command':sys.argv,
                'source_run':str(source), 'source_sha256':source_hashes(ROOT),
                'backend_sha256':source_hashes(BACKEND), 'input_sha256':file_hashes(source),
                'dependencies':{n:importlib.metadata.version(n) for n in ('numpy','scipy','matplotlib','cvxpy','clarabel')},
                'numerical_processes':1, 'numerical_threads':1,
                'evidence_policy':'Numerical residuals and empirical variances are not interval or statistical certificates.'}
    save_json(output/'manifest.json',manifest)
    summary = {'status':'running','output':str(output),'stages':[]}

    def checkpoint(name):
        def save(report, arrays):
            save_json(output/('partial_'+name+'.json'),report)
            save_arrays(output/('partial_'+name+'.npz'),**arrays)
        return save

    def stage(name, action, finalization=False):
        print(f'[{name}] starting', flush=True)
        budget = Budget(config['max_seconds'],config['max_memory_mib'],None if finalization else deadline)
        result = None
        try:
            with budget:
                result = action(budget)
                budget.check()
            entry = {'stage':name,'status':'complete',**budget.usage()}
            if isinstance(result,tuple) and isinstance(result[0],dict):
                returned_status=result[0].get('status','complete')
                if returned_status not in ('complete','passed'):
                    entry['status']=returned_status
        except Exception as exc:
            entry = {'stage':name,'status':'resource_limit' if isinstance(exc,ResourceLimit) else 'error',
                     'error':str(exc),**budget.usage()}
            (output/(name+'_error.log')).write_text(traceback.format_exc())
        summary['stages'].append(entry)
        save_json(output/'summary.json',summary)
        print(f'[{name}] {entry["status"]} ({entry["seconds"]:.2f}s)',flush=True)
        return result

    def save_result(name, result):
        if result is not None:
            report, arrays = result
            save_json(output/(name+'.json'),report)
            save_arrays(output/(name+'.npz'),**arrays)
        return result

    from fk.model import build_model
    from fk.exact import run_exact
    model = stage('model',lambda b:build_model())
    if model is None:
        summary['status']='failed_model'
        save_json(output/'summary.json',summary)
        return summary
    physics = stage('physics',lambda b:run_exact(model,config['t_values']))
    if physics is not None:
        save_json(output/'physics.json',physics)
    structure_ok = physics is not None and physics['structure']['verified']
    save_arrays(output/'model.npz',S=model.S,even_masks=model.even_masks,
                targets=np.array([model.target(t) for t in config['t_values']]))
    exact = None
    rb_results = []
    finite = None
    if structure_ok:
        from .exact_moments import run_exact_reference
        exact = save_result('exact_k2',stage('exact_k2',lambda b:run_exact_reference(model,config,b)))
        from .rb import run_rb
        extras = []
        if exact is not None and exact[0].get('verified',False):
            for t in config['t_values']:
                key = f'witness_t{t:g}'
                extras.append({'label':f'exact_k2_t{t:g}', 't':t, 'k':2,
                               'u':exact[1][key], 'expected_reconstruction':model.target(t)})
        for name in config['dictionaries']:
            for seed in config['seeds']:
                label = f'rb_{name}_{seed}'
                stage_config = dict(config,dictionary=name,seed=seed,
                                    enable_exact_reference=bool(exact and exact[0].get('verified',False)))
                result = save_result(label,stage(label,lambda b,n=name,c=stage_config,lab=label:
                    run_rb(model,source,c,b,checkpoint(lab),extra_witnesses=extras if n=='full' else None)))
                if result is not None:
                    rb_results.append(result[0])
        with np.load(source/'baseline_vectors.npz') as a:
            source_arrays = {key:a[key] for key in a.files}
        from .ablation import run_ablation
        save_result('ablation',stage('ablation',lambda b:run_ablation(model,source_arrays,config,b,checkpoint('ablation'))))
        from .orbits import run_finite_orbits
        orbit_config = dict(config,t=config['orbit_t'],R_values=config['orbit_sizes'])
        finite = save_result('finite_orbits',stage('finite_orbits',lambda b:
                      run_finite_orbits(model,source_arrays,orbit_config,b,checkpoint('finite_orbits'))))
    else:
        summary['stages'].append({'stage':'dependent_experiments','status':'skipped_invalid_structure'})
    summary.update({'status':'complete_with_limits', 'structure_verified':structure_ok,
                    'exact_k2_verified':bool(exact and exact[0].get('verified',False)),
                    'rb_stage_count':len(rb_results), 'finite_orbits_completed':finite is not None,
                    'elapsed_seconds':time.monotonic()-started,'peak_mib':peak_mib()})
    rb_rows=[r for result in rb_results for r in result.get('rows',[])]
    final_rows=[r for r in rb_rows if r['samples']==max(config['sample_sizes'])]
    frozen=[r for r in final_rows if r['sourcekind']=='frozen_pilot']
    summary.update({'rb_record_count':len(rb_rows), 'rb_final_record_count':len(final_rows),
                    'frozen_final_raw_accepted':sum(r['raw_within_tolerance'] for r in frozen),
                    'frozen_final_orbit_accepted':sum(r['orbit_within_tolerance'] for r in frozen),
                    'frozen_final_count':len(frozen),
                    'rb_statistical_certificates':sum(r.get('formally_certified',False) for r in rb_rows),
                    'probe_calibration_status':{f'{r["dictionary"]}_{r["seed"]}':r.get('probe_calibration_status') for r in rb_results}})
    save_json(output/'summary.json',summary)
    from .report import make_report
    stage('report',lambda b:make_report(output),finalization=True)
    summary.update({'elapsed_seconds':time.monotonic()-started,'peak_mib':peak_mib()})
    manifest['source_unchanged_during_run'] = manifest['source_sha256']==source_hashes(ROOT)
    manifest['backend_unchanged_during_run'] = manifest['backend_sha256']==source_hashes(BACKEND)
    manifest['inputs_unchanged_during_run'] = manifest['input_sha256']==file_hashes(source)
    save_json(output/'manifest.json',manifest)
    save_json(output/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--config',default=str(ROOT/'configs/followup.json'))
    args=parser.parse_args()
    result=run(args.output,json.loads(Path(args.config).read_text()))
    print(json.dumps({'status':result['status'],'output':result['output']},indent=2))
    return int(result['status'].startswith('failed'))
