"""Single-process pipeline with gates and preserved negative results."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
from .runtime import Budget, ResourceLimit, save_json, save_arrays, source_hashes, peak_mib

ROOT = Path(__file__).resolve().parents[1]


def validate_config(config):
    if config['dictionaries'] != ['full','product']:
        raise ValueError('This pilot compares exactly full and product dictionaries')
    if not set(config['orders']).issubset({1,2,3,4}) or not config['orders']:
        raise ValueError('The first pilot is limited to k=1,2,3,4')
    sizes = config['sample_sizes']
    if sizes != sorted(set(sizes)) or min(sizes)<1 or max(sizes)>65536:
        raise ValueError('Sample sizes must increase and not exceed 65536')
    for key,ceiling in [('max_seconds',180),('max_total_seconds',1800),('max_memory_mib',512),('batch_size',256)]:
        if not 0 < config[key] <= ceiling:
            raise ValueError(f'{key} exceeds the pilot budget')
    if not all(np.isfinite(config['t_values'])) or min(config['t_values'])<0:
        raise ValueError('t values must be nonnegative and finite')
    if not 0 < config['total_failure_probability'] < 1:
        raise ValueError('Failure probability must be in (0,1)')


def run(output,config):
    validate_config(config)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing run: {output}')
    output.mkdir(parents=True)
    save_json(output/'config.json',config)
    started = time.monotonic()
    deadline = started+config['max_total_seconds']
    manifest = {'python':sys.version,'platform':platform.platform(),
                'dependencies':{name:importlib.metadata.version(name) for name in ('numpy','scipy','matplotlib','cvxpy','clarabel')},
                'source_sha256':source_hashes(ROOT),'command':sys.argv,
                'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                'time_zone_for_report':'Asia/Shanghai',
                'threads':1,'numerical_processes':1}
    manifest['git_status_before'] = subprocess.run(['git','status','--short'],cwd=ROOT,text=True,capture_output=True).stdout
    save_json(output/'manifest.json',manifest)
    summary = {'status':'running','stages':[],'output':str(output)}

    def stage(name,action,finalization=False):
        print(f'[{name}] starting',flush=True)
        budget = Budget(config['max_seconds'],config['max_memory_mib'],None if finalization else deadline)
        try:
            with budget:
                value = action(budget)
                budget.check()
            entry = {'stage':name,'status':'complete',**budget.usage()}
            print(f'[{name}] complete in {entry["seconds"]:.2f}s',flush=True)
        except Exception as exc:
            kind = 'resource_limit' if isinstance(exc,ResourceLimit) else 'error'
            entry = {'stage':name,'status':kind,'error':str(exc),**budget.usage()}
            (output/(name+'_error.log')).write_text(traceback.format_exc())
            value = None
            print(f'[{name}] {kind}: {exc}',flush=True)
        summary['stages'].append(entry)
        save_json(output/'summary.json',summary)
        return value

    from .model import build_model
    model = stage('model',lambda budget:build_model())
    if model is None:
        summary['status'] = 'model_failed'
        save_json(output/'summary.json',summary)
        return summary
    from .exact import run_exact
    exact = stage('exact',lambda budget:run_exact(model,config['t_values']))
    if exact is not None:
        save_json(output/'exact.json',exact)
        save_arrays(output/'exact_vectors.npz',S=model.S,
                            targets=np.array([model.target(t) for t in config['t_values']]),
                            even_masks=model.even_masks,
                            reduced_interaction=model.reduced_interaction,
                            reduced_hopping=model.reduced_hopping)
    structure_valid = exact is not None and exact['structure']['verified']
    if structure_valid:
        from .baselines import run_baselines
        baseline = stage('baselines',lambda budget:run_baselines(model,config['t_values'],config['seeds'][0],config['max_seconds']))
        if baseline is not None:
            report,arrays = baseline
            save_json(output/'baselines.json',report)
            save_arrays(output/'baseline_vectors.npz',**arrays)
    else:
        summary['stages'].append({'stage':'baselines','status':'skipped_invalid_symmetry'})

    from .pilot import calibrate,train,validate
    def partial_checkpoint(label):
        def checkpoint(report,arrays):
            save_json(output/('partial_'+label+'.json'),report)
            save_arrays(output/('partial_'+label+'.npz'),**arrays)
        return checkpoint
    calibrations = {}
    for name in config['dictionaries']:
        for seed in config['seeds']:
            label = f'calibration_{name}_{seed}'
            result = stage(label,lambda budget,n=name,s=seed,lab=label:calibrate(model,n,s,config,budget,partial_checkpoint(lab)))
            if result is not None:
                save_json(output/(label+'.json'),result)
                calibrations[(name,seed)] = result
    allowed = {}
    for name in config['dictionaries']:
        allowed[name] = [k for k in config['orders'] if all(
            (name,seed) in calibrations and any(r['k']==k and r['samples']==max(config['sample_sizes'])
                and r['diagnostic_calibration_passed'] for r in calibrations[(name,seed)]['rows'])
            for seed in config['seeds'])]
    summary['calibrated_orders'] = allowed
    all_rows = []
    if structure_valid:
        for name in config['dictionaries']:
            for seed in config['seeds']:
                label = f'{name}_{seed}'
                if not allowed[name]:
                    save_json(output/(f'pilot_{label}.json'),{'status':'skipped_calibration_failed','rows':[]})
                    continue
                def checkpoint(report,arrays,lab=label):
                    save_json(output/(f'training_{lab}.json'),report)
                    save_arrays(output/(f'witnesses_{lab}.npz'),**arrays)
                trained = stage('training_'+label,lambda budget,n=name,s=seed:train(model,n,s,config,budget,allowed[n],checkpoint))
                if trained is None:
                    continue
                training,arrays = trained
                checkpoint(training,arrays)
                result = stage('validation_'+label,lambda budget,n=name,s=seed,lab=label:validate(model,n,s,config,training,arrays,budget,partial_checkpoint('validation_'+lab)))
                if result is not None:
                    report,arrays,dist = result
                    report['skipped'] = training['skipped']
                    save_json(output/(f'pilot_{label}.json'),report)
                    save_arrays(output/(f'witnesses_{label}.npz'),**arrays)
                    save_arrays(output/(f'distributions_{label}.npz'),**dist)
                    all_rows.extend(report['rows'])
    if all_rows:
        scalar_keys = sorted({k for row in all_rows for k,v in row.items() if not isinstance(v,(dict,list,np.ndarray))})
        with (output/'metrics.csv').open('w',newline='') as handle:
            writer = csv.DictWriter(handle,scalar_keys,extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_rows)
    limited = (not all_rows or any(s['status']!='complete' for s in summary['stages'])
               or any(len(allowed[n])!=len(config['orders']) for n in config['dictionaries'])
               or any(not r['reconstruction_accepted'] or not r['statistically_certified'] for r in all_rows))
    summary.update({'status':'complete_with_limits' if limited else 'complete',
                    'candidate_count':len(all_rows),'reconstruction_accepted_count':sum(r['reconstruction_accepted'] for r in all_rows),
                    'k_ge_2_reconstruction_accepted_count':sum(r['reconstruction_accepted'] and r['k']>=2 for r in all_rows),
                    'statistically_certified_count':sum(r['statistically_certified'] for r in all_rows),
                    'distribution_interpretation_allowed_count':sum(r['distribution_interpretation_allowed'] for r in all_rows),
                    'elapsed_seconds':time.monotonic()-started,'peak_mib':peak_mib()})
    save_json(output/'summary.json',summary)
    from .report import make_report
    report_result = stage('report',lambda budget:make_report(output),finalization=True)
    if report_result is None:
        summary['status'] = 'complete_with_limits'
    summary['elapsed_seconds'] = time.monotonic()-started
    summary['peak_mib'] = peak_mib()
    save_json(output/'summary.json',summary)
    manifest['source_unchanged_during_run'] = manifest['source_sha256']==source_hashes(ROOT)
    save_json(output/'manifest.json',manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--config',default=str(ROOT/'configs/pilot.json'))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    result = run(args.output,config)
    print(json.dumps({key:result[key] for key in ('status','output')},indent=2))
    return 0 if result['status'] in ('complete','complete_with_limits') else 1
