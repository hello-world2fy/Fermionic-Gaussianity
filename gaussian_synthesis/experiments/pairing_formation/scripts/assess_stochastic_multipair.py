#!/usr/bin/env python3
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.pilot_pair_symbols import digest


def main(source,out):
    out.mkdir(parents=True,exist_ok=False);records=json.loads((source/'summary.json').read_text())['jobs'];protocol=json.loads((source/'protocol.json').read_text())
    inputs={str(source/'summary.json'):digest(source/'summary.json'),str(source/'protocol.json'):digest(source/'protocol.json')};loaded={};diagnostics=[]
    for row in records:
        for field,hashfield in (('file','sha256'),('report_file','report_sha256')):
            p=source/row[field];assert digest(p)==row[hashfield];inputs[str(p)]=row[hashfield]
        loaded[row['L'],row['kind'],row['base_seed'],row['dt']]=np.load(source/row['file'])['blocks']
        diagnostics.append(json.loads((source/row['report_file']).read_text()))
    results=[];step_checks=[];initial_checks=[]
    for L in (10,14,18):
        exact=[]
        for q in (1,3):
            path=source.parent/f'multipair_readout_01/L{L}_q{q}.json';inputs[str(path)]=digest(path);rows=json.loads(path.read_text())['rows']
            for phi in (0.,float(np.pi/4)):
                exact.append(next(r['analytic_coefficient'] for r in rows if r['method']=='exact' and r['phase']==phi and r['epsilon']==.01))
        exact=np.array(exact)
        for kind in ('gaussian','discrete'):
            for dt in (.004,.002):
                blocks=np.concatenate([loaded[L,kind,seed,dt][-1,:,3:] for seed in (2101,2102)])
                average=blocks.mean(axis=0);se=blocks.std(axis=0,ddof=1)/np.sqrt(len(blocks))
                results.append({'L':L,'method':kind,'dt':dt,'samples':2*protocol['samples_per_run'],
                    'mean':average.tolist(),'estimated_SE':se.tolist(),'exact':exact.tolist(),
                    'absolute_errors':abs(average-exact).tolist(),'error_over_estimated_SE':(abs(average-exact)/se).tolist(),
                    'correct_signs_of_four':int(np.sum(average*exact>0))})
            for seed in (2101,2102):
                delta=loaded[L,kind,seed,.004]-loaded[L,kind,seed,.002]
                step_checks.append({'L':L,'method':kind,'seed':seed,'max_mean_difference':float(np.max(abs(delta.mean(axis=1)))),
                    'max_block_difference':float(np.max(abs(delta))),'readout_time2_mean_differences':delta[-1,:,3:].mean(axis=0).tolist()})
            blocks=np.concatenate([loaded[L,kind,seed,.002][0,:,3:] for seed in (2101,2102)])
            initial_checks.append({'L':L,'method':kind,'max_initial_zero_error_over_SE':float(np.max(abs(blocks.mean(axis=0))/(blocks.std(axis=0,ddof=1)/np.sqrt(len(blocks)))))})
    report={'classification':'Retrospective SMF/discrete-TWA comparison for a newly screened physical observable; no canonical-specific discovery or new parameter forecasts.',
        'input_hashes':inputs,'independent_samples':protocol['independent_samples_excluding_same_seed_step_repeats'],
        'runs_including_time_step_repeats':len(records),'rows':results,'step_checks':step_checks,'initial_zero_checks':initial_checks,
        'max_spin_length_squared_drift':max(r['max_spin_length_squared_drift'] for r in diagnostics),
        'max_number_drift':max(r['max_number_drift'] for r in diagnostics),
        'max_energy_symbol_drift':max(r['max_energy_symbol_drift'] for r in diagnostics),
        'scope':'Monte Carlo standard errors are estimated, not rigorous confidence intervals. The measurement symbol excludes every repeated spin site before factorization. Same-seed finer-step runs are not additional independent samples.'}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'refined_rows':[r for r in results if r['dt']==.002],
        'max_step_mean_difference':max(r['max_mean_difference'] for r in step_checks),
        'max_initial_error_over_SE':max(r['max_initial_zero_error_over_SE'] for r in initial_checks)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.source,a.output)
