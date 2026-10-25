#!/usr/bin/env python3
"""Physical circulating-pair spectra before any canonical geometry claim."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.chirality_spectrum import ChiralitySpectrum


def job(task):
    L,eta,out=task;root=out.parent;tag=f'L{L}_eta{eta:g}';model,_=angular_model(L,relative_strength=eta)
    probe=ChiralitySpectrum(model);result={'L':L,'eta':eta,'models':{}}
    paths={method:root/'nonlinear_channels_01'/f'{tag}_{method}.npz' for method in ('exact','full_cc','source_only_cc','riccati')}
    paths['projective']=root/'projective_channels_01'/f'{tag}_projective.npz'
    for q in (2,4):
        path=root/'nonlinear_channel_spans_01'/f'{tag}_q{q}.npz'
        if path.exists():paths[f'Gaussian_span_q{q}']=path
    for method,path in paths.items():
        data=np.load(path);states=data['states'];times=data['times'];rows=[]
        for t in (.5,1.,1.5,2.,2.5):
            index=int(np.argmin(abs(times-t)));assert abs(times[index]-t)<1e-10
            rows.append({'time':t,**probe.evaluate(states[index])})
        result['models'][method]=rows
    (out/f'{tag}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'tag':tag,'time':2.,'rows':{method:{k:rows[3][k] for k in ('chi_second_per_r_squared','macroscopic_tail_probability','two_circular_orbit_projection_excluding_endpoints')} for method,rows in result['models'].items()}}),flush=True)
    return {'L':L,'eta':eta,'file':f'{tag}.json'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps({'classification':'Retrospective physical gate on previously inspected competing-channel conditions; not a forecast.',
      'observable':'Physical circulating-pair population difference in both electron bands; holes have the opposite sign.',
      'gate':'Look for resolved separated branches and a discrepancy beyond conventional models, with size and initial-count controls. Binder positivity alone is not evidence.',
      'cases':[[10,1],[14,1],[18,1],[14,.85]],'times':[.5,1,1.5,2,2.5],
      'tail_threshold':'|chi|>0.15r, fixed across sizes; also retain full spectra and explicit circular-orbit references.'},indent=2))
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(job,(*case,args.output)) for case in ((10,1.),(14,1.),(18,1.),(14,.85))]):results.append(f.result())
    (args.output/'summary.json').write_text(json.dumps(results,indent=2))
