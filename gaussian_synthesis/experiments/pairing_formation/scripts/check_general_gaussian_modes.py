#!/usr/bin/env python3
"""General pairing-matrix local variations around selected onsite modes."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,time,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.pairing_channels import angular_model
from pairing_formation.columns import FullGaussianOracle


def job(task):
    device,tag,kind,source,out=task;begin=time.monotonic();training=np.load(source/f'{tag}_training.npz');L=int(training['L'])
    model=angular_model(L)[0] if tag.startswith('angular') else picket_fence(L,2.)
    rows=json.loads((source/f'{tag}_{kind}.json').read_text());saved=np.load(source/f'{tag}_{kind}.npz')
    best=max(range(len(rows)),key=lambda i:rows[i]['score_normalized_source'])
    vector=training[kind];oracle=FullGaussianOracle(model,device=f'cuda:{device}')
    atom,Z,report=oracle.optimize(vector,saved['parameters'][best],starts=4,seed=900+device,maxiter=250)
    fock=int(np.argmax(abs(vector)));fock_score=float(abs(vector[fock])**2/np.linalg.norm(vector)**2)
    report.update({'tag':tag,'kind':kind,'device':device,'seconds':time.monotonic()-begin,
      'onsite_score_normalized':rows[best]['score_normalized_source'],
      'general_score_normalized':float(abs(np.vdot(vector,atom))**2/np.linalg.norm(vector)**2),
      'best_Fock_score_normalized':fock_score,'best_Fock_index':fock,
      'limitation':'Four local starts around the selected onsite solution. No full-measure sampling or global maximum certificate.'})
    np.savez_compressed(out/f'{tag}_{kind}.npz',atom=atom,qp_Z=Z)
    (out/f'{tag}_{kind}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('results/gaussian_modes_01'));p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    cases=[(tag,kind) for tag in ('angular_L10_t0.5','angular_L10_t1','angular_L10_t2','uniform_L12_t2.15') for kind in ('target','witness')]
    (args.output/'protocol.json').write_text(json.dumps({'classification':'Local general-Gaussian support probe; not a global support certificate or sampling experiment.','cases':cases,'starts_per_case':4},indent=2))
    reports=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(job,(d,*case,args.source,args.output)) for d,case in enumerate(cases)]):reports.append(future.result())
    (args.output/'summary.json').write_text(json.dumps(reports,indent=2))
