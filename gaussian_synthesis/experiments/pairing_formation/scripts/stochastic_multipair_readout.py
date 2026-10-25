#!/usr/bin/env python3
"""Strong stochastic controls for the newly identified phase-cycled curvature."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,argparse,multiprocessing,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.kernel_stochastic import torch_kernel_run
from pairing_formation.multipair_noise import distinct_transfer_square_symbol
from scripts.pilot_pair_symbols import digest


def worker(task):
    out,device,kind,seed,dt,count=task;rows=[];begin=time.monotonic()
    names=['readout_q1_real','readout_q1_imag','readout_q3_real','readout_q3_imag']
    for L in (10,14,18):
        model,_=angular_model(L);r=model.r;theta=2*np.pi*np.arange(r)/r
        weights=[np.cos(q*(theta[:,None]-theta[None,:]))/r for q in (1,3)];cache=[]
        def extra(state,torch):
            if not cache:cache.extend(torch.as_tensor(W,device=state.device,dtype=torch.complex128) for W in weights)
            plus=state[:,r:,0]+1j*state[:,r:,1];minus=state[:,:r,0]-1j*state[:,:r,1];values=[]
            for W in cache:
                second=distinct_transfer_square_symbol(plus,minus,W);values.extend((-2*second.real,-2*second.imag))
            return torch.stack(values,dim=-1)
        actual_seed=seed+10000*L+(1000 if kind=='discrete' else 0)
        report,blocks=torch_kernel_run(model,[],count=count,kind=kind,seed=actual_seed,times=[0.,1.,1.35,2.],dt=dt,device=device,extra_measure=extra,extra_names=names)
        tag=f'L{L}_{kind}_seed{seed}_dt{dt:g}';path=out/(tag+'.npz');np.savez_compressed(path,blocks=blocks,times=[0.,1.,1.35,2.])
        (out/(tag+'.json')).write_text(json.dumps(report,indent=2))
        row={'L':L,'kind':kind,'base_seed':seed,'seed':actual_seed,'dt':dt,'samples':count,'device':device,'seconds_from_worker_start':time.monotonic()-begin,
            'file':path.name,'sha256':digest(path),'report_file':tag+'.json','report_sha256':digest(out/(tag+'.json'))}
        rows.append(row);print(json.dumps(row),flush=True)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=131072);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    protocol={'classification':'Retrospective strong stochastic controls for two-transfer readout curvature; previously inspected conditions, no new parameter holdouts.',
        'sizes':[10,14,18],'G':4.,'eta':1.,'times':[0.,1.,1.35,2.],'kinds':['gaussian','discrete'],'base_seeds':[2101,2102],'dt':[.004,.002],
        'samples_per_run':a.samples,'independent_samples_excluding_same_seed_step_repeats':12*a.samples,
        'measurement':'X^2 symbol excludes repeated upper or lower spin sites before factorization. Outputs -2 Re and -2 Im <X_q^2>, q1 and3. Gaussian-seeded SMF and discrete TWA are approximations, not positive quantum density matrices or pure-state echoes.',
        'refinement':'Both time steps reuse the same samples within each kind,size,seed; paired block differences isolate time-step error.'}
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
    tasks=[(a.output,d,kind,seed,dt,a.samples) for d,(kind,seed,dt) in enumerate((kind,seed,dt) for kind in ('gaussian','discrete') for seed in (2101,2102) for dt in (.004,.002))]
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(worker,t) for t in tasks]):rows.extend(f.result())
    (a.output/'summary.json').write_text(json.dumps({'jobs':rows},indent=2))
