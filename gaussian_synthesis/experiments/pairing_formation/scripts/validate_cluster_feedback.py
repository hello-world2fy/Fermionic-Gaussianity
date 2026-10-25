#!/usr/bin/env python3
"""Tight integration and count-sector Ehrenfest checks for the cluster pilot."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.cluster_dynamics import ClusterDynamics


def job(task):
    method,q,output,solver,error_control,settings=task
    root=output.parent/'cluster_feedback_pilot_01';data=np.load(root/'frozen_training.npz')
    model=picket_fence(12,2.)
    clusters=[] if method=='radial_cc' else [(2,data['common_radial'])]
    if method in ('direct','witness','Hamiltonian'):clusters += [(2,x) for x in data[method+'_modes'][:q]]
    name=f'{method}_q{q}'
    comparison=settings['compare_directory']
    old=np.load(root/f'{name}_prediction.npz' if comparison is None else comparison/f'{name}.npz');times=old['times']
    d=ClusterDynamics(model,clusters,radial=True,metric_cutoff=settings['metric_cutoff'],metric_solver=solver)
    original=d.derivative;last_log=time.monotonic()
    def logged_derivative(t,coordinates,diagnostics=False):
        nonlocal last_log
        now=time.monotonic()
        if now-last_log>20:
            print(json.dumps({'name':name,'progress_time':t,'calls':d.calls,'solver':solver}),flush=True)
            last_log=now
        return original(t,coordinates,diagnostics)
    d.derivative=logged_derivative
    states,coordinates,report=d.evolve(times,rtol=settings['rtol'],atol=settings['atol'],max_step=settings['max_step'],error_control=error_control)
    rays=[]
    for x,y in zip(old['states'],states):
        x=x/np.linalg.norm(x);y=y/np.linalg.norm(y);phase=np.vdot(y,x);phase/=abs(phase)
        rays.append(float(np.linalg.norm(x-phase*y)))
    continuity=[]
    for k in (1,5,10,20,30,43,52):
        _,_,rate=d.derivative(times[k],coordinates[k],True);state=states[k]
        actual=np.bincount(model.m,weights=2*np.real(state.conj()*rate),minlength=model.r+1)
        exact=np.bincount(model.m,weights=2*np.real(state.conj()*(-1j*(model.H@state))),minlength=model.r+1)
        continuity.append({'time':float(times[k]),'max_count_sector_rate_error':float(np.max(abs(actual-exact)))})
    result={'name':name,'max_normalized_ray_difference':max(rays),'ray_differences':rays,
            'count_Ehrenfest_checks':continuity,'integration':report}
    np.savez_compressed(output/f'{name}.npz',times=times,states=states,coordinates=coordinates)
    (output/f'{name}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'name':name,'max_normalized_ray_difference':max(rays),
                      'max_count_Ehrenfest_error':max(x['max_count_sector_rate_error'] for x in continuity)}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--metric-solver',choices=['gram','svd'],default='gram')
    p.add_argument('--error-control',choices=['coordinates','physical'],default='coordinates')
    p.add_argument('--metric-cutoff',type=float,default=1e-12)
    p.add_argument('--rtol',type=float,default=3e-10);p.add_argument('--atol',type=float,default=3e-12)
    p.add_argument('--max-step',type=float,default=.02);p.add_argument('--compare-directory',type=Path)
    args=p.parse_args()
    settings={key:getattr(args,key) for key in ('metric_cutoff','rtol','atol','max_step','compare_directory')}
    args.output.mkdir(parents=True,exist_ok=False)
    tasks=[('radial_cc',0),('common',0),('direct',2),('witness',2),('Hamiltonian',2)]
    (args.output/'protocol.json').write_text(json.dumps({'classification':'Post-pilot numerical refinement, not additional forecasts.',
      'models':tasks,'metric_solver':args.metric_solver,'error_control':args.error_control,
      'settings':settings,
      'changes':'See explicit settings for tolerances, cutoff and maximum step. Direct SVD uses the same squared-singular-value cutoff as the Gram solver.',
      'physical_invariant':'Each transfer-count sector rate from the variational tangent should agree with the true Hamiltonian current on that same variational state.'},indent=2,default=str))
    results=[]
    with ProcessPoolExecutor(max_workers=5) as pool:
        for f in as_completed([pool.submit(job,(*x,args.output,args.metric_solver,args.error_control,settings)) for x in tasks]):results.append(f.result())
    (args.output/'validation.json').write_text(json.dumps(results,indent=2))
