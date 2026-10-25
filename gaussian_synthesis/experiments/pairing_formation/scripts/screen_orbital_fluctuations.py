#!/usr/bin/env python3
"""Bounded local search for arbitrary-orbital PG count-bound violations.

Positive feasible violations are certificates under the existing analytic
bound. Failure of local search is never a global nonviolation certificate.
This is a retrospective diagnostic of saved target states, not a prediction.
"""
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
from pairing_formation.pairing_channels import angular_model
from pairing_formation.orbital_fluctuations import reduced_moments,orbital_count,torch_count
from pairing_formation.projected_count_bound import moment_fidelity_bound


def job(task):
    kind,L,t,rank,device,output=task
    import sys
    sys.path.append('/usr/local/lib/python3.11/dist-packages')
    import torch
    torch.set_num_threads(1);torch.cuda.set_device(device)
    dev=f'cuda:{device}';start=time.monotonic();root=Path(__file__).resolve().parents[1]/'results'
    if kind=='uniform':
        model=picket_fence(L,2.);target=np.load(root/f'replica2_L12_G2_01/t{t:g}.npz')['target']
    else:
        model,_=angular_model(L,4.);target=np.load(root/f'nonlinear_channels_01/L{L}_eta1_exact.npz')['states'][int(round(t/.05))]
    moments=reduced_moments(model,target)
    tm={k:torch.as_tensor(v,device=dev) for k,v in moments.items()}
    rng=np.random.default_rng(720+device)
    count=64;n=2*L
    samples=rng.normal(size=(count,n,rank))+1j*rng.normal(size=(count,n,rank))
    # Include near energy-count and near least-occupied subspaces, as well as
    # fully mixed random starts. None use an exact future state or an optimizer
    # bound on the free-state overlap.
    orders=[np.argsort(np.repeat(moments['occupations'],2)),np.r_[np.arange(L,n),np.arange(L)]]
    for j in range(count//2):
        samples[j]=np.eye(n)[:,orders[j%2][:rank]]+.08*samples[j]
    samples[0]=np.eye(n)[:,orders[0][:rank]]
    samples[1]=np.eye(n)[:,orders[1][:rank]]
    Z=torch.tensor(samples,device=dev,dtype=torch.complex128,requires_grad=True)
    optimizer=torch.optim.Adam([Z],lr=.04)
    best=torch.full((count,),-torch.inf,device=dev,dtype=torch.float64)
    bestP=torch.empty((count,n,n),device=dev,dtype=torch.complex128)
    history=[]
    def evaluate():
        Q=torch.linalg.qr(Z,mode='reduced').Q
        P=Q@Q.mH
        mean,var=torch_count(tm,P)
        return var-mean,P
    for iteration in range(800):
        optimizer.zero_grad();value,P=evaluate()
        with torch.no_grad():
            improve=value>best;best[improve]=value[improve];bestP[improve]=P[improve]
        (-value.sum()).backward();optimizer.step()
        if iteration%100==0:history.append({'iteration':iteration,'max_excess':float(best.max())})
    # Refine all starts jointly; their coordinates and objectives are separable.
    optimizer=torch.optim.LBFGS([Z],max_iter=180,tolerance_grad=1e-10,tolerance_change=1e-13,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();value,P=evaluate()
        with torch.no_grad():
            improve=value>best;best[improve]=value[improve];bestP[improve]=P[improve]
        loss=-value.sum();loss.backward();return loss
    optimizer.step(closure)
    projectors=bestP.detach().cpu().numpy();values=best.detach().cpu().numpy()
    selected=int(values.argmax());P=projectors[selected]
    result=orbital_count(moments,P)
    direct_error=abs(result['excess']-values[selected]);assert direct_error<2e-10
    bound=moment_fidelity_bound(result['mean'],result['variance'],min(rank/2,model.r))
    tag=f'{kind}_L{L}_t{t:g}_rank{rank}'
    np.savez_compressed(output/f'{tag}.npz',projectors=projectors,excesses=values,
                        **moments,target=target)
    report={'name':tag,'model':kind,'L':L,'time':t,'rank':rank,'device':device,
            'starts':count,'best_found':result,'bound':bound,'selected_start':selected,
            'max_Torch_numpy_difference':direct_error,'all_excesses':values.tolist(),
            'history':history,'seconds':time.monotonic()-start,
            'local_search_not_global_optimality_certificate':True,
            'scope':'Retrospective observable search; a positive witness excludes all normalized total-number-projected pure Gaussians by the analytic arbitrary-orbital bound. It does not demonstrate canonical-specific discovery.'}
    (output/f'{tag}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('name','best_found','bound','seconds')}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    cases=[('uniform',12,2.15,k) for k in (4,6,8,10,12)]+[('uniform',12,1.,12),('angular',14,2.,8),('angular',14,2.,14)]
    (args.output/'protocol.json').write_text(json.dumps({'scope':'Bounded retrospective observable search, not predictive evidence.',
       'cases':cases,'starts_per_case':64,'optimizer':'Batched complex QR projectors; Adam 800 iterations plus L-BFGS 180 iterations.',
       'positive_control':'Uniform L12 G2 t1 upper-band count is already known to violate the bound.',
       'physical_operator':'m=c^dagger P c /2, with a general 2L-mode orthogonal projector P; no seniority projection of the operator.',
       'failure_rule':'A nonpositive best found value does not prove a global absence of violations.'},indent=2))
    results=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(job,(*case,d,args.output)) for d,case in enumerate(cases)]):results.append(f.result())
    (args.output/'summary.json').write_text(json.dumps(results,indent=2))
