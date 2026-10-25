#!/usr/bin/env python3
"""Direct Hamiltonian Taylor coefficients for higher-transfer dark onset."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from math import floor
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.band_transfer import BandTransfer


def job(task):
    r,out=task;begin=time.monotonic()
    target=min(4,(r-1)//2);order=2*target+1
    maximum=min(r,(order+target)//2)
    model=BandTransfer(r,maximum=maximum)
    coefficients=model.taylor(order)
    cc=model.single_cluster_series(coefficients,target)
    rows=[];arrays={}
    for m in range(1,target+1):
        exact=[];single=[]
        for n in range(order+1):
            exact.append(float(np.linalg.norm(model.projected_dark(coefficients[n][m],m))))
            single.append(float(np.linalg.norm(model.projected_dark(cc[n][m],m))))
        leading=next((n for n,value in enumerate(exact) if value>1e-8),None)
        single_leading=next((n for n,value in enumerate(single) if value>1e-8),None)
        if leading is not None:arrays[f'm{m}_first_dark_coefficient']=model.projected_dark(coefficients[leading][m],m)
        rows.append({'m':m,'dark_band_dimension':model.dark_dimensions[m],
            'exact_dark_coefficient_norms':exact,'single_cluster_dark_coefficient_norms':single,
            'first_resolved_order':leading,'single_cluster_first_resolved_order':single_leading,
            'resolution_threshold':1e-8,
            'lower_order_max':max(exact[:leading]) if leading else None})
    # Small-time series is used only to confirm the local power law, with
    # an explicit extra-order comparison to the same matrix-free recurrence.
    extended=BandTransfer(r,maximum=min(r,(order+2+target)//2))
    extra=extended.taylor(order+2)
    times=np.array([.01,.02,.04]);scaling=[]
    for row in rows:
        m,n=row['m'],row['first_resolved_order']
        if n is None:continue
        values=[]
        for t in times:
            a=sum((t**k)*coefficients[k][m] for k in range(order+1))
            b=sum((t**k)*extra[k][m] for k in range(order+3))
            x=model.projected_dark(a,m);y=extended.projected_dark(b,m)
            values.append({'time':float(t),'dark_amplitude_norm':float(np.linalg.norm(y)),
                'scaled_by_leading_power':float(np.linalg.norm(y)/t**n),
                'extra_order_change':float(np.linalg.norm(x-y))})
        scaling.append({'m':m,'values':values})
    result={'r':r,'L':2*r,'G':4.,'shell_gap':1.,'energy_shift':'Fermi expectation -r-G',
        'maximum_transfer':maximum,'maximum_Taylor_order':order,'rows':rows,'local_scaling':scaling,
        'seconds':time.monotonic()-begin,
        'scope':'Floating-point Taylor coefficients, not interval proof or a general-r onset theorem. Single-cluster comparison uses exact m1 intermediate-normalized coefficients, giving it stronger information than a fitted single-cluster trajectory.'}
    np.savez_compressed(out/f'r{r}_coefficients.npz',**arrays)
    (out/f'r{r}_onset.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'r':r,'seconds':result['seconds'],'onsets':[{k:row[k] for k in ('m','dark_band_dimension','first_resolved_order','single_cluster_first_resolved_order')} for row in rows]}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps({'classification':'Operator-based mechanism check, not a canonical discovery or new late-time forecast.',
        'sizes':[5,7,9,11],'candidate':'Higher-transfer dark amplitudes may appear before a permanent lift of even the exact one-transfer amplitude can generate them.',
        'finite_size_caution':'Special highest-occupancy one-dimensional dark sectors may obey additional selection rules; do not extrapolate a universal formula from nonzero ranks.'},indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(job,(r,args.output)) for r in (5,7,9,11)]):f.result()
