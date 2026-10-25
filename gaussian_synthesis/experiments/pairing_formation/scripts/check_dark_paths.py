#!/usr/bin/env python3
"""Resolve the fifth-order dark pair coefficient by Hamiltonian paths."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,time,itertools
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.band_transfer import BandTransfer


def leading(r,G=4.,gap=1.):
    model=BandTransfer(r,G,maximum=3,shell_gap=gap)
    c=model.initial();norms=[];saved=[]
    for order in range(1,6):
        c=[(-1j/order)*x for x in model.apply(c)]
        dark=model.projected_dark(c[2],2)
        norms.append(float(np.linalg.norm(dark)))
        if order==5:saved=dark
    return model,saved,norms


def path_action(model,word):
    m=0;C=np.ones((1,1),complex)
    for kind in word:
        if kind=='U':
            m+=1
            if m>model.maximum:return None
            terms=[]
            for L in model.lowering[m]:
                tmp=L.T@C;terms.append((L.T@tmp.T).T)
            C=-model.G/2*sum(terms)
        elif kind=='D':
            if not m:return None
            terms=[]
            for L in model.lowering[m]:
                tmp=L@C;terms.append((L@tmp.T).T)
            C=-model.G/2*sum(terms);m-=1
        else:
            if not m:return None
            if kind=='F':C=2*model.gap*m*C
            elif kind=='S':C=2*model.G/model.r*m*C
            elif kind=='P':C=-model.G/2*(model.K[m]@C)
            elif kind=='H':C=-model.G/2*(model.K[m]@C.T).T
            else:raise ValueError(kind)
    assert m==2
    return (-1j)**5/120*model.projected_dark(C,2)


def job(task):
    r,out=task;start=time.monotonic()
    model,coefficient,norms=leading(r)
    report={'r':r,'L':2*r,'G':4.,'gap':1.,'seconds':0.,
        'dark_dimensions':model.dark_dimensions[2],
        'coefficient_norms_order1_through5':norms,
        'leading_coefficient_norm':float(np.linalg.norm(coefficient)),
        'leading_probability_coefficient_t10':float(np.linalg.norm(coefficient)**2),
        'scope':'Fifth-order amplitude only. No claim of accurate long-time propagation or thermodynamic population.'}
    if r in (7,9):
        contributions=[];summed=np.zeros_like(coefficient);groups={k:np.zeros_like(coefficient) for k in 'FSPH'}
        for waiting in 'FSPH':
            for word in sorted(set(itertools.permutations('UUUD'+waiting))):
                value=path_action(model,word)
                if value is None:continue
                summed+=value;groups[waiting]+=value
                contributions.append({'word':''.join(word),'norm':float(np.linalg.norm(value)),
                    'parallel_coefficient_real':float((np.vdot(coefficient,value)/np.vdot(coefficient,coefficient)).real),
                    'parallel_coefficient_imag':float((np.vdot(coefficient,value)/np.vdot(coefficient,coefficient)).imag)})
        report['path_contributions']=contributions
        report['one_return_path_sum_error']=float(np.linalg.norm(summed-coefficient))
        report['waiting_groups']={k:{'norm':float(np.linalg.norm(v)),
            'parallel_coefficient_real':float((np.vdot(coefficient,v)/np.vdot(coefficient,coefficient)).real)} for k,v in groups.items()}
        _,changed,_=leading(r,G=2.,gap=3.)
        report['G5_gap_scaling_check_error']=float(np.linalg.norm(changed*(4/2)**5-coefficient))
        # Different shell gap and coupling distinguish pure interaction order
        # from energy-dependent timing. These are coefficient checks, not forecasts.
        np.savez_compressed(out/f'r{r}_coefficients.npz',coefficient=coefficient,
                            **{k:v for k,v in groups.items()})
    report['seconds']=time.monotonic()-start
    (out/f'r{r}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['path_contributions','waiting_groups']}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=7)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps({'sizes_r':[5,7,9,11,15,21,31],
        'order':5,'maximum_intermediate_transfer':3,'target_transfer':2,
        'path_symbols':{'U':'interband pair transfer up','D':'interband pair transfer down',
            'F':'free shell energy','S':'hole self shift','P':'particle-band exchange','H':'hole-band exchange'},
        'path_scope':'All words with three up steps, one down step and one waiting term. Paths with two up steps and three waiting terms remain bright and are tested by comparison to the independent full Taylor coefficient.',
        'purpose':'Determine the first dark-pair source and finite-grid coefficient dependence, not a universal-size or long-time result.'},indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(job,(r,args.output)) for r in (5,7,9,11,15,21,31)]):rows.append(f.result())
    (args.output/'summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x['r']),indent=2))
