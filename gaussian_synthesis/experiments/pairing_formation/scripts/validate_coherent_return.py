#!/usr/bin/env python3
"""Numerical cross-check and exact same-initial-state channel deletion control."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import PairModel
from pairing_formation.frozen import propagate
from pairing_formation.coherent_return import factor_observables


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]/'results'
    frozen=root/'coherent_return_pilot_01'
    holdout=root/'coherent_return_holdouts_01'
    records=json.loads((holdout/'predictions_saved.json').read_text())
    rows=[]
    for L in (12,16):
        for source in (('direct','rpa','witness') if L==12 else ('direct','rpa')):
            selected=[r for r in records if r['model']['L']==L and r['model']['source']==source
                      and r['case']['G']==3.6 and r['case']['spectrum']=='cubic']
            two=next(r for r in selected if r['model']['q']==2)
            three=next(r for r in selected if r['model']['q']==3)
            d2=np.load(frozen/two['model']['frozen_file'])
            d3=np.load(frozen/three['model']['frozen_file'])
            B2,B3=d2['basis'],d3['basis']
            initial=B2@d2['coefficients']
            c3=B3.conj().T@initial
            initial_difference=float(np.linalg.norm(B3@c3-initial))
            assert initial_difference<2e-11
            model=PairModel(np.array(three['case']['energies']),3.6)
            H=B3.conj().T@(model.H@B3)
            coefficients=propagate(H,c3,[2.])[0]
            independent=expm_multiply(-2j*H,c3,traceA=-2j*np.trace(H))
            check=float(np.linalg.norm(coefficients-independent))
            assert check<2e-10
            pred3=B3@coefficients
            saved=np.load(holdout/two['file'])
            pred2=saved['states'][np.argmin(abs(saved['times']-2.))]
            exact=model.evolve([2.],np.column_stack((d2['training'],initial)))[0]
            truth,own=exact[:,0],exact[:,1]
            truth_obs=factor_observables(model.m,truth)
            own_obs=factor_observables(model.m,own)
            metrics={}
            for name,pred in [('two_channels',pred2),('three_channels_same_initial',pred3)]:
                obs=factor_observables(model.m,pred)
                metrics[name]={'echo_excess':obs['echo_excess'],
                               'echo_error':obs['echo_excess']-truth_obs['echo_excess'],
                               'dynamics_only_echo_error':obs['echo_excess']-own_obs['echo_excess'],
                               'dephased_echo_excess':.75**2*obs['echo_difference']-.04*obs['mean'],
                               'count_TV':float(np.sum(abs(np.array(obs['count'])-truth_obs['count']))/2),
                               'fidelity':float(abs(np.vdot(truth,pred))**2)}
            rows.append({'L':L,'source':source,'case':three['case']['name'],'elapsed':2.,
                         'initial_state_difference':initial_difference,'eigh_vs_expm_difference':check,
                         'exact_echo_excess':truth_obs['echo_excess'],
                         'exact_dephased_echo_excess':.75**2*truth_obs['echo_difference']-.04*truth_obs['mean'],
                         'metrics':metrics})
    (args.output/'validation.json').write_text(json.dumps({'scope':'Post-holdout mechanism control, not new independent forecasts. Q2 initial state is identical in both Q2 and Q3 evolutions.',
                                                          'records':rows},indent=2))
    print(json.dumps({'controls':len(rows),'max_initial_difference':max(r['initial_state_difference'] for r in rows),
                      'max_eigh_expm_difference':max(r['eigh_vs_expm_difference'] for r in rows)}),flush=True)
