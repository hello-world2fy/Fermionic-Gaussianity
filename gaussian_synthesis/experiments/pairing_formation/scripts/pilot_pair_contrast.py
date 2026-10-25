#!/usr/bin/env python3
"""Post-pilot Gaussian-vanishing pair-symbol contrast, with prior controls."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.pilot_pair_symbols import digest,modes,predict,evaluate


def freeze(source,out):
    out.mkdir(parents=True,exist_ok=False);old=np.load(source/'frozen_spaces.npz');base=old['shared']
    data={k:old[k] for k in ('Q','H','target','F','dark','counts','pair_channels')};spectra={};hashes={str(source/'frozen_spaces.npz'):digest(source/'frozen_spaces.npz')}
    dimensions={}
    for method in ('canonical','unoptimized'):
        columns=[]
        for i in range(4):
            path=source/f'{method}_{i}.npz';hashes[str(path)]=digest(path);x=np.load(path)
            keep=x['pairs'][:,0]!=x['pairs'][:,1]
            columns.append((x['anomalous'][keep]-x['full'][keep]/3).T)
        columns=np.column_stack(columns)
        for rank in (8,16):
            name=f'{method}_contrast_q{rank}';V,s=modes(base,columns,rank)
            data[name]=V;spectra[name]=s.tolist();dimensions[name]=V.shape[1]
    np.savez_compressed(out/'frozen_spaces.npz',**data)
    protocol={'classification':'Post-pilot physical-coordinate diagnostic on inspected conditions. No independent new-parameter forecasts.',
        'mechanism':'For i!=j, M_anomalous-M_full/3 =2(Y+2Z)/(3 Dplus). This vanishes for every paired Slater determinant because its two-copy product lies in the Cartan range and has no number-changing or broken-pair matrix entries. The label is not pointwise zero on Gaussian atoms, and its integral is not a target observable marginal.',
        'selection':'Four existing training times; exclude onsite labels. Leading8/16 normalized residual contrast directions. No future target enters selection, but the prior full/split pilot has already been inspected.',
        'input_hashes':hashes,'frozen_sha256':digest(out/'frozen_spaces.npz'),'dimensions':dimensions,
        'baseline_evaluation':str(source/'evaluation.json'),'baseline_evaluation_sha256':digest(source/'evaluation.json'),
        'primary_time':2.,'last_time':2.5,'step':.05,
        'gate':'Same three physical-error comparisons against corresponding unoptimized contrast and prior equal-dimension ordinary operator-history/H Krylov. Original objective unchanged.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'geometry.json').write_text(json.dumps(spectra,indent=2));predict(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('results/pair_symbol_pilot_01'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['freeze','evaluate'],required=True);a=p.parse_args()
    if a.stage=='freeze':freeze(a.source,a.output)
    else:evaluate(a.output)
