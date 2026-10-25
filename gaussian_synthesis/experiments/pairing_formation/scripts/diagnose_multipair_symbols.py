#!/usr/bin/env python3
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,argparse
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.pilot_pair_symbols import digest


def main(out):
    out.mkdir(parents=True,exist_ok=False);source=out.parent/'multipair_symbol_pilot_01';data=np.load(source/'frozen_spaces.npz');exact=np.load(source/'exact_references.npz')
    records=json.loads((source/'frozen_predictions.json').read_text());H=data['H'];capacity=[];checks=[];geometry=[]
    selected=('canonical_full_q16','canonical_cross_copy_q8','canonical_cross_copy_q16','canonical_joint_q16','canonical_single_H_q16','ordinary2_q16','Hamiltonian_q16')
    for row in records:
        path=source/row['file'];assert digest(path)==row['sha256'];saved=np.load(path);V=data[row['method']]
        if row['start']=='restart':
            for t in (2.,2.5):
                reference=exact['states'][round(t*20)];prediction=saved['states'][np.argmin(abs(saved['times']-t))]
                capacity.append({'method':row['method'],'time':t,'dimension':V.shape[1],
                    'subspace_max_fidelity':float(np.linalg.norm(V.conj().T@reference)**2),'propagation_fidelity':float(abs(np.vdot(reference,prediction))**2)})
        if row['method'] in selected:
            initial=data['F'] if row['start']=='Fermi' else data['target'];Hr=V.conj().T@H@V
            states=expm_multiply(-1j*Hr,V.conj().T@initial,start=0,stop=saved['times'][-1]-saved['times'][0],num=len(saved['times']),traceA=-1j*np.trace(Hr))@V.T
            checks.append({'method':row['method'],'start':row['start'],'max_state_difference':float(np.max(np.linalg.norm(states-saved['states'],axis=1)))})
    for method in ('canonical','unoptimized'):
        A=data[method+'_contrast_q8'];J=data[method+'_joint_q16'];R=data[method+'_single_H_q16']
        geometry.append({'method':method,'joint_nesting_residual':float(np.linalg.norm(A-J@(J.conj().T@A))),
            'H_repair_nesting_residual':float(np.linalg.norm(A-R@(R.conj().T@A)))})
    paths=[source/name for name in ('frozen_spaces.npz','exact_references.npz','frozen_predictions.json')]
    report={'classification':'Post-pilot numerical and saved-space capacity checks, not new physical holdouts.',
        'input_hashes':{str(p):digest(p) for p in paths},'capacity':capacity,'independent_exponential_repeats':checks,'nested_deletion':geometry,
        'scope':'Float64 capacities apply only to each saved linear space; no interval or full-dictionary bound.'}
    (out/'diagnostics.json').write_text(json.dumps(report,indent=2));print(json.dumps({'max_exponential_state_difference':max(r['max_state_difference'] for r in checks),'repeats':len(checks),'nested':geometry},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
