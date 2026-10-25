#!/usr/bin/env python3
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,argparse
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.pilot_pair_symbols import digest


def main(out):
    out.mkdir(parents=True,exist_ok=False);root=out.parent;rows=[];refinements=[];hashes={};geometry=[]
    for directory in ('pair_symbol_pilot_01','pair_contrast_01'):
        source=root/directory;data=np.load(source/'frozen_spaces.npz');exact=np.load(source/'exact_references.npz')
        frozen=json.loads((source/'frozen_predictions.json').read_text());H=data['H']
        for p in (source/'frozen_spaces.npz',source/'exact_references.npz',source/'frozen_predictions.json'):hashes[str(p)]=digest(p)
        for row in frozen:
            path=source/row['file'];assert digest(path)==row['sha256'];saved=np.load(path);V=data[row['method']]
            if row['start']=='restart':
                for t in (2.,2.5):
                    state=exact['states'][round(t*20)];pred=saved['states'][np.argmin(abs(saved['times']-t))]
                    rows.append({'directory':directory,'method':row['method'],'time':t,'dimension':V.shape[1],
                        'subspace_max_fidelity':float(np.linalg.norm(V.conj().T@state)**2),
                        'propagation_fidelity':float(abs(np.vdot(state,pred))**2)})
            selected=('canonical_anomalous_q8','canonical_full_q16','canonical_split_q16','canonical_anomalous_H_q16','Hamiltonian_q16','canonical_contrast_q8','canonical_contrast_q16')
            if row['method'] in selected:
                initial=data['F'] if row['start']=='Fermi' else data['target'];Hr=V.conj().T@H@V
                states=expm_multiply(-1j*Hr,V.conj().T@initial,start=0,stop=saved['times'][-1]-saved['times'][0],num=len(saved['times']),traceA=-1j*np.trace(Hr))@V.T
                error=float(np.max(np.linalg.norm(states-saved['states'],axis=1)))
                refinements.append({'directory':directory,'method':row['method'],'start':row['start'],'max_state_difference':error})
        if directory=='pair_symbol_pilot_01':
            for method in ('canonical','unoptimized'):
                A=data[method+'_anomalous_q8'];S=data[method+'_split_q16'];N=S[:,A.shape[1]:]
                geometry.append({'method':method,'nested_anomalous_residual':float(np.linalg.norm(A-S@(S.conj().T@A))),
                    'new_normal_count_profile_trace':[float(np.sum(abs(N[data['counts']==m])**2)) for m in range(8)],
                    'new_normal_dark_trace':float(np.trace(N.conj().T@data['dark']@N).real),
                    'added_normal_dimension':N.shape[1]})
    report={'classification':'Post-pilot numerical and frozen-space capacity diagnostics, not new physical holdouts.',
        'input_hashes':hashes,'capacity':rows,'independent_exponential_repeats':refinements,'nested_deletion_geometry':geometry,
        'capacity_scope':'Best overlap with each saved numerical linear space; floating-point projector calculation, not a general Gaussian or interval-certified bound.'}
    (out/'diagnostics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'max_independent_exponential_state_difference':max(r['max_state_difference'] for r in refinements),'repeats':len(refinements),
        'canonical_time2_capacity':[r for r in rows if r['method'].startswith('canonical') and r['time']==2.],'nested':geometry},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
