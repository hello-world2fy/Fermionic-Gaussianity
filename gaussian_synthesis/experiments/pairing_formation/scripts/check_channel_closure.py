#!/usr/bin/env python3
"""Check the Hamiltonian symmetry alternative before a discovery claim."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import sys
from pathlib import Path
import argparse
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.frozen import channel_basis


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for L,ratio in ((12,1.),(12,.85),(16,1.)):
        model,_=angular_model(L,4.,ratio)
        r=model.r
        fourier=np.exp(2j*np.pi*np.arange(r)[:,None]*np.arange(r)[None,:]/r)/np.sqrt(r)
        odd=fourier[:,1::2]
        projector=odd@odd.conj().T
        B,_=channel_basis(model,projector,r//2)
        HB=model.H@B
        Hred=B.conj().T@HB
        residual=float(np.linalg.norm(HB-B@Hred))
        initial_residual=float(np.linalg.norm(model.initial-B@(B.conj().T@model.initial)))
        rows.append({'L':L,'ratio':ratio,'collective_directions_per_band':r//2,
                     'full_dimension':model.dim,'closed_dimension':B.shape[1],
                     'invariance_residual_Frobenius':residual,
                     'relative_invariance_residual':residual/float(np.linalg.norm(HB)),
                     'initial_state_residual':initial_residual})
        assert residual<2e-9 and initial_residual<2e-11
        print(json.dumps(rows[-1]),flush=True)
    (args.output/'evaluation.json').write_text(json.dumps({'scope':'The angular flat-shell model has a Hamiltonian-derived exact symmetry space. Its availability prevents attributing the odd-harmonic structure solely to canonical synthesis.',
                                                          'records':rows},indent=2))
