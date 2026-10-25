#!/usr/bin/env python3
"""Locate finite-grid angular aliases in the exact Taylor coefficients."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
from pathlib import Path
import json,argparse,math
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for L in (10,14,18):
        model,_=angular_model(L);r=model.r
        V=np.exp(2j*np.pi*np.arange(r)[:,None]*np.arange(r)[None,:]/r)/np.sqrt(r)
        energy=float(np.vdot(model.initial,model.H@model.initial).real)
        vector=model.initial.copy();orders=[]
        for n in range(1,r+1):
            vector=(model.H@vector-energy*vector)/n
            C=V.conj().T@vector[model.blocks[1][1]]@V
            orders.append({'order':n,'absolute_diagonal_coefficients':abs(C.diagonal()).tolist(),
                           'coefficient_vector_norm':float(np.linalg.norm(vector)),
                           'offdiagonal_norm':float(np.linalg.norm(C-np.diag(C.diagonal())))})
        first=[]
        for k in range(r):
            observed=[x['order'] for x in orders if x['absolute_diagonal_coefficients'][k]>1e-11*max(1.,x['coefficient_vector_norm'])]
            # Odd harmonics are represented modulo the odd grid size r.
            odd_min=min(abs(a) for a in range(-r,r+1) if a%2 and a%r==k)
            first.append({'discrete_index':k,'lowest_odd_representative_absolute_value':odd_min,
                          'first_resolved_order':observed[0] if observed else None})
        result={'L':L,'r':r,'orders':orders,'first_orders':first}
        results.append(result);print(json.dumps({'L':L,'first_orders':first}),flush=True)
    (args.output/'evaluation.json').write_text(json.dumps({
       'scope':'Retrospective exact Taylor-coefficient diagnostic. Coefficients use (H-E_F)^n/n!, removing a common phase. Floating-point resolution threshold is declared, not an interval proof.',
       'interpretation':'The grid k=0 component appears when an odd angular harmonic wraps modulo r. It must not be called an independently established continuum s-wave channel or phase.',
       'relative_resolution_threshold':1e-11,'results':results},indent=2))
