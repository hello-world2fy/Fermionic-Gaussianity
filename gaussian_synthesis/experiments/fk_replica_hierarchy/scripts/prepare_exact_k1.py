#!/usr/bin/env python3
"""Analytic first-order canonical witness: q1(u)=||u||² and C1(u)=u."""
from pathlib import Path
import argparse
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import iv
from fk4.io import write_json,source_manifest
from fk4.certificates import _c,_bounds,exact_basis_action
from fk4.state_enclosure import enclose_ground_state
from fk.model import build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--t',type=float,nargs='+',default=[0,5,9,9.1,10,10.9225,11,12,20]);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    model=build_model();rows=[];started=time.monotonic();iv.dps=70
    for t in args.t:
        target=model.target(t);z=model.S.conj().T@target
        q=sum((a*a+b*b for a,b in map(_c,z)),iv.mpf(0))
        upper=_bounds(iv.sqrt(q))[1]
        safe=z/(upper*(1+32*np.finfo(float).eps))
        C=exact_basis_action(z);w=exact_basis_action(safe)
        rho=iv.sqrt(sum(((a-_c(v)[0])**2+(b-_c(v)[1])**2 for (a,b),v in zip(C,target)),iv.mpf(0)))
        L=sum((a*_c(v)[0]+b*_c(v)[1] for (a,b),v in zip(w,target)),iv.mpf(0))
        enclose=enclose_ground_state(model,t,target)
        Lphysical=_bounds(L-iv.mpf(enclose['saved_target_to_exact_ground_state_distance_upper']))[0]
        np.savez_compressed(args.output/f't{t:g}_k1_frozen.npz',z=z,witness=model.S@z,target=target,z_feasible=safe)
        rows.append({'t':t,'k':1,'q_interval':_bounds(q),
            'L_certified_saved_target':_bounds(L)[0],'L_certified_physical_target':Lphysical,
            'full_reconstruction_residual_upper':_bounds(rho)[1],
            'canonical_B_cauchy_upper':_bounds(iv.sqrt(128*q))[1],
            'proof':'The first coherent Cartan projector is the identity on the full even128 spinor space; C1 is the identity map.',
            'physical_target_enclosure':enclose})
    write_json(args.output/'summary.json',{'status':'complete','rows':rows,'elapsed_seconds':time.monotonic()-started,'analytic_identity_certificate':True})


if __name__=='__main__':main()
