#!/usr/bin/env python3
"""Reproduce the L18 symmetry-preserving tight-integration control."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.pair_cc_dynamics import PairCCDynamics,circulant_parameter_map,product_parameter_map

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps({'L':18,'eta':1.,
      'purpose':'Known translation symmetry reduces full CC coordinates from 81 to 9 without changing the invariant trajectory; compare against unrestricted integration. Tightened tolerances /100, step /2, metric cutoff /10.',
      'historical_order':'The archived _01 run was saved before reading its late exact references; a later rerun is a numerical replication, not a new forecast.'},indent=2))
    model,forms=angular_model(18);U,_=np.linalg.qr(forms);times=np.linspace(0.,2.5,51)
    for name,M in [('full_cc',circulant_parameter_map(model.r)),('source_only_cc',product_parameter_map(U,U))]:
        start=time.monotonic();d=PairCCDynamics(model,M,metric_cutoff=1e-12)
        states,coordinates,report=d.evolve(times,rtol=2e-10,atol=2e-12,max_step=.02)
        np.savez_compressed(args.output/(name+'.npz'),times=times,states=states,coordinates=coordinates)
        (args.output/(name+'.json')).write_text(json.dumps(report,indent=2))
        print(name,time.monotonic()-start,report['max_norm_error'],report['max_energy_drift'],flush=True)
    (args.output/'frozen_forecasts.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.npz')},indent=2))
