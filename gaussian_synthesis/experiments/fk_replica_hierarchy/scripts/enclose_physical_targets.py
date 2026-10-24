#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import os
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4 import paths
from fk4.io import source_manifest,write_json
from fk4.state_enclosure import enclose_ground_state
from fk4.certificates import _bounds
from mpmath import iv
from fk.model import build_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('canonical',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    summary=json.loads((args.canonical/'summary.json').read_text())
    model=build_model()
    rows=[]
    for source in summary['rows']:
        t,k=source['t'],source['k']
        data=np.load(args.canonical/f't{t:g}_k{k}_frozen.npz')
        result=enclose_ground_state(model,t,data['target'])
        error=iv.mpf(result['saved_target_to_exact_ground_state_distance_upper'])
        L=iv.mpf(source['L_certified_saved_target'])-iv.mpf(source['witness_hilbert_norm_upper'])*error
        result.update(t=t,k=k,L_certified_exact_ground_state=_bounds(L)[0],
                      source_canonical=str(args.canonical.resolve()))
        if 'repaired_decomposition_cost_upper' in source:
            U=iv.mpf(source['repaired_decomposition_cost_upper'])+iv.sqrt(128)*error
            result['repaired_cost_upper_exact_ground_state']=_bounds(U)[1]
        rows.append(result)
        write_json(args.output/'summary.json',{'status':'running','rows':rows})
        print(t,'state error upper',result['saved_target_to_exact_ground_state_distance_upper'],flush=True)
    write_json(args.output/'summary.json',{'status':'complete','rows':rows})


if __name__=='__main__':
    main()
