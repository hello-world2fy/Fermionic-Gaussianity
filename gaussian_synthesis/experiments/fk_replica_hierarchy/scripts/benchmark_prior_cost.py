#!/usr/bin/env python3
"""Remeasure the exact analytic prior independently of column optimization.

This is a later timing measurement of identical candidates, not a replacement
of the historical runtime record. Initialization is included, imports excluded.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4.io import source_manifest,write_json
from fk4.columns import independent_arc_dictionary
from fk4.finite_twirl import FiniteSpin7Twirl
from fk.model import build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--columns',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    source=args.columns/'model_and_prior.npz'
    old=np.load(source)['arc_atoms'];rows=[]
    for repeat in range(5):
        started=time.monotonic()
        model=build_model();twirl=FiniteSpin7Twirl(model)
        arc,angles=independent_arc_dictionary(model,len(old))
        indices=np.linspace(1,len(arc)-2,16).round().astype(int)
        pool=np.concatenate([twirl.orbit(arc[j])[0] for j in indices])
        elapsed=time.monotonic()-started
        np.testing.assert_array_equal(arc,old)
        expected=np.concatenate([twirl.orbit(old[j])[0] for j in indices])
        np.testing.assert_array_equal(pool,expected)
        assert pool.shape==(128,128)
        rows.append({'repeat':repeat,'generation_seconds_including_model_and_group':elapsed,
                     'source_arc_bitwise_equal':True,'pool_bitwise_equal':True})
        print(rows[-1],flush=True)
    np.savez_compressed(args.output/'prior_pool.npz',atoms=pool,arc_atoms=arc,arc_angles=angles)
    write_json(args.output/'summary.json',{'status':'complete','rows':rows,
        'mean_generation_seconds':float(np.mean([r['generation_seconds_including_model_and_group'] for r in rows])),
        'original_arc_file':str(source.resolve()),'original_arc_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'actual_gaussian_branches':128,'uses_target_wavefunction':False,'uses_column_optimization':False,
        'timing_scope':'Later repeated wall-time measurement of the identical deterministic prior, including build_model, full Gaussian group verification, 129 analytic arc atoms and 128 actual prior branches; Python imports excluded.',
        'historical_runtime_not_replaced':True})


if __name__=='__main__':main()
