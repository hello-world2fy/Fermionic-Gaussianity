#!/usr/bin/env python3
"""Measure canonical toy component weights after freezing exact-moment directions."""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk5.io import source_manifest,write_json
from fk4.reference import log_dimension
from sample_fixed_gpu import device_worker


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--moments',type=Path,default=Path('results/two_contact_toy_01'))
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    orders=[8,16,32,64,128];seeds=[52001,52002,52003,52004,52005]
    config=dict(output=str(out),count=65536,sweeps=2,orders=orders,seeds=seeds,devices=list(range(8)))
    write_json(out/'config.json',config)
    write_json(out/'manifest.json',source_manifest(out/'source_snapshot'))
    specs=[]
    for k in orders:
        source=args.moments/f'k{k}.json';theory=json.loads(source.read_text())
        r=float(theory['canonical_direction_ratio'])
        w=np.zeros(128,complex);w[0]=1;w[-1]=r
        name=f'toy_canonical_k{k}'
        np.savez_compressed(out/f'{name}_frozen.npz',witness=w,z=np.empty(0))
        specs.append(dict(name=name,t=0,orders=[k],reference_support=1.,support_is_exact=True,
            support_status='Exact max(|alpha|,|beta|) for the full Gaussian dictionary.',
            anchors={1.:float(np.log(1+r*r)-log_dimension(8,1))},
            source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    write_json(out/'frozen_witnesses.json',specs)
    jobs=[(s,seed) for seed in seeds for s in specs];done=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn')) as pool:
        fs=[pool.submit(device_worker,i,jobs[i::8],config) for i in range(8)]
        for future in as_completed(fs):
            done.extend(future.result());write_json(out/'summary.json',dict(status='running',completed=done))
    write_json(out/'summary.json',dict(status='complete',completed=done,independent_runs=len(done)))


if __name__=='__main__':main()
