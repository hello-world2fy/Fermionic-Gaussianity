#!/usr/bin/env python3
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fk5.io import source_manifest,write_json
from fk5.structure_ablation import prepare,solve_one


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--t',nargs='+',type=float,default=[9,10.9225,11,20]);p.add_argument('--replicates',nargs='+',type=int,default=[1,2,3,4,5])
    p.add_argument('--R',nargs='+',type=int,default=[2,4,8,16,32,64]);p.add_argument('--workers',type=int,default=20)
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    tick=time.monotonic();tasks=prepare(args.output,args.t,args.replicates);rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        pending=[pool.submit(solve_one,str(args.output),row,R) for row in tasks for R in args.R]
        for future in as_completed(pending):
            rows.append(future.result());write_json(args.output/'summary.json',dict(status='running',rows=rows,expected_rows=len(pending)))
    write_json(args.output/'summary.json',dict(status='complete',rows=rows,elapsed_seconds=time.monotonic()-tick,
        cost_constrained_and_unconstrained_results_separate=True,actual_distinct_Gaussian_counts_used=True))


if __name__=='__main__':main()
