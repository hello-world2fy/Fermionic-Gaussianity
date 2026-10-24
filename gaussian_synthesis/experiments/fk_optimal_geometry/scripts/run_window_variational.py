#!/usr/bin/env python3
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk5.io import source_manifest,write_json
from fk5.window_variational import make_dictionary,solve_budgets
from fk.model import build_model


def worker(output,rows,random_directions):
    t,delta=rows[0]['t'],rows[0]['delta'];out=Path(output)/f't{t:g}_delta{delta:g}'
    out.mkdir(exist_ok=False);model=build_model()
    try:
        atoms,dictionary=make_dictionary(model,t,delta,random_directions=random_directions)
        np.savez_compressed(out/'dictionary.npz',atoms=atoms,features=dictionary.pop('features'))
        write_json(out/'dictionary.json',dictionary)
        results,arrays=solve_budgets(model,atoms,t,[r['coefficient_budget'] for r in rows])
        for result,original in zip(results,rows):
            result.update(delta=delta,budget_multiplier=original['budget_multiplier'],original_strict_bounds=original)
            assert result['fidelity_optimized']<=original['fidelity_upper']+1e-8
            assert result['energy_error_optimized']>=original['ground_energy_error_lower']-1e-8
        np.savez_compressed(out/'solutions.npz',**arrays)
        write_json(out/'summary.json',dict(status='complete',rows=results))
        print(t,delta,'complete',[(r['fidelity_optimized'],r['energy_error_optimized']) for r in results],flush=True)
        return results
    except Exception:
        write_json(out/'failure.json',dict(traceback=traceback.format_exc()));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12);p.add_argument('--random-directions',type=int,default=32)
    p.add_argument('--t',type=float,nargs='+',default=[9,11,20]);p.add_argument('--delta',type=float,nargs='+',default=[.001,.01,.1,1.])
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    write_json(args.output/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    source=json.loads(Path('results/initial_certificates_02/summary.json').read_text())['D1_D2_rows']
    groups=[[r for r in source if r['t']==t and r['delta']==d] for t in args.t for d in args.delta]
    rows=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(worker,str(args.output),g,args.random_directions) for g in groups]
        for future in as_completed(futures):
            rows.extend(future.result());write_json(args.output/'summary.json',dict(status='running',rows=rows))
    write_json(args.output/'summary.json',dict(status='complete',rows=rows,elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':main()
