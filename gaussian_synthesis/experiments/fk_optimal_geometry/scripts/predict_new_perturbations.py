#!/usr/bin/env python3
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import datetime
import json
import multiprocessing as mp
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fk5.io import write_json
from fk5.perturbation_transfer import predict_case,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--workers',type=int,default=24);p.add_argument('--limit',type=int)
    args=p.parse_args();registration=json.loads((args.run/'preregistration.json').read_text())
    assert sha(args.run/'frozen_inputs.npz')==registration['frozen_input_sha256']
    assert not (args.run/'evaluation').exists()
    rows=[];pending=[];cases=registration['cases'][:args.limit];start=time.monotonic()
    for case in cases:
        path=args.run/'predictions'/case['key']/'summary.json'
        if path.exists():
            result=json.loads(path.read_text());assert sha(path.parent/'predictions.npz')==result['prediction_file_sha256'];rows.append(result)
        else:pending.append(case)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(predict_case,str(args.run),case) for case in pending]
        for future in as_completed(futures):
            rows.append(future.result());write_json(args.run/'prediction_summary.json',dict(status='running',rows=rows))
    complete=len(rows)==len(registration['cases'])
    write_json(args.run/'prediction_summary.json',dict(status='complete' if complete else 'partial',rows=rows,
        preregistration_sha256=sha(args.run/'preregistration.json'),elapsed_seconds_this_invocation=time.monotonic()-start,
        complete_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_exact_ground_states_accessed=False))
    print('Predicted',len(rows),'of',len(registration['cases']),'cases; complete:',complete,flush=True)


if __name__=='__main__':main()
