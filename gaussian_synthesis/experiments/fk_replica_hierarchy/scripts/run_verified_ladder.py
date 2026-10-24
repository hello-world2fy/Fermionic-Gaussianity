#!/usr/bin/env python3
"""Run an explicitly declared order ladder, advancing only after both gates.

Every child completes, artifacts are independently verified, and the five-
seed reconstruction/mass gate is read before starting the next order. A
failed gate ends this ladder; it is never automatically relabeled or skipped.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fk4.io import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--t',type=float,required=True);p.add_argument('--initial-order',type=int,required=True)
    p.add_argument('--orders',type=int,nargs='+',required=True);p.add_argument('--device',type=int,required=True)
    p.add_argument('--count',type=int,default=524288);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--suffix',default='01');p.add_argument('--wait-source',action='store_true')
    args=p.parse_args()
    if args.state.exists():raise FileExistsError('inspect existing ladder state before restarting')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_NUM_THREADS='12')
    current=args.source;order=args.initial_order;completed=[]
    def checkpoint(status,**extra):write_json(args.state,{'status':status,'device':args.device,'t':args.t,'current_source':str(current),'source_order':order,'declared_orders':args.orders,'completed':completed,**extra})
    def verify(directory):
        checkpoint('verifying',verifying_run=str(directory))
        if not (directory/'verification.json').exists():
            subprocess.run([sys.executable,'scripts/verify_results.py',str(directory)],check=True,env=env)
        v=json.loads((directory/'verification.json').read_text());assert v['artifact_verification_passed']
        s=json.loads((directory/'summary.json').read_text())
        if not s['next_order_gate_pass']:
            checkpoint('stopped_at_failed_empirical_gate',failed_run=str(directory))
            raise SystemExit(2)
        subprocess.run([sys.executable,'scripts/make_report.py',str(directory)],check=True,env=env)
    while True:
        file=current/'summary.json'
        if file.exists() and json.loads(file.read_text())['status']=='complete':break
        if not args.wait_source:raise ValueError('source not complete')
        checkpoint('waiting_for_source_completion')
        time.sleep(3)
    verify(current)
    for next_order in args.orders:
        out=Path(f'results/fk_k{next_order}_radial_t{args.t:g}_n{args.count}_{args.suffix}')
        if out.exists():raise FileExistsError(f'{out} exists; inspect it instead of duplicating')
        command=[sys.executable,'scripts/run_fk_gate.py','--output',str(out),'--t',str(args.t),
                 '--k',str(next_order),'--count',str(args.count),'--rounds','2',
                 '--time-reversal','--full-distribution','--cartan-anchors',
                 'results/cartan_exact_k3_01','results/cartan_exact_k4_01',
                 '--initial-from',str(current),'--initial-order',str(order),
                 '--backend','cuda','--device',str(args.device),'--radial-elimination','--uncompressed-distributions']
        checkpoint('running',running_order=next_order,running_output=str(out),command=command)
        with Path(str(out)+'.log').open('x') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            checkpoint('running',running_order=next_order,running_output=str(out),pid=proc.pid,command=command)
            code=proc.wait()
        if code:
            checkpoint('child_failed',exit_code=code,failed_run=str(out));raise SystemExit(code)
        verify(out)
        completed.append(str(out));current=out;order=next_order
    checkpoint('complete')


if __name__=='__main__':main()
