#!/usr/bin/env python3
"""Explicit per-GPU job queue; each child writes an independent result directory."""
from pathlib import Path
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fk4.io import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--devices',type=int,nargs='+',required=True)
    p.add_argument('--cpu-threads-per-job',type=int,default=8)
    args=p.parse_args()
    pending=json.loads(args.queue.read_text())
    running={}
    finished=[]
    if args.state.exists():
        raise FileExistsError('queue state exists; inspect it before starting another scheduler')
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_NUM_THREADS=str(args.cpu_threads_per_job))
    while pending or running:
        for device in args.devices:
            if device in running:
                proc,job,handle=running[device]
                code=proc.poll()
                if code is not None:
                    handle.close()
                    job.update(exit_code=code,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    finished.append(job)
                    del running[device]
                    print('finished',job['name'],'device',device,'exit',code,flush=True)
            if device not in running and pending:
                job=pending.pop(0)
                handle=open(job['log'],'x',buffering=1)
                command=[sys.executable,*job['arguments'],'--backend','cuda','--device',str(device)]
                proc=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env)
                job.update(device=device,pid=proc.pid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command)
                running[device]=(proc,job,handle)
                print('started',job['name'],'device',device,'pid',proc.pid,flush=True)
        write_json(args.state,{'status':'running' if pending or running else 'complete',
                   'pending':pending,'running':[j for _,j,_ in running.values()],'finished':finished})
        if pending or running:
            time.sleep(2)
    if any(j['exit_code'] for j in finished):
        raise SystemExit(1)


if __name__=='__main__':
    main()
