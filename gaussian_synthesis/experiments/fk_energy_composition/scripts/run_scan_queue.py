#!/usr/bin/env python3
"""Run assigned preregistered (t,k) cells serially on one GPU, without retries."""
from pathlib import Path
import argparse, datetime, json, subprocess, sys, time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',type=int,required=True)
    p.add_argument('--jobs',nargs='+',required=True,help='Pairs t:k')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];rows=[]
    def save():
        state={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'device':args.device,'rows':rows,'automatic_retries':False}
        tmp=args.output/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n')
        tmp.replace(args.output/'state.json')
    for pair in args.jobs:
        t,k=map(int,pair.split(':'));out=root/'results'/f'm2_scan_t{t}_k{k}_01'
        if out.exists():raise FileExistsError(f'Refusing to overwrite or restart {out}')
        command=[sys.executable,str(root/'scripts/run_weighted.py'),'--output',str(out),
                 '--t',str(t),'--k',str(k),'--devices',str(args.device),
                 '--count','131072','--validation-count','262144','--rounds','3',
                 '--moves','96','--penalties','0','.25','.5','1','2','4','8','--compact-particles']
        started=time.monotonic()
        with out.with_suffix('.log').open('w') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            row={'t':t,'k':k,'pid':process.pid,'command':command,'status':'running'}
            rows.append(row);save();code=process.wait()
        row.update(exit_code=code,elapsed_seconds=time.monotonic()-started,
                   status='complete' if code==0 else 'failed');save()
        if code:raise SystemExit(code)


if __name__=='__main__':main()
