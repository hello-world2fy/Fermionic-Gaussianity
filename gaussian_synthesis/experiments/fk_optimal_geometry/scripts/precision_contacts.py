#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fk5.io import source_manifest,write_json
from fk5.precision_geometry import stationary_data,mp_contact_hessian
from fk.model import build_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--digits',type=int,nargs='+',default=[50,90])
    p.add_argument('--names',nargs='+',default=['t0','t9','t11','t20_joint','t20_main','t20_secondary'])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    model=build_model();rows=[];started=time.monotonic()
    for name in args.names:
        for digits in args.digits:
            w,g,a=stationary_data(model,name,digits)
            row=mp_contact_hessian(w,g)
            row.update(name=name,arc_parameter=str(a))
            rows.append(row)
            write_json(args.output/f'{name}_d{digits}.json',row)
            write_json(args.output/'summary.json',dict(status='running',rows=rows))
            print(name,digits,'gradient',row['horizontal_gradient_norm'],
                  'eig6',row['eigenvalues'][6],flush=True)
    write_json(args.output/'summary.json',dict(status='complete',rows=rows,
               elapsed_seconds=time.monotonic()-started,
               high_precision_is_not_interval_certification=True))


if __name__=='__main__':main()
