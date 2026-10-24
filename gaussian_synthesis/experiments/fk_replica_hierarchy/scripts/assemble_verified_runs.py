#!/usr/bin/env python3
"""Index verified single-point runs without copying large Gaussian populations."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fk4.io import write_json,source_manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    summaries=[];provenance=[];seen=set();rows=[]
    for directory in args.runs:
        directory=directory.resolve()
        s=json.loads((directory/'summary.json').read_text())
        v=json.loads((directory/'verification.json').read_text())
        assert s['status']=='complete' and v['artifact_verification_passed']
        for row in s['rows']:
            key=(row['t'],row['k'])
            if key in seen:raise ValueError(f'duplicate physical/order point {key}')
            seen.add(key)
        summaries.append(s)
        provenance.append({'directory':str(directory),'summary_sha256':hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest(),
                           'verification_sha256':hashlib.sha256((directory/'verification.json').read_bytes()).hexdigest()})
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    for directory,s in zip(args.runs,summaries):
        directory=directory.resolve()
        for row in s['rows']:
            key=f"t{row['t']:g}_k{row['k']}"
            shutil.copy2(directory/f'{key}_frozen.npz',args.output/f'{key}_frozen.npz')
            row={**row,'assembled_source_run':str(directory)}
            for obs in row['validation_rows']:
                if 'P_source_file' not in obs:
                    source_file=directory/f"{key}_validation_seed{obs['seed']}.npz"
                    obs['P_source_file']=str(source_file)
                    obs['P_source_key']='P_all_kets'
                    obs['P_source_sha256']=hashlib.sha256(source_file.read_bytes()).hexdigest()
            rows.append(row)
    orders={r['k'] for r in rows}
    if len(orders)!=1:raise ValueError('one replica order per assembled index')
    write_json(args.output/'summary.json',{'status':'complete','rows':rows,'k':next(iter(orders)),
        'next_order_gate_pass':all(s['next_order_gate_pass'] for s in summaries),
        'elapsed_seconds':sum(s['elapsed_seconds'] for s in summaries),
        'timing_scope':'sum of complete upstream single-point runs, not parallel elapsed wall time',
        'assembled_index_not_new_measurements':True,'provenance':provenance})
    write_json(args.output/'config.json',{'output':str(args.output),'assembled_runs':[str(d.resolve()) for d in args.runs]})
    write_json(args.output/'verification.json',{'artifact_verification_passed':True,
        'verification_kind':'composition of explicitly verified immutable source runs; all P file references retain hashes',
        'source_verifications':provenance,'empirical_scientific_gate_passed':all(s['next_order_gate_pass'] for s in summaries)})


if __name__=='__main__':main()
