#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fk5.io import source_manifest,write_json
from fk5.two_contact_toy import finite_order


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    rows=[]
    for k in [1,2,3,4,8,12,16,24,32,48,64,96,128]:
        row=finite_order(k);rows.append(row)
        write_json(args.output/f'k{k}.json',row)
        print(k,'logq',row['balanced_log_q'],'(2k-1)logr',row['scaled_log_height_ratio'][:22],flush=True)
    write_json(args.output/'summary.json',dict(status='complete',rows=rows,
        target='(3 vacuum + full eight-mode occupation)/sqrt(10)',
        globally_known_atomic_norm='4/sqrt(10)',
        full_dictionary_support='h(alpha vacuum+beta full)=max(|alpha|,|beta|)',
        contact_components=2,contact_dimensions=[0,0]))


if __name__=='__main__':main()
