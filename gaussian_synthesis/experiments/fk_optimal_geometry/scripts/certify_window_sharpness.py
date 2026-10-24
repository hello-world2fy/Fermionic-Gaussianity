#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fk5.io import source_manifest,write_json
from fk5.window_sharp_bounds import certify_window
from fk5.excitation_geometry import exact_cayley_pairing_proof
from fk.model import build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    proof,_,_=exact_cayley_pairing_proof(build_model())
    write_json(args.output/'exact_Cayley_proof.json',proof)
    source=json.loads(Path('results/initial_certificates_02/summary.json').read_text());rows=[]
    for row in source['D1_D2_rows']:
        spectrum=next(r for r in source['spectra'] if r['t']==row['t'])
        result=certify_window(row,spectrum);rows.append(result)
        print(result['t'],result['delta'],result['budget_multiplier'],'Fup',result['fidelity_upper'],'Elo',result['best_energy_error_lower'],flush=True)
    write_json(args.output/'summary.json',dict(status='complete',rows=rows,independent_verification_required=True))


if __name__=='__main__':main()
