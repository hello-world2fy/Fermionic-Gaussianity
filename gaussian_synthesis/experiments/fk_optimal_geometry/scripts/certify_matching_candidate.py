#!/usr/bin/env python3
"""Bounded global-normalization attempt for the two-orbit matching candidate."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk5.direction_certification import certify_direction
from fk5.two_orbit import arc_dual
from fk5.paths import PREVIOUS
from fk4.exact_cartan import ExactCartan
from fk5.io import source_manifest,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    initial=json.loads(Path('results/initial_certificates_02/summary.json').read_text())
    construction=json.loads(Path('results/two_orbit_02/summary.json').read_text())['construction']
    a=sum(construction['main_arc_parameter_interval'])/2;z=arc_dual(a)
    row=certify_direction(z,20,4,next(r for r in initial['spectra'] if r['t']==20),ExactCartan(PREVIOUS/'results/cartan_exact_k4_01/proof.json'))
    row.update(status='bounded_global_support_attempt_complete',primal_dual_matching_proved=False,
               comparison_cost_upper=construction['exact_construction_cost_interval'][1])
    write_json(args.output/'summary.json',row);np.savez_compressed(args.output/'candidate.npz',z=z,a=a)
    print('Global candidate support upper:',row['global_support_upper'],'matching is not proved.')


if __name__=='__main__':main()
