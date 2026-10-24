#!/usr/bin/env python3
"""Certify already frozen discoveries; compare only after their freeze."""
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk6.paths import ROOT,PREVIOUS
from fk6.io import source_manifest,write_json
from fk6.restricted_certificate import certify


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=128)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    inputs=json.loads((args.input/'summary.json').read_text())
    initial=json.loads((PREVIOUS/'results/initial_certificates_02/summary.json').read_text())
    reference=json.loads((ROOT/'results/analytic_reference_02/summary.json').read_text())['rows']
    rows=[];started=time.monotonic()
    for source in inputs['rows']:
        path=args.input/(source['key']+'_frozen.npz')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=source['frozen_sha256']:raise ArithmeticError('Frozen artifact changed')
        f=np.load(path);ell=f['witness'];t=source['t']
        spectrum=next(s for s in initial['spectra'] if s['t']==t)
        row=certify(ell,t,source['delta'],source['budget'],spectrum,args.steps)
        ref=next(r for r in reference if r['t']==t and r['budget_multiplier']=='1'
                 and abs(float(__import__('fractions').Fraction(r['correlation_fraction']))-source['fraction'])<1e-12)
        row.update(key=source['key'],k=source.get('k'),seed=source['seed'],fraction=source['fraction'],
                   method=source.get('method','canonical_energy_bins'),frozen_sha256=digest,
                   frozen_source=str(path.resolve()),discovery=source['discovery'],
                   analytic_e4_mass_lower=ref['high_plain_coefficient_mass_lower']['stronger']['0'],
                   analytic_joint_fidelity_upper=ref['stronger_normalized_physical_bounds']['fidelity_upper'],
                   analytic_reference_delta=ref['delta_used_outward'],
                   reference_delta_difference=source['delta']-ref['delta_used_outward'])
        row['mass_ratio_to_analytic_e4']=None if row['analytic_e4_mass_lower']==0 else row['high_plain_coefficient_mass_lower']['0']/row['analytic_e4_mass_lower']
        rows.append(row);write_json(args.output/(source['key']+'.json'),row)
    write_json(args.output/'summary.json',dict(status='complete_pending_independent_audit',rows=rows,
        source_direction_freeze_completed_utc=inputs.get('direction_freeze_completed_utc'),
        source_summary_sha256=hashlib.sha256((args.input/'summary.json').read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-started,independent_gain_claimed=False))
    for t in [11,20]:
        group=[r for r in rows if r['t']==t and r['fraction']==.25]
        if group:print(t,'quarter-window mass range',min(r['high_plain_coefficient_mass_lower']['0'] for r in group),
                       max(r['high_plain_coefficient_mass_lower']['0'] for r in group),'analytic',group[0]['analytic_e4_mass_lower'],flush=True)


if __name__=='__main__':main()
