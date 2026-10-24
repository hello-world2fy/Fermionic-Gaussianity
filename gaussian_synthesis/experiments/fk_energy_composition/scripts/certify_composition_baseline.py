#!/usr/bin/env python3
"""Strong analytic composition-energy control, available equally to all methods."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mpmath import iv,mp
from fk6.paths import PREVIOUS
from fk6.io import source_manifest,write_json
from fk5.two_orbit import ground_root_enclosure,interval_fraction
from fk5.window_sharp_bounds import exact_hopping_coordinates
from fk4.certificates import _bounds


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False);write_json(out/'manifest.json',source_manifest(out/'source_snapshot'))
    initial=json.loads((PREVIOUS/'results/initial_certificates_02/summary.json').read_text())
    two=json.loads((PREVIOUS/'results/two_orbit_02/summary.json').read_text())['construction']['exact_construction_cost_interval'][1]
    iv.dps=100;mp.dps=120;rows=[]
    for t in [11,20]:
        spectrum=next(r for r in initial['spectra'] if r['t']==t)
        q=exact_hopping_coordinates(t,spectrum['ground_energy_interval'])
        lo,hi=ground_root_enclosure(t,spectrum['ground_energy_interval'])
        E=iv.mpf([interval_fraction(lo).a,interval_fraction(hi).b]);corr=-14-4*t-E
        U=two if t==20 else next(r for r in initial['A1_rows'] if r['t']==t)['cost_upper']
        for error in [0,.01]:
            q4=iv.mpf(max(0.,_bounds(q[1]-iv.mpf(error))[0]));q8=iv.mpf(max(0.,_bounds(q[2]-iv.mpf(error))[0]))
            X1=q4/iv.sqrt(iv.mpf(3)/2)
            X2=iv.mpf(max(_bounds(4*q8)[0],_bounds(X1*X1/iv.mpf(U))[0]))
            A=(18*X2+4*(t-9)*X1)/corr;lower=max(0.,_bounds(A)[0])
            off=3*mp.sqrt(14);ev,v=mp.eigsy(mp.matrix([[-14-4*t,-off,0],[-off,-50,-off],[0,-off,-14+4*t]]))
            qexact=v[:,0]*mp.sign(v[0,0]);first=max(0,qexact[1]-mp.mpf(error))/mp.sqrt(mp.mpf(3)/2)
            second=max(4*max(0,qexact[2]-mp.mpf(error)),first**2/mp.mpf(U))
            expected=(18*second+4*(t-9)*first)/(-14-4*t-ev[0]);assert mp.mpf(lower)<=expected
            rows.append(dict(t=t,U=U,hilbert_error=error,composition_energy_lower=lower,
                uses_distribution=False,strict_global_atom_bound=True,independent_120_digit_check=True))
            print('analytic control',t,'epsilon',error,'A>=',lower,flush=True)
    write_json(out/'summary.json',dict(status='complete',rows=rows,
        registration_timing='Added as a stronger analytic control after M2 pilot, before any G3 success comparison; not falsely labelled preregistered.',
        derivation='For Xj=sum |c| x^j: X1>=q4/sqrt(3/2), X2>=4q8 and X2>=X1^2/U. Then A>=[18X2+4(t-9)X1]/(EG-E0). Error allowances subtract epsilon from q4,q8.',
        parent_global_theorem='../fk_replica_hierarchy/fk4/energy_windows.py and ../fk_optimal_geometry/NORMALIZED_WINDOW_BOUNDS.md',
        not_distribution_discovery=True))


if __name__=='__main__':main()
