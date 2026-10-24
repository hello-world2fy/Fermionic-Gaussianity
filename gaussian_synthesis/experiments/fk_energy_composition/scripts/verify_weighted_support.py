#!/usr/bin/env python3
"""Independent 120-digit audit of every covering cell and physical lower bound."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mpmath import mp
from fk6.io import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);args=p.parse_args()
    path=args.input/'summary.json';data=json.loads(path.read_text());mp.dps=120;checked=0
    for r in data['rows']:
        t=mp.mpf(r['t']);penalty=mp.mpf(r['penalty']);a=3*mp.sqrt(14)
        eig,vec=mp.eigsy(mp.matrix([[-14-4*t,-a,0],[-a,-50,-a],[0,-a,-14+4*t]]))
        q=vec[:,0]*mp.sign(vec[0,0]);corr=-14-4*t-eig[0]
        z=[mp.mpf(v) for v in r['real_z']]
        transform=mp.matrix([[mp.mpf(1)/4,-1/mp.sqrt(2),mp.sqrt(7)/4],
            [mp.sqrt(14)/4,0,-mp.sqrt(2)/4],[mp.mpf(1)/4,1/mp.sqrt(2),mp.sqrt(7)/4]])
        w=transform*mp.matrix(z);norm=mp.sqrt(sum(v*v for v in z));N=r['partition_steps']
        assert N==len(r['global_partition_support_uppers']) and t>=9 and penalty>=0 and corr>0
        for j,upper in enumerate(r['global_partition_support_uppers']):
            left=mp.mpf(2*j)/N;right=mp.mpf(2*(j+1))/N
            h4=min(mp.mpf(1),mp.sqrt(mp.mpf(3)/2)*right)
            if right<mp.mpf('.5'):h4=min(h4,3*right/mp.sqrt(14*(1-2*right)))
            numerator=min(norm,abs(w[0])*(1-left/2)**2+abs(w[1])*h4+abs(w[2])*right**2/4)
            denominator=1+penalty*(18*left**2+4*(t-9)*left)/corr
            assert mp.mpf(upper)>=numerator/denominator,(r['t'],r['k'],j)
            checked+=1
        assert r['gaussian_weighted_support_upper']==max(r['global_partition_support_uppers'])
        support=mp.mpf(r['gaussian_weighted_support_upper']);overlap=sum(a*b for a,b in zip(w,q))
        assert mp.mpf(r['weighted_cost_lower'])<=overlap/support
        if penalty>0:
            U=mp.mpf(r['reference_plain_budget'])
            assert mp.mpf(r['composition_energy_lower'])<=max(0,(overlap/support-U)/penalty)
            assert mp.mpf(r['composition_energy_lower_with_hilbert_error_001'])<=max(0,((overlap-norm/100)/support-U)/penalty)
    write_json(args.input/'verification.json',dict(passed=True,physical_witnesses=len(data['rows']),
        checked_global_covering_cells=checked,independent_decimal_digits=120,
        global_proof_scope='Full Gaussian dictionary via previously proved coercivity, pair-number bounds and complete rational covering; odd rays have zero overlap',
        sampled_or_local_maximum_used=False,source_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    print('verified',len(data['rows']),'global weighted witnesses;',checked,'covering cells',flush=True)


if __name__=='__main__':main()
