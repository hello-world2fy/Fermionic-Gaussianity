#!/usr/bin/env python3
"""Replay every repaired cost; independently check selected covariances at 90 digits.

All-branch validity follows from the documented arithmetic bounds, not from
the spot checks. The latter independently exercise the error bound against
dense signed Majorana operators instead of the sampling permutations.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import mp
from fk4.io import write_json
from fk4.rounding import corrected_cost
from fk4.state_enclosure import enclose_ground_state
from fk.model import build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    s=json.loads((args.run/'summary.json').read_text());assert s['status']=='complete'
    manifest=json.loads((args.run/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((args.run/'source_snapshot'/name).read_bytes()).hexdigest()==digest
    model=build_model();mp.dps=90;operators=[]
    for a in range(16):
        for b in range(a+1,16):
            O=model.gammas[a]@model.gammas[b];rr,cc=np.nonzero(O)
            operators.append((a,b,rr,cc,O[rr,cc]))
    checked=0;spots=0
    for r in s['rows']:
        path=Path(r['source_file']);assert hashlib.sha256(path.read_bytes()).hexdigest()==r['source_sha256']
        data=np.load(path);g=data['finite_kets'];c=data['finite_coefficients'];target=model.target(r['t'])
        enclosure=enclose_ground_state(model,r['t'],target)
        assert enclosure['saved_target_to_exact_ground_state_distance_upper']<=r['physical_target_enclosure']['saved_target_to_exact_ground_state_distance_upper']*(1+1e-14)
        bound=corrected_cost(g,c,target,model.j_matrices,r['physical_target_enclosure']['saved_target_to_exact_ground_state_distance_upper'])
        for key,value in bound.items():
            if isinstance(value,(float,int)):assert value==r[key],(key,value,r[key])
        for index in sorted(set([0,len(g)//2,len(g)-1])):
            v=[mp.mpc(float(x.real),float(x.imag)) for x in model.embed(g[index])]
            q=mp.fsum(abs(x)**2 for x in v);C=mp.matrix(16)
            for a,b,rr,cc,vv in operators:
                value=-mp.fsum(mp.conj(v[i])*v[j]*mp.mpc(float(z.real),float(z.imag)) for i,j,z in zip(rr,cc,vv)).imag/q
                C[a,b]=value;C[b,a]=-value
            defect=C*C+mp.eye(16)
            norm_upper=max(mp.fsum(abs(defect[a,b]) for b in range(16)) for a in range(16))
            assert norm_upper<=mp.mpf(r['normalized_exact_covariance_defect_upper'])
            spots+=1
        checked+=len(g)
        print('verified',r['t'],len(g),flush=True)
    write_json(args.run/'verification.json',{'artifact_verification_passed':True,
        'full_source_branches_replayed':checked,'independent_dense_Majorana_90digit_spot_checks':spots,
        'all_branch_guarantee_source':'Proved IEEE arithmetic bounds and near-Gaussian-to-exact-Gaussian repair, not statistical spot checks.',
        'arithmetic_proof':'GAUSSIAN_ROUNDING_CERTIFICATES.md'})


if __name__=='__main__':main()
