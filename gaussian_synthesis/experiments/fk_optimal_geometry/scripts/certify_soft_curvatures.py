#!/usr/bin/env python3
"""Exact relative-rotation Hessian identities and directed soft eigenvalues."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import iv
from fk5.paths import PREVIOUS
from fk5.contact_geometry import exact_relative_rotation_hessian
from fk5.io import source_manifest,write_json
from fk4.certificates import _bounds
from fk.model import build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    proof=exact_relative_rotation_hessian(build_model());iv.dps=80;rows=[]
    for t in [9,11]:
        source=PREVIOUS/f'results/fixed_geometry_01_qualified/t{t}_fixed.npz'
        y=[iv.mpf(float(x)) for x in np.load(source)['z'].real]
        z=y[2]/iv.sqrt(7);C=(y[0]+z)/2;D=(-y[0]+5*z)/4
        a=y[1]/(2*iv.sqrt(2)*D);h=C-D*a*a
        curvature=4*(z-D*a*a)/h
        assert _bounds(a)[0]>0 and _bounds(a)[1]<1 and _bounds(curvature)[0]>0
        boundary=(y[0]-iv.sqrt(y[0]**2+10*y[1]**2))/10
        displacement=y[2]-iv.sqrt(7)*boundary
        rows.append(dict(t=t,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            exact_binary64_witness_coefficients=[float(x) for x in np.load(source)['z'].real],
            stationary_arc_parameter_interval=_bounds(a),
            six_relative_rotation_curvatures_interval=_bounds(curvature),
            curvature_multiplicity_at_least=6,
            change_in_y2_to_zero_curvature_surface_interval=_bounds(-displacement),
            reciprocal_curvature_interval=_bounds(1/curvature),
            global_support_certified=False,
            applies_to_frozen_candidate_not_unknown_exact_optimal_witness=True))
    write_json(args.output/'summary.json',dict(status='complete',exact_identities=proof,rows=rows,
        interval_decimal_digits=80,positive_soft_curvatures_certified=True,
        full_transverse_spectrum_interval_certified=False))
    for r in rows:print(r['t'],r['six_relative_rotation_curvatures_interval'],'delta_y2',r['change_in_y2_to_zero_curvature_surface_interval'],flush=True)


if __name__=='__main__':main()
