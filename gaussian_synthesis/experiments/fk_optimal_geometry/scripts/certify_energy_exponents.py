#!/usr/bin/env python3
"""Finite Gaussian upper bounds matching the energy-window divergence powers."""
from pathlib import Path
from fractions import Fraction
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import iv
from fk5.io import source_manifest,write_json
from fk5.window_sharp_bounds import exact_hopping_coordinates
from fk5.two_orbit import actual_arc_atoms,interval_fraction
from fk5.joint_dual_certificates import exact_schmidt_polar
from fk4.certificates import _bounds
from fk.model import build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    initial=json.loads(Path('results/initial_certificates_02/summary.json').read_text());model=build_model();rows=[]
    for t in [9,11,20]:
        spectrum=next(r for r in initial['spectra'] if r['t']==t)
        for delta in ['.001','.01','.1','1']:
            iv.dps=90;d=interval_fraction(Fraction(delta));a=4*(iv.mpf(t)-9)
            cap=2*d/(iv.sqrt(a*a+72*d)+a);x=iv.mpf(99)*cap/100
            q=exact_hopping_coordinates(t,spectrum['ground_energy_interval'])
            maximum_excess=18*x*x+a*x;assert _bounds(maximum_excess)[1]<float(delta)
            xf=float(x.mid)
            atoms=np.concatenate([(exact_schmidt_polar()[model.even_masks]/4)[None,:],
                                  actual_arc_atoms(model,1-xf/2),actual_arc_atoms(model,1-xf)])
            np.savez_compressed(args.output/f't{t}_delta{delta}_atoms.npz',atoms=atoms)
            for epsilon in ['0','.001','.01']:
                eps=interval_fraction(Fraction(epsilon));q2=q[2]-eps
                assert _bounds(q2)[0]>0
                m2=4*q2;m1=iv.sqrt(14)*q[1]/3+2*q2;m0=q[0]+iv.sqrt(14)*q[1]/3+q2
                coefficients=[m0-3*m1/x+2*m2/x**2,4*m1/x-4*m2/x**2,2*m2/x**2-m1/x]
                points=[iv.mpf(0),x/2,x]
                P=[[(1-y/2)**2,3*(2*y-y*y)/(2*iv.sqrt(14)),y*y/4] for y in points]
                target=[q[0],q[1],q2]
                residual=[sum(c*p[j] for c,p in zip(coefficients,P))-target[j] for j in range(3)]
                assert all(_bounds(r)[0]<=0<=_bounds(r)[1] for r in residual)
                lower=4*q2/cap**2;upper=sum(abs(c) for c in coefficients)
                rows.append(dict(t=t,delta=float(delta),epsilon=float(epsilon),x_interval=_bounds(x),
                    lower_cost_interval=_bounds(lower),upper_cost_interval=_bounds(upper),
                    asymptotic_exponent=1 if t==9 else 2,
                    exact_Hilbert_error=float(epsilon),approximate_vector_normalized=epsilon=='0',
                    exact_coefficient_formula='three-point inverse formula in ENERGY_EXPONENT_THEORY.md; split the last two orbit coefficients equally over 8 actual atoms',
                    orbit_coefficient_intervals=[_bounds(c) for c in coefficients],
                    actual_Gaussian_terms=17,maximum_atom_energy_excess_upper=_bounds(maximum_excess)[1],
                    reconstruction_basis='hopping singlets; finite8-point full-Fock identities proved in TWO_ORBIT_CONSTRUCTION.md',
                    numerical_float_arrays_are_not_exact_reconstruction_certificates=True))
    toy=[]
    for delta in ['.001','.01','.1','1']:
        iv.dps=90;p=interval_fraction(Fraction(delta))/8
        lower=1/(iv.sqrt(10)*p*p);upper=lower+3/(iv.sqrt(10)*(1-p)**2)
        toy.append(dict(delta=float(delta),pair_probability_interval=_bounds(p),
            exact_Gaussian_terms=5,lower_cost_interval=_bounds(lower),upper_cost_interval=_bounds(upper),
            ratio_upper=_bounds(upper/lower)[1],target='(3 vacuum + full)/sqrt10',
            energy_constraint='expectation of total number <= delta',
            exact_reconstruction_proof='fifth-root Fourier identity; degrees 1,2,3 canceled, degrees 0,4 matched'))
    write_json(args.output/'summary.json',dict(status='complete',FK_rows=rows,toy_rows=toy,
        finite_matching_exponents=True,abstract_error_term_includes_functional_norm=True,
        requires_uniform_coercivity_near_all_minimizers=True))
    print('Certified',len(rows),'finite FK constructions and',len(toy),'five-term sharp-constant toy bounds',flush=True)


if __name__=='__main__':main()
