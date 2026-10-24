#!/usr/bin/env python3
"""Independently rounded analytic references; explicitly not distribution discoveries."""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import iv,mp
from fk6.paths import PREVIOUS
from fk6.io import source_manifest,write_json
from fk5.two_orbit import ground_root_enclosure,interval_fraction
from fk5.window_sharp_bounds import exact_hopping_coordinates,certify_window
from fk4.certificates import _bounds


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False);iv.dps=100
    write_json(out/'manifest.json',source_manifest(out/'source_snapshot'))
    initial=PREVIOUS/'results/initial_certificates_02/summary.json'
    construction=PREVIOUS/'results/two_orbit_02/summary.json'
    old=json.loads(initial.read_text());two=json.loads(construction.read_text())['construction']
    rows=[]
    for t in [11,20]:
        spectrum=next(r for r in old['spectra'] if r['t']==t)
        reference=next(r for r in old['A1_rows'] if r['t']==t)
        U=reference['cost_upper'] if t==11 else two['exact_construction_cost_interval'][1]
        lo,hi=ground_root_enclosure(t,spectrum['ground_energy_interval'])
        E=iv.mpf([interval_fraction(lo).a,interval_fraction(hi).b]);corr=-14-4*t-E
        q=exact_hopping_coordinates(t,spectrum['ground_energy_interval'])
        for fraction in [Fraction(1,10),Fraction(1,4),Fraction(1,2),Fraction(1)]:
            interval=interval_fraction(fraction)*corr
            delta=_bounds(interval)[1];D=iv.mpf(delta);a=4*(iv.mpf(t)-9)
            X=2*D/(iv.sqrt(a*a+72*D)+a)
            h_simple=iv.sqrt(iv.mpf(3)/2)*X;h_strong=h_simple
            if _bounds(X)[1]<.5:
                cayley=3*X/(iv.sqrt(14)*iv.sqrt(1-2*X))
                if _bounds(cayley)[1]<_bounds(h_simple)[0]:h_strong=cayley
            for multiplier in [Fraction(1),Fraction(11,10),Fraction(5,4)]:
                B=_bounds(iv.mpf(U)*interval_fraction(multiplier))[1]
                record=dict(t=t,U=U,budget_multiplier=str(multiplier),coefficient_budget=B,
                    correlation_fraction=str(fraction),physical_delta_interval=_bounds(interval),
                    delta_used_outward=delta,X_interval=_bounds(X),h4_simple_interval=_bounds(h_simple),
                    h4_stronger_interval=_bounds(h_strong),h8_interval=_bounds(X*X/4),q_intervals=[_bounds(x) for x in q])
                masses={}
                for label,h in [('simple',h_simple),('stronger',h_strong)]:
                    masses[label]={}
                    for epsilon in [Fraction(0),Fraction(1,100)]:
                        bound=(q[1]/B-interval_fraction(epsilon)/B-h)/(1-h)
                        masses[label][str(epsilon)]=max(0.,_bounds(bound)[0])
                record['high_plain_coefficient_mass_lower']=masses
                b=B*h_simple
                if _bounds(b)[1]<_bounds(q[1])[0]:
                    F=(b*q[1]+iv.sqrt(1-b*b)*iv.sqrt(1-q[1]*q[1]))**2
                    record['single_e4_fidelity_upper']=min(1.,_bounds(F)[1])
                else:record['single_e4_fidelity_upper']=1.
                bstrong=B*h_strong
                if _bounds(bstrong)[1]<_bounds(q[1])[0]:
                    Fstrong=(bstrong*q[1]+iv.sqrt(1-bstrong*bstrong)*iv.sqrt(1-q[1]*q[1]))**2
                    record['single_e4_stronger_fidelity_upper']=min(1.,_bounds(Fstrong)[1])
                else:record['single_e4_stronger_fidelity_upper']=1.
                rational=Fraction(delta)
                base=dict(t=t,delta=delta,delta_numerator=rational.numerator,delta_denominator=rational.denominator,
                    coefficient_budget=B,budget_multiplier=str(multiplier),ground_energy_error_lower=0.,fidelity_upper=1.)
                record['stronger_normalized_physical_bounds']=certify_window(base,spectrum)
                record['analytic_reference_not_distribution_discovery']=True
                rows.append(record)
    write_json(out/'summary.json',dict(status='complete',rows=rows,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [initial,construction]},
        global_atom_proof='../fk_optimal_geometry/NORMALIZED_WINDOW_BOUNDS.md',
        independent_verification_required=True,G1_credited=False,G3_credited=False))
    # Separate high-precision implementation: diagonalize the explicit hopping
    # 3x3 Hamiltonian instead of invoking the certificate coordinate formula.
    mp.dps=120;max_prediction_difference=0.
    for r in rows:
        t=mp.mpf(r['t']);A=3*mp.sqrt(14)
        H=mp.matrix([[-14-4*t,-A,0],[-A,-50,-A],[0,-A,-14+4*t]])
        eig,V=mp.eigsy(H);q=V[:,0]*mp.sign(V[0,0]);D=mp.mpf(r['delta_used_outward']);B=mp.mpf(r['coefficient_budget'])
        x=(mp.sqrt(16*(t-9)**2+72*D)-4*(t-9))/36
        simple=mp.sqrt(mp.mpf(3)/2)*x
        strong=min(simple,3*x/mp.sqrt(14*(1-2*x))) if x<mp.mpf('.5') else simple
        for label,h in [('simple',simple),('stronger',strong)]:
            for eps,bound in r['high_plain_coefficient_mass_lower'][label].items():
                e=Fraction(eps);epsilon=mp.mpf(e.numerator)/e.denominator
                expected=max(mp.mpf(0),((q[1]-epsilon)/B-h)/(1-h))
                assert mp.mpf(bound)<=expected
        b=B*simple;expectedF=1 if b>=q[1] else (b*q[1]+mp.sqrt(1-b*b)*mp.sqrt(1-q[1]*q[1]))**2
        assert mp.mpf(r['single_e4_fidelity_upper'])>=expectedF
        b=B*strong;expectedF=1 if b>=q[1] else (b*q[1]+mp.sqrt(1-b*b)*mp.sqrt(1-q[1]*q[1]))**2
        assert mp.mpf(r['single_e4_stronger_fidelity_upper'])>=expectedF
        joint=r['stronger_normalized_physical_bounds'];caps=[B*strong,B*x*x/4]
        lam=[mp.mpf(v) for v in joint['fidelity_test_weights']]
        expected_joint=(mp.sqrt(q[0]**2+(q[1]-lam[0])**2+(q[2]-lam[1])**2)+sum(v*c for v,c in zip(lam,caps)))**2
        assert mp.mpf(joint['fidelity_upper'])>=min(1,expected_joint)
        mu=[mp.mpf(v) for v in joint['energy_penalties']]
        hp=H+mp.diag([0,*mu]);lowest=mp.eigsy(hp,eigvals_only=True)[0]
        rational_lo,rational_hi=[Fraction(*v) for v in joint['penalized_singlet_eigenvalue_rational_bracket']]
        assert mp.mpf(rational_lo.numerator)/rational_lo.denominator<=lowest<=mp.mpf(rational_hi.numerator)/rational_hi.denominator
        diag=[Fraction(-14-4*r['t']),Fraction(-50)+Fraction(joint['energy_penalties'][0]),Fraction(-14+4*r['t'])+Fraction(joint['energy_penalties'][1])]
        d1=diag[0]-rational_lo;d2=(diag[1]-rational_lo)*d1-126;d3=(diag[2]-rational_lo)*d2-126*d1
        assert min(d1,d2,d3)>0
        assert joint['non_ground_full_spectrum_lower']>-14-4*r['t']
        energy_bound=lowest-sum(v*c*c for v,c in zip(mu,caps))
        assert mp.mpf(joint['direct_energy_error_lower'])<=max(0,energy_bound-eig[0])
        for value,interval in zip(q,r['q_intervals']):assert mp.mpf(interval[0])<=value<=mp.mpf(interval[1])
        if r['correlation_fraction']=='1/4' and r['budget_multiplier']=='1':
            print(r['t'],'Delta',r['physical_delta_interval'],'mass',r['high_plain_coefficient_mass_lower'],
                  'F_e4',r['single_e4_stronger_fidelity_upper'],'F_joint',r['stronger_normalized_physical_bounds']['fidelity_upper'],flush=True)
    write_json(out/'verification.json',dict(passed=True,rows=len(rows),digits=120,
        independently_checked=['high-energy mass, both h4 envelopes and both errors','single-e4 normalized fidelity, both envelopes','joint normalized fidelity','exact rational spectral-penalty certificate and physical energy error','physical target coordinates'],
        source_summary_sha256=hashlib.sha256((out/'summary.json').read_bytes()).hexdigest()))


if __name__=='__main__':main()
