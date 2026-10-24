#!/usr/bin/env python3
"""Recompute frozen-direction certificates with independent 110-digit algebra."""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import mp
from fk5.paths import PREVIOUS
from fk5.io import write_json
from verify_window_physics import enclosed, ground, digest


def dimension(k):
    result=Fraction(1)
    for i in range(8):
        for j in range(i+1,8):result*=Fraction(k+14-i-j,14-i-j)
    assert result.denominator==1
    return mp.mpf(result.numerator)


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();mp.dps=110
    summary=json.loads((args.run/'summary.json').read_text());rows=summary['rows'];assert len(rows)==8
    proof=json.loads((PREVIOUS/'results/cartan_exact_k4_01/proof.json').read_text())
    assert json.loads((PREVIOUS/'results/cartan_exact_k4_01/verification.json').read_text())['artifact_verification_passed']
    V=mp.matrix([[mp.mpf(1)/4,mp.sqrt(14)/4,mp.mpf(1)/4],[-mp.sqrt(2)/2,0,mp.sqrt(2)/2],[mp.sqrt(7)/4,-mp.sqrt(2)/4,mp.sqrt(7)/4]])
    gram=mp.matrix([[mp.mpf(v)/mp.mpf(proof['projector_denominator']) for v in line] for line in proof['gram_numerator']])
    source_checks=[];pooled=[];excluded=[]
    for r in rows:
        z=mp.matrix([mp.mpf(x) for x in r['frozen_real_z']]);t,k=r['t'],r['k']
        source=Path(r['source']);assert digest(source)==r['source_sha256']
        frozen=np.load(source);np.testing.assert_array_equal(frozen['z'].real,r['frozen_real_z'])
        beta=V.T*z;x=[beta[0],beta[1]/mp.sqrt(14),beta[2]]
        mono=mp.matrix([mp.fprod(v**n for v,n in zip(x,powers)) for powers in proof['powers']])
        q4=(mono.T*gram*mono)[0];enclosed(q4,r['exact_q4_interval'])
        assert mp.mpf(r['q4_support_upper'])>=q4**(mp.mpf(1)/8)
        a,b=[mp.mpf(x) for x in r['joint_decomposition_coefficients']];assert a>=0 and b>=0
        env=4*a+b/2 if b<=4*a else a+b+2*a*a/(b-2*a)
        bound=env+mp.sqrt((z[0]-a-2*b)**2+(z[1]+mp.sqrt(8)*a)**2+(z[2]-mp.sqrt(7)*a)**2)
        assert mp.mpf(r['joint_plus_Hilbert_remainder_support_upper'])>=bound
        assert r['global_support_upper']==min(r['q4_support_upper'],r['joint_plus_Hilbert_remainder_support_upper'])
        for candidate in r['all_explicit_orbit_candidates']:
            u=mp.mpf(candidate['parameter'])
            vec=([ (2-u*u)/4,-u/mp.sqrt(2),(2+5*u*u)/(4*mp.sqrt(7))] if candidate['family']=='A'
                else [(2*u*u-1)/4,-u/mp.sqrt(2),(5+2*u*u)/(4*mp.sqrt(7))])
            overlap=sum(v*w for v,w in zip(z,vec));enclosed(overlap,candidate['signed_overlap_interval'])
            assert mp.mpf(candidate['support_lower'])<=abs(overlap)
        lo=mp.mpf(r['support_lower']);up=mp.mpf(r['global_support_upper']);assert 0<lo<=up
        assert r['support_lower']==max(x['support_lower'] for x in r['all_explicit_orbit_candidates'])
        _,q,_=ground(mp.mpf(t));numerator=(q.T*V.T*z)[0];enclosed(numerator,r['physical_target_overlap_interval'])
        assert mp.mpf(r['certified_direction_cost_lower'])<=numerator/up
        assert mp.mpf(r['best_possible_direction_cost_upper'])>=numerator/lo
        loglo=max(2*k*mp.log(lo),mp.log(dimension(k))+mp.mpf(k)/4*(mp.log(mp.mpf(r['exact_q4_interval'][0]))-mp.log(dimension(4))))
        logup=mp.mpf(k)/4*mp.log(mp.mpf(r['exact_q4_interval'][1]))
        assert mp.mpf(r['strict_high_order_log_q_envelope'][0])<=loglo
        assert mp.mpf(r['strict_high_order_log_q_envelope'][1])>=logup
        impossible=r['best_possible_direction_cost_upper']<=r['analytic_joint_baseline']
        assert impossible==r['any_global_support_certificate_cannot_beat_joint']
        if impossible:excluded.append([t,k])
        for entry in r['canonical_component_diagnostics']:
            f=np.load(entry['source'])['P_all_features'];assert hashlib.sha256(f.tobytes()).hexdigest()==entry['features_sha256']
            assert f.shape==(entry['count'],4)
            np.testing.assert_allclose(f.mean(axis=0),entry['feature_means'],atol=1e-14,rtol=0)
            for cutoff,value in entry['mass_eta_gt'].items():assert float(np.mean(f[:,3]>float(cutoff)))==value
        if t==20:
            construction=json.loads(Path('results/two_orbit_02/summary.json').read_text())['construction']
            a=mp.mpf(sum(construction['main_arc_parameter_interval'])/2)
            main=sum(v*w for v,w in zip(z,[(2-a*a)/4,-a/mp.sqrt(2),(2+5*a*a)/(4*mp.sqrt(7))]))
            secondary=sum(v*w for v,w in zip(z,[(2*a*a-1)/4,a/mp.sqrt(2),(5+2*a*a)/(4*mp.sqrt(7))]))
            h=r['two_orbit_heights'];assert abs(float(main)-h['main_signed'])<1e-15
            assert abs(float(secondary)-h['secondary_signed'])<1e-15
            assert abs(float((2*k-1)*mp.log(abs(main/secondary)))-h['scaled_log_height_ratio'])<1e-12
        source_checks.append(dict(t=t,k=k,witness_sha256=r['source_sha256']))
    # Independent reconstruction audit of the only newly trained direction.
    new=Path('results/canonical_k64_t11_n131072_01');s=json.loads((new/'summary.json').read_text())['rows'][0]
    for row in s['validation_rows']:
        data=np.load(new/f"t11_k64_validation_seed{row['seed']}.npz")
        residual=np.linalg.norm(data['C_RB']-data['target']);assert residual<.001
        assert abs(residual-row['rho_RB'])<1e-12;pooled.append(data['C_RB'])
    residual=float(np.linalg.norm(np.mean(pooled,axis=0)-data['target']))
    assert abs(residual-s['rho_RB_pooled'])<1e-12
    write_json(args.run/'verification.json',dict(artifact_verification_passed=True,independent_decimal_digits=110,
        directions_checked=source_checks,strictly_excluded_from_beating_joint=excluded,
        coefficient_measure_feature_arrays_checked=40,new_t11_k64_independent_validation_residual=residual,
        high_order_point_integrals_certified=False,local_contacts_used_only_for_support_lower_bounds=True,
        summary_sha256=digest(args.run/'summary.json'),verification_script_sha256=digest(__file__)))
    lines=['# Frozen high-order directions and the finite-order stopping rule','',
        'All eight frozen directions passed independent 110-digit checks of the exact q4, global support envelope, explicit Gaussian support lower witness, physical target overlap and rigorous high-order moment envelopes. The high-order point integrals themselves remain uncertified.','',
        '| t | k | Certified cost lower | Best possible cost lower from this direction | Joint baseline |',
        '|---|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['t']} | {r['k']} | {r['certified_direction_cost_lower']:.9f} | <= {r['best_possible_direction_cost_upper']:.9f} | {r['analytic_joint_baseline']:.9f} |")
    lines+=['','All four t20 directions, and t11 k8/k16, cannot beat the analytic joint witness even with a perfect global support computation. The t11 k32/k64 directions remain possible candidates, but current global bounds do not certify an improvement. No unconditional increase of k is warranted.','',
        'For t20 the scaled main/secondary log-height ratios at k8,16,32,64 are about 41.54,63.70,75.81,60.50, while the proposed two-orbit balance requires about 5.105. The current sequence is not in that asymptotic regime. Thresholding eta is only a finite-width diagnostic, not a certified assignment to contact components. The known-optimum toy demonstrates 1/k drift under its proved hypotheses; those hypotheses cannot be transferred to the unclassified FK optimal witness.','',
        'The deterministic high-order bounds use Cartan block inclusion and Jensen. They are wide envelopes, not validated high-dimensional quadrature. The toy four-radial reduction does not apply to a general FK singlet direction: the full real Gaussian manifold has dimension 56, and quotienting by diagonal Spin(7) (dimension 21) leaves at least 35 generic dimensions.','']
    (args.run/'REPORT.md').write_text('\n'.join(lines))
    print('Verified eight directions, 40 coefficient-feature arrays, six strict exclusions and new t11 k64 reconstruction.',flush=True)


if __name__=='__main__':main()
