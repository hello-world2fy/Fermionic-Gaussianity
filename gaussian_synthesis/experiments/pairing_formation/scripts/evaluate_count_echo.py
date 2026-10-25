#!/usr/bin/env python3
"""Synthetic two-copy tests of a finite-angle projected-Gaussian exclusion.

The time/angle/noise grid is fixed before measurement generation. These are
retrospective consequences of saved states, not new dynamical predictions.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
from pathlib import Path
import argparse
import json
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.projected_count_bound import quantum_fisher


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {'levels': [8, 12, 16], 'G': [2., 4.], 'times': [.5, 1., 2.5, 3.],
                'dephasing_fractions': [0., .1, .25, .5, 1.], 'theta': .2,
                'shots_per_overlap_or_count_setting': 1_000_000,
                'settings_per_case': 3, 'physical_copies_per_case': 5_000_000,
                'familywise_failure_probability': .01, 'seed': 20261122,
                'scope': 'Retrospective finite-angle certificate with synthetic IID SWAP and counting outcomes; no laboratory or new forecast claim.',
                'free_control': 'For each size/coupling/time, the full Fock-diagonal mixture has the same count distribution and is a convex mixture of Slater determinants.',
                'criterion': 'K(0)-K(theta)>theta^2*mean(m) excludes the convex hull of normalized total-number-projected pure fermionic Gaussians.'}
    count = len(protocol['levels'])*len(protocol['G'])*len(protocol['times'])*(len(protocol['dephasing_fractions'])+1)
    protocol['cases'] = count
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    root = Path(__file__).resolve().parents[1]/'results'
    rng = np.random.default_rng(protocol['seed'])
    failure = protocol['familywise_failure_probability']/(3*count)
    shots = protocol['shots_per_overlap_or_count_setting']
    overlap_radius = float(np.sqrt(2*np.log(1/failure)/shots))
    records = []
    for L in protocol['levels']:
        for G in protocol['G']:
            folder = 'physics_L16_01' if L == 16 else 'physics_screen_01'
            data = np.load(root/folder/f'L{L}_G{G:g}_states.npz')
            model = picket_fence(L, G)
            m = np.arange(model.r+1)
            mean_radius = float(model.r*np.sqrt(np.log(1/failure)/(2*shots)))
            for time in protocol['times']:
                j = int(np.argmin(abs(data['times']-time)))
                assert abs(data['times'][j]-time) < 1e-12
                state = data['states'][j]; state = state/np.linalg.norm(state)
                probability = np.array(model.observe(state)['count']); probability /= probability.sum()
                mean = float(probability@m)
                characteristic = probability@np.exp(-1j*protocol['theta']*m)
                for fraction in protocol['dephasing_fractions']+[None]:
                    if fraction is None:
                        kind = 'free_Fock_mixture'
                        purity = float(np.sum(abs(state)**4)); overlap = purity; fisher = 0.
                    else:
                        kind = 'collective_count_dephasing'
                        a = 1-fraction
                        purity = float(a*a+(1-a*a)*np.sum(probability**2))
                        overlap = float(a*a*abs(characteristic)**2+(1-a*a)*np.sum(probability**2))
                        rho = a*np.outer(np.sqrt(probability), np.sqrt(probability))+(1-a)*np.diag(probability)
                        fisher = quantum_fisher(rho, np.diag(m))
                    k0 = float(2*rng.binomial(shots, np.clip((1+purity)/2, 0., 1.))/shots-1)
                    kt = float(2*rng.binomial(shots, np.clip((1+overlap)/2, 0., 1.))/shots-1)
                    counts = rng.multinomial(shots, probability)
                    mean_estimate = float(counts@m/shots)
                    true_excess = purity-overlap-protocol['theta']**2*mean
                    estimate = k0-kt-protocol['theta']**2*mean_estimate
                    radius = 2*overlap_radius+protocol['theta']**2*mean_radius
                    records.append({'L': L, 'G': G, 'time': time, 'kind': kind, 'dephasing_fraction': fraction,
                                    'mean': mean, 'true_purity': purity, 'true_rotated_overlap': overlap,
                                    'QFI': fisher, 'free_QFI_bound': 4*mean,
                                    'finite_angle_excess': true_excess, 'estimate': estimate,
                                    'simultaneous_one_sided_radius': radius, 'lower_confidence_bound': estimate-radius,
                                    'certified_outside_projected_Gaussian_convex_hull': bool(estimate-radius > 0),
                                    'synthetic_only': True})
    (args.output/'evaluation.json').write_text(json.dumps({'protocol': protocol, 'records': records}, indent=2))
    print(json.dumps({'cases': len(records), 'certified': sum(r['certified_outside_projected_Gaussian_convex_hull'] for r in records),
                      'free_control_false_positives': sum(r['certified_outside_projected_Gaussian_convex_hull'] for r in records if r['kind']=='free_Fock_mixture')}))
