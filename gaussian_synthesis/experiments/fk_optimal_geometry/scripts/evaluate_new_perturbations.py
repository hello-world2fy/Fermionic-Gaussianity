#!/usr/bin/env python3
"""ED access is gated on complete, immutable predictions for all new tests."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import datetime
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.linalg import eigh
from fk5.io import write_json,source_manifest
from fk5.perturbation_transfer import sha
from fk.model import build_model
from fk.observables import exact_metrics
from fk2.orbits import covariance_from_kets


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    registration=json.loads((args.run/'preregistration.json').read_text());prediction=json.loads((args.run/'prediction_summary.json').read_text())
    assert prediction['status']=='complete' and len(prediction['rows'])==160
    assert prediction['preregistration_sha256']==sha(args.run/'preregistration.json')
    assert sha(args.run/'frozen_inputs.npz')==registration['frozen_input_sha256']
    assert {r['key'] for r in prediction['rows']}=={r['key'] for r in registration['cases']}
    hashes={}
    for r in prediction['rows']:
        assert len(r['rows'])==15 and not r['new_target_accessed']
        file=args.run/'predictions'/r['key']/'predictions.npz';hashes[r['key']]=sha(file)
        assert hashes[r['key']]==r['prediction_file_sha256']
    # This file is written only after every prediction has passed the barrier,
    # and before any perturbed eigensystem is constructed below.
    barrier=args.run/'ED_ACCESS_BARRIER.json';assert not barrier.exists()
    write_json(barrier,dict(all_2400_predictions_complete_and_hashed=True,
        prediction_hashes=hashes,preregistration_sha256=sha(args.run/'preregistration.json'),
        prediction_summary_sha256=sha(args.run/'prediction_summary.json'),
        prediction_complete_utc=prediction['complete_utc'],first_ED_access_after_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    output=args.run/'evaluation';output.mkdir(exist_ok=False);write_json(output/'manifest.json',source_manifest(output/'source_snapshot'))
    model=build_model();masks=model.even_masks;rows=[];checks=[];tick=time.monotonic()
    original={r['key']:r for r in registration['cases']}
    for r in prediction['rows']:
        case=original[r['key']];data=np.load(args.run/'predictions'/r['key']/'predictions.npz');H=data['hamiltonian_full']
        # Reassemble independently from the exact operators and frozen pattern.
        expected=model.h_int.copy()
        hopping=np.ones(8)
        if case['kind']=='hopping':hopping+=case['amplitude']*np.array(case['pattern'])
        for j,value in enumerate(hopping):expected+=.5j*case['t']*value*(model.gammas[j]@model.gammas[j+8])
        if case['kind']=='interaction':
            for x in range(2):
                for a in range(7):expected-=case['amplitude']*case['pattern'][x][a]*(model.q_ops[x,a]@model.q_ops[x,a])
        assert np.max(abs(H-expected))<3e-14
        E,V=eigh(H);ground=V[:,0];E0=float(E[0]);gap=float(E[1]-E[0]);assert gap>1e-8
        residual=float(np.linalg.norm(H@ground-E0*ground));assert residual<1e-11
        parity_even_weight=float(np.sum(abs(ground[masks])**2));reference=exact_metrics(ground,model)
        np.savez_compressed(output/(r['key']+'_exact.npz'),energies=E,ground=ground)
        maximum_purity=0.;case_rows=[]
        for pred in r['rows']:
            method=pred['method'];R=pred['requested_maximum_terms'];name=f'{method}_R{R}'
            atoms=data[name+'_atoms'];c=data[name+'_coefficients'];state=c@atoms
            assert len(atoms)==pred['actual_terms']<=R
            assert max(abs(np.linalg.norm(atoms,axis=1)-1))<1e-10 and abs(np.linalg.norm(state)-1)<1e-10
            cov=covariance_from_kets(atoms);purity=float(np.max(abs(cov@cov+np.eye(16))));maximum_purity=max(maximum_purity,purity)
            assert purity<1e-9
            cost=float(sum(abs(c)));energy=float(np.vdot(state,H[np.ix_(masks,masks)]@state).real)
            assert abs(cost-pred['coefficient_cost'])<1e-10 and abs(energy-pred['predicted_energy'])<1e-10
            metrics=exact_metrics(state,model);F=float(abs(np.vdot(ground[masks],state))**2);err=energy-E0
            assert err>=-1e-10 and err>=gap*(1-F)-2e-10
            record=dict(key=r['key'],t=case['t'],kind=case['kind'],amplitude=case['amplitude'],pattern_id=case['pattern_id'],
                method=method,R=R,actual_terms=len(atoms),coefficient_cost=cost,fidelity=F,energy_error=err,
                mass_correlation=metrics['mass_correlation'],mass_correlation_error=metrics['mass_correlation']-reference['mass_correlation'],
                wick_deviation=metrics['wick_deviation'],wick_deviation_error=metrics['wick_deviation']-reference['wick_deviation'],
                prediction_seconds=pred['prediction_seconds'],ground_energy=E0,full_spectral_gap_numeric=gap,
                exact_ground_even_weight=parity_even_weight)
            rows.append(record);case_rows.append(record)
        checks.append(dict(key=r['key'],Hamiltonian_reassembled=True,exact_eigen_residual=residual,maximum_Gaussian_purity_error=maximum_purity))
        write_json(output/(r['key']+'.json'),dict(rows=case_rows,reference=reference,check=checks[-1]))
    assert len(rows)==2400
    # Paired comparisons preserve t, perturbation pattern and amplitude.
    paired=[]
    for contact in [r for r in rows if r['method']=='two_orbit']:
        for competitor in ['physics_prior','dual_columns','canonical_Lambda_k8','canonical_Lambda_k16','canonical_Lambda_k32','adaptive_energy_columns']:
            other=next(x for x in rows if x['key']==contact['key'] and x['R']==contact['R'] and x['method']==competitor)
            paired.append(dict(key=contact['key'],t=contact['t'],kind=contact['kind'],amplitude=contact['amplitude'],R=contact['R'],competitor=competitor,
                contact_minus_other_energy_error=contact['energy_error']-other['energy_error'],
                contact_minus_other_cost=contact['coefficient_cost']-other['coefficient_cost'],
                contact_energy_no_worse=contact['energy_error']<=other['energy_error']+1e-9,
                contact_cost_no_greater=contact['coefficient_cost']<=other['coefficient_cost']+1e-9,
                contact_jointly_no_worse=contact['energy_error']<=other['energy_error']+1e-9 and contact['coefficient_cost']<=other['coefficient_cost']+1e-9))
    write_json(output/'summary.json',dict(status='complete',rows=rows,paired_comparisons=paired,elapsed_seconds=time.monotonic()-tick,
        all_predictions_preceded_new_ED=True,no_post_test_dictionary_changes=True,upstream_costs_not_all_comparable=True))
    write_json(output/'verification.json',dict(artifact_verification_passed=True,predictions_checked=2400,new_Hamiltonians_checked=160,
        checks=checks,all_predictions_complete_before_first_ED=True,barrier_sha256=sha(barrier),
        summary_sha256=sha(output/'summary.json'),verification_script_sha256=sha(__file__),
        proof_scope='Independent full-Hamiltonian assembly, full-spectrum ED validation and all actual Gaussian trial states; numerical transfer evidence, not a global prediction theorem.'))
    with (output/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# New hopping and interaction perturbations: frozen transfer test','',
        'All 2400 predictions across 160 Hamiltonians were saved and hashed before the first new full-spectrum ED. The Hamiltonians were then independently reassembled, every trial state checked for Gaussian atoms and normalization, and energy, fidelity, mass and Wick errors evaluated. No dictionary was modified using the validation targets.','',
        'Means across the preregistered couplings, amplitudes and five perturbation patterns (not independent error bars):','',
        '| Perturbation | R | Method | Mean fidelity | Mean energy error | Mean coefficient cost | Mean prediction seconds |',
        '|---|---:|---|---:|---:|---:|---:|']
    methods=['two_orbit','physics_prior','dual_columns','canonical_Lambda_k8','canonical_Lambda_k16','canonical_Lambda_k32','adaptive_energy_columns']
    for kind in ['hopping','interaction']:
        for R in [16,32]:
            for method in methods:
                group=[r for r in rows if r['kind']==kind and r['R']==R and r['method']==method]
                means=[np.mean([r[key] for r in group]) for key in ['fidelity','energy_error','coefficient_cost','prediction_seconds']]
                lines.append(f'| {kind} | {R} | {method} | '+ ' | '.join(f'{x:.8g}' for x in means)+' |')
    lines+=['','Matched-term contact comparisons (energy and cost jointly no worse; tolerance 1e-9):','',
        '| Competitor | Contact energy no worse | Contact energy and cost jointly no worse | Total paired tests |','|---|---:|---:|---:|']
    for method in methods[1:]:
        group=[r for r in paired if r['competitor']==method]
        lines.append(f"| {method} | {sum(r['contact_energy_no_worse'] for r in group)} | {sum(r['contact_jointly_no_worse'] for r in group)} | {len(group)} |")
    lines+=['','HFB is a separate 32-start, one-Gaussian baseline. Adaptive columns use the reused Gaussian overlap oracle on the energy Ritz residual, with no new exact target; their timing includes the full shared 32-column calculation and lists HFB setup separately in prediction metadata. Frozen column and canonical preparation costs are preserved in preregistration.json.','',
        'The contact construction has an untimed upstream discovery/proof cost, so these results do not establish an end-to-end computational advantage. Canonical sampling is the original coefficient measure Lambda, historically named P. All cost and actual-term differences must be retained when interpreting accuracy comparisons. Full paired cases and HFB results are in metrics.csv.','']
    (output/'REPORT.md').write_text('\n'.join(lines))
    print('Validated 2400 preregistered predictions against 160 full-spectrum exact ground states.',flush=True)


if __name__=='__main__':main()
