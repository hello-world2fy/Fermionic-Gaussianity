#!/usr/bin/env python3
"""Audit finite ablations, keep feasible nested candidates and make figures."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.linalg import svd
from fk5.io import write_json
from fk.model import build_model
from fk.observables import exact_metrics
from fk2.orbits import covariance_from_kets
from verify_window_physics import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();model=build_model()
    summary=json.loads((args.run/'summary.json').read_text());assert summary['status']=='complete'
    rows=summary['rows'];assert len(rows)==720
    dictionaries=json.loads((args.run/'dictionaries.json').read_text())['rows'];assert len(dictionaries)==120
    config=json.loads((args.run/'config.json').read_text());assert config['t']==[9,10.9225,11,20]
    assert config['R']==[2,4,8,16,32,64] and config['replicates']==[1,2,3,4,5]
    bound=json.loads(Path('results/window_sharp_bounds_02/summary.json').read_text())['rows']
    orbit=json.loads(Path('results/coherent_orbits_01/summary.json').read_text())['rows']
    orbit={r['t']:r['whole_orbit_deletion_certificates'] for r in orbit}
    grouped=defaultdict(list)
    for row in rows:grouped[row['t'],row['method'],row['replicate']].append(row)
    all_selected=[];maxpurity=0.;state_count=0;reused=0;warning_count=0
    variants=['unconstrained_fidelity','unconstrained_energy','budget_fidelity','budget_energy']
    for dictionary in dictionaries:
        t=dictionary['t'];path=args.run/'dictionaries'/(dictionary['key']+'.npz')
        assert digest(path)==dictionary['file_sha256'];atoms=np.load(path)['atoms'];assert atoms.shape==(64,128)
        gram=abs(atoms.conj()@atoms.T);np.fill_diagonal(gram,0)
        assert np.max(gram)<1-1e-10
        cov=covariance_from_kets(atoms);purity=float(np.max(abs(cov@cov+np.eye(16))));maxpurity=max(maxpurity,purity)
        assert purity<1e-10 and max(abs(np.linalg.norm(atoms,axis=1)-1))<1e-11
        H=model.full_hamiltonian(t)[np.ix_(model.even_masks,model.even_masks)];target=model.target(t)
        E0=float(np.vdot(target,H@target).real);reference=exact_metrics(target,model)
        if dictionary['method']=='low_energy_delta0.01':
            assert max(np.einsum('ni,ij,nj->n',atoms.conj(),H,atoms).real+14+4*t)<=.01+1e-10
        records=sorted(grouped[t,dictionary['method'],dictionary['replicate']],key=lambda x:x['R'])
        assert [r['R'] for r in records]==config['R']
        best={}
        for row in records:
            R=row['R'];small=atoms[:R];directory=args.run/'runs'/(dictionary['key']+f'_R{R}')
            data=np.load(directory/'states.npz');np.testing.assert_array_equal(data['atoms'],small)
            selected={};arrays={}
            U,s,_=svd(small.T,full_matrices=False);keep=s>1e-10*s[0]
            span_fidelity=float(np.linalg.norm(U[:,keep].conj().T@target)**2)
            energy_keep=s*s>1e-10*max(1.,s[0]*s[0])
            span_energy=float(np.linalg.eigvalsh(U[:,energy_keep].conj().T@H@U[:,energy_keep])[0])
            for name in variants:
                c=data[name+'_coefficients'];state=c@small
                norm=float(np.linalg.norm(state));assert abs(norm-1)<2e-11
                cost=float(sum(abs(c)));F=float(abs(np.vdot(target,state))**2);E=float(np.vdot(state,H@state).real)
                metrics=exact_metrics(state,model);r=row['metrics'][name]
                assert abs(cost-r['coefficient_cost'])<2e-10 and abs(F-r['fidelity'])<2e-12 and abs(E-r['energy'])<3e-11
                for key in ['mass_correlation','wick_prediction','wick_deviation','mass_a','mass_b','mass_dot','eta']:
                    assert abs(metrics[key]-r[key])<3e-11,(dictionary['key'],R,name,key)
                if name.startswith('budget'):assert cost<=row['budget']+1e-10
                if name=='unconstrained_fidelity':assert abs(F-span_fidelity)<2e-9
                if name=='unconstrained_energy':assert abs(E-span_energy)<2e-7,(dictionary['key'],R,E,span_energy)
                method=dictionary['method']
                if method in ['main_only','secondary_only','joint_contacts']:
                    certificate=orbit[t][method if method!='joint_contacts' else 'joint_contacts_only']
                    assert F<=certificate['fidelity_ceiling_interval'][1]+3e-10
                    assert E-E0>=certificate['energy_error_lower']-3e-10
                if method=='low_energy_delta0.01' and t in [9,11,20] and name.startswith('budget'):
                    b=next(x for x in bound if x['t']==t and x['delta']==.01 and x['budget_multiplier']=='1')
                    # The D table rounded the reference budget outward. Its
                    # (possibly one-ulp larger) budget also bounds this subset.
                    assert row['budget']<=b['coefficient_budget']
                    assert F<=b['fidelity_upper']+3e-11 and E-E0>=b['best_energy_error_lower']-3e-10
                state_count+=1
                objective='fidelity' if name.endswith('fidelity') else 'energy'
                better=(name not in best or (F>best[name][0]['fidelity'] if objective=='fidelity' else E<best[name][0]['energy']))
                if better:best[name]=(r,c.copy(),R)
                chosen,coeff,origin=best[name];selected[name]=dict(**chosen,candidate_source_R=origin)
                arrays[name+'_coefficients']=np.pad(coeff,(0,R-len(coeff)))
                if origin!=R:reused+=1
            solver=row['normalized_budget_solver'];warning_count+=solver['fidelity_convex_solver_status']!='optimal'
            warning_count+=sum(e['status']!='optimal' for h in solver['energy_multistart_histories'] for e in h)
            np.savez_compressed(directory/'selected_states.npz',**arrays)
            record=dict(t=t,method=dictionary['method'],replicate=dictionary['replicate'],R=R,
                dictionary_distinct_atoms_available=R,budget=row['budget'],metrics=selected,
                selection='Best feasible result among this dictionary and all smaller nested prefixes; raw solver results preserved',
                reference_mass_correlation=reference['mass_correlation'],reference_wick_deviation=reference['wick_deviation'])
            write_json(directory/'selection.json',record);all_selected.append(record)
    assert state_count==2880
    write_json(args.run/'selected_summary.json',dict(status='complete',rows=all_selected,nested_candidate_reuses=reused,
        actual_nonzero_terms_recorded_separately=True,all_raw_solver_results_preserved=True))
    write_json(args.run/'verification.json',dict(artifact_verification_passed=True,dictionaries_checked=120,
        distinct_atoms_checked=7680,solver_configurations_checked=720,normalized_states_checked=state_count,
        maximum_gaussian_purity_error=maxpurity,solver_precision_warning_entries=warning_count,
        nested_feasible_candidate_reuses=reused,full_scope=[[t,R] for t in config['t'] for R in config['R']],
        numerical_feasibility_and_observable_checks_not_interval_atom_certificates=True,
        all_applicable_global_window_and_orbit_bounds_respected=True,
        summary_sha256=digest(args.run/'summary.json'),verification_script_sha256=digest(__file__)))
    flattened=[]
    for r in all_selected:
        for name,values in r['metrics'].items():flattened.append(dict(t=r['t'],method=r['method'],replicate=r['replicate'],R=r['R'],objective=name,budget=r['budget'],**values))
    with (args.run/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flattened[0]));w.writeheader();w.writerows(flattened)
    methods=['two_orbit','main_only','secondary_only','joint_contacts','low_energy_delta0.01','physics_prior','dual_columns','canonical_Lambda_k8','canonical_Lambda_k16','canonical_Lambda_k32']
    lines=['# Four-coupling structure ablation','',
        'Completed and independently checked 720 configurations: four couplings, six dictionary sizes, five analytic/deletion/window dictionaries, and five independent replicas of each prior/column/canonical baseline. Each configuration has unconstrained fidelity and energy solutions plus normalized cost-budget fidelity and energy trials (2880 saved states).','',
        'The budget is the explicit original plan reference U at each coupling. Canonical pools are the coefficient measure Lambda proportional to |overlap|^(2k-1); legacy upstream filenames called this P. No claim that these are the overlap^(2k) Husimi distributions is made. Every pool is deduplicated before target-independent QR selection.','',
        'Numerical budget optimizations provide feasible candidates; the normalized fidelity relaxation may have an interior solution and energy optimization is local. The report retains the best feasible candidate from each nested smaller dictionary, so increasing the allowed term count cannot worsen the reported result. All original solver results and statuses remain saved.','',
        f'Solver warnings: {warning_count} entries including repeated subproblems. Nested candidate reuse: {reused}. All applicable whole-orbit and energy-window bounds are respected.','',
        'At R=16, means over the five repeats where applicable:','',
        '| t | Method | Budget fidelity | Budget energy error | Mass-correlation error (energy state) | Coefficient cost (energy state) |',
        '|---|---|---:|---:|---:|---:|']
    for t in config['t']:
        for method in methods:
            group=[r for r in all_selected if r['t']==t and r['method']==method and r['R']==16]
            F=np.mean([r['metrics']['budget_fidelity']['fidelity'] for r in group]);E=np.mean([r['metrics']['budget_energy']['energy_error'] for r in group])
            M=np.mean([abs(r['metrics']['budget_energy']['mass_correlation']-r['reference_mass_correlation']) for r in group]);C=np.mean([r['metrics']['budget_energy']['coefficient_cost'] for r in group])
            lines.append(f'| {t:g} | {method} | {F:.9f} | {E:.8g} | {M:.8g} | {C:.8f} |')
    lines+=['','The two-orbit construction reaches the exact target with 16 actual Gaussians without a budget constraint. Its exact costs at t9,10.9225,11,20 are approximately 1.984228,1.793456,1.785866,1.295570. Only t20 lies within the corresponding reference U; the first three budget-constrained results remain inexact. Removing the secondary orbit leaves a provable singlet-amplitude obstruction, though the t20 fidelity loss is small. Restricting all atoms to the low-energy window gives the much stronger budget-dependent physical obstruction certified in D1/D2.','',
        'These are representation and structure comparisons on known FK targets. They do not establish a prediction advantage on new Hamiltonians; that claim requires the separately preregistered F tests. Unconstrained solutions and their potentially large coefficient costs are in metrics.csv.','']
    (args.run/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,constrained_layout=True)
    colors=plt.get_cmap('tab10')
    display=['two_orbit','main_only','low_energy_delta0.01','physics_prior','dual_columns','canonical_Lambda_k8','canonical_Lambda_k16','canonical_Lambda_k32']
    for ax,t in zip(axs.ravel(),config['t']):
        for i,method in enumerate(display):
            mean=[];spread=[]
            for R in config['R']:
                values=[max(1e-12,1-r['metrics']['budget_fidelity']['fidelity']) for r in all_selected if r['t']==t and r['method']==method and r['R']==R]
                mean.append(np.mean(values));spread.append(np.std(values))
            ax.plot(config['R'],mean,'.-',label=method,color=colors(i))
            ax.fill_between(config['R'],np.maximum(1e-12,np.array(mean)-spread),np.array(mean)+spread,color=colors(i),alpha=.1)
        ax.set_xscale('log',base=2)
        ax.set(yscale='log',title=f't = {t:g}',xlabel='Distinct atoms available R',ylabel='Budget-constrained infidelity')
        from matplotlib.ticker import NullLocator
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks(config['R'],[str(x) for x in config['R']]);ax.grid(alpha=.2)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=4,fontsize=8)
    fig.savefig(args.run/'budget_infidelity.png',dpi=180);fig.savefig(args.run/'budget_infidelity.pdf');plt.close(fig)
    print(f'Verified {state_count} physical states; all 720 configurations and complete R/t coverage. Report, CSV and figures saved.',flush=True)


if __name__=='__main__':main()
