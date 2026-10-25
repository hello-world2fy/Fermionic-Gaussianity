#!/usr/bin/env python3
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,json,hashlib,re,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(out):
    out.mkdir(parents=True,exist_ok=False);root=out.parent;p=root/'temporal_cubic_01'
    load=lambda path:json.loads(path.read_text())
    evaluated=load(p/'evaluation.json');protocol=load(p/'protocol.json');geometry=load(p/'basis_geometry.json')
    checks=load(p/'tensor_checks.json');synthesis=load(p/'synthesis_summary.json');jobs=load(p/'tensor_jobs.json')
    diagnostics=load(root/'temporal_cubic_diagnostics_01/protocol.json');fits=load(root/'temporal_cubic_diagnostics_01/fits.json')
    late=load(root/'late_witness_01/summary.json');forecasts=load(p/'frozen_predictions.json');refined=load(p/'refined_predictions.json')
    verified=[]
    def verify(path,expected):
        assert digest(path)==expected,str(path);verified.append({'path':str(path),'sha256':expected})
    verify(p/'early_states.npz',protocol['early_states_sha256']);verify(p/'frozen_bases.npz',geometry['frozen_bases_sha256'])
    verify(p/'frozen_tensors.npz',checks['frozen_tensors_sha256'])
    for row in synthesis+jobs+forecasts+refined:verify(p/row['file'],row['sha256'])
    for row in fits:verify(root/'temporal_cubic_diagnostics_01'/row['file'],row['sha256'])
    for row in late:verify(root/'late_witness_01'/row['file'],row['sha256'])
    tests=(root/'temporal_cubic_tests_01.log').read_text();match=re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK',tests);assert match
    shutil.copyfile(root/'temporal_cubic_tests_01.log',out/'tests.log')
    T=np.load(p/'frozen_tensors.npz');ranks={}
    for name in T.files:
        singular=np.linalg.svd(T[name].reshape(170,-1),compute_uv=False)
        ranks[name]={'polynomial_linear_span_rank_at_1e-11':int(sum(singular>1e-11*singular[0])),
            'largest_singular':float(singular[0]),'smallest_singular':float(singular[-1])}
    audit={'classification':'Fourteenth round: progress; original important-physics and unseen-prediction goal remains active.',
        'training_times':protocol['training_times'],'early_full_Gaussian_syntheses':8,'late_diagnostic_syntheses':2,
        'GPUs_used':8,'positive_canonical_measure_draws':0,'actual_full_Gaussian_coherent_map':True,
        'complex_coordinates':10,'physical_real_tangent_dimension':18,'full_symmetry_dimension':170,
        'frozen_prediction_models':8,'same_initial_trajectories':len(forecasts),'numerical_refinement_trajectories':len(refined),
        'posthoc_local_fit_cases':len(fits),'posthoc_local_fit_starts':6*len(fits),'new_physical_holdout_predictions':0,
        'gate':evaluated['gate'],'canonical_unique_predictive_structure_established':False,
        'new_important_physics_established':False,'novelty_priority_established':False,
        'analytical_monotonicity_domain':'Full-Gaussian k2 map restricted to fixed-number seniority-zero amplitudes. Exact rational compressed-kernel Gershgorin floor; not arbitrary full-Fock-state coercivity.',
        'exact_rational_mu_L14':'1301961/8388608','physical_fidelity_interval_certificate':False,
        'max_early_synthesis_residual':max(r['polished_residual'] for r in synthesis),
        'max_late_synthesis_residual':max(r['residual'] for r in late),
        'max_tensor_random_reconstruction_error':max(r.get('random_reconstruction_error',0.) for r in checks['checks']),
        'max_initial_state_error':max(r['initial_state_error'] for r in forecasts),
        'max_norm_error':max(r['max_norm_error'] for r in forecasts),
        'max_energy_drift':max(r['max_energy_drift'] for r in forecasts),
        'min_whitened_symplectic_singular':min(r.get('minimum_whitened_symplectic_singular',1.) for r in forecasts),
        'max_refinement_state_change':max(r['max_phase_aligned_state_change'] for r in diagnostics['numerical_refinements']),
        'tests_passed':int(match[1]),'tests_seconds':float(match[2]),'verified_file_count':len(verified),'verified_files':verified,
        'primary_time2':evaluated['primary'],'basis_geometry':geometry,'polynomial_span':ranks,
        'endpoint_tangent_diagnostics':diagnostics['endpoint_controls'],'local_feasible_fits':fits,'late_witness_obstructions':late,
        'remaining_gate':'A physically interpretable canonical-derived structure must explain and predict an important deviation missed by strong conventional controls, then survive genuinely new conditions and same-initial-state deletion. Temporal weight dynamics and analytical obstruction do not meet that gate.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    selected=[('canonical','Canonical history, cubic','#0072B2','-'),('direct','Direct history + endpoint witness, cubic','#D55E00','-'),
        ('Hamiltonian','Hamiltonian directions, cubic','#009E73','-'),('linear_history','History + response, linear','#CC79A7','--'),
        ('linear_Hamiltonian','Hamiltonian Krylov, linear','black','--')]
    rows={r['method']:r['rows'] for r in evaluated['models'] if r['start']=='restart'}
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for name,label,color,style in selected:
        t=[r['time'] for r in rows[name]]
        for ax,key in zip(axes,['fidelity','count_TV','current_L1_error']):ax.plot(t,[r[key] for r in rows[name]],style,color=color,label=label)
    for ax,label in zip(axes,['Fidelity','Count TV','Sector-current L1 error']):ax.set(xlabel='Time',ylabel=label)
    axes[0].legend(fontsize=7);fig.suptitle('Same exact t=1.35 initial state; 18 real projective coordinates per main model')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'temporal_predictions.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.6));methods=['canonical','direct','Hamiltonian'];x=np.arange(3)
    for ax,time in zip(axes,[2.,2.5]):
        local=[next(r for r in fits if r['method']==m and r['time']==time) for m in methods]
        lower=next(r for r in late if r['time']==time)['obstructions']
        ax.bar(x-.23,[1-r['actual_prediction_fidelity'] for r in local],.23,label='Propagation error')
        ax.bar(x,[1-r['local_feasible_fidelity'] for r in local],.23,label='Best local feasible fit error')
        ax.bar(x+.23,[lower[m]['infidelity_lower_expression'] for m in methods],.23,label='Analytic lower expression (float evaluation)')
        ax.set(xticks=x,xticklabels=['Canonical','Direct','Hamiltonian'],ylabel='Infidelity',yscale='log',title=f't={time:g}')
    axes[0].legend(fontsize=7);fig.suptitle('Retrospective capacity diagnostics; no global local-fit claim or interval certificate')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'temporal_capacity.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,3.6))
    for name,label,color,style in selected[:3]:
        ax.plot([r['time'] for r in rows[name]],[r['multiple_dark_probability'] for r in rows[name]],style,color=color,label=label)
    exact=rows['exact_symmetry'];ax.plot([r['time'] for r in exact],[r['multiple_dark_probability'] for r in exact],'k--',label='Exact')
    ax.set(xlabel='Time',ylabel='Multiple-transfer dark probability');ax.legend(fontsize=7);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'temporal_dark_population.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps({k:audit[k] for k in ['same_initial_trajectories','numerical_refinement_trajectories','gate','max_energy_drift','max_refinement_state_change','tests_passed','verified_file_count']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();main(args.output)
