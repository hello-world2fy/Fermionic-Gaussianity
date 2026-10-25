#!/usr/bin/env python3
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,argparse,hashlib,re,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out):
    out.mkdir(parents=True,exist_ok=False);root=out.parent;p=root/'occupation_response_01'
    load=lambda path:json.loads(path.read_text())
    protocol=load(p/'protocol.json');frozen=load(p/'frozen_predictions.json');evaluated=load(p/'evaluation.json')
    jobs=load(p/'moment_jobs.json');mc=load(root/'occupation_symbol_checks_01/summary.json')['jobs']
    diagnostics=load(root/'occupation_response_diagnostics_01/diagnostics.json');verified={}
    def verify(path,expected):
        assert digest(path)==expected,str(path);verified[str(path)]=expected
    for path,value in protocol['input_hashes'].items():verify(Path(path),value)
    verify(p/'training.npz',protocol['training_sha256']);verify(p/'frozen_spaces.npz',protocol['frozen_sha256'])
    verify(p/'background_prediction.npz',frozen['background_sha256'])
    for row in jobs+frozen['models']:verify(p/row['file'],row['sha256'])
    for row in mc:verify(root/'occupation_symbol_checks_01'/row['file'],row['sha256'])
    for path,value in diagnostics['source_hashes'].items():verify(Path(path),value)
    for row in diagnostics['kicks']:verify(root/'occupation_response_diagnostics_01'/row['file'],row['sha256'])
    log=(root/'occupation_response_tests_01.log').read_text();match=re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK',log);assert match
    shutil.copyfile(root/'occupation_response_tests_01.log',out/'tests.log')
    audit={'classification':'Fifteenth round: progress; physically important canonical-specific prediction objective remains active and unmet.',
        'actual_full_Fock_Gaussian_occupation_coordinates':True,'physical_modes':28,'GPUs_used':8,
        'Haar_check_samples':sum(r['samples'] for r in mc),'iid_canonical_measure_draws':0,
        'training_times':protocol['training_times'],'reused_canonical_syntheses':4,'fresh_canonical_syntheses':0,
        'deterministic_single_mode_contractions':14*len(jobs),'Cartan_derivative_calls':28*len(jobs),
        'full_paired_dimension':3432,'previous_fully_invariant_dimension':170,'new_response_sectors':6,
        'response_space_dimensions':[1,3,9],'common_background_H_Krylov_dimension':32,
        'frozen_response_predictions':len(frozen['models']),'passing_canonical_gates':[r for r in evaluated['gate'] if r['passes']],
        'same_initial_deletion_cases':6,'independent_exponential_repeats':len(diagnostics['independent_exponential_checks']),
        'posthoc_signed_finite_kick_trajectories':2*len(diagnostics['kicks']),
        'new_physical_parameter_holdouts':0,'new_important_physics_established':False,
        'canonical_unique_predictive_structure_established':False,'novelty_priority_established':False,
        'equal_total_compute_advantage_established':False,
        'single_mode_rank_limit':'At replica k2, all integrable functions of one full-Gaussian mean occupation produce a coherent vector in three contracted copy-occupation sectors. Does not apply to joint modes or pair-field phases.',
        'max_ward_residual':max(r['max_ward_residual'] for r in jobs),
        'max_reused_canonical_synthesis_residual':max(r['synthesis_residual'] for r in jobs if r['method']=='canonical'),
        'max_Haar_moment_error_over_estimated_Frobenius_SE':max(x['error_over_SE'] for r in mc for x in r['moments']),
        'max_Haar_linear_frame_error_over_estimated_SE':max(r['linear_error_over_SE'] for r in mc),
        'rigorous_MC_confidence_certificate':False,'interval_certified_physical_projection_bound':False,
        **evaluated['checks'],
        'max_independent_exponential_state_difference':max(r['max_state_difference'] for r in diagnostics['independent_exponential_checks']),
        'max_SVD_cutoff_subspace_residual':max(r['subspace_residual'] for r in diagnostics['SVD_cutoff_checks']),
        'max_finite_kick_Richardson_response_error':max(r['Richardson_max_error'] for r in diagnostics['physical_response_checks']),
        'tests_passed':int(match[1]),'tests_seconds':float(match[2]),'verified_file_count':len(verified),'verified_files':verified,
        'primary':[{'case':r['case'],'method':r['method'],'dimension':r['dimension'],**r['primary']} for r in evaluated['models']],
        'capacity':diagnostics['capacity'],'canonical_unoptimized_geometry':load(p/'geometry.json'),
        'remaining_gate':'A physically interpretable canonical-derived structure must predict an important discrepancy beyond strong conventional controls, survive genuinely new conditions and same-initial-state deletion. Occupation moments do not pass this gate.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    cases=[f'q{q}_{parity}' for q in (1,2,3) for parity in ('plus','minus')]
    lookup={(r['case'],r['method']):r for r in evaluated['models']}
    styles=[('canonical_history','Canonical occupation moments','#0072B2','-'),
        ('unoptimized_history','Unoptimized occupation moments','#D55E00',':'),
        ('ordinary_history','Ordinary response history','#009E73','--'),
        ('Hamiltonian_d9','Hamiltonian Krylov, dimension 9','black','-.')]
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),sharex=True)
    for ax,case in zip(axes.ravel(),cases):
        for method,label,color,style in styles:
            rows=lookup[case,method]['rows'];ax.plot([r['elapsed'] for r in rows],[r['chi'] for r in rows],style,color=color,label=label,lw=1.5)
        rows=lookup[case,'canonical_history']['rows']
        ax.plot([r['elapsed'] for r in rows],[r['chi_exact'] for r in rows],color='0.6',lw=3,alpha=.5,label='Exact')
        ax.axvline(.65,color='0.7',ls=':',lw=.8);ax.set(title=case.replace('_',' '),ylabel='Density response chi')
    for ax in axes[-1]:ax.set_xlabel('Elapsed time after t=1.35')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=8)
    fig.tight_layout(rect=(0,0,1,.91))
    for ext in ('png','pdf'):fig.savefig(out/f'occupation_density_response.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(cases))
    for k,(method,label,color,_) in enumerate(styles):
        ax.bar(x+(k-1.5)*.19,[lookup[case,method]['primary']['complex_correlation_relative_RMS'] for case in cases],.19,color=color,label=label)
    ax.set(xticks=x,xticklabels=[c.replace('_',' ') for c in cases],yscale='log',ylabel='Relative complex correlation RMS error',xlabel='Physical density harmonic and band-exchange parity')
    ax.legend(fontsize=8,ncol=2);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'occupation_prediction_errors.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.8));capacity={(r['case'],r['method']):r for r in diagnostics['capacity']}
    ax.bar(x-.18,[lookup[c,'canonical_history']['primary']['fidelity'] for c in cases],.36,label='Canonical moment propagation')
    ax.bar(x+.18,[capacity[c,'canonical_history']['time_065_subspace_max_fidelity'] for c in cases],.36,label='Best fidelity in the same saved subspace')
    ax.set(xticks=x,xticklabels=[c.replace('_',' ') for c in cases],ylim=(0,1),ylabel='Response-state fidelity at elapsed time 0.65')
    ax.legend(fontsize=8);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'occupation_capacity.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps({k:audit[k] for k in ['Haar_check_samples','frozen_response_predictions','passing_canonical_gates','max_finite_kick_Richardson_response_error','tests_passed','verified_file_count']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
