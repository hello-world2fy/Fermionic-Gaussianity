#!/usr/bin/env python3
"""Round-seventeen saved-evidence audit and physical-noise figures."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,hashlib,re,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False);root=out.parent;experiment=root.parent
    load=lambda p:json.loads(p.read_text());verified={}
    def verify(path,value):
        path=Path(path)
        if not path.is_absolute():path=experiment/path
        path=path.resolve();assert digest(path)==value,str(path)
        verified[str(path.relative_to(experiment))]=value
    source=root/'multipair_symbol_pilot_01';protocol=load(source/'protocol.json');frozen=load(source/'frozen_predictions.json');evaluation=load(source/'evaluation.json');jobs=load(source/'moment_jobs.json')
    for path,value in protocol['input_hashes'].items():verify(path,value)
    for path,value in load(root/'pair_symbol_pilot_01/protocol.json')['input_hashes'].items():verify(path,value)
    verify(source/'training.npz',protocol['training_sha256']);verify(source/'frozen_spaces.npz',protocol['frozen_sha256'])
    for row in jobs+frozen:verify(source/row['file'],row['sha256'])
    assert [(r['method'],r['start'],r['sha256']) for r in frozen]==[(r['method'],r['start'],r['sha256']) for r in evaluation['models']]
    mc=load(root/'multipair_symbol_checks_01/summary.json')['jobs']
    for row in mc:verify(root/'multipair_symbol_checks_01'/row['file'],row['sha256'])
    diagnostics=load(root/'multipair_symbol_diagnostics_01/diagnostics.json')
    screen=load(root/'multipair_noise_screen_01/screen.json')
    for record in (diagnostics,screen):
        for path,value in record['input_hashes'].items():verify(path,value)
    readout=load(root/'multipair_readout_01/summary.json');readout_rows=[];readout_checks=[]
    for row in readout['jobs']:
        path=root/'multipair_readout_01'/row['file'];verify(path,row['sha256']);data=load(path)
        for filename,value in data['input_hashes'].items():verify(filename,value)
        readout_rows.extend(data['rows']);readout_checks.extend(data['checks'])
    stochastic=load(root/'stochastic_multipair_assessment_01/evaluation.json')
    uniform=load(root/'uniform_multipair_controls_01/evaluation.json')
    for record in (stochastic,uniform):
        for path,value in record['input_hashes'].items():verify(path,value)
    log=root/'multipair_symbol_tests_03.log';match=re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK',log.read_text());assert match
    shutil.copyfile(log,out/'tests.log')
    data=np.load(source/'frozen_spaces.npz');exact=np.load(source/'exact_references.npz');operators=data['pair_channels']
    def coefficients(v):
        return np.array([-2*np.real(np.exp(-2j*phi)*np.vdot(v,operators[q]@operators[q]@v)) for q in (1,3) for phi in (0.,np.pi/4)])
    exact_coefficient=coefficients(exact['states'][40]);curvatures=[]
    for row in frozen:
        saved=np.load(source/row['file']);v=saved['states'][np.argmin(abs(saved['times']-2))];c=coefficients(v)
        curvatures.append({'method':row['method'],'start':row['start'],'coefficients':c.tolist(),'L2_error':float(np.linalg.norm(c-exact_coefficient)),
            'correct_signs_of_four':int(np.sum(c*exact_coefficient>0))})
    readout_lookup={(r['L'],r['q'],r['method'],r['phase'],r['epsilon']):r for r in readout_rows}
    wrong_signs=0
    for L in (10,14,18):
        for q in (1,3):
            for phi in (0.,float(np.pi/4)):
                a=readout_lookup[L,q,'exact',phi,.01]['analytic_coefficient'];b=readout_lookup[L,q,'full_cc',phi,.01]['analytic_coefficient'];wrong_signs+=a*b<0
    audit={'classification':'Seventeenth round: physical multi-transfer-noise and readout progress; original important canonical-specific discovery objective remains active and unmet.',
        'physical_modes':28,'GPUs_used':8,'CPU_readout_workers':6,'Haar_check_samples':sum(r['samples'] for r in mc),
        'iid_canonical_measure_draws':0,'synthesis_replica_order':2,
        'training_times':protocol['training_times'],'reused_canonical_syntheses':4,'fresh_canonical_syntheses':0,
        'two_transfer_label_cases':sum(r['labels'] for r in jobs),'frozen_spaces':len(protocol['dimensions']),
        'new_trajectories':sum(not r['reused'] for r in frozen),'reused_trajectories':sum(r['reused'] for r in frozen),
        'evaluated_trajectories':len(frozen),'canonical_gates':len(evaluation['gate']),
        'passing_canonical_gates':[r for r in evaluation['gate'] if r['passes']],
        'independent_exponential_repeats':len(diagnostics['independent_exponential_repeats']),
        'physical_readout_sizes':[10,14,18],'physical_readout_pulse_pairs':len(readout_rows),
        'physical_readout_channel_evolutions':2*len(readout_rows),'full_CC_wrong_readout_signs_of_twelve':int(wrong_signs),
        'angular_stochastic_independent_samples':stochastic['independent_samples'],
        'angular_stochastic_runs_with_step_repeats':stochastic['runs_including_time_step_repeats'],
        'angular_stochastic_refined_rows':[r for r in stochastic['rows'] if r['dt']==.002],
        'angular_stochastic_max_mean_step_difference':max(r['max_mean_difference'] for r in stochastic['step_checks']),
        'angular_stochastic_max_initial_zero_error_over_SE':max(r['max_initial_zero_error_over_SE'] for r in stochastic['initial_zero_checks']),
        'angular_stochastic_drifts':{k:v for k,v in stochastic.items() if k.startswith('max_')},
        'uniform_stochastic_independent_samples':uniform['independent_stochastic_samples'],
        'uniform_stochastic_runs_with_step_repeats':uniform['stochastic_runs_with_step_repeats'],
        'uniform_stochastic_refined_rows':[r for r in uniform['stochastic_rows'] if r['dt']==.002],
        'uniform_stochastic_max_mean_step_difference':max(r['max_mean_difference'] for r in uniform['step_checks']),
        'uniform_stochastic_drifts':{k:v for k,v in uniform.items() if k.startswith('max_')},
        'uniform_snapshot_states':len(uniform['snapshot_rows']),
        'uniform_snapshot_pulse_pairs':sum(len(r['pulse_rows']) for r in uniform['snapshot_rows']),
        'uniform_snapshot_rows':uniform['snapshot_rows'],
        'stochastic_scope':'Gaussian-seeded SMF and discrete TWA use Pauli-reduced distinct-site X squared symbols. Estimated block standard errors are not rigorous confidence intervals. Same-seed time-step repeats do not add independent samples.',
        'uniform_snapshot_scope':'Existing best feasible state fits selected by their original fidelity. These readout evaluations are representation controls, not new predictions. Conditioned orbitals and conditioned Jastrow greatly reduce the original-model discrepancy without canonical synthesis.',
        'max_Haar_error_over_estimated_Frobenius_SE':max(c['error_over_SE'] for r in mc for c in r['checks']),
        'max_reused_canonical_synthesis_residual':max(r['synthesis_residual'] for r in jobs if r['method']=='canonical'),
        'max_initial_state_error':max(r['initial_error'] for r in frozen),'max_norm_error':max(r['norm_error'] for r in frozen),
        'max_independent_exponential_state_difference':max(r['max_state_difference'] for r in diagnostics['independent_exponential_repeats']),
        'max_nested_deletion_residual':max(max(r['joint_nesting_residual'],r['H_repair_nesting_residual']) for r in diagnostics['nested_deletion']),
        'max_readout_initial_diagonal_difference':max(r['initial_diagonal_difference'] for r in readout_checks if 'initial_diagonal_difference' in r),
        'max_readout_all_pair_correlator_difference':max(r['all_pair_correlator_difference'] for r in readout_checks if 'all_pair_correlator_difference' in r),
        'max_readout_Richardson_coefficient_error':max(r['Richardson_variance_coefficient_error'] for r in readout_checks if 'Richardson_variance_coefficient_error' in r),
        'max_readout_symmetry_residual':max(r['symmetry_residual'] for r in readout_checks if 'symmetry_residual' in r),
        'max_readout_trace_error':max(r['max_trace_error'] for r in readout_rows),
        'max_noise_quadrature_identity_error':max(r['variance_identity_error'] for m in evaluation['models'] for r in m['rows']),
        'max_survival_pulse_Richardson_error':max(r['Richardson_error'] for r in screen['pair_pulse_checks']),
        'new_physical_parameter_holdouts':0,'new_important_physics_established':False,'canonical_unique_predictive_structure_established':False,
        'novelty_priority_established':False,'equal_total_compute_advantage_established':False,'rigorous_MC_confidence_certificate':False,
        'interval_certified_physical_projection_bound':False,
        'readout_scope':'Phase twirling and multiple-quantum coherence readout have established antecedents. This is a finite-model application and mechanism diagnostic, not a new phase or canonical-only discovery. Count diagonal and all four-fermion expectations match before the pulse; the mixed states themselves differ by the deliberate coherence deletion.',
        'cross_copy_scope':'A coproduct contribution from the actual full-Gaussian symbol, not a probability bin or the physical connected-noise cumulant.',
        'tests_passed':int(match[1]),'tests_seconds':float(match[2]),'test_log_sha256':digest(log),
        'verified_file_count':len(verified),'verified_files':verified,
        'primary':[{'method':r['method'],'start':r['start'],'dimension':r['dimension'],**{k:v for k,v in r['primary'].items() if k not in ('connected_real','connected_imag','pulse_variances')}} for r in evaluation['models']],
        'noise_screen_time2':[{k:v for k,v in r.items() if k not in ('connected_real','connected_imag','pulse_variances')} for r in screen['rows'] if abs(r['time']-2)<1e-12],
        'capacity':diagnostics['capacity'],'cross_copy_fractions':[{k:r[k] for k in ('method','time','cross_fraction')} for r in jobs],
        'readout_coefficients_exact':exact_coefficient.tolist(),'posthoc_frozen_readout_predictions':curvatures,
        'remaining_gate':'A distribution-derived mechanism must predict an important physical discrepancy beyond strong controls, then pass genuinely new conditions and same-initial deletion. The higher-symbol coordinate pilot does not pass.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    lookup={(r['method'],r['start']):r for r in evaluation['models']}
    styles=[('canonical_full_q16','Two-transfer full','#0072B2','-'),('canonical_cross_copy_q16','Two-transfer cross-copy','#D55E00','-'),
        ('canonical_joint_q16','Single + two-transfer','#CC79A7','-'),('unoptimized_joint_q16','Unoptimized joint','#009E73',':'),
        ('canonical_single_H_q16','Same single + H','#E69F00','--'),('Hamiltonian_q16','Hamiltonian','black','-.')]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for method,label,color,line in styles:
        rows=lookup[method,'restart']['rows'];t=[r['time'] for r in rows]
        for ax,key in zip(axes,('count_TV','two_transfer_connected_L2_error','pair_pulse_variance_L2_error')):
            ax.plot(t,np.maximum([r[key] for r in rows],1e-14),line,color=color,lw=1.5,label=label)
    for ax,title in zip(axes,('Count total-variation error','Connected transfer-noise error','Pair-pulse variance error')):
        ax.set(title=title,xlabel='Time',xlim=(1.4,2.5),yscale='log',ylim=(1e-12,3));ax.axvline(2,color='0.75',ls=':',lw=.8)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=8)
    fig.suptitle('Identical t=1.35 restart; all displayed spaces have dimension 25',y=.83,fontsize=10)
    fig.tight_layout(rect=(0,0,1,.8));save(fig,'multipair_noise_predictions')
    fig,axes=plt.subplots(2,3,figsize=(11,6),sharex=True)
    for col,L in enumerate((10,14,18)):
        for row,phi in enumerate((0.,float(np.pi/4))):
            ax=axes[row,col]
            for method,label,color,line in [('exact','Exact','#0072B2','o-'),('full_cc','Full pair-CC','#D55E00','s--')]:
                rows=sorted([r for r in readout_rows if r['L']==L and r['q']==1 and r['phase']==phi and r['method']==method],key=lambda r:r['epsilon'])
                ax.plot([r['epsilon']**2 for r in rows],[r['variance_coefficient'] for r in rows],line,color=color,label=label,ms=4)
                ax.scatter([0],[rows[0]['analytic_coefficient']],marker='x',color=color,s=35)
            ax.axhline(0,color='0.8',lw=.8);ax.set(title=f'L={L}, pulse phase '+('0' if row==0 else 'pi/4'),ylabel=r'$\Delta\mathrm{Var}(m)/\epsilon^2$')
    for ax in axes[-1]:ax.set_xlabel('Pulse area squared')
    axes[0,0].legend(fontsize=8);fig.suptitle('Even-count coherence retained vs fully dephased: identical initial counts and four-point data',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94));save(fig,'multipair_phase_readout')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method,label,color in [('exact','Exact','#0072B2'),('full_cc','Full pair-CC','#D55E00')]:
        r=readout_lookup[14,1,method,0.,.4]
        axes[0].plot(np.arange(8),np.array(r['even_twirl_counts'])-r['full_twirl_counts'],'o-',color=color,label=label)
    axes[0].axhline(0,color='0.8',lw=.8);axes[0].set(xlabel='Transferred pair count m',ylabel='Difference in count probability',title='L14, q1, phase 0, pulse area 0.4');axes[0].legend(fontsize=8)
    names=['canonical_contrast_q8','canonical_joint_q16','unoptimized_joint_q16','canonical_single_H_q16']
    labels=['Single d=17','Joint d=25','Unoptimized\njoint d=25','Single + H\nd=25'];x=np.arange(4)
    c={(r['method'],r['start']):r for r in curvatures}
    axes[1].bar(x,[c[n,'restart']['L2_error'] for n in names]);axes[1].set(xticks=x,xticklabels=labels,yscale='log',ylabel='Four readout curvatures: L2 error',title='Previously frozen restart predictions');axes[1].tick_params(axis='x',labelsize=8)
    fig.tight_layout();save(fig,'multipair_readout_controls')
    fig,axes=plt.subplots(1,4,figsize=(13,3.7),sharex=True)
    refined={(r['L'],r['method']):r for r in stochastic['rows'] if r['dt']==.002}
    sizes=[10,14,18]
    for j,ax in enumerate(axes):
        ax.plot(sizes,[refined[L,'gaussian']['exact'][j] for L in sizes],'ko-',label='Exact')
        for method,label,color,marker in [('gaussian','Gaussian SMF','#0072B2','s'),('discrete','Discrete TWA','#009E73','^')]:
            ax.errorbar(sizes,[refined[L,method]['mean'][j] for L in sizes],
                yerr=[2*refined[L,method]['estimated_SE'][j] for L in sizes],color=color,marker=marker,label=label,capsize=3)
        ax.axhline(0,color='0.8',lw=.8)
        ax.set(xlabel='Pair levels L',xticks=sizes,title=f'q={1 if j<2 else 3}, phase '+('0' if j%2==0 else 'pi/4'))
    axes[0].set_ylabel('Weak-pulse variance curvature')
    axes[0].legend(fontsize=8)
    fig.suptitle('Known angular conditions at t=2; error bars: two estimated standard errors',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.93));save(fig,'multipair_stochastic_controls')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    fits={r['method']:r for r in uniform['snapshot_rows'] if r['time']==2.15}
    stoch={r['method']:r for r in uniform['stochastic_rows'] if r['time']==2.15 and r['dt']==.002}
    names=['pair_CC_rank6','radial_full_CC','conditioned_orbitals','conditioned_Jastrow','two_conditioned_orbits']
    labels=['Pair CC','Radial CC','Conditional','Cond. Jastrow','Two orbits']
    x=np.arange(len(names)+2)
    values=[fits[n]['coefficient_L2_error'] for n in names]+[stoch[n]['L2_error'] for n in ('gaussian','discrete')]
    axes[0].bar(x,values,color=['#0072B2']*5+['#D55E00']*2)
    axes[0].set(xticks=x,xticklabels=labels+['SMF','DTWA'],yscale='log',ylabel='Two readout curvatures: L2 error',title='Original L12 G2, first pulse t=2.15')
    axes[0].tick_params(axis='x',rotation=35,labelsize=8)
    for name,label in [('exact','Exact'),('radial_full_CC','Radial CC'),('conditioned_orbitals','Conditional'),('conditioned_Jastrow','Cond. Jastrow')]:
        axes[1].plot([0,1],[p['variance_difference'] for p in fits[name]['pulse_rows']],'o-',label=label)
    axes[1].set(xticks=[0,1],xticklabels=['0','pi/4'],xlabel='Pulse phase',ylabel='Count-variance difference',title='Finite pulse area 0.4')
    axes[1].legend(fontsize=8)
    fig.suptitle('Blue bars use existing fitted states; orange bars propagate stochastic initial samples',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.94));save(fig,'multipair_uniform_controls')
    print(json.dumps({k:audit[k] for k in ('Haar_check_samples','new_trajectories','reused_trajectories','canonical_gates','passing_canonical_gates','physical_readout_pulse_pairs','full_CC_wrong_readout_signs_of_twelve','max_initial_state_error','max_norm_error','tests_passed','verified_file_count')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
