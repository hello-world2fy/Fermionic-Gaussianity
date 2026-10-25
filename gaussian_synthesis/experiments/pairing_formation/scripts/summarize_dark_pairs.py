#!/usr/bin/env python3
"""Twelfth-round fixed archive: higher-transfer dark sectors and phase moments."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,hashlib,json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(out):
    out.mkdir(parents=True,exist_ok=False);root=out.parent;verified=[]
    def read(path):return json.loads(path.read_text())
    def verify(path,value):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==value,path
        verified.append(str(path))
    dark={L:read(root/'dark_pairs_01'/f'L{L}_diagnostic.json') for L in (10,14,18)}
    for case in dark.values():
        for path,value in case['source_files'].items():verify(Path(path),value)
    controls={L:read(root/'dark_pair_controls_01'/f'L{L}_evaluation.json') for L in (14,18)}
    for L,case in controls.items():verify(root/'dark_pair_controls_01'/f'L{L}_frozen_predictions.npz',case['frozen']['sha256'])
    onset={r:read(root/'dark_onsets_01'/f'r{r}_onset.json') for r in (5,7,9,11)}
    training=read(root/'dark_gaussian_training_01/protocol.json')
    verify(Path(training['source']),training['source_sha256'])
    verify(root/'dark_gaussian_training_01/training.npz',training['training_sha256'])
    synthesis=read(root/'dark_gaussian_training_01/synthesis.json')
    backends=read(root/'dark_gaussian_training_01/backend_checks.json')
    quadrature={name:read(root/name/'assessment.json') for name in ('dark_gaussian_assessment_01','dark_gaussian_assessment_02')}
    for name,assessment in quadrature.items():
        for path,value in assessment['source_files'].items():verify(Path(path),value)
        verify(root/name/'coherent_moments.npz',assessment['moments_sha256'])
        assert all(group['largest_passing_rank']==0 for method in assessment['methods'].values() for group in method['groups'].values())
    phase_dir=root/'dark_phase_parts_01';phase=read(phase_dir/'evaluation.json');protocol=read(phase_dir/'protocol.json')
    verify(phase_dir/'frozen_spaces.npz',protocol['frozen_sha256'])
    for record in phase['models']:verify(phase_dir/record['file'],record['sha256'])
    refinement=read(root/'dark_phase_refinement_01/refinement.json')
    verify(root/'dark_gaussian_training_01/training.npz',refinement['input_sha256'])
    verify(root/'dark_phase_refinement_01/refinement.npz',refinement['output_sha256'])
    log=(root/'dark_pair_tests_01.log').read_text();match=re.search(r'Ran (\d+) tests in ([\d.]+)s',log)
    assert match and log.rstrip().endswith('OK');(out/'tests.log').write_text(log)
    audit={'classification':'Twelfth-round retrospective mechanism and coherent-information experiment; original physical discovery goal remains active.',
        'physical_dark_conditions':3,'dark_model_diagnostics':sum(len(x['models']) for x in dark.values()),
        'same_initial_deletion_and_full_records':12,'short_time_sizes':[10,14,18,22],
        'full_Gaussian_training_L':14,'full_Gaussian_physical_modes':28,'Haar_rotation_group':'SO(56)',
        'Haar_coherent_integration_samples':sum(x['samples'] for x in quadrature.values()),'calibration_samples':4096,'GPUs_used':8,
        'stable_population_coordinate_directions':0,'canonical_iid_measure_draws':False,
        'actual_full_Gaussian_coherent_moments':True,'deterministic_complex_label_integrals':True,
        'phase_partition_spaces':len(protocol['dimensions']),'phase_partition_trajectories':len(phase['models']),
        'phase_refinement_trajectories':len(refinement['checks']),'new_parameter_holdouts':0,
        'source_annihilator_darkness_is_full_H_invariance':False,'dark_population_proves_long_lifetime':False,
        'general_size_onset_theorem_proved':False,'new_important_physics_established':False,'novelty_priority_established':False,
        'canonical_unique_predictive_structure_established':False,'equal_compute_advantage_established':False,
        'verified_file_count':len(verified),'verified_files':verified,
        'max_phase_initial_error':max(x['initial_error'] for x in phase['models']),
        'max_phase_norm_error':max(x['norm_error'] for x in phase['models']),
        'max_deletion_initial_error':max(x['initial_error'] for case in controls.values() for x in case['models']),
        'max_deletion_norm_error':max(x['norm_error'] for case in controls.values() for x in case['models']),
        'max_symmetry_reference_difference':max(x['max_full_reference_difference'] for x in controls.values()),
        'tests':{'passed':int(match[1]),'seconds':float(match[2]),'log':'tests.log'},
        'dark_primary':{str(L):{name:rows[40] for name,rows in case['models'].items()} for L,case in dark.items()},
        'deletion_primary':{str(L):[{'name':x['name'],'dimension':x['dimension'],**x['rows'][40]} for x in case['models']] for L,case in controls.items()},
        'onset':onset,'synthesis':synthesis,'GPU_backend_checks':backends,
        'quadrature':quadrature,'phase_geometry':protocol['geometry'],'phase_part_diagnostics':protocol['part_diagnostics'],
        'phase_primary':[{**{k:r[k] for k in ('name','start','dimension')},**next(x for x in r['rows'] if abs(x['time']-2)<1e-12)} for r in phase['models']],
        'phase_refinement':refinement,
        'remaining_gate':'Explain and predict the significant higher-transfer dark-sector effect using canonical-derived connected coherent structure that survives physical dark projections, nonlinear cluster alternatives and Hamiltonian response, then verify genuinely unseen conditions and identical-initial-state deletion.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    colors={'exact':'black','full_cc':'#cc79a7','matched':'#0072b2','Hamiltonian_block_response':'#d55e00'}
    labels={'exact':'Exact','full_cc':'Full pair-CC','matched':'Matched rotations','Hamiltonian_block_response':'Hamiltonian response'}
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for ax,L in zip(axes,(14,18)):
        for method in colors:
            rows=dark[L]['models'][method]
            ax.plot([x['time'] for x in rows],[x['multiple_transfer_either_dark'] for x in rows],color=colors[method],label=labels[method])
        ax.set_title(f'L={L}');ax.set_xlabel('time');ax.grid(alpha=.2);ax.axvline(2,color='gray',ls=':',lw=.7)
    axes[0].set_ylabel('either-band dark probability, m >= 2');axes[0].legend(fontsize=8)
    fig.suptitle('Source-annihilator dark components in higher transfer sectors\nPopulation alone does not establish trapping or a lifetime',fontsize=10);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'dark_population.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    names={'delete_either_m1':'Delete m=1 darkness','delete_either_multiple':'Delete m>=2 darkness','delete_both_multiple':'Delete both-band m>=2 darkness'}
    for ax,L in zip(axes,(14,18)):
        for row in controls[L]['models']:
            if row['name'] not in names:continue
            ax.plot([x['time'] for x in row['rows']],[x['count_TV'] for x in row['rows']],label=names[row['name']])
        ax.set_title(f'L={L}');ax.set_xlabel('time');ax.grid(alpha=.2);ax.axvline(2,color='gray',ls=':',lw=.7)
    axes[0].set_ylabel('count total variation');axes[0].legend(fontsize=7)
    fig.suptitle('Identical Fermi sea under projected-Hamiltonian deletion\nThe higher-transfer effect is larger than the m=1-only effect',fontsize=10);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'dark_deletion.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2));a=quadrature['dark_gaussian_assessment_02'];x=np.arange(3)
    for shift,group,label in ((-.18,'linear_count','Count coordinate'),(.18,'linear_dark','Dark coordinate')):
        checks=[a['methods'][method]['groups'][group]['rank_checks'][0] for method in ('canonical','husimi_moment','linear_frame')]
        axes[0].bar(x+shift,[r['principal_cosines'][0] for r in checks],width=.34,label=label)
        axes[1].bar(x+shift,[r['smallest_singular_over_Frobenius_SE'] for r in checks],width=.34,label=label)
    axes[0].axhline(.95,color='black',ls='--',lw=1);axes[1].axhline(3,color='black',ls='--',lw=1)
    for ax in axes:ax.set_xticks(x,['Canonical','Cubic','Linear']);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('independent-half direction cosine');axes[0].set_ylim(.7,1.);axes[1].set_ylabel('signal singular value / Frobenius SE');axes[0].legend(fontsize=8)
    fig.suptitle('4,194,304 independent Haar atoms: both gates must pass\nPopulation-coordinate directions were not used for forecasts',fontsize=10);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'coherent_moment_stability.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.2))
    selected={'shared':'Shared (22)','canonical_complex_q2':'Canonical A (24)',
              'canonical_phase_pair_q4':'Canonical A+B (26)','physical_dark_q2':'Physical dark (24)',
              'Hamiltonian_Krylov_q4':'Hamiltonian Krylov (26)'}
    for row in phase['models']:
        if row['start']!='restart' or row['name'] not in selected:continue
        times=[v['time'] for v in row['rows']]
        for ax,key in zip(axes,('count_TV','current_L1_error','multiple_dark_error')):
            ax.plot(times,[abs(v[key]) for v in row['rows']],label=selected[row['name']])
    for ax,label in zip(axes,('count total variation','sector-current L1 error','absolute higher-dark error')):
        ax.set_xlabel('time');ax.set_ylabel(label);ax.grid(alpha=.2);ax.axvline(2,color='gray',ls=':',lw=.7)
    axes[0].legend(fontsize=7)
    fig.suptitle('Exact training-state restart: deterministic complex phase labels\nNo simultaneous advantage over strong ordinary response controls',fontsize=10);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'phase_partition_predictions.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps({k:audit[k] for k in ('verified_file_count','Haar_coherent_integration_samples','phase_partition_trajectories','max_phase_initial_error','max_phase_norm_error','tests')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();main(args.output)
