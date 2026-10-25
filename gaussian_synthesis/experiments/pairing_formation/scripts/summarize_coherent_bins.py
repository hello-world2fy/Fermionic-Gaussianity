#!/usr/bin/env python3
"""Fixed tenth-round archive; all physical comparisons are retrospective."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,json,hashlib,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(out):
    out.mkdir(parents=True,exist_ok=False);root=out.parent
    bins=json.loads((root/'gaussian_bins_assessment_01/assessment.json').read_text())
    prediction=root/'gaussian_bins_prediction_pilot_01';evaluation=json.loads((prediction/'evaluation.json').read_text())
    protocol=json.loads((prediction/'protocol.json').read_text())
    assert hashlib.sha256((prediction/'frozen_spaces.npz').read_bytes()).hexdigest()==protocol['spaces_hash']
    verified=['frozen_spaces.npz']
    for r in json.loads((prediction/'frozen_predictions.json').read_text()):
        assert hashlib.sha256((prediction/r['file']).read_bytes()).hexdigest()==r['sha256'];verified.append(r['file'])
    noise=json.loads((root/'gaussian_bins_noise_01/sensitivity.json').read_text())
    noise_hash=json.loads((root/'gaussian_bins_noise_01/frozen_record.json').read_text())['sha256']
    assert hashlib.sha256((root/'gaussian_bins_noise_01/frozen_quadrature_variants.npz').read_bytes()).hexdigest()==noise_hash
    spectra=[json.loads((root/'chirality_spectrum_01'/name).read_text()) for name in ('L10_eta1.json','L14_eta1.json','L18_eta1.json','L14_eta0.85.json')]
    spectra_primary=[]
    for case in spectra:
        spectra_primary.append({'L':case['L'],'eta':case['eta'],'models':{name:{k:rows[3][k] for k in
          ('chi_second_per_r_squared','macroscopic_tail_probability','two_circular_orbit_projection_excluding_endpoints')} for name,rows in case['models'].items()}})
    testlog=(root/'coherent_bins_tests_01.log').read_text();match=re.search(r'Ran (\d+) tests in ([\d.]+)s',testlog)
    assert match and testlog.rstrip().endswith('OK');(out/'tests.log').write_text(testlog)
    exact_rows=[y for x in evaluation['models'] if x['name']=='exact_symmetry_space' for y in x['rows']]
    audit={'classification':'Tenth-round retrospective physical screen and actual full-Gaussian coherent-integral pilot. No new-parameter forecasts.',
      'physical_conditions':4,'physical_spectral_primary':spectra_primary,
      'full_Gaussian_integration':bins,'coherent_bin_geometry':protocol['geometry'],
      'pilot_predictions':len(evaluation['models']),'verified_frozen_files':len(verified),
      'max_initial_state_difference':evaluation['max_initial_error'],'max_norm_error':evaluation['max_norm_error'],
      'max_closed_symmetry_count_TV':max(x['count_TV'] for x in exact_rows),
      'max_closed_symmetry_infidelity':max(1-x['fidelity'] for x in exact_rows),
      'pilot_primary':[{'name':r['name'],'start':r['start'],'dimension':r['dimension'],
          **next(x for x in r['rows'] if x['time']==2.)} for r in evaluation['models']],
      'quadrature_sensitivity':noise['summary'],'quadrature_sensitivity_records':len(noise['records']),
      'tests':{'passed':int(match[1]),'seconds':float(match[2]),'log':'tests.log'},
      'actual_full_Gaussian_canonical_integrals':True,'iid_canonical_measure_draws':False,
      'canonical_unique_predictive_structure_established':False,'new_important_physics_established':False,
      'two_simple_circular_branches_explain_late_state':False,
      'thermodynamic_chiral_phase_or_cat_excluded':False,
      'equal_total_compute_advantage_established':False,'rigorous_MC_confidence_certificate':False,
      'remaining_gate':'Find physically interpretable canonical structure beyond the tested count/circulation and conventional response spaces; freeze it before genuinely new physical conditions and test identical-initial-state deletion. Current pilot does not pass.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    labels={'canonical':'Canonical k=2','husimi_moment':'Unoptimized cubic moment','linear_frame':'Linear Haar frame'}
    colors={'canonical':'#0072b2','husimi_moment':'#cc79a7','linear_frame':'#009e73'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for name in labels:
        for ax,group in zip(axes,('count_and_circulation','count_only')):
            row=bins['methods'][name][group];sv=np.array(row['singular_values']);sv=sv[sv>1e-12]
            ax.semilogy(np.arange(1,len(sv)+1),sv/row['block_Frobenius_standard_error'],'o-',color=colors[name],label=labels[name])
    for ax,title in zip(axes,('Count x circulation bins','Count bins only')):
        ax.axhline(3,color='gray',ls=':',label='Declared amplitude gate')
        ax.set_xlabel('singular-value rank');ax.set_ylabel('singular value / Frobenius SE')
        ax.set_title(title);ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle('Full Gaussian quadrature: 1,048,576 atoms, two independent half pools\nKnown F, training state and C2(training state) removed; angle gate also required',fontsize=11)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'bin_stability.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    methods=[('shared','Shared directions only','#777777'),('canonical_count_only_q2','Canonical +2','#0072b2'),
      ('linear_frame_count_only_q2','Linear frame +2','#009e73'),('count_polynomial_q2','Count polynomials +2','#e69f00'),
      ('Hamiltonian_response_q2','Hamiltonian response +2','#d55e00')]
    for ax,key,title in zip(axes.ravel(),('fidelity','count_TV','current_L1_error','chi_second_error'),
      ('State infidelity','Counting distribution error','Resolved current error','Circulation second-moment error')):
        for name,label,color in methods:
            row=next(x for x in evaluation['models'] if x['name']==name and x['start']=='restart')
            times=[r['time'] for r in row['rows']];y=np.array([r[key] for r in row['rows']])
            if key=='fidelity':y=1-y
            if key=='chi_second_error':y=abs(y)
            ax.plot(times,y,color=color,label=label)
        ax.axvline(2,color='gray',ls=':',lw=.8);ax.set_xlabel('time');ax.set_title(title);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Identical exact initial state at t=1; structures frozen before propagation\nL=10, G=4, eta=1: inspected-condition pilot, not a new-parameter forecast',fontsize=11)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'bin_predictions.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for name,label,color in [('exact','Exact','#111111'),('full_cc','Variational pair-CC','#d55e00'),('riccati','Riccati','#cc79a7'),('Gaussian_span_q2','Generated Gaussian q=2','#009e73')]:
        for ax,key in zip(axes,('chi_second_per_r_squared','two_circular_orbit_projection_excluding_endpoints')):
            ax.plot([10,14,18],[next(c for c in spectra if c['L']==L and c['eta']==1)['models'][name][3][key] for L in (10,14,18)],'o-',label=label,color=color)
    axes[0].set_ylabel('physical <chi^2> / r^2');axes[1].set_ylabel('squared projection, excluding m=0,r')
    axes[0].set_title('Circulating-pair fluctuations');axes[1].set_title('Two fixed circular reference branches')
    for ax in axes:ax.set_xlabel('pair levels L');ax.set_xticks([10,14,18]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Finite-size physical screen at t=2, G=4, eta=1\nNo thermodynamic phase or chiral-cat conclusion',fontsize=11)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'physical_circulation.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps({k:audit[k] for k in ('pilot_predictions','verified_frozen_files','max_initial_state_difference','max_norm_error','max_closed_symmetry_count_TV','tests')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();main(args.output)
