#!/usr/bin/env python3
"""Fixed ninth-round archive: connected-transfer pilot and weighted bound."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,hashlib,re
from fractions import Fraction
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.weighted_count_bound import weighted_statistics


def main(out):
    out.mkdir(parents=True,exist_ok=False);root=out.parent;pilot=root/'cluster_feedback_pilot_01'
    evaluation=json.loads((pilot/'evaluation.json').read_text());training=np.load(pilot/'frozen_training.npz')
    verified=[]
    record=json.loads((pilot/'training_hash.json').read_text())
    assert hashlib.sha256((pilot/record['file']).read_bytes()).hexdigest()==record['sha256'];verified.append(record['file'])
    for name,digest in json.loads((pilot/'frozen_forecasts.json').read_text()).items():
        assert hashlib.sha256((pilot/name).read_bytes()).hexdigest()==digest;verified.append(name)
    geometry={'scope':'Retrospective geometry; double projection removes all one-sided variations of the common rank-one m=2 shape.'}
    U,s,Vh=np.linalg.svd(training['common_radial'],full_matrices=False)
    Qp=np.eye(len(U))-np.outer(U[:,0],U[:,0].conj());Qh=np.eye(len(Vh))-np.outer(Vh[0].conj(),Vh[0])
    for method in ('direct','witness','Hamiltonian'):
        tensor=training[method+'_tensor'];normal=Qp@tensor@Qh
        geometry[method]={'tensor_norm':float(np.linalg.norm(tensor)),'double_orthogonal_norm':float(np.linalg.norm(normal)),
          'double_orthogonal_squared_fraction':float(np.linalg.norm(normal)**2/np.linalg.norm(tensor)**2),
          'double_orthogonal_singular_values':np.linalg.svd(normal,compute_uv=False).tolist()}
    for q in (1,2,4):
        A=training['direct_modes'][:q].reshape(q,-1);row={}
        for method in ('witness','Hamiltonian'):
            B=training[method+'_modes'][:q].reshape(q,-1)
            row['direct_'+method+'_subspace_cosines']=np.linalg.svd(A.conj()@B.T,compute_uv=False).tolist()
        geometry[f'q{q}']=row
    (out/'geometry_diagnostic.json').write_text(json.dumps(geometry,indent=2))
    old_geometry=json.loads((pilot/'geometry_diagnostic.json').read_text())
    assert max(abs(geometry[x]['double_orthogonal_squared_fraction']-old_geometry[x]['double_orthogonal_squared_fraction']) for x in ('direct','witness','Hamiltonian'))<1e-14
    orbit=json.loads((root/'orbital_fluctuations_01/summary.json').read_text())
    weighted=json.loads((root/'weighted_count_bound_02/evaluation.json').read_text())
    random=json.loads((root/'weighted_count_bound_02/independent_checks.json').read_text())
    rational=json.loads((root/'rational_pulse_certificate_01/certificate.json').read_text())
    def fraction(key):
        row=rational[key];return Fraction(int(row['numerator']),int(row['denominator']))
    assert fraction('excess_lower')>0 and fraction('excess_upper')>fraction('excess_lower')
    distance=fraction('trace_distance_lower');R=fraction('full_number_operator_range_upper');RX=fraction('full_number_X_range')
    assert R*distance+Fraction(5,6)*RX*RX*distance*distance<fraction('excess_lower')
    assert distance*distance==fraction('infidelity_lower')
    validations=[json.loads((root/f'cluster_feedback_physical_validation_0{i}/validation.json').read_text()) for i in (1,2)]
    model=picket_fence(12,2.);pulse=next(x for x in weighted['cases'] if x['name']=='uniform_L12_t2.15');w=np.array(pulse['weights'])
    differences=[]
    for row in validations[-1]:
        name=row['name'];old=np.load(pilot/f'{name}_prediction.npz')['states']
        fine=np.load(root/'cluster_feedback_physical_validation_02'/f'{name}.npz')['states']
        count_errors=[];energies=[]
        for a,b in zip(old,fine):
            a=a/np.linalg.norm(a);b=b/np.linalg.norm(b)
            ca=np.bincount(model.m,weights=abs(a)**2,minlength=7)
            cb=np.bincount(model.m,weights=abs(b)**2,minlength=7)
            count_errors.append(float(.5*np.sum(abs(ca-cb))))
            energies.append(float(np.vdot(b,model.H@b).real))
        differences.append({'name':name,'max_count_TV_change_from_pilot':max(count_errors),
          'max_normalized_energy_drift':float(np.max(abs(np.array(energies)-energies[0]))),
          'fine_fixed_pulse_weight':weighted_statistics(model,fine[43],w)})
    tests=(root/'cluster_feedback_tests_02.log').read_text();match=re.search(r'Ran (\d+) tests in ([\d.]+)s',tests)
    assert match and tests.rstrip().endswith('OK')
    (out/'tests.log').write_text(tests)
    audit={'classification':'Ninth-round retrospective structure pilot and analytic-family certification. Original three-stage research goal remains active.',
      'pilot_predictions':11,'frozen_files_verified':len(verified),'verified_files':verified,
      'max_initial_state_difference':evaluation['max_initial_state_difference'],
      'pilot_primary_time':2.15,'pilot_primary':[{ 'name':x['name'],**x['rows'][43]} for x in evaluation['models']],
      'geometry':geometry,'orbit_search_cases':len(orbit),'orbit_search_starts':sum(x['starts'] for x in orbit),'GPUs_used':8,
      'orbit_search':[{'name':x['name'],'best_found':x['best_found'],'bound':x['bound']} for x in orbit],
      'refinement_max_ray_differences':[max(x['max_normalized_ray_difference'] for x in v) for v in validations],
      'max_count_Ehrenfest_error':max(x['max_count_sector_rate_error'] for v in validations for row in v for x in row['count_Ehrenfest_checks']),
      'refined_observable_comparisons':differences,
      'coordinate_error_refinements_terminated_incomplete':2,
      'weighted_bound_independent_Fock_checks':sum(x['cases'] for x in random['arbitrary_Hermitian_onebody']),
      'weighted_bound_independent_Pfaffian_checks':sum(x['cases'] for x in random['independent_rotated_Pfaffian']),
      'weighted_bound_cases':weighted['cases'],'weighted_bound_failed_setup_attempts':1,
      'rational_late_pulse_certificate':{key:rational[key] for key in ('excess_lower','excess_upper','infidelity_lower','Taylor_remainder_norm_upper','float64_reference_state_difference')},
      'tests':{'passed':int(match[1]),'seconds':float(match[2]),'log':'tests.log'},
      'canonical_measure_samples_used':False,'equal_compute_budget_advantage_established':False,
      'canonical_unique_prediction_established':False,'new_important_physics_established':False,
      'novelty_priority_established':False,'all_results_interval_certified':False,
      'late_pulse_fixed_rational_observable_has_exact_enclosure':True,
      'remaining_gate':'Actual canonical-derived structure must predict a physically important strong-baseline discrepancy on new parameters and survive identical-initial-state deletion. The weighted late-pulse certificate is retrospective, extremely weak, and independent of canonical synthesis.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    names=['radial_cc_q0','common_q0','direct_q2','witness_q2','Hamiltonian_q2']
    labels=['Radial pair-CC','Shared m=2 cluster','Direct +2 directions','Witness +2 directions','Hamiltonian +2 directions']
    colors=['#666666','#e69f00','#0072b2','#cc79a7','#009e73']
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,key,title in zip(axes.ravel(),('count_TV','current_L1_error','fidelity','m2_negativity'),
                           ('Counting distribution error','Resolved current error','State infidelity','Conditional m=2 negativity')):
        for name,label,color in zip(names,labels,colors):
            rows=next(x['rows'] for x in evaluation['models'] if x['name']==name)
            t=[x['time'] for x in rows];y=np.array([np.nan if x[key] is None else x[key] for x in rows])
            if key=='fidelity':y=1-y
            ax.plot(t,y,label=label,color=color,lw=1.6)
        if key=='m2_negativity':ax.plot(t,[x[key] for x in evaluation['exact']],color='black',ls='--',label='Exact')
        ax.axvline(1,color='gray',lw=.7,ls=':');ax.axvline(2.15,color='gray',lw=.7,ls=':')
        ax.set_title(title);ax.set_xlabel('time');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[1,1].legend(fontsize=8)
    fig.suptitle('L=12, G=2: all trajectories start at the same Fermi sea\nRetrospective pilot; extra connected directions do not give a canonical advantage')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'cluster_feedback.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].bar(model.energies,w,width=.13,color=['#0072b2']*6+['#d55e00']*6)
    axes[0].set_xlabel('single-particle energy');axes[0].set_ylabel('unit-norm weight')
    axes[0].set_title('Late-pulse weighted witness (selected retrospectively)');axes[0].axhline(0,color='gray',lw=.7)
    selected=[x for x in weighted['cases'] if 'infidelity_lower_bound' in x['weighted'] and x['weighted']['infidelity_lower_bound']>0]
    axes[1].bar(np.arange(len(selected)),[x['weighted']['infidelity_lower_bound'] for x in selected],color='#009e73')
    axes[1].set_yscale('log');axes[1].set_xticks(np.arange(len(selected)),['BCS t=.5','BCS t=1','BCS t=2.15','Angular L10','Angular L14'],rotation=20)
    axes[1].set_ylabel('certified lower bound on 1 - F (float64 evaluation)')
    axes[1].set_title('Global pure projected-Gaussian exclusions; weak bounds')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'weighted_certificates.{ext}',dpi=170)
    plt.close(fig)
    print(json.dumps({k:audit[k] for k in ('frozen_files_verified','refinement_max_ray_differences','max_count_Ehrenfest_error','tests')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();main(args.output)
