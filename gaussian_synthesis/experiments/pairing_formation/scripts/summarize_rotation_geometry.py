#!/usr/bin/env python3
"""Eleventh-round fixed archive: nonlinear field shapes and strong controls."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
from fractions import Fraction
import argparse,json,hashlib,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.gaussian_bins import angular_invariant_basis
from pairing_formation.finite_dictionary_cost import FiniteDictionaryCertificate


def main(out):
    out.mkdir(parents=True,exist_ok=False);root=out.parent;verified=[];geometry={};frozen={}
    runs=('rotation_geometry_01','rotation_geometry_precision_01','rotation_geometry_enlarged_01','rotation_geometry_shifted_01','rotation_geometry_independent_01')
    def verify(path,digest):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,str(path);verified.append(str(path.relative_to(root)))
    for name in runs:
        geometry[name]={};frozen[name]={}
        for L in (14,18):
            d=root/name;record=json.loads((d/f'L{L}_frozen.json').read_text());frozen[name][L]=record
            verify(d/f'L{L}_frozen_spaces.npz',record['frozen_spaces_sha256'])
            for r in record['predictions']:verify(d/r['file'],r['sha256'])
            geometry[name][L]=json.loads((d/f'L{L}_evaluation.json').read_text())
    rank={L:json.loads((root/'rotation_rank_controls_02'/f'L{L}_evaluation.json').read_text()) for L in (14,18)}
    for L,result in rank.items():
        for filename,digest in result['frozen']['files'].items():verify(root/'rotation_rank_controls_02'/filename,digest)
    mechanism={L:json.loads((root/'rotation_mechanism_audit_01'/f'L{L}_audit.json').read_text()) for L in (14,18)}
    for L,r in mechanism.items():verify(root/'rotation_mechanism_audit_01'/f'L{L}_frozen_predictions.npz',r['predictions_sha256'])
    certificates={}
    for L in (14,18):
        d=root/'rotation_dictionary_cost_01';r=json.loads((d/f'L{L}_certificate.json').read_text());verify(d/f'L{L}_certificate_inputs.npz',r['input_sha256'])
        x=np.load(d/f'L{L}_certificate_inputs.npz');checker=FiniteDictionaryCertificate(x['atoms'],x['free_basis'],x['state'])
        for w,cert in zip(x['witnesses'],r['certificates']):
            e=cert['error_allowance'];epsilon=Fraction(int(e['numerator']),int(e['denominator']))
            assert checker.evaluate(w,epsilon)==cert
        certificates[L]=r
    mode_dir=root/'gaussian_modes_01';mode_protocol=json.loads((mode_dir/'protocol.json').read_text());modes=[]
    for case in mode_protocol['cases']:
        tag=case['tag'];chosen={};parameters={}
        if case['source_sha256'] is not None:verify(Path(case['source']),case['source_sha256'])
        for kind in ('target','witness'):
            rows=json.loads((mode_dir/f'{tag}_{kind}.json').read_text());data=np.load(mode_dir/f'{tag}_{kind}.npz')
            i=max(range(len(rows)),key=lambda j:rows[j]['score_normalized_source']);chosen[kind]=data['atoms'][i];parameters[kind]=data['parameters'][i]
        row={**case}
        if tag.startswith('angular'):
            model,forms=angular_model(10);Q,_=angular_invariant_basis(model);chosen={k:Q.T@v for k,v in chosen.items()}
            U,_=np.linalg.qr(forms);shapes={}
            for kind,p in parameters.items():
                z=p[:10]+1j*p[10:];z[:5]*=-1;z=np.exp(z)
                mean_qp=float(np.sum(abs(z)**2/(1+abs(z)**2)))
                shapes[kind]={'mean_quasiparticle_pairs_before_number_projection':mean_qp,
                  'near_Fermi_boundary':bool(mean_qp<1e-6),
                  'first_harmonic_profile_fraction':None if mean_qp<1e-6 else [float(np.linalg.norm(U.T@v)**2/np.linalg.norm(v)**2) for v in (z[:5],z[5:])]}
            row['onsite_mode_shapes']=shapes
        a,b=chosen['target'],chosen['witness'];row['direct_witness_projected_orbit_fidelity']=float(abs(np.vdot(a,b))**2/(np.linalg.norm(a)*np.linalg.norm(b))**2);modes.append(row)
    general=json.loads((root/'general_gaussian_modes_01/summary.json').read_text())
    refinements=[]
    for L in (14,18):
        a=np.load(root/'rotation_geometry_01'/f'L{L}_rotation_profiles16.npz')['states']
        b=np.load(root/'rotation_geometry_precision_01'/f'L{L}_rotation_profiles16.npz')['states']
        refinements.append({'L':L,'max_same_profile_state_difference_after_rank_cutoff_change':float(np.max(np.linalg.norm(a-b,axis=1))),
          'old_rank':next(r['dimension'] for r in frozen['rotation_geometry_01'][L]['predictions'] if r['name']=='rotation_profiles16'),
          'new_rank':next(r['dimension'] for r in frozen['rotation_geometry_precision_01'][L]['predictions'] if r['name']=='rotation_profiles16'),
          'interpretation':'More retained directions change the approximation space; this is not a time-integrator error estimate or a proof of convergence.'})
    log=(root/'rotation_geometry_tests_01.log').read_text();match=re.search(r'Ran (\d+) tests in ([\d.]+)s',log);assert match and log.rstrip().endswith('OK');(out/'tests.log').write_text(log)
    primary={name:{str(L):[{'name':r['name'],'dimension':r['dimension'],**r['rows'][40]} for r in case['models']] for L,case in cases.items()} for name,cases in geometry.items()}
    audit={'classification':'Eleventh-round retrospective structure and strong-baseline audit. No new-parameter physical forecasts or canonical-only discovery.',
      'Gaussian_mode_cases':5,'onsite_mode_starts':320,'general_Gaussian_mode_cases':8,'general_Gaussian_local_starts':32,'GPUs_used':8,
      'mode_geometry':modes,'general_mode_results':general,
      'max_general_over_onsite_score_gain':max(r['general_score_normalized']-r['onsite_score_normalized'] for r in general),
      'geometry_model_and_refinement_records':sum(len(x['models']) for cases in geometry.values() for x in cases.values()),
      'mechanism_records':sum(len(x['models']) for x in mechanism.values()),'equal_rank_records':8,
      'physical_conditions':2,'primary_time':2.,'geometry_primary':primary,
      'equal_rank_primary':{str(L):[{'name':r['name'],'dimension':r['dimension'],**r['rows'][40]} for r in case['models']] for L,case in rank.items()},
      'mechanism_controls':mechanism,'rank_cutoff_checks':refinements,
      'stored_finite_dictionary_cost_certificates':certificates,
      'verified_files':verified,'verified_file_count':len(verified),
      'max_initial_state_difference':max(r['initial_error'] for cases in frozen.values() for case in cases.values() for r in case['predictions']),
      'max_norm_error':max(r['max_norm_error'] for cases in frozen.values() for case in cases.values() for r in case['predictions']),
      'max_exact_symmetry_projection_error':max(x['max_exact_symmetry_projection_error'] for cases in geometry.values() for x in cases.values()),
      'tests':{'passed':int(match[1]),'seconds':float(match[2]),'log':'tests.log'},
      'failed_serialization_attempts_retained':1,'actual_canonical_measure_draws':False,
      'rotation_geometries_are_canonical_selected':False,'same_dimension_advantage_of_independent_fields_established':False,
      'canonical_unique_physical_prediction_established':False,'new_important_physics_established':False,
      'full_Gaussian_extent_bounds_from_rotation_costs':False,'ideal_Gaussian_atom_generation_interval_certified':False,
      'remaining_gate':'Find a robust, physically important higher-transfer coherent structure beyond the conventional nonlinear-field and block-Hamiltonian-response controls; select it from actual canonical information before genuinely new physical conditions and test identical-initial-state deletion.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    colors={'matched':'#0072b2','shifted':'#cc79a7','independent':'#009e73','Hamiltonian_block_response':'#d55e00'}
    labels={'matched':'Matched fields','shifted':'Relative angle pi/4','independent':'Independent fields','Hamiltonian_block_response':'Hamiltonian response'}
    for col,L in enumerate((14,18)):
        for r in rank[L]['models']:
            t=[x['time'] for x in r['rows']]
            axes[0,col].plot(t,[x['count_TV'] for x in r['rows']],color=colors[r['name']],label=labels[r['name']])
            axes[1,col].plot(t,[1-x['fidelity'] for x in r['rows']],color=colors[r['name']])
        axes[0,col].set_title(f'L={L}, equal dimension {rank[L]["frozen"]["dimension"]}')
        for ax in axes[:,col]:ax.set_xlabel('time');ax.axvline(2,color='gray',ls=':',lw=.7);ax.grid(alpha=.2)
    axes[0,0].set_ylabel('count total variation');axes[1,0].set_ylabel('state infidelity');axes[0,0].legend(fontsize=8)
    fig.suptitle('Same exact Fermi sea; known-condition mechanism controls\nIndependent fields do not improve the equal-rank comparison',fontsize=11);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'equal_rank_controls.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,L in zip(axes,(14,18)):
        s=np.array(frozen['rotation_geometry_01'][L]['singular_values']['rotation_profiles16']);s=s[s>s[0]*1e-15]
        ax.semilogy(np.arange(1,len(s)+1),s/s[0],color='#0072b2');ax.axhline(1e-9,color='#d55e00',ls='--',label='Original cutoff');ax.axhline(1e-10,color='#009e73',ls=':',label='More retained directions')
        ax.set_xlabel('singular-value rank');ax.set_ylabel('relative singular value');ax.set_title(f'L={L}');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Matched rotation configurations: strong near-linear dependence\nColumn count and collective-coordinate count do not certify low synthesis cost',fontsize=11);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'rotation_conditioning.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.6));x=np.arange(4)
    for j,L in enumerate((14,18)):
        values=[r['coefficient_l1_lower']['float'] for r in certificates[L]['certificates']]
        ax.bar(x+(j-.5)*.35,values,width=.35,label=f'L={L}',color=('#0072b2','#d55e00')[j])
    ax.set_yscale('log');ax.set_xticks(x,['0','.01','.05','.10']);ax.set_xlabel('allowed state-vector norm error');ax.set_ylabel('lower bound on coefficient l1 mass');ax.legend();ax.grid(axis='y',alpha=.2)
    ax.set_title('Saved finite rotation-column model, source space treated as free\nExact dyadic matrix bounds; not full-Gaussian extent bounds',fontsize=10);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'finite_dictionary_cost.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps({k:audit[k] for k in ('verified_file_count','geometry_model_and_refinement_records','mechanism_records','equal_rank_records','max_initial_state_difference','max_norm_error','tests')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();main(args.output)
