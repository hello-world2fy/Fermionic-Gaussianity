#!/usr/bin/env python3
"""Audit nonlinear controls and compare only jointly defined observables."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from run_nonlinear_channels import CASES,TIMES,name,state_rows


def verify(folder):
    obj=json.loads((folder/'frozen_forecasts.json').read_text())
    files=obj.get('files',obj)
    for filename,digest in files.items():
        assert hashlib.sha256((folder/filename).read_bytes()).hexdigest()==digest,filename
    return len(files)


def stochastic_rows(folder,tag,kind):
    blocks=np.concatenate([np.load(folder/f'{tag}_{kind}_seed{seed}_dt0.004.npz')['blocks'] for seed in (2026,2027)],axis=1)
    avg=blocks.mean(axis=1);se=blocks.std(axis=1,ddof=1)/np.sqrt(blocks.shape[1])
    base=np.load(folder/f'{tag}_{kind}_seed2026_dt0.004.npz')['blocks']
    fine=np.load(folder/f'{tag}_{kind}_seed2026_dt0.002.npz')['blocks']
    rows=[]
    for j,t in enumerate(TIMES):
        rows.append({'time':float(t),'mean':float(avg[j,0]),'mean_se':float(se[j,0]),
                     'variance':float(avg[j,1]-avg[j,0]**2),
                     'variance_se':float(np.std(blocks[j,:,1]-2*avg[j,0]*blocks[j,:,0],ddof=1)/np.sqrt(blocks.shape[1])),
                     'upper_source_occupation':float(avg[j,2]),'upper_source_occupation_se':float(se[j,2]),
                     'upper_third_harmonic_occupation':float(avg[j,3]),'upper_third_harmonic_occupation_se':float(se[j,3]),
                     'classical_variance':float(avg[j,4]-avg[j,0]**2)})
    return rows,{'max_paired_block_step_difference':float(np.max(abs(fine-base))),
                 'max_mean_step_difference':float(np.max(abs((fine-base).mean(axis=1)))),
                 'independent_samples':int(2*262144),'error_blocks':int(blocks.shape[1])}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--refresh',action='store_true',help='Regenerate derived audit/plots; never change forecasts')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=args.refresh)
    root=Path(__file__).resolve().parents[1]/'results'
    base=root/'nonlinear_channels_01';extra=root/'projective_channels_01';fine18=root/'L18_nonlinear_refinement_01'
    verified=sum(verify(f) for f in (base,extra,fine18))
    results=[];refinements=[];step=[];spectra=[]
    for L,eta in CASES:
        tag=name(L,eta)
        report=json.loads((base/f'{tag}_evaluation.json').read_text())
        model,forms=angular_model(L,4.,eta)
        p=np.load(extra/f'{tag}_projective.npz')
        assert len(p['times'])==len(TIMES)
        rows=state_rows(model,p['states'],forms)
        exact_states=np.load(base/f'{tag}_exact.npz')['states']
        V=np.exp(2j*np.pi*np.arange(model.r)[:,None]*np.arange(model.r)[None,:]/model.r)/np.sqrt(model.r)
        third=np.array(sorted({3%model.r,(-3)%model.r}))
        spectrum=[]
        for t,state in zip(TIMES,exact_states):
            C=V.conj().T@state[model.blocks[1][1]]@V
            spectrum.append({'time':float(t),'diagonal_probabilities':(abs(C.diagonal())**2).tolist(),
                             'offdiagonal_probability':float(np.linalg.norm(C-np.diag(C.diagonal()))**2),
                             'joint_third_harmonic_probability':float(np.linalg.norm(C[np.ix_(third,third)])**2)})
        spectra.append({'condition':tag,'L':L,'eta':eta,'rows':spectrum})
        for row,true,a,b in zip(rows,report['exact'],p['states'],exact_states):
            row['fidelity']=float(abs(np.vdot(a,b))**2)
            row['count_TV']=float(.5*np.sum(abs(np.array(row['count'])-true['count'])))
        report['predictions']['projective_cc']=rows
        report['stochastic']={}
        for kind in ('gaussian','discrete'):
            s,diagnostic=stochastic_rows(base,tag,kind)
            report['stochastic'][kind]=s
            step.append({'condition':tag,'method':kind,**diagnostic})
        for method in ('full_cc','source_only_cc'):
            path=fine18/(method+'.npz') if L==18 else extra/f'{tag}_{method}_refined.npz'
            original=np.load(base/f'{tag}_{method}.npz')['states']
            refined=np.load(path)['states']
            errors=[]
            for a,b in zip(original,refined):
                a=a/np.linalg.norm(a);b=b/np.linalg.norm(b)
                phase=np.vdot(b,a);phase/=abs(phase)
                errors.append(float(np.linalg.norm(a-phase*b)))
            refinements.append({'condition':tag,'method':method,'max_normalized_ray_difference':max(errors)})
        results.append(report)
    (args.output/'evaluation.json').write_text(json.dumps(results,indent=2))
    (args.output/'fourier_diagnostics.json').write_text(json.dumps(spectra,indent=2))
    short=[]
    for p in extra.glob('L*.json'):
        record=json.loads(p.read_text())
        if 'short_time' in record:
            short.append({'L':record['L'],'rows':record['short_time']})
    controls=[]
    for result in results:
        j=40
        values={k:result['exact'][j][k] for k in ('mean','variance','m1_dark_probability','m1_dark_fraction','upper_third_harmonic_occupation','echo_witness_theta02')}
        control={'condition':name(result['L'],result['eta']),'exact':values,'deterministic':{},'stochastic':{}}
        for method,rows in result['predictions'].items():
            row=rows[j]
            control['deterministic'][method]={k:row[k] for k in ('fidelity','count_TV','m1_dark_probability','m1_dark_fraction','upper_third_harmonic_occupation','mean','variance')}
        for method,rows in result['stochastic'].items():
            row=rows[j];control['stochastic'][method]=row
        controls.append(control)
    audit={'verified_forecast_files':verified,'physical_conditions':len(results),
           'gpu_runs':24,'GPUs_used':8,'samples_per_run':262144,'independent_samples_per_condition_and_method':524288,
           'deterministic_methods':4,'same_initial_state':'Exact Fermi sea for all deterministic predictions.',
           'numerical_refinements':refinements,'stochastic_step_checks':step,'short_time_CC_source_checks':short,
           'primary_time':2.,'primary_results':controls,
           'new_important_canonical_specific_physics_established':False,
           'scope':'Odd-grid robustness and strong nonlinear controls. The conventional projective CC equation contains the Pauli cubic source explicitly. This is not evidence of a canonical-only discovery.',
           'stochastic_scope':'Gaussian-noise spin mean field and discrete TWA with exact onsite spin symbols. No pure-state echo or coherent postselection is assigned to these ensembles.',
           'floating_point_not_interval_certified':True}
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2))
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(2,3,figsize=(15,8.7))
    case=next(x for x in results if x['L']==14 and x['eta']==1)
    colors={'full_cc':'#d55e00','source_only_cc':'#009e73','riccati':'#56b4e9','projective_cc':'#cc79a7'}
    labels={'full_cc':'Variational pair-CC','source_only_cc':'Source-only pair-CC','riccati':'Riccati lift','projective_cc':'Projective pair-CC ket'}
    for ax,key,title in zip(axes[0],('m1_dark_fraction','upper_third_harmonic_occupation','variance'),
                            ('L=14: conditional dark fraction','L=14: third angular pair occupation','L=14: transferred-pair variance')):
        ax.plot(TIMES,[r[key] for r in case['exact']],color='black',lw=2,label='Exact')
        for method,rows in case['predictions'].items():
            y=[np.nan if key=='m1_dark_fraction' and r['m1_probability']<=1e-14 else r[key] for r in rows]
            ax.plot(TIMES,y,color=colors[method],label=labels[method],ls='--' if method=='source_only_cc' else '-')
        if key!='m1_dark_fraction':
            for kind,c in [('gaussian','#e69f00'),('discrete','#0072b2')]:
                y=np.array([r[key] for r in case['stochastic'][kind]])
                se=np.array([r[key+'_se'] for r in case['stochastic'][kind]])
                ax.plot(TIMES,y,color=c,ls=':',label='Gaussian SMF' if kind=='gaussian' else 'DTWA')
                ax.fill_between(TIMES,y-2*se,y+2*se,color=c,alpha=.17)
        ax.set_title(title);ax.set_xlabel('time');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    axes[0,1].legend(fontsize=7)
    ax=axes[1,0]
    for rec in short:
        ax.loglog([r['time'] for r in rec['rows']],
                  [abs(r['DF_probability_over_leading']-1) for r in rec['rows']],'-o',label=f"L={rec['L']} variational CC")
        ax.loglog([r['time'] for r in rec['rows']],
                  [abs(r['projective_T_probability_over_leading']-1) for r in rec['rows']],'--s',label=f"L={rec['L']} projective CC")
    ax.set_title('Conventional CC recovers the cubic source');ax.set_xlabel('time');ax.set_ylabel('relative correction to leading t^6 probability');ax.legend(fontsize=7);ax.grid(alpha=.2)
    ax=axes[1,1]
    sizes=[10,14,18]
    equal=[next(x for x in results if x['L']==L and x['eta']==1) for L in sizes]
    ax.plot(sizes,[x['exact'][40]['m1_dark_probability'] for x in equal],'-ok',label='Exact')
    for method in colors:
        ax.plot(sizes,[x['predictions'][method][40]['m1_dark_probability'] for x in equal],'-o',color=colors[method],label=labels[method])
    ax.set_title('t=2: dark-channel probability, odd grids');ax.set_xlabel('pair levels L');ax.grid(alpha=.2)
    ax=axes[1,2];x=np.arange(len(results))
    for k,method in enumerate(colors):
        ax.bar(x+(k-1.5)*.2,[r['predictions'][method][40]['count_TV'] for r in results],width=.2,color=colors[method],label=labels[method])
    ax.set_xticks(x,[name(r['L'],r['eta']) for r in results],rotation=15,fontsize=8)
    ax.set_title('t=2: full counting distribution error');ax.set_ylabel('total variation distance');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Nonlinear pairing controls: physical channel generation is not canonical-specific',fontsize=13)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(args.output/f'nonlinear_channel_controls.{ext}',dpi=170)
    print(json.dumps({'verified_files':verified,'maximum_ray_refinement_difference':max(x['max_normalized_ray_difference'] for x in refinements),
                      'primary':controls},indent=2))
