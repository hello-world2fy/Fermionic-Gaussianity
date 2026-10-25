#!/usr/bin/env python3
"""Archive seventh-round predictions, controls, theory checks and limitations."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
from pathlib import Path
import argparse
import json
import hashlib
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def primary(record):
    return next(r for r in record['rows'] if r['elapsed']==2.)


def save(fig, path):
    for suffix in ('png','pdf'):
        fig.savefig(path.with_suffix('.'+suffix),bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    results=root/'results'
    pilot=read(results/'coherent_return_pilot_01/evaluation.json')
    holdout=read(results/'coherent_return_holdouts_01/evaluation.json')
    controls=read(results/'coherent_return_pilot_01/phase_controls.json')
    validations=read(results/'coherent_return_validation_01/validation.json')
    screen=read(results/'pairing_channels_screen_01/summary.json')
    geometry=read(results/'pairing_channels_geometry_01/evaluation.json')
    pauli=read(results/'pauli_channels_check_01/evaluation.json')
    closure=read(results/'pairing_channels_closure_01/evaluation.json')
    checked=[]
    for record in read(results/'coherent_return_pilot_01/manifest.json')['records']:
        for field in ('frozen','prediction'):
            path=results/'coherent_return_pilot_01'/record[field+'_file']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==record[field+'_sha256']
            checked.append(str(path.relative_to(root)))
    for record in read(results/'coherent_return_holdouts_01/predictions_saved.json'):
        path=results/'coherent_return_holdouts_01'/record['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
        checked.append(str(path.relative_to(root)))
    aggregates=[]
    for L in (12,16):
        for q in (2,3):
            for source in ('direct','rpa','witness'):
                rows=[primary(x) for x in holdout if x['model']['L']==L and x['model']['q']==q and x['model']['source']==source]
                if not rows:
                    continue
                aggregates.append({'L':L,'q':q,'source':source,'conditions':len(rows),
                    'echo_mean_absolute_error':float(np.mean([abs(x['echo_error']) for x in rows])),
                    'count_TV_mean':float(np.mean([x['count_TV'] for x in rows])),
                    'fidelity_mean':float(np.mean([x['fidelity'] for x in rows])),
                    'dephased_echo_mean_absolute_error':float(np.mean([abs(x['dephased_echo_error']) for x in rows])),
                    'dephased_echo_sign_matches':sum((x['predicted']['dephased_echo_excess']>0)==(x['exact']['dephased_echo_excess']>0) for x in rows),
                    'initial_echo_mean_absolute_error':float(np.mean([abs(x['initial_only_echo_error']) for x in rows])),
                    'dynamics_echo_mean_absolute_error':float(np.mean([abs(x['dynamics_only_echo_error']) for x in rows]))})
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'tests'),'-v'],capture_output=True,text=True)
    log=test.stdout+test.stderr
    (args.output/'tests.log').write_text(log)
    assert test.returncode==0,log
    summary=re.search(r'Ran \d+ tests in [^\n]+',log).group()
    report={'scope':'Uniform-BCS coherent return is predicted by conventional RPA-generated channels; no canonical-specific gain. Competing-channel screening identifies an exact Pauli cubic source and rejects a false chirality diagnostic, but has a Hamiltonian-derived symmetry alternative and has not passed the full strong-baseline prediction gate.',
            'pilot_model_predictions':len(pilot),'new_holdout_model_predictions':len(holdout),
            'new_holdout_physical_conditions':len({r['case']['name'] for r in holdout}),
            'exact_own_initial_state_controls':len(pilot)+len(holdout),
            'same_initial_state_controls':validations,'verified_files':checked,'verified_file_count':len(checked),
            'holdout_aggregate':aggregates,
            'max_prediction_norm_error':max(r['predicted']['norm_error_before_normalization'] for x in pilot+holdout for r in x['rows']),
            'channel_screen_cases':len(screen),'channel_screen':screen,
            'geometry_counterexample':geometry['counterexample'],
            'Pauli_short_time_checks':pauli['checks'],
            'max_Pauli_identity_error':max(max(x['identity_errors']) for x in pauli['checks']),
            'max_Riccati_dark_norm':max(x['max_Riccati_dark_norm'] for x in pauli['checks']),
            'Hamiltonian_symmetry_alternative':closure,
            'tests':{'returncode':test.returncode,'summary':summary},
            'goal_complete':False,'new_important_physics_established':False,
            'canonical_unique_mechanism_established':False,'new_chiral_phase_or_cat_established':False,
            'new_kernel_strong_nonlinear_baselines_complete':False}
    (args.output/'audit.json').write_text(json.dumps(report,indent=2))
    plt.rcParams.update({'font.size':10,'figure.dpi':150})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,L in zip(axes[0],(12,16)):
        ref=next(x for x in pilot if x['model']['L']==L)
        t=[r['elapsed'] for r in ref['rows']]
        ax.plot(t,[r['exact']['echo_excess'] for r in ref['rows']],color='black',lw=2,label='Exact')
        for source,q,style in [('rpa',2,':'),('rpa',3,'--'),('direct',3,'-.')]+([('witness',3,'-')] if L==12 else []):
            data=next(x for x in pilot if x['model']['L']==L and x['model']['source']==source and x['model']['q']==q)
            ax.plot(t,[r['predicted']['echo_excess'] for r in data['rows']],ls=style,label=f'{source}, q={q}')
        ax.axhline(0,color='gray',lw=.8)
        ax.set(title=f'L={L}, original G=4: retrospective pilot',xlabel='Time after training',ylabel='Finite-angle echo excess')
        ax.legend(fontsize=8)
    selected=sorted([x for x in holdout if x['model']['source']=='rpa' and x['model']['q']==3],key=lambda x:x['case']['name'])
    ax=axes[1,0]
    x=np.arange(len(selected))
    ax.plot(x,[primary(r)['exact']['dephased_echo_excess'] for r in selected],'kx',ms=8,label='Exact')
    ax.plot(x,[primary(r)['predicted']['dephased_echo_excess'] for r in selected],'o',mfc='none',label='RPA q=3')
    ax.axhline(0,color='gray',lw=.8)
    ax.set(xticks=x,xticklabels=[str(i+1) for i in x],xlabel='Eight new physical conditions (L12, then L16)',
           ylabel='Echo excess after 25% count dephasing',title='New spectrum / coupling forecasts, elapsed time 2')
    ax.legend(fontsize=8)
    ax=axes[1,1]
    rows=[r for r in validations['records'] if r['L']==16]
    for row in rows:
        ax.plot([2,3],[row['metrics'][key]['dephased_echo_excess'] for key in ('two_channels','three_channels_same_initial')],'o-',label=row['source'])
    ax.axhline(rows[0]['exact_dephased_echo_excess'],color='black',ls='--',label='Exact')
    ax.axhline(0,color='gray',lw=.8)
    ax.set(xticks=[2,3],xlabel='Retained orbital directions per band',ylabel='Dephased echo excess',
           title='Identical initial state: L16, G=3.6, cubic spectrum')
    ax.legend(fontsize=8)
    fig.suptitle('The late coherent return does not establish a canonical-specific advantage',fontsize=14)
    save(fig,args.output/'coherent_return_controls')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for record in pauli['late']:
        t=[r['time'] for r in record['rows']]
        axes[0,0].plot(t,[r['dark_fraction_given_m1'] for r in record['rows']],label=f'L={record["L"]}')
        axes[0,1].plot(t,[max(r['dark_probability'],1e-14) for r in record['rows']],label=f'L={record["L"]}')
    axes[0,0].axhline(0,color='gray',ls='--',label='Riccati prediction')
    axes[0,0].set(xlabel='Time',ylabel='Fraction within one-transfer sector',title='Pauli-generated angular channels outside the interaction')
    axes[0,0].legend()
    axes[0,1].set(xlabel='Time',ylabel='Probability in the full state',yscale='log',ylim=(1e-9,.2),title='Conditional fractions retain their success probability')
    axes[0,1].legend()
    for ratio in (0.,1.):
        data=read(results/f'pairing_channels_screen_01/L12_ratio{ratio:g}.json')
        axes[1,0].plot([r['time'] for r in data['rows']],[r['normalized_phase_locking'] if r['normalized_phase_locking'] is not None else np.nan for r in data['rows']],label=f'eta={ratio:g}')
    axes[1,0].set(xlabel='Time',ylabel='Normalized fourth pair moment',title='Negative values also arise with only one interaction channel')
    axes[1,0].legend()
    counter=geometry['counterexample']
    axes[1,1].bar(['Real orbital','Chiral +','Chiral -'],[counter['geometry']['best_real_joint_conditional_overlap']]+counter['geometry']['chiral_joint_conditional_overlaps'])
    axes[1,1].set(ylabel='Squared overlap',ylim=(0,1.1),title='Explicit single-real-orbital counterexample: phase moment = -1')
    fig.suptitle('Competing-channel screen: a concrete Pauli mechanism, with no chirality claim',fontsize=14)
    save(fig,args.output/'pauli_channel_diagnostics')
    print(json.dumps({'tests':summary,'verified_files':len(checked),'holdout_models':len(holdout),'physical_conditions':8}),flush=True)
