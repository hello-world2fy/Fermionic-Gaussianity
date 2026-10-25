#!/usr/bin/env python3
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,hashlib,re,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.coherent_response import complement


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out):
    out.mkdir(parents=True,exist_ok=False);root=out.parent;p=root/'coherent_response_01'
    protocol=json.loads((p/'protocol.json').read_text());evaluated=json.loads((p/'evaluation.json').read_text())
    geometry=json.loads((p/'geometry.json').read_text());verified=[]
    def verify(path,expected):
        assert digest(path)==expected,str(path);verified.append({'path':str(path),'sha256':expected})
    for path,expected in protocol['input_hashes'].items():verify(Path(path),expected)
    verify(p/'training.npz',protocol['training_sha256']);verify(p/'frozen_spaces.npz',protocol['frozen_sha256'])
    for record in json.loads((p/'kernel_jobs.json').read_text()):verify(p/record['file'],record['sha256'])
    forecasts=json.loads((p/'frozen_predictions.json').read_text())
    for record in forecasts:verify(p/record['file'],record['sha256'])
    primary={r['name']:next(x for x in r['rows'] if x['time']==2.) for r in evaluated['models'] if r['start']=='restart'}
    return_data={}
    for L in (14,18):
        d=json.loads((root/'return_pair_mode_01'/f'L{L}_evaluation.json').read_text())
        verify(root/'return_pair_mode_01'/f'L{L}_frozen.npz',d['checks']['frozen_sha256'])
        return_data[str(L)]=d
    formulas=json.loads((root/'return_pair_mode_01/formula_checks.json').read_text())
    paths=json.loads((root/'dark_paths_01/summary.json').read_text())
    tests=(root/'coherent_response_tests_01.log').read_text()
    match=re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK',tests);assert match
    shutil.copyfile(root/'coherent_response_tests_01.log',out/'tests.log')
    arrays=np.load(p/'frozen_spaces.npz');base=arrays['shared'];Z=complement(base)
    phi=np.load(root/'return_pair_mode_01/L14_frozen.npz')['mode'];v=phi-base@(base.conj().T@phi);v/=np.linalg.norm(v)
    values,V=np.linalg.eigh(Z.conj().T@arrays['canonical_A']@Z);modes=Z@V
    overlap=abs(modes.conj().T@v)**2
    spectrum={'residual_eigenvalues_ascending':values.tolist(),'return_mode_weight':overlap.tolist(),
        'top16_return_mode_fraction':float(sum(overlap[-16:])),
        'bottom32_return_mode_fraction':float(sum(overlap[:32])),
        'interpretation':'Posthoc mechanism diagnostic. No selection or holdout claim follows from this comparison.'}
    audit={'classification':'Thirteenth goal round: progress; the original physical discovery and prediction goal remains active.',
        'full_gaussian_kernel_coordinates':170,'physical_fermion_modes':28,'GPUs_used':8,
        'kernel_derivative_calls':680,'kernel_monte_carlo_samples':0,
        'actual_full_gaussian_coherent_integrals':True,'positive_canonical_measure_samples':False,
        'frozen_spaces':len(protocol['dimensions']),'same_initial_trajectories':len(forecasts),
        'passing_canonical_primary_selections':[r['method'] for r in evaluated['gate'] if r['passes_primary_all_controls']],
        'new_physical_holdout_predictions':0,'new_important_physics_established':False,
        'canonical_unique_predictive_structure_established':False,'novelty_priority_established':False,
        'general_size_formula_proved':False,'long_time_L62_simulation':False,
        'max_initial_error':max(r['initial_error'] for r in forecasts),
        'max_norm_error':max(r['norm_error'] for r in forecasts),
        'max_kernel_Hermitian_error':max(r['Hermitian_error'] for r in geometry['kernels'].values()),
        'max_kernel_symmetric_error':max(r['symmetric_error'] for r in geometry['kernels'].values()),
        'full_gaussian_reconstruction_error':geometry['kernels']['canonical']['self_reconstruction_error'],
        'tests_passed':int(match[1]),'tests_seconds':float(match[2]),'verified_file_count':len(verified),
        'verified_files':verified,'primary_time2_restart':primary,'return_mode':return_data,
        'finite_formula_checks':formulas,'short_time_sizes':paths,'response_mode_content':spectrum,
        'remaining_gate':'Explain a physically significant deviation with an actual distribution-selected coherent structure beyond conventional nonlinear clusters, adaptive Gaussian and Hamiltonian response controls; then freeze genuinely new physical predictions and same-initial-state deletion.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    ranks=[1,2,4,8,16];colors=['#0072B2','#D55E00','#009E73']
    for ax,metric,title in zip(axes,['count_TV','current_L1_error','multiple_dark_error'],['Count TV','Sector-current L1 error','Absolute dark-probability error']):
        for kind,color in zip(['eigen','anomalous','cross'],colors):
            for method,style in [('canonical','-'),('unoptimized','--')]:
                y=[abs(primary[f'{method}_{kind}_q{q}'][metric]) for q in ranks]
                ax.plot(ranks,y,style,marker='o',ms=3,color=color,label=f'{method}: {kind}')
        ax.plot(ranks,[abs(primary[f'Hamiltonian_Krylov_q{q}'][metric]) for q in ranks],'k-s',ms=3,label='Hamiltonian Krylov')
        ax.set(xlabel='Added directions (common dimension 22)',ylabel=title,yscale='log',xticks=ranks)
    axes[0].legend(fontsize=7);fig.suptitle('L=14, same exact t=1 initial state, prediction at t=2')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'response_predictions.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.7))
    for L,color in [(14,'#0072B2'),(18,'#D55E00')]:
        rows=return_data[str(L)]['rows'];time=[r['time'] for r in rows]
        axes[0].plot(time,[np.nan if r['fraction_m2_dark'] is None else r['fraction_m2_dark'] for r in rows],color=color,label=f'L={L}, fraction of m=2 dark')
        axes[0].plot(time,[np.nan if r['fraction_all_multiple_dark'] is None else r['fraction_all_multiple_dark'] for r in rows],'--',color=color,label=f'L={L}, fraction of all m>=2 dark')
        axes[1].plot(time,[r['deletion_count_TV'] for r in rows],color=color,label=f'L={L}, delete one return mode')
        previous=json.loads((root/'dark_pair_controls_01'/f'L{L}_evaluation.json').read_text())
        row=next(x for x in previous['models'] if x['name']=='delete_either_multiple')
        axes[1].plot(time,[r['count_TV'] for r in row['rows']],'--',color=color,label=f'L={L}, delete all m>=2 dark')
    axes[0].set(xlabel='Time',ylabel='Fraction of exact dark probability',ylim=(0,1.03));axes[1].set(xlabel='Time',ylabel='Count TV after same-Fermi-state deletion')
    for ax in axes:ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'return_mode_dynamics.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    r=np.array([x['r'] for x in paths]);norm=np.array([x['leading_coefficient_norm'] for x in paths])
    predicted=np.sqrt(4.**10*(r-3)**2*(r-4)/(1800*r**7))
    axes[0].loglog(r,norm,'o',label='Independent Taylor coefficient')
    axes[0].loglog(r,predicted,'-',label='Inferred finite-size formula')
    axes[0].set(xlabel='Levels per band r',ylabel='Norm of fifth-order amplitude coefficient');axes[0].legend(fontsize=8)
    axes[1].plot(np.arange(1,len(overlap)+1),np.cumsum(overlap[::-1]))
    axes[1].set(xlabel='Number of largest A eigenvectors retained',ylabel='Captured return-mode squared overlap',ylim=(0,1.03))
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'return_mode_scaling_and_geometry.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps({k:audit[k] for k in ['frozen_spaces','same_initial_trajectories','passing_canonical_primary_selections','max_initial_error','max_norm_error','tests_passed','verified_file_count']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();main(args.output)
