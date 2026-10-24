#!/usr/bin/env python3
"""Compare full canonical P populations, keeping each independent run separate."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import t as student_t
from fk4.io import write_json,source_manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--endpoint',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    groups={};seeds=[];histograms={}
    for run in args.runs+([args.endpoint] if args.endpoint else []):
        s=json.loads((run/'summary.json').read_text())
        assert s['status']=='complete' and json.loads((run/'verification.json').read_text())['artifact_verification_passed']
        endpoint=run==args.endpoint
        for r in s['rows']:
            t,k=r['t'],r['k']
            if endpoint and (t,k) in groups:continue
            if (t,k) in groups:raise ValueError('duplicate point')
            observations=r.get('observations',r.get('validation_rows'))
            values=[];hist=np.zeros((32,32))
            for o in observations:
                if endpoint:
                    file=run/(o['key']+'.npz');featurekey='features';expected=o['full_distribution']['feature_means']
                elif 'observations' in r:
                    file=run/f"t{t:g}_k{k}_seed{o['seed']}.npz";featurekey='features';expected=o['P']['feature_means']
                else:
                    file=Path(o['P_source_file']) if 'P_source_file' in o else run/f"t{t:g}_k{k}_validation_seed{o['seed']}.npz"
                    featurekey='P_all_features';expected=o['P_full_distribution']['feature_means']
                a=np.load(file);features=a[featurekey]
                np.testing.assert_allclose(features.mean(axis=0),expected,atol=3e-11)
                assert features.shape[1]==4 and len(features)>0
                v=np.r_[features.mean(axis=0),np.mean(features[:,3]>=.2),np.mean(features[:,0]>=.8)]
                values.append(v)
                h,edges,_=np.histogram2d(features[:,0],features[:,3],bins=(np.linspace(0,1,33),)*2)
                hist+=h/h.sum()/len(observations)
                seeds.append({'t':t,'k':k,'seed':o['seed'],'count':len(features),
                    'source_file':str(file.resolve()),'feature_array_sha256':hashlib.sha256(features.tobytes()).hexdigest(),
                    'statistics':v.tolist()})
            v=np.array(values);assert len(v)>=5
            half=student_t.ppf(.975,len(v)-1)*v.std(axis=0,ddof=1)/np.sqrt(len(v))
            groups[t,k]={'t':t,'k':k,'independent_repeats':len(v),'mean':v.mean(axis=0).tolist(),
                'repeat_95_halfwidth':half.tolist(),'source_run':str(run.resolve()),
                'reconstruction_and_mass_gate_pass':r.get('all_seeds_reconstruction_pass',True) and r['mass_repeat_pass']}
            histograms[f't{t:g}_k{k}']=hist
    names=['mass_a_squared','mass_b_squared','mass_dot','eta','fraction_eta_ge_0.2','fraction_mass_a_squared_ge_0.8']
    rows=[groups[key] for key in sorted(groups)]
    write_json(args.output/'summary.json',{'status':'complete','rows':rows,'seed_observations':seeds,
        'statistic_names':names,'uncertainty':'Empirical Student-t repeat intervals across independent random streams; particles within each SMC run are not independent.',
        'interpretation':'Finite-order canonical coefficient distribution only; does not identify unique atoms or prove nonzero-t global contact recovery.'})
    np.savez_compressed(args.output/'mass_eta_histograms.npz',**histograms,bin_edges=np.linspace(0,1,33))
    flattened=[{'t':r['t'],'k':r['k'],**{name:r['mean'][i] for i,name in enumerate(names)},
                **{name+'_95_halfwidth':r['repeat_95_halfwidth'][i] for i,name in enumerate(names)}} for r in rows]
    with (args.output/'composition.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flattened[0]));writer.writeheader();writer.writerows(flattened)
    os.environ['MPLCONFIGDIR']=str(args.output/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ts=sorted({r['t'] for r in rows})
    fig,axes=plt.subplots(3,3,figsize=(13,10),sharex=True,sharey=True)
    for ax,t in zip(axes.flat,ts):
        rs=[r for r in rows if r['t']==t]
        for index,label in [(0,'E |m_A|²'),(2,'E m_A·m_B'),(3,'E eta')]:
            ax.errorbar([r['k'] for r in rs],[r['mean'][index] for r in rs],yerr=[r['repeat_95_halfwidth'][index] for r in rs],fmt='o-',capsize=2,label=label)
        ax.set(xscale='log',title=f't={t:g}',xlabel='Replica order k',ylabel='Canonical P mean')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=9)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(args.output/f'canonical_composition.{ext}',dpi=180)
    plt.close(fig)
    selected=[t for t in [0,9,11,20] if t in ts]
    fig,axes=plt.subplots(2,len(selected),figsize=(3.5*len(selected),6),squeeze=False)
    for col,t in enumerate(selected):
        ks=sorted(r['k'] for r in rows if r['t']==t)
        for row,k in enumerate([min((v for v in ks if v>=2),default=ks[0]),ks[-1]]):
            ax=axes[row,col];h=histograms[f't{t:g}_k{k}']
            from matplotlib.colors import LogNorm
            im=ax.imshow(h.T,origin='lower',extent=[0,1,0,1],aspect='auto',norm=LogNorm(vmin=1e-6,vmax=1),cmap='viridis')
            ax.set(title=f't={t:g}, k={k}',xlabel='|m_A|²',ylabel='eta')
    fig.subplots_adjust(wspace=.35,hspace=.45,right=.90)
    cb=fig.add_axes([.92,.12,.015,.76]);fig.colorbar(im,cax=cb,label='Coefficient probability per bin')
    for ext in ['png','pdf']:fig.savefig(args.output/f'mass_eta_low_high.{ext}',dpi=180,bbox_inches='tight')
    lines=['# Canonical Gaussian组成','',
        '所有点均从独立核验过的完整P粒子数组重新计算；保留每个种子的统计量和特征数组哈希。误差条来自至少五条独立随机流的经验Student-t半宽，未把同一SMC内的粒子当作独立样本。',
        '', '![组成均值](canonical_composition.png)','', '![低阶与最高已测阶数](mass_eta_low_high.png)','',
        '这些是有限阶canonical系数分布的组成，不能直接解释为唯一最优构型或所有最优表示的必要原子。t=0的全字典必要性另外有解析证明；非零t的全局接触集合仍未完成尖锐认证。',
        '', '| t | 最低/最高k | E eta（低阶→高阶） | E massA²（低阶→高阶） |',
        '|---:|---|---|---|']
    for t in ts:
        rs=[r for r in rows if r['t']==t];a,b=rs[0],rs[-1]
        lines.append(f"| {t:g} | {a['k']} / {b['k']} | {a['mean'][3]:.6g} → {b['mean'][3]:.6g} | {a['mean'][0]:.6g} → {b['mean'][0]:.6g} |")
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('composition points',len(rows))


if __name__=='__main__':main()
