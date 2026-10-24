#!/usr/bin/env python3
"""Check saved samples and report independent-run uncertainty and fit sensitivity."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import t as student_t
from mpmath import mp
from fk5.io import write_json
from fk5.paths import ROOT,PREVIOUS
from fk5.precision_geometry import stationary_data
from fk4.geometry import contact_hessian
from fk3.discovery import covariance_from_kets
from fk.model import build_model


def interval_mean(values):
    v=np.array(values)
    half=student_t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
    return dict(mean=float(v.mean()),repeat_95_halfwidth=float(half),
                finite_order_bias_included=False,rigorous_coverage_guarantee=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path)
    p.add_argument('--precision',type=Path,default=Path('results/precision_contacts_01'))
    args=p.parse_args()
    config=json.loads((args.run/'config.json').read_text())
    assert json.loads((args.run/'summary.json').read_text())['status']=='complete'
    manifest=json.loads((args.run/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((args.run/'source_snapshot'/name).read_bytes()).hexdigest()==digest
    hp=json.loads((args.precision/'summary.json').read_text());assert hp['status']=='complete'
    mp.dps=100
    hessian_checks=[];model=build_model()
    for name in sorted({r['name'] for r in hp['rows']}):
        low=next(r for r in hp['rows'] if r['name']==name and r['precision_digits']==50)
        high=next(r for r in hp['rows'] if r['name']==name and r['precision_digits']==90)
        assert max(abs(mp.mpf(a)-mp.mpf(b)) for a,b in zip(low['eigenvalues'],high['eigenvalues']))<mp.mpf('1e-40')
        assert mp.mpf(high['horizontal_gradient_norm'])<mp.mpf('1e-80')
        w,g,_=stationary_data(model,name,90)
        w=np.array([complex(v) for v in w]);g=np.array([complex(v) for v in g])
        _,independent=contact_hessian(w,g)
        spectrum=np.array(list(map(float,high['eigenvalues'])))
        assert np.max(abs(independent['hessian_eigenvalues']-spectrum))<1e-12
        positive=spectrum[spectrum>1e-30]
        hessian_checks.append(dict(name=name,zero_modes=int(np.sum(abs(spectrum)<1e-30)),
            minimum_positive_curvature=float(positive.min()),
            largest_precision_difference=str(max(abs(mp.mpf(a)-mp.mpf(b)) for a,b in zip(low['eigenvalues'],high['eigenvalues']))),
            soft_direction_crossover_order_estimate=float(1/positive.min()),
            globally_certified_support=name in ['t0','t20_joint']))
    os.environ['MPLCONFIGDIR']=str(args.run/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,len(config['names']),figsize=(19,10))
    fits=[];checked=0;lines=['# 固定 witness 的接触几何与 replica 扫描','',
        '5 个冻结方向，每个方向 5 个独立种子、65536 个 SMC 粒子、9 个阶数，共 225 个检查点。误差条以独立运行作为统计单位，不把重采样粒子当作独立样本。',
        '', '| witness | 有效维数拟合（均值 ± 重复 95% 半宽） | 零模 | 最小正曲率 |',
        '|---|---:|---:|---:|']
    for col,name in enumerate(config['names']):
        rows=[json.loads((args.run/f'{name}_seed{seed}/summary.json').read_text()) for seed in config['seeds']]
        frozen=np.load(args.run/f'{name}_frozen.npz');w=frozen['witness']
        spec=rows[0]['witness'];h=spec['reference_support']
        if name in ['t9','t11']:
            h=float(next(r for r in hp['rows'] if r['name']==name and r['precision_digits']==90)['overlap'])
        k=np.array(config['orders']);Y=[];W=[];mixture=[]
        for seed,row in zip(config['seeds'],rows):
            assert row['status']=='complete'
            assert [o['k'] for o in row['observations']]==config['orders']
            y=[];width=[];weights=[]
            for o in row['observations']:
                data=np.load(args.run/f"{name}_seed{seed}/k{o['k']}.npz")
                g=data['sample_kets'];lr=data['log_r']
                assert len(lr)==config['count']
                np.testing.assert_allclose(2*np.log(abs(g.conj()@w)),lr[:len(g)],atol=2e-11,rtol=0)
                cov=covariance_from_kets(g)
                assert np.max(abs(cov@cov+np.eye(16)))<1e-9
                assert data['purity_errors'].max()<1e-9
                assert np.max(abs(data['features'].mean(axis=0)-o['P_features']['feature_means']))<1e-12
                deficit=2*np.log(h)-lr
                y.append(o['log_q']-2*o['k']*np.log(h))
                width.append(np.sqrt(max(0.,deficit.mean())))
                weights.append(o['heuristic_eta_gt_08_P_mass'])
                checked+=1
            Y.append(y);W.append(width);mixture.append(weights)
        Y=np.array(Y);W=np.array(W);mixture=np.array(mixture)
        sensitivity=[]
        for kmin in [8,16,24,32]:
            select=k>=kmin;x=k[select].astype(float)
            for degree in [1,2,3]:
                X=np.column_stack([np.ones(len(x)),np.log(x)]+[x**(-j) for j in range(1,degree+1)])
                if len(x)<=X.shape[1]:continue
                dimensions=[2*np.linalg.lstsq(X,y[select],rcond=None)[0][1] for y in Y]
                sensitivity.append(dict(k_min=kmin,correction_degree=degree,**interval_mean(dimensions)))
        default=next(r for r in sensitivity if r['k_min']==16 and r['correction_degree']==2)
        geometry_name='t20_main' if name=='t20_two_orbit' else name
        geometry=next(r for r in hessian_checks if r['name']==geometry_name)
        high=k>=48
        width_slopes=[np.polyfit(np.log(k[high]),np.log(width[high]),1)[0] for width in W]
        record=dict(name=name,support_reference_used=h,support_is_exact=spec['support_is_exact'],
                    fit_sensitivity=sensitivity,width_raw_power=interval_mean(width_slopes),
                    heuristic_high_eta_mass_k128=interval_mean(mixture[:,-1]),
                    geometry=geometry,finite_order_dimension_is_not_contact_classification=True)
        fits.append(record)
        lines.append(f"| {name} | {default['mean']:.4f} ± {default['repeat_95_halfwidth']:.4f} | {geometry['zero_modes']} | {geometry['minimum_positive_curvature']:.8g} |")
        axes[0,col].errorbar(k,Y.mean(axis=0),yerr=Y.std(axis=0,ddof=1),fmt='o-',capsize=3)
        axes[0,col].set(xscale='log',title=name,xlabel='k',ylabel='log q - 2k log h')
        axes[1,col].loglog(k,W.mean(axis=0),'o-')
        axes[1,col].loglog(k,W.mean(axis=0)[-1]*(k/k[-1])**(-.5),'--')
        axes[1,col].set(xlabel='k',ylabel='RMS log-overlap width')
        axes[2,col].errorbar(k,mixture.mean(axis=0),yerr=mixture.std(axis=0,ddof=1),fmt='o-')
        axes[2,col].set(xscale='log',xlabel='k',ylabel='P mass with eta > 0.8',ylim=(-.02,1.02))
    fig.tight_layout();(args.run/'figures').mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(args.run/f'figures/fixed_geometry.{ext}',dpi=160)
    lines+=['','![扫描](figures/fixed_geometry.png)','',
        '50/90 位计算在至少 40 位上吻合；独立的全生成元双精度 Hessian 也与之相符。t9、t11 的解析驻点有 6 个对称零模和 6 个约 10⁻⁶ 的真实软方向。估算 k×曲率达到 1 需要约 1.4–1.5×10⁵ 阶，因此 k≤128 的有效维数约 12 并不矛盾；没有据此启动更大 k。',
        '', 't20 联合 witness 的全局接触集合由解析饱和条件确定为 S6。有限阶拟合有修正项和窗口依赖，误差条仅反映重复运行波动。两个轨道候选的拟合尚未稳定到局部几何维数 6，不能用这个拟合证明新的接触分量或维数。',
        '', 't20 两轨道候选的两个解析驻点具有相同的 50 个正曲率和相同的轨道诱导度量。条件于它们穷尽全局接触集，Laplace 极限权重相同。eta>0.8 仅为有限阶启发式分区，不能视为严格接触分量概率。',
        '', 't9、t11 使用解析驻点的重叠值修正此前局部搜索参考值；仍没有全局支持证书。t20 两轨道候选同样未获全局认证。所有高阶积分为点估计。']
    (args.run/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(args.run/'analysis.json',dict(rows=fits,hessian_checks=hessian_checks))
    write_json(args.run/'verification.json',dict(artifact_verification_passed=True,
        sample_checkpoints_checked=checked,independent_runs=len(config['seeds'])*len(config['names']),
        high_precision_Hessians_crosschecked=len(hessian_checks),
        high_order_integrals_certified=False,interval_Hessians_certified=False))
    print('Verified',checked,'checkpoints and',len(hessian_checks),'precision spectra',flush=True)


if __name__=='__main__':main()
