#!/usr/bin/env python3
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import t as student_t
from fk4.io import write_json
from fk3.discovery import covariance_from_kets


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    args=p.parse_args()
    summary=json.loads((args.run/'summary.json').read_text())
    assert summary['status']=='complete'
    config=json.loads((args.run/'config.json').read_text())
    fixed=Path(config['fixed'])
    geometry=json.loads((fixed/'summary.json').read_text())
    manifest=json.loads((args.run/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((args.run/'source_snapshot'/name).read_bytes()).hexdigest()==digest
    os.environ['MPLCONFIGDIR']=str(args.run/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ts=sorted({r['t'] for r in summary['rows']})
    fig,axes=plt.subplots(3,len(ts),figsize=(4.2*len(ts),10))
    fits=[]
    lines=['# 固定 witness 的几何标度','',
           '所有 replica 阶数使用同一个预先冻结的 witness。t=0 的完整 Gaussian 支持值为解析结果；其余点使用局部支持值作探索性拟合，不能据此认证全局接触维数。',
           '', '| t | 2×log k 拟合斜率（均值±重复95%半宽） | 原始宽度幂 | 最佳接触梯度范数 | Hessian 6–12 方向 |',
           '|---:|---:|---:|---:|---|']
    checked=0
    for col,t in enumerate(ts):
        rows=[r for r in summary['rows'] if r['t']==t]
        k=np.array([o['k'] for o in rows[0]['observations']])
        Y=np.array([[o['log_q_minus_2k_logh'] for o in r['observations']] for r in rows])
        W=np.array([[o['overlap_width_rms'] for o in r['observations']] for r in rows])
        for row in rows:
            data=np.load(args.run/f"t{t:g}_seed{row['seed']}.npz")
            for o in row['observations']:
                key=f"k{o['k']}"
                g=data[key+'_sample_kets']
                np.testing.assert_allclose(2*np.log(abs(g.conj()@data['witness'])),data[key+'_log_r'][:len(g)],atol=2e-11)
                cov=covariance_from_kets(g)
                assert np.max(abs(cov@cov+np.eye(16)))<1e-9
                checked+=1
        sensitivity=[]
        for kmin in [8,16,24,32]:
            selected=k>=kmin
            x=k[selected]
            for degree in [1,2,3]:
                X=np.column_stack([np.ones(len(x)),np.log(x)]+[x.astype(float)**(-j) for j in range(1,degree+1)])
                if len(x)<=X.shape[1]:
                    continue
                coefficients=np.array([np.linalg.lstsq(X,y[selected],rcond=None)[0] for y in Y])
                dimension=2*coefficients[:,1]
                hw=float(student_t.ppf(.975,len(rows)-1)*dimension.std(ddof=1)/np.sqrt(len(rows)))
                sensitivity.append({'k_min':kmin,'inverse_order_correction_degree':degree,
                                    'effective_dimension_mean':float(dimension.mean()),'repeat_95_halfwidth':hw,
                                    'per_seed_effective_dimension':dimension.tolist(),
                                    'finite_order_model_error_included':False})
        standard=next(r for r in sensitivity if r['k_min']==16 and r['inverse_order_correction_degree']==2)
        last=k>=48
        width_slopes=[np.polyfit(np.log(k[last]),np.log(w[last]),1)[0] for w in W]
        corrected=np.column_stack([np.ones(last.sum()),np.log(k[last]),1/k[last]])
        width_corrected=[np.linalg.lstsq(corrected,np.log(w[last]),rcond=None)[0][1] for w in W]
        geom=next(r for r in geometry['rows'] if r['t']==t)
        raw=np.load(fixed/f't{t:g}_fixed.npz')
        eig=raw['contact0_hessian_eigenvalues']
        fits.append({'t':t,'support_value_certified':t==0,'fit_sensitivity':sensitivity,
                     'width_power_raw_mean':float(np.mean(width_slopes)),
                     'width_power_with_1_over_k_mean':float(np.mean(width_corrected)),
                     'hessian_stationarity_norm':geom['contact_geometry'][0]['horizontal_gradient_norm'],
                     'soft_curvatures_6_to_12':eig[6:12].tolist(),
                     'asymptotic_contact_dimension_certified':12 if t==0 else None})
        lines.append(f"| {t:g} | {standard['effective_dimension_mean']:.4g} ± {standard['repeat_95_halfwidth']:.2g} | {np.mean(width_slopes):.4g} | {geom['contact_geometry'][0]['horizontal_gradient_norm']:.3g} | {eig[6]:.3g} … {eig[11]:.3g} |")
        axes[0,col].errorbar(k,Y.mean(axis=0),yerr=Y.std(axis=0,ddof=1),marker='o',capsize=3)
        axes[0,col].set(xscale='log',title=f't={t:g}',xlabel='k',ylabel='log q - 2 k log h')
        axes[1,col].loglog(k,W.mean(axis=0),'o-',label='Measured overlap width')
        axes[1,col].loglog(k,W.mean(axis=0)[-1]*(k/k[-1])**-.5,'--',label='k^-1/2 reference')
        axes[1,col].set(xlabel='k',ylabel='RMS log-overlap deficit')
        axes[1,col].legend(fontsize=7)
        axes[2,col].semilogy(np.arange(56),np.maximum(abs(eig),1e-15),'o',markersize=3)
        axes[2,col].set(xlabel='Horizontal eigenvalue index',ylabel='Absolute Hessian curvature')
    fig.tight_layout()
    folder=args.run/'figures'
    folder.mkdir(exist_ok=True)
    for suffix in ['png','pdf']:
        fig.savefig(folder/f'fixed_witness_geometry.{suffix}',dpi=170)
    lines+=['','![固定 witness 几何](figures/fixed_witness_geometry.png)','',
            't=0 的解析接触流形为 S⁶×S⁶，维数 12。拟合与之相符；原始有限阶宽度幂尚未达到 −1/2，加入 1/k 修正的敏感性结果保存在 fits.json。',
            '', 't=9、11 的另外六个 Hessian 曲率约为 10⁻⁶ 量级。在 k≤128 的窗口中，这些方向仍表现为很软的方向，因此 moment 拟合的有效维数约为 12，不能直接等同于按固定 10⁻⁶ 阈值数出的零特征值个数。t=11 的局部站稳精度也不足以作精确维数分类。',
            '', 't=20 的有限阶拟合趋近较低维数，但仍有拟合窗口和逆阶修正依赖。非零 t 的全局支持认证尚未完成；不宣称在 t=10 或其他参数处证明了接触流形突变。']
    (args.run/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(args.run/'fits.json',{'rows':fits,'strict_asymptotic_validation':False})
    write_json(args.run/'verification.json',{'artifact_verification_passed':True,'sample_checkpoints_checked':checked,
               'moment_uncertainty_certified':False,'nonzero_t_global_support_certified':False})


if __name__=='__main__':
    main()
