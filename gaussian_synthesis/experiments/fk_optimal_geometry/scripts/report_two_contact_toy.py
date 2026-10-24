#!/usr/bin/env python3
"""Independent tensor/q-quadrature checks and two-component statistical report."""
from pathlib import Path
from fractions import Fraction
from math import comb
import argparse
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import mp
from scipy.special import roots_legendre
from scipy.stats import t as student_t
from fk5.io import write_json
from fk5.two_contact_toy import coefficients,dimension
from fk2.exact_moments import full_tensor_reconstruction
from fk3.discovery import covariance_from_kets


def stats(v):
    v=np.array(v);se=float(v.std(ddof=1)/np.sqrt(len(v)))
    return dict(mean=float(v.mean()),standard_error=se,repeat_95_halfwidth=float(student_t.ppf(.975,len(v)-1)*se))


def independent_quadrature(k):
    # Each radial variable has polynomial degree k+12. Gauss-Legendre at
    # this order integrates that polynomial exactly in exact arithmetic.
    n=(k+14)//2
    x,w=roots_legendre(n);x=(x+1)/2;w=w/2
    P=np.array(np.meshgrid(x,x,x,x,indexing='ij')).reshape(4,-1)
    W=np.prod(np.array(np.meshgrid(w,w,w,w,indexing='ij')).reshape(4,-1),axis=0)
    for i in range(4):
        for j in range(i+1,4):W*=np.power(P[i]-P[j],4)
    A=np.prod(1-P,axis=0);B=np.prod(P,axis=0)
    integral=np.sum(W*sum(comb(k,l)**2*A**(k-l)*B**l for l in range(k+1)))/np.sum(W)
    return float(dimension(k)*integral)


def check_run(run,moments,canonical):
    summary=json.loads((run/'summary.json').read_text());assert summary['status']=='complete'
    config=json.loads((run/'config.json').read_text());records=[];checked=0
    for k in ([8,16,32,64,128] if canonical else config['orders']):
        name=f'toy_canonical_k{k}' if canonical else 'toy_balanced'
        theory=json.loads((moments/f'k{k}.json').read_text())
        w=np.load(run/f'{name}_frozen.npz')['witness']
        r=mp.mpf(float(w[-1].real))
        c=[mp.mpf(n)/d for n,d in theory['coefficients_rational']]
        exact=float(mp.log(mp.fsum(x*r**(2*l) for l,x in enumerate(c))))
        values=[];masses=[];coefficient_masses=[]
        for seed in config['seeds']:
            folder=run/f'{name}_seed{seed}'
            row=json.loads((folder/'summary.json').read_text());assert row['status']=='complete'
            o=next(o for o in row['observations'] if o['k']==k)
            data=np.load(folder/f'k{k}.npz');g=data['sample_kets'];prob=data['vacuum_full_probabilities']
            np.testing.assert_allclose(abs(g[:,[0,-1]])**2,prob[:len(g)],atol=1e-14,rtol=0)
            np.testing.assert_allclose(2*np.log(abs(g.conj()@w)),data['log_r'][:len(g)],atol=2e-11,rtol=0)
            cov=covariance_from_kets(g);assert np.max(abs(cov@cov+np.eye(16)))<1e-9
            assert data['purity_errors'].max()<1e-9
            assert abs(np.mean(prob[:,1]>prob[:,0])-o['full_contact_P_mass'])<1e-15
            values.append(o['log_q']);masses.append(o['full_contact_P_mass'])
            coefficient_masses.append(o['full_contact_Lambda_mass']);checked+=1
        est=stats(values)
        records.append(dict(k=k,log_q_estimate=est,log_q_exact=exact,
            moment_error_in_standard_errors=(est['mean']-exact)/est['standard_error'],
            full_contact_P_mass=stats(masses),full_contact_Lambda_mass=stats(coefficient_masses),
            limiting_mass=.25 if canonical else .5))
    return records,checked


def main():
    p=argparse.ArgumentParser();p.add_argument('--moments',type=Path,default=Path('results/two_contact_toy_01'))
    p.add_argument('--fixed',type=Path,default=Path('results/toy_fixed_gpu_n65536_s5_01'))
    p.add_argument('--canonical',type=Path,default=Path('results/toy_canonical_gpu_n65536_s5_01'))
    args=p.parse_args();mp.dps=100
    tensor_checks=[]
    for beta in [1.,2.,.5,1j,.5j]:
        w=np.zeros(128,complex);w[0]=1;w[-1]=beta
        q=np.vdot(w,full_tensor_reconstruction(w)).real
        # All these tensor operations are dyadic, exact in this magnitude range.
        r2=Fraction(float(abs(beta)**2));expected=sum(c*r2**l for l,c in enumerate(coefficients(2)))
        assert Fraction(float(q))==expected
        tensor_checks.append(dict(beta=str(beta),q2_exact=[expected.numerator,expected.denominator]))
    quadrature=[]
    for k in [2,4,8,16]:
        value=independent_quadrature(k);exact=float(sum(coefficients(k),Fraction()))
        assert abs(value-exact)<2e-11
        quadrature.append(dict(k=k,direct_radial_quadrature=value,exact=exact,error=value-exact))
    fixed,n1=check_run(args.fixed,args.moments,False)
    canonical,n2=check_run(args.canonical,args.moments,True)
    os.environ['MPLCONFIGDIR']=str(args.moments/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    k=np.array([r['k'] for r in fixed])
    axes[0].errorbar(k,[r['log_q_estimate']['mean'] for r in fixed],
                    yerr=[r['log_q_estimate']['repeat_95_halfwidth'] for r in fixed],fmt='o',label='SMC')
    axes[0].plot(k,[r['log_q_exact'] for r in fixed],label='Exact rational moments')
    axes[0].axhline(np.log(2),color='grey',linestyle='--');axes[0].set(xscale='log',xlabel='k',ylabel='log q balanced');axes[0].legend()
    for rows,label in [(fixed,'Fixed balanced'),(canonical,'Moving canonical')]:
        axes[1].errorbar([r['k'] for r in rows],[r['full_contact_Lambda_mass']['mean'] for r in rows],
                        yerr=[r['full_contact_Lambda_mass']['repeat_95_halfwidth'] for r in rows],fmt='o-',label=label)
    axes[1].axhline(.5,color='grey',linestyle='--');axes[1].axhline(.25,color='grey',linestyle='--')
    axes[1].set(xscale='log',xlabel='k',ylabel='Full-contact coefficient mass');axes[1].legend()
    moments=json.loads((args.moments/'summary.json').read_text())['rows']
    axes[2].plot([r['k'] for r in moments],[float(r['scaled_log_height_ratio']) for r in moments],'o-')
    axes[2].axhline(np.log(1/3),color='grey',linestyle='--')
    axes[2].set(xscale='log',xlabel='k',ylabel='(2k-1) log canonical height ratio')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(args.moments/f'two_contact_toy.{ext}',dpi=170)
    lines=['# 已知最优解的两接触轨道测试','',
        '目标为 (3|vac⟩+|full⟩)/√10，严格 Gaussian 成本为 4/√10。完整接触集合只有 vacuum 和 full 两条射线，两者维数均为 0。证明及精确矩公式见 TWO_CONTACT_TOY.md。',
        '', '| k | 固定 witness 的 full 系数质量 | canonical witness 的 full 系数质量 |',
        '|---:|---:|---:|']
    for c in canonical:
        f=next(r for r in fixed if r['k']==c['k'])
        lines.append(f"| {c['k']} | {f['full_contact_Lambda_mass']['mean']:.6f} | {c['full_contact_Lambda_mass']['mean']:.6f} |")
    lines+=['','![两接触测试](two_contact_toy.png)','',
        '固定 witness 权重趋向 1/2；随 k 移动的 canonical witness 权重趋向目标要求的 1/4。解析矩方程给出 (2k−1)log(r_k)→log(1/3)，即 1/k 高度调整。本例不需要 log(k)/k，因为两个接触分量维数相同。',
        '', 'k2 用完整 128×128 张量上的独立 Cartan projector 精确核对；k2/4/8/16 又用独立四维 Gauss–Legendre 多项式积分核对。所有阶数均用未调用高阶公式的独立 SMC 运行作数值交叉验证。统计置信区间不是严格概率包围；canonical 方向的 90 位根求解也没有伪装成区间认证。']
    (args.moments/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(args.moments/'analysis.json',dict(fixed=fixed,canonical=canonical))
    write_json(args.moments/'verification.json',dict(artifact_verification_passed=True,
        exact_full_tensor_k2_checks=tensor_checks,independent_quadrature=quadrature,
        sample_checkpoints_checked=n1+n2,
        maximum_absolute_SMC_moment_error_in_standard_errors=max(abs(r['moment_error_in_standard_errors']) for r in fixed+canonical),
        SMC_integrals_interval_certified=False))
    print('Verified tensor checks, radial quadrature and',n1+n2,'GPU checkpoints',flush=True)


if __name__=='__main__':main()
